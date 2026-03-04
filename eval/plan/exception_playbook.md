# SFT 학습 예외 상황 플레이북

**프로젝트:** Korean 1B SFT 재학습  
**서버:** 8× B200 183GB, Driver 580.95.05, CUDA 13.1, PyTorch 2.10  
**작성일:** 2026-02-26  
**설정:** bs=4 × 8GPU × grad_accum=2 = effective batch 64, max_steps=10000, lr=2e-5, FP8

---

## 시나리오 1: Loss가 0으로 떨어지는 경우

### 감지 기준
- **즉각 경고:** loss < 0.01이 3 step 연속 발생
- **주의:** loss < 0.1이 10 step 이상 지속
- **정상 범위:** 1B SFT에서 수렴 시 loss ≈ 1.5~2.0. 0에 가까우면 100% 비정상

### 즉각 대응
1. 학습 즉시 중단 (Ctrl+C 또는 `kill -SIGINT <PID>`)
2. 가장 최근 정상 체크포인트 확인:
   ```bash
   ls -lt checkpoints/korean_1b_sft/checkpoint-* | head -5
   ```

### 원인별 진단 및 대응

#### 1-A. Labels Shift 버그 재발
**확인 방법:**
```python
# 데이터에서 샘플 하나 로드 후 labels 검증
from data.sft_dataset import SFTDataset
from tokenizers import Tokenizer
tok = Tokenizer.from_file("tokenizer/korean_sp/tokenizer.json")
ds = SFTDataset("data/sft/train.jsonl", tok, max_seq_len=4096)
ids, labels = ds[0]
# labels에서 -1이 아닌 부분이 input_ids의 다음 토큰과 일치하는지 확인
mask = labels != -1
print(f"유효 labels 수: {mask.sum()}")
print(f"첫 유효 label 위치: {mask.nonzero()[0].item() if mask.any() else 'NONE'}")
# labels[i]는 input_ids[i+1]과 같아야 함 (autoregressive)
# 만약 labels == input_ids 이면 shift 안 됨 → 버그
```
**수정:** `sft_dataset.py`에서 `labels = input_ids[1:]`, `input_ids = input_ids[:-1]` shift 확인

#### 1-B. 데이터 오염
**확인 방법:**
```python
# 랜덤 배치에서 실제 학습 토큰 검사
for batch in train_loader:
    ids, labels, mask = batch
    valid = (labels != -1)
    print(f"유효 토큰 비율: {valid.float().mean():.4f}")
    # 유효 토큰이 0이면 모든 labels가 -1 → loss=0
    if valid.sum() == 0:
        print("🔴 모든 labels가 ignore_index! 데이터 문제")
    break
```
**대응:** 데이터 재생성, `prepare_sft_data.py` 재실행

#### 1-C. Learning Rate 문제
**확인:** loss가 갑자기 0이면 lr 문제보다는 labels 버그일 가능성이 훨씬 높음. 그래도 확인:
```bash
grep "lr " checkpoints/korean_1b_sft/train.log | tail -20
# lr이 비정상적으로 높으면 (>1e-3) 수정
```

---

## 시나리오 2: Loss Spike (급등)

### 감지 기준
- **Spike 정의:** 이전 log_interval 평균 대비 **3배 이상** 급등
- **예:** 평균 loss 1.9에서 갑자기 5.7 이상
- **GNorm 기준:** grad_norm > 10.0이면 주의, > 50.0이면 심각

### 원인별 대응

| 원인 | 진단 | 대응 |
|------|------|------|
| Bad batch (이상 데이터) | 해당 step의 배치 내용 확인 | 1~2회 spike 후 자연 복구되면 무시 |
| LR 문제 | warmup 직후 spike → lr 너무 높음 | lr을 1e-5로 낮추고 재시작 |
| GNorm 폭발 | gnorm > 50 | max_grad_norm을 0.5로 강화 |
| FP8 수치 불안정 | FP8 관련 warning 확인 | `--use_fp8` 제거하고 BF16으로 전환 |

### 대응 절차
1. **1회 spike:** 무시 (단발성 bad batch). 다음 log에서 복구 확인
2. **연속 3회 spike:** 학습 중단
3. **복구 방법:**
   ```bash
   # 마지막 정상 체크포인트에서 재시작, lr 낮추기
   bash scripts/launch_sft.sh --resume checkpoints/korean_1b_sft/checkpoint-XXXX --lr 1e-5
   ```

### 현재 코드의 보호 장치
- ✅ `max_grad_norm=1.0` (gradient clipping 활성화)
- ✅ Non-finite loss 감지 → RuntimeError 발생 (trainer.py `_step()`)
- ❌ Loss spike 자동 감지/skip은 미구현 → `monitor_training.sh`로 보완

---

## 시나리오 3: 과적합 (val_loss > train_loss 지속)

### 감지 기준
- **주의:** val_loss - train_loss > 0.15 (상대갭 8% 이상)
- **심각:** val_loss가 3회 연속 eval에서 상승 (train_loss는 하강 중)
- **eval_interval:** 현재 250 steps → 매 250 step마다 val_loss 기록됨

### 현재 코드 상태
- ✅ `val_loader` 지원 (sft.py에서 `--val_data` 인자 있음)
- ✅ `_run_validation()` 구현됨 (trainer.py)
- ✅ Best checkpoint 자동 저장 (`val_loss < self._best_val_loss`)
- ❌ **Early stopping 미구현** — val_loss가 올라도 max_steps까지 학습 계속

### 대응

#### 즉시 가능한 조치
1. **수동 early stop:** 모니터링 스크립트가 경고 → 수동 중단
2. **Best checkpoint 사용:** `checkpoint-best` 디렉토리에 자동 저장됨
   ```bash
   ls checkpoints/korean_1b_sft/checkpoint-best/
   ```

#### 과적합 해소 방법 (재학습 시)
| 방법 | 설정 변경 |
|------|-----------|
| LR 낮추기 | `--lr 1e-5` |
| Weight decay 높이기 | `--weight_decay 0.05` |
| 데이터 augmentation | NEFTune 이미 활성화 (noise_alpha=10.0) ✅ |
| Steps 줄이기 | `--max_steps 7000` (과적합 시작 전 step에서 멈춤) |
| Dropout | 모델 구조 수정 필요 (현재 코드에서 쉽지 않음) |

#### Early Stopping 추가 방법 (trainer.py 수정)
```python
# trainer.py의 train() 메서드에서 validation 후:
if val_loss > self._best_val_loss:
    self._patience_counter += 1
    if self._patience_counter >= 5:  # 5회 연속 개선 없으면 중단
        self._log("Early stopping triggered")
        break
else:
    self._patience_counter = 0
    self._best_val_loss = val_loss
```

---

## 시나리오 4: OOM (Out of Memory)

### 현재 메모리 추정

| 항목 | 추정 |
|------|------|
| 모델 파라미터 (1.19B, BF16) | ~2.4 GB |
| 옵티마이저 상태 (AdamW, fp32) | ~9.5 GB |
| Gradient (BF16) | ~2.4 GB |
| Activation (bs=4, seq=4096, gradient checkpointing ON) | ~8-15 GB |
| **Peak 총합 (per GPU)** | **~25-35 GB** |
| **B200 여유** | **183 - 35 = ~148 GB 여유** |

→ 1B 모델에서 OOM 가능성 **극히 낮음**

### 만약 발생한다면
1. **증상:** `torch.cuda.OutOfMemoryError` → trainer.py에서 이미 catch하여 상세 메시지 출력
2. **즉시 대응:**
   ```bash
   # batch_size 줄이기 (4→2), grad_accum 늘리기 (2→4) → effective batch 동일
   bash scripts/launch_sft.sh --batch_size 2 --grad_accum 4 --resume <last_ckpt>
   ```
3. **Gradient checkpointing:**
   - ✅ **이미 활성화됨** (sft.py에서 `model.gradient_checkpointing_enable()`)
4. **추가 조치:**
   ```bash
   # 메모리 fragmentation 방지
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   ```

### 메모리 모니터링
```bash
watch -n 5 nvidia-smi  # 실시간 확인
# 또는 monitor_training.sh 사용 (아래 참조)
```

---

## 시나리오 5: GPU Hang / NCCL 통신 장애

### 감지 방법
- **증상:** 학습 로그가 멈춤 (새 step이 N분 이상 안 나옴)
- **NCCL timeout:** 기본 30분 후 에러 발생
- `nvidia-smi`에서 특정 GPU utilization 0%

### 진단
```bash
# 1. GPU 상태 확인
nvidia-smi

# 2. NCCL 디버그 활성화하여 재시작
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# 3. 프로세스 상태 확인
ps aux | grep torchrun
```

### 복구 방법
```bash
# 1. 기존 프로세스 정리
pkill -f torchrun
sleep 5

# 2. 가장 최근 체크포인트 자동 감지
LATEST_CKPT=$(ls -d checkpoints/korean_1b_sft/checkpoint-* 2>/dev/null \
  | grep -v best | sort -t- -k2 -n | tail -1)
echo "Latest checkpoint: ${LATEST_CKPT}"

# 3. 재시작
bash scripts/launch_sft.sh --resume "${LATEST_CKPT}"
```

### 최근 체크포인트 자동 감지 스크립트
```bash
#!/bin/bash
# find_latest_checkpoint.sh
CKPT_DIR="${1:-checkpoints/korean_1b_sft}"
LATEST=$(ls -d "${CKPT_DIR}"/checkpoint-[0-9]* 2>/dev/null \
  | sort -t- -k2 -n | tail -1)
if [[ -z "$LATEST" ]]; then
    echo "No checkpoint found in ${CKPT_DIR}" >&2
    exit 1
fi
echo "$LATEST"
```

### 예방
- `save_interval=500` (현재 설정) → 최대 500 step 손실
- NCCL timeout 조정: `export NCCL_TIMEOUT=1800` (30분 → 필요 시 줄이기)

---

## 시나리오 6: 학습 완료 후 반복률 >15%

### 판단 기준

| 반복률 | 판단 | 대응 |
|--------|------|------|
| <5% (rep_penalty 없이) | ✅ 성공 | 배포 가능 |
| 5-10% | 🟡 OK | rep_penalty=1.1로 배포 |
| 10-20% | 🟠 경계 | 아래 파라미터 조정 시도 |
| >20% | 🔴 실패 | 재학습 필요 |

### 파라미터 조정으로 해결 시도 (재학습 없이)

```python
# 추론 시 적용
generate_kwargs = {
    "repetition_penalty": 1.1,      # 1.05~1.2 범위 탐색
    "no_repeat_ngram_size": 3,      # 3-gram 반복 차단
    "temperature": 0.7,             # 약간 낮추면 반복 감소
    "top_p": 0.9,
}
```

### 재학습이 필요한 경우
- rep_penalty=1.2 + no_repeat_3gram에서도 >10%
- 원인 분석:
  1. **데이터 내 반복 패턴:** `data_quality_audit.py`로 재확인
  2. **Epoch 과다:** 5+ epoch은 반복 패턴 암기 유발 → 3-4 epoch이 적정
  3. **EOS 학습 부족:** truncation 시 EOS 손실 여부 확인

### 고급 대응 (추가 학습 방법)
| 방법 | 설명 | 소요 |
|------|------|------|
| ORPO | Preference optimization, 반복 패턴 직접 penalize | +3-6시간 |
| DPO | Chosen(비반복) vs Rejected(반복) 쌍 필요 | +4-8시간 |
| rep_penalty fine-tuning | 추론 시 penalty 결과를 reward로 RL | 복잡 |

---

## 시나리오 7: ko_ifeval 기대치 미달 (<15%)

### 원인 분석 방법

#### Step 1: 모델 출력 직접 확인
```bash
# ko_ifeval 실패 샘플 분석
python -c "
# lm_eval 결과에서 실패 케이스 추출
# 지시문 이해 부족 vs 포맷 오류 vs 한국어 능력 부족 구분
"
```

#### Step 2: 카테고리별 분석
| 실패 유형 | 의미 | 대응 |
|-----------|------|------|
| 지시 무시 (wrong format) | instruction following 약함 | SFT 데이터에 format-constrained 샘플 추가 |
| 한국어 이해 실패 | 한국어 능력 부족 | 한국어 비율 높이기 (현재 ~70%) |
| 추론 오류 | 1B 모델 한계 | 모델 크기 한계 → 3B 전환 |

#### Step 3: 모델 한계 vs 데이터 문제 구분
```
1B 모델 ko_ifeval 현실적 범위: 15-30%
- <15%: 데이터/학습 문제 가능성 높음
- 15-25%: 정상 범위, 데이터로 개선 여지 있음
- 25-30%: 1B 한계에 근접, 3B 전환 필요
- >30%: 1B에서 달성하기 어려움
```

### 데이터 추가 수집 방향
1. **Korean instruction-following 데이터:** KoAlpaca, KULLM 등에서 format-constrained 샘플
2. **Multi-turn 한국어 대화:** 지시 따르기 능력 강화
3. **ko_ifeval과 유사한 포맷 데이터:** "~형식으로 답하시오" 유형

---

## 시나리오 8: 디스크 공간 부족

### 현재 상태
```
/PROJECT: 3.5TB 총, 1.4TB 사용, 2.2TB 가용 (39% 사용)
```

### 체크포인트 크기 추정
| 항목 | 크기 |
|------|------|
| model.pt (1.19B BF16) | ~2.4 GB |
| optimizer.pt (AdamW states) | ~9.5 GB |
| scheduler + meta | ~1 MB |
| **체크포인트 1개** | **~12 GB** |
| 10,000 steps / 500 save = 20개 | **~240 GB** |
| + best checkpoint | +12 GB |
| + tensorboard logs | ~100 MB |
| **총 예상** | **~252 GB** |

→ 2.2TB 가용 대비 충분하지만, 여러 실험 시 누적 주의

### 체크포인트 관리 전략

#### 저장 주기 최적화
- **현재:** 500 step마다 (추천 유지)
- 디스크 부족 시: 1000 step으로 변경 → 120 GB로 절반 감소
- `train_config.save_interval = 1000`

#### 오래된 체크포인트 자동 삭제
```bash
#!/bin/bash
# cleanup_checkpoints.sh — 최신 N개만 유지, best는 항상 보존
CKPT_DIR="${1:-checkpoints/korean_1b_sft}"
KEEP="${2:-5}"  # 최신 5개 유지

CKPTS=$(ls -d "${CKPT_DIR}"/checkpoint-[0-9]* 2>/dev/null | sort -t- -k2 -n)
TOTAL=$(echo "$CKPTS" | wc -l)
DELETE=$((TOTAL - KEEP))

if [[ $DELETE -gt 0 ]]; then
    echo "$CKPTS" | head -n "$DELETE" | while read ckpt; do
        echo "Removing: $ckpt"
        rm -rf "$ckpt"
    done
    echo "Kept latest $KEEP checkpoints + checkpoint-best"
else
    echo "Only $TOTAL checkpoints, nothing to delete (keep=$KEEP)"
fi
```

### 디스크 모니터링
```bash
# 학습 중 주기적 확인
df -h /PROJECT | awk 'NR==2 {if ($5+0 > 80) print "🔴 DISK >80%: "$5}'
```

---

## 학습 재시작 가이드

### 현재 코드의 Resume 지원

✅ **완전 지원됨:**
- `sft.py`에 `--resume` 인자 있음
- `load_checkpoint()`으로 model, optimizer, scheduler 상태 모두 복원
- `start_step` 반환 → 이어서 학습

### 재시작 명령어
```bash
# 방법 1: 최신 체크포인트에서 자동 재시작
LATEST=$(ls -d checkpoints/korean_1b_sft/checkpoint-[0-9]* 2>/dev/null \
  | sort -t- -k2 -n | tail -1)
bash scripts/launch_sft.sh --resume "${LATEST}"

# 방법 2: 특정 체크포인트 지정
bash scripts/launch_sft.sh --resume checkpoints/korean_1b_sft/checkpoint-0003000

# 방법 3: LR 변경하며 재시작 (과적합/spike 대응)
bash scripts/launch_sft.sh --resume "${LATEST}" --lr 1e-5
```

### 주의사항
- **cosine schedule:** resume 시 scheduler가 중간 step에서 복원됨 → LR이 올바른 위치에서 재개
- **max_steps 변경 시:** 원래 5000 step 기준 schedule인데 10000으로 변경하면 LR curve가 달라짐 → 처음부터 재학습 권장
- **DDP seed:** resume 시 동일 seed 사용해야 데이터 순서 재현 (현재 코드에서 자동 처리)

---

## 모니터링 자동화

별도 스크립트: `scripts/monitor_training.sh` 참조

### 감시 항목 요약

| 항목 | 임계값 | 의미 |
|------|--------|------|
| loss = 0.0000 (3 step 연속) | 🔴 Critical | Labels 버그 |
| loss spike (3× 평균) | 🟠 Warning | Bad batch / LR |
| gnorm > 10.0 | 🟠 Warning | 불안정 |
| gnorm > 50.0 | 🔴 Critical | 발산 직전 |
| GPU util < 50% | 🟡 Info | 병목 (data loading?) |
| 로그 5분 이상 멈춤 | 🔴 Critical | Hang / NCCL 장애 |
| 디스크 사용 > 80% | 🟠 Warning | 체크포인트 정리 필요 |

---

## 위험도 순위 (높음 → 낮음)

| 순위 | 시나리오 | 위험도 | 예방 |
|------|----------|--------|------|
| 1 | **Loss → 0 (Labels 버그)** | 🔴🔴🔴 | 학습 전 labels shift 검증 스크립트 실행 |
| 2 | **GPU Hang (NCCL)** | 🔴🔴 | save_interval=500, NCCL 환경변수 설정 |
| 3 | **과적합** | 🔴 | val_data 필수, 모니터링 |
| 4 | **반복률 >15%** | 🟠🟠 | 깨끗한 데이터, 적정 epoch |
| 5 | **Loss Spike** | 🟠 | grad_clip=1.0, 이미 설정됨 |
| 6 | **ko_ifeval 미달** | 🟠 | 1B 한계 인지, 데이터 다양성 |
| 7 | **디스크 부족** | 🟡 | 2.2TB 여유, 자동 정리 |
| 8 | **OOM** | 🟢 | 183GB에 1B 모델, 거의 불가능 |

---

## 학습 전 체크리스트

```
□ 데이터 필터링 완료 (data_quality_audit.py)
□ Val split 생성 (90/10)
□ Labels shift 검증 (위 코드 스니펫 실행)
□ sft_dataset.py 수정 확인 (dynamic padding, EOS 보존)
□ launch_sft.sh 설정 확인 (max_steps, val_data, lr)
□ 디스크 공간 확인 (df -h /PROJECT)
□ GPU 상태 확인 (nvidia-smi)
□ monitor_training.sh 백그라운드 실행
□ tensorboard 실행: tensorboard --logdir checkpoints/korean_1b_sft/tensorboard
```
