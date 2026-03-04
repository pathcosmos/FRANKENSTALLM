# 🚀 3B 한국어 LLM 마스터 플랜

**작성일**: 2026-02-27 04:27 KST
**프로젝트**: `/PROJECT/0325120031_A/ghong/taketimes/llm-bang/`
**결정**: 1B → 3B 전환 (1B 구조적 한계 확인)
**총 예상 기간**: ~35시간

---

## 0. 현황 요약

### 1B에서 확인된 것

| 항목 | 결과 |
|------|------|
| 반복률 (raw, 올바른 포맷) | 30.7% |
| 반복률 (rep_penalty=1.1) | 18.0% |
| val_loss | 2.2062 |
| 자연 종료율 | 60% |
| 짧은 QA 품질 | ✅ 양호 (수도, 김치 등) |
| 복잡한 질문 품질 | ❌ 반복 퇴화 심각 |

### 3B 전환 근거

1. **반복률 18%는 1B 구조적 한계** — d_model=2048, 24 layers로는 긴 시퀀스에서 hidden state 붕괴 불가피
2. **Scaling law 예측**: 3B는 loss ~7% 감소 → 반복률 5~8% 예상
3. **ORPO 없이도 목표 달성 가능**: 3B SFT만으로 <10%, +ORPO로 <3%
4. **총 소요시간 ORPO 실패→3B (39h) vs 3B 직행 (30h)** — 직행이 빠름

---

## 1. 3B 모델 아키텍처

| 파라미터 | 1B (현재) | 3B (목표) | 변화 |
|---------|----------|----------|------|
| d_model | 2048 | **3072** | 1.5× |
| n_layers | 24 | **32** | 1.33× |
| n_heads | 16 | **24** | 1.5× |
| n_kv_heads (GQA) | 4 | **8** | 2× (GQA 3:1) |
| d_ffn (SwiGLU) | 5472 | **8192** | 1.5× |
| vocab_size | 64000 | **64000** | 동일 |
| max_seq_len | 4096 | **4096** | 동일 |
| rope_theta | 500000 | **500000** | 동일 |
| **총 파라미터** | **1.19B** | **~3.42B** | 2.9× |

### 파라미터 수 상세

```
Embedding:      64000 × 3072                    = 196.6M
Attention/layer: Q(3072×3072) + K(3072×1024) + V(3072×1024) + O(3072×3072) = 25.1M
FFN/layer:      SwiGLU gate(3072×8192) + up(3072×8192) + down(8192×3072) = 75.5M
Layer total:    25.1 + 75.5 = 100.6M × 32 layers = 3,219M
LM Head:        tied with embedding
총계:           196.6M + 3,219M ≈ 3.42B
```

### GPU 메모리 예상 (8× B200 183GB)

```
모델 (FP8):           3.42 GB
Optimizer (FP32):     27.4 GB (DDP 분산 → ~3.4 GB/GPU)
Gradients (BF16):     6.84 GB (분산 → ~0.86 GB/GPU)
Activations (bs=4):   ~15-25 GB (gradient checkpointing)
Per GPU 합계:         ~28 GB → B200의 15% → 매우 여유
```

### Config 파일

```yaml
# configs/korean_3b_fp8.yaml
model:
  vocab_size: 64000
  d_model: 3072
  n_layers: 32
  n_heads: 24
  n_kv_heads: 8
  d_ffn: 8192
  max_seq_len: 4096
  rope_theta: 500000.0
  dropout: 0.0
  bias: false
  use_flash_attn: true
  use_fp8: true

train:
  max_steps: 34000       # 8.91B × 4 epoch
  batch_size: 4
  grad_accum_steps: 8    # eff_batch: 4 × 8 × 8GPU × 4096 ≈ 1M tok/step
  lr: 1.5e-4
  min_lr: 1.5e-5
  weight_decay: 0.1
  warmup_steps: 2000
  max_grad_norm: 1.0
  log_interval: 10
  save_interval: 500
  eval_interval: 200
  fp8_format: "MXFP8"

tokenizer:
  vocab_size: 64000
  type: sentencepiece_unigram
```

---

## 2. 데이터 파이프라인

### 즉시 사용 가능

| 소스 | 크기 | 토큰 수 | 상태 |
|------|------|---------|------|
| korean_c4_train.bin | 15.1 GB | 7.56B | ✅ 토큰화 완료 |
| korean_namuwiki_train.bin | 2.2 GB | 1.08B | ✅ 토큰화 완료 |
| korean_wiki_train.bin | 0.5 GB | 0.26B | ✅ 토큰화 완료 |
| **합계 (korean_train.bin)** | **17.8 GB** | **8.91B** | ✅ **즉시 시작 가능** |

### 추가 준비 필요 (병렬 토큰화)

| 소스 | 크기 | 추정 토큰 | 작업 | 예상 소요 |
|------|------|----------|------|----------|
| culturax_ko | 60 GB | ~30-40B | parquet→토큰화 | 4-6h |
| hplt_ko | 23 GB | ~12-15B | 토큰화 | 2-3h |
| cc100_ko | 14 GB | ~8-10B | xz해제+토큰화 | 2h |
| oscar_ko | 9.2 GB | ~5-6B | 토큰화 | 1-2h |
| korean_textbooks | 6.4 GB | ~3-4B | 토큰화 | 1h |
| **합계** | **~123 GB** | **~70-80B** | | **8-12h (병렬)** |

### Chinchilla 분석

```
3.42B × 20 = 68.4B tokens (최적)
즉시 사용 가능: 8.91B × 4 epoch = 35.6B (최적의 52%)
extra 포함 시: ~80-90B → 충분 (131%)
```

### 데이터 타임라인

| 시점 | 행동 |
|------|------|
| **지금** | korean_train.bin 8.91B로 사전학습 시작 (4 epoch) |
| **병렬** | korean_extra 토큰화 + MinHash 중복제거 + PPL 필터 진행 |
| **Phase 2** | 전체 60-80B 토큰으로 extended pretrain (선택) |

### SFT 데이터

| 항목 | 값 |
|------|-----|
| 현재 클린 데이터 | ~120-135K 샘플 (필터링 후) |
| Val split | 10% (~12-13K) |
| 3B에 충분? | ✅ (7B Alpaca도 52K로 학습) |

### 추가 고품질 SFT 소스 (선택)

- `hPark/orca-ko` (~200K)
- `maywell/synatra-orca` (~300K)
- `HAERAE-HUB/qarv-instruct-100k` (100K)
- 필터링 후 200-300K 사용 가능

---

## 3. 사전학습 계획

### 하이퍼파라미터

| 파라미터 | 값 | 근거 |
|---------|-----|------|
| LR | 1.5e-4 | 3B 표준 (1B의 3e-4 대비 보수적) |
| Min LR | 1.5e-5 | LR의 10% |
| Warmup | 2,000 steps | ~6% |
| Weight Decay | 0.1 | Pretrain 표준 |
| Batch Size | 4/GPU × 8GPU × 8 grad_accum = 256 | eff ~1M tok/step |
| Max Steps | 34,000 | 8.91B × 4 epoch |
| Precision | MXFP8 | B200 최적화 |
| Grad Clip | 1.0 | 표준 |

### 예상 소요 시간

```
1B 실측: ~75,700 tok/s (단일 B200)
3B 예상: 파라미터 3× → throughput ~40-50% 감소
         BUT batch 최적화 + FP8 → 보정

보수적 추정:
  8.91B × 4 epoch = 35.6B tokens
  처리량: ~400K tok/s (8× B200, FP8, 최적 배치)
  소요: 35.6B / 400K = 89,000초 ≈ 24.7시간

낙관적 추정:
  처리량: ~600K tok/s → 16.5시간
  
채택 추정: ~26시간
```

### 모니터링

```bash
# 실시간 로그
tail -f checkpoints/korean_3b_fp8/train.log

# TensorBoard
tensorboard --logdir checkpoints/korean_3b_fp8/tensorboard --port 6007

# GPU 상태
watch -n 10 nvidia-smi
```

**핵심 관찰 수치**:

| 수치 | 정상 범위 | 경고 | 즉시 중단 |
|------|----------|------|----------|
| Train Loss | 시작 ~10, 수렴 ~3-4 | 정체 5000+ steps | 발산 (상승) |
| GNorm | 0.5-2.0 | >5.0 | >50 |
| PPL | 하강 추세 | 정체 | 상승 |
| GPU Util | >90% | <70% | <50% (병목) |
| tok/s | >300K | <200K | <100K |

### 체크포인트 전략

| Step | 행동 |
|------|------|
| 500 | Sanity check — loss 하강 중? OOM 없나? |
| 5,000 | 1 epoch 완료 — PPL 측정, 한국어 텍스트 perplexity <20? |
| 10,000 | 중간점 — loss 추세 확인, 과적합 징후? |
| 17,000 | 2 epoch — PPL < 15? |
| 25,000 | 3 epoch — PPL < 12? |
| 34,000 | 최종 — PPL < 10 목표 |

**디스크**: 체크포인트 1개 ~27GB (모델 7GB + optimizer 20GB) × save_interval=500 → 68개 = ~1.8TB
→ **save_interval=2000으로 변경 권장** → 17개 = ~460GB

---

## 4. SFT 계획

### 1B 교훈 전부 적용

| 교훈 | 1B에서 발견 | 3B에 적용 |
|------|------------|-----------|
| Dynamic padding 필수 | 4096 고정으로 90% 낭비 | ✅ sft_dataset.py 수정 완료, 그대로 사용 |
| EOS 보존 | 트렁케이션 시 EOS 손실 | ✅ `response_ids[-1] = eos_id` 강제 |
| Val split 필수 | 과적합 모니터링 불가했음 | ✅ 10% split |
| 3-4 epoch | 2 epoch은 underfitting | ✅ max_steps 계산 |
| OpenOrca 과대표집 방지 | 5× 가중치로 과적합 | ✅ 2.0× 이하 |
| 데이터 품질 필터 | `</s>` 리터럴, Q/A 마커 오염 | ✅ 필터 스크립트 완성 |
| 올바른 포맷 통일 | 학습/추론 포맷 불일치 | ✅ `<\|user\|>/<\|assistant\|>` 일관 |
| Early stopping | val_loss 상승해도 학습 계속됨 | ✅ patience=5 구현 |
| NEFTune alpha | 10.0은 과도 | ✅ 5.0으로 조정 |

### 하이퍼파라미터

| 파라미터 | 값 | 근거 |
|---------|-----|------|
| LR | 2e-5 | SFT 표준 (Alpaca, Vicuna 동일) |
| Warmup | 300 steps | ~3% |
| Max Steps | 10,000 | ~3-4 epoch (데이터 크기 따라 조정) |
| Batch Size | 4/GPU × 2 grad_accum × 8GPU = 64 | SFT 표준 |
| Weight Decay | 0.01 | SFT 표준 (pretrain 0.1보다 낮게) |
| NEFTune | alpha=5.0 | 과적합 방지 |
| Eval Interval | 500 steps | |
| Early Stopping | patience=5 | 2,500 steps 무개선 시 중단 |
| Dropout | 0.05 | 과적합 방지 (1B에서 0.0이었음) |

### 실행 명령어

```bash
RUN_NAME=korean_3b_sft \
BASE_CHECKPOINT=checkpoints/korean_3b_fp8/checkpoint-BEST \
SFT_DATA=data/sft_v2/train.jsonl \
VAL_DATA=data/sft_v2/val.jsonl \
MAX_STEPS=10000 \
LR=2.0e-5 \
WARMUP_STEPS=300 \
bash scripts/launch_sft.sh
```

**예상 시간**: ~2시간 (3B는 1B 대비 ~2.5× 느림)

### 성공 기준

| 지표 | 목표 | 실패 기준 |
|------|------|----------|
| Train Loss | < 1.85 | > 2.00 |
| Val Loss | Train의 1.1배 이내 | 1.2배 초과 |
| 반복률 (raw) | < 10% | > 15% |
| 반복률 (rep_penalty=1.1) | < 3% | > 8% |
| EOS 종료율 | > 80% | < 60% |

---

## 5. ORPO 계획

### 타이밍: SFT 완료 후, 반복률 >5%일 때만

### 데이터

| 소스 | 샘플 수 | 유형 |
|------|---------|------|
| maywell/ko_Ultrafeedback_binarized | ~60K | 일반 도메인 preference |
| kuotient/orca-math-korean-dpo-pairs | 수천 | 수학 도메인 |
| **자체 생성** (3B SFT 모델로) | ~2-5K | 반복 타겟 preference |

**자체 생성 방법**:
1. 3B SFT 모델로 1000 프롬프트 × 4 temperature 생성
2. 반복 출력 → rejected, 깨끗한 출력 → chosen
3. 3B에서는 반복률 낮으므로 **1B보다 훨씬 편향 적음**

### 하이퍼파라미터

```python
ORPOConfig(
    learning_rate=5e-7,    # 매우 낮은 LR (정렬 학습)
    num_train_epochs=1,    # 1 epoch 충분
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    beta=0.1,              # ORPO coefficient
)
```

### 예상 시간: 1-2시간

### 목표: 반복률 <3% (raw), <1% (rep_penalty=1.1)

---

## 6. 평가 계획

### 벤치마크

| 벤치마크 | 도구 | 1B 예상 | 3B 목표 |
|---------|------|---------|---------|
| ko_ifeval | lm-eval-harness | 15-25% | **35-45%** |
| ko_winogrande | lm-eval-harness | 53-58% | **60-68%** |
| KoBEST (5 tasks avg) | lm-eval-harness | 55-60% | **65-75%** |
| 반복률 (raw) | test_generation_params.py | 18% | **<8%** |
| 반복률 (+rep_penalty) | test_generation_params.py | ~5-8% | **<3%** |

### 실행 명령어

```bash
# ko_ifeval
lm_eval --model hf \
    --model_args pretrained=checkpoints/korean_3b_sft/checkpoint-BEST,dtype=bfloat16 \
    --tasks ko_ifeval --device cuda:0

# KoBEST
lm_eval --model hf \
    --model_args pretrained=checkpoints/korean_3b_sft/checkpoint-BEST,dtype=bfloat16 \
    --tasks kobest_boolq,kobest_copa,kobest_wic,kobest_hellaswag,kobest_sentineg \
    --device cuda:0
```

### 판단 기준

```
                    [3B SFT 평가 결과]
                          │
    ┌─────────────────────┼─────────────────────┐
    │                     │                     │
  ✅ PASS              ⚠️ PARTIAL            ❌ FAIL
 반복률<5%            반복률 5-10%          반복률>10%
 ko_ifeval>35%       ko_ifeval 25-35%      ko_ifeval<25%
    │                     │                     │
    ▼                     ▼                     ▼
 배포 준비             ORPO 적용             원인 분석
 (Phase 배포)        (Phase 5)           (데이터/아키텍처 점검)
```

---

## 7. 전체 타임라인

### 한눈에 보기

```
Day 0 (지금, 04:30)
  ├── [04:30] Config 작성 + sanity check        30분
  ├── [05:00] 🔥 사전학습 시작                   ← 오늘 밤 시작
  ├── [05:00] (병렬) korean_extra 토큰화 시작    8-12h
  │
Day 1 (내일)
  ├── [~07:00] 사전학습 진행 중... (~26시간)
  ├── [중간] 체크포인트 PPL 확인
  │
Day 1.5
  ├── [~07:00+26h = Day 2 07:00] 사전학습 완료
  ├── [07:00] SFT 시작                          2시간
  ├── [09:00] SFT 완료 → 평가
  ├── [09:30] 반복률 <5%? → 배포
  ├── [09:30] 반복률 5-10%? → ORPO            1-2시간
  ├── [11:30] ORPO 완료 → 최종 평가
  │
Day 2
  ├── 벤치마크 풀 스위트                        2시간
  ├── HuggingFace 업로드                        1시간
  ├── vLLM 서빙 테스트                          1시간
  └── 🎉 배포 완료
```

### 표 형식

| 단계 | 시작 | 소요 | 완료 | 의존성 |
|------|------|------|------|--------|
| **0. Config + Sanity** | Day 0 04:30 | 30분 | 05:00 | 없음 |
| **1. 사전학습** | Day 0 05:00 | **26시간** | Day 1 ~07:00 | Config |
| **(병렬) Extra 토큰화** | Day 0 05:00 | 8-12시간 | Day 0 ~17:00 | 없음 |
| **2. SFT** | Day 1 07:00 | **2시간** | Day 1 09:00 | 사전학습 완료 |
| **3. 1차 평가** | Day 1 09:00 | 30분 | Day 1 09:30 | SFT 완료 |
| **4. ORPO (조건부)** | Day 1 09:30 | 1-2시간 | Day 1 11:30 | 반복률 >5% |
| **5. 풀 벤치마크** | Day 1 11:30 | 2시간 | Day 1 13:30 | |
| **6. 배포** | Day 1 13:30 | 2시간 | Day 1 15:30 | 벤치마크 통과 |

---

## 8. 의사결정 트리

### Phase 1: 사전학습 중 (Step 5000, 10000, ...)

```
Loss 하강 중?
├── YES → 계속
└── NO (정체 3000+ steps)
    ├── 데이터 품질 문제? → PPL 필터 강화 + 재시작
    ├── LR 문제? → LR 반감 후 resume
    └── 모델 아키텍처? → d_model/n_layers 조정 (최후 수단)

PPL (한국어 텍스트)?
├── < 15 at 2 epoch → 정상
├── 15-20 at 2 epoch → 주의 (데이터 부족?)
└── > 20 at 2 epoch → 문제 (데이터 품질 or 하이퍼파라미터)

OOM?
├── batch_size 4→2, grad_accum 8→16
└── gradient checkpointing 확인
```

### Phase 2: SFT 후

```
반복률 (raw)?
├── < 5%  → ✅ 배포 가능! (ORPO 건너뜀)
├── 5-10% → ⚠️ ORPO 적용
├── 10-15% → 🟠 SFT 하이퍼파라미터 조정 후 재시도
└── > 15% → ❌ 사전학습 품질 문제 → Phase 1 재점검

ko_ifeval?
├── > 35% → ✅ 목표 달성
├── 25-35% → 🟡 데이터 augmentation 고려
└── < 25% → 🔴 3B에서도 이러면 데이터 문제 심각
```

### Phase 3: ORPO 후

```
반복률?
├── < 3% → ✅ 완료
├── 3-5% → 🟡 서빙 시 rep_penalty=1.05로 보완
└── > 5% → 🔴 preference 데이터 재검토
```

---

## 9. 예외 대응

| 시나리오 | 확률 | 대응 |
|---------|------|------|
| **OOM** | 5% | batch_size 4→2, grad_accum 2× |
| **Loss 발산** | 5% | LR 반감, grad_clip 0.5로 강화 |
| **GPU Hang / NCCL** | 10% | `pkill torchrun` → latest checkpoint에서 resume |
| **디스크 부족** | 3% | save_interval 2000→5000, 오래된 ckpt 삭제 |
| **사전학습 후 PPL >20** | 10% | 데이터 추가 (korean_extra) + extended training |
| **SFT 후 반복률 >15%** | 10% | 데이터 필터 재강화 + LR/epoch 조정 |
| **ORPO 후 품질 퇴행** | 15% | ORPO LR 낮추기 (5e-7 → 1e-7), beta 조정 |
| **FP8 수치 불안정** | 5% | BF16으로 폴백 (시간 1.5× 증가) |

### NCCL/GPU 복구 스크립트

```bash
# 프로세스 정리
pkill -f torchrun && sleep 5

# 최신 체크포인트 찾기
LATEST=$(ls -d checkpoints/korean_3b_fp8/checkpoint-[0-9]* 2>/dev/null \
  | sort -t- -k2 -n | tail -1)

# 재시작
bash scripts/run_pretrain.sh --config configs/korean_3b_fp8.yaml --resume "${LATEST}"
```

---

## 10. 1B에서 배운 교훈 체크리스트

### 학습 전 필수 확인

- [ ] **Dynamic padding 작동 확인**: `SFTDataset.__getitem__`이 가변 길이 텐서 반환, `dynamic_collate_fn` 배치별 패딩
- [ ] **EOS 보존 확인**: `grep -n "eos_token_id" data/sft_dataset.py` — 트렁케이션 시 강제 부착
- [ ] **Val split 존재**: `wc -l data/sft_v2/val.jsonl` → 10K+ 확인
- [ ] **데이터 오염 제거**: `</s>` 리터럴, Q/A 마커, 자체 반복 패턴 필터 적용됨
- [ ] **OpenOrca 가중치 ≤ 2.0**: prepare_sft_data.py에서 확인
- [ ] **프롬프트 포맷 통일**: 학습 = 추론 = `<|user|>/<|assistant|>`
- [ ] **Labels shift 정상**: trainer.py에서 `logits[t]` → `targets[t]` 직접 비교, labels에서 shift 처리

### 학습 중 필수 모니터링

- [ ] **Val loss 추적**: 매 eval_interval마다 기록, 3연속 상승 시 주의
- [ ] **Early stopping 활성화**: patience=5
- [ ] **Loss 0 감지**: 3 step 연속 loss < 0.01 → labels 버그 즉시 확인
- [ ] **Grad norm**: > 10이면 경고, > 50이면 중단

### 학습 후 필수 확인

- [ ] **올바른 포맷으로 생성 테스트**: `<|user|>\n{질문}\n<|assistant|>\n`
- [ ] **rep_penalty 없이 반복률 측정**: 목표 <10%
- [ ] **rep_penalty=1.1로 반복률**: 목표 <3%
- [ ] **벤치마크 실행**: ko_ifeval, KoBEST

### 절대 반복하지 말 것

- ❌ 학습/추론 포맷 불일치 상태로 평가하지 말 것
- ❌ Val split 없이 학습하지 말 것
- ❌ 특정 소스 5× 이상 업샘플링하지 말 것
- ❌ 2 epoch 미만으로 학습하지 말 것
- ❌ Dynamic padding 미작동 상태로 학습하지 말 것
- ❌ 반복률 측정 없이 "loss 낮으니 OK" 판단하지 말 것

---

## 🔥 오늘 밤 지금 당장 시작할 첫 번째 명령어

```bash
cd /PROJECT/0325120031_A/ghong/taketimes/llm-bang

# 1. 3B config 작성
cat > configs/korean_3b_fp8.yaml << 'YAML'
model:
  vocab_size: 64000
  d_model: 3072
  n_layers: 32
  n_heads: 24
  n_kv_heads: 8
  d_ffn: 8192
  max_seq_len: 4096
  rope_theta: 500000.0
  dropout: 0.0
  bias: false
  use_flash_attn: true
  use_fp8: true
train:
  max_steps: 34000
  batch_size: 4
  grad_accum_steps: 8
  lr: 1.5e-4
  min_lr: 1.5e-5
  weight_decay: 0.1
  warmup_steps: 2000
  max_grad_norm: 1.0
  log_interval: 10
  save_interval: 2000
  eval_interval: 500
  fp8_format: "MXFP8"
YAML

# 2. 사전학습 시작!
bash scripts/run_pretrain.sh --config configs/korean_3b_fp8.yaml
```

---

## ⚡ 가장 중요한 판단 포인트 3개

### 1️⃣ 사전학습 Step 5,000 (1 epoch 완료) — "기초 체력 확인"
- **PPL < 20?** → 정상, 계속
- **PPL > 20?** → 데이터 품질 or 하이퍼파라미터 문제. 즉시 진단

### 2️⃣ SFT 후 반복률 측정 — "3B의 진짜 실력"
- **<5%?** → 🎉 ORPO 없이 바로 배포. 대성공
- **5-10%?** → ORPO 1라운드로 해결 가능
- **>10%?** → 사전학습 품질 재검토 필요 (이 확률은 낮음)

### 3️⃣ ko_ifeval 점수 — "실사용 가능 수준?"
- **>35%?** → 3B 한국어 모델로서 경쟁력 있음. 배포
- **25-35%?** → 추가 SFT 데이터로 개선 여지 있음
- **<25%?** → 사전학습 데이터가 부족했을 가능성 → extended pretrain 고려

---

*"1B에서 배웠고, 3B에서 증명한다."*

*이 문서는 각 Phase 완료 시 실측 결과로 업데이트할 것.*
