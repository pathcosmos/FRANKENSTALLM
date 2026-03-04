# 3B Korean LLM 학습 파이프라인 — 전체 계획

**작성일:** 2026-02-27  
**서버:** 8× B200 192GB, NVSwitch, CUDA 13.1, PyTorch 2.10, TransformerEngine FP8  
**프로젝트:** `/PROJECT/0325120031_A/ghong/taketimes/llm-bang/`

---

## 1. 모델 아키텍처 (3B)

| 항목 | 1B (현재) | 3B (신규) | 근거 |
|------|-----------|-----------|------|
| d_model | 2048 | 3072 | LLaMA-3 3B 참고 |
| n_layers | 24 | 28 | |
| n_heads | 16 | 24 | |
| n_kv_heads | 4 (GQA 4:1) | 8 (GQA 3:1) | 더 큰 모델에서 KV 좀 더 여유 |
| d_ffn | 5472 | 8192 | ~2.67× d_model, 128배수 (FP8) |
| max_seq_len | 4096 | 4096 | 동일 |
| vocab_size | 64000 | 64000 | 동일 토크나이저 |
| 총 파라미터 | ~1.0B | ~3.0B | |

## 2. 사전학습 하이퍼파라미터

| 항목 | 1B 값 | 3B 값 | 근거 |
|------|-------|-------|------|
| **Learning Rate** | 2e-4 | **1.5e-4** | μP scaling ~1/√(3), LLaMA-3 3B 참고 |
| **LR Schedule** | cosine decay | cosine decay | 동일 |
| **Warmup Steps** | 2000 | 2000 | 57k의 3.5% (적절) |
| **Weight Decay** | 0.1 | 0.1 | 표준 |
| **Gradient Clip** | 1.0 | 1.0 | 표준 |
| **Batch Size (local)** | 8 | 8 | per GPU |
| **Grad Accum** | 4 | 4 | |
| **Eff Batch** | 1M tok/step | 1M tok/step | 8×8×4×4096 |
| **Max Steps** | 34,000 | **57,000** (60B tok) | Chinchilla: 3B → 60B min |
| **총 토큰** | 35.6B | **60B** (최소) / 95k steps=100B (권장) | |
| **Save Interval** | 500 | **2000** | 27GB/체크포인트 → 덜 자주 |
| **Eval Interval** | 200 | **500** | |
| **FP8** | MXFP8 | MXFP8 | B200 네이티브 |

## 3. 자원 예측

### 체크포인트
- model.pt: 3B × 1B(FP8) ≈ **3GB**
- optimizer.pt: 3B × 8B(FP32 states) ≈ **24GB**
- 체크포인트당 **~27GB**
- 2000 step 간격 → 최대 ~28개 = **756GB**
- /PROJECT 여유 19TB → **충분**

### VRAM
- 모델: ~6GB (FP8)
- Optimizer states: ~24GB
- Activations: ~40-60GB (batch 8, seq 4096)
- 총: ~80-90GB/GPU → B200 192GB로 **충분** (47% 사용)

### 학습 시간 예상
- 1B: 34k steps → ~12h (관찰값 기반)
- 3B: step당 ~3배 → step ~1.05s 예상
- 57k steps × 1.05s ≈ **~17h** (60B tokens)
- 95k steps × 1.05s ≈ **~28h** (100B tokens)
- 안전 마진 포함: **24~36h**

## 4. NCCL 최적화 (B200 NVSwitch)

```bash
export NCCL_IB_DISABLE=1          # 단일 노드, IB 불필요
export NCCL_ALGO=Ring,Tree        # 3B gradient 크기에 두 알고리즘 병행
export NCCL_PROTO=Simple          # NVLink bulk transfer 최적
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export NCCL_BUFFSIZE=134217728    # 128MB (1B의 64MB에서 증가)
export NCCL_P2P_LEVEL=NVL         # NVLink 직접 P2P
```

## 5. SFT 계획 (사전학습 완료 후)

### 데이터
- 1B SFT 데이터 (161k 샘플) — 기존 검증 완료
- 추가 고품질 데이터 고려:
  - Ko-Alpaca 확장
  - ShareGPT-ko 추가
  - 목표: **200k+ 샘플**

### 하이퍼파라미터

| 항목 | 1B SFT | 3B SFT | 근거 |
|------|--------|--------|------|
| LR | 2e-5 | **1e-5** | 더 큰 모델 → 더 낮은 LR |
| Batch (local) | 4 | 4 | |
| Grad Accum | 2 | 2 | |
| Eff Batch | 64 | 64 | |
| Max Steps | 9000 | **12000** | 3B는 수렴에 좀 더 필요 |
| Warmup | 300 | 500 | |
| Max Seq Len | 4096 | 4096 | |

### 1B SFT 교훈 반영
- ✅ Labels shift 버그 — sft_dataset.py 이미 수정됨
- ✅ 프로세스 중복 방지 — launch 스크립트에 pgrep 체크 추가
- ✅ Loss 0 감지 — 모니터링에 포함

### 예상 SFT 시간
- 12000 steps × ~0.8s/step ≈ **~3h**

## 6. ORPO 계획 (SFT 완료 후)

### 데이터
- **`maywell/ko_Ultrafeedback_binarized`** — Korean preference 데이터
- 다운로드:
  ```python
  from datasets import load_dataset
  ds = load_dataset("maywell/ko_Ultrafeedback_binarized")
  ```

### 설정
| 항목 | 값 | 근거 |
|------|-----|------|
| LR | 5e-6 | ORPO 표준, SFT보다 낮게 |
| β (ORPO lambda) | 0.1 | 논문 기본값 |
| Batch (local) | 2 | chosen+rejected 쌍 → 메모리 2배 |
| Grad Accum | 4 | eff_batch = 64 |
| Max Steps | ~3000-5000 | 데이터 크기에 따라 |
| Max Seq Len | 4096 | |

### 예상 ORPO 시간
- 5000 steps × ~1.5s/step (쌍 비교) ≈ **~2h**

## 7. 전체 타임라인

```
Phase 1: 사전학습 (60B tokens)
├─ 준비: configs, scripts 확인          ~30분
├─ 학습: 57k steps                     ~24-36h
└─ 평가: eval suite                    ~1h

Phase 2: SFT
├─ 데이터 준비: 161k+ 검증             ~30분
├─ 학습: 12k steps                     ~3h
└─ 평가                                ~30분

Phase 3: ORPO
├─ 데이터 다운로드+처리                 ~30분
├─ 학습: 3-5k steps                    ~2h
└─ 최종 평가                           ~1h

총 예상: 약 3-4일 (사전학습 포함)
        사전학습만: 24-36h
        SFT+ORPO:  ~6h
```

## 8. 예외 대응 플레이북 (3B 특화)

### 8-1. 서버 재시작이 필요한 경우
1. `Ctrl+C`로 graceful stop (SIGINT → 현재 step 완료 후 체크포인트 저장)
2. 안 되면 `kill -15 <PID>` → 10초 대기 → `kill -9`
3. 재시작 후: `bash scripts/launch_3b_pretrain.sh` (자동 resume 감지)

### 8-2. 체크포인트 손상
```bash
# 최근 체크포인트 무결성 확인
python -c "
import torch
ckpt = torch.load('checkpoints/korean_3b_fp8_run1/checkpoint-XXXXX/model.pt', weights_only=True)
print(f'Keys: {len(ckpt)}')
print(f'Total params: {sum(v.numel() for v in ckpt.values()):,}')
"
# 손상 시 → 이전 체크포인트로 resume
bash scripts/launch_3b_pretrain.sh --resume checkpoints/korean_3b_fp8_run1/checkpoint-YYYYY
```

### 8-3. NCCL Hang 감지
- `monitor_3b.sh` 로그 10분 멈춤 → CRITICAL 알림
- `--auto-restart` 옵션으로 자동 kill + 재시작 가이드
- 수동: `kill -9 $(pgrep -f pretrain.py)` → 재실행

### 8-4. 디스크 공간 부족
- `monitor_3b.sh --auto-cleanup`: MAX_CHECKPOINTS(15) 초과 시 오래된 것 자동 삭제
- 수동 정리:
  ```bash
  ls -d checkpoints/korean_3b_fp8_run1/checkpoint-* | sort -V | head -10 | xargs rm -rf
  ```

### 8-5. Loss 발산 (NaN / spike)
1. 즉시 중단
2. 최근 정상 체크포인트에서 resume
3. LR 50% 감소하여 재시작: `--lr 7.5e-5`
4. 반복 시 warmup 늘리기: `--warmup_steps 4000`

## 9. 모니터링 스크립트

### 실시간 감시
```bash
bash scripts/monitor_3b.sh                    # 기본 (60초 간격)
bash scripts/monitor_3b.sh --check-once       # 1회
bash scripts/monitor_3b.sh --auto-cleanup     # 자동 체크포인트 정리
bash scripts/monitor_3b.sh --auto-restart     # NCCL hang 시 자동 kill
```

### TensorBoard
```bash
tensorboard --logdir checkpoints/korean_3b_fp8_run1/tensorboard --port 6007
```

## 10. 실행 커맨드 요약

```bash
# Step 1: 사전학습 시작
bash scripts/launch_3b_pretrain.sh

# Step 1b: 모니터링 (별도 터미널)
bash scripts/monitor_3b.sh --auto-cleanup

# Step 2: 사전학습 완료 후 SFT (launch_sft.sh 3B 버전 필요)
BASE_CHECKPOINT=checkpoints/korean_3b_fp8_run1/checkpoint-XXXXX \
RUN_NAME=korean_3b_sft \
bash scripts/launch_sft.sh --lr 1e-5 --max_steps 12000 --warmup_steps 500

# Step 3: ORPO (별도 스크립트 필요 — train/orpo.py 작성 필요)
# TBD after SFT
```

---

**상태:** ✅ 파이프라인 설계 완료, 스크립트 작성 완료  
**다음 액션:** `bash scripts/launch_3b_pretrain.sh` 실행
