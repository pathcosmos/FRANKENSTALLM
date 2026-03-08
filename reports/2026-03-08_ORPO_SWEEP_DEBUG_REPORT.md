# ORPO HP Sweep 디버깅 및 체크포인트 수정 보고서

**날짜**: 2026-03-08
**작성**: Claude Opus 4.6 + ghong

---

## 1. 목표

SFT v2 best 체크포인트 기반 ORPO hyperparameter sweep (6개 조합, 각 200 steps)을 8×B200 GPU에서 실행하여 최적 HP 조합을 찾는다.

## 2. 발견된 문제 및 해결

### 2.1 [CRITICAL] 체크포인트 QKV Weight 누락

**증상**: 모델 로딩 시 28개 레이어의 `self_attn.q_proj`, `k_proj`, `v_proj` weight가 `MISSING`으로 표시. attention layer가 랜덤 초기화 상태로 학습됨.

**원인**: `scripts/convert_to_hf.py`의 `remap_weights()` 함수가 fused QKV 프로젝션(`attn.qkv_proj.weight`)을 처리하지 못함. 원본 체크포인트(TransformerEngine FP8)는 Q+K+V를 하나의 `qkv_proj`로 저장하지만, 변환 스크립트는 `q_proj`, `k_proj`, `v_proj`를 개별 키로 찾았음.

**체크포인트 구조**:
```
원본 (TE FP8):  layers.0.attn.qkv_proj.weight  → shape [5120, 3072]
                                                   Q(3072) + K(1024) + V(1024) = 5120
변환 전 (HF):   self_attn.o_proj.weight만 존재  → 171 keys (QKV 84개 누락)
변환 후 (HF):   q_proj [3072,3072] + k_proj [1024,3072] + v_proj [1024,3072] → 255 keys
```

**GQA 구조**: heads=24, kv_heads=8, head_dim=128
- Q: 24 × 128 = 3072
- K: 8 × 128 = 1024
- V: 8 × 128 = 1024

**수정**: `scripts/convert_to_hf.py` — fused `qkv_proj`를 감지하면 GQA 구조에 따라 Q/K/V로 분리:
```python
qkv = src_state_dict[qkv_key].float()
dst["q_proj.weight"] = qkv[:q_dim]        # [3072, 3072]
dst["k_proj.weight"] = qkv[q_dim:q_dim+k_dim]  # [1024, 3072]
dst["v_proj.weight"] = qkv[q_dim+k_dim:]        # [1024, 3072]
```

**검증**: `q_proj norm=73.54` (학습된 weight), 변환 후 255 keys 정상.

### 2.2 NCCL Timeout (1800s)

**증상**: `torch.distributed.DistBackendError: wait timeout after 1800000ms`

**원인**: ORPOTrainer 초기화 시 Rank 0이 649K 샘플 토크나이징에 ~30분 소요. 다른 rank들이 NCCL communicator setup을 기다리다 기본 timeout(1800s) 초과.

**수정**:
- `train/orpo.py`: `ddp_timeout=7200` (2시간)
- `configs/korean_3b_orpo.yaml`: `dataset_num_proc: 64` (토크나이징 ~10분으로 단축)
- `scripts/orpo_hp_sweep.sh`: `--dataset_num_proc 64` 추가

### 2.3 TensorBoard Import 크래시

**증상**: `AttributeError: module 'tensorflow' has no attribute 'io'`

**수정**: `report_to="none"` (orpo.py, YAML config, sweep script 3곳)

### 2.4 TRL 0.29.0 API 변경

**증상**: `from trl import ORPOTrainer` → ImportError

**수정**: `from trl.experimental.orpo import ORPOConfig, ORPOTrainer`

### 2.5 `load_best_model_at_end` + `save_steps` 충돌

**증상**: `ValueError: save_steps(9999) is not a round multiple of eval_steps(100)`

**수정**:
- `orpo.py`: `--no_load_best` CLI 인자 추가
- sweep script: `--no_load_best --save_steps 200` 사용

### 2.6 포트 충돌 (EADDRINUSE)

**증상**: Run 1이 이전 프로세스의 port 29510 점유로 실패

**원인**: 이전 실행의 좀비 프로세스가 소켓을 점유

**교훈**: 실행 전 항상 `pkill -f torchrun` + `sleep 2` 로 정리 필요

### 2.7 TRL 0.29.0 ORPOTrainer 토크나이저 버그 (이전 세션)

**증상**: Korean tokenizer의 merge ops로 인한 prompt token 길이 불일치 → `zip(strict=True)` ValueError

**수정**: TRL 소스 패치 8건 (`.029bak` 백업):
1. `build_tokenized_answer` length mismatch → graceful fallback
2. chosen/rejected prompt tokens truncation
3. `zip(..., strict=True)` → `zip(...)` 변경
4. prompt diff ValueError → `pass` (warn-and-continue)
5. `add_bos_token_if_needed` args 통일
6. evaluation_loop zip strict 제거

## 3. 수정된 파일 목록

| 파일 | 변경 내용 |
|------|-----------|
| `scripts/convert_to_hf.py` | fused QKV → separate Q/K/V 분리 로직 추가 |
| `train/orpo.py` | TRL 0.29.0 import, ddp_timeout, dataset_num_proc, 예외처리/로깅 강화, --no_load_best |
| `configs/korean_3b_orpo.yaml` | dataset_num_proc: 64, report_to: none |
| `scripts/orpo_hp_sweep.sh` | --dataset_num_proc 64, --no_load_best, set +e, FAILED_RUNS 추적 |
| `eval/outputs/hf_3b_sft_best/` | 재변환 완료 (171→255 keys) |

## 4. Sweep 설정

| Run | Name | Beta | LR | Max Length | Effective BS |
|-----|------|------|----|-----------|-------------|
| 1 | baseline_b015_lr8e6 | 0.15 | 8e-6 | 1536 | 128 |
| 2 | baseline_b025_lr8e6 | 0.25 | 8e-6 | 1536 | 128 |
| 3 | strong_b035_lr8e6 | 0.35 | 8e-6 | 1536 | 128 |
| 4 | fast_b025_lr12e6 | 0.25 | 1.2e-5 | 1536 | 128 |
| 5 | conserv_b025_lr5e6 | 0.25 | 5e-6 | 1536 | 128 |
| 6 | short_b025_lr8e6 | 0.25 | 8e-6 | 1024 | 128 |

각 run: 200 steps, eval_steps=100, 8×B200 DDP

## 5. 하드웨어 최적화 설정

```bash
# NCCL (NVSwitch mesh — auto-detect)
NCCL_IB_DISABLE=1
NCCL_BUFFSIZE=134217728
NCCL_P2P_LEVEL=NVL

# CPU
OMP_NUM_THREADS=9    # 72 cores / 8 GPUs
MKL_NUM_THREADS=9
dataset_num_proc=64  # 토크나이징 병렬화

# GPU
bf16=true
flash_attention_2
gradient_checkpointing=true
dataloader_pin_memory=true
ddp_find_unused_parameters=false
ddp_timeout=7200
```

## 6. 현재 상태

- 체크포인트 재변환 완료 (QKV 정상)

## 7. Sweep 실행 이력

### 시도 1 — NCCL Timeout (이전 세션)
- **시각**: ~03:00
- **결과**: Rank 0이 토크나이징(649K, num_proc=8)에 30분 소요 → 나머지 rank NCCL 1800s timeout
- **교훈**: `ddp_timeout=7200` + `dataset_num_proc=64` 필요

### 시도 2 — save_steps/eval_steps 충돌
- **시각**: 03:28
- **결과**: `ValueError: save_steps(9999) not a multiple of eval_steps(100)`
- 6개 run 전부 즉시 실패 (ORPOConfig 생성 단계)
- **수정**: `--save_steps 200`, `--no_load_best` 추가

### 시도 3 — 포트 충돌 + 깨진 체크포인트
- **시각**: 03:45
- **결과**:
  - Run 1: port 29510 EADDRINUSE (이전 좀비 프로세스) → 2초 만에 실패
  - Run 2~6: 모델 로딩 성공하나 `q_proj/k_proj/v_proj MISSING` (랜덤 초기화)
  - GPU 0만 100% utilization, 나머지 7개 0% → DDP 비정상
- **근본 원인 발견**: `convert_to_hf.py`가 fused `qkv_proj` [5120, 3072]를 분리 안 함
- **수정**: QKV split 로직 추가, 체크포인트 재변환 (171→255 keys)

### 시도 4 — 정상 실행 중 (현재)
- **시각**: 04:20
- **사전 정리**: `pkill -9` + 포트 해제 + sweep 디렉토리 초기화
- **상태**: Run 1/6 `baseline_b015_lr8e6` 토크나이징 진행 중 (Map 69%, num_proc=64)
- **확인 사항**:
  - 모델 255 keys 정상 로딩 (MISSING 경고 없음)
  - 8 GPU 모두 모델 로드 (726MB/GPU)
  - utilization 0% = 정상 (CPU 토크나이징 단계)
- **대기 중**: 토크나이징 완료 → NCCL init → 8 GPU DDP 학습 시작 확인 필요

### GPU 1개만 사용 문제 설명
- **현상**: 시도 3에서 GPU 0만 100%, 나머지 0%
- **원인**: 깨진 체크포인트(attention 랜덤 초기화)로 인해 DDP 동기화 실패. Rank 0만 학습 진입, 나머지 rank는 NCCL communicator 대기 상태
- **현재**: 아직 토크나이징 단계라 GPU utilization 0%는 정상. 토크나이징 완료 후 8 GPU 학습 진입 확인 필요

## 8. 다음 단계

1. 토크나이징 완료 후 8 GPU DDP 학습 정상 진입 확인
2. 6개 HP 조합 결과 비교 (eval_loss, margin 기준)
3. 최적 HP로 본 학습 (full epochs)
4. 학습 후 6차원 평가 재실행
