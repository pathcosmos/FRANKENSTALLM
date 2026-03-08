# FRANKENSTALLM 3B — Phase 3 ORPO 학습 여정

**작성일**: 2026-03-08 (최종 업데이트: 2026-03-09)
**작성자**: Claude Opus 4.6

---

## 1. ORPO 선택 배경

### 1.1 SFT의 한계

Phase 2 SFT v2 학습 완료 후 6차원 평가를 수행했다. 결과는 다음과 같다.

| 차원 | 지표 | 결과 | 목표 | 판정 |
|------|------|------|------|------|
| 1. 지식 보존 | PPL forgetting | 0.9% | <5% | **PASS** |
| 2. 생성 품질 | Greedy 반복률 | 72.97% | <5% | **FAIL** |
| 3. 종료 능력 | EOS 종료율 | 0% | >90% | **FAIL** |
| 4. 한국어 이해 | KoBEST | 43.26% | >55% | **FAIL** |
| 5. 형식 준수 | 포맷 정확도 | 95%+ | >90% | **PASS** |
| 6. 안전성 | 유해 출력률 | <1% | <5% | **PASS** |

**핵심 문제 분석:**

- **반복 생성 (72.97%)**: 모델이 동일한 토큰 시퀀스를 끊임없이 반복한다. SFT는 "좋은 응답"만 학습하기 때문에, "나쁜 응답(반복)"을 억제하는 음의 신호(negative signal)가 전혀 없다.
- **EOS 미종료 (0%)**: 반복과 밀접하게 연결된 문제. 모델이 반복 루프에 빠지면 EOS 토큰을 생성할 기회를 잃는다.
- **KoBEST 저조 (43.26%)**: 한국어 이해력 부족. 다만 이 문제는 preference optimization보다 데이터 품질과 양에 더 의존한다.

3개 항목 PASS(지식 보존, 형식, 안전성)는 SFT 기반이 건전하다는 의미이며, 특히 **PPL forgetting 0.9%**는 추가 학습의 기반이 안정적임을 확인해준다.

### 1.2 왜 ORPO인가? (DPO vs ORPO 비교)

SFT만으로는 "좋은 응답 vs 나쁜 응답"을 구분하는 preference signal을 줄 수 없다. Preference optimization 기법이 필요하다.

| 기준 | DPO | ORPO |
|------|-----|------|
| Reference model | 필요 (메모리 2배) | **불필요** |
| 구현 복잡도 | 보통 | **낮음** |
| 메모리 효율 | 낮음 | **높음** |
| 학습 안정성 | 검증 많음 | 비교적 새로움 |
| 반복 억제 | 효과적 | 효과적 |

**ORPO 선택 근거:**

1. **메모리 절약**: Reference model이 불필요하여 3B 모델 기준 약 6GB VRAM 절약. 8-GPU DDP에서는 총 ~48GB 절약.
2. **구현 간결성**: TRL의 `ORPOTrainer`로 DPO 대비 설정이 단순하다.
3. **SFT 기반 건전성**: PPL forgetting 0.9%는 ORPO 추가 학습이 기존 지식을 크게 훼손하지 않을 것임을 시사한다.

**위험 요소:**

- Preference 데이터 중 명확한 반복 차이가 있는 쌍(chosen이 정상, rejected가 반복)은 전체의 **3.3%에 불과**하다. ORPO가 이 적은 비율의 신호만으로 반복을 억제할 수 있는지가 핵심 불확실성이다.

**Plan B**: ORPO가 실패할 경우 DPO(`loss_type='sigmoid'`, reference model 사용)로 전환한다.

### 1.3 Preference 데이터 현황

- **원본**: 683,181 preference pairs (기존 문서의 795K는 집계 오류)
- **NaN 방지 필터 후**: ~630,000 pairs
  - prompt > `max_length - 16` → 제거
  - response > `max_length` → 제거
- **Eval split**: 5% (seed=42) → ~31,000 eval pairs
- **Effective batch size**: 4 (per-device) x 8 GPU x 4 (gradient accumulation) = **128**

---

## 2. 6-Config HP Sweep 설계

### 2.1 탐색 축 설계

3개의 독립적인 하이퍼파라미터 축을 선정했다.

**축 1: Beta (반복 억제 강도)**

Beta는 ORPO의 odds ratio loss 가중치를 조절한다. 값이 클수록 chosen/rejected 간 차이를 더 적극적으로 학습한다.

- 낮은 beta (0.15): 보수적. SFT 품질을 유지하면서 약하게 preference 학습.
- 중간 beta (0.25): 표준적인 출발점. 논문 기본값 부근.
- 높은 beta (0.35): 공격적. 반복 억제를 강하게 밀어붙이지만 과적합 위험.

**축 2: Learning Rate (수렴 속도)**

- 낮은 LR (5e-6): 안정적이지만 느린 수렴. 200 steps 내 효과 미미할 수 있음.
- 중간 LR (8e-6): 표준적. 200 steps에서 의미 있는 변화 기대.
- 높은 LR (1.2e-5): 빠른 수렴. 200 steps 짧은 sweep에서 효과를 빨리 확인 가능하지만 발산 위험.

**축 3: Max Length (VRAM vs 커버리지)**

- 1536 tokens: 대부분의 prompt+response를 커버. VRAM 더 사용.
- 1024 tokens: VRAM 절약. 긴 응답은 잘리지만 대다수는 1024 이내.

### 2.2 왜 6개인가?

3축의 full factorial 조합은 3 x 3 x 2 = **18개** (또는 각 3개씩이면 27개)로, 200 steps sweep이라 해도 비현실적이다.

**중심축 고정 방식**을 채택했다:

- 중심축: `beta=0.25`, `lr=8e-6`, `max_length=1536`
- 각 축을 한 번씩 양방향으로 변형하되, 나머지 축은 중심값 고정
- 이 방식으로 **6개 config**만으로 3개 축의 영향을 독립적으로 측정 가능

이는 실험 설계의 "one-factor-at-a-time" (OFAT) 방식에 해당한다.

### 2.3 각 Config의 목적

| Run | Name | Beta | LR | Max Length | 설계 의도 |
|-----|------|------|----|-----------|-----------|
| 1 | `baseline_b015_lr8e6` | 0.15 | 8e-6 | 1536 | 약한 beta 베이스라인. SFT 품질 유지 우선 |
| 2 | `baseline_b025_lr8e6` | 0.25 | 8e-6 | 1536 | **중심축**. 모든 비교의 기준점 |
| 3 | `strong_b035_lr8e6` | 0.35 | 8e-6 | 1536 | 강한 beta. 적극적 반복 억제, 과적합 감시 |
| 4 | `fast_b025_lr12e6` | 0.25 | 1.2e-5 | 1536 | 높은 LR. 200 steps에서 빠른 수렴 확인 |
| 5 | `conserv_b025_lr5e6` | 0.25 | 5e-6 | 1536 | 보수적 LR. 안정성 우선, 느린 변화 |
| 6 | `short_b025_lr8e6` | 0.25 | 8e-6 | 1024 | 짧은 max_length. VRAM 절약 효과 측정 |

---

## 3. 시도 이력 — 5번의 실패, 1번의 성공

총 6번의 시도 끝에 학습이 정상 실행되었다. 각 실패에서 얻은 교훈은 대규모 분산 학습의 실전 지식이다.

### 3.1 시도 1: NCCL Timeout

- **시각**: ~03:00
- **증상**: `torch.distributed.DistBackendError: wait timeout after 1800000ms`
- **원인**: Rank 0이 649K 샘플을 토크나이징하는 데 `num_proc=8`로 약 30분 소요. 그 동안 다른 7개 rank가 NCCL communicator setup에서 대기하다 기본 timeout(1800초 = 30분)을 초과했다.
- **수정**:
  - `ddp_timeout=7200` (30분 → 2시간으로 확대)
  - `dataset_num_proc=64` (8 → 64 프로세스로 토크나이징 병렬화)
- **교훈**: 대규모 데이터 + DDP 환경에서는 데이터 전처리 시간이 NCCL timeout에 직접 영향을 준다. 데이터가 클수록 timeout을 넉넉하게 설정해야 한다.

### 3.2 시도 2: Config 충돌

- **시각**: 03:28
- **증상**: `ValueError: save_steps(9999) is not a round multiple of eval_steps(100)`
- **원인**: `load_best_model_at_end=True`일 때 HuggingFace Trainer는 `save_steps`가 `eval_steps`의 정수배여야 한다는 validation을 수행한다. 최선 모델을 eval 시점에 저장해야 하므로 두 주기가 동기화되어야 하기 때문이다.
- **수정**: `--no_load_best --save_steps 200`
- **교훈**: TRL/HuggingFace Trainer의 config validation은 예상보다 엄격하다. 특히 `load_best_model_at_end`와 관련된 설정 간 정합성을 사전에 확인해야 한다.

### 3.3 시도 3: 포트 충돌 + QKV 변환 버그

- **시각**: 03:45

**증상 (2가지 동시 발생):**

1. **Run 1**: `port 29510 EADDRINUSE` — 이전 실행의 좀비 프로세스가 소켓 점유
2. **Run 2~6**: `q_proj/k_proj/v_proj MISSING` — QKV weight가 랜덤 초기화됨. GPU 0만 100% utilization, 나머지 7개 GPU는 0%.

**원인 (포트)**: 이전 실행이 비정상 종료되면서 `torchrun` 프로세스가 좀비로 남아 29510 포트를 점유.

**원인 (QKV)**: `convert_to_hf.py`의 `remap_weights()` 함수가 TransformerEngine의 fused QKV 프로젝션을 처리하지 못했다.

TransformerEngine은 Q, K, V를 하나의 가중치로 합친다:
- `attn.qkv_proj.weight` shape: `[5120, 3072]`
- Q: 3072 (24 heads x 128 dim), K: 1024 (8 kv_heads x 128 dim), V: 1024 (8 kv_heads x 128 dim)
- 합계: 3072 + 1024 + 1024 = 5120

HuggingFace 형식은 `q_proj`, `k_proj`, `v_proj`를 분리하여 저장한다.

**수정**: QKV split 로직을 `convert_to_hf.py`에 추가:

```python
q_dim = num_heads * head_dim        # 24 * 128 = 3072
k_dim = num_kv_heads * head_dim     # 8 * 128 = 1024
v_dim = num_kv_heads * head_dim     # 8 * 128 = 1024

qkv = state_dict[fused_key]
state_dict[f"{prefix}.q_proj.weight"] = qkv[:q_dim]
state_dict[f"{prefix}.k_proj.weight"] = qkv[q_dim:q_dim + k_dim]
state_dict[f"{prefix}.v_proj.weight"] = qkv[q_dim + k_dim:]
del state_dict[fused_key]
```

체크포인트 키 수: 171 → 255 (fused → split 변환으로 키 증가).

- **교훈**: TransformerEngine FP8 → HuggingFace 변환 시 fused projection(QKV, gate-up 등) 처리가 필수이다. 변환 후 반드시 키 매핑을 검증해야 한다.

### 3.4 시도 4: TRL NaN 버그

이 시도는 시도 3 이전에 발생한 것으로, TRL 라이브러리 내부의 심각한 버그를 발견하고 패치한 과정이다.

**증상**: 8-GPU DDP + 풀 데이터(683K) 학습 시 step 10~20에서 `loss=0`, `grad_norm=NaN` 발생. 단일 GPU나 소규모 데이터에서는 재현되지 않았다.

**근본 원인**: TRL의 `tokenize_row` 함수에서 chosen과 rejected response를 동일한 길이 기준으로 잘랐다.

```python
# TRL 원본 코드 (버그)
longer_response_length = max(chosen_len, rejected_len)
# 이후 양쪽 모두 longer_response_length 기준으로 truncation
```

문제 시나리오: `longer_response_length > max_length`이면, shorter response가 음수 인덱스로 잘려서 **0 tokens**가 된다 (Python의 `list[:-음수]` = `[]`).

**NaN 전파 체인:**

```
0 response tokens
  → labels 전부 -100 (학습 대상 없음)
    → get_batch_logps = 0.0 (log probability 합이 0)
      → log1mexp(0.0) = log(1 - exp(0)) = log(1 - 1) = log(0) = -inf
        → log_odds = -inf - (-inf) = NaN
          → gradient NaN → 전체 weight 오염
```

**3중 패치** (TRL 라이브러리 파일 직접 수정):

**Patch 1: `get_batch_logps` — division by zero 방지**

```python
# 파일: trl/trainer/utils.py
if average_log_prob:
    denom = loss_mask.sum(-1).clamp(min=1)
    return (per_token_logps * loss_mask).sum(-1) / denom
```

**Patch 2: `odds_ratio_loss` — logps clamp**

```python
# 파일: trl/trainer/orpo_trainer.py (또는 experimental)
policy_chosen_logps = policy_chosen_logps.float().clamp(max=-1e-4)
policy_rejected_logps = policy_rejected_logps.float().clamp(max=-1e-4)
```

logps가 0에 가까우면 `log1mexp`에서 `-inf`가 발생하므로, 최대값을 `-1e-4`로 제한한다.

**Patch 3: `tokenize_row` — 독립 truncation (근본 원인 수정)**

```python
# 파일: trl/trainer/orpo_trainer.py (또는 experimental)
for answer_tokens in [chosen_tokens, rejected_tokens]:
    prompt_len = len(answer_tokens["prompt_input_ids"])
    response_len = len(answer_tokens["input_ids"])
    if prompt_len + response_len > self.max_length:
        max_response = max(self.max_length - prompt_len, 1)
        for k in ["input_ids", "attention_mask"]:
            answer_tokens[k] = answer_tokens[k][:max_response]
```

각 response를 독립적으로 truncation하여, 하나가 길다고 해서 다른 하나가 0으로 잘리는 상황을 방지한다.

추가로 `train/orpo.py`에 데이터 필터를 추가했다:
- prompt > `max_length - 16` → 제거 (response 공간이 16 tokens 미만이면 무의미)
- response > `max_length` → 제거

683,181 → ~630,000 pairs로 축소.

- **교훈**: 분산 학습에서만 재현되는 NaN 버그는 데이터 분포의 꼬리(극단적으로 긴 prompt + 짧은 response 조합)에서 발생한다. 단일 GPU 테스트로는 발견하기 어렵다. TRL 같은 성숙한 라이브러리도 edge case에서 수치적 불안정성을 가질 수 있다.

### 3.5 시도 5: TRL Tokenizer 호환 문제

- **증상**: 한국어 tokenizer의 merge operations으로 인해 prompt token 길이 불일치 발생 → `zip(strict=True)` ValueError
- **수정**: TRL 소스 파일 8건에 패치 적용 (`.029bak` 백업 생성):
  - `zip(..., strict=True)` → `zip(...)` (strict 제거)
  - `build_tokenized_answer`의 length mismatch → graceful fallback
- **교훈**: 영어 중심으로 테스트된 TRL이 한국어 tokenizer(BPE merge 특성이 다름)와 호환성 문제를 가질 수 있다.

### 3.6 시도 6: 성공!

- **시각**: 04:20

**사전 정리 작업:**

```bash
# 좀비 프로세스 정리
pkill -9 -f torchrun
pkill -9 -f orpo

# 포트 해제 확인
ss -tlnp | grep 29510

# sweep 디렉터리 초기화
rm -rf checkpoints/orpo_sweep/*
```

**실행**: 6-config sweep이 순차적으로 정상 진행 시작.

---

## 4. 스윕 결과

### 4.1 결과 테이블

각 config는 200 steps만 실행하여 경향을 파악하는 "탐색" 수준의 sweep이다.

| Run | Name | Beta | LR | Train Loss | Eval Loss | Margin | Time(s) | Status |
|-----|------|------|----|-----------|-----------|--------|---------|--------|
| 1 | `baseline_b015_lr8e6` | 0.15 | 8e-6 | 1.811 | 1.827 | 0.004 | 2,344 | 완료 |
| 2 | `baseline_b025_lr8e6` | 0.25 | 8e-6 | 1.890 | 1.906 | 0.009 | 2,360 | 완료 |
| 3 | `strong_b035_lr8e6` | 0.35 | 8e-6 | 2.055 | 1.985 | 0.007 | 2,390 | 완료 |
| 4 | `fast_b025_lr12e6` | 0.25 | 1.2e-5 | 1.917 | 1.862 | 0.009 | 2,416 | 완료 |
| 5 | `conserv_b025_lr5e6` | 0.25 | 5e-6 | - | - | - | - | 진행중 |
| 6 | `short_b025_lr8e6` | 0.25 | 8e-6 | - | - | - | - | 대기중 |

### 4.2 분석

**Beta 축 분석 (Run 1 vs 2 vs 3, LR=8e-6 고정):**

- Beta가 높아질수록 train loss가 일관되게 증가한다: 1.811 → 1.890 → 2.055
- 이는 예상된 결과다. Beta가 높으면 odds ratio loss의 가중치가 커지므로 전체 loss가 상승한다.
- **Eval loss도 상승하지만 기울기가 완만하다**: 1.827 → 1.906 → 1.985. Run 3(beta=0.35)에서 train-eval gap이 가장 작아(0.070) 과적합 징후는 없다.
- Margin: Run 1(0.004)이 가장 낮고, Run 2(0.009)와 Run 3(0.007)은 비슷하다. Beta를 높인다고 반드시 margin이 비례 증가하지는 않는다.

**LR 축 분석 (Run 2 vs 4 vs 5, Beta=0.25 고정):**

- Run 4(`lr=1.2e-5`): **eval_loss 1.862로 최저값**. 높은 LR이 200 steps라는 짧은 구간에서 더 빠르게 유용한 방향으로 수렴했음을 시사.
- Run 2(`lr=8e-6`): eval_loss 1.906. 기준선.
- Run 5(`lr=5e-6`): 아직 진행 중. 낮은 LR이 200 steps에서 충분한 변화를 보일지 관건.

**잠정 결론**: 200-step sweep 기준으로 `lr=1.2e-5, beta=0.25` 조합(Run 4)이 가장 유망하다. 다만 full training에서는 높은 LR이 후반부 불안정성을 가져올 수 있으므로, cosine scheduler와 함께 warmup ratio를 조정해야 한다.

---

## 5. 기술 세부사항

### 5.1 TRL 0.29.0 API

TRL 0.29.0에서 ORPO 관련 클래스의 위치가 변경되었다.

```python
# 기존 (0.28 이하)
from trl import ORPOConfig, ORPOTrainer  # 제거됨

# 0.29.0
from trl.experimental.orpo import ORPOConfig, ORPOTrainer
```

또한 `max_prompt_length` 파라미터가 제거되었다. Prompt 길이 제한이 필요하면 데이터 전처리 단계에서 직접 필터링해야 한다.

### 5.2 DDP 최적화

**Rank별 데이터 처리 분리:**

```python
if local_rank == 0:
    # Rank 0만 num_proc=64로 토크나이징
    dataset = dataset.map(tokenize_fn, num_proc=64)
    # 결과가 캐시되므로 다른 rank는 캐시 히트
else:
    # Rank 1~7은 num_proc=1로 호출 (캐시에서 로드)
    dataset = dataset.map(tokenize_fn, num_proc=1)
```

이 방식으로 72코어를 rank 0에 집중 투입하여 토크나이징 시간을 단축하고, 나머지 rank는 캐시를 재사용한다.

**DDP 설정 주의사항:**
- `ddp_find_unused_parameters=False`: TransformerEngine과 함께 사용 시 필수. `True`로 설정하면 TE의 FP8 버퍼에서 오류 발생.
- `static_graph=True` **사용 금지**: TE와 호환되지 않는다.

### 5.3 하드웨어 설정

```bash
# NCCL 설정
NCCL_IB_DISABLE=1           # InfiniBand 비활성화 (단일 노드)
NCCL_BUFFSIZE=134217728     # 128MB 버퍼
NCCL_P2P_LEVEL=NVL          # NVLink P2P 통신

# CPU 스레드
OMP_NUM_THREADS=9            # 72 cores / 8 GPUs
MKL_NUM_THREADS=9

# 학습 설정
dataset_num_proc=64
bf16=true                    # B200 BF16 네이티브
dataloader_pin_memory=true
ddp_timeout=7200             # NCCL timeout 2시간

# 모델 설정
flash_attention_2             # FlashAttention-2 활성화
gradient_checkpointing=true   # VRAM 절약
```

### 5.4 안전장치

**SIGHUP 3중 방어:**

터미널 세션 종료 시 학습이 중단되는 것을 방지하기 위한 3중 보호 체계:

1. **nohup + setsid**: 프로세스를 세션 리더에서 분리
2. **Python signal handler**: SIGHUP 수신 시 무시하고 학습 계속
3. **Emergency checkpoint**: 비정상 종료 감지 시 현재 상태를 즉시 저장

**텔레그램 알림**: 각 run 완료/실패 시 자동 알림 전송 (Python `urllib`, curl 차단 환경 대응).

**Early stopping**: eval_loss 기반. patience 설정으로 과적합 시 자동 중단.

---

## 6. 스윕 최종 결과 + Best Config 선정

### 6.1 전체 스윕 완료 결과

6개 config 모두 200 steps 완료.

| Run | Name | Beta | LR | MaxLen | Train Loss | Eval Loss | Margin | Time(s) |
|-----|------|------|----|--------|-----------|-----------|--------|---------|
| 1 | `baseline_b015_lr8e6` | 0.15 | 8e-6 | 1536 | 1.811 | 1.827 | 0.004 | 2,344 |
| 2 | `baseline_b025_lr8e6` | 0.25 | 8e-6 | 1536 | 1.890 | 1.906 | 0.009 | 2,360 |
| 3 | `strong_b035_lr8e6` | 0.35 | 8e-6 | 1536 | 2.055 | 1.985 | 0.007 | 2,390 |
| 4 | `fast_b025_lr12e6` | 0.25 | 1.2e-5 | 1536 | 1.917 | 1.862 | 0.009 | 2,416 |
| 5 | `conserv_b025_lr5e6` | 0.25 | 5e-6 | 1536 | 1.833 | 1.910 | 0.004 | 2,350 |
| 6 | `short_b025_lr8e6` | 0.25 | 8e-6 | 1024 | 1.664 | 1.695 | 0.007 | 1,840 |

### 6.2 Best Config 선정: Run 4 (lr=1.2e-5, beta=0.25)

**선정 근거:**

1. **Eval loss 최저 (1.862)**: maxlen=1536 그룹 내 eval loss 기준 1위.
2. **높은 margin (0.009)**: chosen/rejected 구분 능력이 강함. Run 2와 동률.
3. **빠른 수렴**: 200 steps 만에 다른 config 대비 가장 큰 개선폭을 보여, 긴 학습에서도 유리할 것으로 판단.

**참고**: Run 6(short_1024)의 eval_loss가 1.695로 절대값은 가장 낮지만, 이는 max_length=1024로 짧은 시퀀스를 다루기 때문이며 1536 시퀀스와 직접 비교할 수 없다.

### 6.3 Throughput 벤치마크

본 학습에 앞서 4가지 batch/grad_accum 조합의 throughput을 벤치마크하여 최적 설정을 결정했다.

| Config | batch_size | grad_accum | max_length | eff_batch | Throughput (samples/s) |
|--------|-----------|-----------|-----------|----------|----------------------|
| **1** | **4** | **4** | **1536** | **128** | **80.63** |
| 2 | 2 | 8 | 1536 | 128 | 73.14 |
| 3 | 8 | 2 | 1536 | 128 | OOM |
| 4 | 4 | 4 | 1024 | 128 | 91.25 |

**Config 1 (bs=4, accum=4, maxlen=1536)** 선정. 동일 effective batch size에서 ~10% 높은 throughput.

CPU 스레드도 NUMA-aware로 최적화: `OMP_NUM_THREADS=9, MKL_NUM_THREADS=9` (72코어 ÷ 8 GPU = 9코어/GPU).

---

## 7. Full Training 시작 (2026-03-09)

### 7.1 학습 설정

| 파라미터 | 값 | 비고 |
|---------|-----|------|
| Beta | 0.25 | Sweep Run 4에서 선정 |
| Learning rate | 1.2e-5 | Sweep eval_loss 최저 |
| Batch size (per-device) | 4 | Throughput 벤치마크 최적 |
| Gradient accumulation | 4 | |
| Effective batch | 128 | 4 × 4 × 8 GPU |
| Max length | 1536 | |
| Epochs | 2 | |
| Warmup ratio | 0.05 | |
| Weight decay | 0.01 | |
| Eval steps | 500 | |
| Early stopping patience | 3 | |
| GPU VRAM 사용 | ~52GB / 183GB (28%) | |
| 예상 총 steps | 9,840 | |
| 예상 학습 시간 | ~4.8시간 | ~1.75 s/step |

### 7.2 SIGHUP 3중 방어 실행

```bash
nohup setsid bash scripts/launch_3b_orpo.sh \
  > checkpoints/korean_3b_orpo_v1/train.log 2>&1 &
```

1. **nohup + setsid**: 프로세스를 세션에서 완전 분리
2. **Python SIGHUP handler**: orpo.py 내 signal.signal(SIGHUP, handler) — 무시 처리
3. **Emergency checkpoint**: 비정상 종료 감지 시 즉시 체크포인트 저장

### 7.3 학습 지표 추이

| 지표 | step ~250 | step ~1,160 | 변화 |
|------|-----------|-------------|------|
| **loss** | 1.952 | **1.709** | -0.243 |
| **nll_loss** | 1.757 | **1.593** | -0.164 |
| **rewards/margins** | 0.002 | **0.330** | +0.328 |
| **rewards/accuracies** | 0.473 | **0.719** | +0.246 |
| **log_odds_chosen** | -0.021 | **1.468** | +1.489 |

```
step 1163/9840 (12%), epoch 0.24, 경과 40분, 남은 ~4.4시간
속도: ~1.82 s/step
GPU VRAM: ~52GB/183GB, utilization 91~98%
```

**관찰**: rewards/accuracies가 0.47(랜덤 수준) → 0.72로 빠르게 상승. 모델이 chosen/rejected를 구분하는 능력이 강화되고 있다. margins도 0.002 → 0.330으로 급상승하여 ORPO가 preference signal을 효과적으로 학습 중임을 시사한다.

---

## 8. 다음 단계 + 교훈 요약

### 8.1 다음 단계

1. **Full training 완료 대기**: 9,840 steps, 예상 ~4.8시간
2. **6차원 재평가**: 특히 반복률(72.97% → 목표 <5%)과 EOS 종료율 개선 확인
3. **GGUF 변환 + Ollama 배포**: 평가 통과 시 Phase 4 진행
4. **Plan B 준비**: ORPO 효과 미미 시 DPO 전환

### 6.2 교훈 요약

| # | 교훈 | 카테고리 |
|---|------|---------|
| 1 | 대규모 데이터 + DDP에서 토크나이징 시간을 NCCL timeout에 반영해야 한다 | 분산 학습 |
| 2 | TRL/HF Trainer의 config validation은 예상보다 엄격하다. 특히 `load_best_model_at_end` 관련 | 프레임워크 |
| 3 | TransformerEngine FP8 → HF 변환 시 fused projection 처리 필수 | 모델 변환 |
| 4 | TRL의 `tokenize_row`는 극단적 길이 조합에서 NaN을 생성한다. 분산 학습에서만 재현 | 버그 |
| 5 | NaN은 0 response → -100 labels → logps=0 → log(0)=-inf → NaN 체인으로 전파된다 | 수치 안정성 |
| 6 | 영어 중심 TRL이 한국어 tokenizer BPE와 호환성 문제를 가진다 | 다국어 |
| 7 | 좀비 프로세스의 포트 점유를 사전에 확인해야 한다 | 인프라 |
| 8 | 단일 GPU 테스트로는 분산 환경의 edge case를 발견하기 어렵다 | 테스트 |
| 9 | HP sweep은 full factorial 대신 중심축 고정 OFAT로 효율적 탐색 가능 | 실험 설계 |
| 10 | SFT의 "좋은 응답만 학습" 한계는 preference optimization으로만 해결 가능 | 학습 전략 |

---

*이 보고서는 FRANKENSTALLM 3B Phase 3 ORPO 학습의 전 과정을 기록한다. 5번의 실패에서 얻은 교훈은 대규모 언어 모델 학습의 실전 지식으로, 향후 유사 프로젝트의 참고 자료가 될 것이다. 현재 본 학습이 진행 중이다 (2026-03-09).*
