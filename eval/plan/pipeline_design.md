# SFT 파이프라인 완전 재설계 보고서

> 작성일: 2026-02-26  
> 목표: 반복률 <5%, ko_ifeval 20-30%  
> 모델: 1B 한국어 LLM, 188K 학습 샘플

---

## 1. 현재 코드 검증 결과

### ✅ 올바르게 수정된 것들

| 항목 | 상태 | 근거 |
|------|------|------|
| Dynamic padding | ✅ 정상 | `SFTDataset.__getitem__`이 raw-length 텐서 반환, `dynamic_collate_fn`이 batch-local max + 64-align 패딩 |
| EOS 강제 부착 | ✅ 정상 | truncation 시 `response_ids[-1] = self.eos_token_id` 강제 |
| Enhanced quality filter | ✅ 적용됨 | `_enhanced_quality_filter`가 `quality_filter()` 안에서 호출됨 (EOS 리터럴, Q/A 오염, 50자 미만) |
| Val data 연결 | ✅ 정상 | `launch_sft.sh`에 `VAL_DATA="data/sft/val.jsonl"`, `--val_data` 전달됨 |
| Val loss 기록 | ✅ 구현됨 | `trainer.py`의 `_run_validation()` + `eval_interval=250` step마다 실행 |
| Best checkpoint | ✅ 구현됨 | `val_loss < self._best_val_loss` 시 suffix="best" 저장 |
| NEFTune | ✅ 적용됨 | `noise_alpha=10.0`으로 embedding hook 등록 |
| Gradient checkpointing | ✅ 적용됨 | `model.gradient_checkpointing_enable()` 호출 |

### ⚠️ 남은 문제 (수정 필요)

#### 🔴 P0: Labels off-by-one 버그

```python
# sft_dataset.py line ~168
resp_label_start = max(0, resp_start - 1)  # ← 이게 문제
resp_label_end   = resp_label_start + len(response_ids)
labels[resp_label_start:resp_label_end] = torch.tensor(response_ids, dtype=torch.long)
```

**문제**: `resp_start - 1`은 프롬프트의 마지막 토큰 위치에 response_ids[0]을 넣음.  
이는 next-token prediction에서 "프롬프트 마지막 토큰이 들어오면 response 첫 토큰을 예측하라"는 의미로 **의도적 설계**일 수 있음.

**그러나**: 이 경우 labels의 길이가 `len(response_ids)`인데 `resp_label_start`부터 시작하므로 response의 마지막 토큰(EOS)에 대한 label이 한 칸 앞으로 밀려, **마지막 response 토큰의 label이 누락될 수 있음**.

실제로:
- `resp_start = len(prompt_ids)` (예: 50)
- `resp_label_start = 49`
- `resp_label_end = 49 + len(response_ids)`
- `labels[49:49+R] = response_ids[0:R]`

이렇게 하면 position 49의 label = response_ids[0], position 50의 label = response_ids[1], ... 이건 autoregressive LM 관점에서:
- position 49 입력 = prompt 마지막 토큰 → 예측해야 할 것 = response 첫 토큰 ✅
- position (49+R-1) 입력 = response[R-2] → 예측해야 할 것 = response[R-1] (=EOS) ✅
- position (49+R) 입력 = response[R-1] (=EOS) → label = -1 (무시) ✅

**결론: 이 로직은 실제로 올바름.** next-token prediction의 teacher-forcing에서 shift가 모델 내부가 아닌 labels에서 이루어지는 패턴. 단, **모델의 forward가 logits[t]로 position t+1을 예측하는지 확인 필요**. 대부분 decoder LM은 `loss = CE(logits[:, :-1], labels[:, 1:])` 형태를 씀.

**확인 사항**: `trainer.py`의 `_compute_loss`를 보면:
```python
nn.functional.cross_entropy(logits.view(B*T, V), targets.view(B*T), ignore_index=-1)
```
여기서 **shift가 없음**. 즉 `logits[t]`와 `targets[t]`를 직접 비교함.

일반적으로 decoder LM의 logits[t]는 position t의 hidden state로 t+1번째 토큰을 예측. 따라서:
- `logits[t]` = position t 다음에 올 토큰 예측
- `targets[t]`가 t+1 토큰이어야 함

**현재 코드의 labels 배치:**
- `labels[resp_start - 1] = response_ids[0]` → logits[resp_start-1]로 response[0] 예측 ✅ (prompt 마지막 토큰 → 첫 응답)
- `labels[resp_start - 1 + k] = response_ids[k]` → logits[prompt_last + k]로 response[k] 예측

이것이 맞으려면 `input_ids[resp_start - 1 + k]`가 response_ids[k]의 **이전** 토큰이어야 함.
- k=0: `input_ids[resp_start - 1]` = prompt 마지막 토큰, label = response[0] ✅
- k=1: `input_ids[resp_start]` = response[0], label = response[1] ✅
- k=R-1: `input_ids[resp_start + R - 2]` = response[R-2], label = response[R-1] = EOS ✅

**결론: off-by-one이 아님. 이 코드는 shift를 labels 측에서 수행하는 정상적인 패턴.**

#### 🟡 P1: Early stopping 미구현

현재 best checkpoint은 저장하지만 **학습을 중단하는 로직은 없음**. val_loss가 N번 연속 개선 안 되면 중단하는 patience 로직 추가 권장.

#### 🟡 P2: `_quality_filter`의 한국어 비율 30%는 너무 느슨

output의 한국어 문자 비율 30%는 영어/코드가 대부분인 샘플도 통과시킴. 한국어 SFT 목적에는 **50% 이상** 권장.

#### 🟢 P3: weight_decay=0.01이 launch_sft.sh에 명시 안 됨

`sft.py`의 default=0.01을 그대로 사용 중. 명시적으로 적는 게 좋음.

---

## 2. 데이터 파이프라인 설계

### 2.1 포맷 확인

현재 `sft_dataset.py`의 chat template:
```
<|user|>
{instruction}
<|assistant|>
{output}</s>
```

**→ 목표 포맷과 일치함.** 추가 수정 불필요.

### 2.2 샘플 품질 기준 최종안

| 기준 | 현재값 | 권장값 | 근거 |
|------|--------|--------|------|
| 최소 output 길이 | 50자 | **100자** | 50자는 1-2문장 수준으로 너무 짧음. IF-eval 같은 벤치마크는 구조화된 답변 필요 |
| 최대 output 길이 | 4000자 | **3000자** | 극단적으로 긴 응답은 반복 퇴화 리스크 ↑ |
| 한국어 비율 | 30% | **50%** | 한국어 LLM 목적에 30%는 너무 느슨 |
| 3-gram 반복 unique ratio | <0.5 제거 | **<0.6 제거** | 더 엄격한 반복 필터링으로 반복률 <5% 달성 |
| 최소 instruction 길이 | 10자 | **15자** | 극히 짧은 지시문은 모호한 학습 신호 |

### 2.3 데이터 믹싱 전략

현재 가중치 & 권장 조정:

| 소스 | 현재 가중치 | 권장 가중치 | 비고 |
|------|------------|------------|------|
| KOR-OpenOrca-Platypus-v3 | 2.0 | **1.5** | 여전히 과대표집 가능. 다양성 확보 위해 축소 |
| kullm-v2 | 1.0 | **1.0** | 유지 |
| ko-alpaca-12k | 2.0 | **1.5** | 소규모 고품질이지만 2배는 과도. 1.5배 적정 |
| korean_safe_conversation | 1.5 | **1.5** | 안전 정렬 중요, 유지 |
| evol-instruct-korean | 1.5 | **2.0** | 복잡한 추론/코드 능력은 IF-eval에 직결. 상향 |
| kovast | 0.8 | **0.5** | 멀티턴 첫 턴만 추출하므로 품질 불안정, 더 축소 |

**도메인 균형 목표 (대략적):**
- 일반 지식/QA: ~35%
- 안전/정렬: ~15%
- 추론/코드/수학: ~25% (IF-eval 핵심)
- 대화/창작: ~15%
- 기타: ~10%

---

## 3. 하이퍼파라미터 최종 확정

### 3.1 설정 배경

- 모델: 1B params
- 데이터: ~188K samples
- GPU: 8× B200 (FP8)
- Effective batch: 4 × 2 (grad_accum) × 8 (GPU) = **64 sequences/step**
- Steps per epoch: 188K / 64 ≈ **2,940 steps**
- 5000 steps ≈ **1.7 epochs**

### 3.2 최종 권장값

| 파라미터 | 현재값 | **최종 권장값** | 학술 근거 |
|----------|--------|----------------|-----------|
| **learning_rate** | 2e-5 | **2e-5** | Alpaca, Vicuna, LIMA 등 1-7B SFT 표준. 1e-5는 너무 보수적 (underfitting), 5e-6는 1B에 비현실적으로 낮음 |
| **epochs** | ~1.7 (5000 steps) | **3 epochs (~8,820 steps)** | 188K는 중규모. 3 epoch이면 충분한 학습 + overfitting 방지. Stanford Alpaca도 3 epoch 사용 |
| **max_steps** | 5000 | **9,000** | 3 epoch + 약간 여유 |
| **warmup_steps** | 150 | **300** | 9000 steps의 ~3%. Llama-2-chat 논문 기준 2-5% warmup 권장 |
| **weight_decay** | 0.01 | **0.01** | SFT 표준. pretrain(0.1)보다 낮아야 함. 적절 |
| **max_seq_len** | 4096 | **4096** | dynamic padding이 있으므로 4096 유지해도 짧은 샘플에서 낭비 없음. 긴 응답 학습 가능 |
| **max_grad_norm** | 1.0 | **1.0** | 표준값. 변경 불필요 |
| **batch_size (per GPU)** | 4 | **4** | B200 메모리 충분. 유지 |
| **grad_accum** | 2 | **2** | eff_batch=64 적절 |
| **NEFTune alpha** | 10.0 | **5.0** | 원 논문 default=5. 10은 aggressive. 반복률 개선에 도움되지만 과도하면 품질 하락 |
| **eval_interval** | 250 | **500** | 9000 steps에서 250은 너무 빈번. 500이면 ~18회 평가 |
| **save_interval** | 500 | **1000** | 디스크 절약 |

### 3.3 Launch script 수정사항

```bash
MAX_STEPS=9000
WARMUP_STEPS=300
# NEFTune alpha는 sft.py 코드에서 10→5로 변경 필요
```

---

## 4. 학습 모니터링 강화

### 4.1 현재 구현 상태

| 기능 | 상태 | 비고 |
|------|------|------|
| Val loss 주기적 기록 | ✅ eval_interval=250마다 | TensorBoard + stdout |
| Best checkpoint 저장 | ✅ val_loss 기준 | suffix="best" |
| TensorBoard 로깅 | ✅ loss, lr, grad_norm, tok/s | |
| Early stopping | ❌ 미구현 | **추가 필요** |
| 반복률 추적 | ❌ 미구현 | 학습 중 실시간 추적은 비현실적 |

### 4.2 Early Stopping 구현 방안

`trainer.py`에 추가할 로직:

```python
# __init__에 추가
self._patience = 5  # eval_interval * patience = 5 * 500 = 2500 steps 동안 개선 없으면 중단
self._no_improve_count = 0

# _run_validation 후 (train 메서드 내)
if val_loss < self._best_val_loss:
    self._best_val_loss = val_loss
    self._no_improve_count = 0
else:
    self._no_improve_count += 1
    if self._no_improve_count >= self._patience:
        self._log(f"Early stopping at step {step+1} (no improvement for {self._patience} evals)")
        break  # training loop 탈출
```

### 4.3 반복률 추적

학습 중 실시간 반복률 측정은 generation이 필요해 비용이 큼. 대안:
- **eval_interval마다 5개 샘플 생성** → 3-gram 반복 비율 계산 → TensorBoard 기록
- 이는 선택사항이며, 최종 평가 시 체크포인트별로 측정하는 것이 현실적

---

## 5. 재학습 전 체크리스트

### 데이터
- [ ] `data/sft/train.jsonl` 존재 확인 (188,234 lines) ✅
- [ ] `data/sft/val.jsonl` 존재 확인 (9,907 lines) ✅
- [ ] 데이터 품질 필터 재적용 여부 결정 (한국어 비율 50%, min_output 100자로 강화 시 재실행 필요)
- [ ] 재실행 시: `python data/prepare_sft_data.py --min_output_len 100`

### 코드 수정
- [ ] `prepare_sft_data.py`: 한국어 비율 0.3 → **0.5**로 변경
- [ ] `prepare_sft_data.py`: `_quality_filter`의 max output 4000 → **3000** 변경
- [ ] `prepare_sft_data.py`: DATASET_WEIGHTS 조정 (OpenOrca 2.0→1.5, evol-instruct 1.5→2.0, kovast 0.8→0.5)
- [ ] `sft.py`: NEFTune alpha 10.0 → **5.0** 변경
- [ ] `trainer.py`: Early stopping patience=5 로직 추가
- [ ] `launch_sft.sh`: MAX_STEPS=9000, WARMUP_STEPS=300 변경

### 인프라
- [ ] GPU 상태 확인: `nvidia-smi` (8× B200 가용)
- [ ] 디스크 공간: 체크포인트당 ~4GB × ~9개 = ~36GB 필요
- [ ] Base checkpoint 존재: `checkpoints/korean_1b_fp8_run1/checkpoint-0034000`
- [ ] Tokenizer 존재 확인

### 학습 실행
- [ ] TensorBoard 확인: `tensorboard --logdir checkpoints/korean_1b_sft/tensorboard`
- [ ] 처음 100 step에서 loss 급감 확인 (정상이면 2-3 → 1.5 이하)
- [ ] 500 step에서 val_loss 확인
- [ ] 반복 생성 테스트: 1000 step 체크포인트에서 간단한 프롬프트로 생성 테스트

---

## 6. 요약

### 현재 코드 상태: **양호 (minor 수정 6건)**

주요 버그는 이미 수정됨. labels off-by-one으로 보였던 것은 실제로는 정상적인 teacher-forcing shift 패턴. 추가 수정은 품질 향상을 위한 파라미터 튜닝 수준.

### 핵심 변경사항 우선순위

1. **MAX_STEPS 9000, WARMUP 300** (3 epoch 학습)
2. **데이터 품질 필터 강화** (한국어 50%, min output 100자) → `prepare_sft_data.py` 재실행
3. **Early stopping 구현**
4. **NEFTune alpha 5.0으로 하향**
5. **데이터 가중치 미세 조정**

### 예상 학습 시간

- 9000 steps × 8 GPU B200 FP8 ≈ **2-4시간** (seq_len 분포에 따라)
