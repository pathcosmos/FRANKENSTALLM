# 어벤져스 팀 2번 — ORPO + 고품질 데이터로 1B 완성 전략

**작성일:** 2026-02-27  
**전략:** 현재 1B SFT v2 모델을 ORPO로 반복률 <5% 달성  
**현재 상태:** 반복률 18.0%, val_loss 2.2062

---

## 1. 반복률 18% → <5% 달성 로드맵

### Step A: 추론 파라미터 튜닝 (즉시, 0시간)

| 파라미터 | 현재 | 변경 | 
|----------|------|------|
| repetition_penalty | 1.1 | **1.2** |
| no_repeat_ngram_size | 3 | **4** |

**예상 반복률: 18% → 10~12%**

- 근거: 현재 eval에서 repetition_penalty=1.1로 측정. 1.2로 올리면 n-gram 반복이 직접 억제됨
- 한계: 생성 품질 저하 없이 가능한 범위. 1.3 이상은 문맥 coherence 손상
- **독립 효과:** 모델 가중치 변경 없이 즉시 적용. 다른 단계와 완전히 독립

### Step B: ORPO 학습 (핵심, 3~5시간)

**예상 반복률: 10~12% → 4~7%**

ORPO(Odds Ratio Preference Optimization)는 SFT + preference alignment를 단일 목적함수로 통합:
- SFT loss로 chosen 응답 학습
- Odds ratio로 chosen vs rejected 선호도 학습
- DPO 대비 reference model 불필요 → 메모리/시간 절약

**왜 ORPO가 반복 퇴화에 효과적인가:**
1. 반복 응답을 rejected로 명시적 학습 → 모델이 "반복하지 말라"를 직접 배움
2. SFT만으로는 "뭘 하면 안 되는지" 학습 불가 → preference learning이 유일한 해법
3. 1B 모델의 반복은 파라미터 부족이 아닌 **EOS 경계 학습 실패** + **반복 패턴 미벌칙** → ORPO로 직접 교정 가능

**필요 데이터:** 500~2000 preference 쌍 (아래 섹션 2 참조)

### Step C: 데이터 정제 + 추가 SFT (선택적, 2~4시간)

**예상 반복률: 4~7% → 3~5%**

- data_quality_audit에서 발견된 문제 수정:
  - `</s>` 오염 113건 제거
  - 짧은 output(<80자) 16,519건 제거
  - Q/A 마커 ~550건 제거
  - OpenOrca 가중치 5.0→2.0
- 정제된 ~120K 데이터로 추가 SFT 2-3 epochs

**독립 효과:** 데이터 품질 개선은 ORPO와 무관하게 기저 모델 개선. 하지만 ORPO 없이 이것만으로는 반복률 <5% 불가능 (SFT v1→v2에서 이미 데이터 정제했으나 17.7%→18%로 정체)

### 종합 예상

| 단계 | 반복률 | 소요시간 | 누적시간 |
|------|--------|----------|----------|
| 현재 | 18.0% | - | - |
| Step A (추론 파라미터) | 10~12% | 0h | 0h |
| Step B (ORPO) | 4~7% | 3~5h | 3~5h |
| Step C (데이터 정제 SFT) | 3~5% | 2~4h | 5~9h |
| **최종** | **3~5%** | | **5~9h** |

---

## 2. 자체 Preference 데이터 생성 전략

### 방법: Self-Play Rejection Sampling

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("checkpoints/korean_1b_sft/checkpoint-best")
tokenizer = AutoTokenizer.from_pretrained(...)

def generate_preference_pair(prompt, n_samples=8, temp=0.9):
    """프롬프트 당 n_samples개 생성 → chosen/rejected 분류"""
    responses = []
    for _ in range(n_samples):
        output = model.generate(
            tokenizer.encode(f"<|user|>\n{prompt}\n<|assistant|>\n", return_tensors="pt"),
            max_new_tokens=256, temperature=temp, top_p=0.95,
            do_sample=True, repetition_penalty=1.0  # 의도적으로 penalty 없이
        )
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        rep_rate = calc_repetition_rate(text)  # 10-gram 기준
        responses.append((text, rep_rate))
    
    # 분류
    chosen = [r for r in responses if r[1] < 0.05]   # 반복률 5% 미만 → chosen
    rejected = [r for r in responses if r[1] > 0.15]  # 반복률 15% 이상 → rejected
    
    if chosen and rejected:
        return {"prompt": prompt, "chosen": chosen[0][0], "rejected": rejected[0][0]}
    return None
```

### 규모 계산

| 항목 | 값 |
|------|-----|
| 필요 preference 쌍 | 500~1000 (최소 500) |
| 프롬프트 당 샘플 수 | 8 |
| 유효 쌍 생성률 | ~40% (반복률 18%이므로 chosen/rejected 분리 가능) |
| 필요 프롬프트 수 | 500 / 0.4 = **~1,250개** |
| 프롬프트 당 생성 시간 | 8 × 256 tokens × ~0.02s/token ≈ 40s |
| **총 생성 시간** | 1,250 × 40s ≈ **14시간** (GPU 1개) |

⚠️ **자체 생성은 느림.** 대안: 기존 HF preference 데이터 활용 (섹션 3)

### 자동 품질 판단 기준

- **chosen 임계값:** 10-gram 반복률 < 5%, 길이 > 50 tokens, EOS 정상 생성
- **rejected 임계값:** 10-gram 반복률 > 15% OR 동일 문장 2회 이상 반복
- 중간 영역(5~15%)은 버림 → contrastive signal 극대화

### 빠른 대안: 하이브리드 전략 (추천)

1. HF에서 500~1000쌍 다운로드 (즉시)
2. 자체 모델로 200~300쌍 추가 생성 (반복 특화, 3~4시간)
3. 총 700~1300쌍으로 ORPO 학습

---

## 3. HuggingFace 즉시 사용 가능 한국어 Preference 데이터

### 확인된 데이터셋

| 데이터셋 | 크기 | 포맷 | 적합성 |
|----------|------|------|--------|
| `maywell/ko_Ultrafeedback_binarized` | **61,966쌍** | prompt/chosen/rejected | ⭐⭐⭐ 최적 — 바로 ORPO에 사용 가능 |
| `kuotient/orca-math-korean-dpo-pairs` | **192,848쌍** | question/chosen/rejected | ⭐⭐ 수학 특화지만 양 풍부 |
| `nayohan/preference-collection-ko-full` | **199,760쌍** | 복잡 포맷 (score_A/B) | ⭐⭐ 전처리 필요 |
| `jojo0217/korean_rlhf_dataset` | 미확인 | 미확인 | ⭐ 확인 필요 |
| `heegyu/PKU-SafeRLHF-ko` | 미확인 | 미확인 | ⭐ 안전성 특화 |

### 추천 조합

```python
# 1순위: ko_Ultrafeedback_binarized에서 2000쌍 샘플링
from datasets import load_dataset
ds = load_dataset("maywell/ko_Ultrafeedback_binarized", split="train")
# 이미 prompt/chosen/rejected 포맷 → 바로 사용

# 2순위: orca-math에서 500쌍 추가 (다양성)
ds2 = load_dataset("kuotient/orca-math-korean-dpo-pairs", split="train")
```

**준비 시간: 30분 미만** (다운로드 + 포맷 변환)

---

## 4. 1B 모델의 한계와 ORPO 극복 범위

### 반복 퇴화의 근본 원인: 파라미터 수 vs 학습 방법

**파라미터 수가 주 원인이 아닌 근거:**
1. Pretrain 단계에서 반복률 69% → SFT로 18%까지 낮춤. 같은 1B 파라미터로 51%p 개선
2. 반복 패턴은 특정 프롬프트에서만 발생 (짧은 사실 질문은 0%, 긴 설명 질문에서 20~33%)
3. data_quality_audit에서 EOS 학습 실패가 핵심 원인으로 지목됨 → 학습 데이터/방법 문제

**1B에서 반복률 <5% 현실성:**
- Qwen2.5-0.5B, SmolLM-1.7B 등 유사 규모 모델이 RLHF/DPO 후 반복률 <5% 달성 사례 다수
- ORPO 원논문(Hong et al., 2024)에서 Phi-2(2.7B)와 Llama-2-7B 실험 → 소규모 모델에서도 일관된 개선
- 1B급 직접 실험은 드물지만, **반복 퇴화는 alignment 문제이지 capacity 문제가 아님**

**ORPO 특유의 장점 (1B에 유리):**
- Reference model 불필요 → GPU 메모리 절약 (DPO는 2배 메모리)
- 1B 모델을 단일 GPU에서 full fine-tuning 가능
- SFT + preference를 동시에 학습 → 적은 데이터로 효율적

### 현실적 기대치

| 목표 | 달성 가능성 | 조건 |
|------|------------|------|
| 반복률 <10% | **95%** | ORPO 500쌍 + rep_penalty=1.2 |
| 반복률 <5% | **70%** | ORPO 1000쌍 + 데이터 정제 SFT |
| 반복률 <3% | **40%** | ORPO 2000쌍 + 데이터 정제 + 파라미터 튜닝 |

---

## 5. 총 비용 계산

### 1B ORPO 경로 (이 전략)

| 단계 | 작업 | 시간 |
|------|------|------|
| 1 | HF preference 데이터 다운로드 + 전처리 | 0.5h |
| 2 | 자체 preference 생성 (200~300쌍, 선택적) | 3~4h |
| 3 | ORPO 학습 (1000쌍, 1~2 epochs) | 1~2h |
| 4 | 평가 + 반복 | 0.5h |
| 5 | (선택) 데이터 정제 재SFT | 2~4h |
| **총합 (필수만)** | | **2~3h** |
| **총합 (전체)** | | **7~11h** |

### 3B 처음부터 경로 (대안)

| 단계 | 시간 |
|------|------|
| 3B pretrain | 26h |
| SFT | 1~2h |
| 평가 | 1h |
| **총합** | **28~29h** |

### 비교

| 항목 | 1B ORPO | 3B 처음부터 |
|------|---------|------------|
| 소요 시간 | 2~11h | 28~29h |
| 성공 확률 (<5%) | 70% | 80~90% |
| 실패 시 비용 | 3~11h 낭비 | 29h 낭비 |
| 기대값 (시간×확률) | 3~11h / 0.7 = **4~16h** | 29h / 0.85 = **34h** |
| 병렬 가능 | ✅ 3B와 동시 진행 가능 | GPU 점유 |

---

## 6. 최종 권고: 왜 지금 당장 ORPO여야 하는가

### 핵심 논거

1. **시간 효율:** 필수 단계만 2~3시간. 3B의 1/10 시간
2. **리스크 최소:** 실패해도 3시간 손실. 3B는 29시간 손실
3. **이미 데이터 있음:** `maywell/ko_Ultrafeedback_binarized` 61K쌍이 HF에 준비됨. 다운로드만 하면 됨
4. **정확한 문제 해결:** 반복 퇴화의 원인은 "뭘 하면 안 되는지 모름" → preference learning이 정확한 해법
5. **병렬 전략 가능:** ORPO는 2~3시간이므로, 3B 학습과 동시에 시작 가능. 먼저 끝나는 쪽 채택

### 즉시 실행 계획

```bash
# Step 1: preference 데이터 준비 (30분)
python3 scripts/prepare_orpo_data.py \
  --hf_dataset maywell/ko_Ultrafeedback_binarized \
  --sample_size 2000 \
  --output data/orpo/train.jsonl

# Step 2: ORPO 학습 (1~2시간)
python3 scripts/train_orpo.py \
  --model checkpoints/korean_1b_sft/checkpoint-best \
  --data data/orpo/train.jsonl \
  --lr 5e-6 --epochs 2 --batch_size 4 --beta 0.1 \
  --output checkpoints/korean_1b_orpo

# Step 3: 평가 (30분)
python3 eval/comprehensive_eval.py \
  --model checkpoints/korean_1b_orpo \
  --repetition_penalty 1.2 --no_repeat_ngram_size 4
```

### 성공 판정 기준

| 지표 | 목표 | 현재 |
|------|------|------|
| 반복률 | <5% | 18% |
| 자연 종료율 | >80% | 60% |
| 응답 품질 | 유지 또는 개선 | baseline |

---

## 요약

| 항목 | 값 |
|------|-----|
| **전략** | ORPO + 추론 파라미터 튜닝 |
| **예상 반복률** | 3~7% (목표 <5% 달성 확률 70%) |
| **총 소요시간** | 2~3h (필수) / 7~11h (전체) |
| **vs 3B** | 10~15배 빠름, 기대값 기준 2~3배 효율적 |
| **필요 데이터** | HF에서 즉시 사용 가능 (0원, 30분) |
| **핵심 메시지** | SFT만으로는 "하지 말아야 할 것"을 가르칠 수 없다. ORPO가 정확한 해법이다. |
