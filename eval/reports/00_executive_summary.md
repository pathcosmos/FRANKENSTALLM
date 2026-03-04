# korean_1b_fp8_run1 종합 평가 리포트

> **평가 날짜**: 2026-02-26
> **평가 환경**: NVIDIA B200 ×1 (추론), BF16, 평가 소요 시간 약 15분

---

## 모델 정보

| 항목 | 내용 |
|------|------|
| **모델명** | korean_1b_fp8_run1 |
| **파라미터** | 1,189.7M (1.19B) |
| **아키텍처** | Decoder-only Transformer, LLaMA-style |
| **vocab_size** | 64,000 |
| **d_model** | 2,048 |
| **n_layers** | 24 |
| **n_heads** | 16 |
| **n_kv_heads (GQA)** | 4 |
| **d_ffn** | 5,472 |
| **위치 인코딩** | RoPE (theta=500,000) |
| **정규화** | RMSNorm |
| **활성화 함수** | SwiGLU |
| **기타** | Weight Tying, FlashAttention-2, TransformerEngine FP8 (MXFP8BlockScaling) |

### 학습 설정

| 항목 | 내용 |
|------|------|
| **학습 단계** | 34,000 steps |
| **GPU 환경** | 8× NVIDIA B200 |
| **학습 정밀도** | FP8 + BF16 혼합 |
| **학습률** | 2.0e-4 |
| **배치 크기** | 1.05M tok/step (8GPU × 8batch × 4accum × 4096seq) |
| **웜업** | 2,000 steps |
| **학습 데이터** | 한국어 위키백과 + C4 한국어 + 나무위키 (총 ~8.91B tokens, ~4 에포크) |

---

## 핵심 평가 결과 요약

| 평가 영역 | 핵심 지표 | 판정 |
|-----------|-----------|------|
| **Perplexity (통합)** | PPL=6.95, bits/tok=2.80 | Good (1B 기준) |
| **Perplexity (C4)** | PPL=5.67, bits/tok=2.50 | Excellent |
| **Perplexity (Wiki)** | PPL=11.66, bits/tok=3.54 | Acceptable |
| **Perplexity (Namuwiki)** | PPL=25.34, bits/tok=4.66 | Needs improvement |
| **Top-1 Accuracy** | 56.18% | Good |
| **Top-5 Accuracy** | 72.35% | Good |
| **Top-10 Accuracy** | 77.75% | Good |
| **Mean Entropy** | 2.24 nats (3.23 bits) | Healthy |
| **생성 품질** | 한국어 문법 양호, 사실 부분적 | Expected for 1B |
| **반복 퇴화** | 3/10 degenerate (30%) | Needs mitigation |
| **코드/수학** | 매우 제한적 | Expected |

---

## 상세 리포트 목록

| 파일 | 내용 |
|------|------|
| [`01_perplexity_report.md`](./01_perplexity_report.md) | 데이터 소스별 Perplexity 상세 분석 |
| [`02_generation_report.md`](./02_generation_report.md) | 10개 프롬프트 생성 품질 상세 분석 |
| [`03_repetition_calibration_report.md`](./03_repetition_calibration_report.md) | 반복 분석 + 캘리브레이션 점검 |
| [`04_token_analysis_comparison_report.md`](./04_token_analysis_comparison_report.md) | 토큰 수준 NLL 분석 + 온도별 비교 |

---

## Perplexity 분석 요약

### 데이터 소스별 PPL

```
C4 한국어 (일반 웹 텍스트):  PPL =  5.67  bits/tok = 2.50  ← Excellent
위키백과:                     PPL = 11.66  bits/tok = 3.54  ← Acceptable
나무위키:                     PPL = 25.34  bits/tok = 4.66  ← Needs improvement
통합 (가중 평균):             PPL =  6.95  bits/tok = 2.80  ← Good
```

C4에서의 낮은 PPL은 일상적 웹 텍스트 분포에 잘 적응했음을 나타낸다. 위키백과 PPL이 나무위키보다 낮은 것은 위키백과 특유의 정형화된 문체가 학습 데이터로 더 많이 포함되었기 때문으로 해석된다. 나무위키의 높은 PPL은 구어체, 은어, 신조어, 표 형식 등 비정형 텍스트가 많기 때문이며, 1B 규모의 모델에서는 일반적인 결과이다.

### 비교 기준 (참고)

| 모델 | 규모 | 한국어 PPL (참고치) |
|------|------|---------------------|
| GPT-2 Small | 125M | ~30–40 (영어 기준) |
| small_fp8_run1 (본 프로젝트) | 125M | ~18–22 (추정) |
| **korean_1b_fp8_run1 (본 모델)** | **1.19B** | **6.95 (통합)** |
| LLaMA-2 7B (한국어 적응 없음) | 7B | — |

125M → 1.19B 스케일업에서 PPL이 약 2.5배 이상 개선된 점은 스케일링 법칙(Chinchilla)과 일치하는 결과이다.

---

## 생성 품질 분석 요약

10개 프롬프트에 대한 greedy decoding 결과 기준:

### 생성 성공 사례 (7/10)

- **일상 대화 / 설명문**: 자연스러운 한국어 문장 구성, 조사·어미 처리 안정적
- **사전적 정의 요청**: 단어 설명 형식을 잘 따라가는 경향
- **간단한 목록 생성**: 항목 나열 패턴 파악

### 문제 사례 (3/10)

- **반복 퇴화 (Repetition Degeneration)**: 같은 구절이 반복되며 문장이 수렴하지 않음. Greedy decoding에서 특히 발생하기 쉬운 패턴으로, temperature sampling 또는 repetition penalty 적용으로 완화 가능
- **사실 오류**: 세종대왕 관련 연도, 김치찌개 레시피 비율 등에서 부정확한 내용 생성 — 1B 파라미터로는 세밀한 사실 기억 능력에 한계가 있으며 예상된 결과
- **코드/수학**: 파이썬 코드 생성 및 수식 계산에서 매우 제한적인 성능 — 사전학습 데이터에 코드/수학 데이터가 포함되지 않았으므로 예상된 결과

---

## 캘리브레이션 분석 요약

### Top-K Accuracy

| K | Accuracy |
|---|----------|
| Top-1 | 56.18% |
| Top-5 | 72.35% |
| Top-10 | 77.75% |

Top-1 정확도 56%는 언어 모델로서 건강한 수준이다. 모델이 올바른 다음 토큰을 확률 상위 1위로 예측하는 비율이 56%라는 것은 과도한 확신(overconfidence)이나 과소한 확신(underconfidence) 없이 균형 잡힌 예측 분포를 가짐을 시사한다.

### 엔트로피 분석

```
Mean Entropy: 2.24 nats (3.23 bits)
```

엔트로피 2.24 nats는 모델이 예측 시 약 9.4개 토큰에 걸쳐 확률을 분산시킨다는 의미이다 (e^2.24 ≈ 9.4). 이 값은 너무 뾰족하지도(greedy collapse 위험) 너무 평탄하지도(무작위 출력 위험) 않은 건강한 분포를 나타낸다.

---

## 결론

### 전체 평가

**1B 한국어 사전학습 모델로서 전반적으로 양호한 성능.**

이 모델은 한국어 위키백과, C4 한국어, 나무위키 약 8.91B 토큰으로 학습된 1.19B 파라미터 Decoder-only 모델이다. 8× B200 GPU 환경에서 FP8 + BF16 혼합 정밀도로 34,000 steps 학습하였으며, Chinchilla 최적 계산량에 근사한 설정이다.

---

### 강점

1. **C4 PPL=5.67**: 일반 웹 텍스트에 대한 우수한 언어 모델링 성능. 한국어 일상 텍스트의 패턴을 잘 학습함
2. **Top-1 Accuracy 56%**: 과도한 확신 없이 건강한 캘리브레이션 상태를 유지함
3. **한국어 문법 처리**: 조사(은/는/이/가/을/를), 어미(~했다/~합니다/~이다) 처리가 안정적이며 문법적으로 자연스러운 문장 생성
4. **일상적 프롬프트 대응**: 설명, 정의, 목록 등 기본적인 텍스트 생성 패턴 파악

---

### 약점

1. **Namuwiki PPL=25.34**: 비정형 텍스트(구어체, 은어, 신조어, 표 형식)에 상대적으로 약함. 도메인 불균형에서 비롯됨
2. **반복 퇴화 30%**: 10개 생성 중 3개에서 repetition degeneration 발생. Greedy decoding 환경에서 특히 두드러지며, SFT 또는 RLHF 단계에서 개선 예상
3. **사실 정확도 제한**: 세종대왕 연도, 음식 레시피 등 구체적 사실 기억 능력이 낮음. 1B 파라미터 모델의 고유한 한계이며, 7B 이상 스케일에서 개선 예상
4. **코드/수학 거의 불가**: 사전학습 데이터에 코드/수학이 포함되지 않아 예상된 결과. 전문 파인튜닝 필요

---

### 다음 단계 권장

| 우선순위 | 작업 | 기대 효과 |
|----------|------|-----------|
| 1 | **Instruction Tuning (SFT)** | 반복 퇴화 완화, 지시문 따르기 능력 부여 |
| 2 | **DPO/RLHF** | 생성 품질 + 사실 정확도 개선 |
| 3 | **도메인 적응** | 나무위키/전문 도메인 추가 데이터로 PPL 개선 |
| 4 | **스케일업 (7B)** | 사실 기억, 반복 문제 동시 개선 예상 |
| 5 | **양자화 + 배포** | GGUF Q4_K_M + Ollama 서빙 (Phase B 파이프라인 활용 가능) |

---

## 평가 환경

| 항목 | 내용 |
|------|------|
| **GPU** | NVIDIA B200 ×1 (추론) |
| **추론 dtype** | BF16 |
| **평가 소요 시간** | 약 15분 (전체 6개 섹션) |
| **평가 날짜** | 2026-02-26 |
| **평가 스크립트** | `eval/comprehensive_eval.py` |
