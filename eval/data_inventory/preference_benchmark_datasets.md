# Preference/RLHF + Benchmark 데이터 전수 조사

> 조사일: 2026-02-27

---

## Part 1: 한국어 Preference/DPO 데이터

| 데이터셋 | 규모 | 다운로드 | 비고 |
|----------|------|----------|------|
| `kuotient/orca-math-korean-dpo-pairs` | 100K~1M | 111 | 한국어 수학 DPO. 대규모 |
| `nayohan/preference-collection-ko-full` | 100K~1M | 30 | 한국어 종합 preference |
| `jojo0217/korean_rlhf_dataset` | 100K~1M | 54 | 한국어 RLHF |
| `maywell/ko_Ultrafeedback_binarized` | 10K~100K | 108 | UltraFeedback 한국어 번역 |
| `ChuGyouk/argilla-distilabel-math-preference-dpo-korean` | 1K~10K | 10 | 수학 DPO 한국어 |
| `ohsuz/dpo-v1010-korean` | 10K~100K | 3 | 한국어 DPO |
| `ohsuz/dpo-v1010-korean-without-finance` | 10K~100K | 3 | 금융 제외 버전 |
| `tellang/yeji-preference-ko-v1` | 10K~100K | 13 | 한국어 preference |
| `AnonymousLLMer/Safety_preference-ko-cleaned` | 1K~10K | 4 | 안전성 preference |
| `mncai/distilabel-math-preference-dpo-ko` | 1K~10K | 4 | 수학 DPO 한국어 |
| `vaiv/ko-rag-preference` | <1K | 2 | RAG preference (소규모) |

### ❌ 접근 불가 (404)
- `Bongseok/ko-DPO-v0.1` — 삭제됨
- `HAERAE-HUB/KoRA` — 삭제됨
- `maywell/ko_Ultrafeedback` — 삭제됨 (binarized 버전만 존재)

---

## Part 2: 영어 Preference 데이터 (번역 가치 순위)

| 데이터셋 | 규모 | 다운로드 | 번역 가치 |
|----------|------|----------|-----------|
| `HuggingFaceH4/ultrafeedback_binarized` | 100K~1M (~62K쌍) | 5,158 | ⭐⭐⭐ 최고. 이미 ko 번역판 존재(maywell) |
| `Anthropic/hh-rlhf` | 100K~1M | 17,609 | ⭐⭐⭐ 인간 선호도. 대화형 |
| `nvidia/HelpSteer2` | 10K~100K | 15,448 | ⭐⭐⭐ 고품질 세밀 점수 |
| `openbmb/UltraFeedback` | 10K~100K | 2,317 | ⭐⭐ 원본 (binarized 버전 더 유용) |
| `argilla/distilabel-math-preference-dpo` | 1K~10K | 328 | ⭐⭐ 수학 특화 (이미 ko 번역판 존재) |
| `snorkelai/Snorkel-Mistral-PairRM-DPO-Dataset` | 10K~100K | 71 | ⭐ 자동 생성 |
| `HuggingFaceH4/stack-exchange-preferences` | 10M~100M | 3,873 | ⭐ 너무 대규모, 코드 편향 |
| `allenai/preference-test-sets` | 10K~100K | 2,777 | 평가용 (학습 부적합) |

---

## Part 3: 벤치마크/평가 데이터

| 데이터셋 | 규모 | 다운로드 | 용도 |
|----------|------|----------|------|
| **`HAERAE-HUB/KMMLU`** | 100K~1M | 10,537 | 한국어 MMLU. 핵심 벤치마크 |
| `skt/kobest_v1` | 10K~100K | 3,194 | KoBEST 5개 태스크 (BoolQ, COPA, WiC, HellaSwag, SentiNeg) |
| `HAERAE-HUB/HAE_RAE_BENCH_1.0` | 1K~10K | 457 | 해래 벤치 |
| `HAERAE-HUB/K2-Eval` | <1K | 76 | K2 평가 |
| `openai/gsm8k` | 10K~100K | 465,032 | 수학 추론 (영어) |
| `HuggingFaceH4/MATH-500` | <1K | 94,894 | 수학 벤치마크 (영어) |
| `Rowan/hellaswag` | 10K~100K | 213,419 | 상식추론 (영어) |
| `google/IFEval` | <1K | 60,319 | 지시 따르기 평가 (영어) |

### ❌ 접근 불가 (404)
- `coastalcph/mimir`, `kuotient/korean-gsm8k`, `HAERAE-HUB/KorNAT-CV`, `HAERAE-HUB/KorNAT-NL2SQL`, `snunlp/korean-hate-speech`

---

## Part 4: 자체 Preference 데이터 생성 가능성

**SFT v2 모델 (반복률 18%) 기반 Self-Play 방식:**

### 방법
1. SFT 데이터의 프롬프트 풀에서 각 프롬프트당 N=4~8회 샘플링 (temperature 0.7~1.0)
2. 자동 품질 판단으로 chosen/rejected 선별

### 자동 품질 판단 기준
- **반복 탐지**: n-gram 반복률 > 20% → rejected
- **길이 필터**: 너무 짧거나(<50자) 너무 긴(>2000자) → rejected
- **Perplexity 기반**: 외부 judge 모델 (GPT-4 또는 더 큰 모델)로 점수 부여
- **Self-consistency**: 동일 프롬프트 응답 간 reward model 점수 비교

### 예상 생성량
- SFT 프롬프트 10K개 × 4회 샘플링 = 40K 응답
- chosen/rejected 쌍: ~10K~20K쌍 (상위 25% vs 하위 25%)
- **주의**: 반복률 18%인 모델로 생성 시 rejected 품질이 너무 낮을 수 있음 → 유의미한 학습 신호 약화 가능

### 권장
- 자체 생성보다 **기존 한국어 데이터 활용 우선** (아래 추천 참조)
- 자체 생성은 ORPO 1차 학습 후, 개선된 모델로 2차 Self-Play 시 더 효과적

---

## 🎯 ORPO 즉시 시작 가능한 데이터 조합 추천

### Tier 1: 즉시 사용 (한국어, 변환 최소)
| 데이터 | 예상 쌍수 | 우선순위 |
|--------|-----------|----------|
| `jojo0217/korean_rlhf_dataset` | ~100K+ | 🥇 가장 범용적 |
| `maywell/ko_Ultrafeedback_binarized` | ~60K | 🥇 UltraFeedback 한국어, 고품질 |
| `nayohan/preference-collection-ko-full` | ~100K+ | 🥇 종합 preference |
| `kuotient/orca-math-korean-dpo-pairs` | ~100K+ | 🥈 수학 특화 |

### Tier 2: 보충용
| 데이터 | 예상 쌍수 | 용도 |
|--------|-----------|------|
| `ohsuz/dpo-v1010-korean` | ~10K+ | 추가 다양성 |
| `tellang/yeji-preference-ko-v1` | ~10K+ | 추가 다양성 |
| `ChuGyouk/argilla-distilabel-math-preference-dpo-korean` | ~5K | 수학 보충 |

### 추천 조합
```
총 ~200K~300K쌍 확보 가능
1차: jojo0217 + maywell + nayohan 합산 → ~260K쌍 (예상)
2차: kuotient 수학 추가 → 수학 능력 강화
```

### 벤치마크 평가 파이프라인
- **KMMLU** (한국어 지식) + **KoBEST** (한국어 NLU) 필수
- **GSM8K** (수학) + **IFEval** (지시 따르기) 보조
- **HAE_RAE_BENCH** 한국어 종합 평가
