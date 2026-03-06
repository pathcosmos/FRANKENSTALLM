# FRANKENSTALLM 3B SFT 모델 다면적 종합 평가 계획서

> 작성일: 2026-03-06
> 대상 모델: FRANKENSTALLM 3B SFT v1
> 작성 목적: SFT 학습 완료 후 6개 차원 종합 평가 계획 수립

---

## 1. 개요

### 1.1 학습 완료 상태

| 항목 | 값 |
|------|-----|
| Phase | Phase 2 — SFT (Supervised Fine-Tuning) |
| 최종 Step | 25,500 (early stopping) |
| 최종 val_loss | 1.8851 |
| 체크포인트 | `checkpoints/korean_3b_sft_v1/checkpoint-best/` |

### 1.2 모델 구성

| 파라미터 | 값 |
|----------|-----|
| `use_hybrid` | `false` |
| `use_fp8` | `true` |
| `d_model` | 3072 |
| `n_layers` | 28 |
| 총 파라미터 | ~3B |

### 1.3 평가 목표

1. **6개 평가 차원**으로 SFT 모델을 종합적으로 평가
2. **Base 모델 대비 향상 폭** 정량 측정
3. 결과에 따라 **ORPO 진행 여부** 결정 (Phase 3 게이트)

---

## 2. 평가 차원 (6개)

### 차원 1: Perplexity (지식 보존 — Catastrophic Forgetting 검증)

SFT 과정에서 사전학습 지식이 손실되지 않았는지 검증한다.

**방법**: 19개 validation 데이터셋 × sliding window (`seq_len=2048`, `stride=512`)

#### 주요 데이터셋 및 목표

| 데이터셋 | Base PPL | SFT 목표 PPL | 허용 상한 (forgetting < 15%) |
|----------|----------|-------------|---------------------------|
| `3b_val` | 5.2263 | < 6.0 | 6.01 |
| `korean_c4_val` | 5.7173 | < 6.6 | 6.57 |
| `hplt_ko_val` | 2.4028 | < 2.8 | 2.76 |
| `cc100_ko_val` | 21.782 | < 25.0 | 25.05 |

**판정 기준**: 전체 19개 데이터셋 평균 forgetting ratio < 15%

---

### 차원 2: 생성 품질 (반복률 + EOS + 텍스트 자연스러움)

SFT의 핵심 성과인 **반복 생성 해소**와 **EOS 종료 능력**을 검증한다.

#### 실험 설계

| 항목 | 값 |
|------|-----|
| 프롬프트 수 | 15 한국어 프롬프트 |
| 온도 | 4단계 (0.0 / 0.5 / 0.8 / 1.0) |
| 총 생성 수 | 60 (15 × 4) |
| 파라미터 조합 grid search | 12개 조합 |

#### Chat Template 적용

```
<|user|>
{prompt}
<|assistant|>
```

- **Chat template ON**: SFT 모델 주 평가
- **Raw prompt** (template 없음): Base 모델과의 직접 비교용

#### 핵심 지표 및 목표

| 지표 | Base 값 | SFT 목표 | 비고 |
|------|---------|---------|------|
| Greedy 3-gram 반복률 | 60.99% | < 5% | 가장 중요한 SFT 성과 지표 |
| EOS 종료율 | 0% | > 90% | 생성 종료 능력 |

---

### 차원 3: 벤치마크 (한국어 이해력)

SFT를 통한 한국어 이해 능력 향상을 정량 측정한다.

#### KoBEST (5개 태스크)

| 태스크 | Base (%) | SFT 목표 (%) | 평가 방식 |
|--------|----------|-------------|----------|
| `kobest_copa` | 49.30 | > 65 | 0-shot + 5-shot |
| `kobest_boolq` | 50.28 | > 60 | 0-shot + 5-shot |
| `kobest_hellaswag` | 21.60 | > 30 | 0-shot + 5-shot |
| `kobest_sentineg` | 48.61 | > 60 | 0-shot + 5-shot |
| `kobest_wic` | 48.65 | > 55 | 0-shot + 5-shot |
| **KoBEST 평균** | **43.69** | **> 55** | |

#### HAE-RAE

| 지표 | Base (%) | SFT 목표 (%) |
|------|----------|-------------|
| HAE-RAE 평균 | 19.71 | > 25 |

#### MMLU-KO

| 지표 | Base (%) | SFT 목표 (%) |
|------|----------|-------------|
| MMLU-KO 평균 (57개 과목) | 22.75 | > 30 |

---

### 차원 4: 벤치마크 (영어 유지)

SFT 과정에서 영어 능력이 하락하지 않았는지 검증한다. **하락 금지** 원칙 적용.

| 태스크 | Base (%) | SFT 최소 유지 (%) |
|--------|----------|------------------|
| `hellaswag` | 26.00 | >= 25 |
| `arc_easy` | 25.63 | >= 25 |
| `arc_challenge` | 21.67 | >= 21 |
| `winogrande` | 50.59 | >= 49 |
| `piqa` | 52.50 | >= 51 |
| **MMLU-EN 평균** | **25.81** | **>= 25** |

---

### 차원 5: Calibration (확률 분포 품질)

모델의 예측 확률 분포가 적절히 보정되어 있는지 검증한다.

| 지표 | Base 값 | SFT 기준 |
|------|---------|---------|
| Top-1 Accuracy | 68.75% | >= 65% |
| Top-5 Accuracy | 81.64% | >= 78% |
| Top-10 Accuracy | 85.93% | >= 82% |
| Mean Entropy | 1.5682 | < 2.0 |
| Token NLL mean | 1.5561 | < 2.0 |

---

### 차원 6: SFT 고유 평가 (Chat 능력)

SFT 학습의 본질적 목적인 대화 능력을 정성+정량 평가한다.

#### 평가 항목

| 항목 | 설명 |
|------|------|
| Chat template 응답 품질 | `<\|user\|>` / `<\|assistant\|>` 포맷 준수 여부, 응답 완결성 |
| 지시 따르기 (Instruction Following) | 명시적 지시사항 이행률 (형식, 길이, 언어 등) |
| 다국어 전환 | 한국어 질문→한국어 답변, 영어 질문→영어 답변 전환 능력 |
| 코드 생성 | 간단한 Python/SQL 코드 생성 정확도 |

---

## 3. GPU 분배 계획

### Phase 1: 내부 평가 (차원 1, 2, 5)

8개 GPU를 병렬로 활용하여 내부 평가를 동시 실행한다.

| GPU | 태스크 ID | 내용 | 예상 시간 |
|-----|-----------|------|----------|
| GPU 0 | `ppl_single` | PPL — `3b_val.bin` | ~30분 |
| GPU 1 | `ppl_multi` | PPL — `korean_c4_val.bin`, `korean_val.bin` | ~40분 |
| GPU 2 | `ppl_multi` | PPL — `hplt_ko_val.bin`, `cc100_ko_val.bin` | ~40분 |
| GPU 3 | `ppl_multi` | PPL — cosmo 계열 7개 | ~50분 |
| GPU 4 | `ppl_multi` | PPL — wiki/math 계열 7개 | ~50분 |
| GPU 5 | `calib_nll` | Calibration + Token NLL | ~20분 |
| GPU 6 | `generation` | 15 프롬프트 × 4 온도 (chat template ON) | ~30분 |
| GPU 7 | `repetition_grid` | 12 파라미터 조합 × 5 프롬프트 | ~40분 |

### Phase 2: 벤치마크 (차원 3, 4)

Phase 1 완료 후 GPU를 재배정하여 벤치마크를 실행한다.

| GPU | 벤치마크 | 비고 |
|-----|---------|------|
| GPU 0 | `kobest_boolq`, `kobest_copa`, `kobest_wic` | 한국어 (차원 3) |
| GPU 1 | `kobest_hellaswag`, `kobest_sentineg` | 한국어 (차원 3) |
| GPU 2 | `haerae` | 한국어 (차원 3) |
| GPU 3 | `global_mmlu_ko` | 한국어 (차원 3) |
| GPU 4 | `hellaswag`, `arc_easy`, `arc_challenge` | 영어 (차원 4) |
| GPU 5 | `winogrande`, `piqa` | 영어 (차원 4) |
| GPU 6 | `mmlu_humanities`, `mmlu_social_sciences` | 영어 MMLU (차원 4) |
| GPU 7 | `mmlu_stem`, `mmlu_other` | 영어 MMLU (차원 4) |

---

## 4. 판정 기준 (Phase 게이트)

평가 결과에 따라 다음 단계를 결정한다.

| 조건 | 판정 | 다음 단계 |
|------|------|----------|
| 반복률 < 5% **AND** KoBEST > 55% **AND** forgetting < 15% | **PASS** | Phase 4: GGUF 변환 + Ollama 배포 |
| 반복률 5~15% **OR** 벤치마크 부분 달성 | **CONDITIONAL** | Phase 3: ORPO 강화학습 |
| 반복률 > 15% **OR** 벤치마크 하락 **OR** forgetting > 20% | **FAIL** | SFT 재시도 (하이퍼파라미터/데이터 조정) |

---

## 5. 산출물

평가 완료 시 아래 파일들이 생성된다.

```
eval/outputs/3b_sft_eval_YYYYMMDD_HHMM/
├── phase1_results.json          # 내부 평가 결과 (PPL, 생성, Calibration)
└── phase2_results.json          # 벤치마크 결과 (KoBEST, HAE-RAE, MMLU 등)

reports/
└── 2026-03-06_3B_SFT_EVALUATION_REPORT.md   # 종합 평가 보고서
```

---

## 6. 실행 순서 요약

```
1. Phase 1 내부 평가 (8 GPU 병렬)
   ├── GPU 0-4: Perplexity (19개 val 데이터셋)
   ├── GPU 5:   Calibration + Token NLL
   ├── GPU 6:   생성 품질 (chat template)
   └── GPU 7:   반복률 grid search
   → phase1_results.json 저장

2. Phase 1 결과 확인
   ├── 반복률 > 15%  → FAIL → SFT 재시도 (Phase 2 생략)
   └── 반복률 <= 15% → Phase 2 진행

3. Phase 2 벤치마크 (8 GPU 병렬)
   ├── GPU 0-3: 한국어 벤치마크 (KoBEST, HAE-RAE, MMLU-KO)
   └── GPU 4-7: 영어 벤치마크 (HellaSwag, ARC, PIQA, MMLU-EN)
   → phase2_results.json 저장

4. 종합 판정 → 보고서 작성
   → reports/2026-03-06_3B_SFT_EVALUATION_REPORT.md
```
