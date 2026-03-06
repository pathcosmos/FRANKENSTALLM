# FRANKENSTALLM 3B — SFT 완료, 종합 평가, 코드 개선 및 ORPO 준비 보고서

> **작성일**: 2026-03-06
> **Phase**: Phase 2 (SFT) **완료** → Phase 3 (ORPO) 준비 완료
> **이 보고서의 범위**: SFT v1 완료, 6차원 평가, SFT v2 설계, 코드 개선, ORPO 준비

---

## 1. SFT v1 학습 완료

### 1.1 최종 결과

| 항목 | 값 |
|------|-----|
| **최종 Step** | 25,500 / 33,000 (77.3%) |
| **종료 사유** | Early stopping (patience 5 exhausted) |
| **Best val_loss** | **1.8851** (step 23,000) |
| **최종 train_loss** | ~1.80 |
| **학습 시작** | 2026-03-05 22:15 |
| **학습 종료** | 2026-03-06 13:56 |
| **총 학습 시간** | ~15시간 41분 |
| **VRAM** | 24.2 GB / 183 GB (13.2%) — 전 구간 일정 |
| **사고** | 0건 |

### 1.2 Val Loss 전체 추이

```
Step     500: 2.0732
Step   1,000: 2.0035  (-0.070)
Step   2,000: 1.9558  (-0.048)
Step   3,000: 1.9329  (-0.023)
Step   5,000: 1.9107  (-0.022)
Step  10,000: 1.8917  (-0.019)
Step  15,000: 1.8864  (-0.005)
Step  20,000: 1.8853  (-0.001)
Step  23,000: 1.8851  ← BEST
Step  23,500~25,500: 1.8851 (변동 없음, patience 1~5/5)
Step  25,500: Early Stop
```

**수렴 분석**: step 15K 이후 val_loss 변동 < 0.001. Cosine decay LR이 step 23K에서 2.18e-06까지 감소하여 실질적 학습 종료. Early stopping이 정확히 작동.

### 1.3 학습 설정 (v1)

| 항목 | 값 | 근거 |
|------|-----|------|
| LR | 1e-5 | Pretrain LR(1.5e-4)의 1/15 — forgetting 방지 |
| Effective batch | 64 (2 × 8GPU × 4 accum) | |
| NEFTune alpha | 5.0 | 임베딩 노이즈로 생성 다양성 향상 |
| Warmup | 500 steps | |
| Weight decay | 0.01 | |
| Max steps | 33,000 | ~3.3 epochs |

---

## 2. SFT 종합 평가 결과 (6차원)

평가 일시: 2026-03-06 16:27, SFT 체크포인트: checkpoint-best (step 23000), 소요 49분 27초

### 2.1 차원별 판정

| # | 평가 차원 | 결과 | 핵심 수치 |
|---|----------|------|-----------|
| 1 | Perplexity (지식 보존) | **PASS** | 최대 forgetting 0.9% (임계값 15%) |
| 2 | 생성 품질 (반복률/EOS) | **FAIL** | Greedy 반복률 72.97% (목표 <5%) |
| 3 | 한국어 벤치마크 | **FAIL** | KoBEST 평균 43.26% (목표 >55%) |
| 4 | 영어 벤치마크 (유지) | **PASS** | 모든 태스크 하한 초과 |
| 5 | Calibration | **PASS** | Top-1 68.59% (목표 >=65%) |
| 6 | SFT Chat 능력 | **PASS** | EOS 종료율 60% (Base 0% → 60%) |

**종합: 4/6 차원 통과**

### 2.2 Base vs SFT 핵심 비교

| 지표 | Base | SFT | 변화 | 판정 |
|------|------|-----|------|------|
| Val PPL (통합) | 5.2263 | 5.2529 | +0.5% | PASS |
| Greedy 3-gram 반복률 | 60.99% | 72.97% | +12pp | FAIL (악화) |
| EOS 종료율 | 0% | 60% | +60pp | 개선 (목표 미달) |
| KoBEST 평균 | 43.69% | 43.26% | -0.4pp | FAIL |
| MMLU-KO | 22.75% | 26.00% | +3.2pp | 부분 개선 |

### 2.3 Perplexity 상세 (19개 데이터셋)

| 데이터셋 | Base PPL | SFT PPL | Forgetting % |
|---------|---------|---------|-------------|
| 3b | 5.2263 | 5.2529 | +0.5% |
| cc100_ko | 21.7820 | 21.8072 | +0.1% |
| hplt_ko | 2.4028 | 2.4121 | +0.4% |
| korean_c4 | 5.7173 | 5.7617 | +0.8% |
| korean_namuwiki | 25.8814 | 26.1185 | +0.9% |

**평균 Forgetting: +0.4%** — 19개 전체 PASS. 지식 보존 우수.

### 2.4 생성 샘플

**Greedy (반복 문제)**:
- "대한민국의 수도는" → "서울특별시입니다. 대한민국의 수도는 서울이며..." (EOS=True, rep=0%)
- "인공지능이란" → 동일 문장 5회 반복 (EOS=True, rep=82.5%)
- "한국의 전통 음식 중에서" → 김치 설명 5회 반복 (EOS=False, rep=83.6%)

**Sampled (t=0.7, rep_penalty=1.2) → 반복률 0%**:
- 파라미터 검색에서 rep_penalty 1.1~1.3 적용 시 반복률 0% 달성 확인
- ORPO가 이 행동을 내재화할 수 있다는 근거

### 2.5 한국어 벤치마크 (0-shot)

| 태스크 | Base | SFT | 변화 |
|--------|------|-----|------|
| kobest_boolq | 50.28% | 50.14% | -0.1pp |
| kobest_copa | 49.30% | 48.60% | -0.7pp |
| kobest_hellaswag | 21.60% | 19.80% | -1.8pp |
| kobest_sentineg | 48.61% | 49.12% | +0.5pp |
| kobest_wic | 48.65% | 48.65% | +0.0pp |
| **평균** | **43.69%** | **43.26%** | **-0.4pp** |
| haerae | 19.71% | 19.89% | +0.2pp |
| MMLU-KO | 22.75% | 26.00% | **+3.2pp** |

> KoBEST는 거의 변동 없음 (SFT가 0-shot 분류 능력을 크게 바꾸지 않음). MMLU-KO의 +3.2pp 개선은 instruction-following이 다소 반영된 결과.

---

## 3. SFT v1의 한계와 v2 설계

### 3.1 v1 한계 분석

SFT v1은 지식 보존(forgetting 0.9%)에서 우수했으나, 핵심 목표인 반복률 해소에 실패했다.

| 문제 | 원인 분석 |
|------|----------|
| Greedy 반복률 72.97% (base 60.99%보다 악화) | LR 1e-5가 너무 보수적 → SFT 데이터의 패턴을 충분히 학습하지 못함 |
| KoBEST 변동 없음 | 동일 원인: 낮은 LR로 instruction-following이 약하게 학습됨 |
| Val_loss 1.8851 plateau | Cosine decay가 step 20K에서 사실상 0 → 추가 학습 여지 소진 |

### 3.2 SFT v2 설계 (`configs/korean_3b_sft_v2.yaml`)

v1 실패를 바탕으로 SFT v2를 설계하고 설정 및 코드를 준비했다.

| 항목 | SFT v1 | SFT v2 | 변경 근거 |
|------|--------|--------|----------|
| **LR** | 1e-5 | **5e-5** | 5배 상향, 3B SFT 표준 범위 (Llama-3 SFT 참고) |
| **Effective batch** | 64 | **256** | 4배 확대 (bs=4, accum=8) |
| **Warmup** | 500 | **2,000** | 높은 LR 안정화 |
| **Max steps** | 33,000 | **15,000** | 높은 LR+큰 배치 → 빠른 수렴 |
| **Weight decay** | 0.01 | **0.05** | forgetting 억제 강화 |
| **Data mixing** | 없음 | **SFT 70% + Pretrain 30%** | catastrophic forgetting 방지 |

### 3.3 ORPO 경로 선택 이유

SFT v2를 실행하기 전에, 먼저 ORPO로 반복 문제를 해결하는 것이 더 효율적이라고 판단:

1. **SFT v1의 지식 보존이 우수** (forgetting 0.9%) → base가 건강함
2. **반복 문제는 선호도 정렬의 영역** — 반복을 "나쁜 응답"으로 학습시키는 것이 더 직접적
3. **파라미터 검색에서 rep_penalty로 0% 달성** → 모델이 반복하지 않는 능력 자체는 있음
4. **SFT v2 후에도 ORPO가 필요할 가능성 높음** → 단계 절약

---

## 4. 이번 세션의 코드 변경 사항

### 4.1 `train/sft.py` — MixingDataLoader + DDP 최적화 (+238줄)

**MixingDataLoader 클래스**: SFT 데이터와 Pretrain 데이터를 확률적으로 인터리빙하는 DataLoader 래퍼.
- `pretrain_ratio=0.3`이면 30% 배치가 pretrain에서, 70%가 SFT에서 옴
- 양쪽 DataLoader 무한 사이클 (epoch 단위 자동 재시작)
- 빈 DataLoader 방어 (RuntimeError with 상세 메시지)

**DDP Rank 0 전용 토크나이징**:
- 이전: 8개 rank가 각각 독립 토크나이징 → 8배 중복 작업 + 8배 메모리
- 개선: Rank 0만 64-worker 병렬 토크나이즈 + 디스크 캐시 → DDP barrier → 나머지 rank는 캐시 로드
- 효과: 메모리 8배 절감, 재실행 시 ~2분으로 단축 (21GB `.sft_cache.pt`)

**새 CLI 인자**: `--pretrain_data`, `--pretrain_mix_ratio`, `--max_grad_norm`

### 4.2 `train/trainer.py` — Early Stopping DDP 수정 (+17줄)

**문제**: 기존 코드에서 rank 0만 early stopping을 판단하고 `return` → 나머지 rank는 무한 대기 (DDP hang)

**해결**: `torch.distributed.broadcast`로 early stopping 결정을 전 rank에 동기화.
```python
stop_tensor = torch.tensor([1 if should_stop else 0], device=self.device)
torch.distributed.broadcast(stop_tensor, src=0)
```

**추가 변경**: patience 5 → 10 (v2에서 warmup 후 충분한 학습 보장)

### 4.3 `train/orpo.py` — YAML 설정 지원 + 3B 기본값 (+30줄)

- YAML config 파일 로드 기능 추가 (`--config` 인자)
- 3B 최적화 기본값: batch=2, accum=8, max_length=2048, max_prompt_length=1024
- output_dir 기본값: `checkpoints/korean_3b_orpo`

### 4.4 `eval/report_generator.py` — SFT 비교 보고서 생성기 (+831줄)

Base vs SFT 비교 보고서를 자동 생성하는 대규모 확장:
- Base 모델 참조값 내장 (PPL 19개, 벤치마크 전체)
- Forgetting 계산 (PPL 변화율)
- 생성 품질 비교 (반복률, EOS, chat template)
- 벤치마크 비교 (한국어 7개 + 영어 6개, 0-shot + 5-shot)
- Calibration 비교
- ORPO 진행 판정 자동 로직
- Repetition 파라미터 검색 결과 통합

### 4.5 `eval/tasks/generation_task.py` — Chat Template + 다양성 메트릭 (+75줄)

- 환경변수 기반 체크포인트 경로 (`EVAL_CHECKPOINT`, `EVAL_TOKENIZER`)
- Chat template 지원 (`USE_CHAT_TEMPLATE=1` → `<|user|>...<|assistant|>` 포맷)
- `compute_diversity_metrics()`: Distinct-n, Type-Token Ratio 추가

### 4.6 `eval/tasks/{calibration,ppl,token_nll}_task.py` — 로깅 개선 (+35줄)

- `logging` 모듈 도입 (기존 `print` → 구조화된 로깅)

### 4.7 `eval/sft_eval_pipeline.py` — 새 파일 (SFT 6차원 평가 파이프라인)

8-GPU 병렬 SFT 평가를 위한 통합 파이프라인:
- Phase 1: PPL (19개 데이터셋, GPU 0-4 분배)
- Phase 2: 생성 품질 + 파라미터 검색 (GPU 6-7)
- Phase 3: 벤치마크 (KoBEST, HAE-RAE, MMLU-KO, MMLU-EN, 영어 5대)
- Phase 4: 자동 보고서 생성 + ORPO 판정

### 4.8 설정 파일

| 파일 | 설명 |
|------|------|
| `configs/korean_3b_sft_v2.yaml` | SFT v2 설정 (lr=5e-5, data mixing 70/30, 15K steps) |
| `configs/korean_3b_orpo.yaml` | ORPO 설정 (lr=5e-6, beta=0.1, 795K pairs) |
| `scripts/launch_3b_sft_v2.sh` | SFT v2 런처 (NCCL 최적화, pre-flight checks) |
| `scripts/launch_3b_orpo.sh` | ORPO 런처 업데이트 |

---

## 5. Phase 게이트 판정 및 다음 단계

### 5.1 결정: Phase 3 ORPO 진행

| 근거 | 상세 |
|------|------|
| 지식 보존 양호 | forgetting 0.9% — base 지식 파괴 없음 |
| 반복 미해결 | greedy 72.97% — 선호도 정렬이 직접적 해결 경로 |
| 파라미터 검색 희망적 | rep_penalty 1.2 적용 시 0% → ORPO가 내재화 가능 |
| 데이터 준비 완료 | 795,468 preference pairs (7.9GB) |
| 코드/설정 완비 | `train/orpo.py`, `configs/korean_3b_orpo.yaml` |

### 5.2 ORPO 실행 계획

```
1. HF 변환: checkpoint-best → safetensors
2. ORPO 학습: scripts/launch_3b_orpo.sh
3. 평가: eval/sft_eval_pipeline.py
4. 판정: 반복률 < 5% → GGUF + Ollama 배포
```

### 5.3 SFT v2 백업 경로

ORPO가 충분하지 않을 경우:
```
SFT v2 (lr=5e-5, data mixing) → ORPO → 재평가
```

---

## 6. 전체 프로젝트 타임라인

```
Feb 25     125M FP8 검증, 인프라 세팅
Feb 25-26  1B Pretrain (34K steps, loss 1.904)
Feb 26     1B SFT v1 실패 (label off-by-one → loss=0)
Feb 27     5-에이전트 루트 코즈 분석 (5가지 버그 발견)
Feb 27     1B SFT v2 성공 (val_loss 2.206, 반복률 18%)
Feb 27     저스티스리그 토론 → 3B 전환 결정
Feb 27     640GB+ 데이터 조립
Mar 02     Phase 0 완료 (GQA FA, VRAM -20%, SIGHUP 3중 방어)
Mar 02     Phase 1 시작 (3B Pretrain)
Mar 05     Phase 1 완료 (57K steps, loss 1.466, 63시간)
Mar 05     SFT 데이터 준비 (24소스 → 2.44M samples, 5단계 필터)
Mar 05     Phase 2 시작 (3B SFT v1)
Mar 06     Phase 2 완료 (25.5K steps, val_loss 1.8851, early stopping)
Mar 06     SFT 6차원 평가 (4/6 PASS, 반복률 FAIL)
Mar 06     코드 개선 (MixingDataLoader, DDP early stop fix, eval pipeline)
Mar 06     SFT v2 설계 + ORPO 설정 준비
Mar 06     → ORPO 진행 결정
```

---

## 7. 수정 파일 요약

| 파일 | 변경 | 줄 수 |
|------|------|-------|
| `train/sft.py` | MixingDataLoader, DDP 토크나이징, CLI 인자 | +238 |
| `eval/report_generator.py` | SFT 비교 보고서 생성기 | +831 |
| `eval/tasks/generation_task.py` | Chat template, 다양성 메트릭 | +75 |
| `eval/tasks/calibration_task.py` | 로깅 개선 | +13 |
| `eval/tasks/ppl_task.py` | 로깅 개선 | +11 |
| `eval/tasks/token_nll_task.py` | 로깅 개선 | +11 |
| `train/orpo.py` | YAML config, 3B 기본값 | +30 |
| `train/trainer.py` | DDP early stop broadcast, patience 10 | +17 |
| `scripts/launch_3b_orpo.sh` | 3B ORPO 런처 업데이트 | +10 |
| **신규: `eval/sft_eval_pipeline.py`** | SFT 6차원 평가 파이프라인 | 신규 |
| **신규: `configs/korean_3b_sft_v2.yaml`** | SFT v2 설정 | 신규 |
| **신규: `configs/korean_3b_orpo.yaml`** | ORPO 설정 | 신규 |
| **신규: `scripts/launch_3b_sft_v2.sh`** | SFT v2 런처 | 신규 |

**총 변경: +1,312줄 / -132줄**

---

*이 보고서는 SFT v1 완료(early stopping at step 25,500), 6차원 종합 평가, SFT v2 설계, 코드 개선, ORPO 준비를 포괄합니다.*
*상세 평가: `reports/2026-03-06_3B_SFT_EVALUATION_REPORT.md`*
*상세 평가 계획: `reports/2026-03-06_3B_SFT_EVAL_PLAN.md`*
