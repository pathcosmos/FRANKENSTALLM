# FRANKENSTALLM — 프로젝트 진행 현황

> **갱신**: 2026-03-26 (모델 교체 완료 반영)
> **목표**: 한국어 3B LLM을 처음부터 학습하여 Ollama로 배포

---

## 전체 진행률: 100% ✅ 완료

| # | 단계 | 가중치 | 상태 | 완료율 | 기여 |
|---|------|--------|------|--------|------|
| 0 | 기반 구축 & FP8 검증 | 5% | ✅ 완료 | 100% | 5.0% |
| 1 | 모델 아키텍처 구현 | 5% | ✅ 완료 | 100% | 5.0% |
| 2 | 데이터 파이프라인 | 10% | ✅ 완료 | 100% | 10.0% |
| 3 | 3B 사전학습 (Pretrain) | 25% | ✅ 완료 | 100% | 25.0% |
| 4 | SFT (Supervised Fine-Tuning) | 15% | ✅ 완료 | 100% | 15.0% |
| 5 | SFT 종합 평가 | 5% | ✅ 완료 | 100% | 5.0% |
| 6 | ORPO (선호도 정렬) | 15% | ✅ 완료 | 100% | 15.0% |
| 7 | 최종 평가 | 5% | ✅ 완료 | 100% | 5.0% |
| 8 | GGUF 변환 & Ollama 배포 | 10% | ✅ 완료 | 100% | 10.0% |
| 9 | HuggingFace 공개 | 5% | ✅ 완료 | 100% | 5.0% |

**합계: 5.0 + 5.0 + 10.0 + 25.0 + 15.0 + 5.0 + 15.0 + 5.0 + 10.0 + 5.0 = 100%**

---

## Phase별 상세 현황

### ✅ Phase 0: 기반 구축 & FP8 검증 (완료, Feb 25 ~ Mar 2)

- 8x B200 환경 검증, 125M FP8 파이프라인 성공
- GQA FlashAttention native → VRAM 60.4 → 48.3 GB (-20%)
- DDP gradient_as_bucket_view, NCCL NVLS, SIGHUP 3중 방어
- torch.compile 테스트 → 효과 없음 (TE opaque kernel)

### ✅ Phase 1: 3B Pretrain (완료, Mar 2~5)

| 항목 | 값 |
|------|-----|
| 학습 스텝 | 57,000 (100%) |
| 최종 Loss | **1.466** |
| 총 토큰 | ~38.5B tokens |
| 학습 시간 | **62.94시간** |
| 처리 속도 | 292K tok/s (8 GPU 합산) / 36.5K tok/s per GPU |
| MFU | 33.5% |
| VRAM | 48.3 GB (26.4%) |
| 사고 | 0건 |

### ✅ Phase 2: SFT (완료, Mar 5~6)

| 항목 | 값 |
|------|-----|
| 최종 스텝 | **25,500 / 33,000** (77.3%, early stopping) |
| Best val_loss | **1.8851** (step 23,000) |
| 학습 시간 | **~15시간 41분** |
| 데이터 | 24개 소스 → **2,439,397 samples** (7.48 GB) |
| VRAM | 24.2 GB (13.2%) |
| 사고 | 0건 |

**Val Loss 추이**:
```
Step     500: 2.0732
Step   2,000: 1.9558
Step   5,000: 1.9107
Step  10,000: 1.8917
Step  15,000: 1.8864
Step  20,000: 1.8853
Step  23,000: 1.8851 ← BEST
Step  25,500: 1.8851 → Early Stop (patience 5/5)
```

### ✅ Phase 2.5: SFT 종합 평가 (완료, Mar 6)

**6차원 평가 결과**: 4/6 PASS

| 차원 | 결과 | 핵심 수치 |
|------|------|-----------|
| Perplexity (지식 보존) | **PASS** | forgetting 0.9% |
| 생성 품질 | **FAIL** | Greedy 반복률 72.97% |
| 한국어 벤치마크 | **FAIL** | KoBEST 평균 43.26% |
| 영어 벤치마크 | **PASS** | 전 태스크 하한 초과 |
| Calibration | **PASS** | Top-1 68.59% |
| SFT Chat 능력 | **PASS** | EOS 종료율 60% (Base 0%) |

**판정**: ORPO 진행 (지식 보존 양호, 반복률 해결 필요)

### ✅ Phase 3: ORPO (완료, Mar 8~9)

| 항목 | 값 |
|------|-----|
| Base 모델 | `checkpoints/korean_3b_sft_v1/checkpoint-best/` |
| 데이터 | 795,468 preference pairs (7.9 GB) |
| **최종 스텝** | **9,997** (조기 수렴) |
| **eval_loss** | 1.7910 → **1.6250** |
| **Preference Accuracy** | 67.8% → **76.02%** |
| **Reward Margin** | 0.107 → **0.6100** |
| **greedy 반복률** | 72.97% → **30.89%** (↓42pp) |
| **EOS 종료율** | 60% → 67% |
| **KoBEST 0-shot** | **52.75%** |
| **PPL forgetting** | 최대 4.1% (임계값 15% 이하 ✅) |
| HP Sweep | 6개 설정 탐색 후 최적 (beta=0.25, lr=1.2e-5) 선정 |
| 소요 시간 | ~2시간 |
| 사고 | 0건 (TRL ORPO NaN 버그 3중 패치 해결) |

**10차원 종합 평가**: 7/10 PASS, 정량 스코어 **63.7/100**

| # | 차원 | 결과 |
|---|------|------|
| 1 | Perplexity (지식 보존) | ✅ PASS (forgetting 최대 4.1%) |
| 2 | 생성 품질 (greedy) | ❌ FAIL (30.89%, 목표 <5%) |
| 3 | 한국어 벤치마크 KoBEST | ❌ FAIL (52.75%, 목표 >55%) |
| 4 | 영어 벤치마크 | ❌ FAIL |
| 5 | Calibration | ✅ PASS (67.99%) |
| 6 | SFT Chat 능력 | ✅ PASS |
| 7 | Preference Accuracy | ✅ PASS (76.02%) |
| 8 | Reward Margins | ✅ PASS (0.6100) |
| 9 | Parameter Sensitivity | ✅ PASS |
| 10 | SFT→ORPO 개선 | ✅ PASS |

### ✅ Phase 4: GGUF 변환 & Ollama 배포 (완료, Mar 9~10)

**byte-fallback 수정 (v2)**: SentencePiece 토크나이저에 256개 byte-fallback 토큰 추가, 임베딩 64000→64256 리사이즈 → `\n` 포함 프롬프트 Ollama 크래시 해결

**생성된 GGUF 6종**:

| 파일 | 크기 | 설명 |
|------|------|------|
| `frankenstallm-3b-f16.gguf` | 6.0G | v1 f16 (ORPO 원본) |
| `frankenstallm-3b-Q8_0.gguf` | 3.2G | v1 Q8_0 |
| `frankenstallm-3b-Q4_K_M.gguf` | 1.9G | v1 Q4_K_M |
| `frankenstallm-3b-v2-f16.gguf` | 2.3G | **v2 f16 (byte-fallback 수정)** |
| `frankenstallm-3b-v2-Q8_0.gguf` | 1.2G | **v2 Q8_0** |
| `frankenstallm-3b-v2-Q4_K_M.gguf` | 757M | **v2 Q4_K_M ← 권장** |

**Ollama 벤치마크** (frankenstallm-3b-v2:Q4_K_M, 35 테스트):

| 항목 | 값 |
|------|-----|
| 자동 채점 평균 | **46.7** |
| 평균 TPS | **142.5 tok/s** |
| 평균 TTFT | **16.7 ms** |
| korean_nlu | 100.0 |
| reasoning | 50.0 |
| knowledge | 75.0 |
| instruction_following | 66.7 |

**최적 샘플링 파라미터** (`t0.7_rep1.2`):

| 파라미터 | 값 |
|---------|-----|
| temperature | **0.7** |
| repeat_penalty | **1.2** |
| top_p | **0.9** |
| top_k | **50** |
| num_predict | **512** |
| num_ctx | **4096** |

### ✅ Phase 5: HuggingFace 배포 (완료, Mar 10 / 모델 교체 Mar 25~26)

**URL**: https://huggingface.co/pathcosmos/frankenstallm

**업로드 완료 파일**:

| 경로 | 내용 |
|------|------|
| `model.safetensors` | 4.76GB — v2 ORPO 베스트 체크포인트 |
| `config.json`, `tokenizer.json` 등 | HF 모델 설정 파일 |
| `sampling_config.json` | 검증된 샘플링 파라미터 |
| `gguf/frankenstallm-3b-v2-Q4_K_M.gguf` | 757M ← 권장 |
| `gguf/frankenstallm-3b-v2-Q8_0.gguf` | 1.2G |
| `gguf/frankenstallm-3b-v2-f16.gguf` | 2.3G |
| `gguf/frankenstallm-3b-Q4_K_M.gguf` | 1.9G |
| `gguf/frankenstallm-3b-Q8_0.gguf` | 3.2G |
| `gguf/frankenstallm-3b-f16.gguf` | 6.0G |
| `gguf/Modelfile.3b-v2-Q4_K_M` 등 | Ollama Modelfile 6종 |

**모델 교체 완료 (Mar 25~26)**: 초기 HF 배포 시 1.2B 모델이 잘못 업로드된 문제를 확인, 3B 정식 체크포인트로 교체 완료.

---

## 주요 파일 경로

| 파일 | 설명 |
|------|------|
| `checkpoints/korean_3b_fp8_run1/checkpoint-0057000/` | 3B Base 모델 (Phase 1 최종) |
| `checkpoints/korean_3b_sft_v1/checkpoint-best/` | 3B SFT 모델 (Phase 2 최종) |
| `outputs/hf_checkpoint-best/` | ORPO 최종 체크포인트 (HF 형식) |
| `outputs/hf_checkpoint-best-fixed/` | byte-fallback 수정본 (v2) |
| `outputs/gguf/` | GGUF 변환 파일 6종 |
| `configs/korean_3b_orpo.yaml` | ORPO 설정 |
| `data/preference/combined_preference.jsonl` | ORPO 학습 데이터 (795K pairs) |
| `reports/2026-03-10_PROJECT_COMPLETION_REPORT.md` | 프로젝트 완료 종합 보고서 |
| `reports/2026-03-09_ORPO_EVALUATION_REPORT.md` | ORPO 10차원 평가 상세 |
| `reports/2026-03-09_GGUF_DEPLOYMENT_AND_EVAL_REPORT.md` | GGUF 변환·Ollama 배포 보고서 |
| `reports/2026-03-16_GPU_UTILIZATION_REPORT.md` | GPU 활용 현황 보고서 |

---

## 타임라인

```
Feb 25     Phase 0 시작 (기반 구축, 125M FP8 검증)
Feb 25-26  1B Pretrain (34K steps, loss 1.904)
Feb 26     1B SFT v1 실패 (label off-by-one)
Feb 27     1B SFT v2 성공 (val_loss 2.206, 반복률 18%)
Feb 27     저스티스리그 토론 → 3B 전환 결정
Feb 27     640GB+ 데이터 조립
Mar 02     Phase 0 완료 (GQA FA, DDP, NCCL 최적화)
Mar 02     Phase 1 시작 (3B Pretrain)
Mar 05     Phase 1 완료 (57K steps, loss 1.466, 63시간)
Mar 05     Phase 2 시작 (SFT, 2.44M samples)
Mar 06     Phase 2 완료 (25.5K steps, val_loss 1.8851, early stopping)
Mar 06     SFT 6차원 평가 완료 (4/6 PASS)
Mar 06     → ORPO 진행 결정 (Phase 3)
Mar 08     Phase 3 시작 (ORPO HP Sweep 6 configs)
Mar 09     Phase 3 완료 (9,997 steps, eval_loss 1.625, pref_acc 76.02%)
Mar 09     ORPO 10차원 평가 (7/10 PASS, 63.7/100)
Mar 09     Phase 4 시작 (byte-fallback 수정, GGUF 변환)
Mar 10     Phase 4 완료 (GGUF 6종 생성, Ollama 배포, 벤치마크 완료)
Mar 10     Phase 5 완료 (HuggingFace 배포)
Mar 16     GPU 활용 현황 보고서 작성 (~843 GPU-hours, MFU 33.5%)
Mar 25-26  HF 모델 교체 완료 (1.2B → 3B 정식 체크포인트)
```

---

## GPU 활용 요약 (전체 프로젝트)

| 단계 | 소요 시간 | GPU-hours |
|------|----------|-----------|
| Phase 0: 환경 구축 & FP8 검증 | ~24시간 | 192 |
| Phase 1: 3B 사전학습 | 62.94시간 | 503.5 |
| Phase 2: SFT 지시학습 | 15.7시간 | 125.6 |
| Phase 3: ORPO 선호도 정렬 | ~2시간 | 16 |
| Phase 4: GGUF 변환 & 배포 | ~4시간 | 6 |
| **합계** | **~109시간** | **~843 GPU-hours** |

---

## 다음 단계 (Future Work)

프로젝트 완료 이후 수립된 개선 계획 및 후속 연구:

| 문서 | 내용 |
|------|------|
| `reports/2026-03-13_PERFORMANCE_IMPROVEMENT_STRATEGY_AND_ALTERNATIVES.md` | 성능 개선 전략 (데이터 품질, RLHF, DPO 비교 등) |
| `reports/2026-03-13_KOREAN_CODING_LLM_DESIGN_DRAFT.md` | 한국어 코딩 LLM 설계 초안 (데이터·앙상블 중심) |
| `docs/NEXT_OPTIMIZATION_PLAN.md` | MFU 최적화 계획 (33.5% → 47% 목표, FSDP + 커널 퓨전) |

**주요 후속 과제**:
- MFU 33.5% → 47% 개선 (FSDP + kernel fusion 적용)
- 13B 규모 모델 학습 실험 (FSDP 전략, VRAM 70%+ 활용)
- 한국어 코딩 특화 LLM 개발 (예상 8 GPU × 200+ 시간)
- greedy 반복률 추가 개선 (현재 30.89%, 목표 <5%)
- KoBEST 벤치마크 55% 이상 달성 (현재 52.75%)
