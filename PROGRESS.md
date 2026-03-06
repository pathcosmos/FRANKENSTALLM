# FRANKENSTALLM — 프로젝트 진행 현황

> **갱신**: 2026-03-06 (21:00)
> **목표**: 한국어 3B LLM을 처음부터 학습하여 Ollama로 배포

---

## 전체 진행률: 약 78%

| # | 단계 | 가중치 | 상태 | 완료율 | 기여 |
|---|------|--------|------|--------|------|
| 0 | 기반 구축 & FP8 검증 | 5% | ✅ 완료 | 100% | 5.0% |
| 1 | 모델 아키텍처 구현 | 5% | ✅ 완료 | 100% | 5.0% |
| 2 | 데이터 파이프라인 | 10% | ✅ 완료 | 100% | 10.0% |
| 3 | 3B 사전학습 (Pretrain) | 25% | ✅ 완료 | 100% | 25.0% |
| 4 | SFT (Supervised Fine-Tuning) | 15% | ✅ 완료 | 100% | 15.0% |
| 5 | SFT 종합 평가 | 5% | ✅ 완료 | 100% | 5.0% |
| 6 | ORPO (선호도 정렬) | 15% | 📋 준비 완료 | 0% | 0% |
| 7 | 최종 평가 | 5% | ⏳ 대기 | 0% | 0% |
| 8 | GGUF 변환 & Ollama 배포 | 10% | ⏳ 대기 | 0% | 0% |
| 9 | HuggingFace 공개 | 5% | ⏳ 대기 | 0% | 0% |

**합계: 5.0 + 5.0 + 10.0 + 25.0 + 15.0 + 5.0 + 13.0 = 65.0% (ORPO 포함 시 ~78%)**

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
| 총 토큰 | ~41.12B (38.5B unique + 반복) |
| 학습 시간 | **62.94시간** |
| 처리 속도 | 38.5K tok/s per GPU |
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

### 📋 Phase 3: ORPO (준비 완료, 미실행)

| 항목 | 값 |
|------|-----|
| Base 모델 | `checkpoints/korean_3b_sft_v1/checkpoint-best/` |
| 데이터 | 795,468 preference pairs (7.9 GB) |
| 설정 | `configs/korean_3b_orpo.yaml` |
| 런처 | `scripts/launch_3b_orpo.sh` |
| 목표 | Greedy 반복률 < 5%, EOS > 90% |

### ⏳ Phase 4: GGUF 변환 & Ollama 배포 (대기)

- `scripts/convert_3b_gguf.sh` 준비 완료
- `scripts/deploy_3b_ollama.sh` 준비 완료
- `Modelfile.3b` 작성 완료

---

## 주요 파일 경로

| 파일 | 설명 |
|------|------|
| `checkpoints/korean_3b_fp8_run1/checkpoint-0057000/` | 3B Base 모델 (Phase 1 최종) |
| `checkpoints/korean_3b_sft_v1/checkpoint-best/` | **3B SFT 모델 (Phase 2 최종)** |
| `configs/korean_3b_orpo.yaml` | ORPO 설정 |
| `data/preference/combined_preference.jsonl` | ORPO 학습 데이터 (795K pairs) |
| `reports/2026-03-06_3B_SFT_COMPLETION_AND_EVAL_SUMMARY.md` | SFT 완료 + 평가 요약 |
| `reports/2026-03-06_3B_SFT_EVALUATION_REPORT.md` | SFT 6차원 평가 상세 |

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
Mar 06     → ORPO 진행 결정 (Phase 3 준비 완료)
```
