# FRANKENSTALLM 3B — SFT 진행 보고서

> **작성일**: 2026-03-05
> **프로젝트**: FRANKENSTALLM 3B
> **현재 Phase**: Phase 2 (SFT) — 진행 중 (6%)
> **작성 목적**: Phase 0~2 전체 여정 기록 및 팀 내 공유

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [Phase 0: 최적화 (2026-03-02)](#2-phase-0-최적화-2026-03-02)
3. [Phase 1: 사전학습 (57K steps, ~63시간)](#3-phase-1-사전학습-57k-steps-63시간)
4. [사전학습 평가 결과](#4-사전학습-평가-결과)
5. [SFT 데이터 준비](#5-sft-데이터-준비)
6. [Phase 2: SFT 학습 (현재 진행 중)](#6-phase-2-sft-학습-현재-진행-중)
7. [발견된 이슈 및 해결](#7-발견된-이슈-및-해결)
8. [현재 상태 요약 및 전망](#8-현재-상태-요약-및-전망)

---

## 1. 프로젝트 개요

### 목표

소규모 LLM을 처음부터(from scratch) 사전학습하고, SFT를 통해 한국어 지시 수행 능력을 부여하는 실험 프로젝트.
상용 모델에 의존하지 않고, 데이터 수집 → 학습 → 평가 → 배포까지 전 과정을 직접 수행한다.

### 모델 아키텍처

| 항목 | 값 |
|------|-----|
| **모델명** | FRANKENSTALLM 3B |
| **파라미터** | ~3,015M (3B) |
| **아키텍처** | Transformer (Decoder-only, Pre-norm) |
| **d_model** | 3,072 |
| **n_layers** | 28 |
| **n_heads / n_kv_heads** | 24 / 8 (GQA 3:1) |
| **d_ffn** | 8,192 |
| **max_seq_len** | 4,096 |
| **vocab_size** | 64,000 |
| **RoPE theta** | 500,000 |
| **Attention** | FlashAttention-2 (GQA native) |
| **FFN** | SwiGLU (te.LayerNormMLP) |
| **정밀도** | MXFP8 (B200 block scaling) |

### 하드웨어

| 항목 | 사양 |
|------|------|
| **GPU** | 8x NVIDIA B200 (183 GB HBM3e each, 1.47 TB total) |
| **CPU** | 2x AMD EPYC 9365 (72 cores, Zen 5) |
| **RAM** | 2.2 TB DDR5 |
| **CUDA** | 13.1 |
| **PyTorch** | 2.10.0a0+nv25.12 (NVIDIA custom) |
| **Interconnect** | NVLink 5.0 (NV18, 900 GB/s bidirectional) + NVSwitch |

---

## 2. Phase 0: 최적화 (2026-03-02)

### 2.1 OOM 해결

초기 설정(batch_size=8)에서 172+ GB VRAM 사용으로 OOM 발생.
아래 최적화를 통해 **48.3 GB/GPU**까지 감소 (73.4% 절감).

| 최적화 항목 | Before | After | 효과 |
|-------------|--------|-------|------|
| **GQA FlashAttention native** | `_repeat_kv()` → FA2 | 직접 GQA 전달 | VRAM **-12.1 GB** (-20%) |
| **Batch size** | 8 | 4 | OOM 해소 |
| **최종 VRAM** | 60.4 GB → OOM | **48.3 GB** (26.4%) | 안정 구간 |

### 2.2 NCCL/DDP 튜닝

| 설정 | 값 | 목적 |
|------|-----|------|
| `gradient_as_bucket_view` | True | Zero-copy gradient → NCCL buffer |
| `bucket_cap_mb` | 800 (기본 400) | 3B 대형 gradient 대응 |
| `NCCL_ALGO` | Ring,Tree | Tree 단독 실패 → 혼합 |
| `NCCL_NVLS_ENABLE` | 1 | NVSwitch 하드웨어 all-reduce |
| `NCCL_MAX_NCHANNELS` | 32 | 통신 병렬도 최대화 |
| `NCCL_BUFFSIZE` | 67,108,864 (64 MB) | 대형 버퍼 |

### 2.3 GPU-CPU 동기화 최소화

- Gradient accumulation 중 `loss.detach()` 사용 (CPU sync 방지)
- `.item()` 호출을 optimizer step 당 **1회**로 제한
- 결과: **8회 → 1회/step** (-87.5% CPU sync)

### 2.4 torch.compile 시도 및 실패

- **원인**: Transformer Engine(TE)의 opaque FP8 kernel에서 graph break 다발
- **추가 장애**: `/tmp` noexec 마운트로 Triton AOT 컴파일 불가
- **결론**: 실측 **1.00x** — 효과 없음, 비활성화

### 2.5 NUMA 크로스 어피니티 발견

- **문제**: 8개 rank 중 5개가 잘못된 NUMA 노드에서 실행
- **영향**: 69% worker가 크로스-NUMA 메모리 접근 (latency +40%)
- **GPU-NUMA 매핑**: GPU 0-3 → NUMA0 (cores 0-35), GPU 4-7 → NUMA1 (cores 36-71)
- **예상 개선**: +4~9% throughput (미적용 — Phase 1에서는 안정성 우선)

### 2.6 Phase 0 성과 요약

| 지표 | 최적화 전 | 최적화 후 |
|------|-----------|-----------|
| VRAM/GPU | 60.4+ GB (OOM) | **48.3 GB** (26.4%) |
| Throughput (per GPU) | 32K tok/s | **36~38K tok/s** |
| GPU-CPU sync/step | 8회 | **1회** |
| 안정성 | OOM crash | **48시간+ 무사고** |

---

## 3. Phase 1: 사전학습 (57K steps, ~63시간)

### 3.1 학습 설정

| 항목 | 값 |
|------|-----|
| **총 스텝** | 57,000 |
| **Batch size** | 4 per GPU x 8 GPU x 8 accum = **256 sequences** |
| **Effective batch (tokens)** | 256 x 4,096 = **1,048,576 tok/step** |
| **총 학습 토큰** | 57K x 1.05M = **~60B tokens** |
| **학습률** | 1.5e-4 (peak), cosine decay |
| **Warmup** | 2,000 steps (3.5%) |
| **정밀도** | MXFP8 (B200 block scaling) + BF16 autocast |
| **Optimizer** | AdamW (fused, betas=0.9/0.95) |
| **Weight decay** | 0.1 |
| **Max grad norm** | 1.0 |

### 3.2 학습 과정

| 구간 | Step 범위 | Loss | Throughput | 비고 |
|------|-----------|------|------------|------|
| 초기 수렴 | 10~100 | 11.66 → 7.99 | 32~36K tok/s | CUDA warmup |
| 급속 하강 | 100~500 | 7.99 → 5.05 | 36K tok/s | LR warmup 진행 |
| 안정 학습 | 500~3,000 | 5.05 → 2.50 | 36~38K tok/s | LR peak 도달 |
| 미세 수렴 | 3,000~57,000 | 2.50 → 1.47 | 38~39K tok/s | Cosine decay |

### 3.3 최종 결과

| 지표 | 값 |
|------|-----|
| **최종 Training Loss** | **1.466** |
| **최종 Gradient Norm** | 0.097 (매우 안정) |
| **학습 시간** | 62.94시간 (2일 15시간) |
| **VRAM 사용** | 48.3 GB/GPU (일관) |
| **Throughput (per GPU)** | 38,500~38,750 tok/s (steady state) |
| **System throughput** | **~308K tok/s** (8 GPU) |
| **MFU** | ~33.5% |
| **사고** | 0건 (무사고 완료) |
| **체크포인트** | `checkpoints/korean_3b_fp8_run1/checkpoint-0057000` |

---

## 4. 사전학습 평가 결과

### 4.1 Perplexity (PPL)

**통합 검증 세트:**

| 데이터셋 | PPL | BPT (Bits/Token) | 평가 토큰 수 |
|----------|-----|-------------------|-------------|
| **3b_val (통합)** | **5.226** | 2.386 | 226.9M |
| korean_c4_val | 5.717 | 2.515 | 45.4M |
| korean_wiki_val | 11.836 | 3.565 | 1.57M |
| korean_namuwiki_val | 25.881 | 4.694 | 6.49M |

**확장 PPL (19개 데이터셋, 주요 항목):**

| 도메인 | 데이터셋 | PPL | 등급 |
|--------|----------|-----|------|
| 한국어 웹 | hplt_ko | **2.403** | 우수 |
| 수학 | mathpile | **2.724** | 우수 |
| 교육 | cosmo_khanacademy | **2.932** | 우수 |
| 교육 | cosmo_auto_math_text | **3.149** | 우수 |
| 백과 | cosmo_wikihow | 3.310 | 양호 |
| 영어 혼합 | cosmo_web_v2 | 4.166 | 양호 |
| 한국어 일반 | korean_general | 7.016 | 보통 |
| 백과사전 | wikipedia_ko | 10.706 | 보통 |
| 위키 | namuwiki_2023b | 18.917 | 불량 |
| 한국어 웹 | cc100_ko | 21.782 | 불량 |

### 4.2 토큰 수준 분석

| 지표 | 값 |
|------|-----|
| Top-1 정확도 | **68.75%** |
| Top-5 정확도 | **81.64%** |
| Top-10 정확도 | 85.93% |
| Mean Entropy | 1.568 bits |
| NLL > 5 비율 | 10.86% |
| NLL > 10 비율 | 1.18% |

### 4.3 벤치마크 결과

| 벤치마크 | Score | Random Baseline | 판정 |
|----------|-------|-----------------|------|
| **Belebele (kor)** | 0.2189 | 0.25 | Random 수준 |
| **MMLU-KO (전체)** | 0.2339 | 0.25 | Random 수준 |
| MMLU-KO 인문학 | 0.2389 | 0.25 | — |
| MMLU-KO 사회과학 | 0.2301 | 0.25 | — |
| MMLU-KO STEM | 0.2312 | 0.25 | — |

| 벤치마크 | Score | Random | 판정 |
|----------|-------|--------|------|
| **KoBEST BoolQ** | 50.14% | 50% | Random |
| **KoBEST COPA** | 49.40% | 50% | Random |
| **KoBEST HellaSwag** | 21.60% | 25% | Random |
| **KoBEST SentiNeg** | 50.13% | 50% | Random |
| **KoBEST WiC** | 48.81% | 50% | Random |
| **HAE-RAE** | 19.98% | 20% | Random |

> **해석**: Base 모델의 벤치마크 ~Random은 **정상**. 0-shot 벤치마크는 instruction-following 능력을 전제하므로,
> SFT 이전 base 모델에서 random 수준은 예상된 결과다. SFT 후 향상이 핵심 검증 포인트.

### 4.4 생성 품질

| Temperature | 3-gram 반복률 | EOS 종료율 | 비고 |
|-------------|--------------|-----------|------|
| 0.0 (greedy) | **72.75%** | 0% | 심각한 반복 (base 특성) |
| 0.7 | ~40% | 0% | 개선되나 여전히 높음 |
| 1.0 | 24.27% | 0% | 상대적 개선 |
| `no_repeat_ngram=3` | **0.00%** | 0% | 반복 완전 제거 |

> **해석**: EOS 미학습 + greedy 반복은 base model 고유 특성. SFT에서 대화 종료 패턴을 학습하면 해결 예상.

---

## 5. SFT 데이터 준비

### 5.1 데이터 소스

**24개 소스**에서 수집, 한국어/영어 혼합 instruction-following 데이터:

| # | 데이터셋 | 샘플 수 | 크기 | 특징 |
|---|---------|---------|------|------|
| 1 | reasoning_r1_1.4m | 1,400,000 | 14.77 GB | 최대 단일 소스, 추론 |
| 2 | openhermes_2.5 | 1,001,551 | 1.82 GB | 영어 다목적 |
| 3 | AI-MO_NuminaMath-CoT | 859,494 | 2.51 GB | 수학 CoT |
| 4 | korean_instruction_mix | 515,911 | 1.39 GB | 한국어 혼합 |
| 5 | lemon-mint_smol-koreantalk | 460,281 | 5.23 GB | 한국어 대화 |
| 6 | open_korean_instructions | 375,159 | 0.73 GB | 한국어 지시 |
| 7 | magpie_reasoning_v2 | 249,922 | 3.99 GB | 추론 (영어) |
| 8 | magpie_reasoning_ko | 224,929 | 3.19 GB | 추론 (한국어) |
| 9 | ultrachat_200k | 207,865 | 1.34 GB | 대화 |
| 10 | kuotient_orca-math-ko | 193,789 | 0.61 GB | 수학 (한국어) |
| 11 | data/sft/train.jsonl | 161,848 | 0.27 GB | 원본 SFT |
| 12 | kullm_v2 | 152,630 | 0.42 GB | 한국어 지시 |
| 13-24 | 기타 12개 소스 | ~462,000 | ~10.7 GB | 코드, 위키QA, 글쓰기 등 |

### 5.2 처리 파이프라인

```
24개 소스 (6.59M raw)
    ↓ prepare_sft_combined.sh
    ↓ 포맷 통일 (messages/conversations/alpaca → messages)
    ↓ MD5 중복 제거 (첫 user 메시지 기준)
    ↓ 98:2 train/val split (seed=42)
통합 데이터 (2.56M train + 52K val)
    ↓ filter_sft_v2.py
    ↓ 5단계 필터
최종 학습 데이터
```

### 5.3 필터링 (filter_sft_v2.py)

5단계 품질 필터:

| 단계 | 필터 | 기준 |
|------|------|------|
| 1 | EOS 태그 제거 | `</s>` 리터럴 strip |
| 2 | QA 마커 제거 | 질문:/답변:/Q:/A: 등 접두사 제거 |
| 3 | 최소 응답 길이 | < 50자 제거 |
| 4 | 최대 응답 길이 | > 20,000자 제거 |
| 5 | 4-gram 반복 | 반복률 > 30% 제거 |

### 5.4 최종 데이터 통계

| 구분 | 필터 전 | 필터 후 | 제거율 |
|------|---------|---------|--------|
| **Train** | 2,559,492 (7.79 GB) | **2,439,397** (7.48 GB) | 4.69% |
| **Val** | 52,234 (163 MB) | **49,801** (157 MB) | 4.66% |
| **합계** | 2,611,726 (7.95 GB) | **2,489,198** (7.63 GB) | 4.69% |

---

## 6. Phase 2: SFT 학습 (현재 진행 중)

### 6.1 학습 설정

| 항목 | 값 |
|------|-----|
| **Base 모델** | `checkpoint-0057000` (Phase 1 최종) |
| **총 스텝** | 33,000 |
| **Batch size** | 2 per GPU x 8 GPU x 4 accum = **64 sequences** |
| **학습률** | 1.0e-5 (pretrain의 1/15, catastrophic forgetting 방지) |
| **Warmup** | 500 steps |
| **LR Schedule** | Cosine decay (1e-5 → ~0) |
| **Weight decay** | 0.01 |
| **Max grad norm** | 1.0 |
| **NEFTune alpha** | 5.0 (임베딩 노이즈 주입 → 생성 다양성 향상) |
| **정밀도** | MXFP8 + BF16 |
| **Gradient checkpointing** | 활성화 |
| **Loss masking** | Prompt 토큰 = -1, Response 토큰만 학습 |
| **Chat template** | `<\|user\|>\n{input}\n<\|assistant\|>\n{output}</s>` |

### 6.2 진행 상황

**현재**: Step **2,000 / 33,000** (6.06%)
**시작**: 2026-03-05 22:15
**경과**: ~1시간 12분 (step 2000 기준)

### 6.3 Loss 추이

| Step | Train Loss | Val Loss | LR | 비고 |
|------|-----------|----------|-----|------|
| 10 | 2.2567 | — | 2.00e-7 | 초기 |
| 100 | 2.3083 | — | ~2.00e-6 | Warmup 중 |
| 250 | 1.9842 | — | ~5.00e-6 | 급속 하강 |
| 500 | 2.1380 | **2.0732** | **1.00e-5** | Warmup 완료 |
| 1,000 | 2.0748 | **2.0035** | 1.00e-5 | 안정 학습 |
| 1,500 | 2.0040 | **1.9745** | ~9.98e-6 | 수렴 지속 |
| **2,000** | **2.0527** | **1.9558** | ~9.95e-6 | **현재 BEST** |

**Val loss 추이**: 2.073 → 2.004 → 1.975 → **1.956** (일관된 하락)

### 6.4 안정성 지표

| 지표 | 값 | 판정 |
|------|-----|------|
| **Gradient norm** | 1.0~1.2 | 안정 (clipping 미작동) |
| **VRAM** | **24.2 GB/GPU** (13.2%) | 매우 여유 |
| **메모리 누수** | 없음 (48.3→24.2 고정) | 안정 |
| **학습 속도** | ~1.7 step/s 추정 | 정상 |
| **체크포인트** | checkpoint-0002000 (7.6 GB) | 정상 저장 |

> **VRAM 24.2 GB (vs Phase 1 48.3 GB)**: SFT는 batch_size=2 (pretrain의 절반), gradient accumulation도 4 (pretrain의 절반)로
> micro-batch 크기가 작아 activation memory가 대폭 감소.

---

## 7. 발견된 이슈 및 해결

### 전 Phase 누적 이슈 테이블

| # | Phase | 이슈 | 원인 | 해결 | 상태 |
|---|-------|------|------|------|------|
| 1 | 0 | OOM (bs=8) | TE FP8 activation buffer 비선형 증가 | bs=4 + GQA FA native | 해결 |
| 2 | 0 | TensorBoard import crash | 의존성 충돌 | try/except guard | 해결 |
| 3 | 0 | NCCL Tree 알고리즘 실패 | NVSwitch 토폴로지 불일치 | Ring,Tree 혼합 | 해결 |
| 4 | 0 | DDP static_graph + TE 충돌 | TE FP8 kernel 동적 그래프 필요 | static_graph 비활성화 | 해결 |
| 5 | 0 | te.Linear lm_head + weight tying 실패 | TE는 weight tying 미지원 | nn.Linear lm_head | 해결 |
| 6 | 0 | torch.compile 무효 | TE opaque kernel graph break | 비활성화 | 포기 |
| 7 | 0 | NUMA 크로스 어피니티 | OS 스케줄러 NUMA 미인식 | 발견만 (미적용) | 미적용 |
| 8 | 1 | Greedy 반복률 72.75% | Base model EOS 미학습 | SFT로 해결 예정 | SFT 중 |
| 9 | 1 | EOS 종료율 0% | Base model 대화 종료 미학습 | SFT chat template 적용 | SFT 중 |
| 10 | 1 | Namuwiki PPL 25.9 (불량) | 데이터 품질 문제 (위키 마크업) | 허용 (학습 데이터 필터링 한계) | 허용 |
| 11 | 2 | — | — | — | 진행 중, 이슈 없음 |

---

## 8. 현재 상태 요약 및 전망

### 현재 상태

```
Phase 0 (최적화)     ████████████████████ 100%  완료
Phase 1 (사전학습)   ████████████████████ 100%  완료 (57K/57K steps)
Phase 2 (SFT)        █░░░░░░░░░░░░░░░░░░░  6%   진행 중 (2K/33K steps)
```

### 핵심 수치 요약

| 단계 | 핵심 지표 | 값 |
|------|-----------|-----|
| Phase 0 | VRAM 절감 | 60.4 → **48.3 GB** (-20%) |
| Phase 1 | 최종 Loss | **1.466** |
| Phase 1 | 학습 토큰 | **~60B tokens** |
| Phase 1 | 통합 Val PPL | **5.226** |
| Phase 2 | 현재 Val Loss | **1.956** (step 2000, 하락 중) |
| Phase 2 | VRAM | **24.2 GB** (13.2%, 매우 안전) |

### 예상 완료 시간

| 항목 | 추정 |
|------|------|
| 남은 스텝 | 31,000 (93.9%) |
| 추정 속도 | ~1.7 step/s |
| **예상 잔여 시간** | **~5시간** |
| **예상 완료 시각** | 2026-03-06 04:30경 |

### 다음 단계 (Phase 2 완료 후)

1. **SFT 모델 평가**
   - KoBEST, MMLU-KO 재평가 (base→SFT 향상 폭 측정)
   - 생성 품질: 반복률, EOS 종료율 재측정
   - PPL 변화 모니터링 (catastrophic forgetting 확인)

2. **GGUF 변환 및 로컬 배포**
   - llama.cpp 호환 GGUF 형식 변환
   - 양자화 옵션 탐색 (Q4_K_M, Q5_K_M, Q8_0)

3. **선택적 후속 학습**
   - DPO/ORPO (선호도 기반 정렬) — 필요 시
   - 추가 SFT 라운드 (특정 도메인 강화) — 필요 시

---

## 부록: 주요 파일 경로

| 항목 | 경로 |
|------|------|
| **모델 코드** | `model/config.py`, `model/transformer.py` |
| **사전학습 설정** | `configs/3b_pretrain.yaml` |
| **SFT 설정** | `configs/korean_3b_sft.yaml` |
| **사전학습 스크립트** | `train/pretrain.py`, `train/trainer.py` |
| **SFT 스크립트** | `train/sft.py` |
| **SFT 런치** | `scripts/launch_3b_sft.sh` |
| **데이터 준비** | `scripts/prepare_sft_combined.sh` |
| **데이터 필터** | `data/filter_sft_v2.py` |
| **학습 데이터** | `data/sft_combined/train_filtered.jsonl` (2.44M) |
| **검증 데이터** | `data/sft_combined/val_filtered.jsonl` (49.8K) |
| **Pretrain 체크포인트** | `checkpoints/korean_3b_fp8_run1/checkpoint-0057000/` |
| **SFT 체크포인트** | `checkpoints/korean_3b_sft_v1/checkpoint-0002000/` |
| **Phase 0 보고서** | `reports/2026-03-02_0200_FRANKENSTALLM_phase0_optimization_report.md` |
| **Base 평가 보고서** | `reports/2026-03-05_3B_BASE_EVALUATION_REPORT.md` |
| **SFT 학습 로그** | `checkpoints/korean_3b_sft_v1/train.log` |
