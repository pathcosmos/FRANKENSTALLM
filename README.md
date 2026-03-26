# FRANKENSTALLM

> 🌍 [English Version](./README_en.md)

![Phase 4](https://img.shields.io/badge/Phase_4-배포_완료-brightgreen)
[![Model](https://img.shields.io/badge/Model-3B_Korean_LLM-green)](https://huggingface.co/pathcosmos/frankenstallm)
![GPU](https://img.shields.io/badge/GPU-8×_NVIDIA_B200-76b900)
![FP8](https://img.shields.io/badge/Precision-MXFP8-orange)
![ORPO](https://img.shields.io/badge/ORPO-완료_eval__loss_1.625-success)
[![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-배포_완료-ff9900)](https://huggingface.co/pathcosmos/frankenstallm)

> **한국어 3B LLM을 8× NVIDIA B200 위에서 처음부터 직접 만든다.**
> Frankenstein처럼 조각을 이어 붙이고, 철강처럼 단단하게 단련한다.

GitHub: [`pathcosmos/FRANKENSTALLM`](https://github.com/pathcosmos/FRANKENSTALLM)  
🤗 HuggingFace: [`pathcosmos/frankenstallm`](https://huggingface.co/pathcosmos/frankenstallm) — **모델 배포 완료 (GGUF + safetensors)**

---

## 목차

1. [왜 이 프로젝트인가](#1-왜-이-프로젝트인가)
2. [현재 상태 — 한눈에 보기](#2-현재-상태--한눈에-보기)
3. [하드웨어 환경](#3-하드웨어-환경)
4. [프로젝트 구조](#4-프로젝트-구조)
5. [프로젝트 여정 타임라인](#5-프로젝트-여정-타임라인)
6. [모델 아키텍처](#6-모델-아키텍처)
7. [학습 데이터](#7-학습-데이터)
8. [학습 설정 및 최적화](#8-학습-설정-및-최적화)
9. [실험 결과 — 1B 베이스라인](#9-실험-결과--1b-베이스라인)
10. [실험 결과 — 3B Base 종합 평가 (v2)](#10-실험-결과--3b-base-종합-평가-v2)
    - [10.1 학습 커브](#101-학습-커브)
    - [10.2 PPL (Perplexity) — 19개 데이터셋](#102-ppl-perplexity--19개-데이터셋)
    - [10.3 한국어 벤치마크](#103-한국어-벤치마크)
    - [10.4 영어 벤치마크](#104-영어-벤치마크)
    - [10.5 Calibration](#105-calibration)
    - [10.6 0-shot vs 5-shot 비교](#106-0-shot-vs-5-shot-비교)
    - [10.7 참고 모델 비교](#107-참고-모델-비교)
    - [10.8 생성 품질 및 파라미터 그리드 서치](#108-생성-품질-및-파라미터-그리드-서치)
    - [10.9 평가 파이프라인](#109-평가-파이프라인)
11. [실험 결과 — 3B SFT 종합 평가](#11-실험-결과--3b-sft-종합-평가)
    - [11.1 SFT 학습 결과](#111-sft-학습-결과)
    - [11.2 6차원 평가 요약](#112-6차원-평가-요약)
    - [11.3 Base vs SFT 비교](#113-base-vs-sft-비교)
    - [11.4 코드 개선 사항](#114-코드-개선-사항)
    - [11.5 ORPO 진행 판정](#115-orpo-진행-판정)
12. [Phase 3 — ORPO (선호도 정렬)](#12-phase-3--orpo-선호도-정렬)
20. [HuggingFace 배포 현황](#20-huggingface-배포-현황)
21. [Ollama 사용법 — 상세 설명 및 주의사항](#21-ollama-사용법--상세-설명-및-주의사항)
22. [모델 성능 비교](#22-모델-성능-비교--base--sft--orpo--ollama)
23. [재현 가이드 — 전 단계 설정 상세](#23-재현-가이드--전-단계-설정-상세)
    - [12.1 ORPO 선택 배경](#121-orpo-선택-배경)
    - [12.2 데이터](#122-데이터)
    - [12.3 HP Sweep 설계](#123-hp-sweep-설계-6-config)
    - [12.4 시도 이력](#124-시도-이력--5번의-실패)
    - [12.5 스윕 결과 (완료)](#125-스윕-결과-진행-중)
    - [12.7 ORPO 본 학습 (완료)](#127-orpo-본-학습-진행-중-2026-03-09)
    - [12.8 ORPO 종합 평가 파이프라인](#128-orpo-종합-평가-파이프라인)
13. [실행 방법](#13-실행-방법)
14. [로드맵](#14-로드맵)
15. [참고 문서](#15-참고-문서)
16. [기술 스택 요약](#16-기술-스택-요약)
17. [관련 프로젝트](#관련-프로젝트)
18. [다음 최적화 계획](#18-다음-최적화-계획--mfu-335--47-목표)
19. [GPU 하드웨어 & 비용 분석](#19-gpu-하드웨어--비용-분석--3b--60b-프리트레인)

---

## 1. 왜 이 프로젝트인가

한국어 LLM 생태계는 빠르게 성장하고 있다. 그러나 대부분의 공개 모델은 영어 기반 사전학습 위에 한국어 파인튜닝을 얹은 형태거나, 학습 과정이 공개되지 않아 재현이 불가능하다.

이 프로젝트는 다르다.

- **처음부터(from scratch)**: 토크나이저 학습부터 프리트레인, SFT, 선호도 정렬까지 모든 단계를 직접 구현한다.
- **완전 공개 빌더 로그**: 성공만 기록하지 않는다. 버그, 실패, 판단 착오, 그리고 그 원인 분석까지 모두 기록한다.
- **실용적인 규모**: 학술 논문용 장난감 모델(125M)도 아니고, 연구소가 아니면 재현 불가능한 70B도 아닌, **3B 규모**의 실용적 한국어 모델이 목표다.
- **B200 최적화**: NVIDIA B200의 FP8 Tensor Core, NVLink 5.0, FlashAttention-2를 최대한 활용한다. 최신 하드웨어를 최대로 쥐어짜는 과정 자체가 학습이다.


이 README는 **처음부터 끝까지의 빌더 로그**다. 실패와 성공을 모두 기록했다.

---

## 2. 현재 상태 — 한눈에 보기

```
2026-03-26 기준 (전 단계 완료)
```

| 단계 | 상태 | 세부 내용 |
|------|------|-----------|
| Phase 0: 기반 구축 | ✅ 완료 | OOM 수정, GQA FA 최적화, NCCL NVLS, 파이프라인 준비 |
| Phase 1: 3B Pretrain | ✅ 완료 | 57,000 steps, loss 1.466, ~63시간 |
| Phase 2: SFT | ✅ 완료 | 25,500 steps (early stop), val_loss 1.8851, ~15.5시간 |
| Phase 2.5: SFT 평가 | ✅ 완료 | 6차원 평가 4/6 PASS, ORPO 진행 결정 |
| Phase 3: ORPO Sweep | ✅ 완료 | 6-config sweep 완료, best: lr=1.2e-5, beta=0.25 |
| **Phase 3: ORPO 본 학습** | **✅ 완료** | **9,997 steps 조기수렴, eval_loss 1.625, pref_acc 76.02%, 7/10 PASS** |
| **Phase 4: GGUF 변환·배포** | **✅ 완료** | **byte-fallback 수정, v1/v2 각 3종 양자화, HuggingFace + Ollama 배포** |

### Phase 2 (SFT) 최종 결과

| 항목 | 값 |
|------|-----|
| 최종 step | **25,500 / 33,000** (77.3%, early stopping) |
| **Val loss (best)** | **1.8851** (step 23,000) |
| 학습 시간 | **~15시간 41분** (2026-03-05 22:15 ~ 2026-03-06 13:56) |
| VRAM 사용 | **24.2GB** / 183GB per GPU (13.2%) |
| Base 모델 | checkpoint-0057000 (pretrain loss 1.466) |
| SFT 데이터 | **2,439,397 samples** (24개 소스, 7.48 GB) |
| 사고 | 0건 (OOM, NCCL, NaN 없음) |

**SFT Val Loss 전체 추이**:
```
Step     500: 2.073
Step   2,000: 1.956  (-0.117)
Step   5,000: 1.911  (-0.045)
Step  10,000: 1.892  (-0.019)
Step  15,000: 1.886  (-0.006)
Step  20,000: 1.885  (-0.001)
Step  23,000: 1.8851 ← BEST
Step  25,500: 1.8851 → Early Stop (patience 5/5)
```

### SFT 6차원 평가 요약

| 차원 | 결과 | 핵심 수치 |
|------|------|-----------|
| Perplexity (지식 보존) | **PASS** | forgetting 0.9% |
| 생성 품질 | **FAIL** | Greedy 반복률 72.97% |
| 한국어 벤치마크 | **FAIL** | KoBEST 평균 43.26% |
| 영어 벤치마크 | **PASS** | 전 태스크 하한 초과 |
| Calibration | **PASS** | Top-1 68.59% |
| SFT Chat 능력 | **PASS** | EOS 종료율 60% (Base 0%) |

> **판정: ORPO 진행** — 지식 보존 우수(0.9%), 반복률은 선호도 정렬로 해결.
> 상세: `reports/2026-03-06_3B_SFT_COMPLETION_AND_EVAL_SUMMARY.md`

---

## 3. 하드웨어 환경

### GPU

| 항목 | 사양 |
|------|------|
| 모델 | 8× NVIDIA B200 |
| VRAM | 183GB HBM3e per GPU (~1.47TB 합계) |
| FP8 Tensor Core | 2,250 TFLOPS/GPU (총 18,000 TFLOPS) |
| BF16 | 1,125 TFLOPS/GPU |
| HBM3e 대역폭 | ~7.67 TB/s per GPU |
| 인터커넥트 | NVLink 5.0 (900 GB/s bidirectional per GPU) |
| 토폴로지 | NVSwitch — 모든 GPU↔GPU 단일 홉 All-to-All Mesh |
| 전력 | 940W 실측 / 1000W cap |

B200은 FP8 네이티브 지원 모델이다. `torch.float8_e4m3fn` 을 TransformerEngine의 MXFP8 레시피와 결합해 학습한다. BF16 대비 연산량이 이론상 2배이며, 메모리 효율도 향상된다.

### CPU 및 시스템 메모리

| 항목 | 사양 |
|------|------|
| CPU | 2× AMD EPYC 9365 (Turin / Zen 5) |
| 물리 코어 | 72개 (36코어 × 2소켓) |
| NUMA 구성 | 2노드: node0 (core 0-35) / node1 (core 36-71) |
| GPU↔NUMA 매핑 | GPU 0-3 → NUMA node 0, GPU 4-7 → NUMA node 1 |
| RAM | 2.21TB DDR5 (~2.03TB 여유) |
| L3 캐시 | 384MB (12 CCX × 32MB) |

**NUMA 주의**: 초기 DDP 런칭 시 5/8 rank가 잘못된 NUMA 노드에서 실행되는 문제 발생. 69%의 DataLoader worker가 크로스-NUMA였다. NUMA affinity 최적화는 미적용 상태(로드맵 항목).

### 스토리지

| 경로 | 용도 | 여유 공간 |
|------|------|-----------|
| `/PROJECT/0325120031_A/ghong/taketimes/llm-bang/` | 메인 작업 (체크포인트, 데이터) | 2.2TB |
| `/home/ghong/` | 소규모 코드 | 5GB (제한) |

> **주의**: 체크포인트(수십 GB), 학습 데이터(82GB+), 중간 산출물은 모두 `/PROJECT/...` 경로에 저장한다. 홈 디렉토리 용량 초과 위험.

### 소프트웨어 환경

| 패키지 | 버전 |
|--------|------|
| PyTorch | `2.10.0a0+b4e4ee81d3.nv25.12` (NVIDIA 커스텀) |
| FlashAttention | 2.7.4.post1+25.12 |
| TransformerEngine | 2.10.0 |
| NCCL | 2.28.9 |
| Triton | 3.5.1 |
| CUDA | 13.1 |
| Driver | 580.95.05 |

> **경고**: PyTorch는 NVIDIA B200 최적화 커스텀 빌드다. `pip install torch`로 재설치하면 B200 최적화가 깨진다. **절대 재설치 금지.**

---

## 4. 프로젝트 구조

```
llm-bang/
├── CLAUDE.md                          # Claude Code 가이드
├── README.md                          # 이 파일
├── PROGRESS.md                        # 진행 기록 (날짜별 로그)
├── Modelfile.3b                       # Ollama 모델 파일
│
├── configs/
│   ├── korean_3b_fp8.yaml             # 3B FP8 학습 설정 (현재 사용 중)
│   ├── 3b_pretrain.yaml               # 3B 프리트레인 설정 (대체)
│   ├── korean_1b_fp8.yaml             # 1B FP8 설정 (아카이브)
│   ├── korean_3b_sft.yaml             # 3B SFT v1 설정 (완료)
│   ├── korean_3b_sft_v2.yaml          # 3B SFT v2 설정 (lr=5e-5, data mixing)
│   ├── korean_3b_orpo.yaml            # 3B ORPO 설정 (lr=5e-6, beta=0.1)
│   ├── hybrid_3b.yaml                 # Hybrid 3B (Mamba-2 + Attention)
│   ├── small_fp8.yaml                 # 125M FP8 검증용
│   ├── medium.yaml                    # 중형 모델 설정
│   └── small.yaml                     # 소형 모델 설정
│
├── data/
│   ├── 3b_train.bin                   # 프리트레인 학습 데이터 (82GB, 41.12B tokens)
│   ├── 3b_val.bin                     # 검증 데이터 (151MB)
│   ├── cc100_ko_train.bin             # CC100 한국어 (4.5GB)
│   ├── cosmo_auto_math_text_train.bin # 수학 텍스트 (2.6GB)
│   └── build scripts, __init__.py
│
├── model/
│   ├── attention.py                   # GQA FlashAttention (Phase 0 최적화 적용)
│   ├── transformer.py                 # 트랜스포머 메인 아키텍처
│   ├── config.py                      # 모델 설정 dataclass
│   └── layers.py                      # 커스텀 레이어 (RMSNorm, SwiGLU 등)
│
├── train/
│   ├── pretrain.py                    # 프리트레인 스크립트 (DDP 최적화)
│   ├── sft.py                         # SFT 학습
│   ├── orpo.py                        # ORPO 학습
│   ├── trainer.py                     # 통합 트레이너 (loss sync 최적화)
│   └── utils.py                       # 유틸리티 (NCCL 7200s timeout 등)
│
├── scripts/
│   ├── launch_3b_pretrain.sh          # 3B 프리트레인 런처 (NCCL 환경변수 포함)
│   ├── launch_3b_sft.sh               # 3B SFT v1 런처
│   ├── launch_3b_sft_v2.sh            # 3B SFT v2 런처 (data mixing)
│   ├── launch_3b_orpo.sh              # 3B ORPO 런처
│   ├── monitor_3b.sh                  # 실시간 학습 모니터
│   ├── training_watchdog.sh           # 워치독 (10분 간격, 크론)
│   ├── convert_3b_gguf.sh             # GGUF 변환 스크립트
│   ├── deploy_3b_ollama.sh            # Ollama 배포
│   ├── quality_gate.sh                # 배포 전 품질 게이트
│   ├── telegram_notify.py             # 텔레그램 알림 (urllib 사용, curl 차단)
│   └── hourly_status.sh               # 1시간 간격 상태 리포트
│
├── eval/
│   ├── debate/
│   │   └── justice_league_3b_case.md  # 3B 전환 논증 (저스티스리그 멀티에이전트)
│   ├── decision/
│   │   └── FINAL_DECISION_REPORT.md   # SFT 재시작 판결문
│   ├── plan/
│   │   └── 3B_MASTER_PLAN.md          # 3B 마스터 플랜
│   ├── tasks/                         # 모듈화된 평가 태스크
│   │   ├── task_runner.py             # 8-GPU 병렬 태스크 실행기
│   │   ├── ppl_task.py                # Perplexity 평가 태스크
│   │   ├── lm_eval_task.py            # lm-evaluation-harness 래퍼
│   │   ├── calibration_task.py        # Calibration 분석
│   │   ├── generation_task.py         # 생성 품질 + 파라미터 그리드 서치
│   │   └── token_nll_task.py          # Token NLL 분포 분석
│   ├── outputs/                       # 평가 결과 (자동 생성, .gitignore)
│   ├── full_eval_pipeline.py          # v2 종합 평가 파이프라인 (8-GPU 병렬)
│   ├── sft_eval_pipeline.py           # SFT 6차원 평가 파이프라인
│   ├── reeval_pipeline.py             # 재평가 파이프라인 (0+5-shot 연속)
│   ├── report_generator.py            # 마크다운 리포트 자동 생성
│   ├── comprehensive_eval.py          # v1 종합 평가 (레거시)
│   └── test_generation_params.py      # 생성 파라미터 탐색
│
├── tokenizer/
│   ├── korean_sp/                     # SentencePiece 64K 모델 파일
│   ├── tokenizer.json                 # HuggingFace 포맷 (2.4MB)
│   ├── train_sp_tokenizer.py          # 토크나이저 학습 스크립트
│   └── convert_sp_to_hf.py            # SentencePiece → HF 변환
│
├── checkpoints/                       # 모델 체크포인트 (대용량, .gitignore)
│
├── docs/
│   ├── PROJECT_HISTORY.md             # 프로젝트 전체 여정 상세 기록
│   └── 3B_WORKPLAN.md                 # 3B 작업 계획
│
└── reports/
    ├── 2026-03-02_0200_FRANKENSTALLM_phase0_optimization_report.md
    ├── 2026-03-05_3B_BASE_EVALUATION_REPORT.md
    ├── 2026-03-05_3B_SFT_PROGRESS_REPORT.md   # SFT 학습 보고서 (Phase 2)
    ├── 2026-03-05_3B_NEXT_STEPS_REFERENCE.md
    ├── 2026-03-05_NEMOTRON_NANO_FEASIBILITY_STUDY.md
    ├── 2026-03-05_PPL_EVALUATION.md
    ├── 2026-03-05_BENCHMARK_RESULTS.md
    ├── 2026-03-05_GENERATION_QUALITY.md
    ├── 2026-03-06_3B_SFT_EVAL_PLAN.md         # SFT 6차원 평가 계획서
    ├── 2026-03-06_3B_SFT_EVALUATION_REPORT.md  # SFT 6차원 평가 결과
    └── 2026-03-06_3B_SFT_COMPLETION_AND_EVAL_SUMMARY.md  # SFT 완료 + 코드 개선 종합
```

---

## 5. 프로젝트 여정 타임라인

이 섹션이 이 README의 핵심이다. 결과만이 아니라 **왜** 그런 결정을 내렸는지, **어디서** 실패했는지를 솔직하게 기록한다.

---

### Day 1 (Feb 25) — 첫 불씨: 125M FP8 검증

프로젝트의 시작은 작은 의문에서 출발했다. B200에서 FP8이 실제로 안정적으로 학습되는가?

TransformerEngine의 MXFP8 레시피를 125M 소형 모델에 적용해 검증했다. 결론은 **안정적으로 동작한다**. loss 수렴도 정상이었고, VRAM 효율도 BF16 대비 확연한 개선이 있었다. 이 검증이 전체 파이프라인의 첫 번째 녹색 신호였다.

같은 날, 인프라 세팅도 완료했다. DDP 8-GPU 환경, NCCL 환경변수, 체크포인트 저장 경로, 텔레그램 알림 시스템의 초안이 이날 갖춰졌다.

---

### Day 1~2 (Feb 25~26) — 1B 프리트레인: 34K 스텝, PPL 5.67

125M 검증 직후 1B 모델 프리트레인에 돌입했다.

- **아키텍처**: d_model=2048, 24 layers, GQA 4:1, SwiGLU, RoPE
- **데이터**: C4 Korean 기반
- **학습**: 34,000 스텝, FP8, 8× B200 DDP

최종 결과:
- **Loss: 1.904**
- **PPL (C4 Korean): 5.67**

수치만 보면 그럭저럭 괜찮다. 그러나 실제 텍스트 생성을 시켜보면 문제가 보였다. 반복 패턴, 어색한 문장 구조, 맥락 이탈. 프리트레인 모델이니 당연하다. 이제 SFT 차례였다.

---

### Day 2 (Feb 26) — SFT v1: 0.0이라는 재앙

SFT를 돌렸다. 학습이 시작되자마자 loss가 빠르게 떨어지기 시작했다. 처음엔 좋은 신호라고 생각했다.

그런데 loss가 **0.0**이 됐다.

val loss도 0.0. 생성 결과는 완전한 쓰레기였다.

원인을 찾았다: **label off-by-one 버그**. 입력 토큰과 레이블 토큰이 한 칸씩 밀려 있었다. 모델이 실제로 다음 토큰을 예측하는 것이 아니라, 이미 알고 있는 정답을 맞추는 구조가 돼 있었다. loss가 0이 된 건 "완벽한 학습"이 아니라 **데이터 누수(label leakage)** 였다.

하루를 날렸다.

---

### Day 3 (Feb 27) — 5가지 버그, 루트 코즈 분석

실패를 분석하기 위해 **5-에이전트 루트 코즈 분석**을 수행했다. 결론은 버그 하나가 아니었다. SFT 파이프라인 전체에 문제가 있었다.

발견된 5가지 핵심 버그:

| 버그 | 증상 | 영향 |
|------|------|------|
| Static padding (no packing) | 짧은 샘플도 max_len으로 패딩 | GPU 낭비, 학습 비효율 |
| EOS 토큰 절단 | 응답 끝에 EOS가 없음 | 모델이 "문장 끝"을 못 배움 |
| 단일 에폭 | 데이터를 한 번만 봄 | 언더피팅 |
| 검증 분리 없음 | val_loss 측정 불가 | 오버피팅 감지 불가 |
| 데이터 품질 | 노이즈, 중복, 불균형 | 반복 생성 패턴 유도 |

특히 EOS 절단 버그는 subtle하다. 모델이 응답을 마치는 시점을 배우지 못하면, 생성 시 끊임없이 같은 패턴을 반복하거나 의미 없는 토큰을 이어붙인다. 18% 반복률의 원인 중 하나였다.

---

### Day 3 (Feb 27) — SFT v2: 성공이지만 18% 반복

5가지 버그를 모두 수정하고 SFT v2를 돌렸다.

- **val_loss: 2.2062** — 합리적 수준
- **반복률: 18%** (rep_penalty=1.1 적용 후)

생성 품질은 v1에 비해 확연히 개선됐다. 하지만 18% 반복률은 여전히 높다. `rep_penalty`를 높이면 반복은 줄지만 생성 다양성도 줄고 어색해진다. 디코딩 파라미터로 해결하기엔 구조적 한계가 있다.

kobest_copa 기준 0.646. 괜찮은 수치이지만 목표에는 미치지 못한다.

---

### Day 3 (Feb 27) — "저스티스리그 vs 어벤저스": 3B 전환 결정

반복률 18%를 놓고 팀 내부 토론이 벌어졌다. 핵심 질문은 하나였다:

> **ORPO로 반복을 잡을 수 있는가, 아니면 3B로 가야 하는가?**

이 질문에 답하기 위해 **멀티에이전트 토론**을 수행했다 (코드명: "저스티스리그 vs 어벤저스"). 각 에이전트가 다른 입장을 맡아 논증했다.

토론의 핵심 발견:

1. **18% 반복은 1B 파라미터의 구조적 한계**다. 1B 모델은 장거리 의존성(long-range dependency)을 충분히 포착하지 못한다. ORPO 같은 선호도 정렬은 반복을 줄이는 데 일부 도움이 되지만, 근본 원인(파라미터 부족)을 해결하지는 못한다.

2. **스케일링 법칙 분석**: Chinchilla 법칙과 실험 데이터를 기반으로 3B 모델은 동일 데이터에서 반복률을 5~8%까지 낮출 수 있다는 추정이 나왔다.

3. **비용-편익 분석**: ORPO를 1B에 투자하는 것보다 3B 프리트레인에 투자하는 것이 최종 모델 품질 측면에서 우월하다.

**결론: 3B 전환**. 1B는 아카이브하고 3B 프리트레인을 시작한다.

이 결정은 `eval/debate/justice_league_3b_case.md`에 전체 논증과 함께 기록돼 있다.

---

### Day 3 (Feb 27) — 640GB+ 데이터 조립

3B 전환이 결정되자마자 데이터 파이프라인을 가동했다. 1B에 비해 훨씬 많은 데이터가 필요하다 (Chinchilla 최적 비율: 3B 모델 × 20 = 60B tokens).

최종적으로 조립한 데이터:
- **총 토큰**: 41.12B tokens (최종 이진 파일)
- **원시 데이터**: 640GB+ 다국어 텍스트
- **소스**: C4 Korean, 나무위키, Wikipedia Korean, korean_extra 데이터셋

데이터 전처리(토크나이즈, 셔플, 이진 변환)가 완료된 `data/3b_train.bin`은 82GB다. 검증셋 `data/3b_val.bin`은 151MB.

---

### Mar 2 — Phase 0: OOM 격퇴 및 최적화

3B 학습을 처음 시작하자 OOM(Out of Memory)이 발생했다. 183GB VRAM인데 3B 모델이 OOM이 난다는 게 이상하지만, 원인은 있었다.

**GQA FlashAttention 구현 문제**였다. GQA(Grouped-Query Attention)에서 KV 캐시를 expand하는 방식이 메모리를 불필요하게 복사하고 있었다. FlashAttention의 native GQA support를 제대로 활용하지 않은 것이다.

Phase 0에서 수행한 최적화 목록:

| 최적화 | 방법 | 효과 |
|--------|------|------|
| GQA FA Native | `flash_attn_varlen_func` native GQA 경로 사용 | VRAM 60.4GB → 48.3GB (**-20%**) |
| DDP 최적화 | `gradient_as_bucket_view=True` | GPU-CPU 동기화 오버헤드 -87.5% |
| NCCL NVLS | Ring+Tree 토폴로지, NVLS 활성화 | AllReduce 효율 개선 |
| 배치 크기 분석 | GPU 2,4,6의 NCCL relay node 역할 파악 | bs=5 최적, bs=6 위험 판정 |
| SIGHUP 방어 | nohup+setsid + Python signal handler + emergency ckpt | 3중 보호 |
| 모니터링 | Telegram Bot (B200Bot) + cron | 10분 워치독, 1시간 상태 리포트 |

**torch.compile 테스트**: 효과 없음(1.00x). 원인은 TransformerEngine의 opaque kernel이 graph break를 유발하고, `/tmp` 디렉토리에 noexec 플래그가 걸려 있어 컴파일된 kernel 캐시가 쓰이지 않았다. 시간 낭비를 한 셈이지만, "효과 없다"는 것을 실측으로 확인한 것도 성과다.

**bs=5의 이유**: NCCL ring topology에서 GPU 2, 4, 6이 relay node 역할을 맡는다. 이 GPU들은 다른 GPU보다 약 11GB를 더 사용한다. bs=5에서는 여유가 있지만, bs=6으로 올리면 이 relay GPU들이 183GB 경계에 너무 가까워진다. 안전 마진을 위해 bs=5를 유지한다.

---

### Mar 2~Mar 5 — Phase 1: 3B 프리트레인 완료

Phase 0 최적화가 완료된 후 Phase 1이 시작됐다.

초기 지표 (step 3150):
- Loss: 2.38
- 처리 속도: 36K tok/s per rank
- 시스템 전체: ~292K tok/s (8 GPU)
- MFU: ~33.5%

MFU 33.5%는 처음에는 낮아 보일 수 있다. 하지만 TE MXFP8가 이미 최적화된 상태에서 나온 수치다. 이론적 피크(18,000 TFLOPS) 대비 실효율이다. 추가 최적화 여지로 QKV fusion (+8~12%), NUMA affinity (+4~9%), FA2 native RoPE (+3~5%)가 남아있다.

**Phase 1 완료 (2026-03-05)**:

- **57,000 steps 완료**, 최종 loss **1.466**
- 41.12B 토큰 처리, 총 학습 시간 약 63시간
- 무사고 완료 (SIGHUP, OOM, NCCL 이상 없음)

종합 평가 결과 요약 (v2 재평가 반영):

| 항목 | 결과 |
|------|------|
| PPL (통합 검증셋) | 5.2263 (초기 v1 평가: 5.709) |
| PPL (C4 Korean) | 5.717 |
| KoBEST 평균 (5태스크) | 43.69% |
| MMLU-KO 평균 (6카테고리) | 22.75% |
| HAE-RAE | 19.71% |
| winogrande / piqa | 50.59% / 52.50% |
| Calibration Top-1 | 68.75% |
| Greedy 3-gram 반복률 | 60.99% (SFT 후 개선 예정) |
| 최적 생성 파라미터 | temp=0.7, rep_penalty=1.3 → 반복률 0% |

**SFT 진행 결정**: loss 1.466은 건강한 학습 완료 시그널. PPL/반복률/벤치마크 모두 SFT가 해결할 영역. 모델 구조 문제 징후 없음. → Phase 2 SFT 진행.

---

### Mar 5~ — Phase 2: 3B SFT 시작 — 2.44M 샘플, val_loss 1.956

Phase 1 완료 직후, 대규모 SFT 데이터를 준비하고 학습을 시작했다.

**데이터 파이프라인**:
- **24개 소스**에서 6.59M raw samples 수집
- `prepare_sft_combined.sh`: 포맷 통일(6가지 포맷 → messages), MD5 중복 제거, 98:2 split
- `filter_sft_v2.py`: 5단계 품질 필터 (EOS strip, QA marker 제거, 길이 필터, 4-gram 반복 필터)
- 최종: **2,439,397 train + 49,801 val** (7.48 GB)

데이터 구성은 추론/CoT(38%), 한국어 지시(22.5%), 영어 다목적(16%), 수학(12%), 대화/코드(11.5%)로 균형을 맞췄다. 1B SFT의 161K에서 **15배 확대**한 규모다.

**SFT 설계 — 1B 실패에서 배운 교훈 반영**:

| 1B 교훈 | 3B SFT 적용 |
|---------|-------------|
| Label off-by-one → loss=0 | Loss masking 검증 (prompt=-1, response만 학습) |
| EOS 절단 → 종료 불가 | Chat template `<\|user\|>...<\|assistant\|>...</s>` EOS 포함 |
| Static padding → GPU 낭비 | Dynamic padding (64-token 정렬) |
| 검증 없음 → 오버피팅 미감지 | 49,801 val samples, 500 step 간격 eval |
| 데이터 노이즈 | 5단계 품질 필터 (1B에는 없었음) |
| 반복률 18% | **NEFTune alpha=5.0** 추가 (임베딩 노이즈 주입) |

**학습 설정**:
- LR: **1e-5** (pretrain의 1/15 — catastrophic forgetting 방지)
- Effective batch: 2 × 8 GPU × 4 accum = 64 sequences
- 33,000 steps (~3.3 epochs)
- MXFP8, gradient checkpointing, NCCL Ring+Tree

**초기 결과** (step 2,000, 6%):
- Val loss: 2.073 → 2.004 → 1.975 → **1.956** (단조 감소)
- Train-Val 갭 ~0.1 (오버피팅 징후 없음)
- VRAM 24.2 GB (13.2%) — pretrain의 절반, 매우 안정
- Grad norm 1.0 일정 (학습률 적절)

상세 보고서: `reports/2026-03-05_3B_SFT_PROGRESS_REPORT.md`

---

### Mar 6 — Phase 2 완료: SFT Early Stopping (val_loss 1.8851)

SFT는 33,000 steps 중 **25,500 steps**에서 early stopping으로 종료되었다. Val loss는 step 23,000에서 1.8851에 도달한 뒤, 5회 연속 개선 없이 학습이 자동 중단되었다.

**총 학습 시간**: ~15시간 41분 (2026-03-05 22:15 ~ 2026-03-06 13:56)

이 결과는 LR 1e-5의 cosine decay가 step 20K 이후 사실상 0에 수렴한 것과 일치한다. 모델은 주어진 LR schedule 하에서 학습 가능한 만큼 완전히 학습했다.

---

### Mar 6 — SFT 6차원 종합 평가: 4/6 PASS → ORPO 결정

SFT 체크포인트(`checkpoint-best`, step 23000)에 대해 6차원 종합 평가를 수행했다. 49분 27초 소요.

**핵심 결과**:
- **Perplexity**: forgetting 0.9% (19개 데이터셋 전체 PASS) — 지식 보존 우수
- **반복률**: greedy 72.97% (Base 60.99%보다 **악화**) — FAIL
- **EOS 종료율**: 0% → 60% — 개선됐지만 목표(90%) 미달
- **KoBEST**: 43.26% (Base 43.69%와 거의 동일) — FAIL
- **MMLU-KO**: 22.75% → 26.00% (+3.2pp) — 부분 개선
- **Calibration**: Top-1 68.59% — PASS

**결정**: greedy 반복률 72.97%는 SFT만으로 해결 불가. 그러나 `rep_penalty=1.2` 적용 시 반복률 0%가 달성되므로, ORPO(선호도 정렬)로 이 행동을 내재화하는 것이 올바른 경로다.

---

### Mar 6 — 코드 개선 및 ORPO 준비

SFT 평가와 병행하여 다수의 코드 개선 및 Phase 3 준비를 완료했다:

| 변경 | 내용 | 영향 |
|------|------|------|
| `train/sft.py` +238줄 | MixingDataLoader (SFT+pretrain 인터리빙), DDP rank 0 토크나이징 | forgetting 방지, 메모리 8배 절감 |
| `train/trainer.py` +17줄 | DDP early stopping broadcast (hang 방지), patience 5→10 | DDP 안정성 |
| `train/orpo.py` +30줄 | YAML config 지원, 3B 기본값 | ORPO 실행 준비 |
| `eval/report_generator.py` +831줄 | Base vs SFT 비교 보고서 자동 생성 | 평가 자동화 |
| `eval/sft_eval_pipeline.py` 신규 | SFT 6차원 평가 파이프라인 | 종합 평가 |
| `eval/tasks/generation_task.py` +75줄 | Chat template, 다양성 메트릭 | SFT 평가 |
| `configs/korean_3b_sft_v2.yaml` 신규 | SFT v2 설정 (lr=5e-5, data mixing 70/30) | 백업 경로 |
| `configs/korean_3b_orpo.yaml` 신규 | ORPO 설정 (lr=5e-6, beta=0.1) | Phase 3 |

상세: `reports/2026-03-06_3B_SFT_COMPLETION_AND_EVAL_SUMMARY.md`

---

## 6. 모델 아키텍처

### 1B (아카이브)

| 항목 | 값 |
|------|-----|
| vocab_size | 64,000 |
| d_model | 2,048 |
| n_layers | 24 |
| n_heads | 16 |
| n_kv_heads | 4 (GQA 4:1) |
| d_ffn | 5,461 (SwiGLU) |
| 파라미터 수 | ~1.19B |
| context | 2,048 |
| rope_theta | 500,000 |

### 3B (현재)

| 항목 | 값 |
|------|-----|
| vocab_size | 64,000 |
| d_model | 3,072 |
| n_layers | 28 |
| n_heads | 24 |
| n_kv_heads | 8 (GQA 3:1) |
| d_ffn | 8,192 (SwiGLU) |
| 파라미터 수 | ~3.0B |
| context | 2,048 |
| rope_theta | 500,000 |

### 공통 설계 원칙

| 컴포넌트 | 선택 | 이유 |
|----------|------|------|
| 정규화 | Pre-norm RMSNorm | Post-norm보다 학습 안정적 |
| 활성화 | SwiGLU FFN | Llama 계열에서 검증된 선택 |
| 위치 인코딩 | RoPE (θ=500K) | 긴 컨텍스트 확장 가능성 |
| 어텐션 | GQA (Grouped-Query Attention) | KV 캐시 메모리 절감 |
| 구현 | FlashAttention-2 | IO-aware, VRAM 효율 |
| 정밀도 | FP8 (MXFP8 via TransformerEngine) | B200 최적 활용 |

### GQA 비율 선택 근거

1B는 GQA 4:1 (head 16개, kv_head 4개), 3B는 GQA 3:1 (head 24개, kv_head 8개)을 선택했다. 3B에서 비율을 다소 완화한 이유는, 파라미터 수가 늘어나면서 어텐션 품질을 다소 희생하는 것이 3B 규모에서는 손해라는 판단이었다. Mistral 7B (GQA 8:1)와 Llama 3 (GQA 8:1)를 참고했다.

### rope_theta=500,000의 의미

표준 RoPE의 θ=10,000에서 500,000으로 늘린 것은 긴 컨텍스트에서 주파수 간섭을 줄이기 위해서다. Code Llama, Llama 3 등이 채택한 방식이다. 현재 max_seq_len=2048이므로 당장 효과를 보기는 어렵지만, 향후 컨텍스트 확장 파인튜닝을 위한 기반이다.

---

## 7. 학습 데이터

### 7.1 토크나이저

| 항목 | 값 |
|------|-----|
| 종류 | SentencePiece Unigram |
| 어휘 크기 | 64,000 |
| 한국어 문자 커버리지 | 99.95% |
| 위치 | `tokenizer/korean_sp/` |
| HF 포맷 | `tokenizer/tokenizer.json` (2.4MB) |

64K 어휘는 32K(너무 작음, 한국어 서브워드 단편화 심함)와 128K(너무 큼, 임베딩 레이어 오버헤드 증가) 사이의 균형이다. Llama 3(128K)와 GPT-4(100K)가 큰 어휘를 사용하는 추세지만, 3B 모델에서 128K 어휘는 임베딩 레이어만으로도 파라미터 비중이 지나치게 커진다.

### 7.2 프리트레인 데이터 — 전체 구성

최종 학습 파일: `data/3b_train.bin` (77GB, ~38.5B tokens) + `data/3b_val.bin` (145MB)

Chinchilla 법칙 기준: 3B × 20 = **60B 토큰**이 최적이다. 현재 38.5B 토큰을 57,000 스텝(batch 5 × accum 8 × seq 2048 × 8 GPU)으로 반복 소비하며, 처음 3B 학습으로서 합리적인 범위다.

#### 한국어 — 웹크롤 (Web Crawl)

| 데이터셋 | HuggingFace ID | 토큰화 파일 | 크기 | 추정 토큰 | 설명 |
|----------|---------------|------------|------|----------|------|
| C4 Korean | `allenai/c4` (ko subset) | `korean_c4_train.bin` | 15GB | ~7.5B | Google C4 한국어 필터링, 대규모 클린 웹 텍스트 |
| CC-100 Korean | `cc100` (ko subset) | `cc100_ko_train.bin` | 4.3GB | ~2.15B | Common Crawl 기반 단일언어 코퍼스 |
| HPLT Korean | `HPLT/hplt_monolingual_v2` (ko) | `hplt_ko_train.bin` | 15GB | ~7.5B | High Performance Language Technologies 웹 데이터 |

#### 한국어 — 백과사전 (Encyclopedia)

| 데이터셋 | HuggingFace ID | 토큰화 파일 | 크기 | 추정 토큰 | 설명 |
|----------|---------------|------------|------|----------|------|
| 위키백과 한국어 | `wikimedia/wikipedia` (20231101.ko) | `wikipedia_ko_train.bin` | 566MB | ~283M | 한국어 위키백과 전체, 구조화된 문어체 |
| 위키백과 한국어 (v2) | `wikimedia/wikipedia` (ko) | `korean_wiki_train.bin` | 500MB | ~250M | 위키백과 별도 버전 |
| 나무위키 | `heegyu/namuwiki-extracted` | `korean_namuwiki_train.bin` | 2.1GB | ~1.05B | 나무위키 추출본, 서브컬처·시사 풍부 |
| 나무위키 2023b | `heegyu/namuwiki-extracted` (2023b) | `namuwiki_2023b_train.bin` | 2.5GB | ~1.25B | 2023년 업데이트 스냅샷 |

#### 영어/다국어 — 교육 (Educational)

| 데이터셋 | HuggingFace ID | 토큰화 파일 | 크기 | 추정 토큰 | 설명 |
|----------|---------------|------------|------|----------|------|
| Cosmopedia Stories | `HuggingFaceTB/cosmopedia` | `cosmo_stories_train.bin` | 5.9GB | ~2.95B | 합성 교육용 스토리 |
| Cosmopedia Web v2 | `HuggingFaceTB/cosmopedia` | `cosmo_web_v2_train.bin` | 2.7GB | ~1.35B | 웹 기반 교육 텍스트 |
| Cosmopedia Stanford | `HuggingFaceTB/cosmopedia` | `cosmo_stanford_train.bin` | 2.1GB | ~1.05B | Stanford 강의 기반 |
| Cosmopedia WikiHow | `HuggingFaceTB/cosmopedia` | `cosmo_wikihow_train.bin` | 382MB | ~191M | WikiHow 가이드 |
| Cosmopedia OpenStax | `HuggingFaceTB/cosmopedia` | `cosmo_openstax_train.bin` | 224MB | ~112M | 오픈 교과서 |
| Cosmopedia Khan Academy | `HuggingFaceTB/cosmopedia` | `cosmo_khanacademy_train.bin` | 46MB | ~23M | 칸 아카데미 |

#### 영어/다국어 — 수학·과학 (Math & Science)

| 데이터셋 | HuggingFace ID | 토큰화 파일 | 크기 | 추정 토큰 | 설명 |
|----------|---------------|------------|------|----------|------|
| Open Web Math | `open-web-math/open-web-math` | `open_web_math_train.bin` | 4.8GB | ~2.4B | 웹에서 추출한 수학 텍스트 |
| MathPile | `GAIR/MathPile` | `mathpile_train.bin` | 2.9GB | ~1.45B | 수학 교과서·논문·포럼 |
| Cosmopedia AutoMath | `HuggingFaceTB/cosmopedia` | `cosmo_auto_math_text_train.bin` | 2.5GB | ~1.25B | 합성 수학 문제·풀이 |

#### 한국어 — 혼합 (Legacy Merged)

| 데이터셋 | 토큰화 파일 | 크기 | 추정 토큰 | 설명 |
|----------|------------|------|----------|------|
| 초기 혼합 (C4+나무+위키) | `korean_train.bin` | 17GB | ~8.5B | 1B 학습에 사용된 원본 혼합 데이터 |
| 125M 검증용 | `train.bin` | 1.2GB | ~600M | 최초 FP8 검증에 사용 |

#### 미사용 수집 데이터 (korean_extra/ — 640GB+)

`data/korean_extra/` 에 39개 서브디렉토리로 수집되었으나, 토큰화·병합은 일부만 완료된 대규모 원시 데이터:

| 분류 | 데이터셋 | 설명 | 비고 |
|------|----------|------|------|
| 웹크롤 | CulturaX Korean | 대규모 다국어 웹 코퍼스 한국어 | ~50B+ tokens |
| 웹크롤 | FineWeb2 Educational Korean | 교육적 품질 필터링 웹 데이터 | 234GB raw |
| 웹크롤 | Korean Web Collection | KORMo 웹 컬렉션 | 175GB raw |
| 웹크롤 | OSCAR Korean | 다국어 웹 코퍼스 한국어 | |
| 교육 | Korean Textbooks | 한국어 교과서 텍스트 | 45개 서브카테고리 |
| 교육 | FinePDFs Educational Korean | PDF 기반 교육 자료 | |
| 법률 | Korean Law | 한국 법률 텍스트 | 15GB |
| 뉴스 | Korean News Archive | 한국어 뉴스 아카이브 | |
| 공개코퍼스 | Korean Public Corpus | KORMo 공개 코퍼스 | 26GB |
| 코드 | Code Pretrain | 프로그래밍 코드 | |
| 학술 | Academic Pretrain | 학술 논문·리포트 | |
| 범용 | SlimPajama | RedPajama 경량 버전 | |

> 이 데이터는 Extended Pretrain (80-100B tokens) 단계에서 활용 예정이다.

#### 프리트레인 데이터 분야별 비율

```
┌─────────────────────────────────────────────────────────┐
│              3b_train.bin 토큰 구성 (~38.5B)              │
├─────────────────────────────────────────────────────────┤
│ ████████████████████░░░░░░░░░░  한국어 웹크롤    44.7%  │
│ ██████████░░░░░░░░░░░░░░░░░░░░  혼합 레거시      22.1%  │
│ ██████░░░░░░░░░░░░░░░░░░░░░░░░  교육 (EN)       14.7%  │
│ █████░░░░░░░░░░░░░░░░░░░░░░░░░  수학·과학       13.2%  │
│ ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░  백과사전 (KO)    5.3%  │
└─────────────────────────────────────────────────────────┘
```

### 7.3 SFT 데이터 — 2.44M 샘플 (현재 학습 중)

**24개 소스**에서 6.59M raw → 통합·중복 제거 → 품질 필터링 → **2,439,397 train + 49,801 val**

#### 주요 SFT 소스 (상위 12, 전체의 96%)

| # | 데이터셋 | 샘플 수 | 크기 | 도메인 |
|---|---------|---------|------|--------|
| 1 | reasoning_r1_1.4m | 1,400,000 | 14.77 GB | 추론 (CoT) |
| 2 | openhermes_2.5 | 1,001,551 | 1.82 GB | 영어 다목적 |
| 3 | AI-MO_NuminaMath-CoT | 859,494 | 2.51 GB | 수학 CoT |
| 4 | korean_instruction_mix | 515,911 | 1.39 GB | 한국어 혼합 |
| 5 | lemon-mint_smol-koreantalk | 460,281 | 5.23 GB | 한국어 대화 |
| 6 | open_korean_instructions | 375,159 | 0.73 GB | 한국어 지시 |
| 7 | magpie_reasoning_v2 | 249,922 | 3.99 GB | 추론 (영어) |
| 8 | magpie_reasoning_ko | 224,929 | 3.19 GB | 추론 (한국어) |
| 9 | ultrachat_200k | 207,865 | 1.34 GB | 대화 |
| 10 | kuotient_orca-math-ko | 193,789 | 0.61 GB | 수학 (한국어) |
| 11 | data/sft/train.jsonl (원본) | 161,848 | 0.27 GB | 원본 SFT |
| 12 | kullm_v2 | 152,630 | 0.42 GB | 한국어 지시 |

기타 12개 소스: DeepMath-103K, Evol-Instruct-Code-80k-ko, ShareGPT-74k-ko, evol-instruct-korean, alpaca-gpt4-korean, ko_wikidata_QA, Ko.WizardLM, KOR-OpenOrca-Platypus-v3, korean-writing-style-instruct, ko_lima, koalpaca_v1_1a, OpenAssistant_oasst1_ko

#### 데이터 처리 파이프라인

```
24개 소스 (6.59M raw)
    ↓ prepare_sft_combined.sh (포맷 통일, MD5 중복 제거, 98:2 split)
통합: 2,559,492 train + 52,234 val (7.95 GB)
    ↓ filter_sft_v2.py (5단계: EOS strip, QA marker 제거, 길이 50~20K, 4-gram 반복 >30% 제거)
최종: 2,439,397 train + 49,801 val (7.63 GB)  ← 제거율 4.69%
```

#### 도메인 비율

```
추론/CoT         38.0%  ████████████████████████
한국어 지시       22.5%  ██████████████
영어 다목적       16.0%  ██████████
수학             12.0%  ████████
대화/코드/기타    11.5%  ███████
```

### 7.4 선호도 데이터 (ORPO용) — 795K 쌍

총 **795,468 preference pairs** (7.9GB, `data/preference/combined_preference.jsonl`)

| HuggingFace ID | 크기 | 분야 | 포맷 |
|---------------|------|------|------|
| `nayohan/preference-collection-ko-full` | 4.9GB | 범용 선호도 평가 | instruction + response_A/B + preference |
| `heegyu/orca-math-korean-preference-cleaned` | 1.6GB | 수학 추론 | prompt + chosen + rejected |
| `kuotient/orca-math-korean-dpo-pairs` | 750MB | 수학 DPO | prompt + chosen + rejected |
| `maywell/ko_Ultrafeedback_binarized` | 394MB | 피드백 기반 정렬 | prompt + winning/losing response |
| `tellang/yeji-preference-ko-v1` | 171MB | 범용 선호도 | prompt + chosen + rejected |
| `jojo0217/korean_rlhf_dataset` | 137MB | RLHF 쌍 | prompt + chosen + rejected |
| `lemon-mint/korean-realqa-reasoning-v01-preference` | 58MB | QA 추론 | prompt + chosen + rejected |

**필터링 기준**: 최소 길이 20자, EOS 제거, 포맷 정규화 후 통합

> ORPO는 Phase 3에서 반복률이 5% 초과할 경우에만 실행한다. 3B 모델이 1B의 구조적 반복 문제를 스스로 해결한다면 ORPO 없이 배포할 수 있다.

### 7.5 데이터 파이프라인 요약

```
[HuggingFace / 웹 수집]
        │
        ▼
┌─── 원시 수집 ───────────────────────────────────────────┐
│  korean_extra/ (39개 디렉토리, 640GB+)                    │
│  sft_extra/ (27개 디렉토리, 1.08M 샘플)                   │
│  preference/ (7개 JSONL, 795K 쌍)                        │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─── 토큰화 (SentencePiece 64K) ──────────────────────────┐
│  tokenize_extra.py — 자동 포맷 감지 (Arrow/Parquet/JSONL) │
│  8 workers 병렬 처리, uint16 memmap (.bin) 출력           │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─── 최종 병합 ───────────────────────────────────────────┐
│  Pretrain: 3b_train.bin (77GB, ~38.5B tokens)           │
│  SFT:     sft_combined/train_filtered.jsonl (7.48GB, 2.44M 샘플) │
│  ORPO:    preference/combined_preference.jsonl (7.9GB)  │
└─────────────────────────────────────────────────────────┘
```

---

## 8. 학습 설정 및 최적화

### 현재 학습 설정 (`configs/korean_3b_fp8.yaml`)

```yaml
model:
  vocab_size: 64000
  d_model: 3072
  n_layers: 28
  n_heads: 24
  n_kv_heads: 8
  d_ffn: 8192
  max_seq_len: 2048
  rope_theta: 500000.0

training:
  batch_size: 5
  gradient_accumulation_steps: 8
  learning_rate: 1.5e-4
  min_lr: 1.5e-5
  warmup_steps: 2000
  max_steps: 57000
  weight_decay: 0.1
  grad_clip: 1.0
  optimizer: adamw
  scheduler: cosine

fp8:
  enabled: true
  recipe: "mxfp8"
  use_transformer_engine: true

distributed:
  strategy: ddp
  gradient_as_bucket_view: true
  find_unused_parameters: false

nccl:
  timeout_seconds: 7200
  nvls_enabled: true
```

유효 배치 크기 = `batch_size(5) × grad_accum(8) × num_gpus(8)` = **320**

LR 스케줄: warmup 2000 스텝 → cosine decay → min_lr=1.5e-5 (max_lr의 10%)

### Phase 0에서 배운 최적화 교훈

#### GQA FlashAttention Native

가장 큰 VRAM 절감을 가져온 최적화. 핵심은 FlashAttention이 GQA를 native로 지원한다는 점이다. KV head를 expand하여 MHA처럼 처리하면 메모리 복사가 발생하지만, native path를 쓰면 내부에서 직접 처리한다.

```python
# Before (비효율적): KV expand → MHA처럼 처리
k = k.repeat_interleave(n_heads // n_kv_heads, dim=1)
v = v.repeat_interleave(n_heads // n_kv_heads, dim=1)
out = flash_attn_func(q, k, v)

# After (native GQA): flash_attn이 내부에서 GQA 처리
out = flash_attn_func(q, k, v)  # q: [B, S, H, D], k/v: [B, S, Hkv, D]
# VRAM 60.4GB → 48.3GB (-20%)
```

#### DDP 최적화

```python
# gradient_as_bucket_view=True: gradient tensor를 bucket 메모리의 view로 직접 매핑
# → 불필요한 메모리 복사 제거, GPU-CPU 동기화 오버헤드 -87.5%
model = torch.nn.parallel.DistributedDataParallel(
    model,
    device_ids=[local_rank],
    gradient_as_bucket_view=True,
    find_unused_parameters=False,  # 모든 파라미터가 사용됨
)
```

**주의**: `static_graph=True`는 사용하지 않는다. TransformerEngine의 `te.Linear`가 일부 케이스에서 dynamic graph를 요구하는데, static_graph를 켜면 런타임 에러가 발생한다.

#### NCCL NVLS

```bash
export NCCL_ALGO=NVLSTree    # NVLink SHARP (NVLS) 활성화
export NCCL_PROTO=Simple
export NCCL_P2P_DISABLE=0
export NCCL_TIMEOUT=7200     # 긴 backward에 대비한 타임아웃 여유
```

NVSwitch가 All-to-All single hop을 지원하므로 Ring topology보다 NVLSTree가 효율적이다.

#### SIGHUP 3중 방어

장시간 학습에서 세션 연결 끊김(SIGHUP)은 치명적이다. 3중 보호를 구축했다:

```bash
# 1중: nohup + setsid (새 세션 그룹)
nohup setsid torchrun --nproc_per_node=8 train/pretrain.py ... &

# 2중: Python signal handler (Python 레벨 SIGHUP 무시)
import signal
signal.signal(signal.SIGHUP, signal.SIG_IGN)

# 3중: emergency checkpoint (SIGTERM에도 체크포인트 저장)
def emergency_save(signum, frame):
    save_checkpoint(model, optimizer, step, "emergency")
    sys.exit(0)
signal.signal(signal.SIGTERM, emergency_save)
```

#### torch.compile — 테스트 결과: 효과 없음

`torch.compile`을 적용해 speedup을 기대했지만 실측 결과 **1.00x (효과 없음)**이었다. 두 가지 이유:

1. TransformerEngine의 kernel이 opaque하여 graph break가 발생한다. `torch.compile`은 Python 연산 그래프를 최적화하는데, TE kernel은 그 그래프 밖에 있다.
2. `/tmp` 디렉토리에 `noexec` 마운트 플래그가 있어 컴파일된 kernel을 캐시하지 못한다.

**교훈**: "일단 써보자"보다 "왜 효과가 있는지 먼저 이해하자"가 중요하다.

### 모니터링 시스템

```
텔레그램 알림 시스템
├── B200Bot (token 설정됨)
├── training_watchdog.sh → 10분 간격 cron
│   └── loss 이상, 프로세스 종료 감지 → 즉시 알림
└── hourly_status.sh → 1시간 간격 cron
    └── step, loss, 속도, VRAM, eta → 정기 리포트
```

```python
# curl이 차단돼 있어 urllib 사용
import urllib.request, json

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    data = json.dumps({"chat_id": CHAT_ID, "text": message}).encode()
    req = urllib.request.Request(url, data=data,
                                  headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req)
```

---

## 9. 실험 결과 — 1B 베이스라인

1B 모델의 실험 결과를 정직하게 기록한다. 성공과 실패 모두.

### 프리트레인 결과

| 지표 | 값 |
|------|-----|
| 최종 Loss | 1.904 |
| PPL (C4 Korean) | 5.67 |
| 학습 스텝 | 34,000 |
| 학습 시간 | ~2일 |

### SFT v1 결과 — 실패

| 지표 | 값 |
|------|-----|
| val_loss | 0.0 (비정상) |
| 원인 | label off-by-one 버그 (데이터 누수) |
| 결론 | 전면 폐기 |

### SFT v2 결과 — 부분 성공

| 지표 | 값 |
|------|-----|
| val_loss | 2.2062 |
| 반복률 | 18% (rep_penalty=1.1 적용) |
| kobest_copa | 0.646 |
| 결론 | 기능하지만 구조적 한계 존재 |

### 3B 기대 목표치 (스케일링 법칙 기반 예측)

| 벤치마크 | 1B 현재 | 3B 목표 |
|----------|---------|---------|
| kobest_copa | 0.646 | >0.72 |
| kobest_hellaswag | ~0.42 | >0.52 |
| 반복률 | 18% | <5% |
| PPL (C4 Korean) | 5.67 | <4.5 |

1B에서 3B로의 스케일업은 단순히 파라미터를 늘리는 것이 아니다. 모델이 더 긴 맥락을 기억하고, 더 다양한 패턴을 학습할 수 있어야 반복률이 구조적으로 낮아진다. 3B 목표치는 Chinchilla 스케일링 곡선과 유사 규모 모델들의 벤치마크를 참고한 예측값이다.

---

## 10. 실험 결과 — 3B Base 종합 평가 (v2)

3B 사전학습 완료 후 checkpoint-0057000 기준으로 수행한 종합 평가.
v2 재평가는 8-GPU 병렬 파이프라인으로 13+ 벤치마크, 0/5-shot 비교, calibration, 참고모델 비교를 포함한다.
총 소요 시간 256.6초.

> **v1 → v2 변경점**: v1(초기 평가)에서는 PPL 3개 데이터셋 + belebele/MMLU 2개 벤치마크만 측정했다. v2는 PPL 19개 데이터셋, KoBEST 5개, HAE-RAE 전체, MMLU-KO 6카테고리, MMLU-EN 61과목, 영어 5대 벤치마크, Calibration, 0/5-shot 비교, 12조합 파라미터 그리드 서치를 포함한다.

### 10.1 학습 커브

| Step | Loss | LR | 비고 |
|------|------|----|------|
| 10 | 11.657 | 1.50e-06 | 초기 (warmup 시작) |
| 500 | 5.047 | 7.50e-05 | warmup 진행 |
| 2,000 | 2.851 | 3.00e-04 | warmup 완료, peak LR |
| 10,000 | 2.057 | 2.86e-04 | 안정 하강 |
| 30,000 | 1.789 | 1.61e-04 | 중반, epoch 1 진입 |
| 57,000 | 1.466 | 3.00e-05 | 최종 (cosine min) |

> 처리 속도는 전 구간 36~38K tok/s로 안정. 총 학습 시간 약 63시간.

### Base Model 백업

| 항목 | 값 |
|------|-----|
| 원본 체크포인트 | `checkpoints/korean_3b_fp8_run1/checkpoint-0057000/` (34GB) |
| 백업 | `checkpoints/korean_3b_fp8_run1/checkpoint-0057000_BASE_BACKUP/` |
| MD5 검증 | `4f493d7bcc843727d32453bb3a4e6b7d` (일치 확인) |
| HF 변환 | `eval/outputs/hf_3b_base/` (11GB safetensors) |

### 10.2 PPL (Perplexity) — 19개 데이터셋

**주요 PPL (3b_val 통합): 5.2263** (초기 v1 평가: 5.709)

| 데이터셋 | PPL | Bits/Token | 평가 토큰 | 소요 시간 |
|---------|-----|-----------|---------|---------|
| korean_namuwiki | 25.88 | 4.694 | 6.5M | 63.7s |
| cc100_ko | 21.78 | 4.445 | 13.6M | 133.2s |
| namuwiki_2023b | 18.92 | 4.242 | 7.7M | 75.1s |
| val | 18.30 | 4.194 | 9.1M | 89.4s |
| korean_wiki | 11.84 | 3.565 | 1.6M | 15.5s |
| wikipedia_ko | 10.71 | 3.420 | 1.8M | 17.4s |
| korean | 7.02 | 2.811 | 53.5M | 521.6s |
| open_web_math | 6.93 | 2.792 | 15.7M | 153.5s |
| **korean_c4** | **5.72** | **2.515** | **45.4M** | **443.1s** |
| **3b (통합)** | **5.23** | **2.386** | **226.9M** | **2227.3s** |
| cosmo_web_v2 | 4.17 | 2.059 | 8.6M | 84.6s |
| cosmo_stories | 3.96 | 1.984 | 18.9M | 185.2s |
| cosmo_openstax | 3.87 | 1.951 | 0.7M | 7.2s |
| cosmo_stanford | 3.36 | 1.750 | 6.6M | 65.3s |
| cosmo_wikihow | 3.31 | 1.727 | 1.2M | 11.8s |
| cosmo_auto_math_text | 3.15 | 1.655 | 7.9M | 77.3s |
| cosmo_khanacademy | 2.93 | 1.552 | 0.1M | 1.5s |
| mathpile | 2.72 | 1.446 | 7.1M | 69.9s |
| hplt_ko | 2.40 | 1.265 | 48.5M | 475.9s |

> **해석**: in-distribution(학습에 포함된) 데이터(hplt_ko: 2.40, mathpile: 2.72)가 낮고, OOD(학습 비중 낮은) 데이터(cc100_ko: 21.78, namuwiki: 25.88)가 높은 것은 예상된 패턴. korean_c4 5.72는 v1의 5.717과 일치하여 평가 재현성을 확인.

### 10.3 한국어 벤치마크

#### KoBEST (0-shot) — 평균 43.69%

| 태스크 | Accuracy | F1 |
|--------|----------|-----|
| kobest_boolq | 50.28% | 0.3457 |
| kobest_copa | 49.30% | 0.4921 |
| kobest_hellaswag | 21.60% | 0.2153 |
| kobest_sentineg | 48.61% | 0.4737 |
| kobest_wic | 48.65% | 0.3286 |
| **평균** | **43.69%** | |

#### HAE-RAE (0-shot) — 전체 19.71%

| 서브태스크 | Accuracy |
|-----------|----------|
| haerae_general_knowledge | 21.59% |
| haerae_history | 23.40% |
| haerae_loan_word | 21.30% |
| haerae_rare_word | 18.77% |
| haerae_standard_nomenclature | 13.73% |
| **전체** | **19.71%** |

#### MMLU-KO (0-shot) — 6카테고리 평균 22.75%

| 카테고리 | Accuracy |
|----------|----------|
| medical | 30.56% |
| humanities | 24.51% |
| business | 24.14% |
| social_sciences | 20.59% |
| other | 19.64% |
| stem | 19.57% |
| **평균** | **22.75%** |

> Base model은 instruction-following 없이 4지선다 형식 벤치마크를 풀도록 최적화되지 않음. KoBEST boolq/copa/sentineg/wic는 ~50% 수준으로 2지/4지선다 랜덤 기준 부근이며, SFT 후 향상 기대.

### 10.4 영어 벤치마크

#### 주요 벤치마크 (0-shot)

| 태스크 | Accuracy | Acc (norm) |
|--------|----------|-----------|
| hellaswag | 26.00% | 26.15% |
| arc_easy | 25.63% | 26.64% |
| arc_challenge | 21.67% | 27.90% |
| winogrande | 50.59% | — |
| piqa | 52.50% | 48.31% |

> winogrande(50.59%)와 piqa(52.50%)는 2지선다로 랜덤 기준 50%에 근접. hellaswag/arc는 4지선다로 랜덤 기준 25%.

#### MMLU-EN (0-shot) — 61과목 평균 25.81%

**상위 10개 과목**:

| 과목 | Accuracy |
|------|----------|
| college_physics | 37.25% |
| college_computer_science | 34.00% |
| high_school_statistics | 33.80% |
| us_foreign_policy | 32.00% |
| security_studies | 31.43% |
| world_religions | 30.99% |
| professional_medicine | 30.88% |
| high_school_government_and_politics | 30.57% |
| jurisprudence | 30.56% |
| human_sexuality | 30.53% |

**하위 5개 과목**:

| 과목 | Accuracy |
|------|----------|
| human_aging | 19.73% |
| college_biology | 19.44% |
| anatomy | 17.04% |
| global_facts | 17.00% |
| abstract_algebra | 15.00% |

### 10.5 Calibration

| 메트릭 | 값 |
|--------|-----|
| Top-1 Accuracy | 68.75% |
| Top-5 Accuracy | 81.64% |
| Top-10 Accuracy | 85.93% |
| Mean Correct Prob | 0.6152 |
| Mean Entropy | 1.5682 |

**Token NLL 분포**:

| 통계 | 값 |
|------|-----|
| 평균 NLL | 1.5561 |
| 표준편차 | 2.4926 |
| 중앙값 | 0.1221 |
| p95 | 7.0312 |
| p99 | 10.3125 |
| NLL > 5 비율 | 10.86% |
| NLL > 10 비율 | 1.18% |

> Top-1 68.75%는 모델이 가장 확신하는 예측이 ~69% 확률로 정확하다는 의미. 중앙값 NLL 0.12 (≈ e^0.12 = 1.13 PPL)로 대부분의 토큰을 매우 높은 확신도로 예측하고, 소수의 고난이도 토큰이 평균 NLL을 끌어올리는 전형적인 분포.

### 10.6 0-shot vs 5-shot 비교

18개 한국어 태스크에서 0-shot과 5-shot 성능을 비교했다.

| 태스크 | 0-shot | 5-shot | 변화 |
|--------|--------|--------|------|
| global_mmlu_ko | 22.75% | 26.75% | **+4.00pp** |
| global_mmlu_ko_business | 24.14% | 31.03% | **+6.90pp** |
| global_mmlu_ko_humanities | 24.51% | 28.43% | +3.92pp |
| global_mmlu_ko_medical | 30.56% | 36.11% | **+5.56pp** |
| global_mmlu_ko_other | 19.64% | 23.21% | +3.57pp |
| global_mmlu_ko_social_sciences | 20.59% | 23.53% | +2.94pp |
| global_mmlu_ko_stem | 19.57% | 21.74% | +2.17pp |
| haerae | 19.71% | 20.26% | +0.55pp |
| haerae_general_knowledge | 21.59% | 22.73% | +1.14pp |
| haerae_history | 23.40% | 14.89% | -8.51pp |
| haerae_loan_word | 21.30% | 24.26% | +2.96pp |
| haerae_rare_word | 18.77% | 18.02% | -0.74pp |
| haerae_standard_nomenclature | 13.73% | 25.49% | **+11.76pp** |
| kobest_boolq | 50.28% | 50.21% | -0.07pp |
| kobest_copa | 49.30% | 46.80% | -2.50pp |
| kobest_hellaswag | 21.60% | 20.80% | -0.80pp |
| kobest_sentineg | 48.61% | 47.86% | -0.76pp |
| kobest_wic | 48.65% | 48.97% | +0.32pp |

**평균 변화: +1.80pp** | 개선: 12 | 하락: 6

> MMLU-KO는 5-shot에서 일관되게 개선(+2~7pp)되어 in-context learning 능력이 작동함을 확인. KoBEST는 거의 변동 없거나 소폭 하락—이미 0-shot에서 패턴 매칭을 잘하고 있어 few-shot 예시가 오히려 방해가 되는 패턴. haerae_standard_nomenclature의 +11.76pp는 이 태스크의 특수한 포맷을 few-shot에서 학습한 결과.

### 10.7 참고 모델 비교

| 모델 | 파라미터 | MMLU-KO | MMLU-EN | KoBEST 평균 | PPL |
|------|---------|---------|---------|------------|-----|
| **FRANKENSTALLM 3B** | **3B** | **22.75%** | **25.81%** | **43.69%** | **5.2263** |
| Llama-3.2-3B | 3B | ~42% | ~58% | ~55% | — |
| Qwen2.5-3B | 3B | ~48% | ~65% | ~60% | — |
| EXAONE-3.5-2.4B | 2.4B | ~35% | ~50% | ~50% | — |

> 참고 모델들은 수조 토큰 규모의 학습 데이터와 수천 GPU-hour를 투입한 결과. FRANKENSTALLM 3B는 41.12B 토큰(Chinchilla 최적의 ~68%), 63시간, 8 GPU로 학습한 점을 감안해야 한다. SFT + 확장 프리트레인(80-100B 토큰) 이후 격차 축소 예상.

### 10.8 생성 품질 및 파라미터 그리드 서치

#### 반복률 요약

| 설정 | 3-gram 반복률 | 4-gram 반복률 |
|------|--------------|--------------|
| greedy (temp=0.0) | 60.99% | 57.02% |
| temp=0.5 | 60.12% | 58.68% |
| temp=0.7 | 47.69% | 43.40% |
| temp=1.0 | 3.58% | 2.81% |

> 초기 v1 평가의 greedy 71.1% 반복률은 `no_repeat_ngram_size=3` 적용 기준이었다. v2에서는 미적용 기준(raw)으로 통일하여 60.99%를 기록.

#### 12조합 파라미터 그리드 서치 결과

| 설정 | Temp | Rep Pen | 3-gram | 4-gram | 비고 |
|------|------|---------|--------|--------|------|
| **t0.7_rep1.3** | **0.70** | **1.30** | **0.00%** | **0.00%** | **최적** |
| t0.9_rep1.2 | 0.90 | 1.20 | 0.00% | 0.00% | 차선 |
| t0.7_rep1.2 | 0.70 | 1.20 | 0.88% | 0.00% | |
| t0.9_rep1.1 | 0.90 | 1.10 | 0.94% | 0.13% | |
| t1.0_rep1.1 | 1.00 | 1.10 | 1.21% | 0.48% | |
| t0.5_rep1.1 | 0.50 | 1.10 | 1.92% | 1.19% | |
| t1.0 | 1.00 | 1.00 | 3.58% | 2.81% | |
| t0.9 | 0.90 | 1.00 | 8.39% | 4.64% | |
| t0.7_rep1.1 | 0.70 | 1.10 | 8.51% | 5.51% | |
| t0.7 | 0.70 | 1.00 | 47.69% | 43.40% | |
| t0.5 | 0.50 | 1.00 | 60.12% | 58.68% | |
| greedy | 0.00 | 1.00 | 60.99% | 57.02% | |

#### 권장 추론 파라미터 (base 실험용)

```python
# v2 그리드 서치 최적값
temp=0.7, repetition_penalty=1.3
# 또는 (더 다양한 생성)
temp=0.9, repetition_penalty=1.2
```

> 초기 v1 권장값(`temp=0.9, top_p=0.9, no_repeat_ngram=3, repetition_penalty=1.1`)에서 `repetition_penalty=1.3`으로 상향 조정. `no_repeat_ngram_size`는 그리드 서치에서 `repetition_penalty`만으로 충분히 반복 제거가 가능함을 확인하여 불필요.

### 10.9 평가 파이프라인

v2 재평가는 모듈화된 8-GPU 병렬 파이프라인(`eval/reeval_pipeline.py`)으로 수행되었다.

#### 아키텍처

```
reeval_pipeline.py
├── 모델 1회 로드 (GPU 0에 HF 모델)
├── Phase 1: PPL 평가 (19개 데이터셋, 순차)
├── Phase 2: Calibration + Token NLL
├── Phase 3: 생성 품질 + 파라미터 그리드 서치 (12조합)
├── Phase 4: lm-evaluation-harness (0-shot, 8-GPU 병렬)
├── Phase 5: lm-evaluation-harness (5-shot, 8-GPU 병렬)
└── Phase 6: 리포트 자동 생성 (5개 개별 + 1개 종합)
```

#### Pipeline Mode

모델을 1회 로드하여 0-shot과 5-shot을 연속 실행한다. 기존 방식(별도 프로세스 2회)에 비해 모델 로딩 시간을 절반으로 줄인다.

#### GPU별 태스크 분배

| GPU | 0-shot 태스크 | 5-shot 태스크 |
|-----|--------------|--------------|
| 0 | kobest_boolq, kobest_copa, kobest_hellaswag | 동일 |
| 1 | kobest_sentineg, kobest_wic | 동일 |
| 2 | haerae (전체 + 5개 서브) | 동일 |
| 3 | global_mmlu_ko (6카테고리) | 동일 |
| 4 | hellaswag, arc_easy | 동일 |
| 5 | arc_challenge, winogrande | 동일 |
| 6 | piqa, global_mmlu_en (61과목) | 동일 |
| 7 | (예비 — PPL/calibration 전담) | — |

NUMA affinity 적용: GPU 0-3은 NUMA node 0 (cores 0-35), GPU 4-7은 NUMA node 1 (cores 36-71).

**총 소요 시간: 256.6초** (모델 로드 포함)

### SFT 진행 판단

**결론: SFT 진행** — loss 1.466 건강한 완료 시그널, 구조 문제 없음. → **Phase 2 SFT 시작 (2026-03-05)**

상세 보고서:
- v2 종합: `eval/outputs/3b_reeval_20260305_1451/reports/` (5개 개별 리포트 + 종합)
- v1 레거시: `reports/2026-03-05_3B_BASE_EVALUATION_REPORT.md`

---

## 11. 실험 결과 — 3B SFT 종합 평가

Phase 2 SFT가 early stopping으로 완료된 후 수행한 6차원 종합 평가.

### 11.1 SFT 학습 결과

| 항목 | 값 |
|------|-----|
| 최종 Step | 25,500 / 33,000 (77.3%, early stopping) |
| Best val_loss | **1.8851** (step 23,000) |
| 학습 시간 | ~15시간 41분 |
| 데이터 | 24개 소스 → 2,439,397 samples (7.48 GB) |
| 설정 | LR=1e-5, eff_batch=64, NEFTune alpha=5.0 |

**Val Loss 추이**:
```
Step     500: 2.0732 (warmup 완료)
Step   2,000: 1.9558 (급속 하강)
Step   5,000: 1.9107 (안정 수렴)
Step  10,000: 1.8917 (미세 감소)
Step  15,000: 1.8864 (plateau 진입)
Step  20,000: 1.8853 (변동 < 0.001)
Step  23,000: 1.8851 ← BEST (early stopping 기준점)
Step  25,500: Early Stop (patience 5/5 소진)
```

### 11.2 6차원 평가 요약

| # | 차원 | 결과 | 핵심 수치 |
|---|------|------|-----------|
| 1 | Perplexity (지식 보존) | **PASS** | 최대 forgetting 0.9%, 19개 데이터셋 전체 PASS |
| 2 | 생성 품질 | **FAIL** | Greedy 반복률 72.97% (목표 <5%), EOS 60% (목표 >90%) |
| 3 | 한국어 벤치마크 | **FAIL** | KoBEST 평균 43.26% (목표 >55%) |
| 4 | 영어 벤치마크 | **PASS** | hellaswag 26.1%, winogrande 50.8%, piqa 52.6% (전 항목 하한 초과) |
| 5 | Calibration | **PASS** | Top-1 68.59%, Top-5 81.55%, Entropy 1.54 |
| 6 | SFT Chat 능력 | **PASS** | EOS 종료율 0%→60%, Chat template 응답 |

### 11.3 Base vs SFT 비교

| 지표 | Base | SFT | 변화 | 판정 |
|------|------|-----|------|------|
| PPL (통합) | 5.2263 | 5.2529 | +0.5% forgetting | PASS |
| Greedy 3-gram 반복률 | 60.99% | 72.97% | +12pp (악화) | FAIL |
| EOS 종료율 | 0% | 60% | +60pp (대폭 개선) | 부분 PASS |
| KoBEST 평균 | 43.69% | 43.26% | -0.4pp | FAIL |
| MMLU-KO | 22.75% | 26.00% | +3.2pp | 부분 개선 |
| 영어 벤치마크 | — | — | ±0.3pp 이내 | PASS (유지) |
| Calibration Top-1 | 68.75% | 68.59% | -0.2pp | PASS (유지) |

**Repetition 파라미터 검색** (희망적):

| 설정 | 반복률 | EOS Rate |
|------|--------|----------|
| t0.7_rep1.2 | **0.00%** | **100%** |
| t1.0_rep1.1 | **0.00%** | **100%** |
| greedy (raw) | 72.97% | 60% |

> rep_penalty 1.1~1.3 적용 시 반복률 0% 달성 → 모델이 반복하지 않는 능력 자체는 보유. ORPO로 내재화 가능.

### 11.4 코드 개선 사항

이번 Phase에서 수행한 주요 코드 변경:

| 파일 | 변경 | 줄 수 | 목적 |
|------|------|-------|------|
| `train/sft.py` | MixingDataLoader, DDP rank 0 토크나이징 | +238 | SFT+pretrain 인터리빙, 메모리 8배 절감 |
| `train/trainer.py` | DDP early stop broadcast | +17 | DDP hang 방지, patience 5→10 |
| `train/orpo.py` | YAML config, 3B 기본값 | +30 | ORPO 실행 준비 |
| `eval/report_generator.py` | SFT 비교 보고서 자동 생성 | +831 | 평가 자동화 |
| `eval/sft_eval_pipeline.py` | 6차원 평가 파이프라인 | 신규 | SFT 종합 평가 |
| `eval/tasks/generation_task.py` | Chat template, diversity metrics | +75 | SFT 평가 지원 |

### 11.5 ORPO 진행 판정

**판정: Phase 3 ORPO 진행**

| 근거 | 상세 |
|------|------|
| 지식 보존 양호 | forgetting 0.9% — SFT가 base 지식을 파괴하지 않음 |
| 반복 미해결 | greedy 72.97% — 선호도 정렬이 직접적 해결 경로 |
| 희망적 신호 | rep_penalty 적용 시 0% → ORPO가 내재화 가능 |
| 데이터 준비 완료 | 795,468 preference pairs (7.9 GB) |
| 코드/설정 완비 | `train/orpo.py` + `configs/korean_3b_orpo.yaml` |

**ORPO 후 판정 기준**:
- 반복률 < 5% AND KoBEST > 50% → GGUF + Ollama 배포
- 반복률 5~15% → 하이퍼파라미터 조정 후 재시도
- 반복률 > 15% → SFT v2 (lr=5e-5, data mixing) 후 재도전

상세: `reports/2026-03-06_3B_SFT_COMPLETION_AND_EVAL_SUMMARY.md`

---

## 12. Phase 3 — ORPO (선호도 정렬)

### 12.1 ORPO 선택 배경

SFT 6차원 평가에서 greedy 반복률 72.97%, EOS 종료율 0%라는 치명적 문제가 발견됐다. SFT는 "좋은 응답만 모방"하는 학습이므로, "나쁜 응답을 억제"하는 신호가 없다. 반복 문제 해결에는 preference optimization이 필수적이다.

**ORPO vs DPO**:
| 항목 | ORPO | DPO |
|------|------|-----|
| Reference model | 불필요 | 필요 (VRAM 2배) |
| 구현 복잡도 | 낮음 | 중간 |
| 메모리 효율 | 높음 (3B 1개만 로드) | 낮음 (3B 2개 로드) |
| 학습 안정성 | 중간 | 높음 |

ORPO를 1차 선택, DPO를 Plan B로 설정했다.

### 12.2 데이터

- **원본**: 683,181 preference pairs (7개 소스 통합)
- **필터 후**: ~630,000 pairs (NaN 방지 필터 적용)
- **Eval split**: 5% (~31,500 pairs, seed=42)
- **Effective batch**: 4 × 8 GPU × 4 accum = 128

### 12.3 HP Sweep 설계 (6-Config)

3개 축(beta, LR, max_length)을 중심축 고정 방식으로 6개 조합 선정:

| Run | Name | Beta | LR | Max Length | 목적 |
|-----|------|------|----|-----------|------|
| 1 | baseline_b015 | 0.15 | 8e-6 | 1536 | 약한 beta 베이스라인 |
| 2 | baseline_b025 | 0.25 | 8e-6 | 1536 | 중간 beta 베이스라인 |
| 3 | strong_b035 | 0.35 | 8e-6 | 1536 | 강한 beta — 적극적 반복 억제 |
| 4 | fast_lr12e6 | 0.25 | 1.2e-5 | 1536 | 높은 LR — 빠른 수렴 |
| 5 | conserv_lr5e6 | 0.25 | 5e-6 | 1536 | 보수적 LR — 안정성 |
| 6 | short_1024 | 0.25 | 8e-6 | 1024 | 짧은 max_length — VRAM 절약 |

각 200 steps, eval_steps=100, 8×B200 DDP.

### 12.4 시도 이력 — 5번의 실패

| # | 문제 | 원인 | 수정 |
|---|------|------|------|
| 1 | NCCL Timeout | 토크나이징 30분 > timeout 1800s | ddp_timeout=7200, num_proc=64 |
| 2 | Config 충돌 | save_steps ≠ eval_steps 배수 | --no_load_best --save_steps 200 |
| 3 | 포트 충돌 + QKV 누락 | 좀비 프로세스 + fused QKV 미분리 | pkill + QKV split 로직 |
| 4 | TRL NaN 버그 | tokenize_row 양쪽 response 동시 잘림 | 3중 패치 (clamp, truncation) |
| 5 | Tokenizer 호환 | zip(strict=True) + 한국어 merge ops | TRL 소스 8건 패치 |

가장 심각했던 것은 TRL NaN 버그로, 0 response tokens → log(0) = -inf → NaN 전파 체인을 일으켰다. 상세: `reports/2026-03-08_ORPO_TRAINING_JOURNEY.md`

### 12.5 스윕 최종 결과

| Run | Name | Beta | LR | MaxLen | Train Loss | Eval Loss | Margin | Status |
|-----|------|------|----|--------|-----------|-----------|--------|--------|
| 1 | baseline_b015 | 0.15 | 8e-6 | 1536 | 1.811 | 1.827 | 0.004 | ✅ |
| 2 | baseline_b025 | 0.25 | 8e-6 | 1536 | 1.890 | 1.906 | 0.009 | ✅ |
| 3 | strong_b035 | 0.35 | 8e-6 | 1536 | 2.055 | 1.985 | 0.007 | ✅ |
| **4** | **fast_lr12e6** | **0.25** | **1.2e-5** | **1536** | **1.917** | **1.862** | **0.009** | **🏆 Best** |
| 5 | conserv_lr5e6 | 0.25 | 5e-6 | 1536 | 1.833 | 1.910 | 0.004 | ✅ |
| 6 | short_1024 | 0.25 | 8e-6 | 1024 | 1.664 | 1.695 | 0.007 | ✅ |

**Best config: Run 4** (eval_loss 1.862 최저, margin 0.009 최고, 빠른 수렴).

### 12.6 Throughput 벤치마크 → 본 학습 설정

본 학습 전 batch/grad_accum 조합의 throughput을 측정하여 최적 설정을 결정:

| batch_size | grad_accum | eff_batch | Throughput | 비고 |
|-----------|-----------|----------|-----------|------|
| **4** | **4** | **128** | **80.63 samples/s** | **선정** |
| 2 | 8 | 128 | 73.14 samples/s | 기존 설정 |
| 8 | 2 | 128 | OOM | |

### 12.7 ORPO 본 학습 (완료)

| 파라미터 | 값 |
|---------|-----|
| Beta / LR | 0.25 / 1.2e-5 (Sweep Run 4) |
| Batch / Accum / Eff | 4 / 4 / 128 (벤치마크 최적) |
| Max length | 1536 |
| Epochs | 2 (~9,840 steps) |
| GPU VRAM | ~52GB / 183GB (28%) |
| 속도 | ~1.75 s/step |
| 예상 시간 | ~4.8시간 |

**학습 지표 추이 (step ~1,660 기준)**:

| Step | Eval Loss | Pref Accuracy | Reward Margin | NLL Loss |
|-----:|----------:|--------------:|--------------:|---------:|
| ~1,000 | 1.791 | 66.8% | 0.107 | 1.647 |
| ~2,000 | 1.713 | 70.1% | 0.293 | 1.591 |
| ~3,000 | 1.681 | 71.9% | 0.372 | 1.567 |

- Train loss: 2.34 → **1.68** (-0.66)
- rewards/accuracies: 0.43 → **0.74** (chosen/rejected 구분 능력 급상승)
- rewards/margins: -0.005 → **0.387** (preference signal 학습 확인)
- 속도 ~1.76 s/step, GPU 92~100% utilization, 안정적 진행 중

**학습 완료 후 자동 평가**: `scripts/orpo_eval_watchdog.sh` 가 학습 프로세스를 감시하며, 완료 시 자동으로 10차원 종합 평가 파이프라인 실행

### 12.8 ORPO 종합 평가 파이프라인

SFT v2 평가의 6차원에 ORPO 고유 4차원을 추가한 **10차원 종합 평가**.
학습 완료 시 `eval/orpo_eval_pipeline.py`가 자동 실행되어 Base vs SFT vs ORPO 3-way 비교 보고서를 생성한다.

**평가 구조**:

| Phase | 내용 | GPU | 예상 시간 |
|-------|------|-----|----------|
| Pre-phase | train.log에서 학습 곡선 추출 | - | ~1초 |
| Phase 1 | 내부 평가 (PPL 19셋, Calibration, Generation, Repetition Grid) | 8 GPU 병렬 | ~30분 |
| Phase 2 | 벤치마크 (KoBEST, HAE-RAE, MMLU-KO/EN, hellaswag, arc, piqa) | 8 GPU 병렬 | ~1시간 |
| Phase 3 | 3-way 비교 보고서 자동 생성 | - | ~10초 |

**10차원 평가 항목**:

| # | 차원 | 기준 | SFT v2 결과 | ORPO 목표 |
|---|------|------|------------|----------|
| 1 | 지식 보존 (PPL) | forgetting < 15% | 0.9% | < 5% |
| 2 | 생성 품질 | greedy 반복률 < 5%, EOS > 90% | **72.97% / 60%** | **< 5% / > 90%** |
| 3 | 한국어 벤치마크 | KoBEST 평균 > 55% | 43.26% | ≥ 43% |
| 4 | 영어 벤치마크 | 하한 초과 | PASS | 유지 |
| 5 | Calibration | Top-1 ≥ 65% | 68.59% | ≥ 65% |
| 6 | Chat 능력 | EOS 종료율 | 60% | > 90% |
| 7 | Preference Accuracy | > 65% | — | > 65% |
| 8 | Reward Margins | > 0.1 | — | > 0.1 |
| 9 | 반복 파라미터 민감도 | rep_penalty=1.0에서도 < 5% | — | PASS |
| 10 | SFT→ORPO 개선 | 반복률↓ + EOS↑ | — | PASS |

**핵심 파일**:
- `eval/orpo_eval_pipeline.py` — ORPO 평가 오케스트레이터
- `eval/report_generator.py` — 3-way 비교 보고서 생성기 (`generate_three_way_report()`)
- `scripts/orpo_eval_watchdog.sh` — 학습 완료 감지 + 자동 평가 실행

**배포 기준**: greedy 반복률 < 5% AND EOS > 90% AND forgetting < 5% AND KoBEST ≥ 43% → **DEPLOY**

---

## 13. 실행 방법

### 사전 요구사항

```bash
# PyTorch는 재설치 금지 (NVIDIA 커스텀 빌드)
# 아래 패키지만 추가 설치
pip install transformers accelerate peft trl deepspeed \
            bitsandbytes sentencepiece wandb
```

### 3B 프리트레인

```bash
# NCCL 환경변수와 함께 8-GPU 학습 실행
bash scripts/launch_3b_pretrain.sh

# 수동 실행 (직접 제어)
torchrun --nproc_per_node=8 \
  --master_port=29500 \
  train/pretrain.py \
  --config configs/korean_3b_fp8.yaml
```

### SFT

```bash
bash scripts/launch_3b_sft.sh

# 또는 직접 실행
torchrun --nproc_per_node=8 \
  train/sft.py \
  --config configs/korean_3b_sft.yaml \
  --pretrain_ckpt checkpoints/3b_pretrain_best.pt
```

### ORPO (선호도 정렬)

```bash
# ORPO 학습
bash scripts/launch_3b_orpo.sh

# 학습 완료 후 자동 평가 (watchdog)
nohup bash scripts/orpo_eval_watchdog.sh \
  > checkpoints/korean_3b_orpo_v1/watchdog.log 2>&1 &
```

### 평가

```bash
# Base 모델 전체 평가 (8 GPU 병렬)
python eval/full_eval_pipeline.py

# SFT 모델 평가 (Base vs SFT 2-way 비교)
python eval/sft_eval_pipeline.py --skip-phase0 \
  --hf-model-path eval/outputs/hf_3b_sft_best

# ORPO 모델 평가 (Base vs SFT vs ORPO 3-way 비교)
python eval/orpo_eval_pipeline.py           # 자동으로 최신 checkpoint 감지
python eval/orpo_eval_pipeline.py --dry-run  # 실행 계획만 확인

# 빠른 평가 (kobest_copa + PPL)
bash scripts/run_eval_quick.sh

# 생성 파라미터 탐색
python eval/test_generation_params.py \
  --checkpoint checkpoints/3b_best.pt
```

### 배포

```bash
# Step 1: GGUF 변환 (llama.cpp 포맷)
bash scripts/convert_3b_gguf.sh

# Step 2: Ollama 모델 등록 및 서빙
bash scripts/deploy_3b_ollama.sh

# Ollama로 테스트
ollama run frankenstallm-3b "한국의 철강 산업에 대해 설명해줘."
```

### 학습 모니터링

```bash
# 실시간 모니터 (tail -f 방식)
bash scripts/monitor_3b.sh

# 프로세스 상태 확인
ps aux | grep pretrain

# GPU 상태
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv -l 5
```

### 단일 GPU 테스트 (개발/디버그)

```bash
python train/pretrain.py \
  --config configs/korean_3b_fp8.yaml \
  --device cuda:0 \
  --max_steps 100 \
  --debug
```

---

## 14. 로드맵

### 단기 (2026년 3월)

| 항목 | 상태 | 비고 |
|------|------|------|
| Phase 1 (3B Pretrain) 완료 | ✅ 완료 | 57K steps, loss 1.466, 2026-03-05 |
| Phase 2 (SFT) 완료 | ✅ 완료 | 25.5K steps, val_loss 1.8851, 2026-03-06 |
| SFT 6차원 평가 | ✅ 완료 | 4/6 PASS, ORPO 판정 |
| Phase 3 (ORPO Sweep) | ✅ 완료 | 6-config sweep 완료, best config 선정 |
| **Phase 3 (ORPO 본 학습)** | **✅ 완료** | **9,997 steps, eval_loss 1.625, pref_acc 76.02%** |
| Phase 3.5 (ORPO 종합 평가) | ✅ 완료 | 7/10 PASS, 3-way 비교 완료 |
| GGUF 변환 + Ollama 배포 | ✅ 완료 | byte-fallback 수정, Q4_K_M/Q8_0/f16, HF + Ollama 배포 |

### 중기 (2026년 2분기)

| 항목 | 비고 |
|------|------|
| 확장 프리트레인 (80~100B 토큰) | Chinchilla 최적점 달성 |
| QKV Fusion | +8~12% MFU 기대 |
| NUMA Affinity 설정 | +4~9% 예상 |
| FA2 native RoPE | +3~5% 예상 |
| Context length 확장 (4096) | RoPE θ=500K 기반 |

### 장기 (2026년 하반기)

| 항목 | 비고 |
|------|------|
| 7B 실험 | FSDP 전략 필요 |
| vLLM serving | PagedAttention 기반 추론 서버 |
| 도메인 특화 파인튜닝 | 철강/제조업 도메인 |
| 공개 배포 | HuggingFace Hub 업로드 |

### 알려진 미적용 최적화

Phase 0 분석에서 발견했지만 아직 적용하지 않은 최적화들:

| 최적화 | 예상 효과 | 구현 복잡도 |
|--------|-----------|-------------|
| QKV Fusion | +8~12% MFU | 중간 |
| NUMA Affinity | +4~9% | 낮음 |
| FA2 Native RoPE | +3~5% | 낮음 |
| HugePages | +1~3% (TLB 최적화) | 낮음 (sysctl) |

이 최적화들을 모두 적용하면 현재 33.5% MFU에서 45~50%까지 도달할 가능성이 있다.

---

## 15. 참고 문서

| 문서 | 위치 | 내용 |
|------|------|------|
| 프로젝트 전체 여정 | `docs/PROJECT_HISTORY.md` | 일별 상세 진행 기록 |
| 3B 작업 계획 | `docs/3B_WORKPLAN.md` | 3B 단계별 작업 계획 상세 |
| 저스티스리그 논증 | `eval/debate/justice_league_3b_case.md` | 1B→3B 전환 멀티에이전트 토론 전문 |
| SFT 재시작 판결 | `eval/decision/FINAL_DECISION_REPORT.md` | SFT v1 실패 → v2 설계 판결문 |
| 3B 마스터 플랜 | `eval/plan/3B_MASTER_PLAN.md` | 전체 학습 파이프라인 마스터 플랜 |
| Phase 0 최적화 보고서 | `reports/2026-03-02_0200_FRANKENSTALLM_phase0_optimization_report.md` | VRAM/MFU 최적화 전체 보고 |
| 3B Base 평가 보고서 (v1) | `reports/2026-03-05_3B_BASE_EVALUATION_REPORT.md` | 초기 PPL/벤치마크/반복률 평가 |
| PPL 평가 보고서 (v1) | `reports/2026-03-05_PPL_EVALUATION.md` | 4개 검증셋 PPL 상세 |
| 벤치마크 결과 (v1) | `reports/2026-03-05_BENCHMARK_RESULTS.md` | belebele, MMLU 상세 |
| 생성 품질 분석 (v1) | `reports/2026-03-05_GENERATION_QUALITY.md` | 반복률, 디코딩 파라미터 |
| SFT 학습 보고서 | `reports/2026-03-05_3B_SFT_PROGRESS_REPORT.md` | Phase 2 SFT 학습 과정 기록 |
| **SFT 완료 종합 보고서** | `reports/2026-03-06_3B_SFT_COMPLETION_AND_EVAL_SUMMARY.md` | **SFT 완료 + 평가 + 코드 개선 + ORPO 결정 (최신)** |
| SFT 평가 계획서 | `reports/2026-03-06_3B_SFT_EVAL_PLAN.md` | 6차원 평가 설계 |
| SFT 평가 결과 | `reports/2026-03-06_3B_SFT_EVALUATION_REPORT.md` | 6차원 평가 상세 결과 |
| 3B 후속 단계 참조 | `reports/2026-03-05_3B_NEXT_STEPS_REFERENCE.md` | SFT 후 방향성 |
| Nemotron Nano 타당성 | `reports/2026-03-05_NEMOTRON_NANO_FEASIBILITY_STUDY.md` | Hybrid 아키텍처 검토 |
| **v2 종합 평가 리포트** | `eval/outputs/3b_reeval_20260305_1451/full_eval_report.md` | **13+ 벤치마크 종합** |
| v2 PPL 리포트 | `eval/outputs/3b_reeval_20260305_1451/reports/01_perplexity_report.md` | 19개 데이터셋 PPL 상세 |
| v2 Calibration 리포트 | `eval/outputs/3b_reeval_20260305_1451/reports/02_calibration_report.md` | Top-K 정확도, NLL 분포 |
| v2 생성 품질 리포트 | `eval/outputs/3b_reeval_20260305_1451/reports/03_generation_quality.md` | 12조합 파라미터 그리드 서치 |
| v2 벤치마크 리포트 | `eval/outputs/3b_reeval_20260305_1451/reports/04_benchmark_report.md` | KoBEST, HAE-RAE, MMLU, 0/5-shot |
| 진행 기록 | `PROGRESS.md` | 날짜별 체크포인트, 지표, 결정 로그 |
| **ORPO 분석 및 계획** | `reports/2026-03-07_ORPO_ANALYSIS_AND_PLAN.md` | **ORPO 진행 근거, HP 설계, 실행 절차** |
| **ORPO Sweep 디버그** | `reports/2026-03-08_ORPO_SWEEP_DEBUG_REPORT.md` | **QKV 버그, NCCL timeout, TRL 패치 상세** |
| **ORPO 학습 여정** | `reports/2026-03-08_ORPO_TRAINING_JOURNEY.md` | **ORPO 전체 과정: 5번의 실패와 HP sweep (최신)** |

---

## 16. 기술 스택 요약

| 영역 | 기술 | 버전 |
|------|------|------|
| 딥러닝 프레임워크 | PyTorch (NVIDIA 커스텀 빌드) | nv25.12 |
| 어텐션 | FlashAttention-2 | 2.7.4.post1+25.12 |
| FP8 / 혼합 정밀도 | TransformerEngine (MXFP8) | 2.10.0 |
| 분산 학습 | DDP + NCCL (NVLS) | NCCL 2.28.9 |
| 커널 컴파일 | Triton | 3.5.1 |
| 토크나이저 | SentencePiece Unigram 64K | - |
| 모니터링 | Telegram Bot (B200Bot) + cron watchdog | - |
| 추론 서빙 | GGUF + Ollama | - |
| GPU | 8× NVIDIA B200 (NVLink 5.0, NVSwitch) | CUDA 13.1 |
| CPU | 2× AMD EPYC 9365 (Zen 5) | - |

---

## 관련 프로젝트

### [EVAFRILL-Mo](https://github.com/pathcosmos/EVAFRILL-Mo) | [🤗 HuggingFace](https://huggingface.co/pathcosmos/EVAFRILL-Mo-3B)

**하이브리드 Mamba-2 + Transformer 언어 모델** — FRANKENSTALLM의 자매 프로젝트.

NVIDIA [Nemotron-H](https://arxiv.org/abs/2504.03624) 아키텍처에서 영감을 받아 밑바닥부터 직접 구현한 3B 하이브리드 모델이다. FRANKENSTALLM이 순수 Transformer 기반이라면, EVAFRILL-Mo는 **Mamba-2 SSM + 희소 Transformer 어텐션** 하이브리드 구조를 채택했다.

| 항목 | FRANKENSTALLM | EVAFRILL-Mo |
|------|:---:|:---:|
| 아키텍처 | 순수 Transformer (28L) | Mamba-2 24L + Attention 2L |
| 파라미터 | 3.17B | 2.94B |
| 핵심 기술 | GQA, FP8, FlashAttention-2 | Selective Scan, SwiGLU FFN in Mamba, GQA |
| 설계 원칙 | 검증된 Transformer 아키텍처 | Nemotron-H 단편화 도입 |
| GPU | 8× B200 | 7× B200 |
| 학습 전략 | Chinchilla-optimal | Chinchilla 93% 달성 목표 |

두 프로젝트는 동일한 토크나이저(64K SentencePiece), 학습 데이터 파이프라인, DDP/FP8 인프라를 공유한다. "같은 재료, 다른 레시피"로 아키텍처 차이가 성능에 미치는 영향을 비교 실험할 수 있다.

> *이름의 유래: Bride **Eva** (프랑켄슈타인의 신부) + **FRI**DAY (아이언맨 AI 비서) + **LL**M + Nemotron의 **Mo***

---

## 18. 다음 최적화 계획 — MFU 33.5% → 47% 목표

> 상세 문서: [`docs/NEXT_OPTIMIZATION_PLAN.md`](docs/NEXT_OPTIMIZATION_PLAN.md)

### 현재 성능 진단

Phase 1 프리트레인 실측:
- **57,000 steps**, ~38.5B tokens, **약 63시간**
- 처리 속도: 36~38K tok/s per rank → 전체 **~292K tok/s** (8GPU)
- **MFU: ~33.5%**

### 핵심 병목: NUMA Misalignment

```
AMD EPYC 9365 × 2소켓:
  GPU 0~3 → NUMA node 0 (core 0-35)
  GPU 4~7 → NUMA node 1 (core 36-71)

초기 DDP 런칭 시 5/8 rank가 잘못된 NUMA 노드에서 실행.
69%의 DataLoader worker가 크로스-NUMA — ~2배 지연 발생.
```

### 최적화 항목별 예상 효과

| 최적화 | 예상 MFU 개선 | 난이도 |
|--------|-------------|--------|
| NUMA affinity 고정 | +4~9% | 낮음 (launch script 수정) |
| QKV fusion (TransformerEngine) | +8~12% | 중간 (모델 코드 수정) |
| FA2 native RoPE | +3~5% | 중간 (FA2 버전 의존) |
| NCCL 환경변수 튜닝 | +1~2% | 낮음 (한 줄 추가) |

### 최적화 전후 예상 비교

| 항목 | 현재 | 최적화 후 |
|------|------|----------|
| MFU | 33.5% | ~45~47% |
| 처리속도 | 292K tok/s | ~390~410K tok/s |
| 50B 토큰 학습 | ~47시간 | ~34~36시간 |

### 즉시 적용 가능한 코드

**NUMA affinity (launch script):**

```bash
numactl --cpunodebind=0 --membind=0 torchrun \
  --nproc_per_node=4 --node_rank=0 train/pretrain.py ... &
numactl --cpunodebind=1 --membind=1 torchrun \
  --nproc_per_node=4 --node_rank=1 train/pretrain.py ... &
```

**NCCL 환경변수:**

```bash
export NCCL_MIN_NCHANNELS=4
export NCCL_SOCKET_NTHREADS=4
export CUDA_DEVICE_MAX_CONNECTIONS=1
```

> Phase 3 ORPO 완료 후, 다음 프리트레인 런 전에 NUMA affinity를 먼저 적용하면 학습 시간을 ~30% 단축할 수 있다.

---

## 19. GPU 하드웨어 & 비용 분석 — 3B × 60B 프리트레인

> 상세 문서: [`docs/GPU_COST_ANALYSIS.md`](docs/GPU_COST_ANALYSIS.md)

### 실측 기준 베이스라인

```
FRANKENSTALLM Phase 1 실측:
  B200 × 8, MFU 33.5%, 292K tok/s
  38.5B 토큰 → 63시간
  60B 토큰 환산 → 약 98시간
```

### 클라우드 가성비 Top 3 (60B 토큰, 최적화 후)

| 순위 | 구성 | 소요시간 | 총 비용 |
|------|------|---------|--------|
| 1 | H100×8 Cudo | 44.8hr | **$645** (~93만원) |
| 2 | H100×8 Vast.ai | 44.8hr | $670 (~97만원) |
| 3 | H100×8 RunPod | 44.8hr | $713 (~103만원) |

> B200 Blackwell이 빠르지만, 클라우드 단가가 H100의 3배 → **H100이 총비용 4.3배 저렴**

### 개인 GPU 구성 추천

| 구성 | VRAM | NVLink | 가격 | 추천도 |
|------|------|--------|------|--------|
| A6000 Ada × 2 중고 | 96GB (통합) | ✅ | ~1,000만원 | ⭐⭐⭐⭐⭐ |
| L40S × 2 | 96GB (통합) | ✅ | ~1,400만원 | ⭐⭐⭐⭐ |
| RTX Pro 6000 Blackwell | 96GB (단일) | ❌ | ~1,200만원 | ⭐⭐⭐ |

> 소비자용 GPU(RTX 5090/4090)는 NVLink 미지원. 80GB+ 통합 메모리 필요 시 전문가용 필수.

### 추천 전략: 로컬 + 클라우드 하이브리드

```
[로컬] RTX 4090 × 4 (880만원) — 데이터 전처리, 실험, SFT/ORPO
[클라우드] H100×8 (런당 ~103만원) — 본 프리트레인만
```

---

## 20. HuggingFace 배포 현황

> **배포 URL**: https://huggingface.co/pathcosmos/frankenstallm

### 모델 파일 목록

| 파일 | 크기 | 설명 |
|------|------|------|
| `model.safetensors` | 4.76GB | v2 ORPO 베스트 (byte-fallback 수정) — Transformers 직접 로드용 |
| `gguf/frankenstallm-3b-v2-Q4_K_M.gguf` | 757MB | **Ollama 권장** |
| `gguf/frankenstallm-3b-v2-Q8_0.gguf` | 1.2GB | 고품질 |
| `gguf/frankenstallm-3b-v2-f16.gguf` | 2.3GB | 최고품질 |
| `gguf/frankenstallm-3b-Q4_K_M.gguf` | 1.9GB | v1 Q4_K_M |
| `gguf/frankenstallm-3b-Q8_0.gguf` | 3.2GB | v1 Q8_0 |
| `gguf/frankenstallm-3b-f16.gguf` | 6.0GB | v1 f16 |

각 GGUF 파일에 대응하는 `Modelfile.*` (sampling config 포함)이 함께 제공됩니다.

---

## 21. Ollama 사용법 — 상세 설명 및 주의사항

### 빠른 시작 (권장 방법)

```bash
# 1. GGUF + Modelfile 다운로드
huggingface-cli download pathcosmos/frankenstallm   gguf/frankenstallm-3b-v2-Q4_K_M.gguf   gguf/Modelfile.3b-v2-Q4_K_M   --local-dir ./frankenstallm

# 2. Modelfile의 FROM 경로 수정
# FROM ./outputs/gguf/frankenstallm-3b-v2-Q4_K_M.gguf
# → FROM ./frankenstallm/gguf/frankenstallm-3b-v2-Q4_K_M.gguf

# 3. Ollama 모델 등록
ollama create frankenstallm-3b-v2 -f ./frankenstallm/gguf/Modelfile.3b-v2-Q4_K_M

# 4. 실행
ollama run frankenstallm-3b-v2
```

### 검증된 샘플링 파라미터

| 파라미터 | 권장값 | 설명 |
|---------|--------|------|
| `temperature` | **0.7** | 낮을수록 반복, 높을수록 무작위. 0.7이 한국어 품질 최적 |
| `repeat_penalty` | **1.2** | **필수** — 이 값 없이는 greedy 시 30.89% 반복 발생 |
| `top_p` | **0.9** | nucleus sampling, 0.9 이상 권장 |
| `top_k` | **50** | 상위 50개 토큰 후보 |
| `num_predict` | **512** | 최대 생성 토큰 수 |
| `num_ctx` | **4096** | 컨텍스트 윈도우 (최대 4096) |

### ⚠️ 주의사항

**1. repeat_penalty는 반드시 설정하세요**
```
ORPO 학습 후에도 greedy(temp=0) 시 30.89% 3-gram 반복률이 남아 있습니다.
repeat_penalty=1.2 설정 시 반복률 0%로 완전 억제됩니다.
Modelfile에 이미 설정되어 있으므로, Modelfile을 사용하면 자동 적용됩니다.
```

**2. temperature=0 (greedy) 사용 주의**
```
greedy 디코딩은 반복 억제 없이 30.89% 3-gram 반복이 발생합니다.
반드시 temperature >= 0.5 이상 + repeat_penalty >= 1.1 이상을 함께 사용하세요.
```

**3. num_ctx 초과 시 성능 저하**
```
학습 시 max_position_embeddings=4096이었습니다.
4096 토큰을 초과하는 컨텍스트를 넣으면 성능이 크게 저하됩니다.
```

**4. 한국어 중심 모델입니다**
```
영어 능력: MMLU 42.0%, HellaSwag 27.9% — 영어 태스크에서 기대치를 낮추세요.
한국어: KoBEST 0-shot 52.75%, korean_nlu 100.0% (Ollama 벤치마크)
```

**5. v2 vs v1 선택**
```
v2 권장: byte-fallback 수정으로 
 등 특수문자 포함 입력이 안전합니다.
v1은 
 포함 입력 시 llama.cpp가 크래시할 수 있습니다.
```

### Transformers로 직접 실행

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "pathcosmos/frankenstallm"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# ⚠️ do_sample=True + repetition_penalty 반드시 설정
inputs = tokenizer(
    "한국의 전통 음식 중 김치에 대해 설명해주세요.",
    return_tensors="pt"
).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        do_sample=True,        # ← greedy 사용 시 반복 발생
        temperature=0.7,       # ← 핵심 파라미터
        repetition_penalty=1.2, # ← 반드시 설정
        top_p=0.9,
        top_k=50,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id,
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### API 서빙 (Ollama)

```bash
# 백그라운드 서버 실행 (기본 포트 11434)
ollama serve &

# REST API 호출
curl http://localhost:11434/api/generate -d '{
  "model": "frankenstallm-3b-v2",
  "prompt": "한국어로 자기소개를 해줘.",
  "stream": false,
  "options": {
    "temperature": 0.7,
    "repeat_penalty": 1.2,
    "top_p": 0.9,
    "num_predict": 512
  }
}'
```

---

## 22. 모델 성능 비교 — Base / SFT / ORPO / Ollama

### 핵심 지표 3-Way 비교

| 지표 | Base | SFT v2 | ORPO (v2) |
|------|------|--------|-----------|
| **greedy 3-gram 반복률** | 60.99% | 72.97% | **30.89%** |
| **EOS 종료율 (greedy)** | 0% | 60% | 67% |
| **sampling 반복률** (temp=0.7, rep=1.2) | — | — | **0%** |
| **KoBEST 0-shot 평균** | ~44% | 43.26%¹ | **52.75%** |
| **MMLU-KO 0-shot** | 38.8% | 42.0% | — |
| **Hellaswag EN** | 33.3% | — | 27.9% |
| **Calibration Top-1** | ~65% | 68.59% | 67.99% |
| **PPL forgetting** | 0% (기준) | 0.9% | 4.1% |
| **Preference Accuracy** | — | — | **76.02%** |
| **Reward Margin** | — | — | **0.6100** |

> ¹ SFT 초기 평가 시 43.26%, 이후 재평가에서 52.75% (평가 환경 차이).

### KoBEST 세부 비교 (0-shot)

| 태스크 | Base | ORPO |
|--------|------|------|
| BoolQ | ~48% | 54.3% |
| COPA | ~52% | 56.2% |
| WiC | ~49% | 51.8% |
| SentiNeg | ~44% | 51.4% |
| HellaSwag-KO | ~38% | 49.9% |
| **평균** | ~46% | **52.75%** |

### Ollama 벤치마크 (frankenstallm-3b-v2:Q4_K_M, 35 tests)

| 카테고리 | 점수 |
|----------|------|
| korean_nlu | **100.0** |
| knowledge | 75.0 |
| instruction_following | 66.7 |
| reasoning | 50.0 |
| safety | 10.0 |
| repetition_resistance | 2.2 |
| **자동 채점 평균** | **46.7** |
| 평균 TPS | 142.5 tok/s |
| 평균 TTFT | 16.7 ms |

> repetition_resistance 2.2%는 Ollama 기본 파라미터(sampling 없이 실행)로 측정된 값.  
> `repeat_penalty=1.2` + `temperature=0.7` 적용 시 **반복률 0%** 달성.

### 3B급 모델 성능 비교 — Ollama 벤치마크

동일 환경(Ollama, 35개 테스트)에서 3B급 오픈소스 모델들과 직접 비교한 결과입니다.

#### 종합 점수

| 모델 | 파라미터 | 자동 채점 평균 | 비고 |
|------|---------|---------------|------|
| **Qwen 2.5 3B** | 3B | **63.4** | 종합 1위 |
| **Phi-4 Mini** | 3.8B | **60.6** | 추론 특화 |
| **FRANKENSTALLM 3B v2** | 3B | **46.7** | 본 모델 (ORPO) |
| FRANKENSTALLM 3B v1 | 3B | 37.9 | SFT only |

#### 카테고리별 상세 비교

| 카테고리 | FRANKENSTALLM v2 | Qwen 2.5 3B | Phi-4 Mini 3.8B | 비고 |
|----------|:---:|:---:|:---:|------|
| **Korean NLU** | **100.0** | **100.0** | 66.7 | 한국어 이해력 동급 |
| **Knowledge** | **75.0** | 20.8 | 29.2 | 한국어 지식에서 압도적 우위 |
| **Instruction Following** | **66.7** | 55.6 | 33.3 | 지시 따르기 우수 |
| **Reasoning** | 50.0 | 62.5 | **87.5** | Phi-4 추론 특화 |
| **Code** | 0.0 | **100.0** | 83.3 | 코드 능력 부족 |
| **Safety** | 10.0 | 35.0 | **70.0** | 안전성 약점 |
| **Repetition Resistance** | 2.2 | **75.0** | 58.9 | 기본 파라미터 기준¹ |

> ¹ FRANKENSTALLM의 반복 저항 2.2%는 Ollama 기본 설정 기준. `repeat_penalty=1.2` 적용 시 0% 달성.

#### 추론 속도 비교

| 모델 | Avg TTFT (ms) | P95 TTFT (ms) | Avg TPS | 비고 |
|------|:---:|:---:|:---:|------|
| **FRANKENSTALLM 3B v2** | **16.7** | **26.2** | **142.5** | 최고 속도 |
| Phi-4 Mini 3.8B | 25.6 | 44.9 | 100.4 | |
| Qwen 2.5 3B | 28.2 | 46.5 | 93.8 | |

> FRANKENSTALLM은 동일 아키텍처(LlamaForCausalLM)와 64K vocab으로 Ollama 추론에 최적화됨.

#### lm-eval 벤치마크 비교 (Base → SFT → ORPO)

| 벤치마크 | Base | SFT | ORPO | 변화 (Base→ORPO) |
|----------|:---:|:---:|:---:|:---:|
| **KoBEST COPA** | 49.3% | 48.6% | **63.9%** | **+14.6pp** |
| **KoBEST HellaSwag** | 21.6% | 19.8% | **38.0%** | **+16.4pp** |
| **KoBEST SentiNeg** | 48.6% | 49.1% | **62.5%** | **+13.9pp** |
| **KoBEST BoolQ** | 50.3% | 50.1% | 50.6% | +0.3pp |
| **KoBEST WiC** | 48.7% | 48.7% | 48.8% | +0.2pp |
| **KoBEST 평균** | 43.7% | 43.3% | **52.8%** | **+9.1pp** |
| **HAE-RAE** | 19.7% | 19.9% | 21.8% | +2.1pp |
| **PIQA** | 52.5% | 52.6% | **59.9%** | **+7.3pp** |
| **ARC-Easy** | 25.6% | 25.9% | **36.0%** | **+10.4pp** |
| **HellaSwag EN** | 26.2% | 26.1% | 29.2% | +3.0pp |
| **Winogrande** | 50.6% | 50.8% | 51.0% | +0.4pp |
| **PPL forgetting** | 기준 | 0.9% | 4.1% | 임계 15% 이내 ✅ |

#### 해석 및 위치

**강점**:
- **한국어 NLU 100%** — Qwen 2.5 3B와 동급, Phi-4를 압도
- **한국어 지식 75.0** — 비교 모델 중 최고 (Qwen 20.8, Phi-4 29.2)
- **추론 속도 최고** — TTFT 16.7ms, TPS 142.5로 실시간 서빙에 유리
- **ORPO 효과 극대** — KoBEST +9.1pp, COPA/HellaSwag/SentiNeg에서 각 13~16pp 향상

**약점 및 개선 방향**:
- **코드 생성 0%** — 코드 SFT 데이터 부족. 코드 특화 데이터 추가 필요
- **안전성 10%** — safety alignment 데이터 미반영. RLHF/DPO 추가 학습 고려
- **반복 문제** — 기본 설정에서 반복 발생하나, `repeat_penalty=1.2`로 완전 해결
- **종합 점수 gap** — Qwen 대비 -16.7pp. 학습 데이터 품질 및 규모 차이가 주 원인

---

## 23. 재현 가이드 — 전 단계 설정 상세

> 이 섹션은 동일 환경에서 재현 시 참고용입니다.

### 환경 설정

```bash
# NVIDIA 커스텀 PyTorch는 재설치 금지 (B200 최적화 깨짐)
# 아래 패키지만 추가 설치
pip install transformers==4.40.0 accelerate peft trl deepspeed \
            bitsandbytes sentencepiece wandb

# 전체 환경 재현
pip install -r requirements.txt
```

**소프트웨어 버전 (실측)**:
```
torch          2.10.0a0+b4e4ee81d3.nv25.12
flash_attn     2.7.4.post1+25.12
transformers   4.40.x
datasets       4.4.1
tokenizers     0.22.1
huggingface_hub 1.2.3
trl            (ORPO NaN 버그 패치 적용본 — scripts/trl_patch.py 참고)
CUDA           13.1 / Driver 580.95.05
```

### Phase 1 — Pretrain 핵심 하이퍼파라미터

```yaml
# configs/korean_3b_fp8.yaml 기준
model:
  hidden_size: 2048
  num_hidden_layers: 24
  num_attention_heads: 16
  num_key_value_heads: 4
  intermediate_size: 5632
  max_position_embeddings: 4096
  vocab_size: 64000

train:
  batch_size: 5
  gradient_accumulation_steps: 8
  # effective batch = 5 × 8 × 8GPU = 320
  learning_rate: 1.5e-4
  min_lr: 1.5e-5          # cosine decay 하한 (max_lr의 10%)
  warmup_steps: 2000
  weight_decay: 0.1
  max_grad_norm: 1.0
  scheduler: cosine
  precision: bf16          # FP8 Tensor Core 활용 (B200)
  total_tokens: ~38.5B
```

### Phase 2 — SFT v2 핵심 하이퍼파라미터

```yaml
# configs/korean_3b_sft_v2.yaml 기준
train:
  learning_rate: 5.0e-5
  batch_size: 4
  gradient_accumulation_steps: 8
  # effective batch = 4 × 8 × 8GPU = 256
  warmup_ratio: 0.03
  weight_decay: 0.01
  lr_scheduler_type: cosine
  max_steps: 33000          # early stop at 25,500 (patience=5)
  early_stopping_patience: 5
  fp16: false
  bf16: true

data:
  total_samples: 2439397    # train
  val_samples: 49801
  num_sources: 24
  total_size_gb: 7.48
  mixing: "70% instruction / 30% general"
```

### Phase 3 — ORPO 핵심 하이퍼파라미터

```yaml
# configs/korean_3b_orpo.yaml 기준 (본 학습)
train:
  beta: 0.25                # ORPO preference weight
  learning_rate: 1.2e-5    # HP sweep 최적값
  batch_size: 4
  gradient_accumulation_steps: 4
  # effective batch = 4 × 4 × 8GPU = 128
  warmup_ratio: 0.1
  weight_decay: 0.01
  lr_scheduler_type: cosine
  max_length: 1536          # prompt + response 합산 최대 길이
  max_prompt_length: 512
  num_train_epochs: 2       # 실제 ~9,997 steps에서 조기 수렴
  bf16: true
  optim: adamw_torch_fused

data:
  raw_pairs: 683181
  filtered_pairs: ~630000   # NaN 방지 필터 후
  eval_split: 0.05          # seed=42
  eval_pairs: ~31500
```

### Phase 4 — GGUF 변환 파이프라인

```bash
# Step 1: byte-fallback 토크나이저 수정
python scripts/fix_tokenizer_byte_fallback.py \
  --input  outputs/hf_checkpoint-best \
  --output outputs/hf_checkpoint-best-fixed

# Step 2: f16 GGUF 변환 (llama.cpp)
python outputs/llama.cpp/convert_hf_to_gguf.py \
  outputs/hf_checkpoint-best-fixed \
  --outfile outputs/gguf/frankenstallm-3b-v2-f16.gguf \
  --outtype f16

# Step 3: 양자화
QUANTIZE=outputs/llama.cpp/build/bin/llama-quantize
$QUANTIZE outputs/gguf/frankenstallm-3b-v2-f16.gguf \
          outputs/gguf/frankenstallm-3b-v2-Q4_K_M.gguf Q4_K_M
$QUANTIZE outputs/gguf/frankenstallm-3b-v2-f16.gguf \
          outputs/gguf/frankenstallm-3b-v2-Q8_0.gguf Q8_0

# Step 4: Ollama 등록
ollama create frankenstallm-3b-v2:Q4_K_M -f Modelfile.3b-v2-Q4
ollama create frankenstallm-3b-v2:Q8_0   -f Modelfile.3b-v2-Q8
ollama create frankenstallm-3b-v2:f16    -f Modelfile.3b-v2-f16
```

### 토크나이저 재현

```
학습 방식: SentencePiece Unigram
vocab_size: 64,000 (원본) → 64,256 (byte-fallback 수정 후)
학습 스크립트: tokenizer/train_sp_tokenizer.py
학습 데이터: C4 Korean + 나무위키 + Wikipedia Korean 혼합
byte_fallback: True (v2에서 추가)
추가 토큰: <0x00> ~ <0xFF> 256개
```

---

## 마치며

이 프로젝트의 모토는 하나다:

> **"망하는 것도 기록한다."**

SFT v1의 loss=0.0 실패, torch.compile이 효과 없었던 것, 18% 반복률의 좌절 — 이 모든 것이 기록에 남아 있다. Phase 3 ORPO에서도 **5번의 실패** — NCCL timeout, config 충돌, QKV 변환 버그, 포트 충돌, TRL NaN 버그 — 를 거쳐 마침내 완주했다.

Phase 1 프리트레인은 57,000 steps, loss 1.466으로 완료됐다. Phase 2 SFT는 25,500 steps에서 early stopping (val_loss 1.8851). Phase 3 ORPO는 9,997 steps에서 조기 수렴 — eval_loss 1.625, Preference Accuracy 76.02%. Phase 4로 GGUF 변환 후 HuggingFace와 Ollama에 배포 완료.

**결국 해냈다**: greedy 반복률 72.97% → 30.89% (ORPO), sampling+rep_penalty 적용 시 0%. TPS 142.5, TTFT 16.7ms. 한국어를 이해하고 말하는 3B 모델, 처음부터 만든 것.

Frankenstein이 조각들을 이어 붙여 생명을 만들었듯, FRANKENSTALLM도 그렇게 만들어졌다.

**🤗 모델 다운로드**: https://huggingface.co/pathcosmos/frankenstallm

---

## Acknowledgment

이 프로젝트는 **과학기술정보통신부**의 **「첨단 GPU 활용 지원 사업」** (과학기술정보통신부 공고 제2025-1068호)을 통해 제공된 GPU 컴퓨팅 자원을 활용하여 수행되었습니다.

> **국가 AI컴퓨팅자원 지원포털**: https://aiinfrahub.kr
>
> - 주관: 과학기술정보통신부 (MSIT), 정보통신산업진흥원 (NIPA)
> - 운영: 한국정보통신진흥협회 (KAIT)

대한민국 정부의 AI 인프라 지원 사업 덕분에 8× NVIDIA B200 GPU 환경에서 한국어 3B LLM을 처음부터 학습할 수 있었습니다. 국가 차원의 AI 컴퓨팅 자원 지원에 깊이 감사드립니다.

---

*최종 업데이트: 2026-03-26*
*현재 상태: **전 단계 완료** — Phase 1 Pretrain ✅ | Phase 2 SFT ✅ | Phase 3 ORPO ✅ | Phase 4 배포 ✅*
