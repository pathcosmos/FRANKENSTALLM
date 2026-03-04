# Nemotron-3-Nano 스타일 모델 학습 계획

> 작성: 2026-02-25  
> 참조: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16  
> HF Token: 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  

---

## 목표 모델 분석

### 원본 스펙 (NVIDIA Nemotron-3-Nano-30B-A3B)

| 항목 | 내용 |
|------|------|
| 총 파라미터 | 30B |
| 활성 파라미터 | 3.5B (per token, MoE) |
| 아키텍처 | Mamba2 + MoE + Attention 하이브리드 |
| 총 레이어 | 52개 |
| Mamba-2 레이어 | 23개 |
| MoE 레이어 | 23개 (라우팅 전문가 128 + 공유 1, 토큰당 6 활성화) |
| Attention 레이어 | 6개 (GQA, 2 groups) |
| 학습 토큰 | 25T tokens |
| LR 스케줄 | WSD (Warmup-Stable-Decay) |
| Peak LR | 1e-3 |
| Batch size | 3072 |
| 학습 기간 | 2025년 9월 ~ 12월 |

### 성능 하이라이트
- AIME25 (no tools): 89.1 (Qwen3-30B-A3B 85.0 대비 우위)
- GPQA: 73.0
- LiveCodeBench: 68.3
- SWE-Bench (OpenHands): 38.8

---

## 우리 서버 가용 자원

| 항목 | 스펙 |
|------|------|
| GPU | 8× NVIDIA B200 (183GB VRAM each = 1.47TB 합계) |
| RAM | 2.2TB |
| 스토리지 | /PROJECT 3.5TB (여유 2.2TB) |
| CPU | AMD EPYC 9365 72코어 |

---

## 가능성 평가

### ✅ 완전 재현 가능 여부 (처음부터 학습)

| 항목 | 평가 | 비고 |
|------|------|------|
| **VRAM** | ✅ 충분 | 30B BF16 = 60GB, 학습 시 ~240GB → 8×183GB=1.47TB로 여유 |
| **아키텍처 구현** | ⚠️ 복잡 | Mamba2 + MoE 하이브리드, 공개 구현체 있음 (Zamba, Jamba 참조) |
| **25T 토큰 데이터** | ❌ 불가 | 현실적으로 수집 불가능 (수개월 다운로드) |
| **학습 시간** | ❌ 수개월 | 25T tokens / ~1M tok/step = 수백만 step |

→ **원본과 동일한 스케일로 처음부터 학습: 현실적으로 불가**

### ✅ 현실적 대안 3가지

#### 옵션 A: 베이스 모델 다운로드 후 한국어 Continual Pretraining (추천 ⭐)
- HF에서 `NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16` 다운로드
- 한국어 데이터로 continual pretraining (수억~수십억 토큰)
- 이후 한국어 SFT / RLHF
- **장점**: 영어 기반 지식 보존, 한국어 능력 추가
- **예상 기간**: 데이터 준비 1~2주, 학습 수일

#### 옵션 B: Mamba2-MoE 소규모 버전 처음부터 학습
- 구조는 Nemotron과 동일하게, 규모는 축소 (예: 3B 전체, 0.5B 활성)
- 한국어 중심 데이터로 학습 (llm-bang의 korean_train.bin 활용 가능)
- **장점**: 아키텍처 완전 이해 및 커스터마이징 가능
- **예상 기간**: 구현 2~4주, 학습 수일~수주

#### 옵션 C: 원본 모델 다운로드 후 한국어 SFT만 진행
- `NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (instruction 버전) 다운로드
- 한국어 instruction 데이터로 SFT (LoRA 또는 full fine-tuning)
- **장점**: 가장 빠름, 기존 성능 활용
- **예상 기간**: 데이터 준비 수일, 학습 1~2일

---

## 단계별 학습 계획 (옵션 A 기준 - 추천)

### Phase 0: 환경 준비 (1~2일)
- [ ] `mamba-ssm` 패키지 설치 (Mamba2 CUDA 커널)
- [ ] `causal-conv1d` 설치
- [ ] HF 베이스 모델 다운로드 (~60GB)
  ```bash
  huggingface-cli download nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16 \
    --token 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 \
    --local-dir /PROJECT/0325120031_A/ghong/taketimes/llm-bang/nemotron-nano/base_model
  ```

### Phase 1: 한국어 데이터 준비 (1~2주)
- [ ] 기존 llm-bang 한국어 데이터 재활용 (korean_train.bin, 17GB)
- [ ] 추가 한국어 데이터 수집:
  - 나무위키, 한국어 위키백과 (기보유)
  - 한국어 CC-100 (재시도)
  - 한국어 뉴스 / 도서 등
- [ ] Nemotron 토크나이저 기반 재토크나이징 (vocab 64K → 원본 토크나이저)
- 목표: 최소 10B 토큰 한국어 데이터

### Phase 2: Continual Pretraining (2~7일)
```bash
# 학습 커맨드 (예상)
torchrun --nproc_per_node=8 \
  train/continual_pretrain.py \
  --model_path nemotron-nano/base_model \
  --train_data data/korean_nemotron_train.bin \
  --output_dir nemotron-nano/checkpoints/korean_cpt_run1 \
  --max_steps 50000 \
  --batch_size 4 \
  --grad_accum 8 \
  --lr 1e-5 \
  --warmup_steps 500
```

- Effective batch: 4×8GPU×8×4096 = ~1M tok/step
- 50K steps × 1M = 50B 토큰 (한국어 데이터 5회 반복)
- B200 FP8 기준 예상 속도: ~200K tok/s → 약 70시간

### Phase 3: 한국어 SFT (1~3일)
- [ ] 한국어 instruction 데이터 준비
  - KoAlpaca, OpenAssistant-KO 등
  - 자체 생성 (GPT-4o 활용)
- [ ] SFT (full fine-tuning 또는 LoRA)
- 목표: 한국어 instruction following 능력 확보

### Phase 4: 평가 (수일)
- [ ] Ko-MMLU, KoBEST 등 한국어 벤치마크
- [ ] GGUF 변환 → Ollama 배포

---

## 아키텍처 참고 자료

### Mamba2 관련
- Paper: [Mamba-2](https://arxiv.org/abs/2405.21060)
- Code: https://github.com/state-spaces/mamba
- Hybrid: [Zamba2](https://github.com/Zyphra/Zamba2), [Jamba](https://huggingface.co/ai21labs/Jamba-v0.1)

### Nemotron 관련
- Paper: https://arxiv.org/abs/2512.20848
- White Paper: https://arxiv.org/abs/2512.20856
- NeMo Framework: https://github.com/NVIDIA/NeMo

---

## 폴더 구조

```
nemotron-nano/
├── PLAN.md                   # 이 파일
├── base_model/               # HF 다운로드 모델
├── data/                     # 한국어 학습 데이터
├── configs/                  # 학습 설정 파일
├── checkpoints/              # 학습 체크포인트
│   └── korean_cpt_run1/
├── scripts/                  # 학습/평가 스크립트
└── eval/                     # 평가 결과
```

---

## 핵심 판단

| 질문 | 답 |
|------|-----|
| 이 모델과 동일한 것을 만들 수 있나? | ❌ 25T 토큰 데이터와 수개월 학습 필요 |
| 이 모델 구조로 뭔가 만들 수 있나? | ✅ 베이스 다운로드 + 한국어 CPT로 충분히 가능 |
| 우리 서버로 학습 가능한가? | ✅ VRAM 1.47TB로 30B 모델 학습 여유롭게 가능 |
| 가장 빠른 경로는? | 옵션 C (SFT만) → 수일 내 결과물 |
| 가장 의미 있는 경로는? | 옵션 A (CPT + SFT) → 진정한 한국어 Nemotron |
