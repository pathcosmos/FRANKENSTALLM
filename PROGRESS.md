# LLM Bang — 학습부터 Ollama/GGUF 배포까지 진행률

> 갱신: 2026-02-25 (23:10)
> 목표: 한국어 1B 파라미터 LLM을 사전학습하고 Ollama로 배포

---

## 전체 진행률: 약 33%

| # | 단계 | 가중치 | 상태 | 완료율 | 기여 |
|---|------|--------|------|--------|------|
| 1 | 환경 설정 & GPU 인프라 | 5% | ✅ 완료 | 100% | 5.0% |
| 2 | 모델 아키텍처 구현 | 10% | ✅ 완료 | 100% | 10.0% |
| 3 | 한국어 데이터 파이프라인 | 20% | 🔄 진행 중 | ~65% | 13.0% |
| 4 | 사전학습 (Pre-training) | 35% | 🔄 진행 중 | ~15% | 5.3% |
| 5 | 평가 (Evaluation) | 5% | ⏳ 대기 | 0% | 0% |
| 6 | 파인튜닝 (Instruction SFT) | 10% | ⏳ 대기 | 0% | 0% |
| 7 | HuggingFace 포맷 변환 | 5% | ⏳ 대기 | 0% | 0% |
| 8 | GGUF 변환 & 양자화 | 5% | ⏳ 대기 | 0% | 0% |
| 9 | Ollama 배포 | 5% | ⏳ 대기 | 0% | 0% |

**합계: 5.0 + 10.0 + 13.0 + 5.3 = 33.3% ≈ 약 33%**

---

## 단계별 세부 현황

### ✅ 1단계: 환경 설정 & GPU 인프라 (100%)
- ✅ 8× NVIDIA B200 (192GB × 8 = 1.5TB VRAM)
- ✅ CUDA 13.1, cuDNN 9.17.0, NCCL 2.28.9
- ✅ PyTorch 2.10.0a0 NV 커스텀 빌드 (nv25.12, B200 최적화)
- ✅ flash_attn 2.7.4, TransformerEngine 2.10 (FP8)
- ✅ Triton 3.5.1, Apex 설치 완료
- ✅ DDP + FP8 (`MXFP8BlockScaling`) 파이프라인 검증 완료

### ✅ 2단계: 모델 아키텍처 구현 (100%)
- ✅ `model/transformer.py` — GPT-2 스타일 Transformer (RoPE, GQA 지원)
- ✅ `model/attention.py` — FlashAttention-2 통합, FP8 BF16 cast 방어 코드
- ✅ `model/config.py` — `LMConfig` (use_fp8, vocab_size=64K 등)
- ✅ `train/pretrain.py` — DDP + FP8 + gradient checkpointing
- ✅ `configs/korean_1b_fp8.yaml` — 1B 한국어 모델 설정 (vocab=64K)
- ✅ `data/dataset.py` — `PackedDataset` (비겹침, 효율적 DataLoader)

### 🔄 3단계: 한국어 데이터 파이프라인 (~65%)

| 작업 | 상태 | 비고 |
|------|------|------|
| ko_wiki + en_wiki 다운로드 | ✅ 완료 | 770K 문서, 462M 토큰 추정 |
| mC4 Korean (allenai/c4) 다운로드 | ✅ 완료 | 5M 행, 50 샤드, ~30GB |
| Namuwiki 다운로드 | ✅ 완료 | 565K 문서, 6 샤드, ~5.7GB |
| CC-100 Korean 다운로드 | ❌ 실패 | `--text_col text` 버그 (cc100_ko 비어있음), `download_cc100.sh` 재시도 스크립트 준비 완료 |
| SP 64K Unigram 토크나이저 학습 | ✅ 완료 | vocab=64,000, `tokenizer/korean_sp/tokenizer.model` |
| SP → HF tokenizers.json 변환 | ✅ 완료 | `tokenizer/korean_sp/tokenizer.json` (4.1MB) |
| c4_ko 토크나이징 → `.bin` | 🔄 진행 중 | `finish_korean_pipeline.sh` 백그라운드 실행 중 (PID 464940) |
| namuwiki 토크나이징 → `.bin` | ⏳ 미실행 | `finish_korean_pipeline.sh` 실행 필요 |
| ko_wiki 토크나이징 → `.bin` | ⏳ 미실행 | `finish_korean_pipeline.sh` 실행 필요 |
| .bin 병합 → `korean_train.bin` | ⏳ 미실행 | 위 토크나이징 완료 후 자동 실행 |

**다음 실행 명령:**
```bash
# 백그라운드 실행
nohup bash data/finish_korean_pipeline.sh > data/finish_korean_pipeline.log 2>&1 &
tail -f data/finish_korean_pipeline.log
```

### 🔄 4단계: 사전학습 (~8%)

| 실험 | 상태 | 진행 |
|------|------|------|
| `small_fp8_run1` (125M, FP8 파이프라인 검증) | ✅ 완료 | 100,000/100,000 steps, loss 2.36 |
| `korean_1b_fp8_run1` (1B, 한국어) | ⏳ 대기 | 3단계 완료 후 시작 |

- 처리 속도: ~330K tok/s per GPU (≈2.64M tok/s, 8× B200)
- 최종 체크포인트: `checkpoints/small_fp8_run1/checkpoint-0100000`
- 1B 학습: `korean_train.bin` 완료 후 `bash scripts/launch_korean_1b.sh` 실행 예정

### ⏳ 5단계: 평가 (0%)
- `eval/perplexity.py` — 계획됨
- HellaSwag, Ko-NLU 등 downstream 태스크 평가 — 계획됨

### ⏳ 6단계: 파인튜닝 — Instruction SFT (0%)
- 한국어 instruction 데이터셋 선정 (KoAlpaca, KULLM 등)
- LoRA 또는 full fine-tuning

### ⏳ 7단계: HuggingFace 포맷 변환 (0%)
- 체크포인트 → `config.json`, `pytorch_model.bin` / `safetensors`
- `transformers.AutoModelForCausalLM` 호환 형식

### ⏳ 8단계: GGUF 변환 & 양자화 (0%)
- `llama.cpp/convert_hf_to_gguf.py` 사용
- Q4_K_M, Q8_0 양자화 적용

### ⏳ 9단계: Ollama 배포 (0%)
- `Modelfile` 작성
- `ollama create korean-llm-1b -f Modelfile`
- `ollama run korean-llm-1b`

---

## 주요 파일 경로

| 파일 | 설명 |
|------|------|
| `configs/korean_1b_fp8.yaml` | 1B 한국어 모델 학습 설정 |
| `tokenizer/korean_sp/tokenizer.json` | 한국어 64K vocab 토크나이저 |
| `data/finish_korean_pipeline.sh` | 데이터 파이프라인 재개 스크립트 |
| `scripts/launch_korean_1b.sh` | 1B 모델 학습 런처 |
| `checkpoints/small_fp8_run1/checkpoint-0100000` | 125M FP8 검증 실험 최종 체크포인트 (완료) |

---

## 빠른 실행 체크리스트

```bash
# 1. 데이터 파이프라인 완료 (현재 백그라운드 실행 중 — 대기)
tail -f data/finish_korean_pipeline.log

# 2. 데이터 생성 확인
bash scripts/check_korean_data.sh

# 3. 1B 한국어 모델 학습 시작
bash scripts/launch_korean_1b.sh

# 4. 학습 로그 확인
tail -f checkpoints/korean_1b_fp8_run1/train.log
```
