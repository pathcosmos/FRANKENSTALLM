# FRANKENSTALLM 프로젝트 완료 보고서

- **작성일**: 2026-03-10
- **상태**: Phase 1~4 전체 완료, HuggingFace 배포 완료
- **모델 배포**: https://huggingface.co/pathcosmos/frankenstallm

---

## 1. 프로젝트 개요

8× NVIDIA B200 GPU 위에서 한국어 3B LLM을 **처음부터 직접** 구현하고 배포까지 완료한 실험 프로젝트.

| 항목 | 내용 |
|------|------|
| **목표** | 한국어에 특화된 3B 파라미터 언어 모델 개발 |
| **접근** | Pretrain → SFT → ORPO 파인튜닝 → GGUF 변환 → Ollama 배포 |
| **하드웨어** | 8× NVIDIA B200 (183GB VRAM each, 총 ~1.47TB VRAM) |
| **총 학습 기간** | 2026-02-27 ~ 2026-03-09 |

---

## 2. 전체 진행 요약 (4단계)

```
Phase 1: Pretrain ──────── 57,000 steps, loss 1.466  ✅ 완료
Phase 2: SFT v2  ──────── 25,500 steps, val_loss 1.8851 ✅ 완료
Phase 3: ORPO    ──────── 9,997 steps, eval_loss 1.625 ✅ 완료
Phase 4: GGUF 변환·배포 ─ HuggingFace + Ollama       ✅ 완료
```

---

## 3. Phase 1 — 사전학습 (Pretrain)

| 항목 | 값 |
|------|-----|
| **총 학습 스텝** | 57,000 steps |
| **최종 Loss** | 1.466 (수렴) |
| **학습 토큰** | ~38.5B tokens |
| **하드웨어** | 8× B200, DDP |
| **처리 속도** | 292K tok/s (MFU 33.5%) |
| **Precision** | BF16 / FP8 Tensor Core |

### 모델 아키텍처

| 항목 | 값 |
|------|-----|
| Architecture | LlamaForCausalLM |
| Hidden size | 2,048 |
| Layers | 24 |
| Attention heads | 16 |
| KV heads | 4 |
| Max position | 4,096 |
| Vocab size | 64,256 (64,000 + 256 byte-fallback) |

### 주요 이슈 및 해결

- **NUMA affinity**: GPU 0-3 → NUMA node 0, GPU 4-7 → NUMA node 1 최적 매핑 적용
- **FP8 mixed precision**: B200 네이티브 `torch.float8_e4m3fn` 활용
- **DDP static_graph=False**: Transformer Engine과의 호환성 확보

---

## 4. Phase 2 — SFT (지시 학습)

| 항목 | 값 |
|------|-----|
| **총 학습 스텝** | 25,500 steps (early stopping) |
| **val_loss** | 1.8851 |
| **학습률** | 5e-5 |
| **Batch size** | 4 (grad_accum=8 → effective BS 32) |
| **EOS 종료율** | 0% → 60% |
| **greedy 반복률** | 60.99% → 72.97% (ORPO 필요 확인) |

### SFT 결과

- 지식 보존 우수 (forgetting 0.9% — Base PPL 거의 유지)
- 지시 따르기 일부 학습 (EOS 0% → 60%)
- **반복 문제 미해결** (greedy rep 72.97%) → ORPO 진행 결정

---

## 5. Phase 3 — ORPO (선호도 정렬)

### HP Sweep (6 configs)

| Config | beta | lr | eval_loss | pref_acc |
|--------|------|----|-----------|----------|
| baseline | 0.25 | 8e-6 | 1.703 | 72.1% |
| fast | 0.25 | 1.2e-5 | 1.693 | 73.8% |
| **best** ← | **0.25** | **1.2e-5** | **1.693** | **73.8%** |
| conserv | 0.15 | 5e-6 | 1.721 | 70.2% |
| aggressive | 0.30 | 1.5e-5 | 1.709 | 72.5% |

### 본 학습 결과

| 항목 | 값 |
|------|-----|
| **스텝** | 9,997 (조기 수렴) |
| **eval_loss** | 1.7910 → **1.6250** |
| **Preference Accuracy** | 67.8% → **76.02%** |
| **Reward Margin** | 0.107 → **0.6100** |
| **greedy 반복률** | 72.97% → **30.89%** (↓42pp) |
| **EOS 종료율** | 60% → 67% |
| **KoBEST 0-shot** | **52.75%** |
| **PPL forgetting** | 최대 4.1% (임계값 15% 이하 ✅) |

### 10차원 종합 평가

| # | 차원 | 결과 |
|---|------|------|
| 1 | Perplexity (지식 보존) | ✅ PASS |
| 2 | 생성 품질 (greedy) | ❌ FAIL (30.89%, 목표 <5%) |
| 3 | 한국어 벤치마크 KoBEST | ❌ FAIL (52.75%, 목표 >55%) |
| 4 | 영어 벤치마크 | ❌ FAIL |
| 5 | Calibration | ✅ PASS (67.99%) |
| 6 | SFT Chat 능력 | ✅ PASS |
| 7 | Preference Accuracy | ✅ PASS (76.02%) |
| 8 | Reward Margins | ✅ PASS (0.6100) |
| 9 | Parameter Sensitivity | ✅ PASS |
| 10 | SFT→ORPO 개선 | ✅ PASS |
| **종합** | **7/10 PASS** | 정량 스코어 **63.7/100** |

### 주요 이슈 해결: TRL ORPO NaN 버그

TRL 라이브러리의 ORPO 구현에서 `chosen_logps`가 `nan`이 되는 버그를 3중 패치로 해결:
1. `torch.nan_to_num()` 클램핑
2. `log_softmax` 수치 안정화
3. `inf` 마스킹

---

## 6. Phase 4 — GGUF 변환 & 배포

### byte-fallback 문제 해결 (v2)

| 항목 | 내용 |
|------|------|
| **문제** | SentencePiece Unigram 토크나이저에 `byte_fallback` 미적용 |
| **증상** | `\n` 등 미등록 문자 입력 시 llama.cpp 크래시 |
| **해결** | 256개 byte-fallback 토큰 추가, 임베딩 64000→64256 리사이즈 |
| **검증** | `\n` 포함 프롬프트 Ollama 처리 확인 (크래시 없음) |

### 변환 결과

| 파일 | 크기 | 설명 |
|------|------|------|
| `frankenstallm-3b-f16.gguf` | 6.0G | v1 f16 (ORPO, 원본) |
| `frankenstallm-3b-Q8_0.gguf` | 3.2G | v1 Q8_0 |
| `frankenstallm-3b-Q4_K_M.gguf` | 1.9G | v1 Q4_K_M |
| `frankenstallm-3b-v2-f16.gguf` | 2.3G | **v2 f16 (byte-fallback 수정)** |
| `frankenstallm-3b-v2-Q8_0.gguf` | 1.2G | **v2 Q8_0** |
| `frankenstallm-3b-v2-Q4_K_M.gguf` | 757M | **v2 Q4_K_M ← 권장** |

### Ollama 배포 벤치마크

| 항목 | 값 |
|------|-----|
| 모델명 | `frankenstallm-3b-v2:Q4_K_M` |
| 테스트 수 | 35 (자동 20 + 수동 15) |
| 자동 채점 평균 | **46.7** |
| 평균 TPS | **142.5 tok/s** |
| 평균 TTFT | **16.7 ms** |
| korean_nlu | 100.0 |
| reasoning | 50.0 |
| knowledge | 75.0 |
| instruction_following | 66.7 |

### 최적 샘플링 파라미터

ORPO eval grid 실측 최적값 (`t0.7_rep1.2`):

| 파라미터 | 값 | 비고 |
|---------|-----|------|
| temperature | **0.7** | 창의성/일관성 균형점 |
| repeat_penalty | **1.2** | greedy 반복 0%로 억제 |
| top_p | **0.9** | nucleus sampling |
| top_k | **50** | |
| num_predict | **512** | |
| num_ctx | **4096** | |

---

## 7. HuggingFace 배포 현황 (2026-03-10)

**URL**: https://huggingface.co/pathcosmos/frankenstallm

### 업로드 완료 파일

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

---

## 8. 하드웨어 환경

| 항목 | 사양 |
|------|------|
| GPU | 8× NVIDIA B200 |
| VRAM | 183GB HBM3e/GPU (~1.47TB total) |
| FP8 Tensor Core | 2,250 TFLOPS/GPU |
| NVLink | 5.0 NV18, 900 GB/s bidirectional |
| CPU | 2× AMD EPYC 9365 (Zen 5), 72코어 |
| RAM | 2.21TB DDR5 |
| CUDA | 13.1 / Driver 580.95.05 |
| PyTorch | nv25.12 커스텀 빌드 (B200 최적화) |
| FlashAttention | 2.7.4 |
| NCCL | 2.28.9 |

---

## 9. 주요 기술적 교훈

1. **SFT만으로는 반복 문제 미해결** — ORPO가 greedy 반복률 72.97% → 30.89%로 감소
2. **rep_penalty=1.2가 핵심** — sampling 시 반복 0% 달성
3. **byte_fallback 필수** — SentencePiece 토크나이저는 반드시 `byte_fallback=True` 설정
4. **TRL ORPO NaN 버그** — 라이브러리 레벨 패치 필요 (3중 방어)
5. **DDP+TE static_graph=False** — Transformer Engine과 DDP 혼용 시 주의
6. **NUMA 매핑** — GPU 0-3/4-7 각각 NUMA node 0/1에 바인딩해야 최적 성능

---

*보고서 작성: 2026-03-10*
