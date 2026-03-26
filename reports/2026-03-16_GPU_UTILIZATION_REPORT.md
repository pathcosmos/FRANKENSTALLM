# GPU 활용 현황 보고서

- **작성일**: 2026-03-16
- **프로젝트명**: FRANKENSTALLM — 한국어 3B 대규모 언어 모델 개발
- **사용 기간**: 2026-02-25 ~ 2026-03-10 (14일)

---

## 1. GPU 하드웨어 사양

| 항목 | 사양 |
|------|------|
| GPU 모델 | NVIDIA B200 SXM |
| GPU 수량 | 8기 |
| GPU당 VRAM | 183 GB HBM3e |
| 총 VRAM | ~1.47 TB |
| GPU당 FP8 성능 | 2,250 TFLOPS |
| GPU당 BF16 성능 | 1,125 TFLOPS |
| GPU 간 인터커넥트 | NVLink 5.0 (NV18), 양방향 900 GB/s |
| 호스트 CPU | AMD EPYC 9365 (Zen 5) × 2소켓, 72코어 |
| 시스템 메모리 | DDR5 2.21 TB |
| CUDA 버전 | 13.1 |
| GPU 드라이버 | 580.95.05 |
| PyTorch | 2.10.0a0+nv25.12 (NVIDIA B200 최적화 커스텀 빌드) |
| FlashAttention | 2.7.4.post1 |
| NCCL | 2.28.9 |

---

## 2. GPU 활용 요약

### 2.1 전체 학습 단계별 GPU 사용 현황

| 단계 | 기간 | 소요 시간 | GPU 사용량 | GPU-hours |
|------|------|----------|-----------|-----------|
| Phase 0: 환경 구축 및 FP8 검증 | 02/25 ~ 03/02 | ~24시간 | 8 GPU | 192 |
| Phase 1: 3B 사전학습 (Pretrain) | 03/02 ~ 03/05 | 62.94시간 | 8 GPU | 503.5 |
| Phase 2: SFT 지시학습 | 03/05 ~ 03/06 | 15.7시간 | 8 GPU | 125.6 |
| Phase 3: ORPO 선호도 정렬 | 03/08 ~ 03/09 | ~2시간 | 8 GPU | 16 |
| Phase 4: GGUF 변환 및 배포 | 03/09 ~ 03/10 | ~4시간 | 1~2 GPU | 6 |
| **합계** | | **~109시간** | | **~843 GPU-hours** |

### 2.2 핵심 GPU 활용 지표

| 지표 | Phase 1 (Pretrain) | Phase 2 (SFT) | Phase 3 (ORPO) |
|------|-------------------|---------------|----------------|
| GPU당 VRAM 사용량 | 48.3 GB (26.4%) | 24.2 GB (13.2%) | ~40 GB (21.9%) |
| 처리 속도 (전체) | 292,000 tok/s | — | — |
| 처리 속도 (GPU당) | 36,500 tok/s | — | — |
| MFU (Model FLOPs Utilization) | 33.5% | — | — |
| 학습 정밀도 | BF16 + FP8 | BF16 | BF16 |
| 병렬화 전략 | DDP (8 GPU) | DDP (8 GPU) | DDP (8 GPU) |

---

## 3. GPU 최적화 기술 적용 내역

### 3.1 B200 전용 최적화

| 최적화 항목 | 적용 내용 | 효과 |
|------------|----------|------|
| FP8 Tensor Core | `torch.float8_e4m3fn` 네이티브 활용 | BF16 대비 연산 처리량 2배 |
| FlashAttention-2 | B200 네이티브 지원, GQA 결합 | VRAM 20% 절감 (60.4→48.3 GB) |
| NVLink 5.0 | 8 GPU 메시 토폴로지, 900 GB/s | DDP 통신 병목 최소화 |
| NCCL NVLS | `gradient_as_bucket_view=True` | Gradient AllReduce 오버헤드 감소 |
| NUMA Affinity | GPU 0~3→NUMA 0, GPU 4~7→NUMA 1 | CPU-GPU 메모리 접근 지연 최소화 |

### 3.2 소프트웨어 최적화

| 최적화 항목 | 적용 내용 | 효과 |
|------------|----------|------|
| Grouped-Query Attention (GQA) | KV heads 4개 (16:4 비율) | KV 캐시 메모리 75% 절감 |
| Gradient Checkpointing | ORPO 학습 시 적용 | 대용량 배치 학습 가능 |
| Dynamic Batch Collation | SFT 시 동적 패딩 | 패딩 낭비 감소, 학습 효율 향상 |
| Mixed Precision (BF16/FP8) | 사전학습 전 과정 | 메모리 절감 + 연산 가속 |
| DDP Static Graph 최적화 | Transformer Engine 호환 설정 | 안정적 멀티 GPU 학습 |

---

## 4. 단계별 학습 상세

### 4.1 Phase 1: 사전학습 (Pretraining)

| 항목 | 값 |
|------|-----|
| 모델 규모 | 3B 파라미터 (실측 2.39B) |
| 아키텍처 | LlamaForCausalLM (24 layers, 16 heads, hidden 2048) |
| 학습 스텝 수 | 57,000 steps |
| 최종 Loss | 1.466 (수렴) |
| 총 학습 토큰 | ~38.5B tokens |
| 학습 데이터 | 640GB+ (한국어 웹, 공공 코퍼스, 교육 콘텐츠 등) |
| 처리 속도 | 292,000 tokens/sec (8 GPU 합산) |
| 소요 시간 | 62.94시간 |
| 사고/장애 | 0건 |

### 4.2 Phase 2: SFT (Supervised Fine-Tuning)

| 항목 | 값 |
|------|-----|
| 학습 스텝 수 | 25,500 steps (Early Stopping) |
| 최저 검증 Loss | 1.8851 |
| 학습 데이터 | 24개 소스, 2,439,397 샘플 (7.48 GB) |
| 소요 시간 | 15.7시간 |
| 사고/장애 | 0건 |

### 4.3 Phase 3: ORPO (선호도 정렬 학습)

| 항목 | 값 |
|------|-----|
| HP Sweep | 6개 설정 탐색 후 최적 설정 선정 |
| 학습 스텝 수 | 9,997 steps (조기 수렴) |
| Preference Accuracy | 76.02% |
| 학습 데이터 | 795,468 preference pairs (7.9 GB) |
| 소요 시간 | ~2시간 |
| 사고/장애 | 0건 (TRL 라이브러리 NaN 버그 패치 해결) |

---

## 5. GPU 활용 성과

### 5.1 모델 성능 결과

| 평가 항목 | 결과 |
|----------|------|
| 종합 평가 점수 | 63.7/100 (10차원 중 7개 PASS) |
| Perplexity (지식 보존) | PASS (forgetting 0.9%) |
| KoBEST 한국어 벤치마크 | 52.75% |
| Preference Accuracy | 76.02% |
| 반복 생성률 개선 | 72.97% → 30.89% (↓42pp) |
| EOS 종료율 | 0% → 67% |
| Calibration 정확도 | 67.99% |

### 5.2 배포 성과

| 항목 | 내용 |
|------|------|
| HuggingFace 공개 | https://huggingface.co/pathcosmos/frankenstallm |
| 모델 포맷 | SafeTensors (4.76GB) + GGUF 6종 (757MB~6.0GB) |
| 경량 모델 (Q4_K_M) | 757MB — 일반 PC에서 구동 가능 |
| Ollama 추론 속도 | 142.5 tok/s (Q4_K_M 기준) |
| TTFT (첫 토큰 지연) | 16.7ms |

### 5.3 데이터 처리량

| 구분 | 데이터량 |
|------|---------|
| 사전학습 코퍼스 | 640GB+ (한국어 웹, 공공 데이터, 교육 콘텐츠) |
| SFT 지시학습 데이터 | 7.48 GB (24개 소스, 244만 샘플) |
| ORPO 선호도 데이터 | 7.9 GB (79.5만 쌍) |
| 총 학습 토큰 | ~38.5B tokens (사전학습 기준) |

---

## 6. GPU 자원 효율성 분석

### 6.1 GPU 가동률

```
전체 사용 기간: 14일 (2026-02-25 ~ 2026-03-10)
실제 학습 가동 시간: ~109시간
총 GPU-hours: ~843 GPU-hours (8 GPU × 평균 가동)
일평균 GPU 가동: ~7.8시간/일

주요 비가동 사유:
  - 환경 셋업 및 라이브러리 설정
  - HP sweep 설계 및 실험 설정
  - 중간 평가 및 결과 분석
  - 데이터 전처리 및 파이프라인 구성
  - 코드 디버깅 및 버그 수정 (TRL NaN 패치 등)
```

### 6.2 GPU 메모리 활용 효율

| 단계 | GPU당 VRAM 사용 | 전체 VRAM 사용 | 활용률 |
|------|----------------|---------------|--------|
| Phase 1 (Pretrain) | 48.3 GB / 183 GB | 386.4 GB / 1,464 GB | 26.4% |
| Phase 2 (SFT) | 24.2 GB / 183 GB | 193.6 GB / 1,464 GB | 13.2% |
| Phase 3 (ORPO) | ~40 GB / 183 GB | ~320 GB / 1,464 GB | 21.9% |

> 3B 모델은 B200 8기 환경에서 VRAM 여유가 충분하여 안정적으로 학습 완료.
> 향후 13B~70B 규모 모델 학습 시 VRAM 전체를 활용하는 FSDP 전략 적용 예정.

### 6.3 연산 효율 (MFU)

| 항목 | 값 |
|------|-----|
| 이론 최대 처리량 (BF16) | 9,000 TFLOPS (1,125 × 8) |
| 실측 유효 처리량 | ~3,015 TFLOPS |
| MFU (Model FLOPs Utilization) | **33.5%** |
| 업계 평균 MFU (유사 규모) | 30~45% |

> MFU 33.5%는 DDP 기반 3B 모델 학습에서 정상 범위.
> FSDP + 커널 퓨전 최적화 적용 시 47%까지 개선 가능 (계획 수립 완료).

---

## 7. 주요 기술적 성과 및 해결 사례

| # | 이슈 | 해결 방법 | GPU 활용 관련성 |
|---|------|----------|---------------|
| 1 | B200 FP8 Tensor Core 호환 | `torch.float8_e4m3fn` 직접 통합 | FP8로 연산 처리량 2배 향상 |
| 2 | FlashAttention GQA 연동 | GQA + FA2 커널 직접 구현 | VRAM 20% 절감 |
| 3 | DDP + Transformer Engine 충돌 | `static_graph=False` 설정 | 멀티 GPU 안정성 확보 |
| 4 | NCCL 통신 최적화 | `gradient_as_bucket_view` 적용 | GPU 간 통신 오버헤드 감소 |
| 5 | NUMA 토폴로지 최적화 | GPU-NUMA 노드 매핑 | CPU-GPU 데이터 전송 최적화 |
| 6 | TRL ORPO NaN 전파 버그 | 3중 패치 (nan_to_num, log_softmax, inf masking) | 학습 안정성 확보 |

---

## 8. 향후 GPU 활용 계획

| 계획 | 예상 GPU 사용 | 목표 |
|------|-------------|------|
| MFU 최적화 (33.5% → 47%) | 동일 GPU, 소프트웨어 개선 | 학습 속도 40% 향상 |
| 13B 모델 학습 (FSDP) | 8 GPU 전량, VRAM 70%+ 활용 | 더 높은 GPU 활용률 |
| 한국어 코딩 LLM 개발 | 8 GPU, 예상 200+ 시간 | 도메인 특화 모델 |
| Mamba-Transformer 하이브리드 | 8 GPU, 검증 단계 | 차세대 아키텍처 실험 |

---

## 9. 비용 효율성 (참고)

| 항목 | 값 |
|------|-----|
| 총 GPU-hours | ~843 시간 |
| 클라우드 환산 비용 (B200 기준) | ~$4,700 (~680만원, RunPod $5.58/GPU/hr) |
| 클라우드 환산 비용 (H100 동일 작업량) | ~$1,340 (~194만원, RunPod $1.99/GPU/hr) |
| 처리 토큰 대비 비용 | ~$0.12 / 1M tokens (B200 기준) |

> 온프레미스 GPU 자원을 활용하여 클라우드 대비 비용 절감 효과 달성.

---

*본 보고서는 FRANKENSTALLM 프로젝트의 GPU 활용 현황을 정리한 것으로,*
*정부 과제 사용완료 보고서 작성 시 참고 자료로 활용할 수 있습니다.*
