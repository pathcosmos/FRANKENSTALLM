# FRANKENSTALLM 3B v2 — GGUF 변환·배포 및 Ollama 평가 보고서

- **작성일**: 2026-03-09
- **대상**: byte-fallback 수정 적용 체크포인트 → GGUF 변환 → Ollama 배포 → 벤치마크

---

## 1. 요약

| 항목 | 내용 |
|------|------|
| **원인** | SentencePiece Unigram 토크나이저에 `byte_fallback` 미적용 → `\n` 등 미등록 문자 시 llama.cpp 크래시 |
| **조치** | 256개 byte-fallback 토큰 추가, 임베딩 64000→64256 리사이즈, GGUF 재변환, Q4_K_M 양자화 |
| **배포** | Ollama 모델 `frankenstallm-3b-v2:latest` (792 MB, Q4_K_M) |
| **뉴라인 검증** | ✅ 크래시 없이 `\n` 포함 프롬프트 처리 확인 |
| **Ollama 벤치마크** | 35개 테스트, 자동 채점 평균 46.7, 평균 TPS 142.5, TTFT 16.7 ms |

---

## 2. 파이프라인 단계

### 2.1 토크나이저·임베딩 수정

- **스크립트**: `scripts/fix_tokenizer_byte_fallback.py`
- **입력**: `outputs/hf_checkpoint-best`
- **출력**: `outputs/hf_checkpoint-best-fixed`
- **변경 사항**:
  - `tokenizer.json`: `byte_fallback=True`, `<0x00>`~`<0xFF>` 256개 토큰 추가
  - `config.json`: `vocab_size` 64000 → 64256
  - 임베딩 레이어 리사이즈 및 새 토큰 초기화 후 safetensors 저장

### 2.2 GGUF 변환 및 양자화

- **F16 GGUF**: `outputs/llama.cpp/convert_hf_to_gguf.py`  
  `outputs/hf_checkpoint-best-fixed` → `outputs/gguf/frankenstallm-3b-v2-f16.gguf`
- **Q4_K_M 양자화**: `outputs/llama.cpp/build/bin/llama-quantize`  
  → `outputs/gguf/frankenstallm-3b-v2-Q4_K_M.gguf` (약 792 MB)

### 2.3 Ollama 배포

- **Modelfile**: 로컬 GGUF 경로 `FROM` 지정 후 `ollama create`
- **모델 이름**: `frankenstallm-3b-v2:latest`

### 2.4 뉴라인 테스트

- **방법**: Ollama API로 `"첫 줄\n두 번째 줄\n세 번째 줄이라고 말해줘."` 프롬프트 전송
- **결과**: HTTP 200, `done: true`, 크래시 없음 → byte-fallback 수정 검증 완료

---

## 3. Ollama 벤치마크 결과 (frankenstallm-3b-v2)

- **실행**: `python eval/ollama_benchmark.py --models frankenstallm-3b-v2 --output-dir eval/results/frankenstallm-3b-v2`
- **일시**: 2026-03-09 23:24:22
- **총 테스트**: 35 (자동 채점 20 + 수동 검토 15)

### 3.1 전체 자동 채점 평균

| 모델 | Auto Avg |
|------|----------|
| frankenstallm-3b-v2 | **46.7** |

### 3.2 카테고리별 점수 (자동/수동)

| 카테고리 | 점수 | 비고 |
|----------|------|------|
| korean_nlu | 100.0 | 3 자동 / 2 수동 |
| korean_generation | manual | 5 수동 |
| reasoning | 50.0 | 4 자동 / 1 수동 |
| knowledge | 75.0 | 4 자동 / 1 수동 |
| code | 0.0 | 3 자동 |
| safety | 10.0 | 2 자동 / 1 수동 |
| instruction_following | 66.7 | 3 자동 |
| multilingual | manual | 3 수동 |
| repetition_resistance | 2.2 | 3 자동 (반복률 높음) |

### 3.3 지연 시간

| 지표 | 값 |
|------|-----|
| Avg TTFT (ms) | 16.7 |
| P50 TTFT (ms) | 15.8 |
| P95 TTFT (ms) | 26.2 |
| Avg TPS | 142.5 |
| P50 TPS | 142.7 |
| P95 TPS | 143.3 |

### 3.4 반복률 상세 (repetition_resistance)

| Test ID | Rep Rate | Unique/Total N-grams | Score |
|---------|----------|----------------------|-------|
| rep_01 | 73.76% | 122/465 | 0.0 |
| rep_02 | 59.72% | 255/633 | 0.0 |
| rep_03 | 46.70% | 226/424 | 6.6 |

- **원본 ORPO 평가** (HF 체크포인트, Greedy): 3-gram 반복률 30.89%, EOS 67%.  
  Ollama Q4_K_M + 벤치마크 프롬프트에서는 반복이 더 두드러짐.

### 3.5 결과 파일 위치

- **JSON**: `eval/results/frankenstallm-3b-v2/ollama_benchmark_results.json`
- **요약 MD**: `eval/results/frankenstallm-3b-v2/ollama_benchmark_summary.md`

---

## 4. 기존 ORPO 평가와의 연계

- **ORPO 종합 보고서**: `reports/2026-03-09_ORPO_EVALUATION_REPORT.md`
- **정량 스코어**: 63.7/100, 7/10 차원 통과, 최종 판정 **RETRY**
- **v2 배포본**은 동일 ORPO 체크포인트에서 byte-fallback만 수정·GGUF 변환한 버전이며,  
  ORPO 지표(예: preference accuracy, reward margin)는 기존 보고서와 동일한 체크포인트 기준으로 유지됨.

---

## 5. 아티팩트 경로 정리

| 용도 | 경로 |
|------|------|
| 수정된 HF 체크포인트 | `outputs/hf_checkpoint-best-fixed/` |
| F16 GGUF | `outputs/gguf/frankenstallm-3b-v2-f16.gguf` |
| Q4_K_M GGUF (Ollama 배포용) | `outputs/gguf/frankenstallm-3b-v2-Q4_K_M.gguf` |
| Ollama 벤치마크 결과 | `eval/results/frankenstallm-3b-v2/` |
| Byte-fallback 수정 스크립트 | `scripts/fix_tokenizer_byte_fallback.py` |

---

*이 보고서는 GGUF 변환·Ollama 배포 및 Ollama 벤치마크 결과를 정리한 문서입니다.*
