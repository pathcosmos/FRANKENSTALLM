# 업로드 대상 모델 파일 선정 (재현·배포용)

Hugging Face 등에 올릴 때 **꼭 넣을 것**과 **선택/제외**를 정리한 문서입니다.

---

## 1. 필수 업로드 (1개 세트)

재현·배포에 **반드시** 포함하는 것을 권장합니다.

| 경로 | 용량 | 설명 |
|------|------|------|
| **`outputs/hf_checkpoint-best-fixed/`** | **약 4.5 GB** | **ORPO 최종 모델 (byte-fallback 수정)**. Transformers 로드·GGUF 변환·Ollama 배포의 기준 체크포인트. |

**포함 파일:**
- `model.safetensors` — 가중치
- `config.json` — 모델 설정 (vocab 64256)
- `tokenizer.json`, `tokenizer_config.json`, `tokenizer.model` — 토크나이저
- `generation_config.json` — 생성 기본 설정
- `README.md` — 모델 카드 (평가 요약 포함)

→ **이 디렉터리 하나만 올려도** `from_pretrained(...)`, GGUF 변환, 재학습 연계가 가능합니다.

---

## 2. 업로드 포함 (GGUF — 로컬/엣지 배포)

업로드 스크립트에서 **HF 체크포인트와 함께** 위 두 GGUF를 올리도록 되어 있습니다.

| 경로 | 용량 | 용도 |
|------|------|------|
| **`outputs/gguf/frankenstallm-3b-v2-f16.gguf`** | **약 2.3 GB** | F16 풀정밀. GGUF 변환 직후 단계. |
| **`outputs/gguf/frankenstallm-3b-v2-Q4_K_M.gguf`** | **약 757 MB** | Ollama·로컬 추론용 양자화. |

**v1 GGUF (올리지 않아도 됨):**
- `frankenstallm-3b-f16.gguf`, `frankenstallm-3b-Q4_K_M.gguf`, `frankenstallm-3b-Q8_0.gguf`  
  → byte-fallback 미적용, v2로 대체되었으므로 **업로드 제외** 권장.

---

## 3. 업로드 제외 권장

재현에 꼭 필요하지 않거나, 중복·용량 때문에 올리지 않는 것이 좋은 것들입니다.

| 구분 | 경로/대상 | 이유 |
|------|-----------|------|
| HF 이전 버전 | `outputs/hf_checkpoint-best/` | byte-fallback 미적용. **hf_checkpoint-best-fixed**로 대체. |
| 평가용 HF 복사본 | `eval/outputs/hf_3b_base/`, `eval/outputs/hf_3b_sft_best/`, `eval/outputs/*/hf_3b_checkpoint-*` | 평가 파이프라인용. 재현 시 HF에서 받은 모델로 대체 가능. |
| 학습 중간 체크포인트 | `checkpoints/korean_3b_orpo_v1/checkpoint-*` | ORPO 학습 step별. 최종만 **hf_checkpoint-best-fixed**로 올리면 됨. |
| llama.cpp vocab 샘플 | `outputs/llama.cpp/models/ggml-vocab-*.gguf` | llama.cpp 기본 vocab. 우리 모델 업로드와 무관. |
| 데이터 .bin | `data/*.bin` | 1.2TB급. HF 업로드 부적합. 스크립트·설명만 올리기. |

---

## 4. 요약 표

| 우선순위 | 대상 | 용량 | 비고 |
|----------|------|------|------|
| **1** | `outputs/hf_checkpoint-best-fixed/` 전체 | ~4.5 GB | **반드시 업로드** |
| **2** | `outputs/gguf/frankenstallm-3b-v2-f16.gguf` | ~2.3 GB | 업로드 스크립트에 포함 |
| **3** | `outputs/gguf/frankenstallm-3b-v2-Q4_K_M.gguf` | ~757 MB | 업로드 스크립트에 포함 |
| 제외 | `outputs/hf_checkpoint-best/`, v1 GGUF, eval/outputs 내 체크포인트, checkpoints/ 학습 체크포인트, data/*.bin | — | 위 “제외 권장” 참고 |

---

## 5. 재현 시나리오별로 보면

- **Transformers로 추론/파인튜닝**  
  → **hf_checkpoint-best-fixed** 만 있으면 됨.

- **GGUF/Ollama로 배포**  
  → **hf_checkpoint-best-fixed** 올려두고, 문서에 `scripts/fix_tokenizer_byte_fallback.py` + `convert_hf_to_gguf.py` + `llama-quantize` 순서만 적어두면 재현 가능.  
  → **frankenstallm-3b-v2-f16.gguf**, **frankenstallm-3b-v2-Q4_K_M.gguf** 둘 다 업로드 스크립트에 포함됨.

- **학습부터 재현**  
  → 모델 파일은 **hf_checkpoint-best-fixed** (또는 공개된 동일 세트) 하나만 명시하고, 데이터·학습 스크립트는 별도 문서/저장소로 안내.

정리하면, **업로드해야 할 모델은 `outputs/hf_checkpoint-best-fixed/` 한 세트**와 **GGUF v2 (f16, Q4_K_M) 두 파일**이 업로드 스크립트에 포함됩니다.
