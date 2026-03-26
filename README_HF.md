---
library_name: transformers
license: apache-2.0
language:
  - ko
  - en
model_type: llama
tags:
  - 3b
  - korean
  - from-scratch
  - orpo
  - instruction-tuned
  - preference-aligned
  - fp8
  - b200
  - gguf
datasets:
  - cc100
  - allenai/c4
  - heegyu/orca-math-korean-preference-cleaned
  - nayohan/preference-collection-ko-full
  - maywell/ko_Ultrafeedback_binarized
  - HuggingFaceTB/cosmopedia
  - wikimedia/wikipedia
pipeline_tag: text-generation
model-index:
  - name: FRANKENSTALLM-3B
    results:
      - task:
          type: text-generation
        dataset:
          type: kobest
          name: KoBEST (0-shot)
        metrics:
          - name: Average
            type: accuracy
            value: 52.75
          - name: COPA
            type: accuracy
            value: 63.9
          - name: HellaSwag-KO
            type: accuracy
            value: 38.0
          - name: SentiNeg
            type: accuracy
            value: 62.5
          - name: BoolQ
            type: accuracy
            value: 50.6
          - name: WiC
            type: accuracy
            value: 48.8
      - task:
          type: text-generation
        dataset:
          type: haerae
          name: HAE-RAE (0-shot)
        metrics:
          - name: Average
            type: accuracy
            value: 21.81
      - task:
          type: text-generation
        dataset:
          type: piqa
          name: PIQA (0-shot)
        metrics:
          - name: Accuracy
            type: accuracy
            value: 59.9
      - task:
          type: text-generation
        dataset:
          type: ai2_arc
          name: ARC-Easy (0-shot)
        metrics:
          - name: Accuracy
            type: accuracy
            value: 36.0
---

# FRANKENSTALLM 3B

> **⚠️ v2 모델 교체 공지 (2026-03-26)**
>
> v2 GGUF 및 safetensors 파일이 변환 과정의 오류로 **1.2B 모델(hidden_size=2048, 24 layers)**로 잘못 배포되었습니다.
> 2026-03-26에 올바른 **3B ORPO 체크포인트(hidden_size=3072, 28 layers, vocab_size=64256, byte-fallback 적용)**로 교체 완료했습니다.
> 이전에 다운로드한 v2 파일이 있다면 재다운로드를 권장합니다.


> **한국어 3B LLM을 처음부터 직접 만들었습니다 — 토크나이저 학습부터 사전학습, SFT, ORPO까지, 8× NVIDIA B200 GPU 위에서.**

| | |
|---|---|
| **개발자** | [pathcosmos](https://huggingface.co/pathcosmos) |
| **파라미터** | ~24억 (weight tying 적용, 3B급) |
| **언어** | 한국어 (주), 영어 (부) |
| **라이선스** | Apache 2.0 |
| **학습** | 3단계: 사전학습 → SFT → ORPO |
| **하드웨어** | 8× NVIDIA B200 (FP8), 총 ~86시간 |

---

## 빠른 시작

### Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "pathcosmos/frankenstallm"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)

inputs = tokenizer(
    "한국의 전통 음식 중 김치에 대해 설명해주세요.",
    return_tensors="pt"
).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        do_sample=True,
        temperature=0.7,
        repetition_penalty=1.2,  # 권장
        top_p=0.9,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id,
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Ollama (GGUF)

```bash
# GGUF + Modelfile 다운로드
huggingface-cli download pathcosmos/frankenstallm \
  gguf/frankenstallm-3b-v2-Q4_K_M.gguf \
  gguf/Modelfile.3b-v2-Q4_K_M \
  --local-dir ./frankenstallm

# Modelfile 내 FROM 경로 수정 후 생성
ollama create frankenstallm -f ./frankenstallm/gguf/Modelfile.3b-v2-Q4_K_M

# 실행
ollama run frankenstallm
```

---


## 파일 다운로드 링크

### 모델 파일

| 파일 | 크기 | 설명 | 다운로드 |
|------|------|------|----------|
| [`model.safetensors`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/model.safetensors) | 5.7 GB | HF Transformers 네이티브 (3B ORPO, byte-fallback) | [Download](https://huggingface.co/pathcosmos/frankenstallm/resolve/main/model.safetensors) |
| [`config.json`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/config.json) | 1 KB | 모델 설정 (hidden=3072, 28L, vocab=64256) | [Download](https://huggingface.co/pathcosmos/frankenstallm/resolve/main/config.json) |
| [`tokenizer.json`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/tokenizer.json) | 4 MB | 토크나이저 (SentencePiece Unigram) | [Download](https://huggingface.co/pathcosmos/frankenstallm/resolve/main/tokenizer.json) |
| [`tokenizer.model`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/tokenizer.model) | 1.4 MB | SentencePiece 모델 (GGUF 변환용) | [Download](https://huggingface.co/pathcosmos/frankenstallm/resolve/main/tokenizer.model) |
| [`sampling_config.json`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/sampling_config.json) | 1 KB | 권장 샘플링 파라미터 | [Download](https://huggingface.co/pathcosmos/frankenstallm/resolve/main/sampling_config.json) |

### GGUF (Ollama / llama.cpp)

| 파일 | 크기 | 양자화 | 다운로드 |
|------|------|--------|----------|
| [`frankenstallm-3b-v2-Q4_K_M.gguf`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/gguf/frankenstallm-3b-v2-Q4_K_M.gguf) | 1.8 GB | **Q4_K_M (권장)** | [Download](https://huggingface.co/pathcosmos/frankenstallm/resolve/main/gguf/frankenstallm-3b-v2-Q4_K_M.gguf) |
| [`frankenstallm-3b-v2-Q8_0.gguf`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/gguf/frankenstallm-3b-v2-Q8_0.gguf) | 3.0 GB | Q8_0 (고품질) | [Download](https://huggingface.co/pathcosmos/frankenstallm/resolve/main/gguf/frankenstallm-3b-v2-Q8_0.gguf) |
| [`frankenstallm-3b-v2-f16.gguf`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/gguf/frankenstallm-3b-v2-f16.gguf) | 5.7 GB | F16 (무손실) | [Download](https://huggingface.co/pathcosmos/frankenstallm/resolve/main/gguf/frankenstallm-3b-v2-f16.gguf) |
| [`Modelfile.3b-v2-Q4_K_M`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/gguf/Modelfile.3b-v2-Q4_K_M) | 1 KB | Ollama Modelfile (Q4) | [Download](https://huggingface.co/pathcosmos/frankenstallm/resolve/main/gguf/Modelfile.3b-v2-Q4_K_M) |
| [`Modelfile.3b-v2-Q8_0`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/gguf/Modelfile.3b-v2-Q8_0) | 1 KB | Ollama Modelfile (Q8) | [Download](https://huggingface.co/pathcosmos/frankenstallm/resolve/main/gguf/Modelfile.3b-v2-Q8_0) |

> v1 GGUF (byte-fallback 미적용)도 `gguf/frankenstallm-3b-*.gguf`로 제공되지만, **v2 사용을 권장**합니다.

### 학습 데이터 (SFT / ORPO 재현용)

| 파일 | 크기 | 용도 | 다운로드 |
|------|------|------|----------|
| [`train_filtered.jsonl`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/data/sft_combined/train_filtered.jsonl) | 7.5 GB | SFT 학습 데이터 (24개 소스, 240만 샘플, 필터링 완료) | [Download](https://huggingface.co/pathcosmos/frankenstallm/resolve/main/data/sft_combined/train_filtered.jsonl) |
| [`val_filtered.jsonl`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/data/sft_combined/val_filtered.jsonl) | 157 MB | SFT 검증 데이터 | [Download](https://huggingface.co/pathcosmos/frankenstallm/resolve/main/data/sft_combined/val_filtered.jsonl) |
| [`combined_preference.jsonl`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/data/preference/combined_preference.jsonl) | 2.6 GB | ORPO 학습 데이터 (7개 소스 통합, 63만 쌍) | [Download](https://huggingface.co/pathcosmos/frankenstallm/resolve/main/data/preference/combined_preference.jsonl) |

<details>
<summary>ORPO Preference 데이터 개별 소스 (7종)</summary>

| 파일 | 크기 | 다운로드 |
|------|------|----------|
| [`nayohan_preference-collection-ko-full.jsonl`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/data/preference/nayohan_preference-collection-ko-full.jsonl) | 4.9 GB | [Download](https://huggingface.co/pathcosmos/frankenstallm/resolve/main/data/preference/nayohan_preference-collection-ko-full.jsonl) |
| [`heegyu_orca-math-korean-preference-cleaned.jsonl`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/data/preference/heegyu_orca-math-korean-preference-cleaned.jsonl) | 1.6 GB | [Download](https://huggingface.co/pathcosmos/frankenstallm/resolve/main/data/preference/heegyu_orca-math-korean-preference-cleaned.jsonl) |
| [`kuotient_orca-math-korean-dpo-pairs.jsonl`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/data/preference/kuotient_orca-math-korean-dpo-pairs.jsonl) | 750 MB | [Download](https://huggingface.co/pathcosmos/frankenstallm/resolve/main/data/preference/kuotient_orca-math-korean-dpo-pairs.jsonl) |
| [`maywell_ko_Ultrafeedback_binarized.jsonl`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/data/preference/maywell_ko_Ultrafeedback_binarized.jsonl) | 394 MB | [Download](https://huggingface.co/pathcosmos/frankenstallm/resolve/main/data/preference/maywell_ko_Ultrafeedback_binarized.jsonl) |
| [`tellang_yeji-preference-ko-v1.jsonl`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/data/preference/tellang_yeji-preference-ko-v1.jsonl) | 171 MB | [Download](https://huggingface.co/pathcosmos/frankenstallm/resolve/main/data/preference/tellang_yeji-preference-ko-v1.jsonl) |
| [`jojo0217_korean_rlhf_dataset.jsonl`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/data/preference/jojo0217_korean_rlhf_dataset.jsonl) | 137 MB | [Download](https://huggingface.co/pathcosmos/frankenstallm/resolve/main/data/preference/jojo0217_korean_rlhf_dataset.jsonl) |
| [`lemon-mint_korean-realqa-reasoning-v01-preference.jsonl`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/data/preference/lemon-mint_korean-realqa-reasoning-v01-preference.jsonl) | 58 MB | [Download](https://huggingface.co/pathcosmos/frankenstallm/resolve/main/data/preference/lemon-mint_korean-realqa-reasoning-v01-preference.jsonl) |

</details>

### 데이터 파이프라인 스크립트

| 파일 | 설명 |
|------|------|
| [`prepare_sft_data.py`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/data/prepare_sft_data.py) | HF 데이터셋 → JSONL 정규화 (Alpaca 포맷) |
| [`filter_sft_v2.py`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/data/filter_sft_v2.py) | SFT 품질 필터링 (중복 제거, 반복률 필터) |
| [`prepare_preference_combined.py`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/data/prepare_preference_combined.py) | Preference 데이터 통합 (DPO/ORPO용) |
| [`tokenize_extra.py`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/data/tokenize_extra.py) | 대용량 데이터 병렬 토크나이징 |
| [`sft_dataset.py`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/data/sft_dataset.py) | SFT 데이터셋 로더 (Alpaca/대화 포맷) |
| [`dataset.py`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/data/dataset.py) | 사전학습 데이터셋 로더 (memmap .bin) |
| [`build_korean_dataset.sh`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/data/build_korean_dataset.sh) | 한국어 데이터 전체 파이프라인 |

### Phase별 보고서

| 보고서 | 내용 |
|--------|------|
| [`PROJECT_COMPLETION_REPORT`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/reports/2026-03-10_PROJECT_COMPLETION_REPORT.md) | 프로젝트 최종 완료 보고서 |
| [`ORPO_EVALUATION_REPORT`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/reports/2026-03-09_ORPO_EVALUATION_REPORT.md) | ORPO 10차원 종합 평가 |
| [`ORPO_TRAINING_JOURNEY`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/reports/2026-03-08_ORPO_TRAINING_JOURNEY.md) | ORPO 학습 여정 (HP sweep, 디버깅) |
| [`SFT_COMPLETION_AND_EVAL`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/reports/2026-03-06_3B_SFT_COMPLETION_AND_EVAL_SUMMARY.md) | SFT 완료 및 평가 |
| [`3B_BASE_EVALUATION`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/reports/2026-03-05_3B_BASE_EVALUATION_REPORT.md) | 사전학습 베이스 모델 평가 |
| [`Phase0_Optimization`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/reports/2026-03-02_0200_FRANKENSTALLM_phase0_optimization_report.md) | FP8 최적화 보고서 |


---

## 모델 특징

- **처음부터 만든 한국어 토크나이저**: SentencePiece Unigram, 64K 어휘, 한국어 문자 커버리지 99.95%
- **3단계 학습 파이프라인**: 사전학습 (57K 스텝, ~600억 토큰) → SFT (25.5K 스텝, 240만 샘플) → ORPO (10K 스텝, 63만 선호도 쌍)
- **B200 FP8 네이티브 학습**: TransformerEngine MXFP8 — BF16 대비 이론적 2배 처리량
- **GGUF 배포 지원**: Q4_K_M (1.8GB), Q8_0 (3.0GB), F16 (5.7GB) + Ollama Modelfile 제공

---

## 아키텍처

| 구성 요소 | 값 |
|-----------|-----|
| 구조 | Decoder-only Transformer (LLaMA 스타일) |
| Hidden size | 3,072 |
| 레이어 수 | 28 |
| 어텐션 헤드 | 24 |
| KV 헤드 | 8 (GQA 3:1) |
| FFN 차원 | 8,192 (SwiGLU) |
| 어휘 크기 | 64,256 (byte-fallback 적용) |
| 컨텍스트 길이 | 4,096 (학습 시 2,048) |
| 위치 인코딩 | RoPE (θ=500,000) |
| 정규화 | Pre-norm RMSNorm |
| 어텐션 구현 | FlashAttention-2 |
| 정밀도 | FP8 (TransformerEngine MXFP8) |
| Weight tying | 적용 (embedding ↔ lm_head) |

---

## 학습 파이프라인

### Phase 1: 사전학습

| 항목 | 값 |
|------|-----|
| 스텝 수 | 57,000 |
| 최종 loss | 1.466 |
| 학습 토큰 | ~600억 (385억 고유 × ~1.5 에폭) |
| 소요 시간 | ~63시간 |
| 데이터 | CC-100 KO, HPLT KO, C4 KO, 나무위키, 위키피디아 KO, Cosmopedia (EN) |
| 배치 크기 | 5 × 8 GPU × 8 accum × 2,048 seq = ~65만 토큰/스텝 |

### Phase 2: SFT (지도 미세조정)

| 항목 | 값 |
|------|-----|
| 스텝 수 | 25,500 (77.3% 지점에서 조기 종료) |
| 최적 val_loss | 1.8851 (step 23,000) |
| 소요 시간 | ~15.5시간 |
| 데이터 | 24개 소스, 243만 9,397 샘플 (7.48 GB) |
| 구성 | SFT 70% + 사전학습 리플레이 30% (치명적 망각 방지) |
| 지식 망각률 | 0.9% (19개 데이터셋 기준) |

### Phase 3: ORPO (선호도 최적화)

| 항목 | 값 |
|------|-----|
| 스텝 수 | 9,997 (조기 수렴) |
| 최적 eval_loss | 1.625 |
| 선호도 정확도 | 76.02% |
| 보상 마진 | 0.6100 |
| 소요 시간 | ~7시간 |
| 데이터 | 한국어 HF 데이터셋 7종, ~63만 선호도 쌍 |
| 하이퍼파라미터 | beta=0.25, lr=1.2e-5, eff_batch=128 |

**총 학습 시간: 8× B200에서 약 86시간**

---

## 벤치마크

### 학습 단계별 성능 변화 (Base → SFT → ORPO)

| 벤치마크 | Base | SFT | ORPO | 변화 (Base→ORPO) |
|-----------|:----:|:---:|:----:|:---:|
| **KoBEST 평균 (0-shot)** | 43.7% | 43.3% | **52.8%** | **+9.1pp** |
| KoBEST COPA | 49.3% | 48.6% | **63.9%** | +14.6pp |
| KoBEST HellaSwag-KO | 21.6% | 19.8% | **38.0%** | +16.4pp |
| KoBEST SentiNeg | 48.6% | 49.1% | **62.5%** | +13.9pp |
| KoBEST BoolQ | 50.3% | 50.1% | 50.6% | +0.3pp |
| PIQA | 52.5% | 52.6% | **59.9%** | +7.3pp |
| ARC-Easy | 25.6% | 25.9% | **36.0%** | +10.4pp |
| HAE-RAE | 19.7% | 19.9% | 21.8% | +2.1pp |
| HellaSwag EN | 26.2% | 26.1% | 29.2% | +3.0pp |
| Greedy 3-gram 반복률 | 61.0% | 73.0% | **30.9%** | -30.1pp |
| EOS 종료율 | 0% | 60% | **67%** | +67pp |
| PPL 망각률 | — | 0.9% | 4.1% | 15% 이내 ✅ |

### 3B급 모델 비교 (Ollama, 35개 테스트)

| 모델 | 파라미터 | 한국어 NLU | 지식 | 지시 수행 | 추론 | 평균 점수 |
|-------|:------:|:----------:|:----:|:---------:|:----:|:---------:|
| Qwen 2.5 3B | 3B | 100.0 | 20.8 | 55.6 | 62.5 | **63.4** |
| Phi-4 Mini | 3.8B | 66.7 | 29.2 | 33.3 | **87.5** | 60.6 |
| **FRANKENSTALLM 3B** | **3B** | **100.0** | **75.0** | **66.7** | 50.0 | 46.7 |

> FRANKENSTALLM은 **한국어 NLU** (Qwen과 동률), **한국어 지식** (75.0 vs 20.8/29.2), **지시 수행** (66.7 vs 55.6/33.3)에서 앞섭니다.

### 추론 속도 (Ollama, Q4_K_M)

| 모델 | 평균 TTFT | TPS | 비고 |
|-------|:--------:|:---:|------|
| **FRANKENSTALLM 3B** | **16.7ms** | **142.5** | 가장 빠름 |
| Phi-4 Mini 3.8B | 25.6ms | 100.4 | |
| Qwen 2.5 3B | 28.2ms | 93.8 | |

### Perplexity 보존율 (ORPO 지식 유지)

| 데이터셋 | Base PPL | ORPO PPL | 망각률 |
|---------|:--------:|:--------:|:------:|
| Korean C4 | 5.72 | 5.87 | +2.7% |
| Korean Wiki | 11.84 | 12.21 | +3.2% |
| 최대 망각률 | — | — | 4.1% ✅ |

---

## 학습 데이터

### 사전학습 (~385억 토큰)

| 분류 | 소스 | 추정 토큰 수 |
|------|------|:-----------:|
| 한국어 웹 크롤 | C4 KO, CC-100 KO, HPLT KO | ~172억 |
| 한국어 백과사전 | 위키피디아 KO, 나무위키 (2개 버전) | ~28억 |
| 영어 교육 | Cosmopedia (Stories, Web, Stanford, WikiHow, OpenStax, Khan) | ~57억 |
| 영어 수학·과학 | AutoMathText, OpenWebMath, Proof-Pile-2 | ~85억 |
| 코드 | StarCoder (필터링) | ~43억 |

### SFT (240만 샘플, 24개 소스)

| 영역 | 비율 | 주요 데이터셋 |
|------|:----:|-------------|
| 추론/CoT | 38% | reasoning_r1_1.4m, magpie_reasoning |
| 한국어 지시문 | 23% | korean_instruction_mix, open_korean_instructions, kullm_v2 |
| 영어 일반 | 16% | openhermes_2.5, ultrachat_200k |
| 수학 | 12% | NuminaMath-CoT, orca-math-ko |
| 대화/코드/기타 | 11% | smol-koreantalk, Evol-Instruct-Code-80k-ko |

### ORPO (~63만 선호도 쌍, 7개 소스)

| 데이터셋 | 용량 | 영역 |
|---------|:----:|------|
| nayohan/preference-collection-ko-full | 4.9GB | 일반 선호도 |
| heegyu/orca-math-korean-preference-cleaned | 1.6GB | 수학 추론 |
| kuotient/orca-math-korean-dpo-pairs | 750MB | 수학 DPO |
| maywell/ko_Ultrafeedback_binarized | 394MB | 피드백 정렬 |
| tellang/yeji-preference-ko-v1 | 171MB | 일반 선호도 |
| jojo0217/korean_rlhf_dataset | 137MB | RLHF 쌍 |
| lemon-mint/korean-realqa-reasoning-v01-preference | 58MB | QA 추론 |

---

## GGUF & Ollama

### 제공 양자화 파일

| 파일 | 크기 | 설명 |
|------|:----:|------|
| `gguf/frankenstallm-3b-v2-Q4_K_M.gguf` | 1.8GB | **권장** — 크기 대비 최적 품질 |
| `gguf/frankenstallm-3b-v2-Q8_0.gguf` | 3.0GB | 높은 품질 |
| `gguf/frankenstallm-3b-v2-f16.gguf` | 5.7GB | 전체 정밀도 |
| `model.safetensors` | 5.7GB | Transformers 네이티브 (3B ORPO best, byte-fallback 수정, vocab=64256) |

### 권장 샘플링 파라미터

| 파라미터 | 값 | 비고 |
|---------|:---:|------|
| `temperature` | 0.7 | 한국어 생성 품질 최적 |
| `repeat_penalty` | 1.2 | **필수** — 미적용 시 greedy 반복률 30.9% |
| `top_p` | 0.9 | Nucleus 샘플링 |
| `top_k` | 50 | Top-k 후보 수 |
| `max_tokens` | 512 | 최대 생성 길이 |
| `num_ctx` | 4096 | 컨텍스트 윈도우 (초과 금지) |

> ⚠️ 반드시 `repeat_penalty >= 1.2`를 사용하세요. 적용하면 반복률이 **0%** 로 떨어집니다. 미적용 시 greedy 디코딩에서 ~31% 3-gram 반복이 발생합니다.

---

## 제한 사항

- **영어 성능 제한**: MMLU-EN ~23%, HellaSwag-EN ~29% — 한국어 특화 모델입니다
- **코드 생성**: 거의 불가능 (학습 데이터에 코드 비중이 낮음)
- **Greedy 반복**: `repeat_penalty` 미사용 시 30.9% 3-gram 반복 — 반드시 `repeat_penalty >= 1.2` 사용
- **안전성**: 안전 정렬(safety alignment) 데이터가 학습에 포함되지 않았으므로 적절한 가드레일과 함께 사용하세요
- **규모 차이**: 수조 토큰으로 학습된 상용 3B 모델 대비 ~600억 토큰으로 학습 — 전반적 벤치마크 점수는 낮을 수 있습니다

---

## 하드웨어 및 학습 환경

| 구성 요소 | 사양 |
|-----------|------|
| GPU | 8× NVIDIA B200 (183GB HBM3e × 8, 총 ~1.47TB) |
| FP8 연산 | 2,250 TFLOPS/GPU (총 18,000 TFLOPS) |
| 인터커넥트 | NVLink 5.0, NVSwitch all-to-all mesh |
| CPU | 2× AMD EPYC 9365 (72코어, Zen 5) |
| RAM | 2.21 TB DDR5 |
| PyTorch | 2.10.0a0+b4e4ee81d3.nv25.12 (NVIDIA 커스텀) |
| TransformerEngine | 2.10.0 |
| FlashAttention | 2.7.4 |
| NCCL | 2.28.9 |
| CUDA | 13.1 |
| 총 학습 시간 | ~86시간 (사전학습 63h + SFT 15.5h + ORPO 7h) |

---

## 인용

```bibtex
@misc{frankenstallm2026,
  title={FRANKENSTALLM: A Korean 3B LLM Built From Scratch on B200 GPUs},
  author={pathcosmos},
  year={2026},
  url={https://huggingface.co/pathcosmos/frankenstallm},
  note={3-phase training (Pretrain, SFT, ORPO) with FP8 on 8x NVIDIA B200}
}
```

---

## 링크 및 연락처

- **GitHub**: [pathcosmos/FRANKENSTALLM](https://github.com/pathcosmos/FRANKENSTALLM) — 전체 소스코드, 학습 스크립트, 빌더 로그
- **HuggingFace**: [pathcosmos/frankenstallm](https://huggingface.co/pathcosmos/frankenstallm)
- **연락처**: pathcosmos@gmail.com

---

## 감사의 글

이 프로젝트는 **과학기술정보통신부**의 **「첨단 GPU 활용 지원 사업」** (과학기술정보통신부 공고 제2025-1068호)을 통해 제공된 GPU 컴퓨팅 자원을 활용하여 수행되었습니다.

> **국가 AI컴퓨팅자원 지원포털**: https://aiinfrahub.kr
>
> - 주관: 과학기술정보통신부 (MSIT), 정보통신산업진흥원 (NIPA)
> - 운영: 한국정보통신진흥협회 (KAIT)

대한민국 정부의 AI 인프라 지원 사업 덕분에 8× NVIDIA B200 GPU 환경에서 한국어 3B LLM을 처음부터 학습할 수 있었습니다. 국가 차원의 AI 컴퓨팅 자원 지원에 깊이 감사드립니다.

---
---

> 🇺🇸 **English version below**

---

# FRANKENSTALLM 3B

> **⚠️ v2 Model Replacement Notice (2026-03-26)**
>
> The v2 GGUF and safetensors files were incorrectly deployed as a **1.2B model (hidden_size=2048, 24 layers)** due to a conversion pipeline error.
> On 2026-03-26, they were replaced with the correct **3B ORPO checkpoint (hidden_size=3072, 28 layers, vocab_size=64256, byte-fallback applied)**.
> If you downloaded v2 files before this date, please re-download.


> **A Korean 3B LLM built entirely from scratch — tokenizer, pretraining, SFT, and ORPO — on 8× NVIDIA B200 GPUs.**

| | |
|---|---|
| **Developer** | [pathcosmos](https://huggingface.co/pathcosmos) |
| **Parameters** | ~2.4B (3B-class with weight tying) |
| **Languages** | Korean (primary), English (secondary) |
| **License** | Apache 2.0 |
| **Training** | 3-phase: Pretrain → SFT → ORPO |
| **Hardware** | 8× NVIDIA B200 (FP8), ~86 hours total |

---

## Quick Start

### Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "pathcosmos/frankenstallm"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)

inputs = tokenizer(
    "한국의 전통 음식 중 김치에 대해 설명해주세요.",
    return_tensors="pt"
).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        do_sample=True,
        temperature=0.7,
        repetition_penalty=1.2,  # recommended
        top_p=0.9,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id,
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Ollama (GGUF)

```bash
# Download GGUF + Modelfile
huggingface-cli download pathcosmos/frankenstallm \
  gguf/frankenstallm-3b-v2-Q4_K_M.gguf \
  gguf/Modelfile.3b-v2-Q4_K_M \
  --local-dir ./frankenstallm

# Fix FROM path in Modelfile, then create
ollama create frankenstallm -f ./frankenstallm/gguf/Modelfile.3b-v2-Q4_K_M

# Run
ollama run frankenstallm
```

---


## File Downloads

### Model Files

| File | Size | Description | Download |
|------|------|-------------|----------|
| [`model.safetensors`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/model.safetensors) | 5.7 GB | HF Transformers native (3B ORPO, byte-fallback) | [Download](https://huggingface.co/pathcosmos/frankenstallm/resolve/main/model.safetensors) |
| [`config.json`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/config.json) | 1 KB | Model config (hidden=3072, 28L, vocab=64256) | [Download](https://huggingface.co/pathcosmos/frankenstallm/resolve/main/config.json) |
| [`tokenizer.json`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/tokenizer.json) | 4 MB | Tokenizer (SentencePiece Unigram) | [Download](https://huggingface.co/pathcosmos/frankenstallm/resolve/main/tokenizer.json) |
| [`tokenizer.model`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/tokenizer.model) | 1.4 MB | SentencePiece model (for GGUF conversion) | [Download](https://huggingface.co/pathcosmos/frankenstallm/resolve/main/tokenizer.model) |

### GGUF (Ollama / llama.cpp)

| File | Size | Quantization | Download |
|------|------|--------------|----------|
| [`frankenstallm-3b-v2-Q4_K_M.gguf`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/gguf/frankenstallm-3b-v2-Q4_K_M.gguf) | 1.8 GB | **Q4_K_M (Recommended)** | [Download](https://huggingface.co/pathcosmos/frankenstallm/resolve/main/gguf/frankenstallm-3b-v2-Q4_K_M.gguf) |
| [`frankenstallm-3b-v2-Q8_0.gguf`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/gguf/frankenstallm-3b-v2-Q8_0.gguf) | 3.0 GB | Q8_0 (High quality) | [Download](https://huggingface.co/pathcosmos/frankenstallm/resolve/main/gguf/frankenstallm-3b-v2-Q8_0.gguf) |
| [`frankenstallm-3b-v2-f16.gguf`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/gguf/frankenstallm-3b-v2-f16.gguf) | 5.7 GB | F16 (Lossless) | [Download](https://huggingface.co/pathcosmos/frankenstallm/resolve/main/gguf/frankenstallm-3b-v2-f16.gguf) |

### Training Data (for SFT / ORPO reproduction)

| File | Size | Purpose | Download |
|------|------|---------|----------|
| [`train_filtered.jsonl`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/data/sft_combined/train_filtered.jsonl) | 7.5 GB | SFT training data (24 sources, 2.4M samples, filtered) | [Download](https://huggingface.co/pathcosmos/frankenstallm/resolve/main/data/sft_combined/train_filtered.jsonl) |
| [`val_filtered.jsonl`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/data/sft_combined/val_filtered.jsonl) | 157 MB | SFT validation data | [Download](https://huggingface.co/pathcosmos/frankenstallm/resolve/main/data/sft_combined/val_filtered.jsonl) |
| [`combined_preference.jsonl`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/data/preference/combined_preference.jsonl) | 2.6 GB | ORPO training data (7 sources, 630K pairs) | [Download](https://huggingface.co/pathcosmos/frankenstallm/resolve/main/data/preference/combined_preference.jsonl) |

<details>
<summary>Individual ORPO Preference Sources (7 datasets)</summary>

| File | Size | Download |
|------|------|----------|
| [`nayohan_preference-collection-ko-full.jsonl`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/data/preference/nayohan_preference-collection-ko-full.jsonl) | 4.9 GB | [Download](https://huggingface.co/pathcosmos/frankenstallm/resolve/main/data/preference/nayohan_preference-collection-ko-full.jsonl) |
| [`heegyu_orca-math-korean-preference-cleaned.jsonl`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/data/preference/heegyu_orca-math-korean-preference-cleaned.jsonl) | 1.6 GB | [Download](https://huggingface.co/pathcosmos/frankenstallm/resolve/main/data/preference/heegyu_orca-math-korean-preference-cleaned.jsonl) |
| [`kuotient_orca-math-korean-dpo-pairs.jsonl`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/data/preference/kuotient_orca-math-korean-dpo-pairs.jsonl) | 750 MB | [Download](https://huggingface.co/pathcosmos/frankenstallm/resolve/main/data/preference/kuotient_orca-math-korean-dpo-pairs.jsonl) |
| [`maywell_ko_Ultrafeedback_binarized.jsonl`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/data/preference/maywell_ko_Ultrafeedback_binarized.jsonl) | 394 MB | [Download](https://huggingface.co/pathcosmos/frankenstallm/resolve/main/data/preference/maywell_ko_Ultrafeedback_binarized.jsonl) |
| [`tellang_yeji-preference-ko-v1.jsonl`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/data/preference/tellang_yeji-preference-ko-v1.jsonl) | 171 MB | [Download](https://huggingface.co/pathcosmos/frankenstallm/resolve/main/data/preference/tellang_yeji-preference-ko-v1.jsonl) |
| [`jojo0217_korean_rlhf_dataset.jsonl`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/data/preference/jojo0217_korean_rlhf_dataset.jsonl) | 137 MB | [Download](https://huggingface.co/pathcosmos/frankenstallm/resolve/main/data/preference/jojo0217_korean_rlhf_dataset.jsonl) |
| [`lemon-mint_korean-realqa-reasoning-v01-preference.jsonl`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/data/preference/lemon-mint_korean-realqa-reasoning-v01-preference.jsonl) | 58 MB | [Download](https://huggingface.co/pathcosmos/frankenstallm/resolve/main/data/preference/lemon-mint_korean-realqa-reasoning-v01-preference.jsonl) |

</details>

### Data Pipeline Scripts

| File | Description |
|------|-------------|
| [`prepare_sft_data.py`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/data/prepare_sft_data.py) | HF datasets → JSONL normalization (Alpaca format) |
| [`filter_sft_v2.py`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/data/filter_sft_v2.py) | SFT quality filtering (dedup, repetition filter) |
| [`prepare_preference_combined.py`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/data/prepare_preference_combined.py) | Preference data merging (DPO/ORPO format) |
| [`tokenize_extra.py`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/data/tokenize_extra.py) | Large-scale parallel tokenization |
| [`sft_dataset.py`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/data/sft_dataset.py) | SFT dataset loader (Alpaca/conversation format) |

### Phase Reports

| Report | Content |
|--------|---------|
| [`PROJECT_COMPLETION_REPORT`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/reports/2026-03-10_PROJECT_COMPLETION_REPORT.md) | Final project completion report |
| [`ORPO_EVALUATION_REPORT`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/reports/2026-03-09_ORPO_EVALUATION_REPORT.md) | ORPO 10-dimension evaluation |
| [`ORPO_TRAINING_JOURNEY`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/reports/2026-03-08_ORPO_TRAINING_JOURNEY.md) | ORPO training journey (HP sweep, debugging) |
| [`SFT_COMPLETION_AND_EVAL`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/reports/2026-03-06_3B_SFT_COMPLETION_AND_EVAL_SUMMARY.md) | SFT completion and evaluation |
| [`3B_BASE_EVALUATION`](https://huggingface.co/pathcosmos/frankenstallm/blob/main/reports/2026-03-05_3B_BASE_EVALUATION_REPORT.md) | Pretrained base model evaluation |


---

## Model Highlights

- **From-scratch Korean tokenizer**: SentencePiece Unigram, 64K vocab, 99.95% Korean character coverage
- **3-phase training pipeline**: Pretrain (57K steps, ~60B tokens) → SFT (25.5K steps, 2.4M samples) → ORPO (10K steps, 630K preference pairs)
- **B200 FP8 native training**: TransformerEngine MXFP8 on NVIDIA B200 — 2× theoretical throughput vs BF16
- **GGUF deployment ready**: Q4_K_M (1.8GB), Q8_0 (3.0GB), F16 (5.7GB) with optimized Ollama Modelfiles

---

## Architecture

| Component | Value |
|-----------|-------|
| Type | Decoder-only Transformer (LLaMA-style) |
| Hidden size | 3,072 |
| Layers | 28 |
| Attention heads | 24 |
| KV heads | 8 (GQA 3:1) |
| FFN dim | 8,192 (SwiGLU) |
| Vocab size | 64,256 (byte-fallback applied) |
| Context length | 4,096 (trained at 2,048) |
| Position encoding | RoPE (θ=500,000) |
| Normalization | Pre-norm RMSNorm |
| Attention impl | FlashAttention-2 |
| Precision | FP8 (MXFP8 via TransformerEngine) |
| Weight tying | Yes (embedding ↔ lm_head) |

---

## Training Pipeline

### Phase 1: Pretraining

| Detail | Value |
|--------|-------|
| Steps | 57,000 |
| Final loss | 1.466 |
| Tokens seen | ~60B (38.5B unique × ~1.5 epochs) |
| Duration | ~63 hours |
| Data | CC-100 KO, HPLT KO, C4 KO, NamuWiki, Wikipedia KO, Cosmopedia (EN) |
| Batch size | 5 × 8 GPU × 8 accum × 2,048 seq = ~655K tok/step |

### Phase 2: Supervised Fine-Tuning (SFT)

| Detail | Value |
|--------|-------|
| Steps | 25,500 (early stop at 77.3%) |
| Best val_loss | 1.8851 (step 23,000) |
| Duration | ~15.5 hours |
| Data | 2,439,397 samples from 24 sources (7.48 GB) |
| Mix | 70% SFT + 30% pretrain replay (catastrophic forgetting prevention) |
| Knowledge forgetting | 0.9% (19 datasets) |

### Phase 3: ORPO (Odds Ratio Preference Optimization)

| Detail | Value |
|--------|-------|
| Steps | 9,997 (early convergence) |
| Best eval_loss | 1.625 |
| Preference accuracy | 76.02% |
| Reward margin | 0.6100 |
| Duration | ~7 hours |
| Data | ~630K preference pairs from 7 Korean HF datasets |
| Hyperparams | beta=0.25, lr=1.2e-5, eff_batch=128 |

**Total training time: ~86 hours on 8× B200**

---

## Benchmarks

### Training Phase Progression (Base → SFT → ORPO)

| Benchmark | Base | SFT | ORPO | Δ (Base→ORPO) |
|-----------|:----:|:---:|:----:|:---:|
| **KoBEST Avg (0-shot)** | 43.7% | 43.3% | **52.8%** | **+9.1pp** |
| KoBEST COPA | 49.3% | 48.6% | **63.9%** | +14.6pp |
| KoBEST HellaSwag-KO | 21.6% | 19.8% | **38.0%** | +16.4pp |
| KoBEST SentiNeg | 48.6% | 49.1% | **62.5%** | +13.9pp |
| KoBEST BoolQ | 50.3% | 50.1% | 50.6% | +0.3pp |
| PIQA | 52.5% | 52.6% | **59.9%** | +7.3pp |
| ARC-Easy | 25.6% | 25.9% | **36.0%** | +10.4pp |
| HAE-RAE | 19.7% | 19.9% | 21.8% | +2.1pp |
| HellaSwag EN | 26.2% | 26.1% | 29.2% | +3.0pp |
| Greedy 3-gram repetition | 61.0% | 73.0% | **30.9%** | -30.1pp |
| EOS termination rate | 0% | 60% | **67%** | +67pp |
| PPL forgetting | — | 0.9% | 4.1% | within 15% ✅ |

### 3B-class Model Comparison (Ollama, 35 tests)

| Model | Params | Korean NLU | Knowledge | Instruction | Reasoning | Avg Score |
|-------|:------:|:----------:|:---------:|:-----------:|:---------:|:---------:|
| Qwen 2.5 3B | 3B | 100.0 | 20.8 | 55.6 | 62.5 | **63.4** |
| Phi-4 Mini | 3.8B | 66.7 | 29.2 | 33.3 | **87.5** | 60.6 |
| **FRANKENSTALLM 3B** | **3B** | **100.0** | **75.0** | **66.7** | 50.0 | 46.7 |

> FRANKENSTALLM leads in **Korean NLU** (tied with Qwen), **Korean Knowledge** (75 vs 20.8/29.2), and **Instruction Following** (66.7 vs 55.6/33.3).

### Inference Speed (Ollama, Q4_K_M)

| Model | Avg TTFT | TPS | Note |
|-------|:--------:|:---:|------|
| **FRANKENSTALLM 3B** | **16.7ms** | **142.5** | Fastest |
| Phi-4 Mini 3.8B | 25.6ms | 100.4 | |
| Qwen 2.5 3B | 28.2ms | 93.8 | |

### Perplexity Preservation (ORPO Knowledge Retention)

| Dataset | Base PPL | ORPO PPL | Forgetting |
|---------|:--------:|:--------:|:----------:|
| Korean C4 | 5.72 | 5.87 | +2.7% |
| Korean Wiki | 11.84 | 12.21 | +3.2% |
| Max forgetting | — | — | 4.1% ✅ |

---

## Training Data

### Pretraining (~38.5B tokens)

| Category | Sources | Est. Tokens |
|----------|---------|:-----------:|
| Korean Web Crawl | C4 KO, CC-100 KO, HPLT KO | ~17.2B |
| Korean Encyclopedia | Wikipedia KO, NamuWiki (2 versions) | ~2.8B |
| English Educational | Cosmopedia (Stories, Web, Stanford, WikiHow, OpenStax, Khan) | ~5.7B |
| English Math/Science | AutoMathText, OpenWebMath, Proof-Pile-2 | ~8.5B |
| Code | StarCoder (filtered) | ~4.3B |

### SFT (2.4M samples, 24 sources)

| Domain | Share | Key Datasets |
|--------|:-----:|-------------|
| Reasoning/CoT | 38% | reasoning_r1_1.4m, magpie_reasoning |
| Korean Instructions | 23% | korean_instruction_mix, open_korean_instructions, kullm_v2 |
| English General | 16% | openhermes_2.5, ultrachat_200k |
| Math | 12% | NuminaMath-CoT, orca-math-ko |
| Dialog/Code/Other | 11% | smol-koreantalk, Evol-Instruct-Code-80k-ko |

### ORPO (~630K preference pairs, 7 sources)

| Dataset | Size | Domain |
|---------|:----:|--------|
| nayohan/preference-collection-ko-full | 4.9GB | General preference |
| heegyu/orca-math-korean-preference-cleaned | 1.6GB | Math reasoning |
| kuotient/orca-math-korean-dpo-pairs | 750MB | Math DPO |
| maywell/ko_Ultrafeedback_binarized | 394MB | Feedback alignment |
| tellang/yeji-preference-ko-v1 | 171MB | General preference |
| jojo0217/korean_rlhf_dataset | 137MB | RLHF pairs |
| lemon-mint/korean-realqa-reasoning-v01-preference | 58MB | QA reasoning |

---

## GGUF & Ollama

### Available Quantizations

| File | Size | Description |
|------|:----:|-------------|
| `gguf/frankenstallm-3b-v2-Q4_K_M.gguf` | 1.8GB | **Recommended** — best size/quality balance |
| `gguf/frankenstallm-3b-v2-Q8_0.gguf` | 3.0GB | Higher quality |
| `gguf/frankenstallm-3b-v2-f16.gguf` | 5.7GB | Full precision |
| `model.safetensors` | 5.7GB | Transformers native (3B ORPO best, byte-fallback fixed, vocab=64256) |

### Recommended Sampling Parameters

| Parameter | Value | Notes |
|-----------|:-----:|-------|
| `temperature` | 0.7 | Optimal for Korean generation quality |
| `repeat_penalty` | 1.2 | **Required** — without it, greedy repetition is 30.9% |
| `top_p` | 0.9 | Nucleus sampling |
| `top_k` | 50 | Top-k candidates |
| `max_tokens` | 512 | Max generation length |
| `num_ctx` | 4096 | Context window (do not exceed) |

> ⚠️ Always use `repeat_penalty >= 1.2`. With it, repetition drops to **0%**. Without it, greedy decoding produces ~31% 3-gram repetition.

---

## Limitations

- **English performance is limited**: MMLU-EN ~23%, HellaSwag-EN ~29% — this is a Korean-focused model
- **Code generation**: Near zero capability (limited code in training data)
- **Greedy repetition**: 30.9% 3-gram repetition without `repeat_penalty` — always use sampling with `repeat_penalty >= 1.2`
- **Safety**: Safety alignment data was not included in training; use with appropriate guardrails
- **Scale gap**: Compared to commercial 3B models trained on trillions of tokens, this model was trained on ~60B tokens — expect lower overall benchmark scores

---

## Hardware & Training Environment

| Component | Specification |
|-----------|---------------|
| GPU | 8× NVIDIA B200 (183GB HBM3e each, ~1.47TB total) |
| FP8 Compute | 2,250 TFLOPS/GPU (18,000 TFLOPS total) |
| Interconnect | NVLink 5.0, NVSwitch all-to-all mesh |
| CPU | 2× AMD EPYC 9365 (72 cores, Zen 5) |
| RAM | 2.21 TB DDR5 |
| PyTorch | 2.10.0a0+b4e4ee81d3.nv25.12 (NVIDIA custom) |
| TransformerEngine | 2.10.0 |
| FlashAttention | 2.7.4 |
| NCCL | 2.28.9 |
| CUDA | 13.1 |
| Total training | ~86 hours (Pretrain 63h + SFT 15.5h + ORPO 7h) |

---

## Citation

```bibtex
@misc{frankenstallm2026,
  title={FRANKENSTALLM: A Korean 3B LLM Built From Scratch on B200 GPUs},
  author={pathcosmos},
  year={2026},
  url={https://huggingface.co/pathcosmos/frankenstallm},
  note={3-phase training (Pretrain, SFT, ORPO) with FP8 on 8x NVIDIA B200}
}
```

---

## Links & Contact

- **GitHub**: [pathcosmos/FRANKENSTALLM](https://github.com/pathcosmos/FRANKENSTALLM) — Full source code, training scripts, and builder's log
- **HuggingFace**: [pathcosmos/frankenstallm](https://huggingface.co/pathcosmos/frankenstallm)
- **Contact**: pathcosmos@gmail.com

---

## Acknowledgment

This project was conducted using GPU computing resources provided through the **"Advanced GPU Utilization Support Program"** (MSIT Notice No. 2025-1068) by the **Ministry of Science and ICT (MSIT)** of the Republic of Korea.

> **National AI Computing Resource Support Portal**: https://aiinfrahub.kr
>
> - Organized by: Ministry of Science and ICT (MSIT), National IT Industry Promotion Agency (NIPA)
> - Operated by: Korea Association of Information & Telecommunication (KAIT)

We are deeply grateful for the national-level AI computing infrastructure support from the Korean government, which made it possible to train a Korean 3B LLM from scratch on 8× NVIDIA B200 GPUs.
