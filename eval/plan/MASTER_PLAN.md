# 🗺️ MASTER PLAN: 한국어 LLM 1B 재학습 → 3B → 배포

**작성일**: 2026-02-27  
**프로젝트**: `/PROJECT/0325120031_A/ghong/taketimes/llm-bang/`  
**결정**: Restart (base checkpoint에서 클린 재학습)  
**총 예상 기간**: ~35시간 (1B: 3시간 → 3B pretrain: 26시간 → 3B SFT+평가: 6시간)

---

## 📊 전체 타임라인 한눈에 보기

```
Phase 0  ██░░░░░░░░░░░░░░░░░░░░░░  30분    데이터/코드 준비
Phase 1  ████░░░░░░░░░░░░░░░░░░░░  40분    1B SFT 재학습
Phase 2  ██████░░░░░░░░░░░░░░░░░░  2시간   1B 평가
         ────── 여기서 판단 ──────
Phase 3A ████████░░░░░░░░░░░░░░░░  3-5시간  (조건부) 1B 추가 개선
Phase 3B ████████████████████████  26시간   3B 사전학습
Phase 4  ████░░░░░░░░░░░░░░░░░░░░  2시간   3B SFT
Phase 5  ██████░░░░░░░░░░░░░░░░░░  4시간   평가 & 배포
```

---

## Phase 0: 재학습 직전 준비 (오늘, ~30분)

### 체크리스트

#### ☐ 0-1. 데이터 재생성 (~20분)
```bash
cd /PROJECT/0325120031_A/ghong/taketimes/llm-bang

# prepare_sft_data.py 재실행 (강화 필터 + 수정된 가중치)
python data/prepare_sft_data.py \
    --output_dir data/sft_v2/ \
    --val_split 0.1
```

**확인 사항**:
- 필터링 후 **120K-135K 샘플** 남아야 함 (기존 159K에서 저품질 제거)
- `</s>` 리터럴 113건, Q/A 마커 ~550건, 자체반복 57건 제거 확인
- OpenOrca 가중치: 5.0 → 2.0으로 감소 확인
- Val split: ~12-13K 샘플 (10%)
- 짧은 output (<80자) 제거 확인

```bash
# 결과 확인
wc -l data/sft_v2/train.jsonl data/sft_v2/val.jsonl
# 예상: train ~108K-120K, val ~12K-13K
```

**완료 기준**: train 100K+ 샘플, val 10K+ 샘플. 제거된 샘플 spot check 시 실제 저품질.

#### ☐ 0-2. sft_dataset.py 수정 확인 (~5분)

이미 수정된 항목 확인:

| 수정 사항 | 파일 | 확인 |
|-----------|------|------|
| Dynamic padding 실제 작동 | `data/sft_dataset.py` `__getitem__` | ☐ 패딩 없이 실제 길이 텐서 반환 |
| EOS 보존 | `data/sft_dataset.py` L130-134 | ☐ `response_ids[:allowed-1] + [eos_id]` |
| Collate fn | `data/sft_dataset.py` `dynamic_collate_fn` | ☐ 배치별 가변 패딩 |

```bash
# 핵심 코드 확인
grep -n "allowed_response" data/sft_dataset.py
grep -n "eos_token_id" data/sft_dataset.py
grep -n "torch.full" data/sft_dataset.py  # 4096 고정 패딩 없어야 함
```

#### ☐ 0-3. launch_sft.sh 수정 (~5분)

```bash
# 변경할 값들:
# RUN_NAME=korean_1b_sft_v2
# SFT_DATA=data/sft_v2/train.jsonl
# VAL_DATA=data/sft_v2/val.jsonl
# MAX_STEPS=10000  (3-4 epoch, 기존 5000에서 증가)
# WARMUP_STEPS=300  (3%)

cp scripts/launch_sft.sh scripts/launch_sft_v2.sh
# 편집 후 diff 확인
```

#### ☐ 0-4. Sanity Check (~5분)

```bash
# 100 steps만 빠르게 돌려서 파이프라인 정상 확인
bash scripts/launch_sft_v2.sh --max_steps 100

# 확인:
# - Loss가 2.0-2.5 범위에서 시작하는가? ✅
# - 배치 내 시퀀스 길이가 가변적인가? (로그에서 확인) ✅
# - Val loss가 출력되는가? ✅
# - OOM 없는가? ✅
```

**완료 기준**: 100 steps 에러 없이 완료, loss 합리적 범위, val loss 출력 확인.

---

## Phase 1: 1B SFT 재학습 (오늘, ~40분)

### 실행 명령어

```bash
cd /PROJECT/0325120031_A/ghong/taketimes/llm-bang

RUN_NAME=korean_1b_sft_v2 \
BASE_CHECKPOINT=checkpoints/korean_1b_fp8_run1/checkpoint-0034000 \
SFT_DATA=data/sft_v2/train.jsonl \
VAL_DATA=data/sft_v2/val.jsonl \
MAX_STEPS=10000 \
WARMUP_STEPS=300 \
LR=2.0e-5 \
bash scripts/launch_sft.sh
```

### 모니터링

**실시간 로그**:
```bash
tail -f checkpoints/korean_1b_sft_v2/train.log
```

**TensorBoard**:
```bash
tensorboard --logdir checkpoints/korean_1b_sft_v2/tensorboard --port 6007
```

**핵심 수치**:

| 수치 | 정상 범위 | 경고 | 즉시 중단 |
|------|----------|------|----------|
| Train Loss | 시작 2.0-2.5, 최종 <1.90 | >2.5 at step 500+ | >3.0 (발산) |
| Val Loss | Train의 1.0-1.1배 | Train의 1.2배 | Train 대비 계속 상승 (과적합) |
| GNorm | 0.8-1.5 | >2.0 | >5.0 (gradient 폭발) |
| 학습 속도 | 기존 대비 2x+ (dynamic padding 효과) | 기존과 비슷 | 기존보다 느림 |

**체크포인트 관찰**:
- Step 500: 파이프라인 안정성 확인
- Step 2500: 중간 지점, loss 추세 확인
- Step 5000: 기존 학습과 비교 (loss < 1.97이어야 함)
- Step 7500: 수렴 여부 확인
- Step 10000: 최종

### 성공 기준

| 지표 | 목표 | 실패 기준 |
|------|------|----------|
| Final Train Loss | < 1.90 | > 2.00 |
| Final Val Loss | < 2.00 | Train 대비 1.2배 초과 |
| Val Loss 추세 | 하강 or 안정 | 3연속 상승 (과적합) |
| 학습 시간 | ~40-60분 | >2시간 (dynamic padding 미작동) |

### 실패 시 대응

| 상황 | 원인 추정 | 대응 |
|------|----------|------|
| Loss 발산 (>3.0) | LR 과다 or 데이터 버그 | LR=1e-5로 재시도 |
| OOM | 배치 크기 과다 | BATCH_SIZE=2로 감소 |
| Loss 정체 (step 2000+ 변화 없음) | LR 부족 or 데이터 문제 | 데이터 점검, LR=3e-5 시도 |
| Val Loss 발산 (과적합) | Epoch 과다 | Early stop at best val checkpoint |
| 학습 속도 기존과 같음 | Dynamic padding 미작동 | sft_dataset.py 재점검 |

---

## Phase 2: 1B SFT 평가 (~2시간)

### 평가 순서

#### 2-1. 반복률 측정 (30분)

```bash
# 올바른 포맷(<|user|>/<|assistant|>)으로 생성 테스트
python eval/test_generation_params.py \
    --checkpoint checkpoints/korean_1b_sft_v2/checkpoint-0010000

# 다양한 rep_penalty 테스트
# rep_penalty=1.0 (없음): 목표 <10%
# rep_penalty=1.1:        목표 <3%
# rep_penalty=1.2:        목표 <1%
```

#### 2-2. 생성 품질 주관 평가 (30분)

```bash
python eval/generate.py \
    --checkpoint checkpoints/korean_1b_sft_v2/checkpoint-0010000 \
    --prompts_file eval/test_prompts.txt \
    --temperature 0.8 --top_p 0.9
```

**체크**: 한국어 자연스러움, instruction following, EOS 정상 종료

#### 2-3. 공식 벤치마크 (1시간)

```bash
# ko_ifeval
lm_eval --model hf \
    --model_args pretrained=checkpoints/korean_1b_sft_v2/checkpoint-0010000,dtype=bfloat16 \
    --tasks ko_ifeval \
    --device cuda:0 \
    --output_path eval/results/sft_v2_ko_ifeval.json

# ko_winogrande (선택)
lm_eval --model hf \
    --model_args pretrained=checkpoints/korean_1b_sft_v2/checkpoint-0010000,dtype=bfloat16 \
    --tasks ko_winogrande \
    --device cuda:0 \
    --output_path eval/results/sft_v2_ko_winogrande.json
```

### 판단 기준 & 분기

```
                    [Phase 2 평가 결과]
                          │
    ┌─────────────────────┼─────────────────────┐
    │                     │                     │
  ✅ PASS              ⚠️ PARTIAL            ❌ FAIL
 반복률<5%            반복률 5-15%          반복률>15%
 ko_ifeval>25%       ko_ifeval 15-25%      ko_ifeval<15%
    │                     │                     │
    ▼                     ▼                     ▼
 Phase 3B             Phase 3A              원인 분석
 (3B 전환)          (추가 개선)           (데이터/코드 재검토)
```

**상세 기준**:

| 지표 | ✅ Pass | ⚠️ 추가 조정 | ❌ 재학습 |
|------|---------|-------------|----------|
| 반복률 (rep_penalty 없이) | <10% | 10-20% | >20% |
| 반복률 (rep_penalty=1.1) | <5% | 5-15% | >15% |
| ko_ifeval | >25% | 15-25% | <15% |
| EOS 정상 종료율 | >85% | 60-85% | <60% |

---

## Phase 3A: 1B 추가 개선 (조건부, ~3-5시간)

> **Phase 2 결과가 ⚠️ PARTIAL일 때만 진입**

### 옵션 A: ORPO 학습 (~3시간)

#### Preference Data 준비 (1시간)
```bash
# 한국어 preference 데이터 다운로드
python -c "
from datasets import load_dataset
# 옵션 1: ko_Ultrafeedback (60K, 일반 도메인)
ds = load_dataset('maywell/ko_Ultrafeedback')
# 옵션 2: 자체 생성 (현재 모델로 rejected 생성)
"
```

**자체 생성 방법**:
1. 현재 SFT 모델로 동일 프롬프트에 여러 번 생성
2. 반복/저품질 출력 → rejected
3. 깨끗한 데이터의 정답 → chosen
4. ~10K-20K 쌍 생성

#### ORPO 학습 (1.5시간)
```python
from trl import ORPOConfig, ORPOTrainer

config = ORPOConfig(
    learning_rate=5e-7,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    beta=0.1,  # ORPO coefficient
)
trainer = ORPOTrainer(model, config, train_dataset=preference_data)
trainer.train()
```

#### 평가 (30분)
- 반복률 재측정: 목표 <5% (rep_penalty=1.1)
- ko_ifeval 재측정: 목표 >20%

### 옵션 B: 추가 SFT (데이터 보강, ~5시간)

#### 추가 데이터 수집 (2시간)
```python
from datasets import load_dataset

# 고품질 한국어 데이터 추가
datasets = {
    "hPark/orca-ko": 200_000,          # 고품질 합성
    "nayohan/llama3-instruct-ko-dataset": 58_000,  # Llama3 한국어
    "FreedomIntelligence/evol-instruct-korean": 70_000,  # GPT-4 생성
}
# 기존 120K + 추가 ~300K → 필터 후 ~350K
```

#### 재학습 (2시간)
```bash
# 증가된 데이터로 재학습
RUN_NAME=korean_1b_sft_v3 \
SFT_DATA=data/sft_v3/train.jsonl \
MAX_STEPS=15000 \
bash scripts/launch_sft.sh
```

### Phase 3A 성공 기준

| 지표 | 목표 |
|------|------|
| 반복률 (rep_penalty=1.1) | <5% |
| ko_ifeval | >20% |

**실패 시**: 1B 한계 인정, Phase 3B (3B 전환)로 바로 이동.

---

## Phase 3B: 3B 사전학습 (Phase 2 통과 후, ~26시간)

### 3B 모델 아키텍처

| 파라미터 | 1B (현재) | 3B (목표) | 비고 |
|---------|----------|----------|------|
| d_model | 2048 | 2560 | ~1.25x |
| n_layers | 24 | 32 | ~1.33x |
| n_heads | 16 | 32 | 2x |
| n_kv_heads (GQA) | 4 | 8 | 2x |
| d_ffn | 5472 | 6912 | ~1.26x |
| vocab_size | 64000 | 64000 | 동일 |
| max_seq_len | 4096 | 4096 | 동일 |
| **총 파라미터** | **1.19B** | **~3.0B** | ~2.5x |

### 설정 파일 작성

```bash
# configs/korean_3b_fp8.yaml 작성
cat > configs/korean_3b_fp8.yaml << 'EOF'
model:
  d_model: 2560
  n_layers: 32
  n_heads: 32
  n_kv_heads: 8
  d_ffn: 6912
  vocab_size: 64000
  max_seq_len: 4096
  rope_theta: 500000

training:
  lr: 3.0e-4
  min_lr: 3.0e-5
  warmup_steps: 2000
  max_steps: 100000
  batch_size: 4
  grad_accum: 4
  weight_decay: 0.1
  use_fp8: true

data:
  sources:
    - cc100_ko
    - culturax_ko
    - existing_pretrain
EOF
```

### 사전학습 데이터

| 소스 | 토큰 수 | 상태 |
|------|---------|------|
| CulturaX ko | 24.8B | ✅ 보유 |
| cc100 ko (재수집) | ~65-100B | ⚠️ 재수집 필요 (노이즈 필터링) |
| 기존 pretrain 데이터 | ~8.9B | ✅ 보유 |
| 추가 수집 (나무위키, 뉴스 등) | ~20-50B | 선택적 |
| **합계** | **~120-180B** | Chinchilla 60B 최소 충족 |

**데이터 준비 명령어**:
```bash
# cc100 재수집 + 품질 필터링
python scripts/download_cc100_ko.py --quality_filter --dedup
# MinHash dedup + perplexity filter
python scripts/quality_filter.py --input data/pretrain/ --max_ppl 1000
```

### 학습 실행

```bash
# 3B pretrain 시작 (8× B200, ~26시간)
bash scripts/run_pretrain.sh --config configs/korean_3b_fp8.yaml

# 예상 처리 속도: ~1.6M tok/s (8× B200)
# 150B tokens / 1.6M tok/s ≈ 26시간
```

### 모니터링

```bash
# 로그 확인
tail -f checkpoints/korean_3b_fp8/train.log

# 중간 체크포인트에서 base 품질 확인 (step 10000마다)
python eval/perplexity.py --checkpoint checkpoints/korean_3b_fp8/checkpoint-0010000
```

**성공 기준**: PPL < 10 (한국어 텍스트), loss 지속 하강

---

## Phase 4: 3B SFT (~2시간)

### 1B에서 배운 교훈 전부 적용

| 교훈 | 적용 |
|------|------|
| Dynamic padding 작동 확인 | ✅ sft_dataset.py 수정 완료, 그대로 사용 |
| EOS 보존 | ✅ 동일 코드 |
| Val split 필수 | ✅ 10% split |
| 3-4 epoch | ✅ MAX_STEPS 계산하여 설정 |
| OpenOrca 과다 가중치 방지 | ✅ 2.0x 이하 |
| 데이터 품질 필터링 | ✅ Phase 0에서 생성한 클린 데이터 사용 |
| 올바른 프롬프트 포맷 | ✅ `<\|user\|>/<\|assistant\|>` |

### 실행

```bash
RUN_NAME=korean_3b_sft \
BASE_CHECKPOINT=checkpoints/korean_3b_fp8/checkpoint-BEST \
SFT_DATA=data/sft_v2/train.jsonl \
VAL_DATA=data/sft_v2/val.jsonl \
MAX_STEPS=10000 \
LR=2.0e-5 \
WARMUP_STEPS=300 \
bash scripts/launch_sft.sh
```

**예상 시간**: ~2시간 (3B는 1B 대비 ~2.5x 느림)

### 성공 기준

| 지표 | 목표 |
|------|------|
| Train Loss | < 1.85 |
| Val Loss | Train의 1.1배 이내 |
| 반복률 (rep_penalty 없이) | < 10% |
| 반복률 (rep_penalty=1.1) | < 3% |

---

## Phase 5: 평가 및 배포 (~4시간)

### 5-1. 전체 벤치마크 (~2시간)

```bash
# ko_ifeval
lm_eval --model hf \
    --model_args pretrained=checkpoints/korean_3b_sft/checkpoint-BEST,dtype=bfloat16 \
    --tasks ko_ifeval --device cuda:0

# ko_winogrande
lm_eval --model hf \
    --model_args pretrained=checkpoints/korean_3b_sft/checkpoint-BEST,dtype=bfloat16 \
    --tasks ko_winogrande --device cuda:0

# KoBEST (선택)
lm_eval --model hf \
    --model_args pretrained=checkpoints/korean_3b_sft/checkpoint-BEST,dtype=bfloat16 \
    --tasks kobest_boolq,kobest_copa,kobest_wic,kobest_hellaswag,kobest_sentineg \
    --device cuda:0
```

**3B 목표 수치**:

| 벤치마크 | 1B 예상 | 3B 목표 |
|---------|---------|---------|
| ko_ifeval | 20-30% | **35-45%** |
| ko_winogrande | 53-58% | **60-68%** |
| KoBEST (avg) | 55-60% | **65-75%** |
| 반복률 | <5% | **<3%** |

### 5-2. HuggingFace Hub 업로드 (~1시간)

```bash
# HF 포맷 변환
python scripts/convert_to_hf.py \
    --checkpoint checkpoints/korean_3b_sft/checkpoint-BEST \
    --output_dir hf_models/korean-3b-instruct

# Model card 작성
cat > hf_models/korean-3b-instruct/README.md << 'EOF'
---
language: ko
license: apache-2.0
tags:
  - korean
  - llm
  - instruction-tuning
---
# Korean 3B Instruct
...벤치마크 결과, 사용법 등...
EOF

# 업로드
huggingface-cli upload ghong/korean-3b-instruct hf_models/korean-3b-instruct
```

### 5-3. vLLM 서빙 설정 (~1시간)

```bash
# vLLM 서버 시작
python -m vllm.entrypoints.openai.api_server \
    --model hf_models/korean-3b-instruct \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --port 8000

# 테스트
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "korean-3b-instruct",
        "messages": [{"role": "user", "content": "한국의 수도는?"}],
        "temperature": 0.7
    }'
```

**FP8 서빙 (B200 최적)**:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model hf_models/korean-3b-instruct \
    --quantization fp8 \
    --tensor-parallel-size 1 \
    --max-model-len 4096
```

**GGUF 변환 (Ollama/로컬 배포)**:
```bash
bash scripts/convert_to_gguf.sh checkpoints/korean_3b_sft/checkpoint-BEST
# Ollama Modelfile 작성 후
ollama create korean-3b -f Modelfile
```

---

## 📋 Phase별 요약 테이블

| Phase | 소요 시간 | 필요한 것 | 성공 기준 | 실패 시 |
|-------|----------|----------|----------|---------|
| **0: 준비** | 30분 | prepare_sft_data.py, sft_dataset.py 수정 | 클린 데이터 120K+, sanity 100steps 통과 | 코드 디버그 |
| **1: 1B SFT** | 40분 | 8×B200, 클린 데이터, 수정된 코드 | Loss<1.90, ValLoss 안정 | LR 조정 or 데이터 재점검 |
| **2: 1B 평가** | 2시간 | lm-eval-harness, 평가 스크립트 | 반복률<5%, ko_ifeval>25% | Phase 3A |
| **3A: 추가개선** | 3-5시간 | Preference 데이터, ORPO/추가 SFT | 반복률<5% 달성 | 1B 한계 인정→3B |
| **3B: 3B PT** | 26시간 | 150B+ 토큰, configs/korean_3b_fp8.yaml | PPL<10, loss 하강 | 데이터 추가 or 아키텍처 조정 |
| **4: 3B SFT** | 2시간 | Phase 0의 클린 데이터 재사용 | Loss<1.85, 반복률<3% | LR/epoch 조정 |
| **5: 배포** | 4시간 | HF 계정, vLLM | ko_ifeval>35%, 서빙 정상 | 모델 개선 후 재배포 |

---

## 🔥 오늘 당장 시작할 첫 번째 명령어

```bash
cd /PROJECT/0325120031_A/ghong/taketimes/llm-bang
python data/prepare_sft_data.py --output_dir data/sft_v2/ --val_split 0.1
```

이 명령어 하나로 Phase 0의 가장 중요한 작업(클린 데이터 생성)이 시작된다.

---

## ⚡ 가장 중요한 판단 포인트

### 1차 판단: Phase 1 완료 후 (Step 10000)
- **Val Loss가 Train Loss의 1.2배 이상?** → 과적합. Best checkpoint 사용.
- **Train Loss > 2.0?** → 무언가 잘못됨. 코드/데이터 재점검.

### 2차 판단: Phase 2 평가 후 (가장 중요!)
- **반복률 <5% AND ko_ifeval >25%?** → ✅ 3B 전환 (Phase 3B)
- **반복률 5-15%?** → ⚠️ ORPO 시도 (Phase 3A)
- **반복률 >15%?** → ❌ 원인 분석. 데이터/코드 재검토.

### 3차 판단: Phase 3B 중간 (3B pretrain step 50000)
- **Loss 하강 멈춤?** → 데이터 품질 문제. 필터링 강화.
- **PPL > 15?** → 데이터 부족. 추가 수집 필요.

---

## 🛡️ 리스크 매트릭스

| 리스크 | 확률 | 영향 | 예방/대응 |
|--------|------|------|----------|
| Dynamic padding 여전히 미작동 | 10% | 높음 (속도 3-8x 낭비) | Sanity check에서 배치 길이 확인 |
| 데이터 필터링 과다 (100K 미만) | 15% | 중간 | 필터 기준 완화 (80자→50자) |
| 1B 재학습 후에도 반복 >15% | 15% | 중간 | ORPO or 3B 전환 |
| 3B pretrain 중 OOM | 10% | 높음 | batch_size 줄이기, gradient checkpointing |
| cc100 재수집 시간 초과 | 20% | 낮음 | CulturaX만으로 시작 (24.8B) |
| 디스크 공간 부족 | 5% | 높음 | 현재 19TB 가용, 충분 |

---

*"40분 아끼려고 기술 부채를 안고 가지 마라. 3시간 투자해서 깨끗한 기반을 만들어라."*

*이 문서는 각 Phase 완료 시 결과로 업데이트할 것.*
