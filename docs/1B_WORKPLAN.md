# 1B 모델 작업 계획 및 진행 현황

**모델**: Korean LLM 1B (1.19B params)  
**경로**: `checkpoints/korean_1b_*`  
**최종 상태**: SFT v2 완료, 평가 완료, 한계 확인  
**작성일**: 2026-02-27

---

## 전체 작업 상태 요약

```
[✅] 1B Pretrain         34,000 steps, loss 1.904
[✅] SFT v1              실패 (label 버그) → 아카이브
[✅] SFT v2              9,000 steps, val_loss 2.2062
[✅] Pretrain 평가       PPL C4=5.67, Wiki=11.66
[✅] SFT v2 평가         lm-eval 5개 태스크 완료
[✅] 반복률 측정          18.0% (올바른 포맷 + rep_penalty=1.1)
[⏳] ORPO               선택적 — 3B 우선 결정으로 보류
[⏳] 배포 (1B)           3B 완성 후 재고려
```

---

## 1B Pretrain 상세

### 실행 스크립트

```bash
bash scripts/launch_korean_1b.sh
```

### 모델 아키텍처

```yaml
vocab_size: 64000
d_model: 2048
n_layers: 24
n_heads: 16
n_kv_heads: 4        # GQA 4:1
d_ffn: 5472          # SwiGLU
max_seq_len: 4096
rope_theta: 500000
dtype: FP8
total_params: ~1.19B
```

### 학습 설정

```yaml
optimizer: AdamW
lr: 3e-4
lr_schedule: cosine_decay
warmup_steps: 1000
total_steps: 34000
batch_size_per_gpu: 8
num_gpus: 8
seq_len: 4096
effective_batch_tokens: 262144   # 8*8*4096
total_tokens: ~8.93B
dtype: FP8 (Transformer Engine)
```

### 결과

| 지표 | 값 |
|------|-----|
| 최종 loss | **1.904** |
| 학습 토큰 | ~8.93B |
| 체크포인트 경로 | `checkpoints/korean_1b_fp8_run1/checkpoint-0034000` |
| 학습 완료일 | 2026-02-26 |

### Perplexity 평가

| 데이터셋 | PPL |
|---------|-----|
| C4 한국어 | **5.67** |
| Wikipedia 한국어 | **11.66** |
| Namuwiki | **25.34** |

---

## SFT 데이터 준비

### 원본 데이터

```
data/sft/train.jsonl   → 161,848 샘플 (276MB)
data/sft/val.jsonl     → 8,518 샘플 (15MB)
```

### 데이터 소스

- `evol_instruct_ko`: 고품질 instruction following
- `korean_safe_conv`: 일상 대화
- 기타 한국어 instruction 데이터

### SFT v2에서 수행한 데이터 클리닝

| 작업 | 처리 건수 |
|------|---------|
| `</s>` 리터럴 제거 | 113건 |
| Q/A 마커 오염 제거 | ~550건 |
| 자체 반복 샘플 제거 | 57건 |
| 짧은 output 제거 (<80자) | — |
| **최종 train** | **161,848** |
| **최종 val** | **8,518** |

---

## SFT v1 (아카이브)

### 버그 내역

```python
# sft_dataset.py 내 label 생성 로직
# 버그 코드 (off-by-one):
labels = input_ids[1:]      # 전체 shift → 복사 과제가 됨
# loss → 0.000 (완벽한 복사 = 완벽한 loss)

# 올바른 코드:
labels = input_ids.clone()
labels[:prompt_len] = -100  # 질문 부분 마스킹
```

### 결과

- **loss 0으로 수렴** — 언어 생성 능력 미학습
- **보관 경로**: `checkpoints/korean_1b_sft_v1_backup/`
- **절대 사용 금지**

---

## SFT v2 상세

### 실행 스크립트

```bash
bash scripts/launch_sft.sh
# 또는
nohup python -m train.sft \
    --model_path checkpoints/korean_1b_fp8_run1/checkpoint-0034000 \
    --data_path data/sft/train.jsonl \
    --val_path data/sft/val.jsonl \
    --output_dir checkpoints/korean_1b_sft \
    --lr 2e-5 \
    --warmup_steps 300 \
    --max_steps 9000 \
    --neftune_alpha 5 \
    --batch_size 8 &
```

### 수정된 코드 버그 목록

| 파일 | 버그 | 수정 |
|------|------|------|
| `data/sft_dataset.py` | label off-by-one | `labels[:prompt_len] = -100` |
| `data/sft_dataset.py` | static padding 4096 | `dynamic_collate_fn` 도입 |
| `data/sft_dataset.py` | EOS truncation | `response_ids[:allowed-1] + [eos_id]` |
| `train/sft.py` | val loop gradient | `torch.no_grad()` 블록 수정 |

### 학습 설정

```yaml
base_model: checkpoints/korean_1b_fp8_run1/checkpoint-0034000
lr: 2e-5
warmup_steps: 300
total_steps: 9000          # ~3 epochs
neftune_alpha: 5
batch_size_per_gpu: 8
num_gpus: 8
effective_batch: 64
val_interval: 250
save_interval: 250
```

### 학습 경과

| Step | val_loss | 비고 |
|------|----------|------|
| 0 | — | 시작 |
| 1,000 | ~2.35 | 초기 수렴 |
| 3,000 | ~2.28 | 개선 중 |
| 6,000 | — | ⚠️ kill -9 사고 → resume 복구 |
| 8,750 | **2.2062** | ✅ Best checkpoint |
| 9,000 | 2.2079 | 최종 (약간 과적합) |

### 체크포인트

```
checkpoints/korean_1b_sft/
├── checkpoint-best/        ← 사용 권장 (val_loss=2.2062)
├── checkpoint-0009000/     ← 최종
├── checkpoint-0008750/
├── ...
```

---

## 1B 최종 평가 결과

### 올바른 inference 설정

```python
# 반드시 이 포맷 사용 (학습 포맷과 일치)
prompt = f"<|user|>\n{question}\n<|assistant|>\n"

# generation 파라미터
generate_kwargs = {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "no_repeat_ngram_size": 3,
}
```

### lm-eval 벤치마크 (SFT v2 best, 2026-02-27)

| Task | 점수 | Random baseline | 평가 |
|------|------|----------------|------|
| kobest_boolq | 0.5000 | 0.50 | ➖ Random 수준 |
| kobest_copa | **0.6460** | 0.50 | ✅ +14.6%p |
| haerae_general_knowledge | 0.2273 | 0.25 | ❌ 미달 |
| haerae_history | 0.1543 | 0.25 | ❌ 심각 미달 |
| paws_ko | 0.4900 | 0.50 | ➖ Random 수준 |

### 반복률 측정

| 조건 | 반복률 |
|------|--------|
| 잘못된 포맷 (`### 질문/답변`) | 57% |
| 올바른 포맷 | 30.7% |
| + rep_penalty=1.1 | **18.0%** |
| + no_repeat_ngram_size=3 | 17.7% |

### Decision Gate 결과

| 기준 | 임계값 | 실제 | 판정 |
|------|--------|------|------|
| 반복률 | <5% → 3B 바로 | 18% | 🔴 |
| 반복률 | 5~15% → ORPO | 18% | 🟡 (경계) |
| ko_ifeval | >25% → 3B | — | — |
| haerae_history | 기대 수준 | 15.4% | ❌ |

**최종 판정**: 반복률 18% + 지식 취약 = **1B 구조적 한계** → 3B 전환

---

## 1B 구조적 한계 분석

### 왜 1B로는 부족한가

1. **Hidden State Collapse**: d_model=2048, 24 layers는 긴 컨텍스트(200+ 토큰)에서 attention이 특정 패턴에 고착
2. **지식 용량 부족**: 파라미터가 충분히 많지 않아 한국어 역사/상식 지식 저장 한계
3. **Repetition 근본 원인**: 확률 분포가 평탄해져 다음 토큰 선택이 무작위해지면서 반복 루프 발생
4. **Chinchilla 관점**: 1.19B × 20 = ~24B 토큰이 최소 필요 — 8.93B 토큰으로는 underfitting

### ORPO로 해결 가능한가?

- 이론적으로 반복률 5~8%까지 낮출 수 있음
- 그러나 지식 부족(haerae 낮은 점수)은 ORPO로 해결 불가
- **결론**: ORPO는 band-aid — 근본 해결은 3B로 확장

---

## 남은 1B 작업

### 선택적 (우선순위 낮음)

| 작업 | 설명 | 상태 |
|------|------|------|
| ORPO 학습 | `train/orpo.py`, 795K preference pairs 사용 | ⏳ 3B 후 재고려 |
| GGUF 변환 | `scripts/convert_to_gguf.sh` | ⏳ |
| Ollama 배포 | `scripts/deploy_ollama.sh` | ⏳ |
| HuggingFace 업로드 | `somebody-to-love` 계정 | ⏳ |

### 필요한 경우 ORPO 실행 방법

```bash
# TRL 설치 필요
pip install trl

# ORPO 실행
python train/orpo.py \
    --model_path outputs/hf_checkpoint-best \
    --preference_data data/preference/ \
    --output_dir checkpoints/korean_1b_orpo \
    --lr 5e-6 \
    --max_steps 3000
```

---

## 파일 위치 참고

```
/PROJECT/0325120031_A/ghong/taketimes/llm-bang/
├── checkpoints/
│   ├── korean_1b_fp8_run1/checkpoint-0034000   # Pretrain 최종
│   ├── korean_1b_sft/checkpoint-best            # SFT v2 Best ★
│   ├── korean_1b_sft/checkpoint-0009000        # SFT v2 최종
│   └── korean_1b_sft_v1_backup/                # v1 (사용 금지)
├── outputs/
│   └── hf_checkpoint-best/                     # HF 포맷 변환본
├── eval/outputs/quick_sft_v2_best/             # lm-eval 결과
└── data/
    ├── sft/train.jsonl                          # SFT 학습 데이터
    └── preference/                              # ORPO용 795K pairs
```
