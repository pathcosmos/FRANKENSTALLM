# 3B 모델 작업 계획 및 진행 현황

**모델**: Korean LLM 3B (~3.42B params)  
**경로**: `checkpoints/korean_3b_*` (예정)  
**현재 상태**: 사전학습 준비 중  
**작성일**: 2026-02-27

---

## 전체 작업 상태 요약

```
[✅] 3B 아키텍처 설계      configs/korean_3b_fp8.yaml
[✅] 3B 벤치마크           36,250 tok/s, 47.8GB VRAM
[✅] 학습 스크립트 준비    scripts/launch_korean_3b.sh
[✅] 대규모 데이터 확보    640GB+ (10개 도메인)
[🔄] 추가 데이터 다운로드  Cosmopedia, GitHub-code 등 진행 중
[⏳] 데이터 전처리         새 데이터 토큰화 + 품질 필터링
[⏳] 3B 사전학습            ~200B 토큰, 예상 ~25~30시간
[⏳] 3B SFT                1.25M 샘플, 예상 ~4~6시간
[⏳] 3B ORPO               795K+ pairs, 예상 ~3~4시간
[⏳] 3B 평가               lm-eval + 반복률
[⏳] 배포                  GGUF → Ollama
```

---

## 3B 모델 아키텍처

### 설계 근거

1B 대비 각 차원을 1.5× 확장:
- Hidden dimension: 2048 → 3072
- Layers: 24 → 32
- Heads: 16 → 24
- FFN: 5472 → 8192

### 최종 아키텍처

| 파라미터 | 값 |
|---------|-----|
| vocab_size | 64,000 |
| d_model | **3072** |
| n_layers | **32** |
| n_heads | **24** |
| n_kv_heads | **8** (GQA 3:1) |
| d_ffn | **8192** (SwiGLU) |
| max_seq_len | 4,096 |
| rope_theta | 500,000 |
| dtype | FP8 |
| **총 파라미터** | **~3.42B** (실측 2.39B — config 조정 가능) |

> **실측 참고**: 벤치마크 시 d_model=2560, n_heads=32, n_kv_heads=8로 실행 시 **2.39B** 확인됨.  
> 학습 전 최종 config 확인 필요.

### 파라미터 분해

```
Embedding:       64,000 × 3072              = 196.6M
Attention/layer: Q(3072×3072) + K(3072×1024) + V(3072×1024) + O(3072×3072) = 25.1M
FFN/layer:       gate(3072×8192) + up(3072×8192) + down(8192×3072) = 75.5M
Layer subtotal:  100.6M × 32 layers         = 3,219M
LM Head:         tied with embedding
총계:            196.6M + 3,219M            ≈ 3.42B
```

---

## 벤치마크 결과 (2026-02-27 04:32~04:35 KST)

> **Benchmark-First Rule**: 장시간 작업 전 반드시 벤치마크 완료

### 실행 결과

```
체크포인트: checkpoints/korean_3b_bench/
Config: d_model=2560, n_layers=32, n_heads=32, n_kv_heads=8
실측 파라미터: 2,386,987,520 (2.39B)
```

| 지표 | 값 |
|------|-----|
| 처리량 | **36,250 tok/s** |
| 학습 속도 | **0.3 steps/s** |
| VRAM 사용량 | **47.8GB/GPU** (8 GPU 모두) |
| B200 대비 사용률 | **26%** (183GB 기준) |
| Effective batch | 1,048,576 토큰/step |
| 학습 시간 예상 | ~25~30h (200B 토큰, ~190K steps) |

### 메모리 여유

```
B200 183GB/GPU
사용 중: 47.8GB (26%)
여유: 135GB (74%)
→ batch size 증가 여지 충분
→ gradient checkpointing 불필요
```

---

## 3B 사전학습 계획

### Phase 1: 데이터 전처리

**⏳ 아직 시작 안 함**

```bash
# 1단계: 새로 받은 데이터 토큰화
bash scripts/prepare_3b_data.sh

# 처리 대상 (우선순위순)
# 1. KORMo korean-web-collection (175GB) → 추정 ~40B 토큰
# 2. fineweb2_edu_ko (234GB)              → 추정 ~60B 토큰
# 3. KORMo korean-public-corpus (26GB)    → 추정 ~6B 토큰
# 4. korean_law (15GB)                    → 추정 ~4B 토큰
# 5. domain_specific (21GB)              → 추정 ~5B 토큰
# 6. Cosmopedia (진행 중)                 → 추정 ~20B 토큰
# 7. open_web_math (26GB)                 → 추정 ~7B 토큰
# 8. GitHub-code (진행 중, 30GB)          → 추정 ~10B 토큰
```

**예상 총 토큰**: 현재 ~39B + 신규 ~150B+ = **~190B+ 토큰**

**목표**: Chinchilla minimum 60B ~ optimal 210B 달성

#### 품질 필터링 기준

```python
# 한국어 텍스트 필터
min_length = 100  # 글자 수
max_length = 100000
min_korean_ratio = 0.3    # 한국어 비율 30% 이상
max_perplexity = 1000     # LM perplexity 필터 (선택적)
dedup = True              # near-duplicate 제거

# 코드 데이터 필터
languages = ["python", "javascript", "typescript", "java", "cpp", "go", "rust", "sql"]
min_length = 50
```

#### 데이터 믹스 계획

```
한국어 텍스트 (KORMo + fineweb + cc100 등): 55%  → ~110B
영어 코드 (GitHub + SmolLM):                15%  → ~30B
영어 수학/과학 (open_web_math + Cosmo):     15%  → ~30B
한국어 교육 (법률 + 교과서 + 뉴스):          10%  → ~20B
영어 일반 (CulturaX 영어 부분):              5%  → ~10B
```

### Phase 2: 사전학습 실행

**⏳ 아직 시작 안 함**

```bash
bash scripts/launch_korean_3b.sh
# 또는
nohup python -m train.pretrain \
    --config configs/korean_3b_fp8.yaml \
    --data_dir data/tokenized/ \
    --output_dir checkpoints/korean_3b_fp8_run1 \
    --max_steps 195000 \
    --save_interval 1000 \
    --eval_interval 500 &
echo $! > /tmp/3b_pretrain_pid.txt
```

#### 학습 설정 계획

```yaml
# configs/korean_3b_fp8.yaml
optimizer: AdamW
lr: 2e-4                   # 1B보다 약간 낮게
lr_schedule: cosine_decay
warmup_steps: 2000
total_steps: ~195000        # 200B 토큰 / (1M tok/step)
batch_size_per_gpu: 8       # 벤치마크 기준
num_gpus: 8
seq_len: 4096
effective_batch_tokens: 1048576  # ~1M
gradient_clip: 1.0
dtype: FP8
save_interval: 2000
eval_interval: 1000
```

#### 예상 학습 시간

```
총 토큰: ~200B
Steps: 200B / 1,048,576 ≈ 190,700 steps
처리량: 36,250 tok/s
총 시간: 200B / 36,250 ≈ 5,517,000초 ≈ ~65시간

또는 step 기준:
190,700 steps / 0.3 steps/s ≈ 635,000초 ≈ ~176시간 ???

→ 재계산 필요: 벤치마크 batch size 확인 후 조정
→ 벤치마크의 0.3 steps/s는 batch=256 seqs × 4096 = 1,048,576 tok/step
→ 36,250 tok/s × 3600 = 130M tok/hr → 200B / 130M = ~1,538시간 ???

⚠️ 재벤치마크 필요: batch size별 처리량 재측정
```

> **주의**: 위 시간 예상에 불확실성 있음. **3B pretrain 전 Benchmark-First Rule 재적용 필요**

### Phase 3: SFT (3B)

**⏳ Pretrain 완료 후 시작**

```bash
# SFT 데이터 확보 현황
data/sft/train.jsonl         → 161,848 샘플 (기존)
data/sft_extra/              → 1,084,752 샘플 (신규)
# 합계: ~1.25M 샘플

# SFT 실행 (SFT v2와 동일 코드, 모델만 교체)
nohup python -m train.sft \
    --model_path checkpoints/korean_3b_fp8_run1/checkpoint-best \
    --data_path data/sft_combined/train.jsonl \
    --output_dir checkpoints/korean_3b_sft \
    --lr 1e-5 \
    --warmup_steps 500 \
    --max_steps 15000 \
    --neftune_alpha 5 &
echo $! > /tmp/3b_sft_pid.txt
```

#### SFT 예상 시간

- 3B는 1B 대비 ~2.9× 느림
- 1B SFT: ~3시간 (9K steps) → 3B SFT: **~9~12시간** (15K steps 예상)

### Phase 4: ORPO (3B)

**⏳ SFT 완료 후 시작**

```bash
# Preference 데이터
data/preference/   → 795,468쌍 (7.9GB)

# ORPO 실행
pip install trl    # 필요 시
nohup python train/orpo.py \
    --model_path checkpoints/korean_3b_sft/checkpoint-best \
    --preference_data data/preference/ \
    --output_dir checkpoints/korean_3b_orpo \
    --lr 5e-7 \
    --max_steps 5000 &
echo $! > /tmp/3b_orpo_pid.txt
```

#### 목표

- 반복률: 18% (1B SFT) → **<5%** (3B ORPO 후)
- haerae scores: 큰 개선 기대 (파라미터 3× 증가)

---

## 3B 평가 계획

### 빠른 평가 (SFT 직후)

```bash
# HF 변환
python scripts/convert_to_hf.py \
    checkpoints/korean_3b_sft/checkpoint-best \
    outputs/hf_3b_sft_best

# lm-eval
bash scripts/run_eval_quick.sh \
    checkpoints/korean_3b_sft/checkpoint-best \
    eval/outputs/quick_3b_sft
```

### 평가 목표

| Task | 1B 현재 | 3B 목표 |
|------|---------|---------|
| kobest_boolq | 0.50 | **>0.65** |
| kobest_copa | 0.65 | **>0.75** |
| haerae_general_knowledge | 0.23 | **>0.40** |
| haerae_history | 0.15 | **>0.35** |
| paws_ko | 0.49 | **>0.65** |
| 반복률 (SFT) | 18% | **<12%** |
| 반복률 (ORPO) | — | **<5%** |

### 풀 평가 (ORPO 후)

```bash
bash scripts/run_eval_full.sh \
    checkpoints/korean_3b_orpo/checkpoint-best \
    eval/outputs/full_3b_orpo
```

추가 태스크:
- `ko_ifeval` (instruction following)
- `klue_mrc` (독해)
- `ko_hellaswag` (상식 추론)
- `korquad` (한국어 QA)

---

## 3B 배포 계획

**⏳ 평가 통과 후 시작**

```bash
# GGUF 변환
bash scripts/convert_to_gguf.sh \
    outputs/hf_3b_orpo_best \
    korean-3b-instruct

# Ollama 배포
bash scripts/deploy_ollama.sh korean-3b-instruct

# HuggingFace 업로드
huggingface-cli upload \
    somebody-to-love/korean-3b-instruct \
    outputs/hf_3b_orpo_best
```

---

## 현재 병목 및 위험 요소

### 데이터 전처리

- **병목**: 640GB 원시 데이터 토큰화 → 예상 ~10~20시간 (병렬 처리 필요)
- **위험**: 토큰화 품질 (잘못된 포맷, 특수 문자 처리)
- **해결**: `scripts/prepare_3b_data.sh` 병렬 실행

### 학습 시간 불확실성

- 벤치마크에서 측정한 36,250 tok/s는 50 steps 기준 (초반부 warming up 포함 가능)
- 200B 토큰 학습 시 실제 시간은 재측정 필요
- **권장**: 학습 시작 전 1,000 steps 실행해서 실제 처리량 측정

### 스토리지

```
현재 사용:
data/: ~640GB+
checkpoints/: ~100GB+
outputs/: ~수GB
/PROJECT 여유: 19TB+ → 충분

추가 필요:
- 토큰화된 binary: 200B 토큰 × 2bytes = ~400GB
- 3B 체크포인트 (2000 steps마다): ~8GB × 수십 = ~수백GB
- 합계 추가 필요: ~700GB → 총 사용 ~1.4TB → 여유 충분
```

---

## 즉시 실행 가능한 다음 단계

### Step 1: 현재 다운로드 완료 대기

```bash
# 모니터링
tail -f /tmp/dl_more_data.log
tail -f /tmp/dl_pretrain_extra.log
tail -f /tmp/dl_code2.log

# 진행 중:
# - Cosmopedia (합성 교과서)
# - GitHub-code-clean (30GB)
# - 한국어 SFT 추가 데이터
```

### Step 2: SFT v2 eval 결과 확인

```bash
# 반복률 체크 (lm-eval 완료 후)
python /tmp/rep_check_hf.py

# lm-eval 결과 (이미 완료됨)
# kobest_copa: 0.6460, haerae_gk: 0.2273
```

### Step 3: 데이터 전처리 시작

```bash
# 병렬 토큰화 (subagent 활용)
bash scripts/prepare_3b_data.sh --parallel 8

# 대상:
# - KORMo web: 175GB → korean_3b_train.bin 추가
# - fineweb2_edu_ko: 234GB
# - 법률/도메인: 36GB
```

### Step 4: 3B Pretrain 시작

```bash
# Benchmark-First Rule: 학습 전 benchmark 재확인
bash scripts/launch_korean_3b.sh --benchmark_steps 100

# 확인 후 본 학습
nohup bash scripts/launch_korean_3b.sh &
echo $! > /tmp/3b_pretrain_pid.txt
```

---

## 체크리스트

### 데이터 전처리

- [ ] KORMo web-collection 토큰화 (175GB)
- [ ] fineweb2_edu_ko 토큰화 (234GB)
- [ ] KORMo public-corpus 토큰화 (26GB)
- [ ] korean_law 토큰화 (15GB)
- [ ] domain_specific 토큰화 (21GB)
- [ ] Cosmopedia 토큰화 (다운 완료 후)
- [ ] GitHub-code 토큰화 (다운 완료 후)
- [ ] 데이터 믹스 비율 적용 후 최종 binary 생성
- [ ] 토큰 수 확인 (목표: 100B+)

### 3B Pretrain

- [ ] configs/korean_3b_fp8.yaml 최종 확인
- [ ] Benchmark-First: 1,000 steps 실제 처리량 측정
- [ ] 예상 학습 시간 계산
- [ ] nohup 실행 + PID /tmp/3b_pretrain_pid.txt 저장
- [ ] TensorBoard 모니터링 설정
- [ ] 체크포인트 저장 확인

### 3B SFT

- [ ] SFT 데이터 통합 (기존 162K + 신규 1.08M)
- [ ] SFT v2 코드 재사용 (버그 수정 완료)
- [ ] 하이퍼파라미터 조정 (lr, steps)
- [ ] nohup 실행

### 3B ORPO

- [ ] `pip install trl` 확인
- [ ] data/preference/ 포맷 확인 (chosen/rejected 필드)
- [ ] ORPO 실행

### 3B 평가

- [ ] HF 변환: convert_to_hf.py
- [ ] lm-eval 5개 태스크
- [ ] 반복률 측정 (<5% 목표)
- [ ] 추가 벤치마크 (ko_ifeval 등)

### 배포

- [ ] GGUF 변환
- [ ] Ollama 테스트
- [ ] HuggingFace 업로드

---

*최종 업데이트: 2026-02-27 12:xx KST*
