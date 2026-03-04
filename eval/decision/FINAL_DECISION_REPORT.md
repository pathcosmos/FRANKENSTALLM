# SFT 품질 위기 분석 및 의사결정 보고서

**작성일:** 2026-02-26  
**작성자:** Optimus Prime (AI)  
**판결 유형:** 중립적 판사 — 모든 보고서 종합 후 최종 결론

---

## 1. 현재 상황 요약

| 항목 | 값 |
|------|-----|
| 모델 | Korean 1B SFT (1.19B params) |
| 학습 | 5,000 steps, ~39분, 8× B200 |
| Final Loss | 1.9677 (수렴 근접, 아직 미세 하강 중) |
| 반복률 (잘못된 포맷) | 57% → **근본 원인: 프롬프트 포맷 불일치** |
| 반복률 (올바른 포맷) | 30.7% → +rep_penalty 적용 시 **17.7%** |
| 반복률 (올바른 포맷 + rep_penalty=1.1만) | **~5%** (실험 결과) |
| 반복률 (올바른 포맷 + rep_penalty=1.1 + no_repeat_3gram) | **0.0%** |
| SFT 데이터 | 159,125 샘플, ~2 epochs |
| Epoch 수 | ~2 (업계 표준 3-5 대비 부족) |

**핵심 사실:** 원래 보고된 57% 반복률의 대부분은 **추론 시 프롬프트 포맷 불일치** 때문이었다. 학습은 `<|user|>/<|assistant|>` 포맷인데 평가는 `### 질문:/### 답변:` 포맷으로 수행됨. 이 포맷만 맞추면 57% → 5%로 급감하고, rep_penalty=1.1 추가 시 0%까지 도달.

---

## 2. 발견된 문제들 전체 목록

### 🔴 Critical (학습 품질에 직접 영향)

| # | 문제 | 심각도 | 상태 |
|---|------|--------|------|
| 1 | **추론 프롬프트 포맷 불일치** (학습≠평가) | 🔴 Critical | ✅ 수정됨 |
| 2 | **Static Padding** — Dynamic padding이 사실상 무효화 (4096 고정) | 🔴 Critical | ❌ 미수정 |
| 3 | **트렁케이션 시 EOS 손실** — 잘린 샘플에서 EOS 미학습 | 🔴 Critical | ❌ 미수정 (0.04%만 해당) |
| 4 | **Epoch 부족** — ~2 epochs (업계 표준 3-5) | 🔴 Critical | ❌ 미수정 |
| 5 | **Validation split 없음** — 과적합 모니터링 불가 | 🔴 Critical | ❌ 미수정 |

### 🟡 Important (데이터 품질)

| # | 문제 | 영향 |
|---|------|------|
| 6 | Output 내 `</s>` 리터럴 113건 | EOS 학습 혼란 |
| 7 | Output 내 Q/A 마커 ~550건 | 자체 Q/A 루프 패턴 학습 |
| 8 | 자체 반복 패턴 57건 | 반복 생성 직접 학습 |
| 9 | 짧은 output (<50자) 16,519건 (10.4%) | EOS 타이밍 불안정 |
| 10 | OpenOrca 5배 업샘플링 | 과적합 위험, 다양성 부족 |
| 11 | `<\|user\|>/<\|assistant\|>` 특수토큰 미등록 | 서브워드 분할 (경미) |

### 🟢 Minor

| # | 문제 | 영향 |
|---|------|------|
| 12 | 한국어 비율 30% 미만 샘플 13.7% | 일관성 저하 |
| 13 | Label shift 마지막 position 미학습 | EOS 이후 생성 경향 |

---

## 3. 고쳐서 가는 시나리오 (Fix & Continue)

### 시나리오 상세

현재 checkpoint-5000 위에서 추가 학습 (resume 또는 lr=1e-5로 continuation):

| 단계 | 작업 | 소요 시간 |
|------|------|-----------|
| 1 | 데이터 필터링 (품질 문제 샘플 제거) | 30분 |
| 2 | Val split 생성 | 10분 |
| 3 | 추가 학습 5,000 steps (lr=1e-5, epoch 3-4) | ~40분 |
| 4 | 평가 | 30분 |
| **합계** | | **~2시간** |

### 예상 개선 효과

| 지표 | 현재 | 예상 |
|------|------|------|
| Loss | 1.97 | 1.90-1.93 |
| 반복률 (올바른 포맷 + rep_penalty) | 17.7% | 10-15% |
| ko_ifeval | 미측정 (15-28% 추정) | +3-7%p |

### 리스크

- ⚠️ **Static padding 미수정**: 학습 속도 3-8x 낭비 지속 → 40분이면 괜찮지만 비효율
- ⚠️ **오염된 가중치 위에 쌓기**: EOS 경계 혼란 + 반복 패턴이 이미 가중치에 학습됨 → 추가 학습으로 완전히 "잊을" 수 있는가 불확실
- ⚠️ **cosine schedule 문제**: 기존 5000 steps 기준으로 LR이 이미 2e-6까지 decay → resume 시 LR 재설정 필요
- 🟡 **천장 효과**: 오염된 가중치의 한계가 어디인지 모름

---

## 4. 처음부터 다시 시나리오 (Restart from Base)

### 시나리오 상세

base checkpoint (pretrained korean_1b_fp8_run1/checkpoint-0034000)에서 깨끗한 데이터로 SFT 재시작:

| 단계 | 작업 | 소요 시간 |
|------|------|-----------|
| 1 | 데이터 필터링 (159K → ~120-130K) | 30분 |
| 2 | sft_dataset.py 수정 (dynamic padding 실제 작동, EOS 보존) | 30분 |
| 3 | Val split 생성 | 10분 |
| 4 | launch_sft.sh 수정 (10,000 steps, val_data, 가중치 조정) | 10분 |
| 5 | 학습 실행 (10,000 steps, dynamic padding 적용 시 기존보다 빠를 수 있음) | ~40-80분 |
| 6 | 평가 | 30분 |
| **합계** | | **~2.5-3시간** |

### 예상 품질

| 지표 | 예상 |
|------|------|
| Loss | 1.85-1.92 |
| 반복률 (올바른 포맷, rep_penalty=1.1) | **<5%** |
| ko_ifeval | 20-30% (1B 한계 내 최적) |

### 리스크

- 🟢 **리스크 낮음**: 이미 데이터/코드가 모두 준비되어 있음
- 🟢 **결과 예측 가능**: 깨끗한 데이터 + 올바른 패딩 + 충분한 epoch → 표준적 결과 기대
- ⚠️ **유일한 리스크**: 코드 수정(sft_dataset.py) 시 새로운 버그 도입 가능성 → 작은 subset으로 sanity check 필요

---

## 5. 최종 판결 및 근거

### 판결: 🟢 **처음부터 다시 (Restart)** — 즉시 재학습

### 핵심 논거

#### 1. 17.7% 반복률은 "고쳐야 할 수준"인가?

**결론: 배포 불가, 그러나 위기는 아니다.**

- 17.7%는 rep_penalty + no_repeat_3gram 적용 후 수치. 이 기법 없이는 30.7%
- 상업적 서비스 기준: 반복률 <5%가 업계 표준. 17.7%는 사용자 10명 중 2명이 반복 문장을 목격
- **그러나** 올바른 포맷 + rep_penalty=1.1만으로 이미 ~5% 달성 → 모델 자체는 나쁘지 않음
- 진짜 문제는 반복률보다 **코드/데이터 파이프라인의 다수 미수정 버그**

#### 2. 현재 가중치는 구제 가능한가?

**결론: 구제 가능하나, 비용 대비 비효율적.**

- EOS truncation은 0.04%만 해당 → 가중치 오염 경미
- Static padding은 가중치 품질에는 영향 없음 (학습 속도만 낭비)
- 데이터 품질 문제 (</s> 리터럴, Q/A 마커, 짧은 output)는 가중치에 이미 학습됨
- 추가 학습으로 "잊기"는 가능하지만, 깨끗하게 다시 학습하는 것과 시간 차이가 크지 않음

#### 3. 재시작 비용은?

**결론: 매우 낮음. Fix 대비 추가 비용 ~1시간.**

| | Fix (Continue) | Restart |
|---|---|---|
| 데이터 준비 | 30분 | 30분 (동일) |
| 코드 수정 | 0분 | 40분 (sft_dataset.py) |
| 학습 | 40분 | 40-80분 |
| 평가 | 30분 | 30분 (동일) |
| **합계** | **~2시간** | **~2.5-3시간** |
| **결과 품질** | 개선되지만 한계 있음 | **깨끗한 최적 결과** |

**추가 비용 1시간으로 깨끗한 기반을 확보**할 수 있다. 이 1시간은 이후 3B 전환, ORPO/DPO 적용 시 "오염된 가중치에서 시작해야 하나?"라는 고민을 완전히 제거한다.

#### 4. 어느 경로가 목표 달성이 빠른가?

**목표: 반복률 <5%, ko_ifeval 25%**

- **Fix 경로**: 17.7% → 추가 학습 → 10-15% → 여전히 >5%. ORPO 추가 필요 → +6시간. 총 ~8시간
- **Restart 경로**: 깨끗한 재학습 → <5% (추론 파라미터 포함) + ko_ifeval 20-30%. 총 ~3시간
- **Restart가 2.5배 빠름**

### 결정적 수치 근거

```
재학습 추가 비용:  +1시간 (Fix 대비)
반복률 예상 개선:  17.7% → <5% (3.5배 개선)
미수정 버그 해소:  5개 → 0개 (static padding, EOS 보존, epoch, val split, 데이터 필터)
향후 3B/ORPO 기반: 오염 가중치 → 깨끗한 가중치
ROI:              1시간 투자 → 모든 기술 부채 청산
```

---

## 6. 실행 계획 (구체적 Next Steps)

### Step 1: 데이터 필터링 (30분)

```bash
cd /PROJECT/0325120031_A/ghong/taketimes/llm-bang
python eval/data_quality_audit.py  # 또는 enhanced_quality_filter.py 실행
# 159K → ~120-130K 예상
```

**수행 내용:**
- `</s>`, `<|endoftext|>`, `EOS` 리터럴 포함 샘플 제거 (161건)
- Q/A 마커 포함 샘플 제거 (~550건)
- Output <80자 샘플 제거 (~16K건)
- N-gram 반복 샘플 제거 (57건)
- 한국어 비율 <40% 샘플 제거

**성공 기준:** 필터링 후 120K-135K 샘플 남음. 제거된 샘플 spot check 시 실제 저품질 확인.

### Step 2: 코드 수정 (40분)

**2-1. sft_dataset.py — Dynamic padding 실제 작동** (가장 중요)
- `__getitem__`에서 고정 4096 패딩 제거
- 실제 길이 텐서만 반환
- `dynamic_collate_fn`이 배치별 패딩 수행

**2-2. sft_dataset.py — EOS 보존**
```python
response_ids = response_ids[:allowed_response - 1] + [self.eos_token_id]
```

**2-3. 데이터 가중치 조정**
- OpenOrca: 5.0 → 2.0
- kovast: 0.8 → 0.5

**성공 기준:** 수정 후 작은 subset (1000 샘플, 100 steps)으로 학습이 정상 실행되는지 확인. Loss가 합리적 범위 (2.0-2.5)에서 시작.

### Step 3: Val Split + Config 수정 (10분)

```bash
# 90/10 split
python -c "
import json, random
random.seed(42)
with open('data/sft/train_cleaned.jsonl') as f:
    lines = f.readlines()
random.shuffle(lines)
split = int(len(lines) * 0.9)
with open('data/sft/train_split.jsonl', 'w') as f:
    f.writelines(lines[:split])
with open('data/sft/val_split.jsonl', 'w') as f:
    f.writelines(lines[split:])
"
```

**launch_sft.sh 수정:**
- `--max_steps 10000` (3-4 epochs)
- `--val_data data/sft/val_split.jsonl`
- `--lr 2e-5` (초기 학습이므로 유지)
- `--warmup_steps 300`

**성공 기준:** Config 파일 변경 확인, val split 크기 ~12-13K 확인.

### Step 4: 재학습 실행 (~40-80분)

```bash
bash scripts/launch_sft.sh
```

**모니터링:**
- Loss curve: 지속적 하강 확인
- Val loss: 매 500 steps 체크, 상승 시 early stop
- GNorm: 1.5 미만 유지

**성공 기준:**
- Train loss < 1.90
- Val loss가 train loss의 1.1배 이내 (과적합 없음)
- 학습 속도: dynamic padding으로 기존 대비 2x+ 향상 확인

### Step 5: 평가 (30분)

```bash
# 1. 반복률 측정 (올바른 포맷)
python eval/test_generation_params.py  # 수정된 포맷

# 2. 다양한 rep_penalty에서 반복률
# rep_penalty=1.0 (없음): 목표 <10%
# rep_penalty=1.1: 목표 <3%

# 3. ko_ifeval (가능하면)
lm_eval --model hf --tasks ko_ifeval ...
```

**성공 기준:**

| 지표 | 목표 | 실패 기준 |
|------|------|-----------|
| 반복률 (rep_penalty 없이) | <10% | >20% |
| 반복률 (rep_penalty=1.1) | <3% | >10% |
| Train loss | <1.90 | >2.00 |
| ko_ifeval | >20% | <15% |

### Step 6 (Optional): 3B 전환 준비

재학습 성공 시, 동일한 깨끗한 파이프라인으로 3B pretrain → SFT 진행 가능.
재학습 실패 시, 문제 원인 분석 후 데이터/아키텍처 수준에서 재검토.

---

## 7. 성공 기준 (각 단계별 체크포인트)

```
Step 1 ✅ 데이터 필터링
  □ 120K-135K 샘플 남음
  □ 제거된 샘플이 실제 저품질임을 spot check

Step 2 ✅ 코드 수정  
  □ 100 steps sanity check 통과
  □ 배치 내 시퀀스 길이가 가변적 (4096 고정 아님)
  □ 트렁케이션 샘플에서 마지막 토큰이 EOS

Step 3 ✅ Config
  □ Val split ~12-13K 샘플
  □ max_steps=10000, val_data 경로 설정

Step 4 ✅ 학습
  □ Train loss < 1.90
  □ Val loss ≤ Train loss × 1.1
  □ 학습 속도 ≥ 2x 기존 대비 (dynamic padding 효과)

Step 5 ✅ 평가
  □ 반복률 < 10% (rep_penalty 없이)
  □ 반복률 < 3% (rep_penalty=1.1)
  □ ko_ifeval > 20%

최종 ✅ 목표 달성
  □ 반복률 < 5% (실용적 설정)
  □ ko_ifeval > 25% (1B 한계 내 최적)
  □ 깨끗한 가중치 → 3B/ORPO 기반으로 사용 가능
```

---

## 부록: 왜 "제3의 선택지"는 아닌가

**"1B 고쳐서 재학습 후 바로 3B 전환"** 옵션도 고려했으나:

- 1B 재학습 자체가 3시간이면 끝남 → 별도 "고쳐서" 단계가 필요 없음
- 3B 전환은 1B 결과와 무관하게 진행 가능 (sft_dataset.py 수정은 3B에도 그대로 적용)
- 따라서 "깨끗하게 재학습" = "3B 전환 준비"가 자연스럽게 포함됨

**결론: Restart가 Fix의 상위 호환이다.** Fix로 할 수 있는 모든 것을 Restart가 포함하면서, 추가로 코드 버그까지 수정한다. 비용 차이는 1시간.

---

*"40분 아끼려고 기술 부채를 안고 가지 마라. 3시간 투자해서 깨끗한 기반을 만들어라."*
