# 프로젝트 히스토리: 한국어 LLM 개발 전 과정

**프로젝트**: Korean LLM (1B → 3B)  
**경로**: `/PROJECT/0325120031_A/ghong/taketimes/llm-bang/`  
**작성일**: 2026-02-27  
**작성자**: 옵티머스프라임 (AI)

---

## 📖 전체 흐름 요약

```
[1단계] 기반 구축        → 서버 확인, 데이터 수집, 토크나이저 선택
[2단계] 125M 소형 테스트  → 코드베이스 검증, FP8 작동 확인
[3단계] 1B 사전학습       → 34,000 steps, loss 1.904, 8.93B 토큰
[4단계] 1B 평가           → PPL 측정, 반복률 30%+ 확인
[5단계] SFT v1 (실패)     → label 버그로 loss→0, 아카이브
[6단계] 근본 원인 분석     → 5개 subagent 독립 조사
[7단계] SFT v2 (성공)     → 9,000 steps, val_loss 2.2062
[8단계] 1B 한계 확인       → 반복률 18.8%, 구조적 한계
[9단계] 3B 결정            → Chinchilla 스케일링 법칙 기반
[10단계] 데이터 대규모 확보 → 640GB+, 10개 도메인
[현재] 3B 사전학습 준비 중
```

---

## 1단계: 기반 구축 (2026-02-25 초반)

### 서버 환경 확인

| 항목 | 사양 |
|------|------|
| GPU | 8× B200 (183GB/GPU) |
| CPU | 고성능 멀티코어 |
| 스토리지 | /PROJECT 20TB |
| /home/ghong | 5GB (FULL — 모든 파일 /PROJECT 또는 /tmp에 저장) |
| CUDA | FP8 지원 (Transformer Engine) |

### 초기 데이터 현황

- 기존 보유: C4 한국어, 나무위키, 한국어 위키피디아
- **총합**: ~18.4B 토큰 (토큰화 완료 기준)
  - `korean_train.bin`: 8.93B
  - `korean_c4_train.bin`: 7.56B
  - `korean_namuwiki_train.bin`: 1.08B
  - `korean_wiki_train.bin`: 0.26B
  - `train.bin` (영어): 0.60B

### 토크나이저 선택

- **채택**: `EleutherAI/polyglot-ko-12.8b` 토크나이저
- 어휘 크기: 64,000 토큰
- 한국어 형태소 처리 적합

---

## 2단계: 125M 소형 테스트 (2026-02-25)

### 목적
- FP8 학습 파이프라인 검증
- 코드베이스 (Transformer Engine) 오류 확인
- GPU 8개 DDP 정상 작동 확인

### 결과
- **체크포인트**: `checkpoints/small_fp8_run1/` (최다 103개 체크포인트)
- FP8 정상 동작 확인
- Multi-GPU DDP 안정 확인
- 파이프라인 검증 완료 → 1B로 확장 결정

---

## 3단계: 1B 사전학습 (2026-02-25 ~ 2026-02-26)

### 모델 아키텍처

| 파라미터 | 값 |
|---------|-----|
| d_model | 2048 |
| n_layers | 24 |
| n_heads | 16 |
| n_kv_heads | 4 (GQA 4:1) |
| d_ffn | 5472 (SwiGLU) |
| vocab_size | 64,000 |
| max_seq_len | 4,096 |
| rope_theta | 500,000 |
| **총 파라미터** | **~1.19B** |

### 학습 설정

| 하이퍼파라미터 | 값 |
|-------------|-----|
| dtype | FP8 (Transformer Engine) |
| optimizer | AdamW |
| lr | 3e-4 (cosine decay) |
| warmup steps | 1,000 |
| batch size | 8/GPU × 8 GPU = 64 (global) |
| seq_len | 4,096 |
| effective batch | ~262,144 토큰/step |
| 총 토큰 | ~8.93B |
| GPU | 8× B200 |

### 학습 경과

- 총 **34,000 steps** 완주
- 최종 loss: **1.904** (안정적 수렴)
- 체크포인트: `checkpoints/korean_1b_fp8_run1/checkpoint-0034000`

### 주요 milestone

| Step | Loss | 비고 |
|------|------|------|
| 1,000 | ~2.8 | 초기 수렴 |
| 10,000 | ~2.2 | 급격 하락 |
| 20,000 | ~2.0 | 완만한 개선 |
| 34,000 | 1.904 | 최종 수렴 |

---

## 4단계: 1B 기본 평가 (2026-02-26)

### Perplexity 측정

| 데이터셋 | PPL |
|---------|-----|
| C4 (한국어) | **5.67** |
| Wikipedia (한국어) | **11.66** |
| Namuwiki | **25.34** |

### 반복률 초기 측정 (잘못된 방법)

- 최초 측정 시 `### 질문/답변` 포맷 사용 → **57% 반복률**
- → 이후 올바른 포맷으로 재측정 필요 확인

### 평가 결론

- PPL은 준수 (C4 5.67은 양호)
- Pretrain 자체는 성공적
- SFT 필요 — 반복 문제는 SFT 단계에서 해결 가능할 것으로 기대

---

## 5단계: SFT v1 (실패, 2026-02-26)

### 데이터 준비

- SFT 데이터: `data/sft/train.jsonl` 161,848 샘플
- `data/sft/val.jsonl` 8,518 샘플
- 소스: evol_instruct_ko, korean_safe_conv 등

### 발생한 버그

**Label off-by-one 버그** (`data/sft_dataset.py`):
```python
# 버그: input_ids[1:]을 labels로 사용
# 결과: assistant 응답만 학습해야 하는데, 전체 sequence를 shift해서 복사 과제가 됨
# loss → 0 (모델이 단순 복사를 학습)
```

### 결과

- Loss가 0으로 수렴 (복사 과제)
- 모델이 실질적 언어 생성 능력 학습 실패
- **아카이브**: `checkpoints/korean_1b_sft_v1_backup/`
- → **사용 금지** (label 버그로 무효)

---

## 6단계: 근본 원인 분석 (2026-02-27 새벽)

### 5개 독립 Subagent 조사

5개의 Claude Opus 4.6 에이전트가 독립적으로 문제를 분석:

| 에이전트 | 발견 사항 |
|---------|---------|
| Agent 1 | Label off-by-one 버그 (sft_dataset.py) |
| Agent 2 | Static padding 4,096 고정 → FLOPs 85% 낭비 |
| Agent 3 | EOS 토큰 truncation 버그 (response_ids[:allowed] 가 EOS 제거) |
| Agent 4 | 데이터 오염 (</s> 리터럴 113건, Q/A 마커 ~550건) |
| Agent 5 | 평가 포맷 불일치 (학습: `<\|user\|>` 포맷, 평가: `### 질문/답변` 포맷) |

### 핵심 결론: 수정 vs 재시작

- **3 agents 만장일치**: 재시작이 옳다
- Fix 비용 ~1hr vs 5개 technical debt 누적 위험
- **결정**: 재시작 (SFT v2)

### 보고서 위치

`eval/` 디렉토리 내 각 조사 보고서 저장

---

## 7단계: SFT v2 (성공, 2026-02-27 01:18 KST)

### 핵심 수정 사항

| 버그 | 수정 내용 |
|------|---------|
| Label off-by-one | `labels = input_ids.clone(); labels[:prompt_len] = -100` |
| Static padding | `dynamic_collate_fn` 도입 — 배치별 실제 max 길이로 패딩 |
| EOS truncation | `response_ids[:allowed-1] + [eos_id]` |
| 데이터 오염 | `</s>` 리터럴 113건, Q/A 마커 ~550건 필터링 |
| Val loop 버그 | gradient 미계산 문제 수정 |

### 학습 설정 (v2)

| 하이퍼파라미터 | 값 |
|-------------|-----|
| lr | 2e-5 |
| warmup steps | 300 |
| total steps | 9,000 (~3 epochs) |
| NEFTune alpha | 5 |
| effective batch | 64 (8 GPU × 8) |
| val split | 10% |

### 중간 사고

- Step 6,000에서 `kill -9` 오용으로 DataLoader worker 사망
- `--resume` 옵션으로 복구 성공
- **영구 규칙**: kill 시 반드시 메인 PID만 정확히 지정

### 결과

| 체크포인트 | Step | val_loss |
|-----------|------|----------|
| checkpoint-best | 8,750 | **2.2062** ← 최선 |
| checkpoint-0009000 | 9,000 | 2.2079 |

---

## 8단계: 1B SFT v2 평가 (2026-02-27)

### 반복률 측정 (올바른 포맷)

| 조건 | 반복률 |
|------|--------|
| `### 질문/답변` (잘못된 포맷) | 57% |
| `<\|user\|>\n{Q}\n<\|assistant\|>\n` (올바른 포맷) | 30.7% |
| 올바른 포맷 + rep_penalty=1.1 | **18.0%** |
| 올바른 포맷 + rep_penalty=1.1 + no_repeat_ngram=3 | **17.7%** |

### lm-eval 벤치마크 (SFT v2 best, 2026-02-27)

| Task | Score | 비고 |
|------|-------|------|
| kobest_boolq | 0.5000 | Random baseline = 0.50 |
| kobest_copa | **0.6460** | Random baseline = 0.50 ✅ |
| haerae_general_knowledge | 0.2273 | Random ~0.25 |
| haerae_history | 0.1543 | Random ~0.25 ❌ |
| paws_ko | 0.4900 | Random = 0.50 |

### Decision Gate 판정

- **반복률 18%** → 5~15% 구간 → ORPO 권장이지만...
- **haerae scores 낮음** → 도메인 지식 부족
- **근본 원인**: 1B d_model=2048, 24 layers — 긴 컨텍스트에서 hidden state 붕괴
- **결론**: ORPO로 band-aid 붙이는 것보다 **3B로 확장이 올바른 방향**

---

## 9단계: 3B 결정 (2026-02-27)

### 근거

1. **Scaling Law**: 1B → 3B 전환 시 loss ~7% 감소 예상
2. **반복률 예측**: 3B SFT → 5~8%, +ORPO → <3%
3. **Chinchilla optimal**: 3B × 70 = **210B 토큰** 필요
4. **시간 비교**:
   - ORPO 시도 → 3B (ORPO 실패 가정): ~39h
   - 3B 직행: ~30h
   - → **3B 직행이 합리적**

### 3B 벤치마크 (2026-02-27 04:32~04:35)

| 항목 | 값 |
|------|-----|
| 처리량 | **36,250 tok/s** |
| VRAM 사용량 | **47.8GB/GPU** (B200 183GB의 26%) |
| 파라미터 수 | 2,386,987,520 (2.39B, 실제 측정) |
| 학습 속도 | 0.3 steps/s |
| Effective batch | 1,048,576 토큰/step |
| 예상 학습 시간 | ~25~30시간 (200B 토큰 기준) |

---

## 10단계: 대규모 데이터 확보 (2026-02-27)

### 최종 확보 데이터

| 카테고리 | 경로 | 크기 | 샘플 수 |
|---------|------|------|---------|
| 한국어 웹 (KORMo) | korean_web_collection | 175GB | 378 files |
| 한국어 공공 (KORMo) | korean_public_corpus | 26GB | 171 files |
| 한국어 교육 웹 | fineweb2_edu_ko | 234GB | 506 files |
| 한국어 법률 | korean_law | 15GB | 1.94M |
| 도메인 특화 | domain_specific | 21GB | 6.1M+ |
| CulturaX 한국어 | culturax_ko | 60GB | — |
| HPLT 한국어 | hplt_ko | 23GB | — |
| 수학 (영어) | open_web_math | 26GB | — |
| Preference/DPO | data/preference | 7.9GB | 795,468쌍 |
| SFT 추가 | data/sft_extra | 9.6GB | 1,084,752 |
| **합계** | | **~640GB+** | |

### 10개 도메인 조사 완료

`eval/domain_survey/` — news, legal, medical, finance, code_math, academic, literature, government, sft_instruct, preference_pretrain

---

## 현재 상태 (2026-02-27 12:00 KST)

| 항목 | 상태 |
|------|------|
| 1B pretrain | ✅ 완료 |
| 1B SFT v2 | ✅ 완료 |
| 1B eval | ✅ 완료 (반복 18%, kobest_copa 64.6%) |
| 데이터 확보 | ✅ 640GB+ |
| 추가 데이터 다운로드 | 🔄 진행 중 (Cosmopedia, GitHub-code 등) |
| 3B 사전학습 | ⏳ 데이터 전처리 → 학습 시작 대기 |
| ORPO (1B) | ⏳ 선택적 (3B 우선) |

---

*이 문서는 프로젝트 전체 의사결정 흔적 보존용입니다.*
