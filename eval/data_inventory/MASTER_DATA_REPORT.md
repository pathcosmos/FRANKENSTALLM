# 한국어 LLM 데이터 종합 리포트
> 생성: 2026-02-27 | 5개 subagent 조사 결과 통합

---

## 1. 현재 보유 현황

| 카테고리 | 데이터셋 | 디스크 | 추정 토큰 | 품질 |
|---------|---------|--------|---------|------|
| 교육 웹 | fineweb2_edu_ko | 234G | ~50B | A |
| 웹 크롤 | culturax_ko | 60G | ~24B | B+ |
| 수학 | open_web_math | 26G | ~10B | A |
| 웹 크롤 | hplt_ko | 23G | ~9B | B |
| 웹 크롤 | cc100_processed | 19G | ~7B | C+ |
| 웹 크롤 | cc100_ko | 14G | ~5.5B | C |
| 웹 크롤 | oscar_ko | 9.2G | ~3.5B | B |
| 교육 | korean_textbooks | 6.4G | ~1.5B | A |
| 웹 | korean_webtext | 4.2G | ~1B | B+ |
| 백과 | namuwiki_2023 | 2.9G | ~1B | A- |
| 교육 | finepdfs_edu_ko | 2.9G | ~0.7B | A- |
| 백과 | namuwiki_extracted | 2.2G | ~0.5B | A- |
| 백과 | wikipedia_korean | 1.7G | ~0.4B | A |
| 백과 | wikipedia_ko_2024 | 1.4G | ~0.3B | A |
| Instruct | kovast | 449M | ~0.1B | B |
| Instruct | evol_instruct_ko | 144M | ~0.03B | B |
| 대화 | korean_safe_conv | 51M | ~0.01B | B |
| **합계** | | **~410G** | **~114B raw** | |

> ⚠️ 토큰화 완료 `.bin`: korean_train.bin(17G≈8.9B), korean_c4_train(15G≈7.5B) 등 실제 학습 사용 ~39B

---

## 2. 부족 도메인 갭 분석

### 🔴 CRITICAL (없음)
| 도메인 | 현황 | 영향 |
|--------|------|------|
| **Preference/DPO** | 0건 | ORPO 학습 불가 |
| **법률/판례** | 0 | 법률 추론 불가 |
| **의료/의학** | 0 | 헬스케어 응답 불가 |
| **코드 (한국어 주석)** | 0 | 코딩 지원 약함 |
| **뉴스/언론** | 0 | 시사 맥락 약함 |

### 🟡 WEAK (매우 부족)
| 도메인 | 현황 | 영향 |
|--------|------|------|
| **Instruction/SFT** | ~0.6G (644MB) | 지시 따르기 약함 |
| **금융/경제** | 0 | 금융 도메인 응답 약함 |
| **학술논문** | 0 | 학술적 글쓰기 약함 |
| **소설/문학** | 0 | 창작 능력 약함 |

---

## 3. 최고 후보군 — Pretrain 용 (부족 도메인 채우기)

### 🥇 1순위: KORMo-Team/korean-web-collection
- **크기**: ~50~80GB / ~20~30B 토큰
- **특징**: HF에서 가장 큰 한국어 전용 웹 크롤. 현재 보유 데이터와 중복 적음
- **라이선스**: 공개
- **다운로드**: `huggingface-cli download KORMo-Team/korean-web-collection --repo-type dataset --local-dir ./data/korean-web-collection`

### 🥈 2순위: HPLT/HPLT2.0_cleaned (ko)
- **크기**: ~30GB / ~12B 토큰
- **특징**: HPLT v1.2 이미 보유(23G) → v2.0은 더 크고 정제됨. 추가 순수 증가분 존재
- **라이선스**: 공개
- **다운로드**: `python -c "from datasets import load_dataset; ds = load_dataset('HPLT/HPLT2.0_cleaned', 'ko', split='train'); ds.save_to_disk('./data/hplt2-ko')"`

### 🥉 3순위: 법률 도메인 묶음
| 데이터셋 | 크기 | 내용 |
|---------|------|------|
| `joonhok-exo-ai/korean_law_open_data_precedents` | ~1-2G | 법원 판례 전문 |
| `smhilee/korean-law-dataset` | ~1-3G | 법령/법률 텍스트 |
| `Rootpye/korean-lawdata2` | ~0.5-1G | 법률 데이터 |
| `Rootpye/korean-lawdata4` | ~0.5-1G | 법률 데이터 v4 |
| `ducut91/korean-constitutional-court-decisions` | ~0.5G | 헌법재판소 결정 |
- **합계**: ~4~8G / ~1~2B 토큰
- **왜 중요**: 법률은 완전 공백 도메인. 정밀한 한국어 + 논리 구조 → pretrain 품질 향상

### 4순위: mc4 (ko)
- **크기**: ~50GB / ~20B 토큰
- **특징**: CulturaX와 일부 중복이나 원본 mC4 추가 텍스트 존재
- **라이선스**: 공개
- **다운로드**: `python -c "from datasets import load_dataset; ds = load_dataset('mc4', 'ko', split='train'); ds.save_to_disk('./data/mc4-ko')"`

### 5순위: RedPajama-Data-1T (코드+ArXiv)
- **크기**: 선별 ~15~20GB / ~8~10B 토큰
- **특징**: 한국어 모델이라도 코드+과학 영어 데이터 필수 (cross-lingual transfer)
- **서브셋**: `github` (코드 5B) + `arxiv` (과학 3B) + `book` (2B)
- **라이선스**: 공개

---

## 4. 최고 후보군 — SFT 용

### 🥇 1: kuotient/orca-math-word-problems-193k-korean
- **크기**: 193K 샘플
- **내용**: 수학 문제 한국어, Orca Math 기반
- **왜**: 수학 도메인 완전 공백 채움. 검증된 고품질

### 🥈 2: dbdu/ShareGPT-74k-ko
- **크기**: 74K 샘플
- **내용**: ChatGPT 실사용 대화 멀티턴 한국어 번역
- **왜**: 싱글턴 편향인 현재 데이터 보완, 다양한 도메인

### 🥉 3: nayohan/Evol-Instruct-Code-80k-v1-ko
- **크기**: 80K 샘플
- **내용**: WizardCoder 기반 코딩 instruction 한국어
- **왜**: 코딩 도메인 현재 ~5% → 대폭 강화

### 4: nlp-with-deeplearning/Ko.WizardLM_evol_instruct_V2_196k
- **크기**: 196K 샘플
- **내용**: WizardLM Evol Instruct 한국어 — 복잡한 추론 포함

### 5: FreedomIntelligence/alpaca-gpt4-korean
- **크기**: 52K 샘플
- **내용**: GPT-4 생성 Alpaca 한국어 — 고품질 응답

> **SFT 추가 후 예상**: 현재 162K + 595K = **~757K** (4.7배 증가)

---

## 5. 최고 후보군 — Preference/ORPO 용

### 🥇 1: jojo0217/korean_rlhf_dataset
- **크기**: 100K+ 쌍
- **내용**: 한국어 RLHF 종합 — 가장 범용적
- **우선순위**: 즉시 다운로드

### 🥈 2: maywell/ko_Ultrafeedback_binarized
- **크기**: ~60K 쌍
- **내용**: UltraFeedback 한국어 번역, binarized (chosen/rejected)
- **왜**: 이미 chosen/rejected 형식으로 ORPO 바로 사용 가능

### 🥉 3: nayohan/preference-collection-ko-full
- **크기**: 100K+ 쌍
- **내용**: 한국어 종합 preference 컬렉션

### 4: kuotient/orca-math-korean-dpo-pairs
- **크기**: 100K+ 쌍
- **내용**: 수학 특화 DPO 쌍

> **ORPO 추천 조합**: jojo0217 + maywell + nayohan = ~260K쌍 → 바로 시작 가능

---

## 6. 외부 소스 (신청 필요)

| 소스 | 추정량 | 특징 |
|------|--------|------|
| AI Hub (aihub.or.kr) | ~60~100GB | 뉴스, 대화, 의료, 법률, 금융 전문 — 승인 필요, 비상업적 가능 |
| NIKL 모두의 말뭉치 | ~35~50GB | 문어/구어 코퍼스, 비상업적 연구용 신청 |
| 국가법령정보센터 | ~5~10GB | 크롤링 가능 (공공 데이터) |
| KCI 학술논문 | ~3~5GB | 논문 초록, API 제공 |

---

## 7. 다운로드 실행 플랜 (우선순위순)

```bash
cd /PROJECT/0325120031_A/ghong/taketimes/llm-bang

# === Phase 1: Preference (ORPO 즉시 활성화, 소용량) ===
python3 -c "
from datasets import load_dataset
import os
out = 'data/preference'
os.makedirs(out, exist_ok=True)
for name in ['jojo0217/korean_rlhf_dataset', 'maywell/ko_Ultrafeedback_binarized', 'nayohan/preference-collection-ko-full', 'kuotient/orca-math-korean-dpo-pairs']:
    ds = load_dataset(name, split='train')
    ds.to_json(f'{out}/{name.replace(\"/\",\"_\")}.jsonl')
    print(f'✅ {name}: {len(ds)} samples')
" 2>&1 | tee /tmp/preference_dl.log &

# === Phase 2: SFT 보강 (대화/수학/코드) ===
python3 -c "
from datasets import load_dataset
import os
out = 'data/sft_extra'
os.makedirs(out, exist_ok=True)
for name in ['kuotient/orca-math-word-problems-193k-korean','dbdu/ShareGPT-74k-ko','nayohan/Evol-Instruct-Code-80k-v1-ko','nlp-with-deeplearning/Ko.WizardLM_evol_instruct_V2_196k','FreedomIntelligence/alpaca-gpt4-korean']:
    try:
        ds = load_dataset(name, split='train')
        ds.to_json(f'{out}/{name.replace(\"/\",\"_\")}.jsonl')
        print(f'✅ {name}: {len(ds)}')
    except Exception as e:
        print(f'❌ {name}: {e}')
" 2>&1 | tee /tmp/sft_extra_dl.log &

# === Phase 3: 법률 Pretrain 보강 ===
python3 -c "
from datasets import load_dataset
import os
out = 'data/korean_extra/korean_law'
os.makedirs(out, exist_ok=True)
for name in ['joonhok-exo-ai/korean_law_open_data_precedents','smhilee/korean-law-dataset','Rootpye/korean-lawdata2']:
    try:
        ds = load_dataset(name, split='train')
        ds.to_json(f'{out}/{name.replace(\"/\",\"_\")}.jsonl')
        print(f'✅ {name}: {len(ds)}')
    except Exception as e:
        print(f'❌ {name}: {e}')
" 2>&1 | tee /tmp/law_dl.log &

# === Phase 4: 대용량 Pretrain (백그라운드 장시간) ===
# mc4 Korean (~50GB)
# python3 -c "from datasets import load_dataset; ds = load_dataset('mc4', 'ko', split='train'); ds.save_to_disk('data/korean_extra/mc4_ko')"
# KORMo Web Collection
# huggingface-cli download KORMo-Team/korean-web-collection --repo-type dataset --local-dir data/korean_extra/korean_web_collection
```

---

## 8. 추가 후 예상 데이터 구성

| 카테고리 | 현재 토큰 | 추가 후 | 비고 |
|---------|---------|---------|------|
| 한국어 Pretrain | ~39B (토큰화) | ~60~80B | mc4+KORMo+법률 추가 시 |
| SFT | 162K | ~757K | 5개 추가 후 |
| Preference | 0 | ~260K쌍 | jojo+maywell+nayohan |
| 코드/영어 | ~0.6B | ~10B | RedPajama github+arxiv |
| 법률 | 0 | ~1~2B | 법률 묶음 |

**Chinchilla minimum (60B) 달성 가능** ✅

---

_보고서 저장: `/PROJECT/0325120031_A/ghong/taketimes/llm-bang/eval/data_inventory/`_
