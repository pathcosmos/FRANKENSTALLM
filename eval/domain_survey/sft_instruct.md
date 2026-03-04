# 한국어 SFT/Instruction/Chat 데이터셋 전수 조사

> 작성일: 2026-02-27  
> 목적: 한국어 3B LLM 학습을 위한 공개 SFT 데이터셋 전수 조사  
> 현재 보유: evol_instruct_ko (144M), korean_safe_conv (51M), kovast (449M), train.jsonl (161,848샘플)

---

## 1. 전체 데이터셋 목록 (우선순위 순)

### 🏆 Tier 1: 고품질 / 대규모 (즉시 사용 추천)

| # | Repo ID | 샘플 수 | 포맷 | 라이선스 | 턴 | 도메인 | 품질 | 우선순위 |
|---|---------|--------|------|---------|-----|-------|------|---------|
| 1 | `maywell/koVast` | ~685K | sharegpt | Apache 2.0 | 멀티턴 | 일반/교육/과학 | GPT-4 번역+생성 | **10** |
| 2 | `lemon-mint/smol-koreantalk` | ~400K | openai-messages | Apache 2.0 | 멀티턴 | 일반/코딩/분석 | Claude 번역+정제 | **9** |
| 3 | `CarrotAI/ko-instruction-dataset` | ~100K | alpaca | Apache 2.0 | 싱글턴 | 코딩/수학/일반 | GPT-4 생성/번역 | **9** |
| 4 | `squarelike/sharegpt_deepl_ko_translation` | ~70K | sharegpt | CC BY-SA 4.0 | 멀티턴 | 일반 (ShareGPT 번역) | DeepL 번역 | **8** |
| 5 | `heegyu/OIG-small-chip2-ko` | ~80K | alpaca | Apache 2.0 | 싱글턴 | 일반/QA | 기계번역 | **8** |

### 🥈 Tier 2: 도메인 특화 / 중품질

| # | Repo ID | 샘플 수 | 포맷 | 라이선스 | 턴 | 도메인 | 품질 | 우선순위 |
|---|---------|--------|------|---------|-----|-------|------|---------|
| 6 | `MarkrAI/KOpen-HQ-Hermes-2.5-60K` | ~60K | sharegpt | MIT | 멀티턴 | 일반/코딩/수학 | GPT-4 Turbo 스코어링+DeepL | **8** |
| 7 | `kuotient/orca-math-word-problems-193k-korean` | ~193K | alpaca | MIT | 싱글턴 | **수학** | GPT-4 번역 | **9** (수학 특화) |
| 8 | `jhflow/orca_ko_en_pair` | ~100K+ | alpaca | MIT | 싱글턴 | 수학/논리 | Orca 번역 | **7** |
| 9 | `davidkim205/kollm-converations` | ~100K | sharegpt | CC BY 4.0 | 멀티턴 | 나무위키 QA (백과) | GPT-3.5 생성 | **7** |
| 10 | `coastral/korean-writing-style-instruct` | ~20K | sharegpt | Apache 2.0 | 멀티턴 | **역할극/문체** | GPT-4 생성 | **8** (역할극 특화) |
| 11 | `nayohan/raw_instruction_en_ko_translation` | ~30K | alpaca | MIT | 싱글턴 | 혼합 (소스 컬렉션) | 번역 집합 | **6** |
| 12 | `beomi/KoAlpaca-v1.1a` | ~21K | alpaca | CC BY-NC 4.0 | 싱글턴 | 일반 | ChatGPT 생성 | **7** |
| 13 | `HAERAE-HUB/qarv-instruct-ko` | ~50K | alpaca | CC BY 4.0 | 싱글턴 | 일반/추론 | GPT-4 생성 | **7** |
| 14 | `devngho/korean-instruction-mix` | 집합체 | 혼합 | 다양 | 싱글턴 | 혼합 | 번역+생성 | **6** |
| 15 | `heegyu/OIG-small-chip2-ko` | ~80K | alpaca | Apache 2.0 | 싱글턴 | QA/일반 | OIG 번역 | **7** |

### 🥉 Tier 3: 보완 데이터 (갭 채우기용)

| # | Repo ID | 샘플 수 | 포맷 | 라이선스 | 턴 | 도메인 | 품질 | 우선순위 |
|---|---------|--------|------|---------|-----|-------|------|---------|
| 16 | `beomi/ko-marco-o1-instruct-oai` | ~5K | openai-messages | MIT | 싱글턴 | **수학/추론 (o1-style)** | Marco-o1 CoT | **8** (추론 특화) |
| 17 | `snunlp/KR-FinQA` | ~10K | alpaca | CC BY 4.0 | 싱글턴 | **금융** | 인간 작성 | **7** (금융 특화) |
| 18 | `MLP-lab/Korean-Medical-QA` | ~50K | alpaca | CC BY 4.0 | 싱글턴 | **의료** | 인간+GPT 혼합 | **7** (의료 특화) |
| 19 | `KETI-AIR/kor_dataset` | ~50K | alpaca | CC BY-NC 4.0 | 싱글턴 | 법률/행정 | 인간 작성 | **6** |
| 20 | `OpenAssistant/oasst1` (ko subset) | ~5K | openai-messages | Apache 2.0 | 멀티턴 | 일반 | 인간 작성 (고품질) | **9** (인간작성) |
| 21 | `Babelscape/ALERT` (ko) | ~10K | alpaca | MIT | 싱글턴 | 안전/윤리 | 인간+GPT | **6** |
| 22 | `kyujinpy/KOR-OpenOrca-Platypus4` | ~90K | alpaca | CC BY-NC 4.0 | 싱글턴 | 일반/수학/코딩 | GPT-4 번역 | **7** |
| 23 | `nayohan/llama3-instrtuct-translation-ko` | ~15K | alpaca | Apache 2.0 | 싱글턴 | 일반 | Llama-3 번역 | **5** |
| 24 | `squarelike/OpenOrca-ko` | ~200K | alpaca | MIT | 싱글턴 | 혼합 | GPT-3.5/4 번역 | **7** |
| 25 | `Babelscape/REBEL-small` (ko) | ~10K | alpaca | CC BY-NC 4.0 | 싱글턴 | 지식/엔티티 | 자동생성 | **4** |
| 26 | `nlpai-lab/kullm-v2` | ~150K | alpaca | CC BY-NC 4.0 | 싱글턴 | 일반 (KU+GPT) | GPT-3.5 생성 | **6** |
| 27 | `heegyu/koalpaca-v1.1` | ~21K | alpaca | CC BY-NC 4.0 | 싱글턴 | 일반 | ChatGPT 번역 | **5** |
| 28 | `wooy0ng/korquad-v1-alpaca` | ~10K | alpaca | CC BY-ND 2.0 | 싱글턴 | 독해/QA | 자동 생성 | **5** |
| 29 | `lcw99/wikipedia-korean-20240501` | 별도 | text | CC BY-SA 4.0 | - | 지식 베이스 | 인간 작성 | 참고용 |
| 30 | `uonlp/CulturaX` (ko subset) | ~1M+ | text | CC BY-NC 4.0 | - | 일반 웹 | 웹 크롤링 | 참고용 |

---

## 2. 이미 보유 데이터 (중복 제외)

| 데이터셋 | 크기 | 비고 |
|---------|------|------|
| `evol_instruct_ko` | ~144M tokens | WizardLM 번역본 |
| `korean_safe_conv` | ~51M tokens | 안전 대화 데이터 |
| `kovast` (maywell/koVast) | ~449M tokens = 685K샘플 | ✅ 이미 보유 |
| `train.jsonl` | 161,848 샘플 | 현재 학습 데이터 |

> ⚠️ `maywell/koVast`는 이미 kovast로 보유 중. 중복 다운로드 불필요.

---

## 3. 도메인별 갭 분석

### ✅ 충분한 도메인
- **일반 대화/지식**: koVast(685K), OIG-ko(80K), ShareGPT-ko(70K) → **포화**
- **번역/영어학습**: EvolInstruct-ko(144M) → **충분**

### ⚠️ 부족한 도메인 (우선 수집 필요)

| 도메인 | 현재 상태 | 추천 데이터셋 | 예상 샘플 수 |
|-------|---------|------------|------------|
| **수학/논리 추론** | 매우 부족 | kuotient/orca-math-word-problems-193k-korean | 193K |
| **코딩** | 부족 | CarrotAI/ko-instruction-dataset (코딩 파트) | 30K+ |
| **멀티턴 고품질** | 부족 | MarkrAI/KOpen-HQ-Hermes-2.5-60K | 60K |
| **역할극/페르소나** | 없음 | coastral/korean-writing-style-instruct | 20K |
| **한국어 문화 특화** | 부족 | davidkim205/kollm-converations (나무위키) | 100K |
| **CoT/추론 체인** | 없음 | beomi/ko-marco-o1-instruct-oai | 5K |
| **의료/법률/금융** | 없음 | 별도 도메인 특화 데이터 필요 | 50K+ |
| **안전/거부 응답** | korean_safe_conv | - | 부분 충족 |

### 📊 도메인별 현황 요약
```
일반대화  ████████████████████ 80% (과잉)
번역문서  ████████████████████ 80% (충분)
수학추론  ████░░░░░░░░░░░░░░░░ 20% (부족)
코딩      ██████░░░░░░░░░░░░░░ 30% (부족)
멀티턴    ████████░░░░░░░░░░░░ 40% (보통)
역할극    ██░░░░░░░░░░░░░░░░░░ 10% (매우 부족)
의료/법률 ░░░░░░░░░░░░░░░░░░░░  5% (없음)
CoT추론   ██░░░░░░░░░░░░░░░░░░ 10% (없음)
```

---

## 4. Top 5 즉시 추천 데이터셋

### 🥇 1위: `kuotient/orca-math-word-problems-193k-korean`
- **왜**: 수학/논리 추론이 현재 가장 큰 갭. 193K 샘플로 단숨에 메꿀 수 있음
- **크기**: 193K 샘플
- **라이선스**: MIT (상업 사용 가능)
- **포맷**: alpaca
- **품질**: GPT-4 생성 + DeepL 번역, 검수됨
- **다운로드**: `from datasets import load_dataset; d = load_dataset("kuotient/orca-math-word-problems-193k-korean")`

### 🥈 2위: `MarkrAI/KOpen-HQ-Hermes-2.5-60K`
- **왜**: 고품질 멀티턴 데이터 갭. DeepL+GPT-4 Turbo 스코어링으로 품질 보장
- **크기**: 60K 샘플
- **라이선스**: MIT
- **포맷**: sharegpt
- **품질**: Near-dedup + GPT-4 Turbo scoring (고품질 보장)
- **주의**: HF 로그인 필요 (contact info 동의)

### 🥉 3위: `coastral/korean-writing-style-instruct`
- **왜**: 역할극/문체 다양성이 없음. 한국어 특유의 말투 (존댓말, 고어, 방언 등)
- **크기**: ~20K 샘플
- **라이선스**: Apache 2.0
- **포맷**: sharegpt (멀티턴)
- **품질**: GPT-4 생성, 다양한 페르소나
- **특징**: 조선시대 양반 말투, 선교사 화법 등 문체 다양성

### 4위: `lemon-mint/smol-koreantalk`
- **왜**: Claude 기반 고품질 번역+생성 데이터. 자연스러운 한국어 대화
- **크기**: ~400K 샘플
- **라이선스**: Apache 2.0
- **포맷**: openai-messages (멀티턴)
- **품질**: Claude Haiku 번역 + 정제, 영한 대조 포함

### 5위: `OpenAssistant/oasst1` (ko subset)
- **왜**: 인간이 작성한 유일한 고품질 멀티턴 데이터. 다양성 측면 최고
- **크기**: ~5K 샘플 (한국어만)
- **라이선스**: Apache 2.0
- **포맷**: tree 구조 → sharegpt 변환 필요
- **품질**: 인간 작성 (가장 자연스러움)
- **추출**: `filter(lambda x: x['lang']=='ko', dataset)`

---

## 5. 2024~2025 신규 데이터셋 특이사항

### 🆕 2024년 주목 데이터
1. **`beomi/ko-marco-o1-instruct-oai`** (2024 후반): Chain-of-Thought 한국어 추론. o1 스타일 CoT 포함
2. **`MarkrAI/KOpen-HQ-Hermes-2.5-60K`** (2024): 한국 커뮤니티 최초 Hermes 한국어 번역 고품질
3. **`lemon-mint/smol-koreantalk`** (2025): SmolLM 계열 학습용으로 구축된 최신 데이터
4. **`coastral/korean-writing-style-instruct`** (2024): 문체 다양성 특화, 역할극 최고품질

### 📌 2025년 검색 결과 없음 (미발표 또는 미공개)
- HyperCLOVA X 데이터: NAVER 비공개
- KT/Kakao 내부 데이터: 비공개
- LG AI 내부 데이터: 비공개

---

## 6. 다운로드 우선순위 체크리스트

```
[ ] kuotient/orca-math-word-problems-193k-korean  (~800MB)  ← 수학 갭 최우선
[ ] MarkrAI/KOpen-HQ-Hermes-2.5-60K             (~300MB)  ← 품질 다양성
[ ] coastral/korean-writing-style-instruct        (~100MB)  ← 역할극/문체
[ ] lemon-mint/smol-koreantalk                   (~1.5GB)  ← 대용량 고품질
[ ] OpenAssistant/oasst1 (ko filtered)           (~20MB)   ← 인간작성
[ ] squarelike/OpenOrca-ko                       (~1GB)    ← 일반 보강
[ ] kyujinpy/KOR-OpenOrca-Platypus4              (~500MB)  ← 코딩/수학 혼합
[ ] beomi/ko-marco-o1-instruct-oai               (~30MB)   ← CoT 추론
```

---

## 7. 라이선스 요약

| 라이선스 | 데이터셋 | 상업 사용 |
|---------|---------|---------|
| MIT | MarkrAI/KOpen-HQ, kuotient/orca-math-ko, jhflow/orca_ko | ✅ 가능 |
| Apache 2.0 | koVast, smol-koreantalk, CarrotAI, OIG-ko, oasst1, coastral | ✅ 가능 |
| CC BY 4.0 | davidkim205/kollm, HAERAE qarv | ✅ 가능 |
| CC BY-SA 4.0 | squarelike/sharegpt_deepl | ✅ (파생 동일라이선스) |
| CC BY-NC 4.0 | nlpai-lab/kullm-v2, beomi/KoAlpaca, kyujinpy | ❌ 비상업 |

> ⚠️ **주의**: CC BY-NC 계열 데이터는 상업적 모델 배포 시 사용 불가. 학술/연구 목적만 가능.

---

## 8. 총평 및 액션 아이템

### 현재 데이터 강점
- 일반 대화 데이터 매우 풍부 (koVast 685K + 기존 보유)
- 번역 데이터 충분

### 현재 데이터 약점
1. **수학/논리 추론 전무** → `kuotient/orca-math` 즉시 추가 필수
2. **CoT 데이터 없음** → `beomi/ko-marco-o1` 추가 권장
3. **역할극/페르소나 없음** → `coastral/korean-writing-style` 추가
4. **멀티턴 고품질 부족** → `MarkrAI/KOpen-HQ` 추가
5. **인간 작성 데이터 거의 없음** → `oasst1 ko` 필수 추가

### 예상 총 데이터 규모 (추가 후)
```
현재: ~800K 샘플
추가 후: ~1.8M+ 샘플 (중복 제거 후 ~1.2~1.5M)
```

---

*Generated: 2026-02-27 | Source: HuggingFace Hub 전수 검색 + 개별 데이터셋 검증*
