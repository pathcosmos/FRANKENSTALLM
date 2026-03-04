# 한국어 Preference/DPO/RLHF + 대용량 Pretrain 데이터 전수 조사

> 작성일: 2026-02-26  
> 목적: 3B 한국어 LLM 학습용 데이터 소스 파악  
> 조사 방법: HuggingFace 데이터셋 페이지 직접 web_fetch (Brave API 미사용)

---

## 목차
1. [Preference / DPO / RLHF 데이터셋](#preference--dpo--rlhf-데이터셋)
2. [대용량 Pretrain 데이터셋](#대용량-pretrain-데이터셋)
3. [Top 3 권장 - Preference](#top-3-권장---preference)
4. [Top 3 권장 - Pretrain](#top-3-권장---pretrain)
5. [갭 분석 및 메모](#갭-분석-및-메모)

---

## Preference / DPO / RLHF 데이터셋

### 전체 목록

| # | Repo ID | 규모 | 포맷 | 도메인 | 라이선스 | ORPO/DPO 직접 사용 | 우선순위 |
|---|---------|------|------|--------|----------|-------------------|---------|
| 1 | `kuotient/orca-math-korean-dpo-pairs` | ~193k 쌍 | `{prompt, chosen, rejected}` | 수학 | MIT (원본 기반 추정) | ✅ 직접 사용 가능 | **9** |
| 2 | `kuotient/orca-math-korean-preference` | ~193k 쌍 | `{prompt, chosen, rejected}` | 수학 | Apache 2.0 후보 | ✅ 직접 사용 가능 | **9** |
| 3 | `heegyu/orca-math-korean-preference-cleaned` | ~192k 쌍 | `{prompt, chosen, rejected, correctness_label}` | 수학 (KO+EN 이중) | MIT 추정 | ✅ 직접 사용 가능 (correctness로 추가 필터링 가능) | **8** |
| 4 | `maywell/ko_Ultrafeedback_binarized` | ~60k 쌍 추정 | `{prompt, chosen, rejected}` | 일반 (번역) | MIT 추정 | ✅ 직접 사용 가능 | **8** |
| 5 | `lemon-mint/korean-realqa-reasoning-v01-preference` | ~7.77k 쌍 | `{id, prompt, chosen, rejected}` | 일반 QA + 추론 | 미상 | ✅ 직접 사용 가능 (chosen에 `<think>` CoT 포함) | **7** |
| 6 | `ohsuz/dpo-v1010-korean` | ~35.5k | `{prompt, chosen, rejected}` 추정 | 금융 포함 다도메인 | 미상 (gated) | ⚠️ gated, 사전 동의 필요 | **6** |
| 7 | `ChuGyouk/argilla-distilabel-math-preference-dpo-korean` | ~2.42k 쌍 | `{prompt, chosen, rejected}` | 수학 | MIT 추정 | ✅ 직접 사용 가능 (소규모) | **4** |
| 8 | `jojo0217/korean_rlhf_dataset` | ~107k QA | `{question, answer}` single-turn | 과학/역사/문화/음식/의학/법 | 미상 | ❌ DPO 직접 불가 (단일 응답, SFT용) | **3** |

### 주요 데이터셋 상세

#### 1. `kuotient/orca-math-korean-dpo-pairs` ⭐ 최고 우선순위
- **규모**: 193,000 쌍
- **스키마**: `{prompt: str, chosen: str, rejected: str}`
- **특징**: Microsoft OrcaMath 한국어 번역. 수학 문제 풀이 과정 비교. HF에서 가장 많이 다운로드된 한국어 DPO 데이터셋 (111 downloads)
- **사용법**: `load_dataset("kuotient/orca-math-korean-dpo-pairs")`
- **주의**: 수학 도메인에 특화 → 일반 능력 향상에는 보완 필요

#### 2. `kuotient/orca-math-korean-preference` ⭐ 최고 우선순위
- **규모**: 193,000 쌍
- **특징**: dpo-pairs와 동일 소스지만 다른 포맷 버전. 라이선스 더 명확
- **사용법**: 위와 동일 저자, 상호 보완 또는 대체 사용

#### 3. `heegyu/orca-math-korean-preference-cleaned` ✅ 권장
- **규모**: ~192k 쌍
- **스키마**: `{prompt, chosen, rejected, is_correct: bool}` 
- **특징**: `is_correct=True`인 샘플만 필터링 가능 → 고품질 서브셋 추출 가능
- **특이사항**: KO+EN 이중언어 (한국어 번역 + 원문 포함)

#### 4. `maywell/ko_Ultrafeedback_binarized` ✅ 일반 도메인 보완용
- **규모**: UltraFeedback binarized 원본 (~60k) 한국어 번역
- **특징**: 일반 domain preference (수학 외) → 수학 DPO의 편향 보완
- **스키마**: `{prompt, chosen, rejected}` 표준 포맷
- **데이터 예시 확인**: 자연어 처리, 역사, 정치 등 다양한 주제

#### 5. `lemon-mint/korean-realqa-reasoning-v01-preference` ✅ CoT 학습용
- **규모**: 7,770 쌍
- **특징**: chosen에 `<think>...</think>` CoT 추론 흔적 포함 → reasoning 모델 학습에 적합
- **날짜**: 2025년 2월 신규 릴리즈
- **사용법**: ORPO 학습 시 reasoning 능력 부여에 적합

#### 6. `ohsuz/dpo-v1010-korean` ⚠️ 조건부
- **규모**: 35,500 쌍
- **접근**: Gated (로그인 + 연락처 동의 필요)
- **특징**: 금융 버전도 별도 존재 (`ohsuz/dpo-finance-korean`)
- **README**: 비어있음 → 실제 다운로드 전 포맷 미확인

#### 7-8. 소규모 / SFT 전용
- `ChuGyouk/...`: 2.42k로 너무 소규모, 보조용
- `jojo0217/korean_rlhf_dataset`: chosen/rejected 없음 → SFT 데이터로만 활용 가능

---

## 대용량 Pretrain 데이터셋

### 현재 보유 현황
- 토큰화 완료: ~39B 토큰
- Raw 포함: ~114B (중복 포함)
- 주요 소스: CulturaX(ko), HPLT v1.0, cc100-ko, OSCAR 등

### 전체 목록

| # | Repo ID | 크기 | 기존 소스 중복 | 라이선스 | 필터링 수준 | 우선순위 |
|---|---------|------|---------------|----------|------------|---------|
| 1 | `KORMo-Team/korean-web-collection` | ~수십GB (미확인) | ⚠️ 부분 중복 가능 (blog/news) | 미상 | 중간 (cleaned) | **9** |
| 2 | `KORMo-Team/korean-public-corpus` | ~수GB (미확인) | ✅ 비중복 (학술/공공 도메인) | 공공저작물 | 높음 | **9** |
| 3 | `uonlp/CulturaX` (ko) | ~24.8B 토큰 (~20.5M 문서) | ❌ **보유 중** (mC4 + OSCAR) | CC BY-NC 4.0 (gated) | 높음 (deduped) | **이미 보유** |
| 4 | `HAERAE-HUB/KOREAN-WEBTEXT` | 1.28M docs | ⚠️ 중복 (source=oscar2201) | 미상 | 중간 | **5** |
| 5 | `devngho/korean-webtext-edu` | 1.28M docs (edu 필터) | ⚠️ KOREAN-WEBTEXT 기반 | MIT (원본 라이선스 불명확) | 높음 (edu classifier) | **7** |
| 6 | `oz1115/korean-pretraining-corpus` | 1K~10K rows (소규모) | ⚠️ 위키피디아 포함 | MIT | 중간 (이미 토큰화됨, 512 tok chunks) | **2** |
| 7 | `Saxo/Korean-Corpus-From-Various-Task-1` | ~524k rows | ⚠️ 다양한 소스 혼합 | 미상 | 낮음 (raw) | **4** |
| 8 | `91veMe4Plus-Project/korean_*` | 미확인 (도메인별) | ✅ 비중복 가능성 높음 | 미상 | 도메인별 | **5** |

### 주요 데이터셋 상세

#### 1. `KORMo-Team/korean-web-collection` ⭐ 최고 우선순위
- **내용**: 종교, 백과사전, 뉴스, 블로그 등 다양한 한국어 웹 크롤
- **특징**: KORMo 팀의 대규모 한국어 웹 컬렉션. 별도 도메인 서브셋 구성
- **중복 위험**: 뉴스/블로그 부분은 CC100/OSCAR와 일부 겹칠 수 있음
- **권장 사용**: 중복 제거(MinHash LSH) 후 사용

#### 2. `KORMo-Team/korean-public-corpus` ⭐ 최고 우선순위
- **내용**: 논문, 공공 문서, 학술 텍스트
- **특징**: 웹 크롤 기반 코퍼스와 도메인 비중복 → 순수 증가분으로 가치 높음
- **라이선스**: 공공저작물 (사용 가능)
- **권장 사용**: 학술/전문 도메인 커버리지 향상에 핵심

#### 3. `devngho/korean-webtext-edu` ✅ 고품질 선별용
- **기반**: `HAERAE-HUB/KOREAN-WEBTEXT`에 교육 품질 분류기(`ko_edu_classifier_v2`) 적용
- **스코어**: `scored_over_3` 서브셋으로 고품질만 선택 가능
- **하드웨어**: TPU v4-8 × 4 인스턴스로 처리 (~35분)
- **라이선스**: MIT (단, 원본 KOREAN-WEBTEXT 라이선스 불명확 → 확인 필요)
- **접근**: Gated (로그인 + 동의 필요)
- **중복 주의**: KOREAN-WEBTEXT가 oscar2201 기반 → 기존 OSCAR 보유분과 중복 가능

#### 4. `HAERAE-HUB/KOREAN-WEBTEXT`
- **규모**: 1.28M 문서
- **스키마**: `{text, source, token_count, __index_level_0__}`
- **source**: oscar2201 (OSCAR 2022.01 기반)
- **중복 경고**: ❌ 기존 OSCAR 보유 가능성 높음 → 사용 전 중복 체크 필수
- **용도**: 기존 OSCAR 버전 다르다면 보완 가능

#### 5. `uonlp/CulturaX` (ko) — 이미 보유
- **크기**: ~20.5M 문서, ~24.8B 토큰 (전체의 0.39%)
- **소스**: mC4 + OSCAR 혼합
- **라이선스**: CC BY-NC 4.0 (non-commercial, gated)
- **스키마**: `{text, timestamp, url, source}`
- **참고**: 이미 39B 토큰에 포함된 것으로 파악됨

#### 6. `oz1115/korean-pretraining-corpus` — 소규모, 참고만
- **크기**: 1K~10K rows (매우 소규모)
- **내용**: 한국어 Wikipedia + 공개 텍스트
- **형태**: 이미 BPE 토큰화됨, 512 토큰 청크 형식 (raw 텍스트 불가)
- **결론**: 3B 학습용 대용량 소스로 부적합

### 추가 발굴 필요 소스 (web_search 미사용으로 미확인)

| 소스 | 예상 크기 | 조사 방법 |
|------|-----------|---------|
| HPLT v2.0 한국어 | 수백GB 추정 | `web_fetch https://data.hplt-project.org/` 재시도 |
| PleIAs/common_corpus (ko) | 수십GB 추정 | HF 직접 확인 |
| NLLB data (flores 기반) | 미상 | HF 검색 |
| 국립국어원 공개 말뭉치 | ~수GB | 별도 공식 포털 |
| AI Hub 한국어 코퍼스 | 수백GB | 별도 신청 필요 |

---

## Top 3 권장 - Preference

### 🥇 1위: `kuotient/orca-math-korean-dpo-pairs`
- **선정 이유**: 193k쌍 대용량, 표준 {prompt/chosen/rejected} 포맷, 가장 많이 검증된 한국어 DPO 데이터셋
- **바로 사용**: `load_dataset("kuotient/orca-math-korean-dpo-pairs")`
- **주의**: 수학 편향 → 단독 사용 시 일반 능력 저하 가능

### 🥈 2위: `maywell/ko_Ultrafeedback_binarized`
- **선정 이유**: UltraFeedback 일반 도메인 → 수학 편향 보완, 일반 instruction following 향상
- **조합**: orca-math DPO + ko_Ultrafeedback 혼합 사용 권장
- **규모**: ~60k 추정 (원본 UltraFeedback binarized 기준)

### 🥉 3위: `lemon-mint/korean-realqa-reasoning-v01-preference`
- **선정 이유**: 2025년 최신, CoT reasoning traces 포함 → thinking 능력 학습 가능
- **활용**: 소규모(7.77k)이지만 quality가 높고 CoT 형태 데이터는 희귀
- **ORPO 특이사항**: chosen에 `<think>` 태그 포함 → reasoning 모델 특화 훈련에 적합

**권장 혼합 레시피**:
```
orca-math-dpo (~193k) : ko_ultrafeedback (~60k) : realqa-reasoning (~7k) = 약 260k쌍
비율: 수학 74% : 일반 23% : 추론 3%
→ 더 나은 균형 원한다면 orca-math 다운샘플링 고려 (예: 60k 샘플링)
```

---

## Top 3 권장 - Pretrain

### 🥇 1위: `KORMo-Team/korean-public-corpus`
- **선정 이유**: 학술/공공 도메인 → 기존 웹 크롤 기반 코퍼스와 비중복 가능성 최고
- **기대 추가 토큰**: 중복 제거 후 수십억 토큰 순수 증가 예상
- **라이선스**: 공공저작물 (상업 사용 가능)

### 🥈 2위: `KORMo-Team/korean-web-collection`
- **선정 이유**: 대규모 한국어 웹 다양성, 단순 웹 크롤 이상의 도메인 커버리지
- **주의**: MinHash dedup 필수 (CulturaX/OSCAR와 중복 가능)
- **기대 추가 토큰**: 중복 제거 후 10B~30B 예상

### 🥉 3위: `devngho/korean-webtext-edu`
- **선정 이유**: 교육 품질 분류기 필터링 → 고품질 서브셋 (FineWeb-Edu 스타일)
- **주의**: KOREAN-WEBTEXT(oscar2201) 기반 → 기존 OSCAR와 중복 가능, 중복 제거 후 순수 고품질 새 토큰만 추출
- **활용**: 전체를 쓰기보다 `scored_over_3` 고품질 서브셋만 선별 사용

**Pretrain 추가 확보 전략**:
```
현재: ~39B 토큰
목표: Chinchilla optimal ~210B (3B 모델)
부족분: ~171B 토큰

우선순위 소스 (순수 증가분 추정):
1. KORMo-Team/korean-public-corpus   → 5B~20B (학술, 비중복)
2. KORMo-Team/korean-web-collection  → 10B~30B (dedup 후)
3. devngho/korean-webtext-edu        → 5B~10B (고품질 서브셋)
4. AI Hub 한국어 코퍼스 (신청 필요)  → 50B~100B 추정
5. HPLT v2.0 한국어 (재조사 필요)   → 50B~100B 추정

※ 현실적으로 HF 공개 소스만으로는 171B 순수 증가분 달성 어려움
   → AI Hub + 국립국어원 공개 말뭉치 신청 병행 권장
```

---

## 갭 분석 및 메모

### Preference 데이터 갭
1. **일반 도메인 한국어 DPO 데이터 부족**: 수학/추론 외 한국어 일반 대화 preference 쌍은 매우 희소
2. **Human-annotated 데이터 없음**: 모든 발견된 데이터는 LLM 생성 (GPT-4/GPT-3.5 기반)
3. **최신 안전성 데이터 없음**: 한국어 safety/harmlessness 특화 DPO 데이터 미발견
4. **의료/법률 특화 없음**: 한국어 전문 도메인 preference 데이터 공백

### Pretrain 데이터 갭
1. **HPLT v2.0 접근 불가**: 공식 URL 404 → 공식 릴리즈 채널 재확인 필요
2. **AI Hub 미포함**: 가장 큰 공공 한국어 코퍼스지만 별도 신청 프로세스 필요
3. **국립국어원 말뭉치 미포함**: 별도 다운로드 포털 사용 필요
4. **코드 데이터 미포함**: 한국어 주석 코드 데이터 별도 조사 필요

### 라이선스 주의사항
- `devngho/korean-webtext-edu`: MIT 선언이지만 원본 HAERAE-HUB/KOREAN-WEBTEXT 라이선스 불명확 → 상업적 사용 전 확인 필요
- `ohsuz/dpo-v1010-korean`: Gated → 접근 신청 필요
- `uonlp/CulturaX`: CC BY-NC 4.0 → 비상업적 용도만 가능

---

*조사 완료: 2026-02-26 | 조사자: OpenClaw subagent (survey-preference-pretrain)*
