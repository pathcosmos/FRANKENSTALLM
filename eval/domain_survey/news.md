# 한국어 뉴스/언론 도메인 데이터셋 전수 조사

> 조사일: 2026-02-27  
> 목적: 한국어 3B LLM 학습용 뉴스/언론 도메인 데이터 파악  
> 조사범위: HuggingFace Hub, AI-Hub, 모두의 말뭉치, BigKinds, GitHub, 학술 논문 기반

---

## 전체 데이터셋 목록

| # | 데이터셋 / 출처 | 크기 | 라이선스 | 내용 유형 | 날짜범위 | 출처 언론사 | 상업적 이용 | 우선순위 |
|---|---------------|------|---------|---------|---------|-----------|-----------|--------|
| 1 | **[모두의 말뭉치: 신문]** corpus.korean.go.kr | ~350만 문장 | 연구전용 (비공개배포 금지) | 뉴스 기사 전문 + 메타 | 2018~2021 | 한국경제, 동아, 조선 등 다수 | ❌ | **9** |
| 2 | **[BigKinds]** bigkinds.or.kr | 5,000만건+ | 신청 후 제공 (연구·교육) | 뉴스 기사 전문 | 1990~현재 | 54개 언론사 (연합뉴스, 조선, 중앙 등) | ❌ | **9** |
| 3 | **[AI-Hub] 문서요약 텍스트** (#97) | 원문 40만건 (신문 30만건) / 요약 80만건 | 연구전용 | 뉴스 기사 전문 + 추출/생성 요약 | 2020 | 다수 종합일간지 | ❌ | **8** |
| 4 | **[HF] sieu-n/korean-newstext-dump** | 1M~10M건 (텍스트 파일) | 불명확 | 뉴스 기사 전문 (제목+본문) | ~2021 | 복수 언론사 | ❓ | **8** |
| 5 | **[AI-Hub] 뉴스 기사 기계독해** (#577) | 400,056건 QA / 지문 36만건 | 연구전용 | 뉴스 기사 + QA | 2021 | 중앙일보 등 20개 언론사 | ❌ | **7** |
| 6 | **[HF] daekeun-ml/naver-news-summarization-ko** | 24,934건 (train+test) | Apache 2.0 | 뉴스 기사 전문 + 요약 | 2022.07 | 네이버 뉴스 (YTN, 아시아경제 등) | ✅ | **7** |
| 7 | **[HF] sigridjineth/korean-news-small** | 1M~10M건 | 불명확 | 뉴스 텍스트 | 불명 | 불명 | ❓ | **6** |
| 8 | **[HF] klue/klue (ynat subset)** | 54,800건 | CC-BY-SA-4.0 | 뉴스 제목 + 7개 토픽 레이블 | 2020~2021 | 연합뉴스 | ✅ | **6** |
| 9 | **[GitHub] KcBERT 댓글 데이터** beomi/KcBERT | 45GB / 3.4억건 | CC-BY (댓글) | 네이버 뉴스 댓글 + 대댓글 | ~2022 | 네이버 뉴스 댓글 | ❓ | **5** |
| 10 | **[HF] haseong8012/Korean_Political-News_By_Media-Outlet** | 100K~1M건 | 불명확 | 언론사별 정치 뉴스 | 2024 | 조선, 한겨레 등 다수 언론사 | ❓ | **5** |
| 11 | **[AI-Hub] 한국어-영어 번역 말뭉치 (뉴스)** (#87) | 약 160만 문장쌍 | 연구전용 | 뉴스 기사 한-영 병렬 | 2019 | 다수 | ❌ | **5** |
| 12 | **[HF] KETI-AIR/kor_ag_news** | 120K건 | Unknown | AG News 영→한 번역본 (4분류) | 번역 | 영어 원본 | ❓ | **4** |
| 13 | **[HF] BLACKBUN/old_korean_newspaper_1897_1910_economy_politic** | 100K~1M건 | 불명확 | 구한말 신문 기사 (경제/정치) | 1897~1910 | 독립신문 등 구한말 신문 | ❓ | **3** |
| 14 | **[HF] BLACKBUN/old_korean_newspaper_1897_1910_economy_politic_qa** | 1K~10K건 | 불명확 | 구한말 신문 QA | 1897~1910 | 구한말 신문 | ❓ | **3** |
| 15 | **[HF] hugmanskj/korean-news-topic-classification** | ~5K건 | CC-BY-4.0 | 합성 뉴스 헤드라인 (4분류) | 2025 | 합성 데이터 | ✅ | **2** |
| 16 | **[HF] 91veMe4Plus-Project/korean_news_corpus** | 비어 있음 | MIT | 비어 있음 | - | - | ✅ | **1** |

---

## 출처별 상세 분류

### 🔵 HuggingFace Hub

| HF Repo ID | 다운로드수 | 다운로드 명령어 |
|------------|----------|---------------|
| `sieu-n/korean-newstext-dump` | 8 | `load_dataset("sieu-n/korean-newstext-dump")` |
| `sigridjineth/korean-news-small` | 20 | `load_dataset("sigridjineth/korean-news-small")` |
| `daekeun-ml/naver-news-summarization-ko` | 1,133 | `load_dataset("daekeun-ml/naver-news-summarization-ko")` |
| `klue/klue` (ynat subset) | 4,248 | `load_dataset("klue/klue", "ynat")` |
| `haseong8012/Korean_Political-News_By_Media-Outlet` | 34 | `load_dataset("haseong8012/Korean_Political-News_By_Media-Outlet")` |
| `BLACKBUN/old_korean_newspaper_1897_1910_economy_politic` | 5 | `load_dataset("BLACKBUN/old_korean_newspaper_1897_1910_economy_politic")` |
| `BLACKBUN/old_korean_newspaper_1897_1910_economy_politic_qa` | 5 | `load_dataset("BLACKBUN/old_korean_newspaper_1897_1910_economy_politic_qa")` |
| `KETI-AIR/kor_ag_news` | 5 | `load_dataset("KETI-AIR/kor_ag_news")` |
| `hugmanskj/korean-news-topic-classification` | 33 | `load_dataset("hugmanskj/korean-news-topic-classification")` |
| `91veMe4Plus-Project/korean_news_corpus` | 2 | (비어 있음) |

### 🟢 AI-Hub (aihub.or.kr)

> 모두 **국내 기관/개인 가입 + 신청 승인 후** 다운로드 가능. 상업적 이용 불가.

| 데이터셋명 | dataSetSn | 규모 | 주요 언론사 |
|-----------|----------|------|-----------|
| 문서요약 텍스트 | 97 | 원문 40만건 (신문 30만건) | 다수 종합일간지 |
| 뉴스 기사 기계독해 데이터 | 577 | QA 400,056건 / 지문 36만건 | 중앙일보 등 20개 언론사 |
| 한국어-영어 번역 말뭉치 (뉴스 포함) | 87 | ~160만 문장쌍 | 다수 |

신청 URL 패턴: `https://aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&dataSetSn={ID}`

### 🟡 국립국어원 모두의 말뭉치

> 신청 후 수작업 다운로드. 라이선스 엄격함 (재배포 불가, 연구 전용).

- **modu_news (신문)**: 약 350만 문장, 9개 카테고리(정치/경제/사회/생활/IT과학/연예/스포츠/문화/미용건강)
  - 메타: publisher, author, date, topic, original_topic, paragraph
  - 신청: https://corpus.korean.go.kr → 가입 → 신청
  - Korpora 로드: `from Korpora import Korpora; corpus = Korpora.load("modu_news")`

### 🟠 BigKinds (한국언론진흥재단)

> 별도 계약/신청 필요. 5천만건 이상 뉴스 기사 (1990~현재). 54개 주요 언론사 포함.
- 주요 언론사: 연합뉴스, 조선일보, 중앙일보, 동아일보, 한겨레, 경향신문, 매일경제, 한국경제 등
- 학술/연구 목적 데이터 제공: bigkinds.or.kr
- **연구용 샘플 데이터**: 일부 카테고리 무료 제공, 전체는 협약 필요

### ⚪ GitHub 오픈소스

| 프로젝트 | 규모 | 내용 | 라이선스 | URL |
|---------|------|------|---------|-----|
| KcBERT (Beomi) | 45GB / 3.4억건 | 네이버 뉴스 댓글+대댓글 | CC-BY | https://github.com/Beomi/KcBERT |
| Korpora (modu_news 로더) | - | 모두의 말뭉치 로더 | Apache 2.0 | https://github.com/ko-nlp/Korpora |

---

## 🏆 Top 3 상세 설명

---

### 1위 🥇 모두의 말뭉치 신문 (국립국어원)

| 항목 | 내용 |
|------|------|
| **출처** | 국립국어원 (corpus.korean.go.kr) |
| **크기** | ~350만 문장 / train split |
| **라이선스** | 연구전용, 재배포 불가 |
| **내용** | 뉴스 기사 전문. 메타정보(발행일, 언론사, 카테고리 등) 포함 |
| **날짜 범위** | 2018~2021 추정 |
| **출처 언론사** | 한국경제, 동아일보 등 다수 종합일간지 |
| **카테고리** | 정치, 경제, 사회, 생활, IT/과학, 연예, 스포츠, 문화, 미용/건강 (9개) |
| **다운로드** | https://corpus.korean.go.kr 가입→신청→수작업 다운로드 |
| **Korpora 로드** | `corpus = Korpora.load("modu_news")` |
| **상업적 이용** | ❌ (연구전용) |
| **특징** | 대규모 + 고품질 + 메타정보 풍부 + 다양한 언론사 |
| **주의사항** | 가입 필요, 한국 거주/기관 소속 우선, 재배포 불가 |

**평가**: LLM 사전학습용으로 가장 이상적. 350만 문장의 정제된 뉴스 기사. 다만 접근 절차가 복잡하고 라이선스 제약이 있음.

---

### 2위 🥈 BigKinds 뉴스 빅데이터 (한국언론진흥재단)

| 항목 | 내용 |
|------|------|
| **출처** | 한국언론진흥재단 (bigkinds.or.kr) |
| **크기** | 5,000만건 이상 (1990~현재) |
| **라이선스** | 기관 협약 후 연구목적 제공 |
| **내용** | 뉴스 기사 전문, 키워드, 요약, 카테고리 등 |
| **날짜 범위** | 1990~현재 (30년 이상) |
| **출처 언론사** | 54개: 연합뉴스, 조선일보, 중앙일보, 동아일보, 한겨레, 경향신문, 매일경제, 한국경제, YTN, KBS, MBC 등 |
| **다운로드** | bigkinds.or.kr 연구용 데이터 신청 페이지 |
| **상업적 이용** | ❌ |
| **특징** | 국내 최대 규모 뉴스 DB. 언론사 다양성 최고. 30년치 역사 데이터 |
| **주의사항** | 전체 DB 접근은 협약 필요. 부분 샘플만 무료 |

**평가**: 규모와 품질 모두 최상. 연구 기관 협약 가능하다면 최우선 확보 대상.

---

### 3위 🥉 AI-Hub 문서요약 텍스트 (dataSetSn=97)

| 항목 | 내용 |
|------|------|
| **출처** | AI-Hub (aihub.or.kr) |
| **크기** | 원문 40만건 (신문기사 30만 + 기고문 6만 + 잡지 1만 + 판결문 3만) / 요약문 80만건 |
| **라이선스** | 연구전용 (무료, 국내 기관/개인 신청) |
| **내용** | 뉴스 기사 원문 + 추출요약 + 생성요약 |
| **날짜 범위** | 2020년 구축 |
| **출처 언론사** | 종합일간지 다수 |
| **다운로드** | `https://aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&dataSetSn=97` |
| **상업적 이용** | ❌ |
| **특징** | 추출요약+생성요약 쌍 포함. 요약 태스크뿐 아니라 사전학습용 기사 원문으로도 활용 가능 |
| **다운로드수** | 5,912건 (AI-Hub 내 최고 수준) |

**평가**: 요약 태스크 SFT 데이터 + 사전학습 기사 원문 동시 활용 가능. 가입 후 즉시 신청 가능.

---

## 📊 주요 지표 비교

| 데이터셋 | 규모 | 품질 | 접근성 | 라이선스 | LLM 사전학습 적합도 |
|---------|------|------|-------|---------|-----------------|
| 모두의 말뭉치 신문 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| BigKinds | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐ | ⭐⭐⭐⭐⭐ |
| AI-Hub 문서요약 | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| sieu-n/korean-newstext-dump | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| daekeun-ml/naver-news | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| KcBERT 댓글 | ⭐⭐⭐⭐⭐ | ⭐⭐ (댓글) | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ (댓글 특화) |

---

## 🎯 추천 데이터 확보 전략

### 즉시 사용 가능 (공개 라이선스)
1. `daekeun-ml/naver-news-summarization-ko` — Apache 2.0, HF에서 바로 다운로드
2. `klue/klue` (ynat) — CC-BY-SA-4.0, HF에서 바로 다운로드
3. `sieu-n/korean-newstext-dump` — 라이선스 확인 필요하나 HF 공개
4. `sigridjineth/korean-news-small` — HF 공개

### 신청 절차 필요 (고품질)
1. **모두의 말뭉치 신문** → corpus.korean.go.kr 가입 후 신청 (1~2주 소요)
2. **AI-Hub 문서요약** → aihub.or.kr 가입 후 신청 (즉시~수일 소요)
3. **AI-Hub 뉴스 기계독해** → aihub.or.kr 가입 후 신청

### 협약 필요 (대규모)
1. **BigKinds** → 한국언론진흥재단 협약 (기관 필요)

---

## 📝 기타 참고 사항

### 뉴스 포함 대규모 한국어 코퍼스 (뉴스 외 다수 도메인 혼합)
- **mC4 Korean** (`allenai/c4`, language=ko): 웹크롤 데이터, 뉴스 도메인 상당 부분 포함
- **OSCAR 한국어**: CC0, 웹크롤, 뉴스 혼합
- **CC-100 Korean**: 커먼크롤 기반, 뉴스 포함

### 알려진 미확인 데이터셋
- **연합뉴스 코퍼스**: 공식 제공 여부 불명 (KLUE 데이터의 소스)
- **한국 언론 아카이브**: 개별 언론사 API (유료)
- **공공데이터포털 (data.go.kr)**: 검색 결과 뉴스 특화 텍스트 데이터셋 발견 안 됨

---

*조사: 2026-02-27 | 조사자: LLM-Bang 데이터 서브에이전트*
