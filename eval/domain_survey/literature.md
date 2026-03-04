# 한국어 소설/문학/창작/SNS 도메인 데이터 전수 조사

> 조사일: 2026-02-27  
> 조사자: survey-literature 서브에이전트  
> 목적: 한국어 LLM 3B 모델 학습용 소설·문학·창작·SNS 데이터 발굴

---

## 1. 전체 데이터셋 목록

### 1-A. HuggingFace Hub

| # | Repo ID | 크기 | 라이선스 | 내용 | 다운로드 방법 | 저작권 | 우선순위 |
|---|---------|------|---------|------|--------------|--------|--------|
| 1 | `werty1248/Korean-1930-Novel-Scene-Summarize` | 12,108씬 (~10K-100K) | MIT | 1930년대 한국 퍼블릭도메인 소설 96편 씬분리+요약, Gemini-1.5-Flash 생성 | `load_dataset("werty1248/Korean-1930-Novel-Scene-Summarize")` | ✅ 퍼블릭도메인 기반 | **9** |
| 2 | `minpeter/geulgyeol-blog-korean` | 1.75M 샘플 | 미명시 | 한국어 블로그 텍스트 (네이버 블로그 등 실생활 글) | `load_dataset("minpeter/geulgyeol-blog-korean")` | ⚠️ 불명확 | **8** |
| 3 | `HAERAE-HUB/KOREAN-WEBTEXT` | 2.2B 토큰, 1M-10M 문서 | 미명시 | CC100+OSCAR+인터넷 수집 고품질 웹텍스트 (블로그/SNS 포함) | `load_dataset("HAERAE-HUB/KOREAN-WEBTEXT")` | ⚠️ 웹크롤 | **7** |
| 4 | `KORMo-Team/korean-web-collection` | 대용량 | 미명시 | 최신 한국어 웹 컬렉션 (2025년) | `load_dataset("KORMo-Team/korean-web-collection")` | ⚠️ 불명확 | **5** |
| 5 | `heegyu/namuwiki-extracted` | 571,308행, 2.19GB | CC BY-NC-SA 2.0 | 나무위키 2022-03 덤프 전처리버전, 소설/문화/창작 관련 항목 포함 | `load_dataset("heegyu/namuwiki-extracted")` | ⚠️ NC 제한 | **6** |
| 6 | `heegyu/namuwiki` | 867,024행, 3GB | CC BY-NC-SA 2.0 | 나무위키 원본 덤프 (마크업 포함) | `load_dataset("heegyu/namuwiki")` | ⚠️ NC 제한 | **4** |
| 7 | `heegyu/namuwiki-sentences` | 38,015,081 문장 | CC BY-NC-SA 2.0 | 나무위키 문장 단위 분리버전 | `load_dataset("heegyu/namuwiki-sentences")` | ⚠️ NC 제한 | **4** |
| 8 | `LLM-SocialMedia/Korean-YouTube-Comment-Sentiment-Dataset` | 5,482 댓글 | Other | 유튜브 한국어 댓글 (구어체·이모지·줄임말) | `load_dataset("LLM-SocialMedia/Korean-YouTube-Comment-Sentiment-Dataset")` | ⚠️ 불명확 | **3** |
| 9 | `minpeter/fineweb-2-edu-korean-raw` | 10M-100M 문서 | Apache? | FineWeb-2 한국어 서브셋 (웹텍스트 전체) | `load_dataset("minpeter/fineweb-2-edu-korean-raw")` | ⚠️ 웹크롤 | **6** |
| 10 | `eliceai/korean-webtext-edu` | 1M-10M | MIT | KOREAN-WEBTEXT 교육가치 필터링본 | `load_dataset("eliceai/korean-webtext-edu")` | ⚠️ 웹크롤 | **5** |
| 11 | `naem1023/augmented-namuwiki` | 1M-10M | Apache 2.0 | 나무위키 증강버전 | `load_dataset("naem1023/augmented-namuwiki")` | ⚠️ NC원본 기반 | **3** |

### 1-B. AI-Hub (aihub.or.kr) — 회원가입+신청 필요

| # | 데이터셋명 | 크기 | 내용 | URL | 저작권 | 우선순위 |
|---|-----------|------|------|-----|--------|--------|
| 1 | **대규모 구매도서 기반 한국어 말뭉치 데이터** (No.653) | 10억 어절, 18GB+ | 소설·에세이·경제·철학 등 다양한 도서 텍스트, 분야별 분포 (문학 포함) | [링크](https://aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&dataSetSn=653) | 🟡 AI-Hub 이용약관 | **10** |
| 2 | **다양한 문화콘텐츠 스토리 데이터** (No.71562) | 3,953편, 100,077 유닛, ~670MB | 영화·드라마·소설·만화 스토리 분석 데이터, 장르/인물/서사단계 라벨링 | [링크](https://aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&dataSetSn=71562) | 🟡 AI-Hub 이용약관 | **8** |
| 3 | **동화 줄거리 생성 데이터** (No.71696) | 조회11,745, 다운555 | 동화 텍스트+줄거리 생성 | [링크](https://aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&dataSetSn=71696) | 🟡 AI-Hub 이용약관 | **6** |
| 4 | **동화 이해도 테스트를 위한 질의응답쌍 생성 데이터** (No.71649) | 1M-10M | 동화 QA쌍 | [링크](https://aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&dataSetSn=71649) | 🟡 AI-Hub 이용약관 | **5** |
| 5 | **문학작품 낭송·낭독 음성 데이터** (No.485) | 100GB+ (오디오+텍스트) | 시·소설·희곡·시나리오 낭독 (텍스트 스크립트 포함) | [링크](https://aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&dataSetSn=485) | 🟡 AI-Hub 이용약관 | **7** |

### 1-C. 공유마당 (gongu.copyright.or.kr) — 퍼블릭도메인 소설

| # | 소스 | 크기 | 내용 | URL | 저작권 | 우선순위 |
|---|------|------|------|-----|--------|--------|
| 1 | **공유마당 어문 저작물** | 1,107,853건 | 저작권 만료 소설·수필·시 (김유정, 이효석, 현진건 등 1945년 이전 작가) | [링크](https://gongu.copyright.or.kr/gongu/wrt/wrtCl/listWrtText.do?menuNo=200019) | ✅ 퍼블릭도메인 | **9** |

### 1-D. 국립국어원 모두의 말뭉치 (kli.korean.go.kr)

| # | 데이터셋명 | 크기 | 내용 | URL | 저작권 | 우선순위 |
|---|-----------|------|------|-----|--------|--------|
| 1 | **모두의 말뭉치 (NIKL)** — 현대소설 말뭉치 | 미공개 (수백MB-수GB 추정) | 현대소설, 신문기사, 구어 등 다양한 장르 | [링크](https://kli.korean.go.kr/main/requestMain.do) | 🟡 국립국어원 이용약관 | **9** |

### 1-E. 프로젝트 구텐베르크 (gutenberg.org)

| # | 소스 | 내용 | URL | 우선순위 |
|---|------|------|-----|--------|
| 1 | Gutenberg 한국어 | **사실상 없음** — 영-한 사전 1권만 존재, 한국어 문학 작품 미수록 | [링크](https://www.gutenberg.org/browse/languages/ko) | **1** (스킵) |

---

## 2. Top 3 상세 분석

---

### 🥇 1위: AI-Hub — 대규모 구매도서 기반 한국어 말뭉치 (No.653)

| 항목 | 내용 |
|------|------|
| **Repo/URL** | https://aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&dataSetSn=653 |
| **크기** | 10억 어절 (약 5~18GB 추정), 다운로드 2,515건 |
| **라이선스** | AI-Hub 이용약관 (상업적 활용 가능하나 재배포 불가, 연구·학습 목적 OK) |
| **내용** | 실제 구매된 도서 텍스트 말뭉치. 분야별 비율: 사회과학(28.4%), 철학(8.9%), 종교(4.8%), 역사(9.3%), 예술·체육(3.3%), **문학(9.5% 추정)** 등 다양. 소설·에세이·수필 포함. |
| **구축년도** | 2021년 |
| **다운로드** | 회원가입 → 데이터 신청 → 승인 후 API 다운로드 |
| **저작권** | 🟡 AI-Hub 이용약관. 저작권 구매 도서 기반이므로 법적 안전성 높음. 단, 재배포 금지 |
| **강점** | 실제 출판 도서 텍스트 → 고품질 문어체, 다양한 장르. 10억 어절 규모 최대 |
| **약점** | 신청 승인 필요, 문학 비중이 전체의 일부 |
| **우선순위** | **10/10** |

**다운로드 방법:**
```bash
# 1. aihub.or.kr 회원가입
# 2. 해당 데이터셋 페이지에서 신청
# 3. 승인 완료 후 API 다운로드
find "폴더경로" -name "파일명.zip.part*" -print0 | sort -zt'.' -k2V | xargs -0 cat > "파일명.zip"
```

---

### 🥈 2위: 공유마당 — 퍼블릭도메인 한국 고전소설

| 항목 | 내용 |
|------|------|
| **URL** | https://gongu.copyright.or.kr/gongu/wrt/wrtCl/listWrtText.do?menuNo=200019 |
| **크기** | 1,107,853건 (어문 저작물 전체, 소설은 수천~수만건 추정) |
| **라이선스** | ✅ **완전 퍼블릭도메인** — 상업적 활용·재배포 모두 자유 |
| **내용** | 저작권 만료 소설: 김유정(봄봄, 동백꽃), 현진건(운수 좋은 날), 이효석(메밀꽃 필 무렵), 이상(날개) 등 1945년 이전 작가 작품 전부. 2021년 이전 공모전 수상작도 일부 포함 |
| **다운로드** | 사이트에서 개별 파일 다운로드 또는 스크래핑 가능 |
| **저작권** | ✅ 완전 클리어. LLM 학습용으로 가장 안전한 소스 |
| **강점** | 법적 리스크 제로, 근대소설 문체 학습에 최적 |
| **약점** | 현대(1945년 이후) 소설 없음, 텍스트 양이 상대적으로 적음, 고어체 포함 |
| **우선순위** | **9/10** |

**다운로드 방법:**
```python
# Python 크롤링 예시
import requests
from bs4 import BeautifulSoup

base_url = "https://gongu.copyright.or.kr/gongu/wrt/wrtCl/listWrtText.do?menuNo=200019"
# 페이지별 순회 후 개별 작품 텍스트 다운로드
# 또는 werty1248/Korean-1930-Novel-Scene-Summarize에 이미 전처리된 버전 있음
```

---

### 🥉 3위: HuggingFace — minpeter/geulgyeol-blog-korean

| 항목 | 내용 |
|------|------|
| **Repo ID** | `minpeter/geulgyeol-blog-korean` |
| **URL** | https://huggingface.co/datasets/minpeter/geulgyeol-blog-korean |
| **크기** | 1.75M 샘플 (약 수GB 추정) |
| **라이선스** | 미명시 (주의 필요) |
| **내용** | 한국어 블로그 텍스트. 여행기, 일상기록, 레시피, 부동산, 음악 가사 번역 등 다양한 실생활 글쓰기. 구어체+문어체 혼합, SNS스러운 이모지/줄임말 포함 |
| **구축년도** | 2025년 8월 |
| **다운로드** | `load_dataset("minpeter/geulgyeol-blog-korean")` |
| **저작권** | ⚠️ 네이버 블로그 수집 추정 → 라이선스 불명확. 학습용은 괜찮으나 재배포 주의 |
| **강점** | 실제 한국인의 일상 글쓰기 스타일, 다양한 주제의 블로그 텍스트, 175만 샘플로 규모 큼 |
| **약점** | 라이선스 미명시, 정보성 글 위주 (순수 창작 소설 아님) |
| **우선순위** | **8/10** |

**다운로드 방법:**
```python
from datasets import load_dataset
ds = load_dataset("minpeter/geulgyeol-blog-korean")
print(ds)
```

---

## 3. 추가 유망 데이터셋 (보조)

### HAERAE-HUB/KOREAN-WEBTEXT
- **내용**: CC100+OSCAR+자체 수집 웹텍스트, 2.2B 토큰
- **특징**: 블로그·커뮤니티·뉴스 등 다양한 웹소스 혼합, 고품질 필터링 적용
- **용도**: 도메인 사전학습 데이터로 블로그/SNS 텍스트 포함
- `load_dataset("HAERAE-HUB/KOREAN-WEBTEXT")`

### heegyu/namuwiki-extracted  
- **내용**: 한국 최대 위키 나무위키 (571K 문서, 2.19GB)
- **특징**: 소설/영화/드라마/게임 등 문화콘텐츠 관련 항목 대량 포함, 한국어 백과 스타일
- **라이선스**: CC BY-NC-SA 2.0 → **비상업적 사용만 가능**
- `load_dataset("heegyu/namuwiki-extracted")`

### werty1248/Korean-1930-Novel-Scene-Summarize
- **내용**: 공유마당 소설 96편에서 Gemini로 씬 분리+요약 생성
- **특징**: 원작은 퍼블릭도메인, 요약은 AI생성. MIT 라이선스
- `load_dataset("werty1248/Korean-1930-Novel-Scene-Summarize")`

### AI-Hub — 다양한 문화콘텐츠 스토리 데이터 (No.71562)
- **내용**: 3,953편 (영화 40%, 드라마 41%, 소설 5.5%, 만화 12%), 100,077 스토리 유닛
- **특징**: 줄거리+감정+서사단계 라벨링. 창작 AI 학습에 특화
- **장르**: 드라마(38%), 멜로(24%), 스릴러(12%), 판타지(8%) 등

### AI-Hub — 문학작품 낭송·낭독 음성 데이터 (No.485)
- **내용**: 시·소설·희곡·시나리오 텍스트+음성 데이터
- **특징**: 텍스트 스크립트 포함 → 순수 문학 텍스트로 활용 가능

### 국립국어원 모두의 말뭉치 (NIKL)
- **내용**: 현대소설, 신문, 구어, SNS 등 다양한 장르 말뭉치
- **특징**: 국가 공인 품질, 정교한 형태소 분석 포함
- **다운로드**: kli.korean.go.kr 신청 후 무료 다운로드

---

## 4. 저작권 주의사항

### ⚠️ 핵심 원칙

| 구분 | 기준 | 비고 |
|------|------|------|
| **퍼블릭도메인 한국 소설** | 작가 사망 후 70년 이상 경과 | 1945년 이전 작고 작가 작품 대부분 해당 |
| **현대 소설** | 대부분 저작권 보호 중 | 1950~현재 작가 작품은 허가 없이 사용 불가 |
| **나무위키** | CC BY-NC-SA 2.0 | **상업적 사용 불가, 동일 라이선스 공유 의무** |
| **AI-Hub 데이터** | AI-Hub 이용약관 | 연구·학습 목적 OK, 재배포 금지 |
| **웹크롤 데이터** | 사이트별 ToS 적용 | 학습용 사용은 일반적으로 허용 추세 |

### 퍼블릭도메인 한국 소설 주요 작가 (1945년 이전 작고)
- **김유정** (1908~1937): 동백꽃, 봄봄, 만무방 등
- **이효석** (1907~1942): 메밀꽃 필 무렵, 분녀 등
- **현진건** (1900~1943): 운수 좋은 날, 빈처 등
- **이상** (1910~1937): 날개, 봉별기 등
- **염상섭** (1897~1963): ⚠️ 1963년 작고 → **2034년까지 보호** (주의!)
- **채만식** (1902~1950): ⚠️ 1950년 작고 → **2021년 만료** (현재 퍼블릭도메인)

### 현대 소설 저작권 주의
- 박경리 (1926~2008) — 2079년까지 보호
- 이청준 (1939~2008) — 2079년까지 보호
- 조정래, 황석영 등 생존 작가 — 모두 보호 중
- **웹소설 (카카오/네이버 시리즈)** — 플랫폼과 작가 모두 저작권 보유

### SNS/블로그 데이터
- 네이버 블로그 크롤링 → 네이버 ToS 위반 가능성 있음
- 학습 목적 사용은 법적 그레이존 (EU AI Act, 한국 저작권법 35조의5)
- `minpeter/geulgyeol-blog-korean` 등은 라이선스 명시 없으므로 상업 배포 전 검토 필요

---

## 5. 권고 우선순위 요약

```
1. AI-Hub 대규모 구매도서 말뭉치 (10억 어절, 법적 안전)       ⭐⭐⭐⭐⭐ 10/10
2. 공유마당 퍼블릭도메인 소설 (법적 제로리스크)               ⭐⭐⭐⭐⭐  9/10
3. NIKL 모두의 말뭉치 현대소설 (국가 공인, 무료)              ⭐⭐⭐⭐⭐  9/10
4. werty1248/Korean-1930-Novel-Scene-Summarize (MIT, 즉시)    ⭐⭐⭐⭐    9/10
5. minpeter/geulgyeol-blog-korean (블로그 SNS, 175만)        ⭐⭐⭐⭐    8/10
6. AI-Hub 문화콘텐츠 스토리 (창작 특화, 승인 필요)            ⭐⭐⭐⭐    8/10
7. AI-Hub 문학작품 낭독 데이터 (텍스트 포함)                 ⭐⭐⭐⭐    7/10
8. HAERAE-HUB/KOREAN-WEBTEXT (블로그/SNS 포함 웹텍스트)      ⭐⭐⭐     7/10
9. heegyu/namuwiki-extracted (NC 라이선스 주의)              ⭐⭐⭐     6/10
10. minpeter/fineweb-2-edu-korean-raw (대용량 웹크롤)        ⭐⭐⭐     6/10
```

---

## 6. 즉시 실행 가능한 데이터 (추가 승인 불필요)

```python
from datasets import load_dataset

# 1. 퍼블릭도메인 소설 씬 데이터 (MIT)
ds1 = load_dataset("werty1248/Korean-1930-Novel-Scene-Summarize")

# 2. 한국어 블로그 (175만 샘플)
ds2 = load_dataset("minpeter/geulgyeol-blog-korean")

# 3. 나무위키 (비상업 주의)
ds3 = load_dataset("heegyu/namuwiki-extracted")

# 4. 한국어 웹텍스트 (블로그+SNS 포함)
ds4 = load_dataset("HAERAE-HUB/KOREAN-WEBTEXT")
```

---

*조사 소스: HuggingFace Hub API, AI-Hub, 공유마당, 국립국어원, Project Gutenberg*
