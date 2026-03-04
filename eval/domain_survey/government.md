# 한국어 정부/공공/행정/특허 도메인 데이터 전수 조사

> 작성일: 2026-02-27  
> 목적: 한국어 LLM 3B 모델 사전학습/파인튜닝용 공공·정부·법률·특허 도메인 데이터셋 조사

---

## 1. 전체 목록 테이블

### 1-1. AI-Hub (aihub.or.kr) 데이터셋

| # | 데이터셋 명 | DataSetSn | 크기/규모 | 라이선스 | 내용 유형 | 다운로드 방법 | 한국어% | 우선순위 |
|---|------------|-----------|----------|----------|-----------|--------------|---------|---------|
| 1 | 국가기록물 대상 초거대 AI 학습 말뭉치 데이터 | 71788 | **원천 4억 토큰** / QA 50,000건 / 유해질의 10,560건 | 공공누리 (NIA) | 정부간행물, 백서, 연감, 사업보고서, 연구보고서 등 | API 다운로드 (승인 후) | ~100% | **10** |
| 2 | 국회 회의록 기반 지식 검색 데이터 | 71795 | 회의록 11,827건 / QA쌍 44,033건 | 공공누리 (NIA) | 국회 본회의·상임위·소위·국감 회의록 (15~21대) | API 다운로드 (승인 후) | ~100% | **9** |
| 3 | 국가중점기술 대응 특허 데이터 | 71739 | **619,844건** (특허명세서+분류 라벨) | 공공누리 (NIA) | 특허 명칭/요약/청구항 + 기술분류 레이블 | API 다운로드 (승인 후) | ~95% | **9** |
| 4 | 법률/규정 텍스트 분석 (판례 고도화) | 71723 | 원문 25만건 / 라벨링 66,511건 + QA 20,160건 | 공공누리 (NIA) | 대법원·하급심·심결례 판결문, QA, 요약, 키워드 | API 다운로드 (승인 후) | ~100% | **9** |
| 5 | 공공 민원 상담 LLM 사전학습·IT 데이터 | 71852 | 원천 10,182건 / 가공 124,717건 | 공공누리 (NIA) | 중앙/지방행정기관 민원 상담 (분류·요약·QA) | API 다운로드 (승인 후) | ~100% | **8** |
| 6 | 민간 민원 상담 LLM 사전학습·IT 데이터 | 71844 | 원천 ~10K건 / 가공 ~120K건 | 공공누리 (NIA) | 민간 민원 상담 (분류·요약·QA) | API 다운로드 (승인 후) | ~100% | **7** |
| 7 | 법률안 검토 보고서 요약 데이터 | 71794 | 다운로드 675건 (조회 22K) | 공공누리 (NIA) | 국회 법률안 검토보고서 요약 | API 다운로드 (승인 후) | ~100% | **7** |
| 8 | 지식재산권법 LLM 사전학습·IT 데이터 | 71843 | ~720MB | 공공누리 (NIA) | 지식재산권법 조문, QA, 요약 | API 다운로드 (승인 후) | ~100% | **7** |
| 9 | 민사법 LLM 사전학습·IT 데이터 | 71841 | ~785MB | 공공누리 (NIA) | 민사법 조문, QA, 요약 | API 다운로드 (승인 후) | ~100% | **7** |
| 10 | 컴플라이언스 데이터 | 71807 | ~1.7GB | 공공누리 (NIA) | 기업 규정·컴플라이언스 텍스트 | API 다운로드 (승인 후) | ~95% | **6** |

### 1-2. HuggingFace Hub 데이터셋

| # | Repo ID | 크기/규모 | 라이선스 | 내용 유형 | 다운로드 방법 | 한국어% | 우선순위 |
|---|---------|----------|----------|-----------|--------------|---------|---------|
| 11 | `smhilee/korean-law-dataset` | 중규모 (CSV+JSONL) | 미표기 | 법령 조문 전체 (법령명/공포일/시행일/소관부처/조문내용/항/호) | `datasets` 라이브러리 | 100% | **8** |
| 12 | `joonhok-exo-ai/korean_law_open_data_precedents` | 10K~100K건 | OpenRAIL | 법제처 판례 (2023년 기준 전체) | `datasets` 라이브러리 | 100% | **8** |
| 13 | `ducut91/korean-court-judgments` | **163,546건** | MIT | 국가법령정보공동활용서비스 법원 판결문 (GPT-4o-mini 요약 포함) | `datasets` 라이브러리 | 100% | **8** |
| 14 | `ducut91/korean-constitutional-court-decisions` | **35,007건** | MIT | 헌법재판소 결정문 (15개 컬럼 구조화) | `datasets` 라이브러리 | 100% | **7** |
| 15 | `Rootpye/korean-lawdata1~4` | 4개 시리즈 (zip) | Apache-2.0 | 한국 법령 데이터 (상세 불명) | HF 직접 다운로드 | 100% | **6** |
| 16 | `mosshoon/korean-laws` | 1K~10K건 | CC-BY-4.0 | 2025.08 기준 law.go.kr 법령 수집 | `datasets` 라이브러리 | 100% | **6** |
| 17 | `DistressedModel/korean_law` | 100K~1M건 | 미표기 | 한국 법률 텍스트 (상세 불명) | `datasets` 라이브러리 | 100% | **5** |
| 18 | `wisenut-nlp-team/law_korean` | 100K~1M건 | 미표기 | 한국 법률 (상세 불명) | `datasets` 라이브러리 | 100% | **5** |
| 19 | `xaikorea0/taxia-korean-tax-laws` | 소규모 | Apache-2.0 | 한국 세법 조문 | `datasets` 라이브러리 | 100% | **4** |
| 20 | `Jsoo/korean-fair-trade-law-paragraphs-org-v1` | 1K~10K건 | 미표기 | 공정거래법 조항 | `datasets` 라이브러리 | 100% | **4** |
| 21 | `91veMe4Plus-Project/korean_local_government_ordinances` | 소규모 | MIT | 지방자치단체 조례 | `datasets` 라이브러리 | 100% | **5** |

### 1-3. 국가 공식 포털 (직접 수집 필요)

| # | 소스 | URL | 크기 추정 | 라이선스 | 내용 유형 | 다운로드 방법 | 우선순위 |
|---|------|-----|----------|----------|-----------|--------------|---------|
| 22 | 법제처 국가법령정보센터 (Open API) | https://open.law.go.kr | 현행법령 5,000+ / 판례 수십만건 | 공공누리 1유형 | 법령 조문, 판례, 행정규칙 | REST API (인증키 필요) | **9** |
| 23 | 국회 의안정보시스템 회의록 | https://likms.assembly.go.kr | 수십만 건 | 공공누리 | 국회 의사록 (PDF/HWP) | 웹 크롤링 / Open API | **8** |
| 24 | KIPRIS 특허 공개 데이터 | https://www.kipris.or.kr | 수백만 건 | 공공누리 1유형 | 한국 특허·실용신안 명세서 | KIPRIS Plus API / 대용량 다운로드 | **9** |
| 25 | 공공데이터포털 법령·행정 텍스트 | https://www.data.go.kr | 다양 | 공공누리 | 행정처분, 고시, 공고 등 | API / 파일 다운로드 | **7** |
| 26 | 감사원 감사보고서 | https://www.bai.go.kr | ~수천건 | 공공누리 | 감사결과보고서, 처분요구 | 웹 크롤링 / PDF | **5** |
| 27 | 통계청 통계보고서 | https://kostat.go.kr | 다양 | 공공누리 | 각종 통계조사 보고서 | 웹 크롤링 / API | **4** |
| 28 | e-나라지표 | https://www.index.go.kr | 다양 | 공공누리 | 국가 주요 지표 해설 텍스트 | 웹 크롤링 | **3** |
| 29 | 식품의약품안전처 공개 데이터 | https://www.mfds.go.kr | 중규모 | 공공누리 | 식품·의약품 허가심사보고서 | API / 파일 다운로드 | **4** |

---

## 2. Top 3 상세 분석

### 🥇 #1: 국가기록물 대상 초거대 AI 학습 말뭉치 데이터
**[AI-Hub DataSetSn: 71788]**  
URL: https://aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&dataSetSn=71788

#### 개요
| 항목 | 내용 |
|------|------|
| 구축연도 | 2023 (최종개방: 2024-10) |
| 원천규모 | 원시데이터 **4억 토큰** (약 3억 토큰 말뭉치 정제) |
| 라벨링규모 | QA 50,000건 (의문사형 30K + Yes/No 20K) + 유해질의 10,560건 |
| 라이선스 | 공공누리 (과기정통부/NIA) |
| 형식 | JSON |
| 출처 | 국가기록원 정부간행물 (연감·백서·법규집·연구조사보고서·기관지 등) |

#### 데이터 구성
- **문서 유형별**: 연구조사보고서(12,600건), 기관지(8,367건), 사업보고서(7,397건), 교육자료(1,633건), 연감·백서(1,305건), 회의자료(592건), 법규집(271건), 사료·연혁집(9건) 등
- **주제별**: 행정(7,079건), 경제(4,659건), 정치(2,742건), 사회(2,141건), 기타(15,593건)

#### LLM 학습 활용 포인트
- **사전학습용 말뭉치**: 정부 문서 3억 토큰 — 공공/행정 도메인 지식 주입에 최적
- **Instruction Tuning용**: 의문사형·Yes/No 질의응답 50,000건
- 필드: `source_id`, `title`, `publisher_company`, `category_main`, `category_middle`, `collection_name`, `issue_date`, `corpus`

#### 다운로드 방법
```bash
# 1. AI-Hub 회원가입 + 내국인 인증
# 2. 데이터 신청 페이지에서 "다운로드" 클릭 → 승인 대기 (보통 즉시~수일)
# 3. 승인 후 API 다운로드:
aihubshell -mode d -datasetkey 71788

# 분할 압축 병합:
find "폴더경로" -name "파일명.zip.part*" -print0 | sort -zt'.' -k2V | xargs -0 cat > "파일명.zip"
unzip 파일명.zip
```

#### 품질 평가
- 한국어 순도: ~100% (정부 공식 문서)
- 도메인 다양성: 행정·정치·경제·사회·교육 포함
- LLM 학습 적합성: ★★★★★ (사전학습 + SFT 모두 가능)

---

### 🥈 #2: 국가중점기술 대응 특허 데이터
**[AI-Hub DataSetSn: 71739]**  
URL: https://aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&dataSetSn=71739

#### 개요
| 항목 | 내용 |
|------|------|
| 구축연도 | 2023 (최종개방: 2024-10) |
| 규모 | **619,844건** (특허명세서 + 기술분류 라벨) |
| 라이선스 | 공공누리 (과기정통부/NIA) |
| 형식 | JSON |
| 출처 | KIPRIS 특허 DB |

#### 데이터 구성
- **특허 필드**: 출원번호, 발명명칭, 요약, 청구항, IPC 분류, 출원인, 발명자, 등록일
- **분류 필드**: 국가중점기술 대·중·소분류 (생명/보건, ICT/SW, 에너지, 건설, 환경, 기계, 농수산, 우주, 소재 등 10개 대분류)
- 619,844건 전체에 기술분류 라벨 부여 — 분류 학습 + 사전학습 텍스트 동시 활용 가능

#### LLM 학습 활용 포인트
- **특허 명세서 텍스트** (요약 + 청구항): 한국어 기술 도메인 전문 어휘 학습
- **기술분류 태스크**: 분류 파인튜닝, 특허 분류 QA 생성 가능
- 예시: `발명명칭: 차량의 회생 제동 장치 및 그 방법 / 요약: [기술 설명] / 청구항: [청구 내용]`

#### 데이터 포맷
```json
{
  "updateDate": "2023-...",
  "documentId": "KR20120011990b1",
  "country_code": "KR",
  "application_number": "KR 2012-0011990",
  "document_type": "등록",
  "invention_title": "차량의 회생 제동 장치 및 그 방법",
  "abstract": "본 명세서는 차량의 물리 브레이크 사용을 최소화...",
  "claims": "차량의 속도를 검출하는 속도 검출부와...",
  "Lno": "F", "Ltext": "기계_제조",
  "Mno": "FC", "Mtext": "자동차",
  "Sno": "FCA", "Stext": "스마트자동차기술"
}
```

---

### 🥉 #3: 법률/규정 텍스트 분석 데이터 (판례 고도화)
**[AI-Hub DataSetSn: 71723]**  
URL: https://aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&dataSetSn=71723

#### 개요
| 항목 | 내용 |
|------|------|
| 구축연도 | 2023 (최종개방: 2024-12) |
| 규모 | 원문 약 25만건 → 라벨링 **66,511건** + QA 20,160건 |
| 라이선스 | 공공누리 (과기정통부/NIA) |
| 형식 | TXT(원문) + JSON(라벨) |
| 출처 | 대법원, 국회, 법제처 법률정보서비스 |

#### 데이터 구성
- **상황별 판례**: 민사(17K), 행정(21K), 형사(13K), 근로자(3K), 특허/저작권(3K), 금융조세(3K) 등
- **심판 유형**: 대법원 판례(40K) + 하급심(10K) + 심결례(16K)
- **라벨링 내용**: 추출요약, Q&A(판시사항 기반), 키워드, 참조법령, 참조판례, 카테고리
- **QA 데이터셋**: 법률 전문가 작성 20,160건 (질문+답변+해설+참조법령)

#### LLM 학습 활용 포인트
- 판결문 요약 (BART fine-tuning) / 판결 예측 (BERT fine-tuning) 모두 지원
- 청탁금지법, 공직자윤리법 등 행정 도메인 QA 포함
- 실제 법원 텍스트 — 법률 한국어 어휘 학습에 최적

---

## 3. 공공데이터 다운로드 가이드

### 3-1. AI-Hub (aihub.or.kr) — 가장 핵심 소스

```
URL: https://aihub.or.kr
회원가입 조건: 내국인만 신청 가능 (실명인증)
```

#### 다운로드 절차
```
1. 회원가입 → 로그인
2. 데이터 찾기 → 원하는 데이터셋 검색
3. 데이터셋 페이지에서 "다운로드" 버튼 클릭
4. 신청서 작성 (활용목적, 소속기관 등)
5. 승인 완료 후 API 키 발급
6. aihubshell CLI로 다운로드
```

#### aihubshell CLI 사용법
```bash
# 설치
pip install aihubshell

# 로그인
aihubshell -mode login -usr [아이디] -pwd [비밀번호]

# 데이터셋 다운로드 (datasetkey = DataSetSn)
aihubshell -mode d -datasetkey 71788   # 국가기록물 말뭉치
aihubshell -mode d -datasetkey 71795   # 국회 회의록
aihubshell -mode d -datasetkey 71739   # 특허 데이터
aihubshell -mode d -datasetkey 71723   # 판례 데이터
aihubshell -mode d -datasetkey 71852   # 공공 민원 상담

# 분할 압축 병합 (리눅스 필수)
find "다운로드폴더" -name "*.zip.part*" -print0 | sort -zt'.' -k2V | xargs -0 cat > output.zip
unzip output.zip
```

---

### 3-2. 법제처 국가법령정보 Open API

```
URL: https://open.law.go.kr
인증키: 무료 발급 (open.law.go.kr 회원가입)
라이선스: 공공누리 1유형 (자유 이용, 출처 표시)
```

#### 주요 API 엔드포인트
```bash
BASE_URL="https://www.law.go.kr/DRF"

# 현행 법령 목록
curl "${BASE_URL}/lawSearch.do?OC=your_key&target=law&type=JSON&query=행정"

# 특정 법령 조문 전문
curl "${BASE_URL}/lawService.do?OC=your_key&target=law&ID=법령일련번호&type=JSON"

# 판례 검색
curl "${BASE_URL}/lawSearch.do?OC=your_key&target=prec&type=JSON&query=행정처분"

# 판례 전문 조회
curl "${BASE_URL}/lawService.do?OC=your_key&target=prec&ID=판례일련번호&type=JSON"

# 행정규칙 검색
curl "${BASE_URL}/lawSearch.do?OC=your_key&target=admrul&type=JSON"
```

#### Python 예시
```python
import requests
import json

API_KEY = "your_api_key"
BASE = "https://www.law.go.kr/DRF"

def get_law_full_text(law_id):
    url = f"{BASE}/lawService.do"
    params = {"OC": API_KEY, "target": "law", "ID": law_id, "type": "JSON"}
    resp = requests.get(url, params=params)
    return resp.json()

def get_precedents(query, page=1):
    url = f"{BASE}/lawSearch.do"
    params = {"OC": API_KEY, "target": "prec", "type": "JSON",
              "query": query, "page": page, "display": 20}
    resp = requests.get(url, params=params)
    return resp.json()
```

---

### 3-3. KIPRIS 특허 데이터

```
URL: https://www.kipris.or.kr
API: https://plus.kipris.or.kr (KIPRIS Plus)
라이선스: 공공누리 1유형
```

#### KIPRIS Plus API 사용법
```bash
# 특허 검색 (출원인: 삼성)
curl "http://plus.kipris.or.kr/openapi/rest/patUtiModInfoSearchSevice/applicantNameSearch" \
  -G -d "applicantName=삼성전자" \
  -d "ServiceKey=your_key" \
  -d "pageNo=1" \
  -d "numOfRows=100" \
  -d "AbstractEng=true" \
  -d "AbstractKor=true"

# 특허 전문 (출원번호로 조회)
curl "http://plus.kipris.or.kr/openapi/rest/patUtiModInfoSearchSevice/applicationNumberSearchInfo" \
  -G -d "applicationNumber=1020120011990" \
  -d "ServiceKey=your_key" \
  -d "claimInfo=true" \   # 청구항
  -d "drawingInfo=true"
```

#### 대용량 수집 전략
```python
# 연도·기술분류별 전체 수집
# IPC 대분류: A(생활필수품) B(처리조작) C(화학) D(섬유) E(건설) F(기계) G(물리) H(전기)

import time
import requests

def collect_patents_by_ipc(ipc_code, start_year=2000, end_year=2024):
    """IPC 코드별 특허 수집"""
    all_patents = []
    for year in range(start_year, end_year + 1):
        page = 1
        while True:
            # KIPRIS Plus API 호출
            resp = requests.get(
                "http://plus.kipris.or.kr/openapi/rest/patUtiModInfoSearchSevice/ipcCpcSearchInfo",
                params={
                    "ipcNumber": ipc_code,
                    "startDate": f"{year}0101",
                    "endDate": f"{year}1231",
                    "pageNo": page,
                    "numOfRows": 100,
                    "ServiceKey": API_KEY,
                    "AbstractKor": "true",
                    "claimInfo": "true"
                }
            )
            data = resp.json()
            patents = data.get("response", {}).get("body", {}).get("items", [])
            if not patents:
                break
            all_patents.extend(patents)
            page += 1
            time.sleep(0.5)  # Rate limiting
    return all_patents
```

---

### 3-4. 국회 의안정보시스템 회의록

```
URL: https://likms.assembly.go.kr
Open API: https://open.assembly.go.kr
라이선스: 공공누리
```

#### Open API 사용법
```python
import requests

def get_assembly_minutes(era, committee, page=1):
    """국회 회의록 검색"""
    url = "https://open.assembly.go.kr/portal/openapi/NPRLAPASTABMEETX"
    params = {
        "KEY": "your_api_key",
        "Type": "json",
        "pIndex": page,
        "pSize": 100,
        "DAESU": era,        # 대 (21, 22 등)
        "CMTEE_NM": committee # 위원회 명
    }
    return requests.get(url, params=params).json()

# 전체 회의록 URL 패턴
# http://likms.assembly.go.kr/record/mhs-60-010.do?conferNum=XXXXX
```

---

## 4. 전략적 수집 권고사항

### 우선순위 Matrix

| 우선순위 | 데이터셋 | 이유 |
|---------|---------|------|
| 🔴 즉시 (Priority 9-10) | AI-Hub 71788 (국가기록물 4억 토큰) | 최대 규모 공공 텍스트, 즉시 사전학습 가능 |
| 🔴 즉시 (Priority 9-10) | AI-Hub 71739 (특허 62만건) | 기술 도메인 전문어 학습, 대규모 |
| 🔴 즉시 (Priority 9-10) | 법제처 Open API (법령+판례) | 무료 무제한, 즉시 수집 가능 |
| 🟡 단기 (Priority 7-8) | AI-Hub 71723 (판례 고도화) | 법률 QA/요약 데이터 최우선 |
| 🟡 단기 (Priority 7-8) | AI-Hub 71795 (국회 회의록) | 입법 도메인, 정치 어휘 |
| 🟡 단기 (Priority 7-8) | HF `ducut91/korean-court-judgments` (163K) | 즉시 다운로드, 추가 라벨 없이 사용 |
| 🟡 단기 (Priority 7-8) | HF `smhilee/korean-law-dataset` | 법령 전체 조문 구조화, 즉시 사용 |
| 🟢 중기 (Priority 4-6) | KIPRIS Plus API 자체 수집 | 대용량이나 크롤링 필요 |
| 🟢 중기 (Priority 4-6) | 국회 회의록 Open API 자체 수집 | AI-Hub 외 원문 보완 |

### 추정 총 수집 가능 규모

| 소스 | 추정 크기 |
|------|---------|
| AI-Hub 공공 데이터 (4개 주요셋) | ~5억 토큰 (원천 기준) |
| 법제처 API (법령+판례 전체) | ~2억 토큰 |
| KIPRIS 특허 명세서 (AI-Hub 포함) | ~5억 토큰 |
| HuggingFace 법률 데이터셋 | ~1억 토큰 |
| **합계** | **~13억 토큰** |

---

## 5. 주의사항 및 제약

1. **AI-Hub 내국인 제한**: 외국 IP 또는 외국 법인은 신청 불가. VPN 우회도 규약 위반.
2. **공공누리 라이선스**: 출처 표시 의무. 상업적 이용 가능 (1유형). 연구 목적 자유.
3. **개인정보**: 민원 데이터 등 일부에 마스킹 처리 포함.
4. **KIPRIS API 요청 제한**: 일 호출 횟수 제한 있음 (계정당 ~50,000 건/일). 대용량 수집 시 비즈니스 계정 필요.
5. **AI-Hub 데이터 분할 압축**: 리눅스 환경에서 병합 필수. `aihubshell` CLI 사용 권장.
6. **국회 Open API 인증키**: open.assembly.go.kr 에서 무료 발급.
7. **법제처 API**: `OC` 파라미터에 영문 이메일 ID 사용 (별도 발급 불필요, 이메일로 바로 사용).

---

*조사 완료: 2026-02-27 | 데이터 소스: AI-Hub, HuggingFace Hub, 법제처, KIPRIS, 국회 Open API*
