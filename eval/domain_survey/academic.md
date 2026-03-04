# 한국어 학술논문/연구보고서 도메인 데이터 전수 조사

**조사일**: 2026-02-27  
**목적**: 한국어 LLM 3B 모델 학습용 학술논문/연구보고서/학위논문 데이터 공개 현황 파악

---

## 1. 전체 데이터셋 목록

| # | 데이터셋 | 출처 | 크기 | 라이선스 | 내용 | 분야 | 다운로드 | 우선순위 |
|---|---------|------|------|----------|------|------|----------|--------|
| 1 | [amphora/korean_science_papers](https://huggingface.co/datasets/amphora/korean_science_papers) | HF | 17k행, 147MB | 미명시 | **전문(full text)** | 이공계(생물·화학 위주) | HF 직접 ✅ | **9** |
| 2 | [ddokbaro/KCI_data](https://huggingface.co/datasets/ddokbaro/KCI_data) | HF/KCI | 2.34M행 | 미명시 | 초록(영문 포함) | 전분야 (의학 포함) | HF 직접 ✅ | **8** |
| 3 | [minpeter/arxiv-abstracts-korean](https://huggingface.co/datasets/minpeter/arxiv-abstracts-korean) | HF/arXiv | 50행 | 미명시 | 영문 초록 + 한국어 번역 | 이공계 | HF 직접 ✅ | **3** |
| 4 | [AI-Hub: 필수의료 의학지식 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&dataSetSn=71875) | AI-Hub | ~101M 토큰(원문), 19,201 QA쌍 | AI-Hub 이용약관 | 학술논문+가이드라인+교과서 (원문+QA) | 의학(내과·산부인과·소아과·응급) | 신청 후 다운로드 🔐 | **8** |
| 5 | [KCI Open API](https://www.kci.go.kr/) | KCI | ~500만 논문 메타+초록 | KCI 이용약관 | 메타데이터 + 초록 | 전분야(KCI 등재지) | API Key 신청 🔐 | **7** |
| 6 | [KISTI ScienceON API](https://scienceon.kisti.re.kr/) | KISTI | 수백만 논문 | KISTI 이용약관 | 메타데이터 + 일부 전문 | 이공계(SCIE/SCOPUS 포함) | API Key 신청 🔐 | **7** |
| 7 | [RISS Open API](https://www.riss.kr/) | RISS | 수천만 학위논문/학술지 | RISS 이용약관 | 메타+초록+일부 전문(OA) | 전분야(학위논문 포함) | API Key 신청 🔐 | **6** |
| 8 | [NDSL (ScienceON 통합)](https://scienceon.kisti.re.kr/) | KISTI/NDSL | 수백만 건 | KISTI 이용약관 | 메타데이터 + 초록 | 이공계/기술 | API Key 신청 🔐 | **5** |
| 9 | [DBpia 학술논문](https://www.dbpia.co.kr/) | DBpia | 약 400만 논문 | 유료/계약 기반 | 전문(PDF) | 인문·사회·이공 전분야 | **계약 필요** ❌ | **2** |
| 10 | [AI-Hub: 한-영 과학기술 번역 코퍼스](https://www.aihub.or.kr/) | AI-Hub | ~170만 문장쌍 | AI-Hub 이용약관 | 과학기술 논문 번역문 | 이공계 | 신청 후 다운로드 🔐 | **6** |

---

## 2. Top 3 상세 분석

### 🥇 #1: `amphora/korean_science_papers`
**평가 점수: 9/10**

| 항목 | 내용 |
|------|------|
| **URL** | https://huggingface.co/datasets/amphora/korean_science_papers |
| **크기** | 17,000행, 147MB (압축) |
| **라이선스** | 미명시 (README 없음, 출처 확인 필요) |
| **내용** | 한국어 과학 논문 **전문(full text)** — 한자/LaTeX 수식 포함 |
| **분야** | 이공계 중심 (생물학, 화학, 의생명) |
| **업데이트** | 2025-07-02 |
| **다운로드** | HuggingFace 직접 (`datasets.load_dataset("amphora/korean_science_papers")`) |
| **특이사항** | LaTeX 수식 포함, OCR 기반 PDF 변환 추정, 분야 태그 없음 |

**샘플 데이터 형식**:
```json
{
  "text": "한국어 과학논문 전문 텍스트 (수식, 표, 참고문헌 포함)..."
}
```

**장점**: 한국어 학술 전문 텍스트 rare source, 즉시 다운로드 가능  
**단점**: 라이선스 불분명, 메타데이터(분야, 연도, 학술지) 없음, 규모 소규모(17k)

---

### 🥈 #2: `ddokbaro/KCI_data`
**평가 점수: 8/10**

| 항목 | 내용 |
|------|------|
| **URL** | https://huggingface.co/datasets/ddokbaro/KCI_data |
| **크기** | 2,340,000행 (~2.34M) |
| **라이선스** | 미명시 (KCI 원데이터 기반) |
| **내용** | KCI 논문 초록 + 메타데이터 (한영 혼재) |
| **분야** | 전분야 (의학·의생명 비중 높음) |
| **업데이트** | 2025-01-24 |
| **다운로드** | HuggingFace 직접 (`datasets.load_dataset("ddokbaro/KCI_data")`) |
| **특이사항** | 영문 초록 포함, 일부 한국어 초록. KCI API로 수집한 데이터로 추정 |

**샘플 데이터 형식** (Viewer 기준):
```json
{
  "abstracts": {"abstract1": "...", "abstract2": "..."},
  "metadata": { ... }
}
```

**장점**: 대규모(2.34M), 즉시 다운로드 가능, 학술 도메인 어휘 풍부  
**단점**: 영문 비중 불명확, 초록 수준(전문 없음), 라이선스 불분명  

---

### 🥉 #3: `AI-Hub 필수의료 의학지식 데이터`
**평가 점수: 8/10**

| 항목 | 내용 |
|------|------|
| **URL** | https://www.aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&dataSetSn=71875 |
| **크기** | 원문 약 1억 토큰(국문+영문), QA 19,201쌍 |
| **라이선스** | AI-Hub 이용약관 (비상업적 학술 연구 허용) |
| **내용** | 학술논문/저널 원문, 학회 가이드라인, 의학 교과서 + QA 데이터셋 |
| **분야** | 의학 (내과, 산부인과, 소아청소년과, 응급의학) |
| **업데이트** | 2025-06-30 |
| **다운로드** | AI-Hub 회원가입 → 신청 → 승인 후 다운로드 (내국인 한정) |
| **특이사항** | Big5 병원 4곳 참여, 전문 + QA 동시 제공, JSON 포맷 |

**국문 원천데이터 상세**:
| 출처 | 토큰 수 |
|------|---------|
| 학술 논문 및 저널 | 15,928,056 |
| 학회 가이드라인 | 7,709,412 |
| 의학 교과서 | 647,538 |
| 기타(동의서 등) | 39,799,317 |

**장점**: 고품질 QA 포함, 의학 도메인 전문 어휘, JSON 정형화  
**단점**: 의학 단일 분야, 내국인 신청 필요, 기타(동의서) 비중이 높아 정제 필요

---

## 3. API 신청 방법 정리

### KCI (한국학술지인용색인) Open API
- **URL**: https://www.kci.go.kr/
- **제공 데이터**: 논문 메타데이터, 초록, 인용 정보
- **신청 방법**:
  1. https://www.kci.go.kr 회원가입
  2. 상단 메뉴 → 오픈API 신청
  3. 활용목적 기재 후 API Key 발급 (심사 없이 즉시 발급 가능)
- **제약**: 초록만 제공, 전문은 제공 안 함
- **API 예시**: `GET https://www.kci.go.kr/kciportal/po/openapi/openApiSerList.kci?apiCode=...&apiKey=<KEY>`
- **비용**: 무료

### KISTI ScienceON (NDSL 통합) API
- **URL**: https://scienceon.kisti.re.kr/
- **제공 데이터**: 국내외 논문 메타+초록 (KCI, SCOPUS, PubMed 등 통합)
- **신청 방법**:
  1. ScienceON 회원가입
  2. 오픈API 메뉴 → API Key 신청
  3. 활용목적 제출 → 심사 후 발급 (1~3일)
- **제약**: 전문(full text)은 원칙적으로 제공 안 함, 초록 위주
- **비용**: 무료 (상업적 이용 제한)

### RISS Open API
- **URL**: https://www.riss.kr/ (OpenAPI 메뉴)
- **제공 데이터**: 학위논문/학술지/단행본 메타+일부 초록. **OA 논문 전문 링크** 포함
- **신청 방법**:
  1. RISS 회원가입
  2. 마이페이지 → Open API 신청
  3. 목적 기재 → 즉시 또는 1~2일 내 발급
- **특징**: 학위논문(석사/박사) 메타데이터 강점. OA 논문은 PDF 링크 제공
- **비용**: 무료

### AI-Hub 데이터 신청
- **URL**: https://www.aihub.or.kr/
- **신청 방법**:
  1. AI-Hub 회원가입 (내국인 실명인증 필요)
  2. 원하는 데이터셋 페이지 → "다운로드" 버튼
  3. 활용목적 기재 → 자동 승인 (대부분 즉시) 또는 1~3일
  4. 데이터 다운로드 (PC에서만 가능)
- **비용**: 무료 (비상업적 연구 목적)
- **주의**: 데이터 재배포 금지, 논문/결과물 발표 시 AI-Hub 출처 명기

### DBpia (참고 - 권장하지 않음)
- **URL**: https://www.dbpia.co.kr/
- 기관 구독 또는 개인 유료 결제 필요
- 대량 다운로드/API 제공 없음 → **LLM 학습용으로 사용 불가**

---

## 4. 추가 탐색 권장 소스

| 소스 | URL | 내용 | 비고 |
|------|-----|------|------|
| arXiv Korean subset | https://arxiv.org/search/?query=korean&searchtype=all | arXiv 한국어 포함 논문 | Python으로 bulk 수집 가능 |
| PubMed Open Access | https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/ | 의학 OA 전문 | 한국 저자 한국어 초록 포함 |
| DOAJ Korea | https://doaj.org/search/journals?query=korea | OA 학술지 | 학술지 전문 무료 |
| 국회전자도서관 | https://dl.nanet.go.kr/ | 연구보고서 원문 | OA 많음 |
| 한국교육학술정보원(KERIS) | https://www.riss.kr/ | RISS와 동일 | - |

---

## 5. 요약 및 권장 전략

### 즉시 사용 가능 (HuggingFace 직접 다운로드)
1. `amphora/korean_science_papers` — 147MB, 한국어 과학논문 전문. **라이선스 확인 후 즉시 사용 가능**
2. `ddokbaro/KCI_data` — 2.34M행, KCI 초록 대규모. **즉시 사용 가능**
3. `minpeter/arxiv-abstracts-korean` — 소규모(50개), arXiv 초록 한영. 보조 자료 수준

### 신청 후 확보 가능 (1주 이내)
4. AI-Hub 필수의료 의학지식 데이터 — 의학 전문, 고품질 QA 포함
5. KCI Open API — 초록 대규모 수집 가능 (스크래핑 필요)
6. RISS Open API — 학위논문 메타/초록 + OA 전문 링크

### 권장 우선순위 실행 계획
```
1단계 (즉시): HF 직접 다운로드
   - amphora/korean_science_papers (전문 확보)
   - ddokbaro/KCI_data (초록 대규모)

2단계 (1주): AI-Hub 신청
   - 필수의료 의학지식 데이터 (의학 도메인 강화)

3단계 (2-4주): API 신청 후 수집
   - KCI API → 논문 메타+초록 대규모 수집
   - RISS API → 학위논문 초록 + OA 전문
   
4단계 (장기): OA 전문 수집
   - RISS OA 링크 통해 학위논문 전문 PDF → 텍스트 변환
   - PubMed Central OA 한국 저자 논문 수집
```

---

*조사 방법: HuggingFace Hub 키워드 검색(korean academic/science/thesis/KCI/RISS), AI-Hub 웹 크롤링, KCI/RISS/KISTI 공식 홈페이지 직접 확인*
