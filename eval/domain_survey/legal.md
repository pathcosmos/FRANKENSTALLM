# 한국어 법률/판례/법령 도메인 데이터 전수 조사

> 작성일: 2026-02-27  
> 목적: 한국어 LLM 3B 모델 학습용 법률 도메인 데이터 확보 전략 수립  
> 조사 범위: HuggingFace Hub, AI-Hub, law.go.kr, 대법원, GitHub

---

## 1. 전체 목록 테이블

### 1-A. HuggingFace Hub 데이터셋

| # | Repo ID | 다운로드수(월) | 크기/샘플수 | 라이선스 | 내용 | 상업적이용 | 우선순위 |
|---|---------|--------------|------------|---------|------|-----------|---------|
| 1 | `joonhok-exo-ai/korean_law_open_data_precedents` | 115 | 85,830건 (판결문 전문) | 공공저작물 | 대법원 판례 전문 (사건명, 선고일, 판결요지, 전문텍스트) | ✅ 가능 | **10** |
| 2 | `DistressedModel/korean_law` | 15 | 475,000+ rows | Unknown | 법령 전문 (국가법령정보센터 기반, 지방자치단체 규칙 포함) | ❓ 확인필요 | **9** |
| 3 | `LuminaMotionAI/korean-legal-dataset` | 69 | 160,000건 | Unknown | 헌법재판소 결정례 QA (질문+답변 쌍) | ❓ 확인필요 | **9** |
| 4 | `smhilee/korean-law-dataset` | 7 | ~182건(샘플) | Unknown | 법령 전문 (조문 단위, 식품의약품안전처 등) | ❓ 확인필요 | **6** |
| 5 | `mosshoon/korean-laws` | 21 | 5,500건 (법령 전체) | Unknown | 법령 전문 (국가법령정보센터 출처 명시, 조문 통합본) | ✅ 공공저작물 | **8** |
| 6 | `wisenut-nlp-team/law_korean` | 4 | 233,000건 | Unknown | 계약서 전문 (비밀유지계약, 임대차 등 다양한 계약 유형) | ❓ 확인필요 | **8** |
| 7 | `ohsuz/korean_law_edu` | 5 | 224,000건 | 요청필요 | 법률교육 데이터 (접근동의 필요) | ❓ gated | **5** |
| 8 | `psyche/korean-law` | 4 | 5,410건 | Unknown | 법령 조문 단위 데이터 | ❓ 확인필요 | **5** |
| 9 | `JusWis/korean-legal-terminology` | 25 | 17,500건 | Unknown | 법률 용어사전 (한자+한글+정의) | ❓ 확인필요 | **7** |
| 10 | `paperw8/korean_legal_terminology` | 18 | 6,180건 | Unknown | 법률 용어 설명 데이터 | ❓ 확인필요 | **6** |
| 11 | `paperw8/korean_legal_terminology_sharegpt` | 3 | 18,500건 | Unknown | 법률 용어 ShareGPT 포맷 변환본 | ❓ 확인필요 | **6** |
| 12 | `neuralfoundry-coder/korean-legal-instruction-sample` | 30 | 5,470건 | Unknown | 법률 QA instruction (민사법, 형사법, 노동법 등 AI-Hub 기반) | ❓ 확인필요 | **7** |
| 13 | `joonhok-exo-ai/korean_law_case_codes` | 6 | 199건 | 공공저작물 | 판례 사건코드 매핑 | ✅ 가능 | **3** |
| 14 | `Rootpye/korean-lawdata1~4` | ~100 each | 미상 | Unknown | 법률 데이터 (4개 분할) | ❓ 확인필요 | **4** |
| 15 | `xaikorea0/taxia-korean-tax-laws` | 15 | 미상 | Unknown | 세법 전문 | ❓ 확인필요 | **5** |
| 16 | `MisileLab/korean-law-dataset` | 2 | 550건 | Unknown | 법률 데이터셋 | ❓ 확인필요 | **3** |
| 17 | `abraham-diress/korean_land_mgmt_law_exams` | 3 | 766건 | Unknown | 토지관리법 시험문제 | ❓ 확인필요 | **2** |
| 18 | `Jsoo/korean-fair-trade-law-paragraphs-org-v1` | 4 | 1,130건 | Unknown | 공정거래법 단락 단위 | ❓ 확인필요 | **3** |

---

### 1-B. AI-Hub 법률 카테고리 (11건, 회원가입 + 내국인 신청 필요)

| # | 데이터명 | 데이터셋 번호 | 크기 | 내용 | 라이선스 | 상업적이용 | 우선순위 |
|---|---------|------------|------|------|---------|-----------|---------|
| 1 | **민사법 LLM 사전학습 및 Instruction Tuning 데이터** | 71841 | 100,130건 (판결문 91k, 법령, 심결례, 유권해석) | QA + 요약 태스크, JSON | AI-Hub 이용약관 | ❌ 비상업 | **10** |
| 2 | **형사법 LLM 사전학습 및 Instruction Tuning 데이터** | 71848 | 원천 305만문장, 라벨링 100,000건 | QA + 요약, 판결문 83%, 법령 11%, 해석례 6% | AI-Hub 이용약관 | ❌ 비상업 | **10** |
| 3 | **행정법 LLM 사전학습 및 Instruction Tuning 데이터** | 71847 | 256MB 수준 (라벨링 ~100k 추정) | 행정법 판결문, 법령, 심결례 | AI-Hub 이용약관 | ❌ 비상업 | **9** |
| 4 | **지식재산권법 LLM 사전학습 및 Instruction Tuning 데이터** | 71843 | 720MB 수준 | 지식재산권 법령, 심결례 QA | AI-Hub 이용약관 | ❌ 비상업 | **8** |
| 5 | **계약 외 법률 문서 서식 데이터** | 71835 | 10,299건 (라벨링 284,445건) | 소장, 고소장, 신청서, 준비서면 등 서식 | AI-Hub 이용약관 | ❌ 비상업 | **9** |
| 6 | **계약 법률 문서 서식 데이터** | (목록에서 확인됨) | 미상 | 계약서 서식 (약 9,652건 추정) | AI-Hub 이용약관 | ❌ 비상업 | **9** |
| 7-11 | 기타 법률 데이터 5건 | 미상 | 미상 | 법률 관련 추가 데이터셋 | AI-Hub 이용약관 | ❌ 비상업 | **6~8** |

---

### 1-C. 국가법령정보센터 (law.go.kr) 공개 API

| 소스 | URL | 크기 | 내용 | 라이선스 | 상업적이용 | 우선순위 |
|------|-----|------|------|---------|-----------|---------|
| 법령정보 API | `https://open.law.go.kr/LSO/openApi.do` | 현행법령 5,000+개 | 법령 전문, 조문 단위 API | 공공저작물 자유이용허락 | ✅ 가능 | **10** |
| 판례 검색 API | `https://open.law.go.kr` | 대법원·헌법재판소 판례 수십만건 | 판례 원문, 판시사항, 판결요지 | 공공저작물 | ✅ 가능 | **10** |
| 행정규칙 | 동일 API | 수만건 | 훈령, 예규, 고시 등 | 공공저작물 | ✅ 가능 | **8** |

> **특이사항**: law.go.kr API는 **API키 발급 필요** (무료, 회원가입). `joonhok-exo-ai/korean_law_open_data_precedents`는 이 API의 판례 데이터를 HuggingFace에 미러링한 것으로 추정.

---

### 1-D. 대법원 판례 공개 데이터

| 소스 | URL | 크기 | 내용 | 라이선스 | 상업적이용 | 우선순위 |
|------|-----|------|------|---------|-----------|---------|
| 대법원 판례검색 | `https://www.law.go.kr/precSc.do` | 수십만건+ | 대법원, 하급심 판결문 | 공공저작물 | ✅ 가능 | **9** |
| 종합법률정보 | `https://glaw.scourt.go.kr` | 대법원 판결 전문 | 민사·형사·행정 판결 | 공공저작물 | ✅ 가능 | **9** |

---

### 1-E. GitHub NLP 법률 데이터

| 소스 | URL | 내용 | 우선순위 |
|------|-----|------|---------|
| joonhok-exo-ai 관련 repo | GitHub 검색 | 법률 데이터 수집 스크립트 | **5** |
| duck3244/llama_finetune_project | GitHub | 한국 부동산 법률 QA | **3** |
| AI-Hub 활용 NLP 연구들 | 다수 | 법률 NLP 벤치마크 및 파인튜닝 | **4** |

---

## 2. Top 3 데이터셋 상세

---

### 🥇 Top 1: AI-Hub 형사법 LLM 사전학습 및 Instruction Tuning 데이터

| 항목 | 내용 |
|------|------|
| **Repo/URL** | https://aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&dataSetSn=71848 |
| **크기** | 원천 3,050,000 문장 / 라벨링 100,000건 |
| **용량** | ~235MB (라벨링 기준) |
| **라이선스** | AI-Hub 이용약관 (비상업적 연구 허용) |
| **내용** | 법령(11%), 판결문(83%), 해석례(6%), 결정례(0.03%). QA 59%, 요약 41% |
| **데이터 출처** | 법제처 국가법령정보센터, 대한민국 법원, 국세청 직접 수집 |
| **다운로드 방법** | AI-Hub 회원가입 → 데이터 신청(승인 1~3일) → CLI 다운로드 (내국인만 가능) |
| **상업적 이용** | ❌ 불가 (연구·비상업 목적만) |
| **포맷** | JSON (instruction/input/output 구조) |
| **특이사항** | 원천 데이터(305만 문장)가 사전학습에도 활용 가능. Llama-3-Open-Ko-8B로 검증됨. |
| **우선순위** | **10/10** |

**샘플 데이터 구조:**
```json
{
  "DocuType": "02",
  "doc_id": "서울남부지방법원-2017고단2381",
  "announce_date": "2017-10-19",
  "casenames": "자동차관리법위반...",
  "normalized_court": "서울남부지방법원",
  "casetype": "criminal",
  "taskType": "01(QA)",
  "instruction": "...",
  "input": "...",
  "output": "..."
}
```

---

### 🥈 Top 2: joonhok-exo-ai/korean_law_open_data_precedents (HuggingFace)

| 항목 | 내용 |
|------|------|
| **Repo ID** | `joonhok-exo-ai/korean_law_open_data_precedents` |
| **URL** | https://huggingface.co/datasets/joonhok-exo-ai/korean_law_open_data_precedents |
| **크기** | 85,830건 (train split 1개) |
| **라이선스** | 공공저작물 자유이용허락 (대한민국 법원 공개 데이터) |
| **내용** | 대법원 판례 전문. 필드: 판례정보일련번호, 사건명, 사건번호, 선고일자, 법원명, 사건종류(민사/형사/행정 등), 판결유형, 판시사항, 판결요지, 참조조문, 참조판례, **전문(최대 864k자)** |
| **다운로드 방법** | `datasets.load_dataset("joonhok-exo-ai/korean_law_open_data_precedents")` |
| **상업적 이용** | ✅ 가능 (공공저작물) |
| **포맷** | Parquet (HF datasets) |
| **특이사항** | 즉시 다운로드 가능. 판결 전문 포함으로 사전학습 코퍼스로 바로 활용 가능. 가장 오래된 판례는 1947년까지 거슬러 올라감. |
| **우선순위** | **10/10** |

**컬럼 목록:**
```
판례정보일련번호, 사건명, 사건번호, 선고일자, 선고, 법원명,
사건종류명, 판결유형, 판시사항, 판결요지, 참조조문, 참조판례, 전문
```

---

### 🥉 Top 3: law.go.kr 공개 API (법령 + 판례)

| 항목 | 내용 |
|------|------|
| **URL** | https://open.law.go.kr/LSO/openApi.do |
| **크기** | 현행법령 5,000+종 / 판례 수십만건 (지속 업데이트) |
| **라이선스** | **공공저작물 자유이용허락** (공유·변형·상업적이용 모두 가능) |
| **내용** | ① 법령API: 법령명, 조문번호, 조문제목, 조문내용, 별표/서식; ② 판례API: 사건번호, 선고일, 법원명, 판시사항, 판결요지, 전문 |
| **다운로드 방법** | API키 신청 → REST API 호출 (XML/JSON 응답). 예: `https://www.law.go.kr/DRF/lawSearch.do?OC={API키}&target=prec&type=JSON` |
| **상업적 이용** | ✅ 가능 |
| **포맷** | JSON 또는 XML |
| **특이사항** | **가장 공식적이고 완전한 소스**. 최신 법령 반영. `mosshoon/korean-laws`, `smhilee/korean-law-dataset`, `DistressedModel/korean_law` 등 HF 데이터셋 다수가 이 API 기반. API 일일 호출 제한 있음 (보통 1,000건/회 배치). |
| **우선순위** | **10/10** |

**활용 방법:**
```python
import requests
url = "https://www.law.go.kr/DRF/lawSearch.do"
params = {
    "OC": "{발급받은_API키}",
    "target": "prec",      # 판례
    "type": "JSON",
    "query": "",
    "page": 1,
    "display": 100
}
resp = requests.get(url, params=params)
```

---

## 3. 추가 발굴 데이터셋

### LuminaMotionAI/korean-legal-dataset (HF)
- 160,000건의 헌법재판소 결정례 기반 QA
- 질문-답변 쌍으로 Instruction Tuning에 최적
- 라이선스 불명확하나 헌재 공개데이터 기반으로 추정
- 우선순위: **8/10**

### AI-Hub 민사법 LLM 데이터 (71841)
- 100,130건 (판결문 91,285 + 법령 + 심결례 + 유권해석)
- 형사법과 유사 구조, 민사 특화
- 우선순위: **10/10**

### AI-Hub 계약 외 법률 문서 서식 데이터 (71835)
- 10,299건 계약 외 법률 서식 (소장, 신청서, 고소장, 준비서면 등)
- 법률 문서 생성 태스크에 유용
- 우선순위: **9/10**

### wisenut-nlp-team/law_korean (HF)
- 233,000건 계약서 전문
- NDA, 용역계약, 임대차 등 다양한 계약 유형 포함
- 계약서 생성/이해 능력 향상에 최적
- 우선순위: **8/10**

---

## 4. 데이터 수집 우선순위 로드맵

```
Phase 1 (즉시, 상업적이용 가능):
  ✅ joonhok-exo-ai/korean_law_open_data_precedents  → HF datasets 즉시 다운로드
  ✅ law.go.kr API → API키 발급 후 전량 수집 (법령 + 판례)
  ✅ mosshoon/korean-laws                             → HF datasets 즉시 다운로드

Phase 2 (AI-Hub 신청, 비상업 연구용):
  📋 형사법 LLM 데이터 (71848)    → 가장 큰 규모, 즉시 신청
  📋 민사법 LLM 데이터 (71841)    → 두 번째로 많은 QA쌍
  📋 계약 외 법률 문서 서식 (71835) → 법률 문서 서식 특화
  📋 행정법 LLM 데이터 (71847)
  📋 지식재산권법 데이터 (71843)

Phase 3 (라이선스 확인 후):
  ⚠️  DistressedModel/korean_law     → 475k rows, 라이선스 확인 필요
  ⚠️  LuminaMotionAI/korean-legal-dataset → 160k QA, 라이선스 확인 필요
  ⚠️  wisenut-nlp-team/law_korean    → 233k 계약서, 라이선스 확인 필요
  ⚠️  JusWis/korean-legal-terminology → 17.5k 법률 용어사전
```

---

## 5. 예상 총 데이터 볼륨

| 카테고리 | 건수 | 예상 텍스트량 |
|---------|------|-------------|
| HF 즉시 활용 (상업용) | ~92k건 | ~5GB |
| AI-Hub (비상업 연구) | ~500k건+ | ~20GB |
| law.go.kr API 수집 | 법령 5k종 + 판례 수십만 | ~10GB |
| HF 라이선스 확인 후 | ~700k건 | ~15GB |
| **합계** | **~1.3M건+** | **~50GB** |

---

## 6. 권고사항

1. **law.go.kr API 우선 수집**: 공공저작물로 상업적 이용 무제한. 판례+법령 완전 커버리지.
2. **AI-Hub 신청 병행**: 비상업 연구용이지만 가장 고품질의 Instruction Tuning 데이터. 형사법/민사법 동시 신청.
3. **HF 즉시 활용**: `joonhok-exo-ai/korean_law_open_data_precedents` 85k 판례는 오늘 당장 사용 가능.
4. **라이선스 확인 필요**: `DistressedModel/korean_law`(475k), `LuminaMotionAI`(160k)는 라이선스 명확히 확인 후 사용.
5. **계약서 데이터**: `wisenut-nlp-team/law_korean` 233k 계약서는 법률 도메인 다양성 확보에 핵심.

---

*조사일: 2026-02-27 | 조사자: survey-legal subagent*
