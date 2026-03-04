# 한국어 금융/경제/비즈니스 도메인 데이터셋 전수 조사

> **목적**: 한국어 LLM 3B 모델 학습용 금융·경제·주식·비즈니스 도메인 데이터 발굴  
> **조사일**: 2026-02-26  
> **조사 방법**: HuggingFace Hub 전수 검색 (web_fetch), 공공 데이터 포털 확인  
> **검색 키워드**: korean finance, korean financial, korean stock, korean economy, dart korea, korean business

---

## 전체 데이터셋 목록

| # | Repo ID | 샘플수 | 크기 | 라이선스 | 내용 | 태스크 | 상업적 이용 | 우선순위 |
|---|---------|--------|------|----------|------|--------|-------------|----------|
| 1 | [nmixx-fin/opensource_korean_finance_datasets](https://huggingface.co/datasets/nmixx-fin/opensource_korean_finance_datasets) | 502,831 | ~532MB | 오픈소스(혼합) | 한국어 금융 텍스트 다종 합본 (뉴스·리포트·사전·공시 등) | 다목적 (사전학습·SFT) | ⚠️ 출처별 확인 필요 | **10** |
| 2 | [nayohan/Sujet-Finance-Instruct-177k-ko](https://huggingface.co/datasets/nayohan/Sujet-Finance-Instruct-177k-ko) | 177,000 | ~수백MB | Apache 2.0 추정 | Finnish 금융뉴스 기반 한국어 번역 감성분석 instruction | 감성분석·SFT | ✅ 가능 | **9** |
| 3 | [nmixx-fin/twice_kr_finance_news_summ](https://huggingface.co/datasets/nmixx-fin/twice_kr_finance_news_summ) | 54,700 | ~중간 | 오픈소스 | 한국 금융뉴스 요약 (article + summary + quality label 0/1) | 요약·SFT | ⚠️ 확인 필요 | **9** |
| 4 | [imTak/korean-audio-text-economy](https://huggingface.co/datasets/imTak/korean-audio-text-economy) | 43,200 | ~대용량 | 미확인 | 한국어 경제 오디오+텍스트 (음성 전사) | ASR·텍스트추출 | ⚠️ 확인 필요 | **5** |
| 5 | [nmixx-fin/synthetic_financial_report_korean](https://huggingface.co/datasets/nmixx-fin/synthetic_financial_report_korean) | 20,800 | ~소형 | 오픈소스 | 합성 시황 데이터 (category 7종: 시황 등, source=synthetic) | 텍스트생성·SFT | ✅ 가능 (합성) | **7** |
| 6 | [nmixx-fin/NMIXX_train](https://huggingface.co/datasets/nmixx-fin/NMIXX_train) | 18,800 | ~소형 | 오픈소스 | 한국어-영어 금융뉴스 병렬 코퍼스 (KOSPI·KOSDAQ·글로벌 시황) | 번역·사전학습 | ⚠️ 확인 필요 | **6** |
| 7 | [nmixx-fin/twice_kr_finance_reranking](https://huggingface.co/datasets/nmixx-fin/twice_kr_finance_reranking) | 30,500 | ~소형 | 오픈소스 | 한국 금융 문서 리랭킹 (쿼리-문서 쌍) | 검색·랭킹·RAG | ⚠️ 확인 필요 | **6** |
| 8 | [kgmyh/korean_stock_ticker_qa_data](https://huggingface.co/datasets/kgmyh/korean_stock_ticker_qa_data) | 13,800 | ~소형 | 미확인 | 한국 주식 종목코드 QA (종목명→코드 매핑) | QA·도메인지식 | ⚠️ 확인 필요 | **5** |
| 9 | [nmixx-fin/synthetic_dart_report_korean](https://huggingface.co/datasets/nmixx-fin/synthetic_dart_report_korean) | 5,080 | ~소형 | 오픈소스 | DART 사업보고서 기반 합성 요약 (한화리츠 등 실제 상장법인) | 요약·SFT | ✅ 가능 (합성) | **8** |
| 10 | [nmixx-fin/twice_bok_dict_retrieval](https://huggingface.co/datasets/nmixx-fin/twice_bok_dict_retrieval) | 3,000 | ~소형 | 오픈소스 | 한국은행 경제금융용어 사전 검색 | 검색·RAG | ✅ 가능 | **7** |
| 11 | [nmixx-fin/twice_fss_dict_retrieval](https://huggingface.co/datasets/nmixx-fin/twice_fss_dict_retrieval) | 3,000 | ~소형 | 오픈소스 | 금융감독원 금융용어 사전 검색 | 검색·RAG | ✅ 가능 | **7** |
| 12 | [nmixx-fin/twice_kr_market_report_retrieval](https://huggingface.co/datasets/nmixx-fin/twice_kr_market_report_retrieval) | 3,000 | ~소형 | 오픈소스 | 한국 시장 리포트 검색 (쿼리-문서 쌍) | 검색·RAG | ⚠️ 확인 필요 | **6** |
| 13 | [nmixx-fin/twice_kr_news_retrieval](https://huggingface.co/datasets/nmixx-fin/twice_kr_news_retrieval) | 3,000 | ~소형 | 오픈소스 | 한국 금융뉴스 검색 (쿼리-문서 쌍) | 검색·RAG | ⚠️ 확인 필요 | **6** |
| 14 | [nmixx-fin/korfinSTS](https://huggingface.co/datasets/nmixx-fin/korfinSTS) | 1,990 | ~소형 | 오픈소스 | 한국 금융보고서 STS (KOSPI·채권·글로벌 매크로 문장 쌍, label=1) | STS·임베딩 | ⚠️ 확인 필요 | **6** |
| 15 | [Nexdata/215_Hours_Korean_Financial_Entities_Speech_Data](https://huggingface.co/datasets/Nexdata/215_Hours_Korean_Financial_Entities_Speech_Data) | 215시간 | ~대용량 | 상업적(유료 가능성) | 한국 금융 엔티티 음성 데이터 (NER 태깅) | ASR·NER | ❌ 유료/제한 | **3** |

---

## 소스별 보완 정보

### 🔴 HuggingFace 외 공개 소스 (직접 접근 필요)

| 소스 | URL | 내용 | 접근 방법 | 비고 |
|------|-----|------|-----------|------|
| DART 전자공시 API | https://dart.fscr.or.kr | 상장법인 사업보고서·분기보고서·공시문서 | API Key 발급 후 REST API | ✅ 무료, 대량 수집 가능 |
| 한국은행 ECOS | https://ecos.bok.or.kr | 경제통계 수치 데이터 | API Key 발급 후 REST API | ✅ 무료, 시계열 수치 중심 |
| 한국거래소 KRX | http://data.krx.co.kr | 주식·ETF·채권 시장 데이터 | 웹 다운로드 (CSV) | ✅ 무료, 수치 데이터 중심 |
| AI-Hub 금융 카테고리 | https://aihub.or.kr | 금융 도메인 음성·텍스트 | 회원가입 후 신청 | ⚠️ 비상업적 연구용 |
| 법제처 금융법령 | https://law.go.kr | 금융 관련 법령 전문 | 웹 크롤링 (공공저작물) | ✅ 공공저작물 |

---

## Top 3 상세 분석

---

### 🥇 #1. `nmixx-fin/opensource_korean_finance_datasets`

**우선순위: 10/10**

#### 개요
- **HuggingFace**: https://huggingface.co/datasets/nmixx-fin/opensource_korean_finance_datasets
- **샘플수**: 502,831행
- **파일 크기**: ~532MB (Parquet)
- **라이선스**: 혼합 (출처별 상이)
- **업데이트**: 2024–2025년 활발 유지

#### 내용 구성
한국어 금융 특화 텍스트를 다종 병합한 메가 데이터셋. 내부 구성:
- 한국 금융뉴스 (경제·시황·기업·주식)
- 금융보고서·리서치 리포트
- 한국은행·금융감독원 사전 텍스트
- DART 공시 관련 문서
- 합성 금융 텍스트

#### 컬럼 구조
```
text, category, source, token_count (추정)
```

#### 다운로드 방법
```python
from datasets import load_dataset
ds = load_dataset("nmixx-fin/opensource_korean_finance_datasets")
```
또는
```bash
huggingface-cli download nmixx-fin/opensource_korean_finance_datasets --repo-type dataset
```

#### 활용 방안
- **사전학습(Continual Pretraining)**: 502k 규모 금융 도메인 텍스트로 도메인 적응
- **SFT 데이터 소스**: 텍스트에서 instruction 쌍 자동 생성 가능
- **RAG 인덱싱**: 금융 문서 검색 시스템 구축용

#### 주의사항
- 혼합 라이선스이므로 상업적 이용 전 출처별 라이선스 검토 필수
- 합성 데이터 포함 여부 확인 후 학습 파이프라인 분리 권장

---

### 🥈 #2. `nmixx-fin/twice_kr_finance_news_summ`

**우선순위: 9/10**

#### 개요
- **HuggingFace**: https://huggingface.co/datasets/nmixx-fin/twice_kr_finance_news_summ
- **샘플수**: ~54,700행
- **라이선스**: 오픈소스
- **업데이트**: 2025년 1월

#### 내용 구성
한국 금융뉴스 기사 → 한 문장 요약 쌍. 품질 레이블 포함:
- `article`: 전문 금융기사 (항만공사·POSCO·지자체 경제뉴스 등)
- `summary`: 한 문장 요약
- `label`: 품질 지표 (0=저품질, 1=고품질)

#### 다운로드 방법
```python
from datasets import load_dataset
ds = load_dataset("nmixx-fin/twice_kr_finance_news_summ")
# label=1만 필터링 권장
ds_clean = ds.filter(lambda x: x['label'] == 1)
```

#### 활용 방안
- **요약 SFT**: 금융뉴스 요약 능력 특화 파인튜닝
- **instruction 변환**: "다음 금융기사를 한 문장으로 요약하시오" 포맷으로 변환
- **품질 필터**: `label=1` 기준으로 고품질 서브셋 추출 (~수만 샘플)

#### 주의사항
- 뉴스 원문의 저작권 확인 필요 (언론사별 상이)
- `label=0` 데이터는 학습 전 제거 권장

---

### 🥉 #3. `nayohan/Sujet-Finance-Instruct-177k-ko`

**우선순위: 9/10**

#### 개요
- **HuggingFace**: https://huggingface.co/datasets/nayohan/Sujet-Finance-Instruct-177k-ko
- **샘플수**: 177,000행
- **라이선스**: Apache 2.0 추정 (원본 Sujet-Finance-Instruct 기반)
- **업데이트**: 2024년

#### 내용 구성
Finnish 금융뉴스 코퍼스(PhinsAFN)를 한국어로 번역·변환한 감성분석 instruction 데이터:
- `instruction`: 한국어 금융뉴스 문장
- `output`: 감성 레이블 (0=부정, 1=중립, 2=긍정, 3=강한긍정 추정)
- `source`: 뉴스 출처

#### 컬럼 예시
```
{"instruction": "애플 주가가 폭락하면서 나스닥이 하락했다.", "output": "부정", "label": 0}
```

#### 다운로드 방법
```python
from datasets import load_dataset
ds = load_dataset("nayohan/Sujet-Finance-Instruct-177k-ko")
```

#### 활용 방안
- **감성분석 SFT**: 금융텍스트 감성분류 특화 파인튜닝
- **instruction 다양화**: 감성분석 외 다른 태스크로 재포맷 가능
- **대규모 SFT 베이스**: 177k 규모로 instruction-following 능력 강화

#### 주의사항
- 번역 품질 불균일 가능 (자동번역 포함)
- Finnish 금융 뉴스 기반이므로 한국 금융 특화 표현보다는 글로벌 금융 뉴스 중심
- 원본 라이선스(Apache 2.0) 확인 권장

---

## 추가 권장 수집 액션

### 즉시 실행 가능
1. **DART API 크롤링**: `dart.fscr.or.kr` API Key 발급 → 최근 5년 사업보고서 전문 수집 (수십만 문서)
2. **한국은행 통화정책 보고서**: BOK 웹사이트에서 PDF 다운로드 → 텍스트 추출
3. **법제처 금융법령**: 공공저작물로 자유 이용 가능

### 중기 수집 권장
4. **AI-Hub 금융 데이터**: 회원가입 후 신청 (비상업용 연구 라이선스)
5. **증권사 리서치 리포트**: 네이버 증권·한국IR협의회 등에서 공개 PDF 수집
6. **한국경제·매일경제 뉴스**: RSS 또는 공개 아카이브 크롤링

---

## 요약 및 학습 전략 제안

### 우선순위별 활용 로드맵

| 단계 | 데이터셋 | 목적 |
|------|---------|------|
| 1단계 (사전학습) | `nmixx-fin/opensource_korean_finance_datasets` (502k) | 금융 도메인 언어 패턴 학습 |
| 2단계 (SFT-요약) | `nmixx-fin/twice_kr_finance_news_summ` (54k, label=1) | 뉴스 요약 능력 |
| 2단계 (SFT-감성) | `nayohan/Sujet-Finance-Instruct-177k-ko` (177k) | 감성분석·instruction-following |
| 3단계 (SFT-공시) | `nmixx-fin/synthetic_dart_report_korean` (5k) | 공시 문서 이해·요약 |
| 3단계 (RAG준비) | `nmixx-fin/twice_bok_dict_retrieval` + `twice_fss_dict_retrieval` | 금융 용어 검색 |
| 보완 | DART API 직접 수집 | 대규모 실제 공시 문서 |

### 총 예상 학습 데이터 규모
- **즉시 활용 가능**: 약 **800k 샘플** (HuggingFace 공개 데이터 합산)
- **추가 수집 시**: DART 공시 수십만 문서 추가 가능

---

*조사자: survey-finance 서브에이전트 | 모델: claude-sonnet-4-6 | 조사일: 2026-02-26*
