# 한국어 의료/의학/헬스케어 데이터셋 전수 조사

> 작성일: 2026-02-27  
> 목적: 한국어 LLM 3B 모델 학습용 공개 의료 데이터 전수 조사  
> 조사 소스: HuggingFace Hub, AI-Hub, HIRA, NHIS, 공공데이터포털, GitHub

---

## 전체 목록 테이블

| # | 데이터셋 ID / 소스 | 크기 | 라이선스 | 내용 분류 | 다운로드 방법 | 제한사항 | 우선순위 |
|---|------------------|------|---------|---------|------------|---------|---------|
| 1 | `sean0042/KorMedMCQA` (HF) | 7,469 문제 | CC BY-NC 2.0 | 한국 의료면허시험 MCQ (의사/간호사/약사/치과) | `datasets.load_dataset` | 비상업 | **9** |
| 2 | `ChuGyouk/medical-o1-reasoning-SFT-Ko` (HF) | 25,700 행 | Apache 2.0 | 의학 추론 SFT (한국어 번역, CoT 포함) | `datasets.load_dataset` | 없음 | **9** |
| 3 | `HAERAE-HUB/KMMLU` (HF) | 35,030 문제 (의학 서브셋 포함) | CC BY-ND 4.0 | 45개 분야 전문가 MCQ (의학 다수 포함) | `datasets.load_dataset` | 변경불가 | **8** |
| 4 | `squarelike/ko_medical_chat` (HF) | 3,040 대화 | 없음(오픈) | 한국어 의사-환자 대화 (ChatDoctor 기반 번역) | `datasets.load_dataset` | 없음 | **8** |
| 5 | `ChuGyouk/medical-reasoning-train-kormedmcqa` (HF) | ~5,000 행 | CC BY-NC | KorMedMCQA 기반 Gemini 추론 학습 데이터 | `datasets.load_dataset` | 비상업 | **8** |
| 6 | `ih9511/medical-translation-en-ko` (HF) | 1M~10M 행 | 오픈 | 의학 논문/특허 EN↔KO 번역 (한국학술정보 기반) | `datasets.load_dataset` | 없음 | **7** |
| 7 | `GrowingApple/orpo_kor_translated_medical` (HF) | 10K~100K 행 | 없음 | 한국어 의료 ORPO 학습 데이터 (번역) | `datasets.load_dataset` | 없음 | **7** |
| 8 | `ChuGyouk/medical_questions_pairs_ko` (HF) | ~5,000 쌍 | unknown | 의료 질문 유사도 쌍 한국어 번역 | `datasets.load_dataset` | 불명확 | **6** |
| 9 | `ChuGyouk/MMMLU-Ko-Medical` (HF) | 1K~10K | MIT | MMMLU 한국어 의료 서브셋 (clinical/genetics/anatomy 등) | `datasets.load_dataset` | 없음 | **6** |
| 10 | `seongsubae/KorMedMCQA-V` (HF) | 1,534 문제 + 2,043 이미지 | CC BY-NC-SA 4.0 | 한국 의료면허시험 + 의료 이미지 (멀티모달) | `datasets.load_dataset` | 비상업 | **6** |
| 11 | `helenko/medical_DPO_dataset_ko` (HF) | 1K~10K | 없음 | 의료 DPO 학습 데이터 한국어 | `datasets.load_dataset` | 없음 | **5** |
| 12 | `hjkimsun/medical-dpo-ko` (HF) | 1K~10K | 없음 | 의료 DPO 데이터 한국어 | `datasets.load_dataset` | 없음 | **5** |
| 13 | `Saxo/ko_medical_meadow_med_qa_options_...` (HF) | 10K~100K | Apache 2.0 | 한국어 MedQA 옵션 데이터 | `datasets.load_dataset` | 없음 | **5** |
| 14 | `Nexdata/203_Hours_Korean_Medical...` (HF) | 203시간 음성 (샘플) | CC BY-ND 4.0 | 한국어 의료 엔티티 음성/전사 (샘플, 전체 유료) | 샘플만 무료 | 유료 전체 | **3** |
| 15 | `LGAI-EXAONE/KMMLU-Redux` (HF) | 2,587 문제 | CC BY-NC-ND 4.0 | KMMLU 재구성 (오류 제거, 의학 포함) | gated(승인 필요) | 비상업+변경불가 | **6** |
| 16 | `LGAI-EXAONE/KMMLU-Pro` (HF) | 2,822 문제 | CC BY-NC-ND 4.0 | 한국 전문직 면허 시험 (의사 포함) | gated(승인 필요) | 비상업+변경불가 | **7** |
| 17 | AI-Hub 헬스케어 카테고리 전체 | **126개 데이터셋** | 공공누리/연구전용 | 의료 영상/임상/건강검진/의학 NLP 등 | 안심존+IRB 필수 | **IRB 심의 필수** | **8** (접근 어려움) |
| 18 | HIRA 공개 데이터 (opendata.hira.or.kr) | 수십~수백만 건 | 공공누리 1유형 | 의료장비현황, 병의원현황, 건강보험 진료통계 등 | 직접 다운로드 | 없음 (통계 위주) | **3** |
| 19 | NHIS 공개 데이터 (nhis.or.kr) | 수십만 건 | 공공누리 | 지역별 의료이용통계, 진료실적 현황 등 | 직접 다운로드 | 없음 (통계 위주) | **3** |
| 20 | 공공데이터포털 의료 관련 (data.go.kr) | 4,406건 파일/API | 공공누리 | 전국의료기관현황, 응급의료기관, 의료영상정보 등 | 직접 다운로드/API | 없음 (구조 데이터) | **4** |
| 21 | KoreaMed (synapse.koreamed.org) | 수십만 편 논문 초록 | 개별 저작권 | 한국 의학 저널 논문 초록 (영문/한문 혼재) | 웹 스크래핑 | 저작권 주의 | **5** |
| 22 | PubMed 한국어 초록 | 수만 건 | PubMed OA | 한국어로 작성된 PubMed 초록 | PubMed API/NCBI FTP | 제한 없음 | **5** |

---

## 소스별 상세 분석

### 1. HuggingFace Hub

HuggingFace API (`/api/datasets?search=...`) 및 직접 URL 조회 결과, 한국어 의료 데이터셋은 **주로 번역 기반이거나 벤치마크 목적**의 소규모 데이터가 대부분이다.

**주요 특징:**
- 원시(native) 한국어 의료 데이터는 매우 드물다
- 대부분 영어 의료 데이터(ChatDoctor, MedQA, HuatuoGPT 등)를 한국어로 번역한 것
- 한국 의료면허시험 기반의 벤치마크(KorMedMCQA, KMMLU)가 가장 퀄리티가 높음

**수집 기준:**
- `ko_medical`, `medical korean`, `medical ko`, `KorMed`, `KMMLU` 등 검색어 사용
- 총 20+ 쿼리 조회

### 2. AI-Hub (aihub.or.kr)

**헬스케어 카테고리: 총 126개 데이터셋** 보유 (2026-02-27 기준)

- 대부분 의료 영상(MRI, CT, 병리 이미지) 데이터
- NLP/텍스트 관련 데이터도 존재하나 **"안심존(Safe Zone)"** 접근 필수
- 안심존: 인터넷 분리 환경에서만 분석 가능, 데이터 반출 불가
- **IRB 심의 결과 통지서 + 승인된 연구계획서 필수**
- 의료 데이터 특성상 직접 다운로드 불가 (개인정보 비식별화에도 불구)

**접근 프로세스:**
1. 기관생명윤리위원회(IRB) 심의 → 결과 통지서 획득
2. 안심존 이용 신청서 + 보안서약서 제출
3. 구축기관 심사 및 승인
4. 온라인/오프라인 안심존에서 데이터 분석
5. 분석 모델만 반출 가능 (데이터 반출 불가)

**문의:** safezone1@aihub.kr / 02-525-7708

### 3. HIRA 공개 데이터 (opendata.hira.or.kr)

**공공누리 1유형 (자유 이용 가능)**

주요 데이터:
- 의료장비 상세 현황 (2019~2024, CSV/XLSX)
- 전국 병의원 및 약국 현황
- 3단상병별 성별 연령군별 건강보험 진료 통계
- 요양기관별 건강보험 청구 통계

**NLP 활용 가능성: 낮음** — 통계/구조적 데이터로 직접 LLM 학습에는 부적합

### 4. NHIS 공개 데이터

- 지역별 의료이용통계 (XLSX)
- 의료보장(건강보험+의료급여) 시도별 진료실적 현황

**NLP 활용 가능성: 낮음** — 수치 통계 위주

### 5. 공공데이터포털 (data.go.kr)

의료 관련 4,406건 검색 결과:
- 전국의료기관표준데이터 (CSV)
- 전국응급의료기관표준데이터 (XML)
- 전국보건기관표준데이터 (CSV/XML/JSON)
- 의료영상정보 (국가중점데이터)
- 임상연구정보 (국가중점데이터)
- 해부학 및 의료행위 기록설명그림 정보

**NLP 활용 가능성: 중간** — 임상연구정보, 해부학/의료행위 기록 등은 활용 가능

---

## Top 3 상세 분석

---

### 🥇 1위: `sean0042/KorMedMCQA`

**우선순위: 9/10**

| 항목 | 내용 |
|------|------|
| **HuggingFace ID** | `sean0042/KorMedMCQA` |
| **URL** | https://huggingface.co/datasets/sean0042/KorMedMCQA |
| **논문** | https://arxiv.org/abs/2403.01469 |
| **크기** | 7,469 문제 (train 5,902 / dev 755 / test 812) |
| **형식** | Parquet |
| **라이선스** | CC BY-NC 2.0 |
| **HF 다운로드수** | 1,301 (2026-02 기준) |
| **언어** | 한국어 (native) |

**내용:**
- **출처**: 2012~2024년 한국 보건의료 전문면허 시험 실제 문제
- **카테고리**: 의사(Doctor), 간호사(Nurse), 약사(Pharmacist), 치과의사(Dentist)
- **형식**: 4지선다 MCQ (보기 A/B/C/D + 정답)
- **의학 분야**: 내과, 외과, 소아과, 산부인과, 약리학, 병리학, 해부학 등 전 분야

**IRB/비식별화 여부:**
- 원본 데이터가 공개 국가시험 문제이므로 개인정보 없음
- IRB 불필요

**다운로드:**
```python
from datasets import load_dataset
ds = load_dataset("sean0042/KorMedMCQA")
# 서브셋: "doctor", "nurse", "pharmacist", "dentist"
ds = load_dataset("sean0042/KorMedMCQA", "doctor")
```

**장점:**
- 한국어 native 의료 데이터 (번역 아님)
- 실제 국가시험 문제 → 의료 도메인 신뢰도 최고
- 의사/간호사/약사/치과의사 4개 직종 커버
- 벤치마크 + 학습 데이터 모두 활용 가능

**제한사항:**
- 비상업 라이선스 (CC BY-NC)
- 이미지 포함 문제는 텍스트만 제공 (이미지 버전은 KorMedMCQA-V 참조)
- 총 7,469문제 (규모 작음)

**활용 방법:**
1. SFT 학습 데이터로 직접 활용
2. Few-shot 예시로 활용
3. 의료 도메인 평가 벤치마크로 활용
4. 추론 데이터 생성의 seed 데이터로 활용

---

### 🥈 2위: `ChuGyouk/medical-o1-reasoning-SFT-Ko`

**우선순위: 9/10**

| 항목 | 내용 |
|------|------|
| **HuggingFace ID** | `ChuGyouk/medical-o1-reasoning-SFT-Ko` |
| **URL** | https://huggingface.co/datasets/ChuGyouk/medical-o1-reasoning-SFT-Ko |
| **크기** | 25,700 행 |
| **형식** | Parquet |
| **라이선스** | Apache 2.0 |
| **HF 다운로드수** | 40 (2026-02 기준) |
| **언어** | 한국어 (번역) |

**내용:**
- **출처**: HuatuoGPT-o1 학습 데이터를 한국어로 번역
- **원본**: GPT-4o가 검증 가능한 의학 문제를 탐색하고 의학 검증자(medical verifier)로 검증
- **번역**: `gemini-2.0-flash-exp` (temperature=0.5)로 번역
- **컬럼**: `Question`, `Complex_Cot`, `Response`
- **특징**: Complex Chain-of-Thought (CoT) 추론 과정 포함

**CoT 구조 예시:**
```
Question: 자신의 음경이 줄어들고 결국 사라져 죽음에 이를 것이라고 믿는 사람의 진단은?
Complex_Cot: [300~3,420 토큰 분량의 한국어 추론 과정]
Response: [최종 답변]
```

**IRB/비식별화 여부:**
- 번역 데이터로 개인정보 없음
- IRB 불필요

**다운로드:**
```python
from datasets import load_dataset
ds = load_dataset("ChuGyouk/medical-o1-reasoning-SFT-Ko")
```

**장점:**
- Apache 2.0 (상업 이용 가능)
- 의학 추론(reasoning) CoT 포함 → 3B 모델 추론력 강화에 최적
- 25K+ 샘플 (KorMedMCQA 대비 규모 큼)
- 오류 검증 과정을 거친 고품질 데이터

**제한사항:**
- 번역 데이터 (원본 영어) → 한국어 의료 표현의 자연스러움 한계 있음
- 번역 오류 가능성 (Gemini 번역)
- 수학/과학 문제 일부 포함 (순수 의료만은 아님)

**활용 방법:**
1. 한국어 의료 추론 SFT 학습 (주력 학습 데이터)
2. CoT 형식으로 의료 응답 품질 향상
3. KorMedMCQA와 결합하여 학습 효과 극대화

---

### 🥉 3위: `HAERAE-HUB/KMMLU` (의학 서브셋)

**우선순위: 8/10**

| 항목 | 내용 |
|------|------|
| **HuggingFace ID** | `HAERAE-HUB/KMMLU` |
| **URL** | https://huggingface.co/datasets/HAERAE-HUB/KMMLU |
| **논문** | https://arxiv.org/abs/2402.11548 |
| **크기** | 35,030 문제 전체 (의학 서브셋은 ~수천) |
| **형식** | CSV |
| **라이선스** | CC BY-ND 4.0 |
| **HF 다운로드수** | 10,537 (2026-02 기준) |
| **언어** | 한국어 (native) |

**내용:**
- **출처**: 한국 국가기술자격시험 실제 문제 (2023~2024)
- **45개 분야**: 회계, 법률, **의학**, **약학**, **간호학** 등
- **의학 관련 서브셋**: `clinical_knowledge`, `medical_genetics`, `anatomy`, `professional_medicine`, `college_biology`, `college_medicine` 등
- **형식**: 4지선다 MCQ + 인간 정확도(Human Accuracy) 제공

**의학 서브셋 접근:**
```python
from datasets import load_dataset
# 의학 관련 서브셋들
medical_subsets = [
    "Clinical-Psychology",
    "Emergency-Medicine", 
    "Health-Insurance-Review",
    "Medical-Examination",
    "Public-Health"
]
for subset in medical_subsets:
    ds = load_dataset("HAERAE-HUB/KMMLU", subset)
```

**IRB/비식별화 여부:**
- 공개 국가시험 문제 → 개인정보 없음
- IRB 불필요

**장점:**
- 가장 높은 다운로드수 (10,537) → 검증된 데이터
- 한국어 native (번역 아님)
- 인간 정확도 레이블 제공 → 문제 난이도 파악 가능
- 45개 서브셋으로 세분화 → 의학 서브셋만 선택 가능
- KMMLU-HARD, KMMLU-Redux, KMMLU-Pro 등 다양한 변형 존재

**제한사항:**
- CC BY-ND 4.0 (변경 불가, 2차 저작물 금지)
- 의학 서브셋이 전체 데이터 일부 (~20%)
- 벤치마크 목적 → 학습 데이터로 전용 시 품질 검토 필요

**활용 방법:**
1. 한국어 의료 도메인 벤치마크 평가 (주 활용)
2. 의학 서브셋만 추출하여 학습 보조 데이터로 활용
3. KMMLU-Pro (전문직 면허 포함) 와 병합하여 확장

---

## 추가 권장 데이터셋

### AI-Hub 헬스케어 (접근 가능한 경우)

접근 방법이 어렵지만 가장 고품질의 한국어 원본 의료 데이터:
- **URL**: https://aihub.or.kr/aihubdata/data/list.do?currMenu=115&topMenu=100&srchDataRealmCode=REALM0014
- **총 126개** 헬스케어 데이터셋
- **IRB 필수**: 기관생명윤리위원회 승인 필요
- **안심존**: 데이터 반출 불가, 현장 분석만 가능
- **주요 NLP 관련 예상 데이터**: 진료 대화, 의무기록, 건강 상담, 의약품 정보

### KMMLU-Pro (LGAI-EXAONE)
- **URL**: https://huggingface.co/datasets/LGAI-EXAONE/KMMLU-Pro
- **크기**: 2,822 문제 (한국 전문직 면허 시험)
- **특징**: 의사 등 전문직 면허 포함, Gated (승인 필요)

### KorMedMCQA-V (멀티모달)
- **URL**: https://huggingface.co/datasets/seongsubae/KorMedMCQA-V
- **크기**: 1,534 문제 + 2,043 이미지
- **활용**: 비전-언어 모델 학습 시 참조

---

## 실용 가이드: 3B 모델 학습을 위한 전략

### Phase 1: 즉시 사용 가능 (IRB 불필요)

```bash
# 1. KorMedMCQA - 한국 의료면허 실제 시험 (benchmark + SFT 모두)
pip install datasets
python -c "from datasets import load_dataset; ds = load_dataset('sean0042/KorMedMCQA', 'doctor'); print(ds)"

# 2. medical-o1-reasoning-SFT-Ko - CoT 추론 학습 데이터
python -c "from datasets import load_dataset; ds = load_dataset('ChuGyouk/medical-o1-reasoning-SFT-Ko'); print(ds)"

# 3. KMMLU 의학 서브셋
python -c "from datasets import load_dataset; ds = load_dataset('HAERAE-HUB/KMMLU', 'Medical-Examination'); print(ds)"

# 4. ko_medical_chat - 대화 형식 SFT
python -c "from datasets import load_dataset; ds = load_dataset('squarelike/ko_medical_chat'); print(ds)"

# 5. medical-translation-en-ko - 대용량 번역 corpus
python -c "from datasets import load_dataset; ds = load_dataset('ih9511/medical-translation-en-ko'); print(ds)"
```

### Phase 2: 접근 신청 필요

| 데이터셋 | 신청 방법 | 예상 소요 시간 |
|---------|---------|------------|
| AI-Hub 헬스케어 | IRB + 안심존 신청 | 4~8주 |
| KMMLU-Redux/Pro | HF Gated 승인 신청 | 수일 |

### 학습 데이터 조합 추천

**규모별 추천 조합:**

| 규모 | 조합 | 예상 총 샘플 |
|------|-----|------------|
| 소규모 | KorMedMCQA + medical-o1-reasoning-SFT-Ko | ~33K |
| 중규모 | 위 + ko_medical_chat + KMMLU 의학 서브셋 + medical_questions_pairs_ko | ~45K |
| 대규모 | 위 + medical-translation-en-ko (필터링) + orpo_kor_translated_medical | ~100K+ |

---

## 주요 고려사항

### 라이선스 분류

| 라이선스 | 데이터셋 | 상업 활용 | 변경 가능 |
|---------|---------|---------|---------|
| Apache 2.0 | medical-o1-reasoning-SFT-Ko | ✅ | ✅ |
| MIT | MMMLU-Ko-Medical | ✅ | ✅ |
| CC BY-ND 4.0 | KMMLU, KorMedMCQA-V(음성) | ✅ | ❌ |
| CC BY-NC 2.0 | KorMedMCQA | ❌ | ✅ |
| CC BY-NC-SA 4.0 | KorMedMCQA-V | ❌ | ✅ |
| CC BY-NC-ND 4.0 | KMMLU-Redux, KMMLU-Pro | ❌ | ❌ |
| 공공누리 1유형 | HIRA, NHIS 통계 | ✅ | ✅ |

### 의료 데이터 특수 고려사항

1. **비식별화 여부**: HuggingFace의 한국어 의료 데이터는 대부분 번역 데이터 or 공개 시험문제 → 비식별화 이슈 없음
2. **IRB**: AI-Hub 헬스케어 데이터만 IRB 필수 (실제 진료 기록 포함)
3. **의료 환각(Hallucination)**: 번역 데이터의 경우 의료 용어 오역 가능 → 검증 필요
4. **진료 가이드라인 최신성**: 시험 문제 기반 데이터는 연도별 의료 가이드라인 변경 반영 필요

---

## 참고 링크

- KorMedMCQA: https://arxiv.org/abs/2403.01469
- KMMLU: https://arxiv.org/abs/2402.11548  
- KMMLU-Pro: https://arxiv.org/abs/2507.08924
- AI-Hub 안심존: https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSn=216
- HIRA 공개데이터: https://opendata.hira.or.kr
- NHIS 연구데이터: https://nhis.or.kr
- 공공데이터포털 의료: https://www.data.go.kr/tcs/dss/selectDataSetList.do?keyword=의료
- KoreaMed (한국의학저널): https://synapse.koreamed.org
