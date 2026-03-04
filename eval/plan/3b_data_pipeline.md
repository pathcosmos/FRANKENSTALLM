# 3B 사전학습 데이터 파이프라인

**작성일:** 2026-02-27
**프로젝트:** `/PROJECT/0325120031_A/ghong/taketimes/llm-bang/`

---

## 1. 현재 데이터 현황

### 토큰화 완료 (즉시 사용 가능)

| 파일 | 크기 | 토큰 수 |
|------|------|---------|
| korean_c4_train.bin | 15 GB | **7.56B** |
| korean_namuwiki_train.bin | 2.1 GB | **1.08B** |
| korean_wiki_train.bin | 500 MB | **262M** |
| korean_train.bin (위 3개 병합) | 17 GB | **9.0B** |
| train.bin (영어 기타) | 1.2 GB | **606M** |

**즉시 사용 가능: ~9.6B 토큰**

### 미토큰화 원본 (korean_extra/)

| 소스 | 디스크 크기 | 형식 | 예상 토큰 수 | 비고 |
|------|------------|------|-------------|------|
| CulturaX (ko) | 60 GB (32 parquet) | parquet, `text` 컬럼 | **~11.6B** | 고품질 웹 + mC4 중복제거 |
| cc100 (ko) | 14 GB xz → 54 GB text | xz 압축 텍스트 | **~13.5B** | 줄 단위 텍스트 |
| HPLT (en-ko) | 23 GB (193 parquet) | 병렬 코퍼스, `tgt_doc.sentences` | **~3.7B** | 한국어 측만 추출 필요 |
| OSCAR (ko) | 9.2 GB (27 parquet) | 중첩 구조, `text[].text` | **~2.0B** | 웹 텍스트 |
| korean_webtext | 4.2 GB (18 parquet) | parquet | **~1.5B** | 한국어 웹 |
| korean_textbooks | 6.4 GB | MMLU 스타일 parquet | **~0.5B** | 교과서/시험 (구조적) |
| finepdfs_edu_ko | 2.9 GB (parquet) | parquet | **~0.8B** | 교육 PDF |
| namuwiki_extracted | 2.2 GB | 텍스트 | **~0.6B** | 이미 namuwiki_train.bin과 중복 가능 |
| kovast | 449 MB | - | **~0.1B** | 소량 |
| korean_safe_conv | 51 MB | jsonl | **~15M** | 대화 데이터 |
| evol_instruct_ko | 144 MB | - | **~40M** | SFT용, pretrain 부적합 |

**미토큰화 총 예상: ~34B 토큰**
**전체 합계: ~43B 토큰 (중복 제거 전)**

---

## 2. 3B 모델 학습 목표

Chinchilla 스케일링 법칙에 따른 최적:
- **최소:** 3B × 20 = **60B 토큰**
- **최적:** 3B × 50 = **150B 토큰**
- **현실적 목표:** 가용 데이터 ~35B 토큰 (중복 제거 후) → **2~3 epoch 반복으로 60-100B 토큰**

### 데이터 믹싱 전략

| 카테고리 | 소스 | 비율 | 이유 |
|----------|------|------|------|
| 고품질 웹 | CulturaX | 35% | 이미 중복 제거된 고품질 |
| 대규모 웹 | cc100 + mC4(기존) | 35% | 양적 확보 |
| 백과사전 | 위키 + 나무위키 | 10% | 사실 지식 |
| 보조 웹 | OSCAR + korean_webtext + HPLT | 15% | 다양성 |
| 전문 도메인 | textbooks + finepdfs | 5% | 교육 품질 |

---

## 3. 데이터 품질 필터링 계획

### Phase 1: 기본 필터 (빠름, 1-2시간)
- **언어 필터:** `langdetect`로 한국어 비율 < 50% 문서 제거
  - HPLT: 병렬 코퍼스라 한국어 추출만 하면 됨
  - cc100: 이미 한국어지만 혼입 확인
- **길이 필터:** 50자 미만 문서 제거
- **중복 줄 제거:** 같은 줄 5회 이상 반복하는 문서 제거

### Phase 2: MinHash 중복 제거 (4-8시간)
```
도구: datasketch (pip install datasketch)
방법: 5-gram MinHash, 128 permutations
임계값: Jaccard > 0.8 → 중복
예상 제거율: 15-25% (특히 cc100 + CulturaX 간)
```
- CulturaX는 이미 내부 중복 제거됨 → 다른 소스와의 교차 중복만 체크
- 72코어 병렬 처리로 ~4시간 예상

### Phase 3: Perplexity 필터 (선택, 12-24시간)
- 현재 1B 모델로 각 문서 perplexity 계산
- 하위 5% (너무 쉬움 = 템플릿/반복) + 상위 5% (노이즈) 제거
- **권장: 3B 첫 학습 후 2차 학습 시 적용** (시간 절약)

---

## 4. 토큰화 파이프라인

### 우선순위 및 예상 시간 (72코어 기준)

| 순위 | 소스 | 예상 시간 | 이유 |
|------|------|----------|------|
| 1 | CulturaX | 3-4시간 | 60GB parquet, 가장 크고 고품질 |
| 2 | cc100 | 2-3시간 | xz 해제 30분 + 토큰화 2시간 |
| 3 | OSCAR | 1시간 | 9.2GB, 구조 파싱 필요 |
| 4 | korean_webtext | 30분 | 4.2GB |
| 5 | HPLT (Korean) | 1-2시간 | 한국어 추출 + 토큰화 |
| 6 | textbooks + finepdfs | 30분 | 소량 |

**총 예상: 8-12시간 (병렬 처리 시)**

---

## 5. 타임라인

### 최소 시작 방안 (즉시, 0시간)
- 기존 `korean_train.bin` (9B 토큰)으로 3B 학습 시작 가능
- 1 epoch에 부족하지만 학습 코드 검증 + 초기 학습에는 충분

### Phase A: 빠른 확장 (8-12시간)
1. CulturaX 토큰화 (3-4시간)
2. cc100 해제 + 토큰화 (2-3시간, CulturaX와 병렬 가능)
3. 기타 소스 토큰화 (2시간)
4. 병합 → ~35B 토큰

### Phase B: 품질 필터링 (추가 4-8시간)
1. 기본 필터 (1-2시간)
2. MinHash 중복 제거 (4-8시간)
3. 최종 병합 → ~28-30B 토큰 (깨끗한 데이터)

### Phase C: 학습
- 30B 토큰 × 2 epoch = 60B 토큰 (Chinchilla 최소)
- 또는 30B × 3 epoch = 90B 토큰 (안전)

---

## 6. 파일 구조 (최종)

```
data/
├── korean_train.bin         # 기존 9B (c4+wiki+namuwiki)
├── korean_val.bin
├── culturax_train.bin       # ~11.6B
├── culturax_val.bin
├── cc100_train.bin          # ~13.5B
├── cc100_val.bin
├── oscar_train.bin          # ~2B
├── oscar_val.bin
├── webtext_train.bin        # ~1.5B
├── webtext_val.bin
├── hplt_ko_train.bin        # ~3.7B
├── hplt_ko_val.bin
├── extra_train.bin          # textbooks + finepdfs + kovast
├── extra_val.bin
├── merged_3b_train.bin      # 전체 병합 (~35B)
└── merged_3b_val.bin
```

---

## 7. 요약

| 항목 | 값 |
|------|-----|
| 즉시 사용 가능 토큰 | **9.6B** |
| 추가 토큰화 후 | **~35B** (중복 제거 전) |
| 중복 제거 후 예상 | **~28-30B** |
| 3B Chinchilla 최소 | 60B (2 epoch) |
| 데이터 준비 최소 시간 | **0시간** (기존 데이터 사용) |
| 전체 파이프라인 완료 | **12-20시간** |
| 학습 시작 추천 | 기존 9B로 즉시 시작 + CulturaX 병렬 준비 |
