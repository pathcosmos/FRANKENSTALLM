# 한국어 공개 Pretrain 데이터셋 전수 조사

> 조사일: 2026-02-27
> HuggingFace API 실접근 확인 완료

---

## 1. 이미 보유 데이터셋

| 데이터셋 | 보유 크기 | 한국어 토큰 수 (추정) | 비고 |
|---|---|---|---|
| `uonlp/CulturaX` (ko) | 60GB | ~24.8B | mC4+OSCAR 정제본, GATED |
| `cc100` (ko) | 14GB | ~5.5B | Common Crawl 100 |
| `oscar-corpus/mOSCAR` (ko) | 9.2GB | ~3.5B | OSCAR multilingual |
| `HPLT/hplt_monolingual_v1_2` (ko) | 23GB | ~9B | Internet Archive 기반 |
| `HAERAE-HUB/KOREAN-WEBTEXT` | 보유 | ~1.5B | 고품질 한국어 웹텍스트 |
| `maywell/korean_textbooks` | 보유 | ~0.2B | 교과서 스타일 합성 데이터 |

**보유 합계: ~106GB+ / ~44.5B 토큰**

---

## 2. HuggingFace 접근 가능 - 추가 다운로드 필요

### 2-1. 대형 웹 코퍼스 (한국어 부분)

| 데이터셋 | 한국어 크기 (추정) | 토큰 수 (추정) | 접근성 | 우선도 |
|---|---|---|---|---|
| `mc4` (ko) | ~50GB | ~20B | ✅ 공개 | ⭐⭐⭐ |
| `allenai/c4` (ko multilingual) | ~15GB | ~6B | ✅ 공개 | ⭐⭐ |
| `HPLT/HPLT2.0_cleaned` (ko) | ~30GB | ~12B | ✅ 공개 | ⭐⭐⭐ |
| `PleIAs/common_corpus` (ko) | ~10-20GB | ~5-8B | ✅ 공개 | ⭐⭐⭐ |
| `minpeter/fineweb-2-edu-korean-raw` | ~20-30GB | ~8-12B | ✅ 공개 | ⭐⭐⭐⭐ |
| `minpeter/fineweb-2-edu-korean` | ~5-10GB | ~2-4B | ✅ 공개 (edu 필터링) | ⭐⭐⭐⭐ |
| `Viet-Mistral/CulturaY` (ko) | ~5GB | ~2B | ✅ 공개 | ⭐⭐ |
| `allenai/dolma` (ko 부분) | ~3-5GB | ~1-2B | ✅ 공개 | ⭐⭐ |

### 2-2. 한국어 전용 데이터셋

| 데이터셋 | 크기 (추정) | 토큰 수 (추정) | 접근성 | 비고 |
|---|---|---|---|---|
| `KORMo-Team/korean-web-collection` | ~50-80GB | ~20-30B | ✅ 공개, dl=2.7k | 한국어 웹 크롤, 가장 큰 한국어 전용 |
| `KORMo-Team/korean-public-corpus` | ~10-20GB | ~4-8B | ✅ 공개 | 공공 데이터 기반 |
| `eliceai/korean-webtext-edu` | ~2-5GB | ~1-2B | ✅ 공개 | 교육 품질 필터링 |
| `CocoRoF/cc-100-korean-processing` | ~14GB | ~5.5B | ✅ 공개 | cc100 한국어 처리본 |
| `MyeongHo0621/korean-quality-cleaned` | ~5-10GB | ~2-4B | ✅ 공개 | 품질 정제 |
| `opendatalab/WanJuan-Korean` | ~3-5GB | ~1-2B | ✅ 공개 | 중국 AI 연구소 제공 |

### 2-3. 위키/나무위키/백과

| 데이터셋 | 크기 | 토큰 수 (추정) | 접근성 |
|---|---|---|---|
| `wikimedia/wikipedia` (ko) | ~2GB | ~0.8B | ✅ 공개 |
| `lcw99/wikipedia-korean-20240501` | ~1.5GB | ~0.6B | ✅ 공개 |
| `heegyu/namuwiki-extracted` | ~5-8GB | ~2-3B | ✅ 공개 |
| `heegyu/namuwiki` | ~5-8GB | ~2-3B | ✅ 공개 |
| `seyoungsong/Open-Korean-Historical-Corpus` | ~1-2GB | ~0.3-0.5B | ✅ 공개 |

### 2-4. 법률/금융/도메인 특화

| 데이터셋 | 크기 | 토큰 수 (추정) | 접근성 |
|---|---|---|---|
| `smhilee/korean-law-dataset` | ~1-3GB | ~0.3-1B | ✅ 공개 |
| `joonhok-exo-ai/korean_law_open_data_precedents` | ~1-2GB | ~0.3-0.5B | ✅ 공개 |
| `Rootpye/korean-lawdata2` | ~0.5-1GB | ~0.2-0.3B | ✅ 공개 |
| `Rootpye/korean-lawdata4` | ~0.5-1GB | ~0.2-0.3B | ✅ 공개 |
| `ducut91/korean-constitutional-court-decisions` | ~0.5GB | ~0.1-0.2B | ✅ 공개 |

### 2-5. 코드 데이터 (다국어)

| 데이터셋 | 전체 크기 | 한국어 관련성 | 접근성 |
|---|---|---|---|
| `codeparrot/github-code` | ~1TB+ | 코드 자체 (언어 무관) | ✅ 공개 |
| `bigcode/the-stack-v2` | ~3TB+ | 코드 (한국어 주석 포함) | ✅ 공개 |

---

## 3. AI Hub / 국립국어원 / 정부 데이터 (HF 외부)

### 3-1. AI Hub (aihub.or.kr) - 회원가입+승인 필요

| 데이터셋 | 규모 (추정) | 비고 |
|---|---|---|
| 한국어 대화 데이터 | ~10-20GB | 일상대화, 목적대화 등 |
| 한국어 뉴스 기사 | ~30-50GB | 수백만 건 |
| 한국어 문서 요약 | ~5-10GB | 뉴스/문서 요약 쌍 |
| 한국어 기계독해 | ~3-5GB | QA 데이터 |
| 전문분야 한국어 | ~5-10GB | 의료/법률/금융/과학 |
| 한국어 SNS 데이터 | ~5-10GB | 소셜미디어 텍스트 |
| **AI Hub 합계** | **~60-100GB** | **승인 후 다운로드, 상업적 이용 제한 확인 필요** |

### 3-2. 국립국어원 모두의 말뭉치 (corpus.korean.go.kr)

| 데이터셋 | 규모 (추정) | 비고 |
|---|---|---|
| 문어 말뭉치 (신문, 잡지, 책) | ~15-20GB | 2020년대 기준 |
| 구어 말뭉치 (대화, 강연) | ~5-10GB | 전사 데이터 |
| 웹 말뭉치 | ~10-15GB | 웹 수집 텍스트 |
| 메신저 말뭉치 | ~1-2GB | 카카오톡 등 |
| 전문분야 말뭉치 | ~3-5GB | 법률/의학/과학 |
| **NIKL 합계** | **~35-50GB** | **비상업적 연구용, 신청 필요** |

### 3-3. 기타 정부/공공 데이터

| 소스 | 규모 | 비고 |
|---|---|---|
| 국가법령정보센터 (law.go.kr) | ~5-10GB | 법령/판례 전문 크롤 가능 |
| 한국학술지인용색인 (KCI) | ~3-5GB | 논문 초록 |
| 국회 회의록 | ~2-3GB | 공개 |
| 특허 데이터 (KIPRIS) | ~5-10GB | 한국어 특허 |

---

## 4. 접근 불가 / 확인 불가

| 데이터셋 | 상태 | 비고 |
|---|---|---|
| `snunlp/korean-hate-speech` | ❌ 404 | 삭제됨 |
| `Bingsu/KoCC` | ❌ 404 | 삭제됨 |
| `nindanaoto/ko-books` | ❌ 404 | 삭제됨 |
| `snunlp/KR-FinPen` | ❌ 404 | 삭제됨 |
| `bigscience/roots_ko_*` | ❌ 404 | BigScience 프로젝트 종료 |
| `open-llm-leaderboard/korean-fineweb` | ❌ 미확인 | 존재 여부 불명 |

---

## 5. 총 가용 토큰 수 추정

| 카테고리 | 토큰 수 (추정) |
|---|---|
| 이미 보유 | ~44.5B |
| HF 추가 다운로드 가능 (대형 웹) | ~55-75B |
| HF 추가 다운로드 가능 (한국어 전용) | ~30-50B |
| HF 추가 (위키/나무위키) | ~5-7B |
| HF 추가 (법률/도메인) | ~1-2B |
| AI Hub + NIKL (신청 필요) | ~35-55B |
| 기타 공공 데이터 (크롤 필요) | ~5-10B |
| **총 가용** | **~175-240B 토큰** |

> ⚠️ 중복 주의: CulturaX, mc4, HPLT, cc100 등은 Common Crawl 기반으로 상당 부분 중복됨.
> 중복 제거 후 유니크 토큰은 **~80-120B** 수준으로 추정.

---

## 6. 즉시 다운로드 권장 Top 5

| 순위 | 데이터셋 | 이유 |
|---|---|---|
| 🥇 1 | `KORMo-Team/korean-web-collection` | 한국어 전용 최대 규모, 기존 보유 데이터와 중복 적음 |
| 🥈 2 | `minpeter/fineweb-2-edu-korean-raw` | FineWeb2 기반 한국어 교육 품질, 최신 고품질 |
| 🥉 3 | `HPLT/HPLT2.0_cleaned` (ko) | v1.2 이미 보유, v2.0은 더 크고 정제됨 |
| 4 | `mc4` (ko) | CulturaX와 일부 중복이나 mC4 원본으로 추가 데이터 확보 가능 |
| 5 | `heegyu/namuwiki-extracted` + `wikimedia/wikipedia` (ko) | 백과사전 품질, 사실 정보 풍부 |

### 다운로드 명령 예시

```bash
# 1. KORMo korean-web-collection
huggingface-cli download KORMo-Team/korean-web-collection --repo-type dataset --local-dir ./data/korean-web-collection

# 2. FineWeb2 Korean
huggingface-cli download minpeter/fineweb-2-edu-korean-raw --repo-type dataset --local-dir ./data/fineweb2-korean

# 3. HPLT 2.0 Korean only
# (config 지정 필요 - ko subset)
python -c "from datasets import load_dataset; ds = load_dataset('HPLT/HPLT2.0_cleaned', 'ko', split='train'); ds.save_to_disk('./data/hplt2-ko')"

# 4. mC4 Korean
python -c "from datasets import load_dataset; ds = load_dataset('mc4', 'ko', split='train'); ds.save_to_disk('./data/mc4-ko')"

# 5. 나무위키 + 위키피디아
huggingface-cli download heegyu/namuwiki-extracted --repo-type dataset --local-dir ./data/namuwiki
python -c "from datasets import load_dataset; ds = load_dataset('wikimedia/wikipedia', '20231101.ko', split='train'); ds.save_to_disk('./data/wiki-ko')"
```

---

## 7. 참고사항

- **중복 처리 필수**: 대부분의 대형 웹 코퍼스(CulturaX, mc4, cc100, OSCAR, HPLT)는 Common Crawl이 원천이므로 MinHash 등으로 dedup 필요
- **품질 필터링**: FineWeb2-edu-korean은 교육 품질 스코어로 필터링되어 있어 pretrain 품질이 높음
- **라이선스 확인**: AI Hub/NIKL 데이터는 상업적 이용 제한이 있을 수 있음. 사전 확인 필요
- **코드 데이터**: 한국어 LLM이라도 코드 능력을 위해 `the-stack-v2` 또는 `github-code`에서 Python/JS/etc 포함 권장 (별도 50-100B 토큰)
