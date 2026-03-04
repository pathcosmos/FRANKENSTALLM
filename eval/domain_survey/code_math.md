# 코드 / 수학 / 과학 데이터셋 전수 조사

> **목적**: 한국어 LLM 3B 모델 학습용 코딩·수학·과학 데이터셋 전수 조사  
> **작성일**: 2026-02-27  
> **조사 범위**: HuggingFace Hub, bigcode, AI-Hub 등

---

## 1. 코드 데이터셋

### 1.1 전체 목록 테이블

| # | 데이터셋 | 규모 | 언어 | 한국어 주석 | 라이선스 | 형태 | 추천도 |
|---|---------|------|------|------------|---------|------|--------|
| 1 | **bigcode/the-stack-v2-dedup** | 32.1TB / ~900B tok | 600+ 언어 | 일부 포함 (필터 필요) | 혼합 (permissive only) | raw code | ★★★★★ |
| 2 | **bigcode/starcoderdata** | 783GB / ~250B tok | 86 언어 | 일부 포함 | 혼합 (permissive) | clean code+docs | ★★★★☆ |
| 3 | **nayohan/Evol-Instruct-Code-80k-v1-ko** | 78.3k samples | 한국어+코드 | ✅ 한국어 질문 | 미상 (GPT-4 번역) | instruction-output | ★★★★☆ |
| 4 | **nickrosh/Evol-Instruct-Code-80k-v1** | 78.3k samples | 영어+코드 | ❌ | MIT | instruction-output | ★★★☆☆ |
| 5 | **CodeResearch/Code-Evol-Instruct-OSS** | 4.31k samples | 영어+코드 | ❌ | 오픈소스 | instruction-output | ★★☆☆☆ |
| 6 | **bigcode/the-stack-v2** | 67.5TB full | 600+ 언어 | 일부 포함 | 혼합 | raw code (SWHID) | ★★★★☆ |

---

### 1.2 Top 3 상세 분석

---

#### 🥇 1위: `bigcode/the-stack-v2-dedup`

| 항목 | 내용 |
|------|------|
| **HuggingFace URL** | https://huggingface.co/datasets/bigcode/the-stack-v2-dedup |
| **전체 크기** | Full: 67.5TB / **Dedup: 32.1TB** / Train tokens: ~900B |
| **파일 수** | 3.28B unique files, 104.2M GitHub repositories |
| **언어 수** | 658개 프로그래밍/마크업 언어 |
| **수집 기간** | GitHub 2023-09-06 기준 |
| **근중 언어** | Python, JavaScript, TypeScript, Java, C++, C#, Go, Rust 등 |
| **한국어 주석 비율** | 직접 측정 없음. GitHub 한국어 레포 기준 추정 ~1-3% |
| **라이선스 구조** | permissive 라이선스만 포함 (MIT, Apache-2.0, BSD 등), 파일별 provenance 제공 |
| **접근 방법** | SoftwareHeritage+INRIA 동의 필요 (AWS S3 bulk download) |
| **전처리 수준** | Near-dedup 완료, PII 제거 필요, 언어별 필터링 가능 |
| **주요 메타데이터** | repo_name, detected_licenses, star/fork count, language, is_vendor, length_bytes |
| **특이사항** | 실제 파일 콘텐츠는 SWH S3에서 별도 다운로드 필요 |

**추천 이유**:  
- 최대 규모의 오픈소스 코드 데이터셋  
- permissive 라이선스만 포함해 법적 리스크 낮음  
- 언어별 서브셋 로드 가능 (`load_dataset("bigcode/the-stack-v2-dedup", "Python")`)  
- StarCoder2 학습 베이스 데이터

**한국어 LLM 활용 전략**:  
```python
# Python 서브셋만 로드
ds = load_dataset("bigcode/the-stack-v2-dedup", "Python", split="train")
# 한국어 주석 포함 파일 필터링 (heuristic)
korean_ds = ds.filter(lambda x: any(ord(c) > 0xAC00 for c in x.get("content", "")))
```

---

#### 🥈 2위: `bigcode/starcoderdata`

| 항목 | 내용 |
|------|------|
| **HuggingFace URL** | https://huggingface.co/datasets/bigcode/starcoderdata |
| **전체 크기** | **783GB / ~250B tokens** |
| **언어 수** | 86개 프로그래밍 언어 |
| **추가 데이터** | GitHub Issues (54GB), Jupyter Notebooks (13GB), GitHub Commits (32GB) |
| **한국어 주석 비율** | 직접 통계 없음. GitHub 한국 개발자 레포 포함 |
| **라이선스** | 원본 레포 라이선스 준수, Terms 동의 필요 |
| **전처리 수준** | **이미 dedup + clean + PII 제거 완료** |
| **Downloads** | 15,556/월 (인기 데이터셋) |
| **사용 모델** | StarCoder, StarCoderBase 학습 데이터 |

**추천 이유**:  
- The Stack v2보다 작지만 **이미 정제된 상태** (바로 학습 가능)  
- GitHub Issues/Jupyter/Commits 포함으로 다양한 코드 컨텍스트  
- StarCoder 논문에서 검증된 품질

**활용법**:  
```python
# Python만 로드
ds = load_dataset("bigcode/starcoderdata", data_dir="python", split="train")
# jupyter notebooks
ds = load_dataset("bigcode/starcoderdata", data_dir="jupyter-scripts-dedup-filtered")
```

---

#### 🥉 3위: `nayohan/Evol-Instruct-Code-80k-v1-ko`

| 항목 | 내용 |
|------|------|
| **HuggingFace URL** | https://huggingface.co/datasets/nayohan/Evol-Instruct-Code-80k-v1-ko |
| **샘플 수** | **78,326개** |
| **형태** | instruction-output 페어 (SFT용) |
| **한국어** | ✅ 질문(instruction)이 한국어로 번역됨 |
| **코드 언어** | Python 중심, 알고리즘/자료구조/코딩문제 |
| **원본** | nickrosh/Evol-Instruct-Code-80k-v1 (GPT-4 번역) |
| **라이선스** | 미명시 (GPT-4 output 포함 주의) |
| **Downloads** | 23/월 |
| **전처리** | 번역 품질 일부 이슈 (기계번역 오류 존재) |

**추천 이유**:  
- **즉시 SFT에 활용 가능한 한국어 코딩 instruction 데이터**  
- 78k 규모로 파인튜닝용으로 충분  
- instruction이 한국어로 됨 → 한국어 질문에 코드 응답하는 능력 학습

**주의사항**:  
- GPT-4 번역 기반 → 라이선스 불명확 (상업 사용 주의)  
- 번역 품질 검토 후 필터링 권장  
- 일부 instruction이 어색한 한국어

---

### 1.3 코드 데이터 수집 전략 요약

```
Pretrain용:
  우선순위 1: bigcode/starcoderdata (Python, JavaScript, etc.) → 즉시 사용 가능
  우선순위 2: bigcode/the-stack-v2-dedup (필요 언어 서브셋) → 규모 확대 시

SFT용:
  우선순위 1: nayohan/Evol-Instruct-Code-80k-v1-ko → 한국어 코딩 Q&A
  우선순위 2: nickrosh/Evol-Instruct-Code-80k-v1 (영어) → 번역 또는 직접 사용

한국어 주석 코드 추출:
  the-stack-v2-dedup에서 한글 포함 파일 필터링 (regex: [\uAC00-\uD7A3])
  → 한국 개발자가 작성한 코드 추출 가능
```

---

## 2. 수학 데이터셋

### 2.1 전체 목록 테이블

| # | 데이터셋 | 규모 | 언어 | 난이도 | 풀이과정 | 라이선스 | 추천도 |
|---|---------|------|------|--------|---------|---------|--------|
| 1 | **kuotient/orca-math-word-problems-193k-korean** | 193k | 한국어+영어 | 초등~중학 | ✅ | 미상 | ★★★★★ |
| 2 | **re2panda/grade_school_math_korean** | 7.47k | 한국어 | 초등~중학 | ✅ | MIT | ★★★★☆ |
| 3 | **openai/gsm8k** | 8.5k | 영어 | 초등~중학 | ✅ (CoT) | MIT | ★★★★☆ |
| 4 | **open-web-math/open-web-math** | 6.3B tok | 영어 | 전 난이도 | ❌ (raw) | ODC-By | ★★★☆☆ |
| 5 | **hendrycks/math** | 12.5k | 영어 | 고등~대학 | ✅ | MIT | ★★★☆☆ |
| 6 | **Quadyun/Korean_SAT_MATH** | 120 | 한국어 | 수능 수준 | 일부 | 미상 | ★★☆☆☆ |
| 7 | **kuotient/orca-math-korean-dpo-pairs** | 193k | 한국어 | 초등~중학 | ✅ (DPO) | 미상 | ★★★★☆ |

---

### 2.2 Top 3 상세 분석

---

#### 🥇 1위: `kuotient/orca-math-word-problems-193k-korean`

| 항목 | 내용 |
|------|------|
| **HuggingFace URL** | https://huggingface.co/datasets/kuotient/orca-math-word-problems-193k-korean |
| **샘플 수** | **193,264개** |
| **언어** | 한국어 + 영어 (이중 언어) |
| **난이도** | 초등~중학교 수준 수학 문장제 |
| **문제 유형** | 수 계산, 비율, 나이 문제, 기하, 확률, 방정식 등 |
| **풀이 과정** | ✅ 상세 단계별 풀이 포함 |
| **형태** | 문제(한국어) + 풀이(한국어) + 문제(영어) + 풀이(영어) |
| **원본** | Microsoft Orca-Math (Synthetic data) |
| **Downloads** | 396/월 |

**데이터 예시**:  
```
문제: 정국이 5위입니다. 정국보다 결승선을 먼저 통과한 사람의 수를 찾아보세요.
풀이: 정국이 5위라면 4명이 정국보다 먼저 결승선을 통과한 셈입니다.

문제: 숫자를 10으로 나눈 값은 6입니다. 윤기는 특정 숫자로부터 15를 빼서 결과를 얻었습니다.
풀이: x / 10 = 6 → x = 60 → 결과 = 60 - 15 = 45
```

**추천 이유**:  
- **가장 큰 한국어 수학 데이터셋** (193k)  
- 이중언어로 한국어-영어 수학 추론 능력 동시 학습  
- 단계별 풀이로 Chain-of-Thought 학습에 최적  
- BTS 멤버 이름 사용 (한국 문화 맥락 자연스럽게 포함)

---

#### 🥈 2위: `kuotient/orca-math-korean-dpo-pairs`

| 항목 | 내용 |
|------|------|
| **HuggingFace URL** | https://huggingface.co/datasets/kuotient/orca-math-korean-dpo-pairs |
| **샘플 수** | 193k DPO pairs |
| **언어** | 한국어 |
| **형태** | chosen / rejected 쌍 (DPO 학습용) |
| **활용** | RLHF/DPO 단계에서 수학 추론 품질 향상 |

**추천 이유**:  
- 위 193k와 세트로 사용 가능  
- DPO 방식으로 수학 답변 품질 향상

---

#### 🥉 3위: `openai/gsm8k`

| 항목 | 내용 |
|------|------|
| **HuggingFace URL** | https://huggingface.co/datasets/openai/gsm8k |
| **샘플 수** | **8,500개** (train: 7,473 / test: 1,319) |
| **언어** | 영어 |
| **난이도** | 초등~중학교 (8.5세~12세 수준) |
| **문제 유형** | 수학 문장제 (1~8단계 추론) |
| **풀이 과정** | ✅ CoT 단계별 풀이 + 최종 답 |
| **라이선스** | MIT |
| **Downloads** | 매우 높음 (표준 벤치마크) |

**특징**:  
- `main` split: 자연어 CoT 풀이  
- `socratic` split: 서브문제 분해 방식  
- 표준 LLM 수학 벤치마크로 re2panda/grade_school_math_korean이 이를 한국어로 번역

---

### 2.3 수학 데이터 추가 후보

| 데이터셋 | 규모 | 특징 |
|---------|------|------|
| `Quadyun/Korean_SAT_MATH` | 120문제 | 한국 수능 수학, 소규모지만 고품질 |
| `open-web-math/open-web-math` | 6.3B tok | 웹 수학 raw 텍스트, 영어, pretrain용 |
| `hendrycks/math` (MATH) | 12.5k | 경시대회 수준 수학, 영어, 고난이도 |

---

## 3. 과학 데이터셋

### 3.1 전체 목록 테이블

| # | 데이터셋 | 규모 | 언어 | 분야 | 난이도 | 라이선스 | 추천도 |
|---|---------|------|------|------|--------|---------|--------|
| 1 | **amphora/korean_science_papers** | 17k papers | 한국어 | 생명/화학/의학/식품 | 대학원 | 공개 (학술지) | ★★★★★ |
| 2 | **hiteshpatel945/korean-stem** | 316k | 한국어 | STEM 전반 | 다양 | 미상 | ★★★☆☆ |
| 3 | **minpeter/arxiv-abstracts-korean** | 50 | 한국어 | CS/물리/수학 | 대학원 | 미상 | ★☆☆☆☆ |
| 4 | **minpeter/arxiv-papers-korean-nllb-600M** | 10 | 한국어 | 전반 | 대학원 | 미상 | ★☆☆☆☆ |

---

### 3.2 Top 3 상세 분석

---

#### 🥇 1위: `amphora/korean_science_papers`

| 항목 | 내용 |
|------|------|
| **HuggingFace URL** | https://huggingface.co/datasets/amphora/korean_science_papers |
| **샘플 수** | **17,000+ 논문** |
| **언어** | 한국어 (일부 영어 키워드/단위 혼재) |
| **분야** | 생명과학, 식품과학, 의학, 화학, 환경 등 |
| **난이도** | 학술 대학원 수준 |
| **형태** | 논문 전문 (서론, 재료/방법, 결과/고찰, 결론) |
| **특이사항** | LaTeX 수식 포함, category 필드 있음 (생명, 화학 등) |
| **접근성** | 공개 (별도 동의 없음) |
| **Downloads** | 17k (최신) |

**데이터 구조**:
```json
{
  "title": "논문 제목",
  "context": "논문 전문 (섹션 포함)",
  "category": "생명"  // 생명, 화학, 의학 등
}
```

**예시 데이터**:  
```
[생명과학 논문]
지방세포로의 분화 초기단계에서 contact inhibition에 의해 증식이 정지되어 있던 
세포는 지방세포 유도 복합체에 의해 다시 세포 증식을 시작하는데...
C/EBPβ 발현이 RLE에 의해 저해됨을 확인하였기에...

[식품과학 논문]
쌀은 동남북아시아 국가에서 주식으로 사용되는 주요 곡물로서 전 세계적으로 
5,670만톤이 생산되며... 단백질 농축물을 제조하였으며...
```

**추천 이유**:  
- **유일한 대규모 한국어 과학 논문 데이터셋**  
- 과학적 전문 용어, 실험 방법, LaTeX 수식 포함  
- 카테고리별 필터링 가능  
- 한국 과학 어휘 및 표현 학습에 최적

---

#### 🥈 2위: `hiteshpatel945/korean-stem`

| 항목 | 내용 |
|------|------|
| **HuggingFace URL** | https://huggingface.co/datasets/hiteshpatel945/korean-stem |
| **샘플 수** | **316k** |
| **언어** | 한국어 |
| **분야** | STEM 전반 |
| **업데이트** | 2025년 (최신) |
| **접근성** | 공개 |
| **Downloads** | 2/월 (신규 데이터셋) |
| **주의** | 데이터 품질 및 출처 미상, 검증 필요 |

**추천 이유**:  
- 대규모 한국어 STEM 데이터  
- 교과서 수준 과학 지식 포함 가능성

**주의사항**:  
- 다운로드 수 낮아 품질 검증 필요  
- 출처 및 라이선스 확인 필수

---

#### 🥉 3위: `minpeter/arxiv-abstracts-korean`

| 항목 | 내용 |
|------|------|
| **HuggingFace URL** | https://huggingface.co/datasets/minpeter/arxiv-abstracts-korean |
| **샘플 수** | 50 (매우 소규모) |
| **언어** | 한국어 |
| **분야** | CS, 물리, 수학 (arXiv) |
| **형태** | arXiv 논문 초록 번역 |

**한계**: 50개 샘플로 실용적 학습 불가. 참고용에 그침.

---

### 3.3 과학 데이터 보완 전략

현재 한국어 과학 데이터는 극히 부족한 상황. 보완 방법:

```
1. AI-Hub 코딩/IT 카테고리 데이터 (계정 신청 필요)
   - URL: https://aihub.or.kr/
   - 한국 정부 지원 고품질 데이터
   - IT/과학 교육 콘텐츠 포함

2. 웹 크롤링 (한국 과학 사이트)
   - 네이버 학술 (scholar.naver.com)
   - RISS (riss.kr) 학위논문
   - KISS (kiss.kstudy.com) 학술지
   - 한국과학기술정보연구원 (KISTI)

3. 한국 교과서 데이터
   - 국가교육과정정보센터 디지털 교과서
   - 중/고등학교 과학 교과서 OCR

4. Wikipedia 한국어판 과학 문서
   - 이미 많은 한국어 LLM 학습에 포함
   - 물리, 화학, 생물, 지구과학 문서
```

---

## 4. 종합 추천 및 우선순위

### 4.1 즉시 사용 가능 (High Priority)

| 우선순위 | 데이터셋 | 도메인 | 토큰 수 추정 | 이유 |
|---------|---------|--------|------------|------|
| 🔴 P1 | bigcode/starcoderdata (Python subset) | 코드 | ~50B | 즉시 pretrain 가능, 검증됨 |
| 🔴 P1 | kuotient/orca-math-word-problems-193k-korean | 수학 | ~200M | 최대 한국어 수학, SFT/pretrain |
| 🔴 P1 | amphora/korean_science_papers | 과학 | ~150M | 유일한 한국어 과학 논문 |
| 🟡 P2 | nayohan/Evol-Instruct-Code-80k-v1-ko | 코드 | ~80M | 한국어 코딩 SFT |
| 🟡 P2 | re2panda/grade_school_math_korean | 수학 | ~15M | 한국어 GSM8K SFT |
| 🟡 P2 | openai/gsm8k | 수학 | ~10M | 영어 CoT, 번역 or 직접 사용 |

### 4.2 조사 중 미확인 / 추가 조사 필요

| 데이터셋 | 현황 | 비고 |
|---------|------|------|
| AI-Hub 코딩/IT | 계정 신청 필요 | 고품질 한국어 IT 데이터 기대 |
| hiteshpatel945/korean-stem | 품질 미검증 | 316k, 신규 데이터셋 |
| GitHub 한국어 레포 직접 수집 | 별도 작업 필요 | 한국 개발자 공개 레포 크롤링 |
| 수능/내신 수학 문제집 OCR | 별도 수집 필요 | 고품질 한국 수학 |

### 4.3 라이선스 위험도 정리

| 위험도 | 데이터셋 | 이유 |
|--------|---------|------|
| 🟢 안전 | bigcode/the-stack-v2, starcoderdata | permissive 라이선스만, provenance 제공 |
| 🟢 안전 | openai/gsm8k, hendrycks/math | MIT |
| 🟢 안전 | re2panda/grade_school_math_korean | MIT |
| 🟡 주의 | nayohan/Evol-Instruct-Code-80k-v1-ko | GPT-4 output 포함 (OpenAI ToS 이슈) |
| 🟡 주의 | amphora/korean_science_papers | 학술지 저작권 (연구 목적은 fair use 가능성) |
| 🔴 불명확 | hiteshpatel945/korean-stem | 출처 미상 |

---

## 5. 한국어 코드 주석 추출 방법

The Stack v2에서 한국어 주석이 포함된 코드 추출:

```python
from datasets import load_dataset
import re

def has_korean_text(text, min_korean_chars=10):
    """한글 10글자 이상 포함 여부 확인"""
    korean_chars = re.findall(r'[\uAC00-\uD7A3]', text)
    return len(korean_chars) >= min_korean_chars

def extract_korean_code(examples):
    """한국어 주석 포함 코드 필터링"""
    content = examples.get("content", "")
    return has_korean_text(content)

# Python 서브셋 로드 (streaming 권장)
ds = load_dataset(
    "bigcode/the-stack-v2-dedup", 
    "Python",
    split="train",
    streaming=True
)

# 한국어 포함 파일만 필터
korean_code_ds = ds.filter(extract_korean_code)
```

**예상 비율**: Python의 경우 한국어 주석 포함 파일 ~0.5-2% (GitHub 한국 사용자 비율 기반 추정)

---

## 6. 데이터 조합 추천 (3B 모델 학습 기준)

### Pretrain 믹스 (코드+수학+과학)

```yaml
pretrain_mix:
  code:
    - source: bigcode/starcoderdata
      languages: [python, javascript, java, cpp, typescript]
      sampling_weight: 0.35
      tokens: ~50B
    - source: bigcode/the-stack-v2-dedup (한국어 주석 필터)
      sampling_weight: 0.05
      tokens: ~5B
  
  math:
    - source: open-web-math/open-web-math
      sampling_weight: 0.10
      tokens: ~10B
    - source: kuotient/orca-math-word-problems-193k-korean
      sampling_weight: 0.05
      tokens: ~200M

  science:
    - source: amphora/korean_science_papers
      sampling_weight: 0.03
      tokens: ~150M

# 나머지는 일반 한국어/영어 텍스트로 채움
```

### SFT 믹스 (코드+수학)

```yaml
sft_mix:
  code_ko: nayohan/Evol-Instruct-Code-80k-v1-ko  # 78k
  code_en: nickrosh/Evol-Instruct-Code-80k-v1    # 78k (선택)
  math_ko: kuotient/orca-math-word-problems-193k-korean  # 193k
  math_ko_gsm: re2panda/grade_school_math_korean  # 7.5k
```

---

*조사일: 2026-02-27 | 조사자: survey-code-math subagent*
