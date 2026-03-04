# 데이터 전수 실측 조사 결과
> 조사일: 2026-02-27 | 총 디스크 사용량: **195GB**

---

## 1. Pretrain 데이터 (.bin 파일) — 즉시 사용 가능

| 파일 | 크기 | 추정 토큰 수 | 비고 |
|------|------|-------------|------|
| `korean_train.bin` | 17GB | **8.9B** | 통합 (c4+wiki+namuwiki 머지) |
| `korean_val.bin` | 35MB | 17.9M | 통합 val |
| `korean_c4_train.bin` | 15GB | **7.5B** | C4 한국어 |
| `korean_c4_val.bin` | 29MB | 15.2M | |
| `korean_namuwiki_train.bin` | 2.1GB | **1.1B** | 나무위키 |
| `korean_namuwiki_val.bin` | 4.2MB | 2.2M | |
| `korean_wiki_train.bin` | 500MB | **261.8M** | 한국어 위키 |
| `korean_wiki_val.bin` | 1.1MB | 524K | |
| `train.bin` | 1.2GB | **605M** | 영어 위키 (Shakespeare 등) |
| `val.bin` | 5.8MB | 3.0M | |

### Pretrain 토큰 합계
- **korean_train.bin (통합)**: 8.9B tokens ← C4 + Wiki + Namuwiki 머지본
- **개별 합산** (c4 7.5B + wiki 0.26B + namuwiki 1.1B = 8.86B) → 통합본과 일치
- **영어 train.bin**: 605M tokens
- ⚠️ **korean_train.bin은 개별 .bin의 머지이므로 중복 계산 주의**
- **비중복 Pretrain 총합: ~9.5B tokens** (한국어 8.9B + 영어 0.6B)

---

## 2. korean_extra (HuggingFace 다운로드) — 처리 필요

| 디렉토리 | 크기 | 포맷 | 추정 토큰 |
|----------|------|------|----------|
| `culturax_ko` | 60GB | parquet | ~15B+ |
| `hplt_ko` | 23GB | parquet | ~6B |
| `cc100_ko` | 14GB | parquet/txt | ~3.5B |
| `oscar_ko` | 9.2GB | parquet | ~2.3B |
| `korean_textbooks` | 6.4GB | parquet | ~1.6B |
| `korean_webtext` | 4.2GB | parquet | ~1B |
| `finepdfs_edu_ko` | 2.9GB | parquet | ~700M |
| `namuwiki_extracted` | 2.2GB | parquet | ~550M |
| `wikipedia_korean` | 1.7GB | parquet | ~400M |
| `kovast` | 449MB | parquet | ~110M |
| `evol_instruct_ko` | 144MB | parquet/json | ~35M (SFT용) |
| `korean_safe_conv` | 51MB | parquet/json | ~12M (SFT용) |

**korean_extra 총합: ~123GB, 추정 ~30B+ tokens** (토큰화 전, 원문 기준)

---

## 3. SFT 데이터 — 즉시 사용 가능

| 파일 | 크기 | 샘플 수 |
|------|------|---------|
| `sft/train.jsonl` | 276MB | **161,848** |
| `sft/val.jsonl` | 15MB | **8,518** |

- **총 SFT 샘플: 170,366**
- 포맷: instruction/output 쌍, 한국어 번역 데이터
- 품질: 양호 (자연스러운 한국어, 다양한 주제)

---

## 4. Raw 텍스트 데이터 — 이미 .bin으로 변환 완료

| 디렉토리 | 크기 | 파일 수 | 비고 |
|----------|------|---------|------|
| `raw/c4_ko/` | 30GB | 50개 txt | → korean_c4_train.bin으로 변환됨 |
| `raw/namuwiki_ko/` | 5.7GB | 6개 txt | → korean_namuwiki_train.bin으로 변환됨 |
| `raw/ko_wiki_*.txt` | 1.2GB | 5개 txt | → korean_wiki_train.bin으로 변환됨 |
| `raw/en_wiki_*.txt` | 1.2GB | 3개 txt | → train.bin으로 변환됨 |
| **raw 합계** | **38GB** | **64개** | 삭제 가능 (디스크 절약) |

---

## 5. 종합 요약

### 즉시 사용 가능
| 용도 | 데이터 | 규모 |
|------|--------|------|
| **Pretrain** | korean_train.bin + train.bin | **9.5B tokens** |
| **SFT** | sft/train.jsonl | **161,848 샘플** |

### 처리하면 추가 확보 가능
| 소스 | 추정 규모 | 필요 작업 |
|------|----------|----------|
| korean_extra (전체) | **~30B+ tokens** | 토큰화 → .bin 변환 |
| evol_instruct_ko + korean_safe_conv | **~47M tokens (SFT)** | JSONL 변환 |

### 디스크 절약 가능
- `raw/` 38GB → 이미 .bin 변환 완료, 삭제 가능
- 개별 .bin (c4/wiki/namuwiki) → korean_train.bin 머지 후 중복, 삭제 가능 (~18GB)

### 최종 잠재력
- **Pretrain**: 현재 9.5B + korean_extra 30B+ = **~40B tokens 확보 가능**
- **SFT**: 현재 162K + 추가 변환 = **~200K+ 샘플 가능**
