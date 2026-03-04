# 한국어 SFT/Instruction 데이터셋 전수 조사

**조사일**: 2026-02-27
**조사 범위**: HuggingFace Hub 한국어 SFT/Instruction 데이터셋

---

## 1. 현재 SFT 데이터 현황

| 항목 | 값 |
|------|-----|
| 파일 | `/PROJECT/.../data/sft/train.jsonl` |
| 총 건수 | **161,848** |
| 포맷 | `instruction` / `input` / `output` (Alpaca 형식) |
| 소스 필드 | ❌ 없음 (`source` 키 미존재) |

> ⚠️ 소스 추적이 불가능하여 중복/출처 검증이 어려움. 향후 데이터 추가 시 `source` 필드 필수 권장.

---

## 2. HuggingFace 한국어 SFT 데이터셋 목록

### Tier 1 — 최고품질 (인간 작성 / 강력 필터링 / GPT-4 생성+검증)

| 데이터셋 | 크기 | 언어 | 설명 | DL |
|----------|------|------|------|-----|
| `nlpai-lab/kullm-v2` | 10K~100K | 🇰🇷 | GPT-4 기반 한국어 instruction, 커뮤니티 검증 | 730 |
| `FreedomIntelligence/alpaca-gpt4-korean` | ~52K | 🇰🇷 | GPT-4로 생성한 한국어 Alpaca | 158 |
| `dbdu/ShareGPT-74k-ko` | 10K~100K | 🇰🇷 | ShareGPT 한국어 번역, 멀티턴 대화 | 169 |
| `squarelike/sharegpt_deepl_ko_translation` | ~50K+ | 🇰🇷 | ShareGPT DeepL 번역, 고품질 번역체 | 41 |
| `kuotient/orca-math-word-problems-193k-korean` | 100K~1M | 🇰🇷 | 수학 문제 한국어 번역, 대규모 | 396 |
| `HuggingFaceH4/no_robots` | ~10K | 🇬🇧 | 인간 작성 고품질 (영어, 번역 가치 높음) | 5,211 |
| `allenai/tulu-3-sft-mixture` | 100K~1M | 다국어 | Allen AI 최신 SFT 믹스, 고품질 큐레이션 | 22,453 |
| `HAERAE-HUB/K2-Feedback` | ~수천 | 🇰🇷 | 한국어 평가/피드백 데이터 | 54 |

### Tier 2 — 중간 품질 (GPT-3.5/4 생성, 부분 검증)

| 데이터셋 | 크기 | 언어 | 설명 | DL |
|----------|------|------|------|-----|
| `beomi/KoAlpaca-v1.1a` | ~52K | 🇰🇷 | 한국어 Alpaca, 널리 사용 | 3,096 |
| `kyujinpy/KOR-OpenOrca-Platypus-v3` | 10K~50K | 🇰🇷 | OpenOrca+Platypus 한국어 병합 | 612 |
| `kyujinpy/OpenOrca-KO` | 10K~50K | 🇰🇷 | OpenOrca 한국어 번역 | 139 |
| `squarelike/OpenOrca-gugugo-ko` | **10M~100M** | 🇰🇷 | 초대규모 OpenOrca 한국어 번역 | 82 |
| `nlp-with-deeplearning/Ko.WizardLM_evol_instruct_V2_196k` | ~196K | 🇰🇷 | WizardLM Evol Instruct 한국어 | 20 |
| `heegyu/open-korean-instructions` | 다양 | 🇰🇷 | 여러 한국어 instruction 통합 | 214 |
| `nayohan/instruction_en_ko_translation_1.4m` | **1.4M** | 🇰🇷 | 대규모 영→한 instruction 번역 | 11 |
| `nayohan/Evol-Instruct-Code-80k-v1-ko` | ~80K | 🇰🇷 | 코드 instruction 한국어 | 23 |
| `changpt/ko-lima-vicuna` | <1K | 🇰🇷 | LIMA+Vicuna 한국어 (소량 고품질) | 43 |
| `OpenLab-NLP/tiny-instruct-ko` | ~수만 | 🇰🇷 | 한국어 instruction 소규모 | 127 |
| `nlpai-lab/openassistant-guanaco-ko` | 1K~10K | 🇰🇷 | OpenAssistant Guanaco 한국어 | 48 |
| `HuggingFaceH4/ultrachat_200k` | 100K~1M | 🇬🇧 | 고품질 대화 (영어, 번역 가치) | 33,729 |
| `kyujinpy/KOpen-platypus` | ~25K | 🇰🇷🇬🇧 | Platypus 한국어 | 306 |

### Tier 3 — 참고용 (노이즈 가능성, 추가 필터링 필요)

| 데이터셋 | 크기 | 언어 | 설명 | DL |
|----------|------|------|------|-----|
| `CarrotAI/ko-instruction-dataset` | 1K~10K | 🇰🇷 | 소규모 | 71 |
| `CarrotAI/ko-code-alpaca-QA` | 소규모 | 🇰🇷 | 코드 QA | 71 |
| `causal-lm/instructions-ko` | 불명 | 🇰🇷 | | 21 |
| `junelee/sharegpt_deepl_ko` | ~수만 | 🇰🇷 | DeepL 번역 | 86 |
| `neuralfoundry-coder/aihub-korean-education-instruct-sample` | 샘플 | 🇰🇷 | 교육 도메인 | 32 |
| `neuralfoundry-coder/korean-legal-instruction-sample` | 샘플 | 🇰🇷 | 법률 도메인 | 30 |

### 영어 대규모 (번역 파이프라인으로 활용 가능)

| 데이터셋 | 크기 | 설명 | DL |
|----------|------|------|-----|
| `Open-Orca/OpenOrca` | ~4M | FLAN 기반 대규모 | - |
| `teknium/OpenHermes-2.5` | ~1M | 고품질 혼합 | - |
| `WizardLM/WizardLM_evol_instruct_V2_196k` | 196K | Evol Instruct | - |
| `stingning/ultrachat` | 1M~10M | 대화형 | 2,838 |
| `iamtarun/python_code_instructions_18k_alpaca` | 18K | 코드 | 6,499 |
| `sahil2801/CodeAlpaca-20k` | 20K | 코드 | 12,060 |

---

## 3. 도메인 커버리지 분석

### 현재 데이터 (161K) 추정 도메인 분포

데이터에 `source` 필드가 없어 정확한 분석 불가. 데이터 내용 샘플링 기반 추정:

| 도메인 | 추정 비율 | 상태 |
|--------|----------|------|
| 일반 지식/QA | ~40% | ✅ 충분 |
| 번역체 대화 | ~25% | ✅ 충분 |
| 창작/글쓰기 | ~15% | ⚠️ 보통 |
| 코딩 | ~5% | ❌ **부족** |
| 수학/과학 | ~5% | ❌ **부족** |
| 한국어 특화 (문화/역사/법률) | ~5% | ❌ **부족** |
| 롤플레이/페르소나 | ~5% | ⚠️ 보통 |

### 도메인 갭 (부족한 영역)

1. **수학/논리 추론** — 현재 거의 없음. `kuotient/orca-math-word-problems-193k-korean` (193K)로 즉시 보완 가능
2. **코딩** — 한국어 코드 instruction 극소. `nayohan/Evol-Instruct-Code-80k-v1-ko` (80K) 활용 필요
3. **한국어 특화 지식** — 한국 문화, 역사, 법률, 수능 등 도메인 특화 데이터 부족
4. **멀티턴 대화** — 싱글턴 QA 위주. `dbdu/ShareGPT-74k-ko`, `ultrachat_200k` 번역으로 보완
5. **Safety/거절 응답** — 유해 요청 거절 학습 데이터 부재

---

## 4. 즉시 다운로드 권장 Top 5

### 🥇 1. `kuotient/orca-math-word-problems-193k-korean`
- **크기**: ~193K
- **이유**: 수학 도메인 완전 보완. 한국어 네이티브 번역. 대규모.
- **품질**: Tier 1-2 (Orca Math 기반, 검증됨)
- **우선도**: ★★★★★

### 🥈 2. `dbdu/ShareGPT-74k-ko`
- **크기**: ~74K
- **이유**: 실제 ChatGPT 대화 기반 멀티턴. 다양한 도메인. 번역 품질 양호.
- **품질**: Tier 1 (실사용자 대화 기반)
- **우선도**: ★★★★★

### 🥉 3. `nayohan/Evol-Instruct-Code-80k-v1-ko`
- **크기**: ~80K
- **이유**: 코딩 도메인 유일한 대규모 한국어 데이터. WizardCoder 기반.
- **품질**: Tier 2
- **우선도**: ★★★★☆

### 4️⃣ 4. `nlp-with-deeplearning/Ko.WizardLM_evol_instruct_V2_196k`
- **크기**: ~196K
- **이유**: Evol Instruct로 난이도 다양. 복잡한 instruction 포함. 대규모.
- **품질**: Tier 2
- **우선도**: ★★★★☆

### 5️⃣ 5. `FreedomIntelligence/alpaca-gpt4-korean`
- **크기**: ~52K
- **이유**: GPT-4 생성으로 응답 품질 높음. 기존 Alpaca 데이터와 상보적.
- **품질**: Tier 1
- **우선도**: ★★★☆☆

---

## 5. 추가 권장 사항

### 즉시 조치
1. 현재 `train.jsonl`에 `source` 필드 추가 (역추적 or 향후 데이터부터)
2. Top 5 데이터셋 다운로드 → 중복 제거 → `source` 태깅 후 병합
3. 예상 추가 데이터: **~595K** (193K + 74K + 80K + 196K + 52K)
4. 병합 후 총 규모: **~757K** (현재 162K + 595K)

### 중기 계획
- `nayohan/instruction_en_ko_translation_1.4m` — 1.4M 대규모이나 품질 검증 필요
- `squarelike/OpenOrca-gugugo-ko` — 초대규모(10M+)이나 노이즈 필터링 필수
- `allenai/tulu-3-sft-mixture` — 다국어 포함, 한국어 부분 추출 가치
- Safety 데이터 자체 구축 (유해 요청 거절 시나리오)

### 도메인 특화 보강
- **법률**: `neuralfoundry-coder/korean-legal-instruction-sample` (샘플만 공개, AI Hub 원본 확인 필요)
- **교육**: `neuralfoundry-coder/aihub-korean-education-instruct-sample`
- **의료**: `squarelike/ko_medical_chat` (25 DL, 소규모)

---

## 6. 404 (삭제/비공개) 데이터셋

다음 데이터셋은 현재 접근 불가:
- `Bingsu/ko-alpaca-cleaned` ❌
- `naver-clova-ix/koco-v1-5` (별도 확인 필요)
- `kuotient/korean-conversation-dataset` (별도 확인 필요)
- `HAERAE-HUB/K2-Bench-Instruction` ❌
- `nayohan/llama3-instruct-ko` ❌
- `Bongseok/Kor-Platypus2` ❌
- `kuotient/orca-math-word-problems-korean` ❌ (→ `orca-math-word-problems-193k-korean`이 정확한 이름)
- `kyujinpy/Kor-Platypus2-T70k` ❌
- `HAERAE-HUB/qarv-instruct-100k` ❌
