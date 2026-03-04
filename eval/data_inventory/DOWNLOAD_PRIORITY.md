# 다운로드 우선순위 계획
> 생성일: 2026-02-27 | 디스크 여유: 19TB

## 즉시 다운로드 Top 5 (우선순위순)

---

### 🥇 Priority 1: FineWeb-Edu (Korean subset)
- **데이터셋:** `HuggingFaceFW/fineweb-edu`
- **왜:** 교육 품질 필터링된 웹 데이터, 고품질(A급). 한국어 서브셋만 추출 가능
- **예상:** 5~15B tokens (한국어 부분)
- **접근:** ✅ 무료, gated 아님
- **임팩트:** 고품질 pretrain 토큰 대량 확보 + 교육 도메인 강화
```bash
# 한국어 서브셋 다운로드
pip install datasets
python3 -c "
from datasets import load_dataset
ds = load_dataset('HuggingFaceFW/fineweb-edu', 'CC-MAIN-2024-10', split='train', streaming=True)
# language filter needed - fineweb-edu is primarily English
# Alternative: fineweb-edu-score filtered Korean web data
"
```
> ⚠️ 주의: fineweb-edu는 대부분 영어. 한국어 비중 적을 수 있음. 영어 고품질 보충용으로도 가치 있음.

---

### 🥈 Priority 2: Korean Preference/DPO 데이터 (다수 소스)
- **데이터셋들:**
  - `kuotient/orca-math-korean-preference` ✅
  - `kuotient/orca-math-korean-dpo-pairs` ✅  
  - `heegyu/orca-math-korean-preference-cleaned` ✅
  - `ohsuz/dpo-v1010-korean` ✅
  - `ChuGyouk/argilla-distilabel-math-preference-dpo-korean` ✅
- **왜:** Preference 데이터 **0건**인 현재 상태에서 ORPO 학습 자체 불가 → 가장 시급
- **예상:** 합계 30~60K 쌍
- **접근:** ✅ 모두 무료
- **임팩트:** ORPO/DPO 학습 파이프라인 활성화
```bash
python3 << 'PYEOF'
from datasets import load_dataset
import json, os

out_dir = "/PROJECT/0325120031_A/ghong/taketimes/llm-bang/data/preference"
os.makedirs(out_dir, exist_ok=True)

datasets_to_dl = [
    ("kuotient/orca-math-korean-preference", None),
    ("kuotient/orca-math-korean-dpo-pairs", None),
    ("heegyu/orca-math-korean-preference-cleaned", None),
    ("ohsuz/dpo-v1010-korean", None),
]

for name, config in datasets_to_dl:
    try:
        ds = load_dataset(name, config, split="train")
        safe_name = name.replace("/", "_")
        ds.to_json(f"{out_dir}/{safe_name}.jsonl")
        print(f"✅ {name}: {len(ds)} samples")
    except Exception as e:
        print(f"❌ {name}: {e}")
PYEOF
```

---

### 🥉 Priority 3: RedPajama-Data-1T (영어 고품질 서브셋)
- **데이터셋:** `togethercomputer/RedPajama-Data-1T`
- **왜:** 영어 데이터 극히 부족 (0.6B). 코드/ArXiv/Book/StackExchange 서브셋 선별 다운로드
- **예상:** 선별 10~20B tokens (코드 5B + ArXiv 3B + Book 2B + SE 2B)
- **접근:** ✅ 무료
- **임팩트:** 코드/과학/추론 능력 + cross-lingual transfer 대폭 강화
```bash
python3 << 'PYEOF'
from datasets import load_dataset

# 코드 서브셋만 먼저 (github subset)
ds = load_dataset("togethercomputer/RedPajama-Data-1T", "github", 
                   split="train", streaming=True,
                   cache_dir="/PROJECT/0325120031_A/ghong/taketimes/llm-bang/data/redpajama")
# ArXiv subset
ds_arxiv = load_dataset("togethercomputer/RedPajama-Data-1T", "arxiv",
                         split="train", streaming=True,
                         cache_dir="/PROJECT/0325120031_A/ghong/taketimes/llm-bang/data/redpajama")
PYEOF
```

---

### 4️⃣ Priority 4: 한국어 SFT 다양성 보강
- **데이터셋들:**
  - `kyujinpy/KOR-OpenOrca-Platypus-v3` ✅ (추론/수학)
  - `maywell/ko_wikidata_QA` ✅ (지식 QA)
  - `nlpai-lab/kullm-v2` ✅ (범용 지시)
- **왜:** 현재 SFT 170K은 양적 충분하나 코드/수학/추론 도메인 부족
- **예상:** +50~100K 다양한 도메인 샘플
- **접근:** ✅ 모두 무료
```bash
python3 << 'PYEOF'
from datasets import load_dataset
import os

out_dir = "/PROJECT/0325120031_A/ghong/taketimes/llm-bang/data/sft_extra"
os.makedirs(out_dir, exist_ok=True)

for name in ["kyujinpy/KOR-OpenOrca-Platypus-v3", "maywell/ko_wikidata_QA", "nlpai-lab/kullm-v2"]:
    try:
        ds = load_dataset(name, split="train")
        safe = name.replace("/","_")
        ds.to_json(f"{out_dir}/{safe}.jsonl")
        print(f"✅ {name}: {len(ds)}")
    except Exception as e:
        print(f"❌ {name}: {e}")
PYEOF
```

---

### 5️⃣ Priority 5: Open-Web-Math (수학 특화)
- **데이터셋:** `open-web-math/open-web-math`
- **왜:** 수학 데이터 전무. 수학 능력은 LLM 벤치마크 핵심 영역
- **예상:** ~14B tokens (영어 수학)
- **접근:** ✅ 무료
- **임팩트:** 수학 추론 능력 기반 확보
```bash
python3 -c "
from datasets import load_dataset
ds = load_dataset('open-web-math/open-web-math', split='train', streaming=True,
                   cache_dir='/PROJECT/0325120031_A/ghong/taketimes/llm-bang/data/open-web-math')
# Stream and save
"
```

---

## 다운로드 후 예상 토큰 분포

| 카테고리 | 현재 | 추가 | 합계 |
|---------|------|------|------|
| 한국어 Pretrain | 39B | +5~10B (fineweb-edu ko) | 44~49B |
| 영어 코드 | 0 | +5B (RedPajama github) | 5B |
| 영어 과학/ArXiv | 0 | +3B (RedPajama arxiv) | 3B |
| 영어 수학 | 0 | +10B (open-web-math) | 10B |
| 영어 기타 고품질 | 0.6B | +5B (RedPajama book+SE) | 5.6B |
| **Pretrain 합계** | **~39B** | **+28~33B** | **~67~72B** |
| SFT | 170K | +50~100K | 220~270K |
| Preference | 0 | +30~60K 쌍 | 30~60K 쌍 |

### 목표 달성 여부
- ✅ Chinchilla minimum (60B) 달성 가능
- ✅ ORPO/DPO 학습 가능
- ✅ 코드/수학/과학 도메인 커버
- 🟡 Chinchilla optimal (210B)에는 여전히 부족 → 추후 CulturaX 전체, SlimPajama 등 추가 검토

---

## 데이터 믹스 권장 비율 (학습 시)

```
한국어 텍스트:  50% (~35B tokens)
영어 코드:     15% (~10B tokens)  
영어 수학/과학: 15% (~10B tokens)
영어 일반:     15% (~10B tokens)
한국어 교육:    5% (~3B tokens)
```

## 주의사항
1. CulturaX는 gated(auto) → HuggingFace에서 동의 필요 (이미 다운받은 60GB 활용)
2. the-stack-dedup도 gated → 승인 필요, RedPajama github로 대체
3. 다운로드 전 `huggingface-cli login --token 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` 실행
4. 대용량 다운로드 시 `HF_HUB_ENABLE_HF_TRANSFER=1` 환경변수 설정 권장
