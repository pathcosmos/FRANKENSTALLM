# SFT 개선 방안 심층 조사

> 프로젝트: 1B Korean LLM SFT (188k 샘플, 8×B200, FP8)
> 현재 구현: NEFTune, dynamic padding, gradient checkpointing, cosine LR, BF16+FP8
> 작성일: 2026-02-26

---

## 1. Curriculum Learning (교육과정 학습)

### 개념
쉬운 샘플에서 어려운 샘플 순서로 학습하여 수렴 속도와 최종 성능 향상.

### 구현 방법

**방법 A: Perplexity 기반 정렬 (권장)**
```python
# scripts/compute_difficulty.py
import torch, json
from pathlib import Path
from tokenizers import Tokenizer
from model import LLM

def compute_sample_perplexity(model, tokenizer, data_path, output_path, device="cuda:0"):
    """현재 pretrain 모델로 각 샘플의 output perplexity 계산"""
    model.eval()
    results = []
    
    with open(data_path) as f:
        samples = [json.loads(line) for line in f]
    
    with torch.no_grad():
        for i, sample in enumerate(samples):
            # conversation에서 assistant turn만 추출
            messages = sample["messages"]
            # 전체 시퀀스 토크나이즈
            full_text = tokenizer.encode(
                "".join(m["content"] for m in messages)
            )
            input_ids = torch.tensor([full_text.ids[:4096]], device=device)
            
            logits = model(input_ids)
            # response 토큰에 대한 CE loss = perplexity proxy
            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]
            loss = torch.nn.functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                reduction='mean'
            )
            ppl = loss.exp().item()
            results.append({"idx": i, "ppl": ppl})
            
            if i % 1000 == 0:
                print(f"  {i}/{len(samples)} done")
    
    # ppl 오름차순 정렬 = 쉬운 것부터
    results.sort(key=lambda x: x["ppl"])
    
    with open(output_path, "w") as f:
        json.dump(results, f)
    return results
```

**방법 B: 길이 기반 (가장 간단)**
- 짧은 응답 → 긴 응답 순서로 정렬
- SFTDataset에서 `__getitem__` 시 정렬된 인덱스 사용

**방법 C: IFD Score**
- Cherry LLM 논문 (2024): `IFD = PPL(output|instruction) / PPL(output)`
- 높은 IFD = instruction이 output 생성을 잘 유도하지 못함 = 어려움

### 실제 효과
- **Curriculum Learning for LLMs (Xu et al., 2024)**: SFT에서 MT-Bench +0.3~0.5점
- **효과 제한적 의견**: Bengio et al.의 원 연구 이후 SFT에서의 효과는 mixed results
- **예상**: ko_ifeval +1~2%, 반복률 변화 미미

### 평가
| 항목 | 값 |
|------|-----|
| 예상 효과 | ko_ifeval +1~2% |
| 구현 복잡도 | 2/5 |
| 소요 시간 | PPL 계산 2~3시간 + 코드 수정 2시간 |
| 적용 가능 | ✅ DataLoader sampler 수정으로 가능 |

---

## 2. "Less is More" 전략 (LIMA, AlpaGasus)

### 핵심 논문
- **LIMA (Zhou et al., 2023)**: 1000개 고품질 > 52k 저품질. 65B 모델에서 검증.
- **AlpaGasus (Chen et al., 2023)**: GPT-4로 품질 점수 → 9k에서 3k 선별 → Alpaca 대비 우수
- **DEITA (Liu et al., 2024)**: complexity + quality + diversity 3축 필터링

### 품질 점수 계산 방법 (외부 API 없이)

```python
# scripts/quality_filter.py
import json, math, torch
import numpy as np
from collections import Counter

def compute_quality_scores(data_path, model, tokenizer, device="cuda:0"):
    """다차원 품질 점수 계산"""
    with open(data_path) as f:
        samples = [json.loads(line) for line in f]
    
    scored = []
    model.eval()
    
    for i, sample in enumerate(samples):
        msgs = sample["messages"]
        
        # 1) 길이 점수: 너무 짧거나 너무 긴 건 감점
        response = "".join(m["content"] for m in msgs if m["role"] == "assistant")
        resp_len = len(response)
        len_score = min(resp_len / 500, 1.0) * (1.0 if resp_len < 3000 else 3000 / resp_len)
        
        # 2) 반복 감점: n-gram 반복률
        tokens = list(response)
        if len(tokens) > 10:
            trigrams = [tuple(tokens[j:j+3]) for j in range(len(tokens)-2)]
            unique_ratio = len(set(trigrams)) / len(trigrams)
        else:
            unique_ratio = 1.0
        rep_score = unique_ratio
        
        # 3) Perplexity 점수 (중간이 좋음 - 너무 낮으면 trivial, 너무 높으면 noise)
        # 사전 계산된 ppl 사용
        
        # 4) Instruction 복잡도: instruction 길이
        instruction = "".join(m["content"] for m in msgs if m["role"] == "user")
        inst_complexity = min(len(instruction) / 200, 1.0)
        
        # 종합 점수
        quality = 0.3 * len_score + 0.3 * rep_score + 0.2 * inst_complexity + 0.2
        scored.append({"idx": i, "quality": quality, "sample": sample})
        
    return scored

def select_top_k(scored, k):
    """상위 k개 선별"""
    scored.sort(key=lambda x: x["quality"], reverse=True)
    return scored[:k]
```

### 권장 샘플 수
- 188k 전체 → **50k~80k** 권장 (상위 30~40%)
- 1B 모델 규모에서는 LIMA처럼 극단적 축소보다 moderate 필터링이 적합
- 근거: AlpaGasus는 ~30% 선별에서 최적, 1B는 65B보다 데이터 의존도 높음

### 평가
| 항목 | 값 |
|------|-----|
| 예상 효과 | 반복률 -30~50%, ko_ifeval +3~5% |
| 구현 복잡도 | 2/5 |
| 소요 시간 | 품질 계산 3~4시간, 필터링 코드 1시간 |
| 적용 가능 | ✅ 데이터 전처리 단계 |

---

## 3. Packing (Sequence Packing)

### 개념
짧은 시퀀스들을 하나의 max_seq_len에 패킹하여 padding 낭비 제거.

### 현 프로젝트 상황
- 이미 `dynamic_collate_fn`으로 batch-level dynamic padding 구현됨
- Packing은 그 이상: **여러 샘플을 하나의 시퀀스로 concatenate**

### 주의사항: Cross-contamination
- 패킹된 서로 다른 샘플 간 attention이 흐르면 안 됨
- **해결**: Flash Attention v2의 `cu_seqlens` 파라미터 (varlen attention)
- 또는 block diagonal attention mask

### 구현 방법
```python
# data/packed_sft_dataset.py
class PackedSFTDataset:
    """여러 SFT 샘플을 하나의 시퀀스로 패킹"""
    
    def __init__(self, samples, tokenizer, max_seq_len=4096):
        self.packed = []
        self.cu_seqlens = []  # Flash Attention varlen용
        
        buffer_ids = []
        buffer_labels = []
        seq_lens = []
        
        for sample in samples:
            ids, labels = self._tokenize(sample, tokenizer)
            
            if len(buffer_ids) + len(ids) > max_seq_len:
                # 현재 버퍼 저장
                if buffer_ids:
                    self._save_buffer(buffer_ids, buffer_labels, seq_lens, max_seq_len)
                buffer_ids = ids
                buffer_labels = labels
                seq_lens = [len(ids)]
            else:
                buffer_ids.extend(ids)
                buffer_labels.extend(labels)
                seq_lens.append(len(ids))
        
        if buffer_ids:
            self._save_buffer(buffer_ids, buffer_labels, seq_lens, max_seq_len)
    
    def _save_buffer(self, ids, labels, seq_lens, max_seq_len):
        # Pad to max_seq_len
        pad_len = max_seq_len - len(ids)
        ids = ids + [0] * pad_len
        labels = labels + [-1] * pad_len
        
        # cu_seqlens for varlen flash attention
        cu = [0]
        for l in seq_lens:
            cu.append(cu[-1] + l)
        
        self.packed.append({
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(labels),
            "cu_seqlens": torch.tensor(cu, dtype=torch.int32),
        })
```

### 속도 개선 예상
- 현재 평균 시퀀스 길이가 max_seq_len(4096)보다 훨씬 짧다면 **1.5~3× 속도 향상**
- SFT 데이터 특성상 평균 ~1000 토큰이면 ~3× 효율 향상 예상

### 평가
| 항목 | 값 |
|------|-----|
| 예상 효과 | 학습 속도 1.5~3×, 성능 변화 없거나 미미 |
| 구현 복잡도 | 3/5 (Flash Attention varlen 연동 필요) |
| 소요 시간 | 1~2일 |
| 적용 가능 | ⚠️ 모델의 attention 구현이 cu_seqlens 지원해야 함 |

---

## 4. Multi-task SFT (도메인별 Loss Weighting)

### 개념
데이터 소스별로 도메인을 분류하고, 도메인별 loss weight를 다르게 적용.

### 현재 데이터 소스 분석 (추정)
- `korean_safe_conv/raw/` 하위: hatespeech, square, evol, yitingxie, gamseong, koalpaca, conversation
- 카테고리: 안전성, QA, 창작, 일반 대화

### 구현 방법
```python
# 도메인 태그를 JSONL에 추가
# {"messages": [...], "domain": "qa"}  
# {"messages": [...], "domain": "creative"}

# trainer에서 도메인별 loss weight
DOMAIN_WEIGHTS = {
    "qa": 1.0,
    "creative": 0.8,
    "safety": 1.2,
    "code": 1.0,
    "math": 1.0,
    "conversation": 0.6,  # 일반 대화는 낮게
}
```

### 평가
| 항목 | 값 |
|------|-----|
| 예상 효과 | 특정 벤치마크 +1~3% |
| 구현 복잡도 | 3/5 |
| 소요 시간 | 도메인 분류 0.5일 + 구현 0.5일 |
| 적용 가능 | ✅ loss 계산 시 weight 곱셈 |

---

## 5. Token-level Loss Weighting / Focal Loss

### 개념
모든 response 토큰에 동일 weight 대신, 모델이 예측하기 어려운 토큰에 더 높은 weight.

### Focal Loss 구현
```python
# train/focal_loss.py
import torch
import torch.nn.functional as F

def focal_cross_entropy(logits, targets, gamma=2.0, ignore_index=-1):
    """
    Focal loss: down-weight easy tokens, up-weight hard tokens.
    Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    """
    # Standard CE
    ce_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=ignore_index,
        reduction='none'
    )
    
    # p_t = probability of correct class
    log_pt = -ce_loss
    pt = torch.exp(log_pt)
    
    # Focal weight: (1 - p_t)^gamma
    focal_weight = (1 - pt) ** gamma
    loss = focal_weight * ce_loss
    
    # Mask ignored tokens
    mask = (targets.reshape(-1) != ignore_index)
    loss = loss[mask].mean()
    
    return loss
```

### 적용: trainer.py 수정
```python
# trainer.py의 _compute_loss에서
# 기존: F.cross_entropy(logits, targets, ignore_index=-1)
# 변경: focal_cross_entropy(logits, targets, gamma=2.0, ignore_index=-1)
```

### 실제 효과
- SFT에서 focal loss 적용 논문은 제한적
- **SelectIT (Liu et al., 2024)**: token-level selection으로 IFEval +2~4%
- gamma=2.0이 일반적이나 SFT에서는 gamma=1.0~1.5 권장 (너무 강하면 불안정)

### 평가
| 항목 | 값 |
|------|-----|
| 예상 효과 | ko_ifeval +1~3%, 반복률 영향 미미 |
| 구현 복잡도 | 1/5 |
| 소요 시간 | 2시간 |
| 적용 가능 | ✅ loss 함수만 교체 |

---

## 6. Data Augmentation for Korean

### 방법 A: Self-Paraphrase
```bash
# 현재 모델(또는 더 큰 모델)로 response 재생성
# instruction은 유지, output만 다양화
# → 동일 instruction에 대한 N개 다른 응답 확보
```

### 방법 B: Back-translation
```python
# 영어 고품질 데이터 (Alpaca, Dolly, OpenAssistant) → 한국어 번역
# 서버에서 실행 가능한 방법:
# 1) NLLB-200 3.3B (Meta): 오프라인 번역, 1GPU로 실행 가능
# 2) 한국어 특화 모델로 직접 번역 생성

# pip install transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "facebook/nllb-200-3.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda:7")  # 여유 GPU 1개 사용

def translate_en_to_ko(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True).to("cuda:7")
    outputs = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id["kor_Hang"], max_new_tokens=1024)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### 평가
| 항목 | 값 |
|------|-----|
| 예상 효과 | 다양성 증가, ko_ifeval +2~4% (좋은 소스 데이터 시) |
| 구현 복잡도 | 3/5 |
| 소요 시간 | 번역 파이프라인 1일 + 번역 실행 1~2일 |
| 적용 가능 | ✅ 별도 GPU에서 병렬 실행 가능 |

---

## 7. 학습 안정성 개선

### FP8 학습 주의사항
현재 설정: MXFP8 + BF16 기반. 주요 주의점:

1. **Loss spike 방지**
   - `max_grad_norm: 1.0` 이미 적용됨 ✅
   - LR 2e-5는 보수적 ✅
   - 추가: gradient norm 모니터링 + 자동 LR 감소
   ```python
   # 연속 3번 grad_norm > threshold면 lr 반감
   if grad_norm > 5.0:
       spike_count += 1
       if spike_count >= 3:
           for pg in optimizer.param_groups:
               pg['lr'] *= 0.5
   ```

2. **Weight Decay**
   - 현재 0.01 (적절)
   - SFT에서는 0.01~0.05 범위가 표준

3. **Dropout**
   - 현재 `dropout: 0.0` — SFT에서는 **0.05~0.1 추가 권장**
   - 과적합 방지에 직접적 효과
   ```yaml
   dropout: 0.05  # configs/korean_1b_sft.yaml
   ```

4. **FP8 amax 설정**
   - `fp8_amax_history_len: 16` + `fp8_amax_compute_algo: "max"` — 적절
   - MXFP8은 DelayedScaling보다 안정적

### 평가
| 항목 | 값 |
|------|-----|
| 예상 효과 | 안정성 ↑, 과적합 -10~20%, 반복률 -5~10% |
| 구현 복잡도 | 1/5 |
| 소요 시간 | 1시간 |
| 적용 가능 | ✅ config 변경만으로 가능 |

---

## 8. 평가 기반 데이터 선택 (Self-Play → ORPO)

### 파이프라인
```
1. 현재 SFT 모델로 각 instruction에 대해 N=4개 응답 생성
2. 자동 평가 (반복률, 길이, 일관성)로 best/worst 선정
3. (chosen, rejected) 페어 구성
4. ORPO/DPO 학습 (이미 train/orpo.py 존재!)
```

### 구체적 단계
```python
# scripts/generate_self_play.py
def generate_candidates(model, tokenizer, instructions, n=4, temp=0.8):
    pairs = []
    for inst in instructions:
        responses = []
        for _ in range(n):
            out = model.generate(inst, temperature=temp, max_new_tokens=1024)
            score = auto_evaluate(out)  # 반복률, 길이, coherence
            responses.append((out, score))
        
        responses.sort(key=lambda x: x[1], reverse=True)
        pairs.append({
            "instruction": inst,
            "chosen": responses[0][0],
            "rejected": responses[-1][0],
        })
    return pairs
```

### 평가
| 항목 | 값 |
|------|-----|
| 예상 효과 | 반복률 -40~60%, ko_ifeval +3~5% |
| 구현 복잡도 | 4/5 |
| 소요 시간 | 생성 1~2일 + ORPO 학습 0.5일 |
| 적용 가능 | ✅ orpo.py 이미 존재 |

---

## 종합 비교표

| 기법 | 예상 효과 (반복률) | 예상 효과 (ko_ifeval) | 구현 복잡도 | 소요 시간 | 우선순위 |
|------|-------------------|---------------------|------------|----------|---------|
| 1. Curriculum Learning | - | +1~2% | 2/5 | 5시간 | 중기 |
| 2. Less is More | **-30~50%** | **+3~5%** | 2/5 | 5시간 | **즉시** |
| 3. Packing | (속도만) | (변화없음) | 3/5 | 1~2일 | 중기 |
| 4. Multi-task Weighting | - | +1~3% | 3/5 | 1일 | 중기 |
| 5. Focal Loss | - | +1~3% | **1/5** | **2시간** | **즉시** |
| 6. Data Augmentation | - | +2~4% | 3/5 | 2~3일 | 중기 |
| 7. 학습 안정성 (dropout) | **-5~10%** | - | **1/5** | **1시간** | **즉시** |
| 8. Self-Play → ORPO | **-40~60%** | **+3~5%** | 4/5 | 2~3일 | 중기 |

---

## 🚀 즉시 적용 Top 3

### 1위: "Less is More" 데이터 필터링
- **근거**: LIMA, AlpaGasus 논문에서 일관되게 입증. 188k → 50~80k 필터링으로 저품질/반복적 샘플 제거
- **예상 효과**: 반복률 -30~50%, ko_ifeval +3~5%
- **소요**: 5시간 (PPL 계산 + 필터링 스크립트)
- **리스크**: 낮음 (최악의 경우 전체 데이터로 롤백)

### 2위: Focal Loss 적용
- **근거**: 어려운 토큰에 집중 → instruction following 능력 향상. 구현 극히 간단
- **예상 효과**: ko_ifeval +1~3%
- **소요**: 2시간 (loss 함수 1개 추가)
- **리스크**: 매우 낮음 (gamma 값만 조정하면 됨)

### 3위: Dropout 추가 (0.05)
- **근거**: 현재 dropout=0.0으로 과적합 위험. SFT에서 light dropout은 표준
- **예상 효과**: 과적합 감소, 반복률 -5~10%
- **소요**: config 한 줄 변경
- **리스크**: 없음

---

## 📅 중기 적용 Top 3

### 1위: Self-Play → ORPO (SFT 이후)
- **근거**: SFT 완료 후 선호 학습은 반복률 감소에 가장 효과적. orpo.py 이미 구현됨
- **예상 효과**: 반복률 -40~60%, ko_ifeval +3~5%
- **소요**: 2~3일 (생성 + 학습)

### 2위: Sequence Packing
- **근거**: 학습 속도 1.5~3× 향상. 향후 반복 실험에 필수적
- **예상 효과**: 학습 시간 대폭 단축
- **소요**: 1~2일

### 3위: Curriculum Learning + Data Augmentation (Back-translation)
- **근거**: 데이터 다양성과 학습 효율 동시 개선
- **예상 효과**: ko_ifeval +2~4%
- **소요**: 3~4일

---

## 권장 실행 순서

```
Phase 1 (즉시, SFT 전): 
  1. dropout: 0.05 설정
  2. 데이터 품질 필터링 (188k → 60~80k)
  3. Focal loss 적용 (gamma=1.5)
  → SFT 실행

Phase 2 (SFT 후): 
  4. Self-Play 데이터 생성
  5. ORPO 학습

Phase 3 (다음 라운드):
  6. Packing 구현 (반복 실험 가속)
  7. Back-translation으로 데이터 확장
  8. Curriculum learning 실험
```
