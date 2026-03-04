"""SFT v2 comprehensive evaluation script."""
import sys, json, os
sys.path.insert(0, "/PROJECT/0325120031_A/ghong/taketimes/llm-bang")

import torch
import torch.nn.functional as F
from pathlib import Path
from eval.generate import load_model_and_tokenizer, generate

CKPT = "/PROJECT/0325120031_A/ghong/taketimes/llm-bang/checkpoints/korean_1b_sft/checkpoint-best"
DEVICE = "cuda:0"
OUTPUT_DIR = "/PROJECT/0325120031_A/ghong/taketimes/llm-bang/eval/sft_v2_eval"

QUESTIONS = [
    "한국의 수도는 어디인가요?",
    "파이썬에서 리스트를 정렬하는 방법을 설명해주세요.",
    "지구온난화의 주요 원인을 설명하세요.",
    "좋은 수면 습관을 만들기 위한 팁을 알려주세요.",
    "한국 전통 음식 중 김치에 대해 설명해주세요.",
    "머신러닝과 딥러닝의 차이점은 무엇인가요?",
    "스트레스 해소 방법을 알려주세요.",
    "효과적인 공부 방법을 설명해주세요.",
    "인공지능의 미래에 대해 어떻게 생각하시나요?",
    "건강한 식습관을 유지하는 방법을 알려주세요.",
]

def calc_repetition_rate(text, n=3):
    tokens = list(text)
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    if not ngrams:
        return 0.0
    unique = set(ngrams)
    return 1.0 - len(unique) / len(ngrams)

def main():
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(CKPT, DEVICE)
    eos_id = tokenizer.token_to_id("</s>")
    
    # Get stop token for <|user|>
    user_token_id = tokenizer.token_to_id("<|user|>")
    
    results = []
    
    print("\n=== Generation Evaluation ===\n")
    for i, q in enumerate(QUESTIONS):
        prompt = f"<|user|>\n{q}\n<|assistant|>\n"
        
        # Collect generated text
        gen_tokens = []
        full_text = ""
        stopped_eos = False
        
        # Use modified generation with repetition penalty and no_repeat_ngram
        input_ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long, device=DEVICE)
        generated_ids = input_ids.clone()
        
        for step in range(200):
            logits_all, _ = model(generated_ids)
            logits = logits_all[:, -1, :].float()
            
            # Temperature
            logits = logits / 0.7
            
            # Repetition penalty
            for token_id in set(generated_ids[0].tolist()):
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= 1.1
                else:
                    logits[0, token_id] *= 1.1
            
            # No repeat 3-gram
            if generated_ids.shape[1] >= 3:
                last_2 = tuple(generated_ids[0, -2:].tolist())
                for j in range(generated_ids.shape[1] - 2):
                    if tuple(generated_ids[0, j:j+2].tolist()) == last_2:
                        blocked = generated_ids[0, j+2].item()
                        logits[0, blocked] = float('-inf')
            
            # Top-p sampling
            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            mask = cumsum - sorted_probs >= 0.9
            sorted_probs[mask] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum()
            
            next_token = sorted_idx[0, torch.multinomial(sorted_probs[0], 1)]  # [1]
            generated_ids = torch.cat([generated_ids, next_token.view(1, 1)], dim=-1)
            
            tid = next_token.item()
            gen_tokens.append(tid)
            
            if tid == eos_id:
                stopped_eos = True
                break
            if user_token_id and tid == user_token_id:
                stopped_eos = True
                break
        
        full_text = tokenizer.decode(gen_tokens)
        # Clean up
        if "<|user|>" in full_text:
            full_text = full_text[:full_text.index("<|user|>")]
        full_text = full_text.replace("</s>", "").strip()
        
        rep_rate = calc_repetition_rate(full_text)
        
        result = {
            "question": q,
            "answer": full_text,
            "repetition_rate": rep_rate,
            "stopped_eos": stopped_eos,
            "num_tokens": len(gen_tokens),
        }
        results.append(result)
        
        print(f"[{i+1}] {q}")
        print(f"    반복률: {rep_rate*100:.1f}% | EOS: {stopped_eos} | 토큰: {len(gen_tokens)}")
        print(f"    답변: {full_text[:100]}...")
        print()
    
    avg_rep = sum(r["repetition_rate"] for r in results) / len(results) * 100
    eos_rate = sum(1 for r in results if r["stopped_eos"]) / len(results) * 100
    
    print(f"\n=== 요약 ===")
    print(f"평균 반복률: {avg_rep:.1f}%")
    print(f"자연 종료율: {eos_rate:.1f}%")
    
    # Val loss calculation
    print("\n=== Val Loss 계산 ===")
    val_path = Path("/PROJECT/0325120031_A/ghong/taketimes/llm-bang/data/sft/val.jsonl")
    if val_path.exists():
        val_data = []
        with open(val_path) as f:
            for i, line in enumerate(f):
                if i >= 100:
                    break
                val_data.append(json.loads(line))
        
        total_loss = 0.0
        count = 0
        model.eval()
        with torch.no_grad():
            for item in val_data:
                # Format as SFT
                if "conversations" in item:
                    convs = item["conversations"]
                    text = ""
                    for c in convs:
                        role = c.get("role", c.get("from", ""))
                        content = c.get("content", c.get("value", ""))
                        if role in ("user", "human"):
                            text += f"<|user|>\n{content}\n"
                        elif role in ("assistant", "gpt"):
                            text += f"<|assistant|>\n{content}\n"
                elif "instruction" in item and "output" in item:
                    text = f"<|user|>\n{item['instruction']}\n<|assistant|>\n{item['output']}\n"
                elif "text" in item:
                    text = item["text"]
                else:
                    continue
                
                ids = tokenizer.encode(text).ids
                if len(ids) < 2:
                    continue
                ids = ids[:512]  # truncate
                
                input_t = torch.tensor([ids], dtype=torch.long, device=DEVICE)
                logits, _ = model(input_t)
                
                # Cross entropy on all tokens
                loss = F.cross_entropy(
                    logits[0, :-1].float().contiguous().view(-1, logits.shape[-1]),
                    input_t[0, 1:].contiguous().view(-1),
                    reduction="mean"
                )
                total_loss += loss.item()
                count += 1
        
        avg_loss = total_loss / max(count, 1)
        print(f"Val loss (100 samples): {avg_loss:.4f}")
    else:
        avg_loss = 2.2062  # from training
        print(f"val.jsonl not found, using training val_loss: {avg_loss}")
    
    # Write report
    report = f"""# SFT v2 체크포인트 종합 평가 보고서

체크포인트: `checkpoints/korean_1b_sft/checkpoint-best`
평가일시: 2026-02-27

## 핵심 지표

| 항목 | Pretrain | SFT v1 (buggy 포맷) | SFT v1 (올바른 포맷) | **SFT v2 (이번)** |
|------|----------|--------------------|--------------------|-----------------|
| 반복률 | 69.4% | 57.1% | 17.7% | **{avg_rep:.1f}%** |
| val_loss | - | 2.69 | - | **{avg_loss:.4f}** |
| 자연 종료율 | - | - | - | **{eos_rate:.1f}%** |

## 목표 달성 여부

- 반복률 <5%: {"✅ 달성" if avg_rep < 5 else "❌ 미달성"} ({avg_rep:.1f}%)
- val_loss <2.2: {"✅ 달성" if avg_loss < 2.2 else "❌ 미달성"} ({avg_loss:.4f})

## 생성 파라미터

- temperature=0.7, top_p=0.9
- repetition_penalty=1.1, no_repeat_ngram_size=3
- max_new_tokens=200
- 프롬프트 포맷: `<|user|>\\n{{질문}}\\n<|assistant|>\\n`

## 생성 샘플 전문

"""
    for i, r in enumerate(results):
        report += f"""### [{i+1}] {r['question']}
- 반복률: {r['repetition_rate']*100:.1f}%
- 종료: {"EOS" if r['stopped_eos'] else "max_tokens"}
- 토큰 수: {r['num_tokens']}

```
{r['answer']}
```

"""
    
    report += f"""## 개선도 분석

- Pretrain → SFT v2: {69.4 - avg_rep:.1f}%p 개선
- SFT v1 (buggy) → SFT v2: {57.1 - avg_rep:.1f}%p 개선
- SFT v1 (올바른 포맷) → SFT v2: {17.7 - avg_rep:.1f}%p 개선

## 권장 다음 단계

"""
    if avg_rep < 5:
        report += """- ✅ 반복률 목표 달성 - 1B SFT 기본 완료
- ORPO/DPO 선호도 학습으로 응답 품질 향상
- 3B 모델로 스케일업 고려
- 더 다양한 벤치마크 (KoBEST, KLUE 등) 평가
"""
    elif avg_rep < 15:
        report += """- 반복률이 아직 목표 미달이지만 상당히 개선됨
- 데이터 다양성 증가 (더 많은 SFT 데이터)
- repetition_penalty 조정 실험
- ORPO로 반복 패턴 추가 교정 가능
"""
    else:
        report += """- 반복률이 여전히 높음 - 추가 SFT 필요
- 학습 데이터 품질 점검
- 학습률/에포크 조정
- 데이터 증강 고려
"""
    
    with open(f"{OUTPUT_DIR}/report.md", "w") as f:
        f.write(report)
    
    # Save raw results
    with open(f"{OUTPUT_DIR}/results.json", "w") as f:
        json.dump({"results": results, "avg_rep": avg_rep, "eos_rate": eos_rate, "val_loss": avg_loss}, f, ensure_ascii=False, indent=2)
    
    print(f"\n보고서 저장: {OUTPUT_DIR}/report.md")

if __name__ == "__main__":
    main()
