#!/usr/bin/env python3
"""
test_ollama_repetition.py — Ollama 배포 모델 반복률 검증

ORPO eval과 동일한 프롬프트로 Ollama API 호출 후 n-gram 반복률 + EOS 종료율 측정.
목표: 3-gram rep < 3% (한국어 자연 반복 고려), EOS 종료율 > 95%

Usage:
    python scripts/test_ollama_repetition.py [--model frankenstallm-3b] [--host localhost:11434]
"""
import argparse
import json
import urllib.request
import urllib.error
import sys
from collections import Counter

# ORPO eval에서 사용한 15개 한국어 프롬프트
TEST_PROMPTS = [
    "대한민국의 수도는 어디인가요?",
    "인공지능이란 무엇인가요?",
    "한국의 전통 음식 중에서 김치에 대해 설명해주세요.",
    "프로그래밍을 배우려면 어떻게 해야 하나요?",
    "지구 온난화의 원인과 대책에 대해 설명해주세요.",
    "한국어의 특징을 3가지 설명해주세요.",
    "좋은 리더의 자질에 대해 논해주세요.",
    "우주 탐사의 의미와 중요성을 설명해주세요.",
    "건강한 생활 습관 5가지를 추천해주세요.",
    "인터넷이 현대 사회에 미친 영향을 분석해주세요.",
    "한국의 교육 제도의 장단점을 설명해주세요.",
    "환경 보호를 위해 개인이 할 수 있는 일을 알려주세요.",
    "4차 산업혁명이 일자리에 미치는 영향을 분석해주세요.",
    "독서의 중요성과 효과적인 독서 방법을 알려주세요.",
    "한국 문화의 세계화에 대해 어떻게 생각하시나요?",
]


def compute_ngram_repetition(text: str, n: int) -> float:
    """n-gram 반복률 계산 (0.0 ~ 1.0)"""
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    counts = Counter(ngrams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / len(ngrams)


def call_ollama(prompt: str, model: str, host: str, timeout: int = 120) -> dict:
    """Ollama API 호출"""
    url = f"http://{host}/api/generate"
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        return {"error": str(e), "response": ""}
    except Exception as e:
        return {"error": str(e), "response": ""}


def main():
    parser = argparse.ArgumentParser(description="Ollama 반복률 검증")
    parser.add_argument("--model", default="frankenstallm-3b", help="Ollama 모델 이름")
    parser.add_argument("--host", default="localhost:11434", help="Ollama 서버 주소")
    args = parser.parse_args()

    print(f"{'='*70}")
    print(f"  Ollama 반복률 검증: {args.model}")
    print(f"  서버: {args.host}")
    print(f"  프롬프트: {len(TEST_PROMPTS)}개")
    print(f"{'='*70}\n")

    results = []
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"[{i:2d}/{len(TEST_PROMPTS)}] {prompt[:40]}...")
        resp = call_ollama(prompt, args.model, args.host)

        if "error" in resp and resp["error"]:
            print(f"  ERROR: {resp['error']}")
            results.append({"prompt": prompt, "error": resp["error"]})
            continue

        text = resp.get("response", "")
        eos_done = resp.get("done", False)

        rep1 = compute_ngram_repetition(text, 1)
        rep2 = compute_ngram_repetition(text, 2)
        rep3 = compute_ngram_repetition(text, 3)
        rep4 = compute_ngram_repetition(text, 4)

        results.append({
            "prompt": prompt,
            "response_len": len(text),
            "word_count": len(text.split()),
            "eos_done": eos_done,
            "rep1": rep1, "rep2": rep2, "rep3": rep3, "rep4": rep4,
        })

        preview = text[:100].replace("\n", " ")
        print(f"  응답: {preview}...")
        print(f"  길이: {len(text)}자, EOS: {eos_done}, "
              f"rep(1/2/3/4): {rep1:.2%}/{rep2:.2%}/{rep3:.2%}/{rep4:.2%}")
        print()

    # --- Summary ---
    valid = [r for r in results if "error" not in r or not r.get("error")]
    if not valid:
        print("ERROR: 유효한 응답 없음")
        sys.exit(1)

    avg_rep3 = sum(r["rep3"] for r in valid) / len(valid)
    eos_rate = sum(1 for r in valid if r["eos_done"]) / len(valid)
    errors = len(results) - len(valid)

    print(f"{'='*70}")
    print(f"  결과 요약")
    print(f"{'='*70}")
    print(f"  유효 응답: {len(valid)}/{len(results)}  (에러: {errors})")
    print(f"  평균 3-gram 반복률: {avg_rep3:.2%}  (목표: < 3%)")
    print(f"  EOS 종료율:        {eos_rate:.0%}  (목표: > 95%)")
    print()

    # Pass/Fail
    # 한국어는 조사/접속사 자연 반복으로 어절 기준 3-gram rep 1.5~2%가 자연 floor
    # 퇴행적 반복(30%+)과 구별하여 3% 기준 적용
    rep_pass = avg_rep3 < 0.03
    eos_pass = eos_rate > 0.95
    overall = rep_pass and eos_pass

    print(f"  3-gram 반복률: {'PASS ✓' if rep_pass else 'FAIL ✗'}  ({avg_rep3:.2%})")
    print(f"  EOS 종료율:    {'PASS ✓' if eos_pass else 'FAIL ✗'}  ({eos_rate:.0%})")
    print(f"  종합:          {'PASS ✓' if overall else 'FAIL ✗'}")
    print(f"{'='*70}")

    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
