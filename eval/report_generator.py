"""
Markdown report generator for FRANKENSTALLM 3B evaluation pipeline.

Generates comprehensive evaluation reports with sections for:
- Perplexity metrics across datasets
- Calibration statistics
- Token NLL distribution
- Generation quality samples
- Repetition parameter search results
- Standard benchmark results (lm-eval) — Korean + English
- 0-shot vs 5-shot comparison
- Comparison with reference models
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
import logging

logger = logging.getLogger(__name__)


def _fmt_seconds(seconds: float) -> str:
    """Format seconds into a human-readable duration string."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


# =========================================================================
# Normalization helpers — map GPU-label keys to logical sections
# =========================================================================

def _normalize_phase1_results(raw: dict) -> dict:
    """Convert GPU-labelled phase1_results into logical sections.

    Returns dict with keys: perplexity, calibration, token_nll, generation, repetition.
    """
    normalized: Dict[str, Any] = {
        "perplexity": {},
        "calibration": {},
        "token_nll": {},
        "generation": {},
        "repetition": {},
    }

    for label, data in raw.items():
        if not isinstance(data, (dict, list)):
            continue

        if "PPL" in label:
            # PPL entries: single dict or list of dicts
            if isinstance(data, dict) and "ppl" in data:
                name = data.get("name", label)
                normalized["perplexity"][name] = data
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "ppl" in item:
                        name = item.get("name", f"unknown_{len(normalized['perplexity'])}")
                        normalized["perplexity"][name] = item
            elif isinstance(data, dict) and "error" in data:
                # Task failed — skip
                pass
        elif "Calibration" in label:
            if isinstance(data, dict):
                if "calibration" in data:
                    normalized["calibration"] = data["calibration"]
                if "token_nll" in data:
                    normalized["token_nll"] = data["token_nll"]
        elif "Generation" in label:
            if isinstance(data, dict):
                normalized["generation"] = data
        elif "Repetition" in label:
            if isinstance(data, dict):
                normalized["repetition"] = data

    return normalized


def _normalize_phase2_results(raw: dict) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Convert GPU-labelled phase2_results into flat task dicts for 0-shot and 5-shot.

    Returns (zero_shot_metrics, five_shot_metrics) where each is:
      {"kobest_boolq": {"acc,none": 0.50, ...}, "haerae": {...}, ...}
    """
    zero_shot: Dict[str, Any] = {}
    five_shot: Dict[str, Any] = {}

    for label, data in raw.items():
        if label == "5shot":
            # Recurse into 5-shot sub-dict
            if isinstance(data, dict):
                for sub_label, sub_data in data.items():
                    if isinstance(sub_data, dict) and "per_task_metrics" in sub_data:
                        for task_name, metrics in sub_data["per_task_metrics"].items():
                            five_shot[task_name] = metrics
            continue

        if isinstance(data, dict) and "per_task_metrics" in data:
            for task_name, metrics in data["per_task_metrics"].items():
                zero_shot[task_name] = metrics

    return zero_shot, five_shot


def _get_acc(metrics: dict, prefer_norm: bool = False) -> Optional[float]:
    """Extract accuracy from lm-eval metrics dict."""
    if prefer_norm and "acc_norm,none" in metrics:
        val = metrics["acc_norm,none"]
        if isinstance(val, (int, float)):
            return float(val)
    if "acc,none" in metrics:
        val = metrics["acc,none"]
        if isinstance(val, (int, float)):
            return float(val)
    return None


def _fmt_pct(val: Optional[float]) -> str:
    """Format as percentage string or N/A."""
    if val is None:
        return "N/A"
    return f"{val * 100:.2f}%"


def _fmt_f(val, decimals: int = 4) -> str:
    """Format float or return N/A."""
    if isinstance(val, (int, float)):
        return f"{val:.{decimals}f}"
    return str(val) if val is not None else "N/A"


# =========================================================================
# Main report generator
# =========================================================================

def generate_report(
    phase1_results: dict,
    phase2_results: dict,
    generation_samples: list,
    output_dir: Path,
    checkpoint_name: str = "checkpoint-0057000",
    total_elapsed_sec: float = 0.0,
) -> str:
    """Generate a comprehensive markdown evaluation report.

    Handles the GPU-labelled key structure from full_eval_pipeline.py
    and produces multiple report files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Normalize data
    p1 = _normalize_phase1_results(phase1_results)
    zero_shot, five_shot = _normalize_phase2_results(phase2_results)

    eval_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ===== Generate individual reports =====
    ppl_report = _generate_perplexity_report(p1["perplexity"])
    cal_report = _generate_calibration_report(p1["calibration"], p1["token_nll"])
    gen_report = _generate_generation_report(p1["generation"], generation_samples)
    bench_report = _generate_benchmark_report(zero_shot, five_shot, p1["repetition"])
    exec_summary = _generate_executive_summary(
        p1, zero_shot, five_shot, checkpoint_name, eval_datetime, total_elapsed_sec,
    )

    # Write individual reports
    (reports_dir / "00_executive_summary.md").write_text(exec_summary, encoding="utf-8")
    (reports_dir / "01_perplexity_report.md").write_text(ppl_report, encoding="utf-8")
    (reports_dir / "02_calibration_report.md").write_text(cal_report, encoding="utf-8")
    (reports_dir / "03_generation_quality.md").write_text(gen_report, encoding="utf-8")
    (reports_dir / "04_benchmark_report.md").write_text(bench_report, encoding="utf-8")

    # Combined full report
    full_report = "\n\n---\n\n".join([
        exec_summary, ppl_report, cal_report, gen_report, bench_report,
    ])

    report_path = output_dir / "full_eval_report.md"
    report_path.write_text(full_report, encoding="utf-8")

    return full_report


# =========================================================================
# Individual report sections
# =========================================================================

def _generate_executive_summary(
    p1: dict,
    zero_shot: dict,
    five_shot: dict,
    checkpoint_name: str,
    eval_datetime: str,
    total_elapsed_sec: float,
) -> str:
    lines = [
        "# FRANKENSTALLM 3B 종합 평가 리포트\n",
        f"- **모델**: FRANKENSTALLM 3B",
        f"- **체크포인트**: {checkpoint_name}",
        f"- **평가 일시**: {eval_datetime}",
        f"- **총 소요 시간**: {total_elapsed_sec:.1f}초\n",
        "## Executive Summary\n",
    ]

    # Main PPL
    main_ppl = "N/A"
    ppl_data = p1.get("perplexity", {})
    for name in ["3b", "3b_val"]:
        if name in ppl_data and isinstance(ppl_data[name], dict):
            main_ppl = _fmt_f(ppl_data[name].get("ppl"))
            break

    # KoBEST average
    kobest_tasks = ["kobest_boolq", "kobest_copa", "kobest_hellaswag",
                    "kobest_sentineg", "kobest_wic"]
    kobest_accs = []
    for t in kobest_tasks:
        if t in zero_shot:
            a = _get_acc(zero_shot[t])
            if a is not None:
                kobest_accs.append(a)
    kobest_avg = _fmt_pct(sum(kobest_accs) / len(kobest_accs)) if kobest_accs else "N/A"

    # MMLU-KO — prefer group-level weighted average from lm-eval
    mmlu_ko_avg = "N/A"
    mmlu_ko_count = 0
    if "global_mmlu_ko" in zero_shot:
        a = _get_acc(zero_shot["global_mmlu_ko"])
        if a is not None:
            mmlu_ko_avg = _fmt_pct(a)
            # Count subtasks for display
            mmlu_ko_count = sum(
                1 for t in zero_shot
                if t.startswith("global_mmlu_ko_") and _get_acc(zero_shot[t]) is not None
            )
            if mmlu_ko_count == 0:
                mmlu_ko_count = 1  # group-level only
    else:
        # Fallback: average subtask-level metrics
        mmlu_ko_accs = []
        for t, m in zero_shot.items():
            if t.startswith("global_mmlu_ko_"):
                a = _get_acc(m)
                if a is not None:
                    mmlu_ko_accs.append(a)
        if mmlu_ko_accs:
            mmlu_ko_avg = _fmt_pct(sum(mmlu_ko_accs) / len(mmlu_ko_accs))
            mmlu_ko_count = len(mmlu_ko_accs)

    # MMLU-EN — exclude group-level keys to avoid double-counting
    _MMLU_EN_GROUPS = {"mmlu", "mmlu_humanities", "mmlu_social_sciences", "mmlu_stem", "mmlu_other"}
    mmlu_en_accs = []
    for t, m in zero_shot.items():
        if (t.startswith("mmlu_") or t == "mmlu") and t not in _MMLU_EN_GROUPS:
            a = _get_acc(m)
            if a is not None:
                mmlu_en_accs.append(a)
    if not mmlu_en_accs:
        # Fallback to group-level if no subtasks
        for t in _MMLU_EN_GROUPS:
            if t in zero_shot:
                a = _get_acc(zero_shot[t])
                if a is not None:
                    mmlu_en_accs.append(a)
    mmlu_en_avg = _fmt_pct(sum(mmlu_en_accs) / len(mmlu_en_accs)) if mmlu_en_accs else "N/A"

    # HAE-RAE
    haerae_acc = "N/A"
    if "haerae" in zero_shot:
        a = _get_acc(zero_shot["haerae"])
        if a is not None:
            haerae_acc = _fmt_pct(a)

    # English benchmarks
    en_benchmarks = {}
    for t in ["hellaswag", "arc_easy", "arc_challenge", "winogrande", "piqa"]:
        if t in zero_shot:
            a = _get_acc(zero_shot[t], prefer_norm=(t in ["hellaswag", "arc_challenge"]))
            if a is not None:
                en_benchmarks[t] = a

    # Top-1 accuracy
    top1 = _fmt_f(p1.get("calibration", {}).get("top1_accuracy"))

    lines.append("| 메트릭 | 값 |")
    lines.append("|--------|-----|")
    lines.append(f"| 주요 PPL (3b_val) | {main_ppl} |")
    lines.append(f"| MMLU-KO 평균 ({mmlu_ko_count}과목) | {mmlu_ko_avg} |")
    lines.append(f"| MMLU-EN 평균 | {mmlu_en_avg} |")
    lines.append(f"| KoBEST 평균 ({len(kobest_accs)}태스크) | {kobest_avg} |")
    lines.append(f"| HAE-RAE | {haerae_acc} |")
    for t, a in en_benchmarks.items():
        lines.append(f"| {t} (0-shot) | {_fmt_pct(a)} |")
    lines.append(f"| Top-1 정확도 (Calibration) | {top1} |")
    lines.append("")

    # Reference comparison
    lines.append("## 참고 모델 비교\n")
    lines.append("| 모델 | 파라미터 | MMLU-KO | MMLU-EN | KoBEST 평균 | PPL |")
    lines.append("|------|---------|---------|---------|------------|-----|")
    lines.append(f"| **FRANKENSTALLM 3B** | 3B | {mmlu_ko_avg} | {mmlu_en_avg} | {kobest_avg} | {main_ppl} |")
    lines.append("| Llama-3.2-3B | 3B | ~42% | ~58% | ~55% | — |")
    lines.append("| Qwen2.5-3B | 3B | ~48% | ~65% | ~60% | — |")
    lines.append("| EXAONE-3.5-2.4B | 2.4B | ~35% | ~50% | ~50% | — |")
    lines.append("")

    return "\n".join(lines)


def _generate_perplexity_report(ppl_data: dict) -> str:
    lines = ["# Perplexity 평가\n"]

    if not ppl_data:
        lines.append("데이터 없음\n")
        return "\n".join(lines)

    rows = []
    for name, metrics in ppl_data.items():
        if isinstance(metrics, dict) and "ppl" in metrics:
            rows.append({
                "name": name,
                "ppl": metrics.get("ppl"),
                "bits": metrics.get("bits_per_token"),
                "n_tokens": metrics.get("n_tokens"),
                "n_eval": metrics.get("n_eval_tokens"),
                "elapsed": metrics.get("elapsed_sec"),
            })

    rows.sort(key=lambda x: x["ppl"] if isinstance(x["ppl"], (int, float)) else float("inf"),
              reverse=True)

    lines.append("| 데이터셋 | PPL | Bits/Token | 전체 토큰 | 평가 토큰 | 소요 시간 |")
    lines.append("|---------|-----|-----------|---------|---------|---------|")
    for r in rows:
        lines.append(
            f"| {r['name']} | {_fmt_f(r['ppl'])} | {_fmt_f(r['bits'])} | "
            f"{r['n_tokens']:,} | {r['n_eval']:,} | {_fmt_f(r['elapsed'], 1)}s |"
            if isinstance(r['n_tokens'], (int, float)) and isinstance(r['n_eval'], (int, float))
            else f"| {r['name']} | {_fmt_f(r['ppl'])} | {_fmt_f(r['bits'])} | "
                 f"{r['n_tokens']} | {r['n_eval']} | {_fmt_f(r['elapsed'], 1)}s |"
        )
    lines.append("")
    return "\n".join(lines)


def _generate_calibration_report(cal_data: dict, nll_data: dict) -> str:
    lines = ["# Calibration 및 Token NLL 분석\n"]

    # Calibration
    lines.append("## Calibration 결과\n")
    if cal_data:
        lines.append("| 메트릭 | 값 |")
        lines.append("|--------|-----|")
        metrics_map = {
            "top1_accuracy": "Top-1 Accuracy",
            "top5_accuracy": "Top-5 Accuracy",
            "top10_accuracy": "Top-10 Accuracy",
            "mean_correct_prob": "Mean Correct Prob",
            "mean_entropy": "Mean Entropy",
        }
        for key, label in metrics_map.items():
            lines.append(f"| {label} | {_fmt_f(cal_data.get(key))} |")
        lines.append("")
    else:
        lines.append("데이터 없음\n")

    # Token NLL
    lines.append("## Token NLL 분포\n")
    if nll_data:
        # Keys may be "mean"/"std" or "nll_mean"/"nll_std"
        stats_map = [
            (["nll_mean", "mean"], "평균"),
            (["nll_std", "std"], "표준편차"),
            (["nll_median", "median"], "중앙값"),
            (["nll_min", "min"], "최솟값"),
            (["nll_max", "max"], "최댓값"),
        ]
        lines.append("| 통계 | 값 |")
        lines.append("|------|-----|")
        for candidates, label in stats_map:
            val = None
            for c in candidates:
                if c in nll_data:
                    val = nll_data[c]
                    break
            lines.append(f"| {label} | {_fmt_f(val)} |")
        lines.append("")

        # Percentiles: "nll_percentiles" (dict) or "percentiles" (dict)
        pct_data = nll_data.get("nll_percentiles", nll_data.get("percentiles"))
        if pct_data and isinstance(pct_data, dict):
            lines.append("### Percentiles\n")
            lines.append("| Percentile | 값 |")
            lines.append("|------------|-----|")
            for pct, value in pct_data.items():
                lines.append(f"| {pct}th | {_fmt_f(value)} |")
            lines.append("")

        # High loss: "high_loss_fractions" (dict) or flat "high_loss_fraction_N" keys
        hlf = nll_data.get("high_loss_fractions")
        if hlf and isinstance(hlf, dict):
            lines.append("### 고손실 토큰 비율\n")
            lines.append("| 임계값 | 비율 |")
            lines.append("|--------|-----|")
            for threshold, fraction in hlf.items():
                lines.append(f"| NLL > {threshold} | {_fmt_f(fraction)} |")
            lines.append("")
        else:
            # Check flat keys: high_loss_fraction_5, high_loss_fraction_10, ...
            hlf_flat = {k.replace("high_loss_fraction_", ""): v
                        for k, v in nll_data.items()
                        if k.startswith("high_loss_fraction_")}
            if hlf_flat:
                lines.append("### 고손실 토큰 비율\n")
                lines.append("| 임계값 | 비율 |")
                lines.append("|--------|-----|")
                for threshold, fraction in sorted(hlf_flat.items()):
                    lines.append(f"| NLL > {threshold} | {_fmt_f(fraction)} |")
                lines.append("")
    else:
        lines.append("데이터 없음\n")

    return "\n".join(lines)


def _generate_generation_report(gen_data: dict, samples: list) -> str:
    lines = ["# 생성 품질 분석\n"]

    if gen_data and "summary" in gen_data:
        lines.append("## 요약 통계\n")
        lines.append("| 메트릭 | 값 |")
        lines.append("|--------|-----|")
        for key, value in gen_data["summary"].items():
            display = key.replace("_", " ").title()
            lines.append(f"| {display} | {_fmt_f(value)} |")
        lines.append("")

    if samples:
        lines.append("## 생성 샘플 (Greedy)\n")
        for i, sample in enumerate(samples[:5], 1):
            if isinstance(sample, dict):
                prompt = sample.get("prompt", "")
                generated = sample.get("generated_text", "")
                if len(generated) > 300:
                    generated = generated[:300] + "..."
                lines.append(f"### 샘플 {i}\n")
                lines.append(f"**Prompt**: {prompt}\n")
                lines.append(f"**Generated**: {generated}\n")
        lines.append("")
    elif not gen_data:
        lines.append("데이터 없음\n")

    return "\n".join(lines)


def _generate_benchmark_report(
    zero_shot: dict,
    five_shot: dict,
    repetition: dict,
) -> str:
    lines = ["# 표준 벤치마크 결과\n"]

    if not zero_shot and not five_shot:
        lines.append("데이터 없음\n")
        return "\n".join(lines)

    # --- Korean Benchmarks ---
    lines.append("## 한국어 벤치마크\n")

    # KoBEST
    kobest_names = ["kobest_boolq", "kobest_copa", "kobest_hellaswag",
                    "kobest_sentineg", "kobest_wic"]
    kobest_0 = {t: zero_shot[t] for t in kobest_names if t in zero_shot}
    if kobest_0:
        lines.append("### KoBEST (0-shot)\n")
        lines.append("| 태스크 | Accuracy | F1 |")
        lines.append("|--------|----------|-----|")
        for t in kobest_names:
            if t in kobest_0:
                m = kobest_0[t]
                acc = _fmt_pct(_get_acc(m))
                f1 = _fmt_f(m.get("f1,none"))
                lines.append(f"| {t} | {acc} | {f1} |")
        kobest_accs = [_get_acc(kobest_0[t]) for t in kobest_names
                       if t in kobest_0 and _get_acc(kobest_0[t]) is not None]
        if kobest_accs:
            lines.append(f"| **평균** | **{_fmt_pct(sum(kobest_accs)/len(kobest_accs))}** | |")
        lines.append("")

    # HAE-RAE
    if "haerae" in zero_shot:
        lines.append("### HAE-RAE (0-shot)\n")
        m = zero_shot["haerae"]
        lines.append(f"- Accuracy: {_fmt_pct(_get_acc(m))}")
        # Check for sub-tasks
        haerae_subs = {t: zero_shot[t] for t in zero_shot if t.startswith("haerae_") and t != "haerae"}
        if haerae_subs:
            lines.append("\n| 서브태스크 | Accuracy |")
            lines.append("|-----------|----------|")
            for t, sm in sorted(haerae_subs.items()):
                lines.append(f"| {t} | {_fmt_pct(_get_acc(sm))} |")
        lines.append("")

    # MMLU-KO
    mmlu_ko_tasks = {t: zero_shot[t] for t in zero_shot
                     if t.startswith("global_mmlu_ko") and t != "global_mmlu_ko"}
    if mmlu_ko_tasks or "global_mmlu_ko" in zero_shot:
        lines.append("### MMLU-KO (0-shot)\n")
        if mmlu_ko_tasks:
            lines.append(f"평가된 과목 수: **{len(mmlu_ko_tasks)}**\n")
            accs = [(t, _get_acc(m)) for t, m in sorted(mmlu_ko_tasks.items())
                    if _get_acc(m) is not None]
            if accs:
                # Prefer group-level weighted average from lm-eval
                group_acc = _get_acc(zero_shot["global_mmlu_ko"]) if "global_mmlu_ko" in zero_shot else None
                avg_acc = group_acc if group_acc is not None else sum(a for _, a in accs) / len(accs)
                lines.append(f"전체 평균: **{_fmt_pct(avg_acc)}**\n")

                # Top 10
                accs_sorted = sorted(accs, key=lambda x: x[1], reverse=True)
                lines.append("**상위 10개 과목**:\n")
                lines.append("| 과목 | Accuracy |")
                lines.append("|------|----------|")
                for t, a in accs_sorted[:10]:
                    subject = t.replace("global_mmlu_ko_", "")
                    lines.append(f"| {subject} | {_fmt_pct(a)} |")
                lines.append("")

                lines.append("**하위 10개 과목**:\n")
                lines.append("| 과목 | Accuracy |")
                lines.append("|------|----------|")
                for t, a in accs_sorted[-10:]:
                    subject = t.replace("global_mmlu_ko_", "")
                    lines.append(f"| {subject} | {_fmt_pct(a)} |")
                lines.append("")
        elif "global_mmlu_ko" in zero_shot:
            a = _get_acc(zero_shot["global_mmlu_ko"])
            lines.append(f"전체 정확도: {_fmt_pct(a)}\n")

    # --- English Benchmarks ---
    lines.append("## 영어 벤치마크\n")

    en_tasks = ["hellaswag", "arc_easy", "arc_challenge", "winogrande", "piqa"]
    en_found = {t: zero_shot[t] for t in en_tasks if t in zero_shot}
    if en_found:
        lines.append("### 주요 벤치마크 (0-shot)\n")
        lines.append("| 태스크 | Accuracy | Acc (norm) |")
        lines.append("|--------|----------|-----------|")
        for t in en_tasks:
            if t in en_found:
                m = en_found[t]
                acc = _fmt_pct(_get_acc(m))
                acc_norm = _fmt_pct(_get_acc(m, prefer_norm=True) if "acc_norm,none" in m else None)
                lines.append(f"| {t} | {acc} | {acc_norm} |")
        lines.append("")

    # MMLU-EN
    mmlu_en_tasks = {t: zero_shot[t] for t in zero_shot
                     if (t.startswith("mmlu_") or t == "mmlu") and not t.startswith("mmlu_ko")}
    if mmlu_en_tasks:
        lines.append("### MMLU-EN (0-shot)\n")
        # Filter out the group-level "mmlu" if sub-tasks exist
        subtasks = {t: m for t, m in mmlu_en_tasks.items() if t != "mmlu"}
        if subtasks:
            lines.append(f"평가된 과목 수: **{len(subtasks)}**\n")
            accs = [(t, _get_acc(m)) for t, m in sorted(subtasks.items())
                    if _get_acc(m) is not None]
            if accs:
                avg_acc = sum(a for _, a in accs) / len(accs)
                lines.append(f"전체 평균: **{_fmt_pct(avg_acc)}**\n")

                accs_sorted = sorted(accs, key=lambda x: x[1], reverse=True)
                lines.append("**상위 10개 과목**:\n")
                lines.append("| 과목 | Accuracy |")
                lines.append("|------|----------|")
                for t, a in accs_sorted[:10]:
                    subject = t.replace("mmlu_", "")
                    lines.append(f"| {subject} | {_fmt_pct(a)} |")
                lines.append("")

                lines.append("**하위 10개 과목**:\n")
                lines.append("| 과목 | Accuracy |")
                lines.append("|------|----------|")
                for t, a in accs_sorted[-10:]:
                    subject = t.replace("mmlu_", "")
                    lines.append(f"| {subject} | {_fmt_pct(a)} |")
                lines.append("")
        elif "mmlu" in mmlu_en_tasks:
            a = _get_acc(mmlu_en_tasks["mmlu"])
            lines.append(f"전체 정확도: {_fmt_pct(a)}\n")

    # --- 0-shot vs 5-shot Comparison ---
    if five_shot:
        lines.append("## 0-shot vs 5-shot 비교\n")

        # Collect all tasks that have both 0-shot and 5-shot results
        common_tasks = sorted(set(zero_shot.keys()) & set(five_shot.keys()))
        if common_tasks:
            lines.append("| 태스크 | 0-shot Acc | 5-shot Acc | 변화 |")
            lines.append("|--------|-----------|-----------|------|")
            for t in common_tasks:
                a0 = _get_acc(zero_shot[t])
                a5 = _get_acc(five_shot[t])
                if a0 is not None and a5 is not None:
                    diff = a5 - a0
                    sign = "+" if diff >= 0 else ""
                    lines.append(
                        f"| {t} | {_fmt_pct(a0)} | {_fmt_pct(a5)} | {sign}{diff*100:.2f}pp |"
                    )
                else:
                    lines.append(f"| {t} | {_fmt_pct(a0)} | {_fmt_pct(a5)} | — |")
            lines.append("")

            # Summary
            diffs = []
            for t in common_tasks:
                a0 = _get_acc(zero_shot[t])
                a5 = _get_acc(five_shot[t])
                if a0 is not None and a5 is not None:
                    diffs.append(a5 - a0)
            if diffs:
                avg_diff = sum(diffs) / len(diffs)
                improved = sum(1 for d in diffs if d > 0)
                degraded = sum(1 for d in diffs if d < 0)
                lines.append(
                    f"평균 변화: {'+' if avg_diff >= 0 else ''}{avg_diff*100:.2f}pp | "
                    f"개선: {improved} | 하락: {degraded} | 동일: {len(diffs) - improved - degraded}\n"
                )

    # --- Repetition ---
    if repetition and repetition.get("grid_results"):
        lines.append("## Repetition 파라미터 검색\n")
        rep_data = repetition["grid_results"]
        rep_rows = []
        # grid_results can be a list of dicts or a dict of dicts
        items = rep_data.items() if isinstance(rep_data, dict) else enumerate(rep_data)
        for key, metrics in items:
            if isinstance(metrics, dict):
                rep_rows.append({
                    "config": metrics.get("params", str(key)),
                    "temp": metrics.get("temperature"),
                    "rep_pen": metrics.get("repetition_penalty"),
                    "3gram": metrics.get("avg_3gram_rep", metrics.get("3gram_repetition", float("inf"))),
                    "4gram": metrics.get("avg_4gram_rep", metrics.get("4gram_repetition")),
                    "eos_rate": metrics.get("eos_rate"),
                    "avg_tokens": metrics.get("avg_tokens"),
                })
        rep_rows.sort(key=lambda x: x["3gram"] if isinstance(x["3gram"], (int, float)) else float("inf"))

        lines.append("| 설정 | Temp | Rep Pen | 3-gram | 4-gram | EOS Rate | Avg Tokens |")
        lines.append("|------|------|---------|--------|--------|----------|-----------|")
        for i, r in enumerate(rep_rows):
            marker = " **← best**" if i == 0 else ""
            lines.append(
                f"| {r['config']} | {_fmt_f(r['temp'], 2)} | {_fmt_f(r['rep_pen'], 2)} | "
                f"{_fmt_f(r['3gram'])} | {_fmt_f(r['4gram'])} | "
                f"{_fmt_f(r['eos_rate'])} | {_fmt_f(r['avg_tokens'], 1)} |{marker}"
            )
        lines.append("")

    lines.append("---\n")
    lines.append("*이 리포트는 자동으로 생성되었습니다.*")
    return "\n".join(lines)


# =========================================================================
# Base vs SFT Comparison Report
# =========================================================================

# Base model reference values (from 3b_reeval_20260305_1451)
_BASE_PPL_REFERENCE = {
    "3b_val": 5.2263,
    "3b": 5.2263,
    "korean_c4_val": 5.7173,
    "korean_c4": 5.7173,
    "hplt_ko_val": 2.4028,
    "hplt_ko": 2.4028,
    "cc100_ko_val": 21.782,
    "cc100_ko": 21.782,
    "korean_val": 9.6505,
    "korean": 9.6505,
}

_BASE_BENCH_REFERENCE = {
    "kobest_boolq": 0.5028,
    "kobest_copa": 0.4930,
    "kobest_hellaswag": 0.2160,
    "kobest_sentineg": 0.4861,
    "kobest_wic": 0.4865,
    "haerae": 0.1971,
    "global_mmlu_ko": 0.2275,
    "hellaswag": 0.2600,
    "arc_easy": 0.2563,
    "arc_challenge": 0.2167,
    "winogrande": 0.5059,
    "piqa": 0.5250,
}

_BASE_GEN_REFERENCE = {
    "greedy_3gram_rep": 0.6099,
    "greedy_4gram_rep": 0.5702,
    "greedy_eos_rate": 0.0,
}

_BASE_CALIB_REFERENCE = {
    "top1_accuracy": 0.6875,
    "top5_accuracy": 0.8164,
    "top10_accuracy": 0.8593,
    "mean_entropy": 1.5682,
}

_BASE_NLL_REFERENCE = {
    "nll_mean": 1.5561,
    "high_loss_fraction_5": 0.1086,
}

# =========================================================================
# Threshold Justification
# =========================================================================
# PPL forgetting 15%: Kirkpatrick et al. (2017) continual learning 기준 10-20%
# KoBEST avg 55%: Random baseline ~40%, Llama 3.2 1B ~52%, Qwen 2.5 3B ~58%
# MMLU-KO 30%: Random 25%, Llama 3.2 3B ~35%
# Greedy 3-gram rep <5%: 인간 한국어 텍스트 256토큰 기준 1-3%, Base 모델 61%
# EOS rate >90%: 챗 모델은 응답을 끝내야 함, 일부 장문 허용
# Calibration top1 65%: Base 68.75%, SFT로 인한 소폭 하락 허용
# Distinct-2 >70%: Li et al. (2016), 다양성 보장 최소선
# =========================================================================

_SFT_TARGETS = {
    # 생성 품질
    "greedy_3gram_rep_max": 0.05,
    "eos_rate_min": 0.90,
    "sampled_eos_min": 0.50,
    "distinct_2_min": 0.70,
    # 지식 보존
    "ppl_forgetting_max_pct": 15.0,
    # 한국어 벤치마크
    "kobest_avg_min": 0.55,
    "haerae_min": 0.25,
    "mmlu_ko_min": 0.30,
    # 칼리브레이션
    "top1_accuracy_min": 0.65,
    # 영어 유지
    "hellaswag_min": 0.25,
    "arc_easy_min": 0.25,
    "arc_challenge_min": 0.21,
    "winogrande_min": 0.49,
    "piqa_min": 0.51,
    "mmlu_en_avg_min": 0.25,
}

_REFERENCE_MODELS = {
    "Llama 3.2 1B":  {"kobest_avg": 0.52, "mmlu_ko": 0.28, "mmlu_en": 0.32},
    "Llama 3.2 3B":  {"kobest_avg": 0.56, "mmlu_ko": 0.35, "mmlu_en": 0.55},
    "Qwen 2.5 3B":   {"kobest_avg": 0.58, "mmlu_ko": 0.42, "mmlu_en": 0.58},
}


def _compute_orpo_score(sft_p1, sft_zero, base_p1, base_zero):
    """ORPO 필요성 정량 판정 (0-100점).

    Returns:
        dict with keys: total_score, dimension_scores, decision, confidence, orpo_gain_estimate
    """
    dimensions = {}
    missing = 0
    total_dims = 7

    # Dim 1: PPL Forgetting (25 pts)
    max_forgetting = _get_max_forgetting(sft_p1, base_p1)
    if max_forgetting is not None:
        threshold = _SFT_TARGETS["ppl_forgetting_max_pct"]
        score = 25 * max(0, 1 - max_forgetting / threshold)
        dimensions["ppl_forgetting"] = {
            "score": round(score, 1), "weight": 25,
            "current": round(max_forgetting, 1), "threshold": f"<{threshold}%",
            "status": "PASS" if max_forgetting < threshold else "FAIL",
        }
    else:
        missing += 1
        dimensions["ppl_forgetting"] = {"score": 0, "weight": 25, "current": "N/A", "threshold": "<15%", "status": "N/A"}

    # Dim 2: Greedy 반복률 (20 pts)
    rep_rate = _get_greedy_3gram_rep(sft_p1)
    if rep_rate is not None:
        threshold = _SFT_TARGETS["greedy_3gram_rep_max"]
        score = 20 * max(0, 1 - rep_rate / threshold)
        dimensions["greedy_rep"] = {
            "score": round(score, 1), "weight": 20,
            "current": f"{rep_rate:.1%}", "threshold": f"<{threshold:.0%}",
            "status": "PASS" if rep_rate < threshold else "FAIL",
        }
    else:
        missing += 1
        dimensions["greedy_rep"] = {"score": 0, "weight": 20, "current": "N/A", "threshold": "<5%", "status": "N/A"}

    # Dim 3: EOS 종료율 (10 pts)
    eos_rate = sft_p1.get("generation", {}).get("summary", {}).get("greedy_eos_rate")
    if eos_rate is not None:
        threshold = _SFT_TARGETS["eos_rate_min"]
        score = 10 * min(eos_rate / threshold, 1)
        dimensions["eos_rate"] = {
            "score": round(score, 1), "weight": 10,
            "current": f"{eos_rate:.0%}", "threshold": f">{threshold:.0%}",
            "status": "PASS" if eos_rate >= threshold else "FAIL",
        }
    else:
        missing += 1
        dimensions["eos_rate"] = {"score": 0, "weight": 10, "current": "N/A", "threshold": ">90%", "status": "N/A"}

    # Dim 4: KoBEST 평균 (20 pts)
    kobest_avg = _get_kobest_avg(sft_zero)
    if kobest_avg is not None:
        threshold = _SFT_TARGETS["kobest_avg_min"]
        score = 20 * min(kobest_avg / threshold, 1)
        dimensions["kobest_avg"] = {
            "score": round(score, 1), "weight": 20,
            "current": f"{kobest_avg:.1%}", "threshold": f">{threshold:.0%}",
            "status": "PASS" if kobest_avg >= threshold else "FAIL",
        }
    else:
        missing += 1
        dimensions["kobest_avg"] = {"score": 0, "weight": 20, "current": "N/A", "threshold": ">55%", "status": "N/A"}

    # Dim 5: Calibration (10 pts)
    top1 = sft_p1.get("calibration", {}).get("top1_accuracy")
    if top1 is not None:
        threshold = _SFT_TARGETS["top1_accuracy_min"]
        score = 10 * min(top1 / threshold, 1)
        dimensions["calibration"] = {
            "score": round(score, 1), "weight": 10,
            "current": f"{top1:.1%}", "threshold": f">={threshold:.0%}",
            "status": "PASS" if top1 >= threshold else "FAIL",
        }
    else:
        missing += 1
        dimensions["calibration"] = {"score": 0, "weight": 10, "current": "N/A", "threshold": ">=65%", "status": "N/A"}

    # Dim 6: 다양성 distinct-2 (10 pts)
    distinct_2 = sft_p1.get("generation", {}).get("summary", {}).get("greedy_avg_distinct_2")
    if distinct_2 is not None:
        threshold = _SFT_TARGETS["distinct_2_min"]
        score = 10 * min(distinct_2 / threshold, 1)
        dimensions["diversity"] = {
            "score": round(score, 1), "weight": 10,
            "current": f"{distinct_2:.0%}", "threshold": f">{threshold:.0%}",
            "status": "PASS" if distinct_2 >= threshold else "FAIL",
        }
    else:
        missing += 1
        dimensions["diversity"] = {"score": 0, "weight": 10, "current": "N/A", "threshold": ">70%", "status": "N/A"}

    # Dim 7: 영어 유지 (5 pts)
    en_tasks = {
        "hellaswag": _SFT_TARGETS["hellaswag_min"],
        "arc_easy": _SFT_TARGETS["arc_easy_min"],
        "arc_challenge": _SFT_TARGETS["arc_challenge_min"],
        "winogrande": _SFT_TARGETS["winogrande_min"],
        "piqa": _SFT_TARGETS["piqa_min"],
    }
    en_all_pass = True
    en_count = 0
    for t, threshold in en_tasks.items():
        a = _get_acc(sft_zero.get(t, {})) if t in sft_zero else None
        if a is not None:
            en_count += 1
            if a < threshold:
                en_all_pass = False
    if en_count > 0:
        score = 5.0 if en_all_pass else 0.0
        dimensions["english"] = {
            "score": score, "weight": 5,
            "current": "전부 통과" if en_all_pass else "일부 미달",
            "threshold": "—", "status": "PASS" if en_all_pass else "FAIL",
        }
    else:
        missing += 1
        dimensions["english"] = {"score": 0, "weight": 5, "current": "N/A", "threshold": "—", "status": "N/A"}

    total_score = sum(d["score"] for d in dimensions.values())
    confidence = round(1.0 - (missing / total_dims), 2)

    if missing >= 2:
        logger.warning("ORPO score has %d/%d missing dimensions — confidence %.0f%%", missing, total_dims, confidence * 100)

    # ORPO gain estimate: dimensions that ORPO can improve
    orpo_improvable = 0.0
    if rep_rate is not None and rep_rate >= _SFT_TARGETS["greedy_3gram_rep_max"]:
        orpo_improvable += 20.0  # repetition
    if eos_rate is not None and eos_rate < _SFT_TARGETS["eos_rate_min"]:
        orpo_improvable += 10.0  # eos
    if distinct_2 is not None and distinct_2 < _SFT_TARGETS["distinct_2_min"]:
        orpo_improvable += 5.0  # partial diversity improvement

    # Decision
    forgetting_ok = max_forgetting is not None and max_forgetting < _SFT_TARGETS["ppl_forgetting_max_pct"]
    if total_score >= 80:
        decision = "DEPLOY"
    elif total_score >= 40 and forgetting_ok:
        decision = "ORPO"
    else:
        decision = "SFT_RETRY"

    return {
        "total_score": round(total_score, 1),
        "dimensions": dimensions,
        "decision": decision,
        "confidence": confidence,
        "orpo_gain_estimate": round(orpo_improvable, 1),
    }


def generate_comparison_report(
    base_results_dir: Path,
    sft_phase1_results: dict,
    sft_phase2_results: dict,
    output_path: Path,
    sft_output_dir: Optional[Path] = None,
    total_elapsed_sec: float = 0.0,
) -> Path:
    """Generate a comprehensive Base vs SFT comparison report.

    Args:
        base_results_dir: Directory containing Base model's phase1/phase2_results.json
        sft_phase1_results: SFT Phase 1 results dict
        sft_phase2_results: SFT Phase 2 results dict
        output_path: Where to write the markdown report
        sft_output_dir: SFT eval outputs directory (for linking)
        total_elapsed_sec: Total pipeline elapsed time

    Returns:
        Path to the generated report
    """
    base_results_dir = Path(base_results_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load Base results
    base_p1 = {}
    base_p2 = {}
    p1_file = base_results_dir / "phase1_results.json"
    p2_file = base_results_dir / "phase2_results.json"
    if p1_file.exists():
        with open(p1_file, encoding="utf-8") as f:
            base_p1 = json.load(f)
    if p2_file.exists():
        with open(p2_file, encoding="utf-8") as f:
            base_p2 = json.load(f)

    # Normalize both
    sft_p1 = _normalize_phase1_results(sft_phase1_results)
    base_p1_norm = _normalize_phase1_results(base_p1)
    sft_zero, sft_five = _normalize_phase2_results(sft_phase2_results)
    base_zero, base_five = _normalize_phase2_results(base_p2)

    eval_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []

    # === Header ===
    lines.append("# FRANKENSTALLM 3B SFT 모델 다면적 종합 평가 보고서\n")
    lines.append(f"- **평가 일시**: {eval_datetime}")
    lines.append(f"- **SFT 체크포인트**: checkpoint-best (val_loss=1.8851, step 25500)")
    lines.append(f"- **Base 참조 결과**: 3b_reeval_20260305_1451")
    lines.append(f"- **총 소요 시간**: {_fmt_seconds(total_elapsed_sec)}")
    if sft_output_dir:
        lines.append(f"- **결과 디렉토리**: {sft_output_dir}")
    lines.append("")

    # === 1. Executive Summary ===
    lines.append("## 1. Executive Summary\n")
    verdicts = _compute_verdicts(sft_p1, sft_zero, base_p1_norm, base_zero)
    lines.append("| 평가 차원 | 결과 | 상세 |")
    lines.append("|----------|------|------|")
    for dim_name, verdict, detail in verdicts:
        icon = "PASS" if verdict else "FAIL"
        lines.append(f"| {dim_name} | **{icon}** | {detail} |")
    lines.append("")

    pass_count = sum(1 for _, v, _ in verdicts if v)
    total_dims = len(verdicts)
    lines.append(f"**종합**: {pass_count}/{total_dims} 차원 통과\n")

    # ORPO verdict — quantitative scoring
    rep_rate = _get_greedy_3gram_rep(sft_p1)
    kobest_avg = _get_kobest_avg(sft_zero)
    max_forgetting = _get_max_forgetting(sft_p1, base_p1_norm)

    lines.append("### ORPO 판정 (정량 스코어)\n")
    orpo_result = _compute_orpo_score(sft_p1, sft_zero, base_p1_norm, base_zero)

    lines.append(f"**결정**: {orpo_result['decision']} (확신도: {orpo_result['confidence']:.0%})\n")
    lines.append(f"**정량 스코어**: {orpo_result['total_score']}/100\n")

    lines.append("| 차원 | 점수 | /가중치 | 현재값 | 기준 | 상태 |")
    lines.append("|------|------|--------|--------|------|------|")
    dim_names = {
        "ppl_forgetting": "PPL Forgetting",
        "greedy_rep": "Greedy 반복률",
        "eos_rate": "EOS 종료율",
        "kobest_avg": "KoBEST 평균",
        "calibration": "Calibration",
        "diversity": "다양성",
        "english": "영어 유지",
    }
    for key, label in dim_names.items():
        d = orpo_result["dimensions"].get(key, {})
        lines.append(
            f"| {label} | {d.get('score', 0)} | /{d.get('weight', 0)} | "
            f"{d.get('current', 'N/A')} | {d.get('threshold', '—')} | {d.get('status', 'N/A')} |"
        )
    lines.append("")

    if orpo_result["orpo_gain_estimate"] > 0:
        lines.append(f"**ORPO 기대 이득**: +{orpo_result['orpo_gain_estimate']}점 "
                     f"(반복률/EOS/다양성 개선 기대, PPL/벤치 변화 없음)\n")

    # Reference model comparison
    lines.append("**참조 모델 비교**:\n")
    for model_name, ref in _REFERENCE_MODELS.items():
        lines.append(f"- {model_name}: KoBEST={ref['kobest_avg']:.0%}, MMLU-KO={ref['mmlu_ko']:.0%}")
    lines.append("")

    # Decision explanation
    if orpo_result["decision"] == "DEPLOY":
        lines.append("**→ Phase 4: GGUF + Ollama 배포** (스코어 ≥80, 모든 핵심 조건 충족)\n")
    elif orpo_result["decision"] == "ORPO":
        lines.append("**→ Phase 3: ORPO** (스코어 40-79, 지식 보존 양호, 생성 개선 필요)\n")
    else:
        lines.append("**→ SFT 재시도** (스코어 <40 또는 심각한 forgetting)\n")

    # === 2. PPL Comparison ===
    lines.append("## 2. Perplexity 비교 (지식 보존)\n")
    lines.append("| 데이터셋 | Base PPL | SFT PPL | 변화 | Forgetting % | 판정 |")
    lines.append("|---------|---------|---------|------|-------------|------|")

    sft_ppl = sft_p1.get("perplexity", {})
    base_ppl = base_p1_norm.get("perplexity", {})

    # Merge all dataset names
    all_ppl_names = sorted(set(list(sft_ppl.keys()) + list(base_ppl.keys())))
    forgetting_values = []
    for name in all_ppl_names:
        sft_val = sft_ppl.get(name, {}).get("ppl") if isinstance(sft_ppl.get(name), dict) else None
        base_val = base_ppl.get(name, {}).get("ppl") if isinstance(base_ppl.get(name), dict) else None
        # Try reference table if base results not available
        if base_val is None:
            base_val = _BASE_PPL_REFERENCE.get(name)

        if sft_val is not None and base_val is not None:
            forgetting = (sft_val - base_val) / base_val * 100
            forgetting_values.append(forgetting)
            verdict = "PASS" if forgetting < _SFT_TARGETS["ppl_forgetting_max_pct"] else "FAIL"
            lines.append(
                f"| {name} | {base_val:.4f} | {sft_val:.4f} | "
                f"{'+' if sft_val >= base_val else ''}{sft_val - base_val:.4f} | "
                f"{forgetting:+.1f}% | {verdict} |"
            )
        elif sft_val is not None:
            lines.append(f"| {name} | — | {sft_val:.4f} | — | — | — |")
        elif base_val is not None:
            lines.append(f"| {name} | {base_val:.4f} | — | — | — | — |")

    if forgetting_values:
        avg_forgetting = sum(forgetting_values) / len(forgetting_values)
        max_f = max(forgetting_values)
        lines.append("")
        lines.append(f"**평균 Forgetting**: {avg_forgetting:+.1f}% | **최대**: {max_f:+.1f}% | "
                      f"**판정**: {'PASS' if max_f < _SFT_TARGETS['ppl_forgetting_max_pct'] else 'FAIL'} (임계값 {_SFT_TARGETS['ppl_forgetting_max_pct']}%)")
    lines.append("")

    # === 3. Generation Quality ===
    lines.append("## 3. 생성 품질 비교\n")
    sft_gen = sft_p1.get("generation", {})
    if not sft_gen:
        logger.warning("Generation results missing from SFT Phase 1")
    sft_summary = sft_gen.get("summary", {})

    lines.append("| 지표 | Base | SFT | 목표 | 판정 |")
    lines.append("|------|------|-----|------|------|")

    greedy_3gram = sft_summary.get("greedy_avg_3gram_rep")
    greedy_4gram = sft_summary.get("greedy_avg_4gram_rep")
    eos_rate = sft_summary.get("greedy_eos_rate")

    rep_threshold = _SFT_TARGETS["greedy_3gram_rep_max"]
    eos_threshold = _SFT_TARGETS["eos_rate_min"]
    greedy_3gram_verdict = "PASS" if greedy_3gram is not None and greedy_3gram < rep_threshold else "FAIL"
    greedy_4gram_verdict = "PASS" if greedy_4gram is not None and greedy_4gram < 0.05 else "FAIL"
    eos_verdict = "PASS" if eos_rate is not None and eos_rate >= eos_threshold else "FAIL"
    lines.append(f"| Greedy 3-gram 반복률 | {_BASE_GEN_REFERENCE['greedy_3gram_rep']:.2%} | "
                 f"{_fmt_pct(greedy_3gram)} | < {rep_threshold:.0%} | {greedy_3gram_verdict} |")
    lines.append(f"| Greedy 4-gram 반복률 | {_BASE_GEN_REFERENCE['greedy_4gram_rep']:.2%} | "
                 f"{_fmt_pct(greedy_4gram)} | < 5% | {greedy_4gram_verdict} |")
    lines.append(f"| EOS 종료율 | {_BASE_GEN_REFERENCE['greedy_eos_rate']:.0%} | "
                 f"{_fmt_pct(eos_rate)} | > {eos_threshold:.0%} | {eos_verdict} |")

    sampled_3gram = sft_summary.get("sampled_avg_3gram_rep")
    sampled_eos = sft_summary.get("sampled_eos_rate")
    if sampled_3gram is not None:
        lines.append(f"| Sampled 3-gram 반복률 | — | {sampled_3gram:.2%} | — | — |")
    if sampled_eos is not None:
        lines.append(f"| Sampled EOS 종료율 | — | {sampled_eos:.2%} | — | — |")
    lines.append("")

    # Chat template status
    chat_status = "활성화" if sft_summary else "비활성화"
    lines.append(f"**Chat Template**: {chat_status}\n")

    # Generation samples
    if sft_gen.get("samples"):
        lines.append("### 생성 샘플 (Greedy, Chat Template)\n")
        greedy_samples = [s for s in sft_gen["samples"] if s.get("temperature") == 0.0]
        for i, s in enumerate(greedy_samples[:5], 1):
            prompt = s.get("prompt", "")
            text = s.get("text", "")[:400]
            hit_eos = s.get("hit_eos", False)
            rep3 = s.get("3gram_rep", 0)
            lines.append(f"**[{i}]** `{prompt}`")
            lines.append(f"> {text}")
            lines.append(f"> *EOS={hit_eos}, 3gram_rep={rep3:.2%}, tokens={s.get('generated_tokens', 0)}*\n")

    # Repetition grid
    sft_rep = sft_p1.get("repetition", {})
    if sft_rep.get("grid_results"):
        lines.append("### Repetition 파라미터 검색 결과\n")
        lines.append("| 설정 | 3-gram | EOS Rate | Avg Tokens |")
        lines.append("|------|--------|----------|-----------|")
        grid = sft_rep["grid_results"]
        items = grid if isinstance(grid, list) else list(grid.values())
        for r in items[:6]:
            if isinstance(r, dict):
                lines.append(
                    f"| {r.get('params', '?')} | "
                    f"{_fmt_f(r.get('avg_3gram_rep'))} | "
                    f"{_fmt_f(r.get('eos_rate'))} | "
                    f"{_fmt_f(r.get('avg_tokens'), 1)} |"
                )
        lines.append("")

    # === 4. Korean Benchmarks ===
    lines.append("## 4. 한국어 벤치마크\n")
    lines.append("### KoBEST (0-shot)\n")
    lines.append("| 태스크 | Base | SFT | 변화 | 목표 | 판정 |")
    lines.append("|--------|------|-----|------|------|------|")

    kobest_tasks = ["kobest_boolq", "kobest_copa", "kobest_hellaswag",
                    "kobest_sentineg", "kobest_wic"]
    kobest_targets = {"kobest_boolq": 0.60, "kobest_copa": 0.65,
                      "kobest_hellaswag": 0.30, "kobest_sentineg": 0.60,
                      "kobest_wic": 0.55}
    sft_kobest_accs = []
    base_kobest_accs = []

    for t in kobest_tasks:
        base_a = _get_acc(base_zero.get(t, {})) if t in base_zero else _BASE_BENCH_REFERENCE.get(t)
        sft_a = _get_acc(sft_zero.get(t, {})) if t in sft_zero else None
        target = kobest_targets.get(t, 0.50)

        if sft_a is not None:
            sft_kobest_accs.append(sft_a)
        if base_a is not None:
            base_kobest_accs.append(base_a)

        diff = ""
        verdict = "—"
        if sft_a is not None and base_a is not None:
            d = (sft_a - base_a) * 100
            diff = f"{'+' if d >= 0 else ''}{d:.1f}pp"
            verdict = "PASS" if sft_a >= target else "FAIL"

        lines.append(f"| {t} | {_fmt_pct(base_a)} | {_fmt_pct(sft_a)} | {diff} | "
                     f"≥{target*100:.0f}% | {verdict} |")

    if sft_kobest_accs:
        sft_avg = sum(sft_kobest_accs) / len(sft_kobest_accs)
        base_avg = sum(base_kobest_accs) / len(base_kobest_accs) if base_kobest_accs else _BASE_BENCH_REFERENCE.get("kobest_avg", 0.4369)
        diff_avg = (sft_avg - base_avg) * 100
        lines.append(f"| **평균** | **{base_avg*100:.2f}%** | **{sft_avg*100:.2f}%** | "
                     f"**{'+' if diff_avg >= 0 else ''}{diff_avg:.1f}pp** | "
                     f"**≥{_SFT_TARGETS['kobest_avg_min']*100:.0f}%** | **{'PASS' if sft_avg >= _SFT_TARGETS['kobest_avg_min'] else 'FAIL'}** |")
    lines.append("")

    # HAE-RAE
    lines.append("### HAE-RAE (0-shot)\n")
    base_haerae = _get_acc(base_zero.get("haerae", {})) if "haerae" in base_zero else _BASE_BENCH_REFERENCE.get("haerae")
    sft_haerae = _get_acc(sft_zero.get("haerae", {})) if "haerae" in sft_zero else None
    if sft_haerae is not None:
        diff_h = (sft_haerae - (base_haerae or 0)) * 100 if base_haerae else 0
        lines.append(f"- Base: {_fmt_pct(base_haerae)} → SFT: {_fmt_pct(sft_haerae)} "
                     f"({'+' if diff_h >= 0 else ''}{diff_h:.1f}pp) | "
                     f"목표 ≥{_SFT_TARGETS['haerae_min']*100:.0f}% | {'PASS' if sft_haerae >= _SFT_TARGETS['haerae_min'] else 'FAIL'}")
    else:
        lines.append(f"- Base: {_fmt_pct(base_haerae)} → SFT: N/A")
    lines.append("")

    # MMLU-KO
    lines.append("### MMLU-KO (0-shot)\n")
    base_mmlu_ko = _get_acc(base_zero.get("global_mmlu_ko", {})) if "global_mmlu_ko" in base_zero else _BASE_BENCH_REFERENCE.get("global_mmlu_ko")
    sft_mmlu_ko = _get_acc(sft_zero.get("global_mmlu_ko", {})) if "global_mmlu_ko" in sft_zero else None
    if sft_mmlu_ko is not None:
        diff_mk = (sft_mmlu_ko - (base_mmlu_ko or 0)) * 100 if base_mmlu_ko else 0
        lines.append(f"- Base: {_fmt_pct(base_mmlu_ko)} → SFT: {_fmt_pct(sft_mmlu_ko)} "
                     f"({'+' if diff_mk >= 0 else ''}{diff_mk:.1f}pp) | "
                     f"목표 ≥{_SFT_TARGETS['mmlu_ko_min']*100:.0f}% | {'PASS' if sft_mmlu_ko >= _SFT_TARGETS['mmlu_ko_min'] else 'FAIL'}")
    else:
        lines.append(f"- Base: {_fmt_pct(base_mmlu_ko)} → SFT: N/A")
    lines.append("")

    # 5-shot comparison
    if sft_five:
        lines.append("### 5-shot 비교 (한국어)\n")
        lines.append("| 태스크 | 0-shot | 5-shot | 변화 |")
        lines.append("|--------|--------|--------|------|")
        for t in kobest_tasks + ["haerae", "global_mmlu_ko"]:
            a0 = _get_acc(sft_zero.get(t, {})) if t in sft_zero else None
            a5 = _get_acc(sft_five.get(t, {})) if t in sft_five else None
            if a0 is not None and a5 is not None:
                d = (a5 - a0) * 100
                lines.append(f"| {t} | {a0*100:.2f}% | {a5*100:.2f}% | {'+' if d >= 0 else ''}{d:.1f}pp |")
        lines.append("")

    # === 5. English Benchmarks ===
    lines.append("## 5. 영어 벤치마크 (유지 확인)\n")
    lines.append("| 태스크 | Base | SFT | 변화 | 하한 | 판정 |")
    lines.append("|--------|------|-----|------|------|------|")

    en_tasks = {
        "hellaswag": _SFT_TARGETS["hellaswag_min"],
        "arc_easy": _SFT_TARGETS["arc_easy_min"],
        "arc_challenge": _SFT_TARGETS["arc_challenge_min"],
        "winogrande": _SFT_TARGETS["winogrande_min"],
        "piqa": _SFT_TARGETS["piqa_min"],
    }
    for t, threshold in en_tasks.items():
        base_a = _get_acc(base_zero.get(t, {}), prefer_norm=(t in ["hellaswag", "arc_challenge"])) \
                 if t in base_zero else _BASE_BENCH_REFERENCE.get(t)
        sft_a = _get_acc(sft_zero.get(t, {}), prefer_norm=(t in ["hellaswag", "arc_challenge"])) \
                if t in sft_zero else None
        diff = ""
        verdict = "—"
        if sft_a is not None and base_a is not None:
            d = (sft_a - base_a) * 100
            diff = f"{'+' if d >= 0 else ''}{d:.1f}pp"
            verdict = "PASS" if sft_a >= threshold else "FAIL"
        lines.append(f"| {t} | {_fmt_pct(base_a)} | {_fmt_pct(sft_a)} | {diff} | "
                     f"≥{threshold*100:.0f}% | {verdict} |")

    # MMLU-EN
    _MMLU_EN_GROUPS = {"mmlu", "mmlu_humanities", "mmlu_social_sciences", "mmlu_stem", "mmlu_other"}
    sft_mmlu_en = []
    base_mmlu_en = []
    for t, m in sft_zero.items():
        if (t.startswith("mmlu_") or t == "mmlu") and t not in _MMLU_EN_GROUPS:
            a = _get_acc(m)
            if a is not None:
                sft_mmlu_en.append(a)
    if not sft_mmlu_en:
        for t in _MMLU_EN_GROUPS:
            if t in sft_zero:
                a = _get_acc(sft_zero[t])
                if a is not None:
                    sft_mmlu_en.append(a)
    for t, m in base_zero.items():
        if (t.startswith("mmlu_") or t == "mmlu") and t not in _MMLU_EN_GROUPS:
            a = _get_acc(m)
            if a is not None:
                base_mmlu_en.append(a)
    if not base_mmlu_en:
        for t in _MMLU_EN_GROUPS:
            if t in base_zero:
                a = _get_acc(base_zero[t])
                if a is not None:
                    base_mmlu_en.append(a)

    sft_mmlu_en_avg = sum(sft_mmlu_en) / len(sft_mmlu_en) if sft_mmlu_en else None
    base_mmlu_en_avg = sum(base_mmlu_en) / len(base_mmlu_en) if base_mmlu_en else 0.2581
    if sft_mmlu_en_avg is not None:
        d = (sft_mmlu_en_avg - base_mmlu_en_avg) * 100
        lines.append(f"| MMLU-EN 평균 | {base_mmlu_en_avg*100:.2f}% | {sft_mmlu_en_avg*100:.2f}% | "
                     f"{'+' if d >= 0 else ''}{d:.1f}pp | ≥25% | "
                     f"{'PASS' if sft_mmlu_en_avg >= _SFT_TARGETS['mmlu_en_avg_min'] else 'FAIL'} |")
    lines.append("")

    # === 6. Calibration ===
    lines.append("## 6. Calibration 비교\n")
    sft_cal = sft_p1.get("calibration", {})
    lines.append("| 지표 | Base | SFT | 목표 | 판정 |")
    lines.append("|------|------|-----|------|------|")

    cal_checks = [
        ("top1_accuracy", "Top-1 Accuracy", _SFT_TARGETS["top1_accuracy_min"], True),
        ("top5_accuracy", "Top-5 Accuracy", 0.78, True),
        ("top10_accuracy", "Top-10 Accuracy", 0.82, True),
        ("mean_entropy", "Mean Entropy", 2.0, False),
    ]
    for key, label, threshold, is_higher_better in cal_checks:
        base_v = _BASE_CALIB_REFERENCE.get(key)
        sft_v = sft_cal.get(key)
        verdict = "—"
        if sft_v is not None:
            if is_higher_better:
                verdict = "PASS" if sft_v >= threshold else "FAIL"
            else:
                verdict = "PASS" if sft_v <= threshold else "FAIL"
        lines.append(f"| {label} | {_fmt_f(base_v)} | {_fmt_f(sft_v)} | "
                     f"{'≥' if is_higher_better else '<'}{threshold} | {verdict} |")

    # Token NLL
    sft_nll = sft_p1.get("token_nll", {})
    nll_mean = sft_nll.get("nll_mean", sft_nll.get("mean"))
    base_nll_mean = _BASE_NLL_REFERENCE.get("nll_mean")
    if nll_mean is not None:
        lines.append(f"| Token NLL mean | {_fmt_f(base_nll_mean)} | {_fmt_f(nll_mean)} | "
                     f"< 2.0 | {'PASS' if nll_mean < 2.0 else 'FAIL'} |")
    hlf5 = sft_nll.get("high_loss_fractions", {}).get("5", sft_nll.get("high_loss_fraction_5"))
    base_hlf5 = _BASE_NLL_REFERENCE.get("high_loss_fraction_5")
    if hlf5 is not None:
        lines.append(f"| NLL > 5 비율 | {_fmt_f(base_hlf5)} | {_fmt_f(hlf5)} | "
                     f"< 0.15 | {'PASS' if hlf5 < 0.15 else 'FAIL'} |")
    lines.append("")

    # === 7. Final Verdict ===
    lines.append("## 7. 종합 판정 및 다음 단계\n")

    lines.append("### 핵심 판정 기준\n")
    lines.append("| 조건 | 현재 값 | 기준 | 충족 |")
    lines.append("|------|---------|------|------|")

    rep_val = rep_rate
    lines.append(f"| Greedy 3-gram 반복률 | {_fmt_pct(rep_val)} | < {_SFT_TARGETS['greedy_3gram_rep_max']:.0%} | "
                 f"{'YES' if rep_val is not None and rep_val < _SFT_TARGETS['greedy_3gram_rep_max'] else 'NO'} |")
    lines.append(f"| KoBEST 평균 | {_fmt_pct(kobest_avg)} | > {_SFT_TARGETS['kobest_avg_min']*100:.0f}% | "
                 f"{'YES' if kobest_avg is not None and kobest_avg > _SFT_TARGETS['kobest_avg_min'] else 'NO'} |")
    lines.append(f"| 최대 Forgetting | {f'{max_forgetting:.1f}%' if max_forgetting is not None else 'N/A'} | "
                 f"< {_SFT_TARGETS['ppl_forgetting_max_pct']}% | {'YES' if max_forgetting is not None and max_forgetting < _SFT_TARGETS['ppl_forgetting_max_pct'] else 'NO'} |")
    lines.append("")

    # Final recommendation
    lines.append("### 권고\n")
    if rep_rate is not None and rep_rate < _SFT_TARGETS["greedy_3gram_rep_max"] and kobest_avg is not None and kobest_avg > _SFT_TARGETS["kobest_avg_min"] and max_forgetting is not None and max_forgetting < _SFT_TARGETS["ppl_forgetting_max_pct"]:
        lines.append("**모든 핵심 조건 충족 → Phase 4: GGUF 변환 + Ollama 배포 진행**\n")
    elif (rep_rate is not None and rep_rate < _SFT_TARGETS["greedy_3gram_rep_max"] * 3) or (kobest_avg is not None and kobest_avg > _SFT_TARGETS["kobest_avg_min"] * 0.82):
        lines.append("**부분 달성 → Phase 3: ORPO 학습 진행** (795K preference pairs 활용)\n")
        lines.append("ORPO 학습 시 주안점:")
        lines.append("- 반복률 추가 감소")
        lines.append("- 벤치마크 점수 유지/향상")
        lines.append("- EOS 종료율 개선")
    else:
        lines.append("**핵심 조건 미달 → SFT 재시도**\n")
        lines.append("재시도 시 검토 사항:")
        lines.append("- 학습률 조정")
        lines.append("- 데이터 구성 재검토")
        lines.append("- 에폭 수 조정")
    lines.append("")

    lines.append("---\n")
    lines.append("*이 보고서는 `eval/sft_eval_pipeline.py`에 의해 자동 생성되었습니다.*")

    report_text = "\n".join(lines)
    output_path.write_text(report_text, encoding="utf-8")

    # Also save to sft_output_dir if provided
    if sft_output_dir:
        (Path(sft_output_dir) / "sft_comparison_report.md").write_text(report_text, encoding="utf-8")

    return output_path


def _compute_verdicts(sft_p1, sft_zero, base_p1, base_zero):
    """Compute pass/fail verdicts for each of the 6 evaluation dimensions."""
    verdicts = []

    # Dim 1: PPL forgetting
    max_forgetting = _get_max_forgetting(sft_p1, base_p1)
    if max_forgetting is not None:
        verdicts.append((
            "차원 1: Perplexity (지식 보존)",
            max_forgetting < _SFT_TARGETS["ppl_forgetting_max_pct"],
            f"최대 forgetting {max_forgetting:.1f}% (임계값 {_SFT_TARGETS['ppl_forgetting_max_pct']}%)",
        ))
    else:
        verdicts.append(("차원 1: Perplexity (지식 보존)", False, "데이터 없음"))

    # Dim 2: Generation quality
    rep_rate = _get_greedy_3gram_rep(sft_p1)
    eos_rate = sft_p1.get("generation", {}).get("summary", {}).get("greedy_eos_rate")
    if rep_rate is not None and eos_rate is not None:
        gen_pass = rep_rate < _SFT_TARGETS["greedy_3gram_rep_max"] and eos_rate > _SFT_TARGETS["eos_rate_min"]
        verdicts.append((
            "차원 2: 생성 품질",
            gen_pass,
            f"반복률 {rep_rate:.2%} (목표 <{_SFT_TARGETS['greedy_3gram_rep_max']:.0%}), EOS {eos_rate:.0%} (목표 >{_SFT_TARGETS['eos_rate_min']:.0%})",
        ))
    else:
        verdicts.append(("차원 2: 생성 품질", False, "데이터 없음"))

    # Dim 3: Korean benchmarks
    kobest_avg = _get_kobest_avg(sft_zero)
    if kobest_avg is not None:
        verdicts.append((
            "차원 3: 한국어 벤치마크",
            kobest_avg > _SFT_TARGETS["kobest_avg_min"],
            f"KoBEST 평균 {kobest_avg*100:.2f}% (목표 >{_SFT_TARGETS['kobest_avg_min']*100:.0f}%)",
        ))
    else:
        verdicts.append(("차원 3: 한국어 벤치마크", False, "데이터 없음"))

    # Dim 4: English benchmarks
    en_tasks = {
        "hellaswag": _SFT_TARGETS["hellaswag_min"],
        "arc_easy": _SFT_TARGETS["arc_easy_min"],
        "arc_challenge": _SFT_TARGETS["arc_challenge_min"],
        "winogrande": _SFT_TARGETS["winogrande_min"],
        "piqa": _SFT_TARGETS["piqa_min"],
    }
    en_pass = True
    en_detail_parts = []
    for t, threshold in en_tasks.items():
        a = _get_acc(sft_zero.get(t, {})) if t in sft_zero else None
        if a is not None:
            if a < threshold:
                en_pass = False
            en_detail_parts.append(f"{t}={a*100:.1f}%")
    if en_detail_parts:
        verdicts.append((
            "차원 4: 영어 벤치마크",
            en_pass,
            ", ".join(en_detail_parts[:3]) + ("..." if len(en_detail_parts) > 3 else ""),
        ))
    else:
        verdicts.append(("차원 4: 영어 벤치마크", False, "데이터 없음"))

    # Dim 5: Calibration
    cal = sft_p1.get("calibration", {})
    top1 = cal.get("top1_accuracy")
    if top1 is not None:
        cal_pass = top1 >= _SFT_TARGETS["top1_accuracy_min"]
        verdicts.append((
            "차원 5: Calibration",
            cal_pass,
            f"Top-1 {top1*100:.2f}% (목표 ≥{_SFT_TARGETS['top1_accuracy_min']*100:.0f}%)",
        ))
    else:
        verdicts.append(("차원 5: Calibration", False, "데이터 없음"))

    # Dim 6: SFT-specific (chat quality) — based on generation + EOS
    if eos_rate is not None:
        chat_pass = eos_rate > 0.50  # relaxed threshold for chat
        verdicts.append((
            "차원 6: SFT Chat 능력",
            chat_pass,
            f"EOS 종료율 {eos_rate:.0%}, 생성 샘플 수동 검토 필요",
        ))
    else:
        verdicts.append(("차원 6: SFT Chat 능력", False, "데이터 없음"))

    return verdicts


def _get_greedy_3gram_rep(p1: dict) -> Optional[float]:
    gen = p1.get("generation", {})
    return gen.get("summary", {}).get("greedy_avg_3gram_rep")


def _get_kobest_avg(zero_shot: dict) -> Optional[float]:
    kobest_tasks = ["kobest_boolq", "kobest_copa", "kobest_hellaswag",
                    "kobest_sentineg", "kobest_wic"]
    accs = []
    for t in kobest_tasks:
        if t in zero_shot:
            a = _get_acc(zero_shot[t])
            if a is not None:
                accs.append(a)
    return sum(accs) / len(accs) if accs else None


def _get_max_forgetting(sft_p1: dict, base_p1: dict) -> Optional[float]:
    sft_ppl = sft_p1.get("perplexity", {})
    base_ppl = base_p1.get("perplexity", {})
    forgetting_values = []
    for name in sft_ppl:
        sft_val = sft_ppl[name].get("ppl") if isinstance(sft_ppl[name], dict) else None
        base_val = base_ppl.get(name, {}).get("ppl") if isinstance(base_ppl.get(name), dict) else None
        if base_val is None:
            base_val = _BASE_PPL_REFERENCE.get(name)
        if sft_val is not None and base_val is not None and base_val > 0:
            forgetting_values.append((sft_val - base_val) / base_val * 100)
    return max(forgetting_values) if forgetting_values else None


if __name__ == "__main__":
    print("report_generator.py — use via full_eval_pipeline.py or sft_eval_pipeline.py")
