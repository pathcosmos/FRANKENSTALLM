"""
Markdown report generator for FRANKENSTALLM 3B evaluation pipeline.

Generates comprehensive evaluation reports with sections for:
- Perplexity metrics across datasets
- Calibration statistics
- Token NLL distribution
- Generation quality samples
- Repetition parameter search results
- Standard benchmark results (lm-eval)
- Comparison with reference models
- GPU/time statistics
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import json


def generate_report(
    phase1_results: dict,
    phase2_results: dict,
    generation_samples: list,
    output_dir: Path,
    checkpoint_name: str = "checkpoint-0057000",
    total_elapsed_sec: float = 0.0,
) -> str:
    """
    Generate a comprehensive markdown evaluation report.
    
    Args:
        phase1_results: Dictionary containing perplexity, calibration, token_nll, 
                       generation, and repetition results
        phase2_results: Dictionary containing benchmark results (KoBEST, MMLU-ko, etc.)
        generation_samples: List of generation sample dictionaries
        output_dir: Directory to save the report
        checkpoint_name: Name of the model checkpoint
        total_elapsed_sec: Total elapsed time for all evaluations
    
    Returns:
        Markdown string content (also written to output_dir / "full_eval_report.md")
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Start building the report
    report = []
    
    # ===== 1. Header =====
    eval_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report.append("# FRANKENSTALLM 3B 종합 평가 리포트\n")
    report.append(f"- **모델**: FRANKENSTALLM 3B")
    report.append(f"- **체크포인트**: {checkpoint_name}")
    report.append(f"- **평가 일시**: {eval_datetime}")
    report.append(f"- **총 소요 시간**: {total_elapsed_sec:.1f}초\n")
    
    # ===== 2. Executive Summary =====
    report.append("## Executive Summary\n")
    report.append("| 메트릭 | 값 |")
    report.append("|--------|-----|")
    
    # Extract key metrics
    main_ppl = phase1_results.get("perplexity", {}).get("3b_val", {}).get("ppl", "데이터 없음")
    if isinstance(main_ppl, (int, float)):
        main_ppl = f"{main_ppl:.4f}"
    
    mmlu_ko_acc = "데이터 없음"
    if phase2_results and "global_mmlu_ko" in phase2_results:
        mmlu_ko_data = phase2_results.get("global_mmlu_ko", {})
        if isinstance(mmlu_ko_data, dict) and "accuracy" in mmlu_ko_data:
            mmlu_ko_acc = f"{mmlu_ko_data['accuracy']:.4f}"
        elif isinstance(mmlu_ko_data, float):
            mmlu_ko_acc = f"{mmlu_ko_data:.4f}"
    
    kobest_avg = "데이터 없음"
    if phase2_results:
        kobest_tasks = {}
        for task_name in ["korsts", "ynat", "klue-sts", "klue-nli"]:
            if task_name in phase2_results:
                task_data = phase2_results[task_name]
                if isinstance(task_data, dict) and "accuracy" in task_data:
                    kobest_tasks[task_name] = task_data["accuracy"]
        if kobest_tasks:
            kobest_avg = f"{sum(kobest_tasks.values()) / len(kobest_tasks):.4f}"
    
    top1_acc = phase1_results.get("calibration", {}).get("top1_accuracy", "데이터 없음")
    if isinstance(top1_acc, (int, float)):
        top1_acc = f"{top1_acc:.4f}"
    
    report.append(f"| 주요 PPL (3b_val) | {main_ppl} |")
    report.append(f"| KMMLU 평균 정확도 | {mmlu_ko_acc} |")
    report.append(f"| KoBEST 평균 | {kobest_avg} |")
    report.append(f"| Top-1 정확도 (Calibration) | {top1_acc} |")
    report.append("")
    
    # ===== 3. Perplexity Table =====
    report.append("## 3. Perplexity 평가\n")
    if phase1_results.get("perplexity"):
        ppl_data = phase1_results["perplexity"]
        
        # Build table rows
        ppl_rows = []
        for dataset_name, metrics in ppl_data.items():
            if isinstance(metrics, dict):
                ppl = metrics.get("ppl", "N/A")
                bits_per_token = metrics.get("bits_per_token", "N/A")
                n_tokens = metrics.get("n_tokens", "N/A")
                n_eval_tokens = metrics.get("n_eval_tokens", "N/A")
                elapsed_sec = metrics.get("elapsed_sec", "N/A")
                
                # Format numeric values
                if isinstance(ppl, (int, float)):
                    ppl = f"{ppl:.4f}"
                if isinstance(bits_per_token, (int, float)):
                    bits_per_token = f"{bits_per_token:.4f}"
                if isinstance(elapsed_sec, (int, float)):
                    elapsed_sec = f"{elapsed_sec:.1f}"
                
                ppl_rows.append({
                    "dataset": dataset_name,
                    "ppl_val": float(ppl) if isinstance(ppl, str) and ppl != "N/A" else float('inf'),
                    "ppl_str": ppl,
                    "bits": bits_per_token,
                    "n_tokens": n_tokens,
                    "n_eval": n_eval_tokens,
                    "time": elapsed_sec,
                })
        
        # Sort by PPL descending
        ppl_rows.sort(key=lambda x: x["ppl_val"], reverse=True)
        
        report.append("| 데이터셋 | PPL | Bits/Token | 전체 토큰 | 평가 토큰 | 소요 시간 |")
        report.append("|---------|-----|-----------|---------|---------|---------|")
        for row in ppl_rows:
            report.append(f"| {row['dataset']} | {row['ppl_str']} | {row['bits']} | {row['n_tokens']} | {row['n_eval']} | {row['time']}s |")
        report.append("")
    else:
        report.append("데이터 없음\n")
    
    # ===== 4. Calibration Results =====
    report.append("## 4. Calibration 결과\n")
    if phase1_results.get("calibration"):
        cal_data = phase1_results["calibration"]
        report.append("| 메트릭 | 값 |")
        report.append("|--------|-----|")
        
        metrics_map = {
            "top1_accuracy": "Top-1 Accuracy",
            "top5_accuracy": "Top-5 Accuracy",
            "top10_accuracy": "Top-10 Accuracy",
            "mean_correct_prob": "Mean Correct Prob",
            "mean_entropy": "Mean Entropy",
        }
        
        for key, label in metrics_map.items():
            value = cal_data.get(key, "N/A")
            if isinstance(value, (int, float)):
                value = f"{value:.4f}"
            report.append(f"| {label} | {value} |")
        report.append("")
    else:
        report.append("데이터 없음\n")
    
    # ===== 5. Token NLL Distribution =====
    report.append("## 5. Token NLL 분포\n")
    if phase1_results.get("token_nll"):
        nll_data = phase1_results["token_nll"]
        report.append("### 기본 통계\n")
        
        stats_map = {
            "mean": "평균",
            "std": "표준편차",
            "median": "중앙값",
            "min": "최솟값",
            "max": "최댓값",
        }
        
        report.append("| 통계 | 값 |")
        report.append("|------|-----|")
        for key, label in stats_map.items():
            value = nll_data.get(key, "N/A")
            if isinstance(value, (int, float)):
                value = f"{value:.4f}"
            report.append(f"| {label} | {value} |")
        report.append("")
        
        # Percentiles
        if "percentiles" in nll_data:
            report.append("### Percentiles\n")
            report.append("| Percentile | 값 |")
            report.append("|------------|-----|")
            for pct, value in nll_data["percentiles"].items():
                if isinstance(value, (int, float)):
                    value = f"{value:.4f}"
                report.append(f"| {pct}th | {value} |")
            report.append("")
        
        # High-loss fractions
        if "high_loss_fractions" in nll_data:
            report.append("### 고손실 토큰 비율\n")
            report.append("| 임계값 | 비율 |")
            report.append("|--------|-----|")
            for threshold, fraction in nll_data["high_loss_fractions"].items():
                if isinstance(fraction, (int, float)):
                    fraction = f"{fraction:.4f}"
                report.append(f"| NLL > {threshold} | {fraction} |")
            report.append("")
    else:
        report.append("데이터 없음\n")
    
    # ===== 6. Generation Quality =====
    report.append("## 6. 생성 품질\n")
    if phase1_results.get("generation"):
        gen_data = phase1_results["generation"]
        
        # Summary stats
        if "summary" in gen_data:
            report.append("### 요약 통계\n")
            summary = gen_data["summary"]
            report.append("| 메트릭 | 값 |")
            report.append("|--------|-----|")
            
            for key, value in summary.items():
                if isinstance(value, (int, float)):
                    value = f"{value:.4f}"
                # Convert snake_case to readable Korean
                key_display = key.replace("_", " ").title()
                report.append(f"| {key_display} | {value} |")
            report.append("")
        
        # Sample outputs (first 5 prompts, greedy only)
        if generation_samples:
            report.append("### 생성 샘플 (Greedy)\n")
            for i, sample in enumerate(generation_samples[:5], 1):
                if isinstance(sample, dict):
                    prompt = sample.get("prompt", "")
                    generated = sample.get("generated_text", "")
                    
                    # Truncate to 200 chars
                    if len(generated) > 200:
                        generated = generated[:200] + "..."
                    
                    report.append(f"#### 샘플 {i}\n")
                    report.append(f"**Prompt**: {prompt}\n")
                    report.append(f"**Generated**: {generated}\n")
            report.append("")
    else:
        report.append("데이터 없음\n")
    
    # ===== 7. Repetition Parameter Search =====
    report.append("## 7. Repetition 파라미터 검색\n")
    if phase1_results.get("repetition") and phase1_results["repetition"].get("grid_results"):
        rep_data = phase1_results["repetition"]["grid_results"]
        
        # Build rows for sorting
        rep_rows = []
        for config, metrics in rep_data.items():
            if isinstance(metrics, dict):
                rep_rows.append({
                    "config": config,
                    "temp": metrics.get("temperature", "N/A"),
                    "rep_pen": metrics.get("repetition_penalty", "N/A"),
                    "3gram": metrics.get("3gram_repetition", float('inf')),
                    "4gram": metrics.get("4gram_repetition", float('inf')),
                    "eos_rate": metrics.get("eos_rate", "N/A"),
                    "avg_tokens": metrics.get("avg_tokens", "N/A"),
                    "metrics": metrics,
                })
        
        # Sort by 3-gram repetition
        rep_rows.sort(key=lambda x: x["3gram"] if isinstance(x["3gram"], (int, float)) else float('inf'))
        
        report.append("| 설정 | Temperature | Rep Penalty | 3-gram Rep | 4-gram Rep | EOS Rate | Avg Tokens |")
        report.append("|------|-------------|------------|-----------|-----------|----------|-----------|")
        
        for i, row in enumerate(rep_rows):
            config_label = row["config"]
            
            # Format numeric values
            temp = f"{row['temp']:.2f}" if isinstance(row['temp'], (int, float)) else row['temp']
            rep_pen = f"{row['rep_pen']:.2f}" if isinstance(row['rep_pen'], (int, float)) else row['rep_pen']
            gram3 = f"{row['3gram']:.4f}" if isinstance(row['3gram'], (int, float)) else row['3gram']
            gram4 = f"{row['4gram']:.4f}" if isinstance(row['4gram'], (int, float)) else row['4gram']
            eos = f"{row['eos_rate']:.4f}" if isinstance(row['eos_rate'], (int, float)) else row['eos_rate']
            tokens = f"{row['avg_tokens']:.1f}" if isinstance(row['avg_tokens'], (int, float)) else row['avg_tokens']
            
            # Highlight best row
            marker = " ← **최적**" if i == 0 else ""
            report.append(f"| {config_label} | {temp} | {rep_pen} | {gram3} | {gram4} | {eos} | {tokens} |{marker}")
        report.append("")
    else:
        report.append("데이터 없음\n")
    
    # ===== 8. Standard Benchmarks (lm-eval) =====
    report.append("## 8. 표준 벤치마크\n")
    if phase2_results:
        has_benchmarks = False
        
        # KoBEST
        kobest_tasks = ["korsts", "ynat", "klue-sts", "klue-nli"]
        kobest_results = {k: v for k, v in phase2_results.items() if k in kobest_tasks}
        
        if kobest_results:
            has_benchmarks = True
            report.append("### KoBEST\n")
            report.append("| 태스크 | 정확도 |")
            report.append("|--------|--------|")
            
            for task_name, task_data in kobest_results.items():
                accuracy = "N/A"
                if isinstance(task_data, dict) and "accuracy" in task_data:
                    accuracy = f"{task_data['accuracy']:.4f}"
                elif isinstance(task_data, (int, float)):
                    accuracy = f"{task_data:.4f}"
                report.append(f"| {task_name} | {accuracy} |")
            report.append("")
        
        # Global MMLU (Korean)
        if "global_mmlu_ko" in phase2_results:
            has_benchmarks = True
            report.append("### Global MMLU (Korean)\n")
            mmlu_data = phase2_results["global_mmlu_ko"]
            
            if isinstance(mmlu_data, dict):
                overall_acc = mmlu_data.get("accuracy", "N/A")
                if isinstance(overall_acc, (int, float)):
                    overall_acc = f"{overall_acc:.4f}"
                report.append(f"**전체 정확도**: {overall_acc}\n")
                
                # Per-subject breakdown if available
                if "subject_breakdown" in mmlu_data:
                    subjects = mmlu_data["subject_breakdown"]
                    # Top 10 / Bottom 10
                    sorted_subjects = sorted(
                        subjects.items(),
                        key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0,
                        reverse=True
                    )
                    
                    report.append("**상위 10개 과목**:\n")
                    report.append("| 과목 | 정확도 |")
                    report.append("|------|--------|")
                    for subject, acc in sorted_subjects[:10]:
                        acc_str = f"{acc:.4f}" if isinstance(acc, (int, float)) else acc
                        report.append(f"| {subject} | {acc_str} |")
                    report.append("")
                    
                    report.append("**하위 10개 과목**:\n")
                    report.append("| 과목 | 정확도 |")
                    report.append("|------|--------|")
                    for subject, acc in sorted_subjects[-10:]:
                        acc_str = f"{acc:.4f}" if isinstance(acc, (int, float)) else acc
                        report.append(f"| {subject} | {acc_str} |")
                    report.append("")
            report.append("")
        
        # PAWS-KO
        if "paws_ko" in phase2_results:
            has_benchmarks = True
            report.append("### PAWS-KO\n")
            paws_data = phase2_results["paws_ko"]
            
            accuracy = "N/A"
            if isinstance(paws_data, dict) and "accuracy" in paws_data:
                accuracy = f"{paws_data['accuracy']:.4f}"
            elif isinstance(paws_data, (int, float)):
                accuracy = f"{paws_data:.4f}"
            
            report.append(f"| 정확도 | {accuracy} |")
            report.append("|--------|--------|")
            report.append("")
        
        if not has_benchmarks:
            report.append("데이터 없음\n")
    else:
        report.append("데이터 없음\n")
    
    # ===== 9. Reference Comparison =====
    report.append("## 9. 참고 모델 비교\n")
    report.append("| 모델 | 파라미터 | MMLU (ko) | KoBEST 평균 | PPL |")
    report.append("|------|---------|-----------|------------|-----|")
    report.append(f"| FRANKENSTALLM 3B | 3B | {mmlu_ko_acc} | {kobest_avg} | {main_ppl} |")
    report.append("| Llama-3.2-3B | 3B | ~42 | ~55 | — |")
    report.append("| Qwen2.5-3B | 3B | ~48 | ~60 | — |")
    report.append("| EXAONE-3.5-2.4B | 2.4B | ~35 | ~50 | — |")
    report.append("")
    
    # ===== 10. GPU/Time Statistics =====
    report.append("## 10. 컴퓨팅 자원 통계\n")
    report.append("| Phase | Task | 소요 시간(s) | 상태 |")
    report.append("|-------|------|------------|------|")
    
    # Extract timing from phase1
    if phase1_results.get("perplexity"):
        report.append("| Phase 1 | Perplexity | - | 완료 |")
    if phase1_results.get("calibration"):
        report.append("| Phase 1 | Calibration | - | 완료 |")
    if phase1_results.get("token_nll"):
        report.append("| Phase 1 | Token NLL | - | 완료 |")
    if phase1_results.get("generation"):
        report.append("| Phase 1 | Generation | - | 완료 |")
    if phase1_results.get("repetition"):
        report.append("| Phase 1 | Repetition Search | - | 완료 |")
    
    # Phase 2 benchmarks
    if phase2_results:
        report.append("| Phase 2 | Standard Benchmarks | - | 완료 |")
    
    report.append(f"| **전체** | **모든 평가** | **{total_elapsed_sec:.1f}** | **완료** |")
    report.append("")
    
    # ===== Footer =====
    report.append("---\n")
    report.append("*이 리포트는 자동으로 생성되었습니다.*")
    
    # Join all lines
    report_markdown = "\n".join(report)
    
    # Write to file
    report_path = output_dir / "full_eval_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_markdown)
    
    return report_markdown


if __name__ == "__main__":
    # Example usage
    example_phase1 = {
        "perplexity": {
            "3b_val": {"ppl": 8.5234, "bits_per_token": 3.0912, "n_tokens": 1000000, "n_eval_tokens": 950000, "elapsed_sec": 123.5},
            "wikitext": {"ppl": 12.3456, "bits_per_token": 3.6543, "n_tokens": 500000, "n_eval_tokens": 480000, "elapsed_sec": 67.2},
        },
        "calibration": {
            "top1_accuracy": 0.7823,
            "top5_accuracy": 0.9234,
            "top10_accuracy": 0.9567,
            "mean_correct_prob": 0.6521,
            "mean_entropy": 1.2345,
        },
        "token_nll": {
            "mean": 2.3456,
            "std": 1.2345,
            "median": 2.1234,
            "min": 0.0234,
            "max": 8.9876,
            "percentiles": {"5": 0.5234, "25": 1.2345, "75": 3.4567, "95": 5.6789, "99": 7.8901},
            "high_loss_fractions": {"5": 0.1234, "10": 0.0567},
        },
        "generation": {
            "summary": {"avg_length": 45.6, "repetition_ratio": 0.0234, "unique_tokens": 8567},
        },
        "repetition": {
            "grid_results": {
                "config_1": {"temperature": 0.7, "repetition_penalty": 1.0, "3gram_repetition": 0.0123, "4gram_repetition": 0.0045, "eos_rate": 0.95, "avg_tokens": 47.2},
                "config_2": {"temperature": 0.8, "repetition_penalty": 1.2, "3gram_repetition": 0.0234, "4gram_repetition": 0.0089, "eos_rate": 0.92, "avg_tokens": 49.1},
            }
        }
    }
    
    example_phase2 = {
        "korsts": {"accuracy": 0.8234},
        "ynat": {"accuracy": 0.7654},
        "klue-sts": {"accuracy": 0.7899},
        "klue-nli": {"accuracy": 0.7432},
        "global_mmlu_ko": {
            "accuracy": 0.4567,
            "subject_breakdown": {
                "biology": 0.5234,
                "chemistry": 0.4123,
                "history": 0.4567,
            }
        },
        "paws_ko": {"accuracy": 0.8765},
    }
    
    example_samples = [
        {"prompt": "한국의 수도는", "generated_text": "서울이다. 서울은 한반도의 중부에 위치한 대한민국의 행정 수도이며, 정치, 경제, 문화의 중심지이다."},
        {"prompt": "AI의 미래는", "generated_text": "밝다고 전문가들은 예측하고 있다. 인공지능 기술의 발전은 다양한 산업에 혁신을 가져올 것으로 기대된다."},
    ]
    
    output_path = Path("/tmp")
    report = generate_report(
        phase1_results=example_phase1,
        phase2_results=example_phase2,
        generation_samples=example_samples,
        output_dir=output_path,
        checkpoint_name="checkpoint-0057000",
        total_elapsed_sec=567.8,
    )
    
    print("Report generated successfully!")
    print(f"Saved to: {output_path / 'full_eval_report.md'}")
