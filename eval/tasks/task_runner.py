"""
task_runner.py — Thin CLI entry point for subprocess GPU workers.

Usage:
    CUDA_VISIBLE_DEVICES=5 python eval/tasks/task_runner.py \
        --task calibration --gpu-id 5 --output /path/to/result.json
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# NUMA affinity helper
# ---------------------------------------------------------------------------

def _set_numa_affinity(gpu_id: int) -> None:
    """Pin the process to the NUMA node that owns the given GPU.

    GPU 0-3 → cores 0-35  (NUMA node 0)
    GPU 4-7 → cores 36-71 (NUMA node 1)
    """
    try:
        import os
        if gpu_id <= 3:
            cores = list(range(0, 36))
        else:
            cores = list(range(36, 72))

        # os.sched_setaffinity is available on Linux
        os.sched_setaffinity(0, cores)
        print(
            f"[TASK_RUNNER gpu_id={gpu_id}] NUMA affinity set: cores {cores[0]}-{cores[-1]}",
            flush=True,
        )
    except Exception as exc:
        # Non-fatal — just warn and continue
        print(
            f"[TASK_RUNNER gpu_id={gpu_id}] WARNING: could not set NUMA affinity: {exc}",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Task dispatch
# ---------------------------------------------------------------------------

VALID_TASKS = {
    "ppl_single",
    "ppl_multi",
    "calibration",
    "token_nll",
    "calib_nll",
    "generation",
    "repetition_grid",
    "lm_eval",
}


def _run_task(args: argparse.Namespace) -> dict:
    task = args.task
    device = "cuda:0"  # CUDA_VISIBLE_DEVICES already set by parent

    if task == "ppl_single":
        if not args.val_file:
            raise ValueError("--val-file is required for ppl_single task")
        from eval.tasks.ppl_task import eval_ppl_single
        result = eval_ppl_single(args.val_file, device)

    elif task == "ppl_multi":
        if not args.val_files:
            raise ValueError("--val-files is required for ppl_multi task")
        val_files_list = [f.strip() for f in args.val_files.split(",") if f.strip()]
        from eval.tasks.ppl_task import eval_ppl_multi
        result = eval_ppl_multi(val_files_list, device)

    elif task == "calibration":
        from eval.tasks.calibration_task import eval_calibration
        result = eval_calibration(device)

    elif task == "token_nll":
        from eval.tasks.token_nll_task import eval_token_nll
        result = eval_token_nll(device)

    elif task == "calib_nll":
        from eval.tasks.calibration_task import eval_calibration
        from eval.tasks.token_nll_task import eval_token_nll
        calib_result = eval_calibration(device)
        nll_result = eval_token_nll(device)
        result = {"calibration": calib_result, "token_nll": nll_result}

    elif task == "generation":
        from eval.tasks.generation_task import eval_generation
        result = eval_generation(device)

    elif task == "repetition_grid":
        from eval.tasks.generation_task import eval_repetition_grid
        result = eval_repetition_grid(device)

    elif task == "lm_eval":
        if not args.hf_model_path:
            raise ValueError("--hf-model-path is required for lm_eval task")
        if not args.lm_eval_tasks:
            raise ValueError("--lm-eval-tasks is required for lm_eval task")
        tasks_list = [t.strip() for t in args.lm_eval_tasks.split(",") if t.strip()]
        from eval.tasks.lm_eval_task import run_lm_eval_tasks
        result = run_lm_eval_tasks(
            args.hf_model_path,
            tasks_list,
            device,
            num_fewshot=args.num_fewshot,
        )

    else:
        raise ValueError(f"Unknown task: {task!r}. Valid tasks: {sorted(VALID_TASKS)}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Thin CLI entry point for subprocess GPU eval workers."
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=sorted(VALID_TASKS),
        help="Eval task to run.",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        required=True,
        help="Original GPU ID (used for NUMA affinity only).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write JSON result file.",
    )
    # --- ppl_single ---
    parser.add_argument(
        "--val-file",
        default=None,
        help="Single validation filename (for ppl_single).",
    )
    # --- ppl_multi ---
    parser.add_argument(
        "--val-files",
        default=None,
        help="Comma-separated validation filenames (for ppl_multi).",
    )
    # --- lm_eval ---
    parser.add_argument(
        "--hf-model-path",
        default=None,
        help="HuggingFace model directory (for lm_eval).",
    )
    parser.add_argument(
        "--lm-eval-tasks",
        default=None,
        help="Comma-separated lm-eval task names (for lm_eval).",
    )
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=0,
        help="Number of few-shot examples (for lm_eval). Default: 0.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    gpu_id = args.gpu_id
    task_name = args.task
    output_path = args.output

    print(f"[TASK_RUNNER gpu_id={gpu_id}] Starting task={task_name}", flush=True)

    # Set NUMA affinity early
    _set_numa_affinity(gpu_id)

    exit_code = 0
    try:
        result = _run_task(args)
        payload = result
    except Exception as exc:
        tb_str = traceback.format_exc()
        print(
            f"[TASK_RUNNER gpu_id={gpu_id}] ERROR in task={task_name}:\n{tb_str}",
            file=sys.stderr,
            flush=True,
        )
        payload = {"error": str(exc), "traceback": tb_str}
        exit_code = 1

    # Write result JSON
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path_obj, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)

    print(
        f"[TASK_RUNNER gpu_id={gpu_id}] Done. Result saved to {output_path}",
        flush=True,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
