#!/usr/bin/env python3
"""Upload FRANKENSTALLM: model, eval reports, source code, and data scripts to Hugging Face.

Usage:
    huggingface-cli login

    # 모델 + README + 평가 결과 + 보고서
    python scripts/upload_to_huggingface.py --repo-id pathcosmos/frankenstallm --create-pr

    # 위 + 소스 코드 + 데이터 스크립트 (모델/데이터/소스 전부)
    python scripts/upload_to_huggingface.py --repo-id pathcosmos/frankenstallm --with-source --with-data --create-pr

    # 평가·보고서만
    python scripts/upload_to_huggingface.py --repo-id pathcosmos/frankenstallm --readme-only --create-pr
"""

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HF_CHECKPOINT = PROJECT_ROOT / "outputs" / "hf_checkpoint-best-fixed"
REPORTS_DIR = PROJECT_ROOT / "reports"
EVAL_RESULTS_DIR = PROJECT_ROOT / "eval" / "results" / "frankenstallm-3b-v2"
DATA_DIR = PROJECT_ROOT / "data"
SOURCE_DIRS = ["train", "model", "configs", "scripts", "tokenizer", "eval"]


def main():
    parser = argparse.ArgumentParser(description="Upload model, eval reports, source, and data scripts to Hugging Face")
    parser.add_argument("--repo-id", type=str, required=True, help="e.g. pathcosmos/frankenstallm")
    parser.add_argument("--readme-only", action="store_true", help="Only push README + eval results + reports (no model)")
    parser.add_argument("--create-pr", action="store_true", help="Create a Pull Request instead of pushing to main")
    parser.add_argument("--with-source", action="store_true", help="Upload full source code (train, eval, model, configs, scripts, tokenizer)")
    parser.add_argument("--with-data", action="store_true", help="Upload data scripts and DATA_README (no .bin files)")
    parser.add_argument("--models-only", action="store_true", help="Only upload model files (HF checkpoint + GGUF + sampling_config + README), no eval/reports")
    args = parser.parse_args()
    create_pr = getattr(args, "create_pr", False)
    models_only = getattr(args, "models_only", False)

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("Install: pip install huggingface_hub")
        raise SystemExit(1)

    api = HfApi()

    # 레포 없으면 생성
    # 레포가 없으면 생성 (본인 계정일 때만 성공)
    try:
        create_repo(args.repo_id, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"Note: create_repo skipped (use Hugging Face website to create repo if needed): {e}")

    if not args.readme_only:
        if not HF_CHECKPOINT.exists():
            print(f"Checkpoint not found: {HF_CHECKPOINT}")
            raise SystemExit(1)
        print(f"Uploading model from {HF_CHECKPOINT} ...")
        api.upload_folder(
            folder_path=str(HF_CHECKPOINT),
            repo_id=args.repo_id,
            repo_type="model",
            create_pr=create_pr,
        )
        print("Model upload done.")

        # GGUF (v2 F16 + Q4_K_M)
        GGUF_DIR = PROJECT_ROOT / "outputs" / "gguf"
        for name in ["frankenstallm-3b-v2-f16.gguf", "frankenstallm-3b-v2-Q4_K_M.gguf"]:
            path = GGUF_DIR / name
            if path.exists():
                print(f"Uploading {name} ...")
                api.upload_file(
                    path_or_fileobj=str(path),
                    path_in_repo=f"gguf/{name}",
                    repo_id=args.repo_id,
                    repo_type="model",
                    create_pr=create_pr,
                )
        print("GGUF upload done.")

    # Sampling config (체크포인트 폴더에 있으면 루트에도 복사 업로드)
    sampling_config = HF_CHECKPOINT / "sampling_config.json"
    if sampling_config.exists():
        print("Pushing sampling_config.json ...")
        api.upload_file(
            path_or_fileobj=str(sampling_config),
            path_in_repo="sampling_config.json",
            repo_id=args.repo_id,
            repo_type="model",
            create_pr=create_pr,
        )
        print("Sampling config upload done.")

    # README는 체크포인트 폴더 것 사용 (이미 평가 요약 포함)
    readme_src = HF_CHECKPOINT / "README.md"
    if readme_src.exists():
        print("Pushing README (model card) ...")
        api.upload_file(
            path_or_fileobj=str(readme_src),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="model",
            create_pr=create_pr,
        )
        print("README upload done.")
    else:
        print("No README.md in checkpoint dir; skipping README push.")

    if models_only:
        print("Models-only mode: skipping eval results and reports.")
        print(f"Done. https://huggingface.co/{args.repo_id}")
        return

    # 평가 결과 JSON
    results_json = EVAL_RESULTS_DIR / "ollama_benchmark_results.json"
    if results_json.exists():
        print("Pushing ollama_benchmark_results.json ...")
        api.upload_file(
            path_or_fileobj=str(results_json),
            path_in_repo="eval/ollama_benchmark_results.json",
            repo_id=args.repo_id,
            repo_type="model",
            create_pr=create_pr,
        )
        print("Eval results upload done.")

    # 배포·평가 보고서 (상세 기록)
    for name, src in [
        ("2026-03-09_GGUF_DEPLOYMENT_AND_EVAL_REPORT.md", REPORTS_DIR / "2026-03-09_GGUF_DEPLOYMENT_AND_EVAL_REPORT.md"),
        ("2026-03-09_ORPO_EVALUATION_REPORT.md", REPORTS_DIR / "2026-03-09_ORPO_EVALUATION_REPORT.md"),
    ]:
        if src.exists():
            print(f"Pushing {name} ...")
            api.upload_file(
                path_or_fileobj=str(src),
                path_in_repo=f"eval_reports/{name}",
                repo_id=args.repo_id,
                repo_type="model",
                create_pr=create_pr,
            )
    print("Reports upload done.")

    # ---------- 소스 코드 (--with-source) ----------
    if getattr(args, "with_source", False):
        print("Uploading source code ...")
        ignore_common = ["**/__pycache__/**", "**/*.pyc", "**/.DS_Store"]
        for dirname in ["train", "model", "configs", "scripts", "tokenizer"]:
            src_dir = PROJECT_ROOT / dirname
            if src_dir.exists():
                api.upload_folder(
                    folder_path=str(src_dir),
                    path_in_repo=f"source/{dirname}",
                    repo_id=args.repo_id,
                    repo_type="model",
                    ignore_patterns=ignore_common,
                    create_pr=create_pr,
                )
                print(f"  source/{dirname}/ done.")
        # eval: outputs/, results/ 제외 (대용량). 허용 확장자만 업로드.
        eval_dir = PROJECT_ROOT / "eval"
        if eval_dir.exists():
            api.upload_folder(
                folder_path=str(eval_dir),
                path_in_repo="source/eval",
                repo_id=args.repo_id,
                repo_type="model",
                ignore_patterns=ignore_common + [
                    "outputs/**",
                    "results/**",
                    ".compile_cache/**",
                    "**/phase1_*.json",
                    "**/phase2_*.json",
                    "**/phase1_*.log",
                    "**/phase2_*.log",
                    "**/*.safetensors",
                    "**/training_curve.json",
                    "**/sft_eval_summary.json",
                    "**/orpo_eval_summary.json",
                ],
                create_pr=create_pr,
            )
            print("  source/eval/ done.")
        # 루트 문서
        for name in ["README.md", "CLAUDE.md", "requirements.txt", "PROGRESS.md"]:
            src_file = PROJECT_ROOT / name
            if src_file.exists():
                api.upload_file(
                    path_or_fileobj=str(src_file),
                    path_in_repo=f"source/{name}",
                    repo_id=args.repo_id,
                    repo_type="model",
                    create_pr=create_pr,
                )
        for p in PROJECT_ROOT.glob("PLAN_*.md"):
            api.upload_file(
                path_or_fileobj=str(p),
                path_in_repo=f"source/{p.name}",
                repo_id=args.repo_id,
                repo_type="model",
                create_pr=create_pr,
            )
        print("Source upload done.")

    # ---------- 데이터 스크립트 (--with-data, .bin 제외) ----------
    if getattr(args, "with_data", False) and DATA_DIR.exists():
        print("Uploading data scripts (no .bin) ...")
        api.upload_folder(
            folder_path=str(DATA_DIR),
            path_in_repo="data",
            repo_id=args.repo_id,
            repo_type="model",
            ignore_patterns=[
                "**/*.bin",
                "**/*.chunk*",
                "**/__pycache__/**",
                "**/code/**",
                "**/*.pyc",
            ],
            create_pr=create_pr,
        )
        print("Data scripts upload done.")

    print(f"Done. https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
