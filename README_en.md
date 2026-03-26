# FRANKENSTALLM

> 🌍 [한국어 버전 (Korean)](./README.md)

![Phase 4](https://img.shields.io/badge/Phase_4-Deployment_Complete-brightgreen)
[![Model](https://img.shields.io/badge/Model-3B_Korean_LLM-green)](https://huggingface.co/pathcosmos/frankenstallm)
![GPU](https://img.shields.io/badge/GPU-8×_NVIDIA_B200-76b900)
![FP8](https://img.shields.io/badge/Precision-MXFP8-orange)
![ORPO](https://img.shields.io/badge/ORPO-Complete_eval__loss_1.625-success)
[![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-Deployed-ff9900)](https://huggingface.co/pathcosmos/frankenstallm)

> **Building a Korean 3B LLM from scratch on 8× NVIDIA B200 GPUs.**
> Stitching pieces together like Frankenstein, forging them as strong as steel.

GitHub: [`pathcosmos/FRANKENSTALLM`](https://github.com/pathcosmos/FRANKENSTALLM)
🤗 HuggingFace: [`pathcosmos/frankenstallm`](https://huggingface.co/pathcosmos/frankenstallm) — **Model deployed (GGUF + safetensors)**

---

## Table of Contents

1. [Why This Project](#1-why-this-project)
2. [Current Status — At a Glance](#2-current-status--at-a-glance)
3. [Hardware Environment](#3-hardware-environment)
4. [Project Structure](#4-project-structure)
5. [Project Journey Timeline](#5-project-journey-timeline)
6. [Model Architecture](#6-model-architecture)
7. [Training Data](#7-training-data)
8. [Training Configuration & Optimization](#8-training-configuration--optimization)
9. [Experiment Results — 1B Baseline](#9-experiment-results--1b-baseline)
10. [Experiment Results — 3B Base Comprehensive Evaluation (v2)](#10-experiment-results--3b-base-comprehensive-evaluation-v2)
    - [10.1 Training Curves](#101-training-curves)
    - [10.2 PPL (Perplexity) — 19 Datasets](#102-ppl-perplexity--19-datasets)
    - [10.3 Korean Benchmarks](#103-korean-benchmarks)
    - [10.4 English Benchmarks](#104-english-benchmarks)
    - [10.5 Calibration](#105-calibration)
    - [10.6 0-shot vs 5-shot Comparison](#106-0-shot-vs-5-shot-comparison)
    - [10.7 Reference Model Comparison](#107-reference-model-comparison)
    - [10.8 Generation Quality & Parameter Grid Search](#108-generation-quality--parameter-grid-search)
    - [10.9 Evaluation Pipeline](#109-evaluation-pipeline)
11. [Experiment Results — 3B SFT Comprehensive Evaluation](#11-experiment-results--3b-sft-comprehensive-evaluation)
    - [11.1 SFT Training Results](#111-sft-training-results)
    - [11.2 6-Dimension Evaluation Summary](#112-6-dimension-evaluation-summary)
    - [11.3 Base vs SFT Comparison](#113-base-vs-sft-comparison)
    - [11.4 Code Improvements](#114-code-improvements)
    - [11.5 ORPO Progression Decision](#115-orpo-progression-decision)
12. [Phase 3 — ORPO (Preference Alignment)](#12-phase-3--orpo-preference-alignment)
20. [HuggingFace Deployment Status](#20-huggingface-deployment-status)
21. [Ollama Usage — Detailed Guide & Notes](#21-ollama-usage--detailed-guide--notes)
22. [Model Performance Comparison](#22-model-performance-comparison--base--sft--orpo--ollama)
23. [Reproduction Guide — Full Stage Configuration Details](#23-reproduction-guide--full-stage-configuration-details)
    - [12.1 Why ORPO](#121-why-orpo)
    - [12.2 Data](#122-data)
    - [12.3 HP Sweep Design](#123-hp-sweep-design-6-config)
    - [12.4 Attempt History](#124-attempt-history--5-failures)
    - [12.5 Sweep Results](#125-sweep-results-in-progress)
    - [12.7 ORPO Main Training](#127-orpo-main-training-in-progress-2026-03-09)
    - [12.8 ORPO Comprehensive Evaluation Pipeline](#128-orpo-comprehensive-evaluation-pipeline)
13. [How to Run](#13-how-to-run)
14. [Roadmap](#14-roadmap)
15. [References](#15-references)
16. [Tech Stack Summary](#16-tech-stack-summary)
17. [Related Projects](#related-projects)
18. [Next Optimization Plans](#18-next-optimization-plans--mfu-335--47-target)
19. [GPU Hardware & Cost Analysis](#19-gpu-hardware--cost-analysis--3b--60b-pretrain)

---

## 1. Why This Project

The Korean LLM ecosystem is growing rapidly. However, most publicly available models are either built on top of English pretraining with Korean fine-tuning layered on, or their training processes are not disclosed, making reproduction impossible.

This project is different.

- **From scratch**: Every stage is implemented directly — from tokenizer training to pretraining, SFT, and preference alignment.
- **Fully transparent builder log**: We don't just record successes. Bugs, failures, judgment errors, and their root cause analyses are all documented.
- **Practical scale**: Neither a toy model for academic papers (125M) nor an unreproducible 70B that only a research lab can train — the goal is a practical Korean model at the **3B scale**.
- **B200 optimization**: Maximizing NVIDIA B200's FP8 Tensor Cores, NVLink 5.0, and FlashAttention-2. The process of squeezing the most out of cutting-edge hardware is itself a learning experience.

This README is not a presentation of a finished product — it is **a builder's log in progress**.

---

## 2. Current Status — At a Glance

```
As of 2026-03-09
```

| Phase | Status | Details |
|-------|--------|---------|
| Phase 0: Foundation | ✅ Complete | OOM fix, GQA FA optimization, NCCL NVLS, pipeline setup |
| Phase 1: 3B Pretrain | ✅ Complete | 57,000 steps, loss 1.466, ~63 hours |
| Phase 2: SFT | ✅ Complete | 25,500 steps (early stop), val_loss 1.8851, ~15.5 hours |
| Phase 2.5: SFT Evaluation | ✅ Complete | 6-dimension evaluation 4/6 PASS, decided to proceed with ORPO |
| Phase 3: ORPO Sweep | ✅ Complete | 6-config sweep done, best: lr=1.2e-5, beta=0.25 |
| **Phase 3: ORPO Main Training** | **✅ Complete** | **9,997 steps early convergence, eval_loss 1.625, pref_acc 76.02%, 7/10 PASS** |
| **Phase 4: GGUF Conversion & Deployment** | **✅ Complete** | **byte-fallback fix, v1/v2 × 3 quantization variants each, HuggingFace + Ollama deployment** |

### Phase 2 (SFT) Final Results

| Metric | Value |
|--------|-------|
| Final step | **25,500 / 33,000** (77.3%, early stopping) |
| **Val loss (best)** | **1.8851** (step 23,000) |
| Training time | **~15 hours 41 minutes** (2026-03-05 22:15 ~ 2026-03-06 13:56) |
| VRAM usage | **24.2GB** / 183GB per GPU (13.2%) |
| Base model | checkpoint-0057000 (pretrain loss 1.466) |
| SFT data | **2,439,397 samples** (24 sources, 7.48 GB) |
| Incidents | 0 (no OOM, NCCL, or NaN issues) |

**SFT Val Loss Trajectory**:
```
Step     500: 2.073
Step   2,000: 1.956  (-0.117)
Step   5,000: 1.911  (-0.045)
Step  10,000: 1.892  (-0.019)
Step  15,000: 1.886  (-0.006)
Step  20,000: 1.885  (-0.001)
Step  23,000: 1.8851 ← BEST
Step  25,500: 1.8851 → Early Stop (patience 5/5)
```

### SFT 6-Dimension Evaluation Summary

| Dimension | Result | Key Metric |
|-----------|--------|------------|
| Perplexity (Knowledge Retention) | **PASS** | forgetting 0.9% |
| Generation Quality | **FAIL** | Greedy repetition rate 72.97% |
| Korean Benchmarks | **FAIL** | KoBEST average 43.26% |
| English Benchmarks | **PASS** | All tasks above lower bound |
| Calibration | **PASS** | Top-1 68.59% |
| SFT Chat Capability | **PASS** | EOS termination rate 60% (Base 0%) |

> **Decision: Proceed with ORPO** — Knowledge retention is excellent (0.9%), and repetition can be addressed via preference alignment.
> Details: `reports/2026-03-06_3B_SFT_COMPLETION_AND_EVAL_SUMMARY.md`

---

## 3. Hardware Environment

### GPU

| Spec | Value |
|------|-------|
| Model | 8× NVIDIA B200 |
| VRAM | 183GB HBM3e per GPU (~1.47TB total) |
| FP8 Tensor Core | 2,250 TFLOPS/GPU (18,000 TFLOPS total) |
| BF16 | 1,125 TFLOPS/GPU |
| HBM3e Bandwidth | ~7.67 TB/s per GPU |
| Interconnect | NVLink 5.0 (900 GB/s bidirectional per GPU) |
| Topology | NVSwitch — Single-hop All-to-All Mesh across all GPUs |
| Power | 940W measured / 1000W cap |

The B200 natively supports FP8. We train using `torch.float8_e4m3fn` combined with TransformerEngine's MXFP8 recipe. Compared to BF16, this theoretically doubles compute throughput while also improving memory efficiency.

### CPU & System Memory

| Spec | Value |
|------|-------|
| CPU | 2× AMD EPYC 9365 (Turin / Zen 5) |
| Physical Cores | 72 (36 cores × 2 sockets) |
| NUMA Configuration | 2 nodes: node0 (cores 0-35) / node1 (cores 36-71) |
| GPU↔NUMA Mapping | GPU 0-3 → NUMA node 0, GPU 4-7 → NUMA node 1 |
| RAM | 2.21TB DDR5 (~2.03TB free) |
| L3 Cache | 384MB (12 CCX × 32MB) |

**NUMA note**: During initial DDP launches, 5 out of 8 ranks ran on the wrong NUMA node. 69% of DataLoader workers were cross-NUMA. NUMA affinity optimization remains unapplied (roadmap item).

### Storage

| Path | Purpose | Free Space |
|------|---------|------------|
| `/PROJECT/0325120031_A/ghong/taketimes/llm-bang/` | Main workspace (checkpoints, data) | 2.2TB |
| `/home/ghong/` | Small code files | 5GB (limited) |

> **Note**: Checkpoints (tens of GB), training data (82GB+), and intermediate artifacts are all stored under the `/PROJECT/...` path. Risk of exceeding home directory quota.

### Software Environment

| Package | Version |
|---------|---------|
| PyTorch | `2.10.0a0+b4e4ee81d3.nv25.12` (NVIDIA custom) |
| FlashAttention | 2.7.4.post1+25.12 |
| TransformerEngine | 2.10.0 |
| NCCL | 2.28.9 |
| Triton | 3.5.1 |
| CUDA | 13.1 |
| Driver | 580.95.05 |

> **Warning**: PyTorch is a custom build optimized for NVIDIA B200. Reinstalling via `pip install torch` will break B200 optimizations. **Never reinstall.**

---

## 4. Project Structure

```
llm-bang/
├── CLAUDE.md                          # Claude Code guide
├── README.md                          # This file
├── PROGRESS.md                        # Progress log (daily entries)
├── Modelfile.3b                       # Ollama model file
│
├── configs/
│   ├── korean_3b_fp8.yaml             # 3B FP8 training config (currently in use)
│   ├── 3b_pretrain.yaml               # 3B pretrain config (alternative)
│   ├── korean_1b_fp8.yaml             # 1B FP8 config (archived)
│   ├── korean_3b_sft.yaml             # 3B SFT v1 config (complete)
│   ├── korean_3b_sft_v2.yaml          # 3B SFT v2 config (lr=5e-5, data mixing)
│   ├── korean_3b_orpo.yaml            # 3B ORPO config (lr=5e-6, beta=0.1)
│   ├── hybrid_3b.yaml                 # Hybrid 3B (Mamba-2 + Attention)
│   ├── small_fp8.yaml                 # 125M FP8 validation
│   ├── medium.yaml                    # Medium model config
│   └── small.yaml                     # Small model config
│
├── data/
│   ├── 3b_train.bin                   # Pretrain training data (82GB, 41.12B tokens)
│   ├── 3b_val.bin                     # Validation data (151MB)
│   ├── cc100_ko_train.bin             # CC100 Korean (4.5GB)
│   ├── cosmo_auto_math_text_train.bin # Math text (2.6GB)
│   └── build scripts, __init__.py
│
├── model/
│   ├── attention.py                   # GQA FlashAttention (Phase 0 optimizations applied)
│   ├── transformer.py                 # Main transformer architecture
│   ├── config.py                      # Model config dataclass
│   └── layers.py                      # Custom layers (RMSNorm, SwiGLU, etc.)
│
├── train/
│   ├── pretrain.py                    # Pretrain script (DDP optimized)
│   ├── sft.py                         # SFT training
│   ├── orpo.py                        # ORPO training
│   ├── trainer.py                     # Unified trainer (loss sync optimized)
│   └── utils.py                       # Utilities (NCCL 7200s timeout, etc.)
│
├── scripts/
│   ├── launch_3b_pretrain.sh          # 3B pretrain launcher (with NCCL env vars)
│   ├── launch_3b_sft.sh               # 3B SFT v1 launcher
│   ├── launch_3b_sft_v2.sh            # 3B SFT v2 launcher (data mixing)
│   ├── launch_3b_orpo.sh              # 3B ORPO launcher
│   ├── monitor_3b.sh                  # Real-time training monitor
│   ├── training_watchdog.sh           # Watchdog (10-min interval, cron)
│   ├── convert_3b_gguf.sh             # GGUF conversion script
│   ├── deploy_3b_ollama.sh            # Ollama deployment
│   ├── quality_gate.sh                # Pre-deployment quality gate
│   ├── telegram_notify.py             # Telegram notifications (urllib, curl blocked)
│   └── hourly_status.sh               # Hourly status report
│
├── eval/
│   ├── debate/
│   │   └── justice_league_3b_case.md  # 3B transition argument (Justice League multi-agent)
│   ├── decision/
│   │   └── FINAL_DECISION_REPORT.md   # SFT restart verdict
│   ├── plan/
│   │   └── 3B_MASTER_PLAN.md          # 3B master plan
│   ├── tasks/                         # Modularized evaluation tasks
│   │   ├── task_runner.py             # 8-GPU parallel task runner
│   │   ├── ppl_task.py                # Perplexity evaluation task
│   │   ├── lm_eval_task.py            # lm-evaluation-harness wrapper
│   │   ├── calibration_task.py        # Calibration analysis
│   │   ├── generation_task.py         # Generation quality + parameter grid search
│   │   └── token_nll_task.py          # Token NLL distribution analysis
│   ├── outputs/                       # Evaluation results (auto-generated, .gitignore)
│   ├── full_eval_pipeline.py          # v2 comprehensive evaluation pipeline (8-GPU parallel)
│   ├── sft_eval_pipeline.py           # SFT 6-dimension evaluation pipeline
│   ├── reeval_pipeline.py             # Re-evaluation pipeline (0+5-shot sequential)
│   ├── report_generator.py            # Markdown report auto-generation
│   ├── comprehensive_eval.py          # v1 comprehensive evaluation (legacy)
│   └── test_generation_params.py      # Generation parameter exploration
│
├── tokenizer/
│   ├── korean_sp/                     # SentencePiece 64K model files
│   ├── tokenizer.json                 # HuggingFace format (2.4MB)
│   ├── train_sp_tokenizer.py          # Tokenizer training script
│   └── convert_sp_to_hf.py            # SentencePiece → HF conversion
│
├── checkpoints/                       # Model checkpoints (large, .gitignore)
│
├── docs/
│   ├── PROJECT_HISTORY.md             # Full project journey detailed log
│   └── 3B_WORKPLAN.md                 # 3B work plan
│
└── reports/
    ├── 2026-03-02_0200_FRANKENSTALLM_phase0_optimization_report.md
    ├── 2026-03-05_3B_BASE_EVALUATION_REPORT.md
    ├── 2026-03-05_3B_SFT_PROGRESS_REPORT.md   # SFT training report (Phase 2)
    ├── 2026-03-05_3B_NEXT_STEPS_REFERENCE.md
    ├── 2026-03-05_NEMOTRON_NANO_FEASIBILITY_STUDY.md
    ├── 2026-03-05_PPL_EVALUATION.md
    ├── 2026-03-05_BENCHMARK_RESULTS.md
    ├── 2026-03-05_GENERATION_QUALITY.md
    ├── 2026-03-06_3B_SFT_EVAL_PLAN.md         # SFT 6-dimension evaluation plan
    ├── 2026-03-06_3B_SFT_EVALUATION_REPORT.md  # SFT 6-dimension evaluation results
    └── 2026-03-06_3B_SFT_COMPLETION_AND_EVAL_SUMMARY.md  # SFT completion + code improvements summary
```

---

## 5. Project Journey Timeline

This section is the heart of this README. It honestly records not just results, but **why** certain decisions were made and **where** things went wrong.

---

### Day 1 (Feb 25) — First Spark: 125M FP8 Validation

The project started from a simple question: Does FP8 actually train stably on the B200?

We applied TransformerEngine's MXFP8 recipe to a 125M small model for validation. The conclusion: **it works stably**. Loss convergence was normal, and VRAM efficiency showed clear improvement over BF16. This validation was the first green light for the entire pipeline.

On the same day, infrastructure setup was also completed. The DDP 8-GPU environment, NCCL environment variables, checkpoint storage paths, and the initial Telegram notification system were all established.

---

### Day 1~2 (Feb 25~26) — 1B Pretrain: 34K Steps, PPL 5.67

Immediately after the 125M validation, we dove into 1B model pretraining.

- **Architecture**: d_model=2048, 24 layers, GQA 4:1, SwiGLU, RoPE
- **Data**: C4 Korean based
- **Training**: 34,000 steps, FP8, 8× B200 DDP

Final results:
- **Loss: 1.904**
- **PPL (C4 Korean): 5.67**

The numbers alone look decent. But when actually generating text, problems were visible — repetitive patterns, awkward sentence structures, context drift. Expected for a pretrain-only model. Time for SFT.

---

### Day 2 (Feb 26) — SFT v1: The 0.0 Disaster

We ran SFT. As soon as training started, the loss began dropping rapidly. At first, this seemed like a good sign.

Then the loss hit **0.0**.

Val loss was also 0.0. The generated output was complete garbage.

We found the root cause: a **label off-by-one bug**. Input tokens and label tokens were shifted by one position. The model wasn't actually predicting the next token — it was matching answers it already had access to. The 0.0 loss wasn't "perfect learning" — it was **data leakage (label leakage)**.

An entire day was wasted.

---

### Day 3 (Feb 27) — 5 Bugs, Root Cause Analysis

To analyze the failure, we conducted a **5-agent root cause analysis**. The conclusion: it wasn't just one bug. The entire SFT pipeline had problems.

5 critical bugs discovered:

| Bug | Symptom | Impact |
|-----|---------|--------|
| Static padding (no packing) | Short samples padded to max_len | GPU waste, training inefficiency |
| EOS token truncation | No EOS at end of responses | Model can't learn "end of sentence" |
| Single epoch | Data seen only once | Underfitting |
| No validation split | Cannot measure val_loss | Cannot detect overfitting |
| Data quality | Noise, duplicates, imbalance | Induces repetitive generation patterns |

The EOS truncation bug is particularly subtle. If the model never learns when to end a response, it endlessly repeats the same patterns or appends meaningless tokens during generation. This was one of the causes behind the 18% repetition rate.

---

### Day 3 (Feb 27) — SFT v2: Success, but 18% Repetition

After fixing all 5 bugs, we ran SFT v2.

- **val_loss: 2.2062** — reasonable level
- **Repetition rate: 18%** (with rep_penalty=1.1 applied)

Generation quality improved significantly compared to v1. However, 18% repetition was still too high. Increasing `rep_penalty` reduces repetition but also reduces generation diversity and makes outputs awkward. There are structural limits to what decoding parameters can fix.

KoBEST COPA scored 0.646. A decent number, but short of the target.

---

### Day 3 (Feb 27) — "Justice League vs Avengers": The 3B Transition Decision

A team debate erupted over the 18% repetition rate. The core question was:

> **Can ORPO fix the repetition, or do we need to move to 3B?**

To answer this, we conducted a **multi-agent debate** (codename: "Justice League vs Avengers"). Each agent took a different position and argued their case.

Key findings from the debate:

1. **18% repetition is a structural limitation of 1B parameters.** A 1B model cannot sufficiently capture long-range dependencies. Preference alignment like ORPO can partially help reduce repetition, but it doesn't address the root cause (insufficient parameters).

2. **Scaling law analysis**: Based on Chinchilla scaling laws and experimental data, a 3B model was estimated to reduce repetition rate to 5~8% on the same data.

3. **Cost-benefit analysis**: Investing in 3B pretraining yields better final model quality than investing ORPO effort on the 1B model.

**Conclusion: Transition to 3B.** Archive 1B and start 3B pretraining.

This decision is documented with the full argumentation in `eval/debate/justice_league_3b_case.md`.

---

### Day 3 (Feb 27) — Assembling 640GB+ of Data

As soon as the 3B transition was decided, the data pipeline kicked into gear. Far more data is needed compared to 1B (Chinchilla optimal ratio: 3B model × 20 = 60B tokens).

Final assembled data:
- **Total tokens**: 41.12B tokens (final binary file)
- **Raw data**: 640GB+ multilingual text
- **Sources**: C4 Korean, Namuwiki, Wikipedia Korean, korean_extra datasets

The preprocessed (tokenized, shuffled, binary-converted) `data/3b_train.bin` is 82GB. The validation set `data/3b_val.bin` is 151MB.

---

### Mar 2 — Phase 0: Defeating OOM & Optimization

When we first started 3B training, OOM (Out of Memory) occurred. Odd for 183GB VRAM with a 3B model, but there was a reason.

It was a **GQA FlashAttention implementation issue**. The KV cache expansion in GQA (Grouped-Query Attention) was unnecessarily copying memory. FlashAttention's native GQA support wasn't being properly utilized.

Optimizations performed in Phase 0:

| Optimization | Method | Effect |
|-------------|--------|--------|
| GQA FA Native | Use `flash_attn_varlen_func` native GQA path | VRAM 60.4GB → 48.3GB (**-20%**) |
| DDP Optimization | `gradient_as_bucket_view=True` | GPU-CPU sync overhead **-87.5%** |
| NCCL NVLS | Ring+Tree topology, NVLS enabled | AllReduce efficiency improved |
| Batch Size Analysis | Identified GPU 2,4,6 as NCCL relay nodes | bs=5 optimal, bs=6 risky |
| SIGHUP Protection | nohup+setsid + Python signal handler + emergency ckpt | Triple protection |
| Monitoring | Telegram Bot (B200Bot) + cron | 10-min watchdog, hourly status report |

**torch.compile test**: No effect (1.00x). The cause was TransformerEngine's opaque kernels triggering graph breaks, and the `/tmp` directory having a noexec flag that prevented compiled kernel caching. A time sink, but confirming "no effect" through actual measurement is also a result.

**Why bs=5**: In the NCCL ring topology, GPUs 2, 4, and 6 serve as relay nodes. These GPUs use roughly 11GB more than the others. At bs=5 there's headroom, but bs=6 pushes these relay GPUs too close to the 183GB boundary. We keep bs=5 for safety margin.

---

### Mar 2~Mar 5 — Phase 1: 3B Pretrain Complete

Phase 1 began after Phase 0 optimizations were complete.

Early metrics (step 3150):
- Loss: 2.38
- Throughput: 36K tok/s per rank
- System total: ~292K tok/s (8 GPUs)
- MFU: ~33.5%

33.5% MFU may initially look low. However, this is the number achieved with TE MXFP8 already optimized. It's the effective utilization against the theoretical peak (18,000 TFLOPS). Remaining optimization headroom includes QKV fusion (+8~12%), NUMA affinity (+4~9%), and FA2 native RoPE (+3~5%).

**Phase 1 Complete (2026-03-05)**:

- **57,000 steps completed**, final loss **1.466**
- 41.12B tokens processed, total training time ~63 hours
- Zero incidents (no SIGHUP, OOM, or NCCL issues)

Comprehensive evaluation results summary (reflecting v2 re-evaluation):

| Metric | Result |
|--------|--------|
| PPL (combined validation set) | 5.2263 (initial v1 evaluation: 5.709) |
| PPL (C4 Korean) | 5.717 |
| KoBEST average (5 tasks) | 43.69% |
| MMLU-KO average (6 categories) | 22.75% |
| HAE-RAE | 19.71% |
| winogrande / piqa | 50.59% / 52.50% |
| Calibration Top-1 | 68.75% |
| Greedy 3-gram repetition rate | 60.99% (expected to improve after SFT) |
| Optimal generation parameters | temp=0.7, rep_penalty=1.3 → 0% repetition |

**SFT progression decision**: Loss 1.466 signals healthy training completion. PPL/repetition/benchmarks are all areas SFT is designed to address. No signs of model architecture issues. → Proceed to Phase 2 SFT.

---

### Mar 5~ — Phase 2: 3B SFT Begins — 2.44M Samples, val_loss 1.956

Immediately after Phase 1 completion, we prepared large-scale SFT data and started training.

**Data Pipeline**:
- Collected **6.59M raw samples** from **24 sources**
- `prepare_sft_combined.sh`: Format unification (6 formats → messages), MD5 deduplication, 98:2 split
- `filter_sft_v2.py`: 5-stage quality filter (EOS strip, QA marker removal, length filter, 4-gram repetition filter)
- Final: **2,439,397 train + 49,801 val** (7.48 GB)

Data composition was balanced across reasoning/CoT (38%), Korean instructions (22.5%), English multi-purpose (16%), math (12%), and conversation/code (11.5%). This is a **15× scale-up** from the 1B SFT's 161K samples.

**SFT Design — Lessons Learned from 1B Failure**:

| 1B Lesson | 3B SFT Application |
|-----------|---------------------|
| Label off-by-one → loss=0 | Loss masking verification (prompt=-1, train on response only) |
| EOS truncation → can't terminate | Chat template `<\|user\|>...<\|assistant\|>...</s>` with EOS included |
| Static padding → GPU waste | Dynamic padding (64-token alignment) |
| No validation → can't detect overfitting | 49,801 val samples, eval every 500 steps |
| Data noise | 5-stage quality filter (absent in 1B) |
| 18% repetition rate | **NEFTune alpha=5.0** added (embedding noise injection) |

**Training Configuration**:
- LR: **1e-5** (1/15 of pretrain — to prevent catastrophic forgetting)
- Effective batch: 2 × 8 GPU × 4 accum = 64 sequences
- 33,000 steps (~3.3 epochs)
- MXFP8, gradient checkpointing, NCCL Ring+Tree

**Initial Results** (step 2,000, 6%):
- Val loss: 2.073 → 2.004 → 1.975 → **1.956** (monotonically decreasing)
- Train-Val gap ~0.1 (no overfitting signs)
- VRAM 24.2 GB (13.2%) — half of pretrain, very stable
- Grad norm steady at 1.0 (learning rate is appropriate)

Detailed report: `reports/2026-03-05_3B_SFT_PROGRESS_REPORT.md`

---

### Mar 6 — Phase 2 Complete: SFT Early Stopping (val_loss 1.8851)

SFT terminated via early stopping at **25,500 out of 33,000 steps**. Val loss reached 1.8851 at step 23,000, then training was automatically stopped after 5 consecutive evaluations with no improvement.

**Total training time**: ~15 hours 41 minutes (2026-03-05 22:15 ~ 2026-03-06 13:56)

This result is consistent with the cosine decay of LR 1e-5 effectively converging to 0 after step 20K. The model learned as much as it could under the given LR schedule.

---

### Mar 6 — SFT 6-Dimension Comprehensive Evaluation: 4/6 PASS → ORPO Decision

A 6-dimension comprehensive evaluation was performed on the SFT checkpoint (`checkpoint-best`, step 23000). Completed in 49 minutes 27 seconds.

**Key Results**:
- **Perplexity**: forgetting 0.9% (all 19 datasets PASS) — excellent knowledge retention
- **Repetition rate**: greedy 72.97% (worse than Base's 60.99%) — FAIL
- **EOS termination rate**: 0% → 60% — improved but below target (90%)
- **KoBEST**: 43.26% (nearly identical to Base's 43.69%) — FAIL
- **MMLU-KO**: 22.75% → 26.00% (+3.2pp) — partial improvement
- **Calibration**: Top-1 68.59% — PASS

**Decision**: The 72.97% greedy repetition rate cannot be resolved by SFT alone. However, since applying `rep_penalty=1.2` achieves 0% repetition, the correct path is to internalize this behavior through ORPO (preference alignment).

---

### Mar 6 — Code Improvements & ORPO Preparation

In parallel with SFT evaluation, numerous code improvements and Phase 3 preparations were completed:

| Change | Description | Impact |
|--------|-------------|--------|
| `train/sft.py` +238 lines | MixingDataLoader (SFT+pretrain interleaving), DDP rank 0 tokenization | Forgetting prevention, 8× memory savings |
| `train/trainer.py` +17 lines | DDP early stopping broadcast (hang prevention), patience 5→10 | DDP stability |
| `train/orpo.py` +30 lines | YAML config support, 3B defaults | ORPO execution ready |
| `eval/report_generator.py` +831 lines | Base vs SFT comparison report auto-generation | Evaluation automation |
| `eval/sft_eval_pipeline.py` new | SFT 6-dimension evaluation pipeline | Comprehensive evaluation |
| `eval/tasks/generation_task.py` +75 lines | Chat template, diversity metrics | SFT evaluation |
| `configs/korean_3b_sft_v2.yaml` new | SFT v2 config (lr=5e-5, data mixing 70/30) | Backup path |
| `configs/korean_3b_orpo.yaml` new | ORPO config (lr=5e-6, beta=0.1) | Phase 3 |

Details: `reports/2026-03-06_3B_SFT_COMPLETION_AND_EVAL_SUMMARY.md`

---

## 6. Model Architecture

### 1B (Archived)

| Spec | Value |
|------|-------|
| vocab_size | 64,000 |
| d_model | 2,048 |
| n_layers | 24 |
| n_heads | 16 |
| n_kv_heads | 4 (GQA 4:1) |
| d_ffn | 5,461 (SwiGLU) |
| Parameters | ~1.19B |
| Context | 2,048 |
| rope_theta | 500,000 |

### 3B (Current)

| Spec | Value |
|------|-------|
| vocab_size | 64,000 |
| d_model | 3,072 |
| n_layers | 28 |
| n_heads | 24 |
| n_kv_heads | 8 (GQA 3:1) |
| d_ffn | 8,192 (SwiGLU) |
| Parameters | ~3.0B |
| Context | 2,048 |
| rope_theta | 500,000 |

### Common Design Principles

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Normalization | Pre-norm RMSNorm | More training-stable than post-norm |
| Activation | SwiGLU FFN | Proven choice in the Llama family |
| Positional Encoding | RoPE (θ=500K) | Potential for long context extension |
| Attention | GQA (Grouped-Query Attention) | KV cache memory savings |
| Implementation | FlashAttention-2 | IO-aware, VRAM efficient |
| Precision | FP8 (MXFP8 via TransformerEngine) | Optimal B200 utilization |

### GQA Ratio Selection Rationale

1B uses GQA 4:1 (16 heads, 4 kv_heads), while 3B uses GQA 3:1 (24 heads, 8 kv_heads). The ratio was relaxed for 3B because, as parameter count increases, sacrificing attention quality becomes a net loss at the 3B scale. We referenced Mistral 7B (GQA 8:1) and Llama 3 (GQA 8:1).

### Why rope_theta=500,000

Increasing from the standard RoPE θ=10,000 to 500,000 reduces frequency interference at long contexts. This approach was adopted by Code Llama, Llama 3, and others. Since the current max_seq_len=2048, there's no immediate benefit, but it provides a foundation for future context extension fine-tuning.

---

## 7. Training Data

### 7.1 Tokenizer

| Item | Value |
|------|-------|
| Type | SentencePiece Unigram |
| Vocabulary Size | 64,000 |
| Korean Character Coverage | 99.95% |
| Location | `tokenizer/korean_sp/` |
| HF Format | `tokenizer/tokenizer.json` (2.4MB) |

A 64K vocabulary strikes a balance between 32K (too small, causing excessive Korean subword fragmentation) and 128K (too large, increasing embedding layer overhead). While Llama 3 (128K) and GPT-4 (100K) trend toward larger vocabularies, a 128K vocabulary on a 3B model would make the embedding layer disproportionately large in terms of parameter share.

### 7.2 Pretraining Data — Full Composition

Final training files: `data/3b_train.bin` (77GB, ~38.5B tokens) + `data/3b_val.bin` (145MB)

Per Chinchilla scaling law: 3B x 20 = **60B tokens** is optimal. Currently, 38.5B tokens are consumed over 57,000 steps (batch 5 x accum 8 x seq 2048 x 8 GPUs) with some repetition, which is a reasonable range for a first 3B training run.

#### Korean — Web Crawl

| Dataset | HuggingFace ID | Tokenized File | Size | Est. Tokens | Description |
|---------|---------------|----------------|------|-------------|-------------|
| C4 Korean | `allenai/c4` (ko subset) | `korean_c4_train.bin` | 15GB | ~7.5B | Google C4 Korean-filtered, large-scale clean web text |
| CC-100 Korean | `cc100` (ko subset) | `cc100_ko_train.bin` | 4.3GB | ~2.15B | Common Crawl-based monolingual corpus |
| HPLT Korean | `HPLT/hplt_monolingual_v2` (ko) | `hplt_ko_train.bin` | 15GB | ~7.5B | High Performance Language Technologies web data |

#### Korean — Encyclopedia

| Dataset | HuggingFace ID | Tokenized File | Size | Est. Tokens | Description |
|---------|---------------|----------------|------|-------------|-------------|
| Wikipedia Korean | `wikimedia/wikipedia` (20231101.ko) | `wikipedia_ko_train.bin` | 566MB | ~283M | Full Korean Wikipedia, structured formal prose |
| Wikipedia Korean (v2) | `wikimedia/wikipedia` (ko) | `korean_wiki_train.bin` | 500MB | ~250M | Separate Wikipedia version |
| Namuwiki | `heegyu/namuwiki-extracted` | `korean_namuwiki_train.bin` | 2.1GB | ~1.05B | Namuwiki extraction, rich in subculture and current affairs |
| Namuwiki 2023b | `heegyu/namuwiki-extracted` (2023b) | `namuwiki_2023b_train.bin` | 2.5GB | ~1.25B | 2023 updated snapshot |

#### English/Multilingual — Educational

| Dataset | HuggingFace ID | Tokenized File | Size | Est. Tokens | Description |
|---------|---------------|----------------|------|-------------|-------------|
| Cosmopedia Stories | `HuggingFaceTB/cosmopedia` | `cosmo_stories_train.bin` | 5.9GB | ~2.95B | Synthetic educational stories |
| Cosmopedia Web v2 | `HuggingFaceTB/cosmopedia` | `cosmo_web_v2_train.bin` | 2.7GB | ~1.35B | Web-based educational text |
| Cosmopedia Stanford | `HuggingFaceTB/cosmopedia` | `cosmo_stanford_train.bin` | 2.1GB | ~1.05B | Based on Stanford lectures |
| Cosmopedia WikiHow | `HuggingFaceTB/cosmopedia` | `cosmo_wikihow_train.bin` | 382MB | ~191M | WikiHow guides |
| Cosmopedia OpenStax | `HuggingFaceTB/cosmopedia` | `cosmo_openstax_train.bin` | 224MB | ~112M | Open textbooks |
| Cosmopedia Khan Academy | `HuggingFaceTB/cosmopedia` | `cosmo_khanacademy_train.bin` | 46MB | ~23M | Khan Academy |

#### English/Multilingual — Math & Science

| Dataset | HuggingFace ID | Tokenized File | Size | Est. Tokens | Description |
|---------|---------------|----------------|------|-------------|-------------|
| Open Web Math | `open-web-math/open-web-math` | `open_web_math_train.bin` | 4.8GB | ~2.4B | Math text extracted from the web |
| MathPile | `GAIR/MathPile` | `mathpile_train.bin` | 2.9GB | ~1.45B | Math textbooks, papers, and forums |
| Cosmopedia AutoMath | `HuggingFaceTB/cosmopedia` | `cosmo_auto_math_text_train.bin` | 2.5GB | ~1.25B | Synthetic math problems and solutions |

#### Korean — Mixed (Legacy Merged)

| Dataset | Tokenized File | Size | Est. Tokens | Description |
|---------|----------------|------|-------------|-------------|
| Initial mix (C4+Namu+Wiki) | `korean_train.bin` | 17GB | ~8.5B | Original mixed data used for 1B training |
| 125M validation | `train.bin` | 1.2GB | ~600M | Used for initial FP8 verification |

#### Unused Collected Data (korean_extra/ — 640GB+)

Collected across 39 subdirectories under `data/korean_extra/`, but only partially tokenized and merged — large-scale raw data:

| Category | Dataset | Description | Notes |
|----------|---------|-------------|-------|
| Web Crawl | CulturaX Korean | Large-scale multilingual web corpus, Korean subset | ~50B+ tokens |
| Web Crawl | FineWeb2 Educational Korean | Educationally quality-filtered web data | 234GB raw |
| Web Crawl | Korean Web Collection | KORMo web collection | 175GB raw |
| Web Crawl | OSCAR Korean | Multilingual web corpus, Korean subset | |
| Educational | Korean Textbooks | Korean textbook text | 45 subcategories |
| Educational | FinePDFs Educational Korean | PDF-based educational materials | |
| Legal | Korean Law | Korean legal text | 15GB |
| News | Korean News Archive | Korean news archive | |
| Public Corpus | Korean Public Corpus | KORMo public corpus | 26GB |
| Code | Code Pretrain | Programming code | |
| Academic | Academic Pretrain | Academic papers and reports | |
| General | SlimPajama | Lightweight version of RedPajama | |

> This data is planned for use in the Extended Pretrain (80-100B tokens) phase.

#### Pretraining Data Domain Proportions

```
+----------------------------------------------------------+
|           3b_train.bin Token Composition (~38.5B)         |
+----------------------------------------------------------+
| ████████████████████░░░░░░░░░░  Korean Web Crawl   44.7% |
| ██████████░░░░░░░░░░░░░░░░░░░░  Mixed Legacy       22.1% |
| ██████░░░░░░░░░░░░░░░░░░░░░░░░  Educational (EN)   14.7% |
| █████░░░░░░░░░░░░░░░░░░░░░░░░░  Math & Science     13.2% |
| ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░  Encyclopedia (KO)   5.3% |
+----------------------------------------------------------+
```

### 7.3 SFT Data — 2.44M Samples (Currently Training)

**24 sources** yielding 6.59M raw -> unified/deduplicated -> quality filtered -> **2,439,397 train + 49,801 val**

#### Major SFT Sources (Top 12, 96% of Total)

| # | Dataset | Samples | Size | Domain |
|---|---------|---------|------|--------|
| 1 | reasoning_r1_1.4m | 1,400,000 | 14.77 GB | Reasoning (CoT) |
| 2 | openhermes_2.5 | 1,001,551 | 1.82 GB | English general-purpose |
| 3 | AI-MO_NuminaMath-CoT | 859,494 | 2.51 GB | Math CoT |
| 4 | korean_instruction_mix | 515,911 | 1.39 GB | Korean mixed |
| 5 | lemon-mint_smol-koreantalk | 460,281 | 5.23 GB | Korean dialogue |
| 6 | open_korean_instructions | 375,159 | 0.73 GB | Korean instructions |
| 7 | magpie_reasoning_v2 | 249,922 | 3.99 GB | Reasoning (English) |
| 8 | magpie_reasoning_ko | 224,929 | 3.19 GB | Reasoning (Korean) |
| 9 | ultrachat_200k | 207,865 | 1.34 GB | Dialogue |
| 10 | kuotient_orca-math-ko | 193,789 | 0.61 GB | Math (Korean) |
| 11 | data/sft/train.jsonl (original) | 161,848 | 0.27 GB | Original SFT |
| 12 | kullm_v2 | 152,630 | 0.42 GB | Korean instructions |

Other 12 sources: DeepMath-103K, Evol-Instruct-Code-80k-ko, ShareGPT-74k-ko, evol-instruct-korean, alpaca-gpt4-korean, ko_wikidata_QA, Ko.WizardLM, KOR-OpenOrca-Platypus-v3, korean-writing-style-instruct, ko_lima, koalpaca_v1_1a, OpenAssistant_oasst1_ko

#### Data Processing Pipeline

```
24 sources (6.59M raw)
    | prepare_sft_combined.sh (format unification, MD5 deduplication, 98:2 split)
Unified: 2,559,492 train + 52,234 val (7.95 GB)
    | filter_sft_v2.py (5 stages: EOS strip, QA marker removal, length 50~20K, 4-gram repetition >30% removal)
Final: 2,439,397 train + 49,801 val (7.63 GB)  <- removal rate 4.69%
```

#### Domain Proportions

```
Reasoning/CoT        38.0%  ████████████████████████
Korean Instructions  22.5%  ██████████████
English General      16.0%  ██████████
Math                 12.0%  ████████
Dialogue/Code/Other  11.5%  ███████
```

### 7.4 Preference Data (for ORPO) — 795K Pairs

Total **795,468 preference pairs** (7.9GB, `data/preference/combined_preference.jsonl`)

| HuggingFace ID | Size | Domain | Format |
|---------------|------|--------|--------|
| `nayohan/preference-collection-ko-full` | 4.9GB | General preference evaluation | instruction + response_A/B + preference |
| `heegyu/orca-math-korean-preference-cleaned` | 1.6GB | Math reasoning | prompt + chosen + rejected |
| `kuotient/orca-math-korean-dpo-pairs` | 750MB | Math DPO | prompt + chosen + rejected |
| `maywell/ko_Ultrafeedback_binarized` | 394MB | Feedback-based alignment | prompt + winning/losing response |
| `tellang/yeji-preference-ko-v1` | 171MB | General preference | prompt + chosen + rejected |
| `jojo0217/korean_rlhf_dataset` | 137MB | RLHF pairs | prompt + chosen + rejected |
| `lemon-mint/korean-realqa-reasoning-v01-preference` | 58MB | QA reasoning | prompt + chosen + rejected |

**Filtering criteria**: minimum length 20 characters, EOS removal, format normalization before merging

> ORPO is only executed in Phase 3 if the repetition rate exceeds 5%. If the 3B model resolves the 1B model's structural repetition issues on its own, it can be deployed without ORPO.

### 7.5 Data Pipeline Summary

```
[HuggingFace / Web Collection]
        |
        v
+--- Raw Collection ----------------------------------------+
|  korean_extra/ (39 directories, 640GB+)                   |
|  sft_extra/ (27 directories, 1.08M samples)               |
|  preference/ (7 JSONL files, 795K pairs)                  |
+-----------------------------------------------------------+
        |
        v
+--- Tokenization (SentencePiece 64K) ---------------------+
|  tokenize_extra.py — auto format detection                |
|  (Arrow/Parquet/JSONL)                                    |
|  8 workers parallel processing, uint16 memmap (.bin)      |
|  output                                                   |
+-----------------------------------------------------------+
        |
        v
+--- Final Merge -------------------------------------------+
|  Pretrain: 3b_train.bin (77GB, ~38.5B tokens)             |
|  SFT:     sft_combined/train_filtered.jsonl               |
|           (7.48GB, 2.44M samples)                         |
|  ORPO:    preference/combined_preference.jsonl (7.9GB)    |
+-----------------------------------------------------------+
```

---

## 8. Training Configuration and Optimization

### Current Training Config (`configs/korean_3b_fp8.yaml`)

```yaml
model:
  vocab_size: 64000
  d_model: 3072
  n_layers: 28
  n_heads: 24
  n_kv_heads: 8
  d_ffn: 8192
  max_seq_len: 2048
  rope_theta: 500000.0

training:
  batch_size: 5
  gradient_accumulation_steps: 8
  learning_rate: 1.5e-4
  min_lr: 1.5e-5
  warmup_steps: 2000
  max_steps: 57000
  weight_decay: 0.1
  grad_clip: 1.0
  optimizer: adamw
  scheduler: cosine

fp8:
  enabled: true
  recipe: "mxfp8"
  use_transformer_engine: true

distributed:
  strategy: ddp
  gradient_as_bucket_view: true
  find_unused_parameters: false

nccl:
  timeout_seconds: 7200
  nvls_enabled: true
```

Effective batch size = `batch_size(5) x grad_accum(8) x num_gpus(8)` = **320**

LR schedule: warmup 2000 steps -> cosine decay -> min_lr=1.5e-5 (10% of max_lr)

### Optimization Lessons Learned from Phase 0

#### GQA FlashAttention Native

The single biggest VRAM saving optimization. The key insight is that FlashAttention natively supports GQA. Expanding KV heads to process them like MHA causes memory copies, but using the native path lets FlashAttention handle it internally.

```python
# Before (inefficient): KV expand -> process like MHA
k = k.repeat_interleave(n_heads // n_kv_heads, dim=1)
v = v.repeat_interleave(n_heads // n_kv_heads, dim=1)
out = flash_attn_func(q, k, v)

# After (native GQA): flash_attn handles GQA internally
out = flash_attn_func(q, k, v)  # q: [B, S, H, D], k/v: [B, S, Hkv, D]
# VRAM 60.4GB -> 48.3GB (-20%)
```

#### DDP Optimization

```python
# gradient_as_bucket_view=True: directly maps gradient tensors as views of bucket memory
# -> eliminates unnecessary memory copies, -87.5% GPU-CPU sync overhead
model = torch.nn.parallel.DistributedDataParallel(
    model,
    device_ids=[local_rank],
    gradient_as_bucket_view=True,
    find_unused_parameters=False,  # all parameters are used
)
```

**Note**: `static_graph=True` is NOT used. TransformerEngine's `te.Linear` requires a dynamic graph in some cases, and enabling static_graph causes runtime errors.

#### NCCL NVLS

```bash
export NCCL_ALGO=NVLSTree    # Enable NVLink SHARP (NVLS)
export NCCL_PROTO=Simple
export NCCL_P2P_DISABLE=0
export NCCL_TIMEOUT=7200     # Generous timeout for long backward passes
```

Since NVSwitch supports all-to-all single-hop communication, NVLSTree is more efficient than Ring topology.

#### SIGHUP Triple Defense

Session disconnection (SIGHUP) is critical for long-running training. A triple protection system was built:

```bash
# Layer 1: nohup + setsid (new session group)
nohup setsid torchrun --nproc_per_node=8 train/pretrain.py ... &

# Layer 2: Python signal handler (ignore SIGHUP at Python level)
import signal
signal.signal(signal.SIGHUP, signal.SIG_IGN)

# Layer 3: emergency checkpoint (save checkpoint even on SIGTERM)
def emergency_save(signum, frame):
    save_checkpoint(model, optimizer, step, "emergency")
    sys.exit(0)
signal.signal(signal.SIGTERM, emergency_save)
```

#### torch.compile — Test Result: No Effect

`torch.compile` was applied expecting a speedup, but the measured result was **1.00x (no effect)**. Two reasons:

1. TransformerEngine kernels are opaque, causing graph breaks. `torch.compile` optimizes Python computation graphs, but TE kernels sit outside that graph.
2. The `/tmp` directory has a `noexec` mount flag, preventing caching of compiled kernels.

**Lesson**: "Just try it" is less valuable than "understand why it would work first."

### Monitoring System

```
Telegram Notification System
+-- B200Bot (token configured)
+-- training_watchdog.sh -> cron every 10 minutes
|   +-- loss anomaly, process termination detection -> immediate alert
+-- hourly_status.sh -> cron every 1 hour
    +-- step, loss, speed, VRAM, ETA -> periodic report
```

```python
# curl is blocked, so urllib is used instead
import urllib.request, json

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    data = json.dumps({"chat_id": CHAT_ID, "text": message}).encode()
    req = urllib.request.Request(url, data=data,
                                  headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req)
```

---

## 9. Experimental Results — 1B Baseline

The 1B model experimental results are recorded honestly. Both successes and failures.

### Pretraining Results

| Metric | Value |
|--------|-------|
| Final Loss | 1.904 |
| PPL (C4 Korean) | 5.67 |
| Training Steps | 34,000 |
| Training Time | ~2 days |

### SFT v1 Results — Failure

| Metric | Value |
|--------|-------|
| val_loss | 0.0 (abnormal) |
| Cause | Label off-by-one bug (data leakage) |
| Conclusion | Completely discarded |

### SFT v2 Results — Partial Success

| Metric | Value |
|--------|-------|
| val_loss | 2.2062 |
| Repetition Rate | 18% (with rep_penalty=1.1) |
| kobest_copa | 0.646 |
| Conclusion | Functional but structurally limited |

### 3B Target Metrics (Scaling Law-based Predictions)

| Benchmark | 1B Current | 3B Target |
|-----------|-----------|-----------|
| kobest_copa | 0.646 | >0.72 |
| kobest_hellaswag | ~0.42 | >0.52 |
| Repetition Rate | 18% | <5% |
| PPL (C4 Korean) | 5.67 | <4.5 |

Scaling from 1B to 3B is not simply about increasing parameters. The model must be able to remember longer contexts and learn more diverse patterns for the repetition rate to structurally decrease. The 3B targets are predictions based on the Chinchilla scaling curve and benchmarks from similarly-sized models.

---

## 10. Experimental Results — 3B Base Comprehensive Evaluation (v2)

Comprehensive evaluation performed on checkpoint-0057000 after completing 3B pretraining.
The v2 re-evaluation used an 8-GPU parallel pipeline covering 13+ benchmarks, 0/5-shot comparison, calibration, and reference model comparison.
Total elapsed time: 256.6 seconds.

> **v1 -> v2 changes**: v1 (initial evaluation) only measured PPL on 3 datasets + 2 benchmarks (belebele/MMLU). v2 includes PPL on 19 datasets, 5 KoBEST tasks, full HAE-RAE, MMLU-KO across 6 categories, MMLU-EN across 61 subjects, 5 major English benchmarks, calibration, 0/5-shot comparison, and a 12-combination parameter grid search.

### 10.1 Training Curve

| Step | Loss | LR | Notes |
|------|------|----|-------|
| 10 | 11.657 | 1.50e-06 | Initial (warmup start) |
| 500 | 5.047 | 7.50e-05 | Warmup in progress |
| 2,000 | 2.851 | 3.00e-04 | Warmup complete, peak LR |
| 10,000 | 2.057 | 2.86e-04 | Stable descent |
| 30,000 | 1.789 | 1.61e-04 | Mid-training, entering epoch 1 |
| 57,000 | 1.466 | 3.00e-05 | Final (cosine min) |

> Throughput was stable at 36-38K tok/s throughout. Total training time approximately 63 hours.

### Base Model Backup

| Item | Value |
|------|-------|
| Original Checkpoint | `checkpoints/korean_3b_fp8_run1/checkpoint-0057000/` (34GB) |
| Backup | `checkpoints/korean_3b_fp8_run1/checkpoint-0057000_BASE_BACKUP/` |
| MD5 Verification | `4f493d7bcc843727d32453bb3a4e6b7d` (confirmed match) |
| HF Conversion | `eval/outputs/hf_3b_base/` (11GB safetensors) |

### 10.2 PPL (Perplexity) — 19 Datasets

**Main PPL (3b_val combined): 5.2263** (initial v1 evaluation: 5.709)

| Dataset | PPL | Bits/Token | Eval Tokens | Elapsed Time |
|---------|-----|-----------|-------------|-------------|
| korean_namuwiki | 25.88 | 4.694 | 6.5M | 63.7s |
| cc100_ko | 21.78 | 4.445 | 13.6M | 133.2s |
| namuwiki_2023b | 18.92 | 4.242 | 7.7M | 75.1s |
| val | 18.30 | 4.194 | 9.1M | 89.4s |
| korean_wiki | 11.84 | 3.565 | 1.6M | 15.5s |
| wikipedia_ko | 10.71 | 3.420 | 1.8M | 17.4s |
| korean | 7.02 | 2.811 | 53.5M | 521.6s |
| open_web_math | 6.93 | 2.792 | 15.7M | 153.5s |
| **korean_c4** | **5.72** | **2.515** | **45.4M** | **443.1s** |
| **3b (combined)** | **5.23** | **2.386** | **226.9M** | **2227.3s** |
| cosmo_web_v2 | 4.17 | 2.059 | 8.6M | 84.6s |
| cosmo_stories | 3.96 | 1.984 | 18.9M | 185.2s |
| cosmo_openstax | 3.87 | 1.951 | 0.7M | 7.2s |
| cosmo_stanford | 3.36 | 1.750 | 6.6M | 65.3s |
| cosmo_wikihow | 3.31 | 1.727 | 1.2M | 11.8s |
| cosmo_auto_math_text | 3.15 | 1.655 | 7.9M | 77.3s |
| cosmo_khanacademy | 2.93 | 1.552 | 0.1M | 1.5s |
| mathpile | 2.72 | 1.446 | 7.1M | 69.9s |
| hplt_ko | 2.40 | 1.265 | 48.5M | 475.9s |

> **Interpretation**: In-distribution data (included in training) shows low PPL (hplt_ko: 2.40, mathpile: 2.72), while OOD data (low training weight) shows high PPL (cc100_ko: 21.78, namuwiki: 25.88) — an expected pattern. korean_c4 at 5.72 matches v1's 5.717, confirming evaluation reproducibility.

### 10.3 Korean Benchmarks

#### KoBEST (0-shot) — Average 43.69%

| Task | Accuracy | F1 |
|------|----------|-----|
| kobest_boolq | 50.28% | 0.3457 |
| kobest_copa | 49.30% | 0.4921 |
| kobest_hellaswag | 21.60% | 0.2153 |
| kobest_sentineg | 48.61% | 0.4737 |
| kobest_wic | 48.65% | 0.3286 |
| **Average** | **43.69%** | |

#### HAE-RAE (0-shot) — Overall 19.71%

| Subtask | Accuracy |
|---------|----------|
| haerae_general_knowledge | 21.59% |
| haerae_history | 23.40% |
| haerae_loan_word | 21.30% |
| haerae_rare_word | 18.77% |
| haerae_standard_nomenclature | 13.73% |
| **Overall** | **19.71%** |

#### MMLU-KO (0-shot) — 6-Category Average 22.75%

| Category | Accuracy |
|----------|----------|
| medical | 30.56% |
| humanities | 24.51% |
| business | 24.14% |
| social_sciences | 20.59% |
| other | 19.64% |
| stem | 19.57% |
| **Average** | **22.75%** |

> The base model is not optimized to solve multiple-choice benchmarks without instruction-following. KoBEST boolq/copa/sentineg/wic hover around ~50%, near the random baseline for binary/4-way multiple choice. Improvement is expected after SFT.

### 10.4 English Benchmarks

#### Major Benchmarks (0-shot)

| Task | Accuracy | Acc (norm) |
|------|----------|-----------|
| hellaswag | 26.00% | 26.15% |
| arc_easy | 25.63% | 26.64% |
| arc_challenge | 21.67% | 27.90% |
| winogrande | 50.59% | — |
| piqa | 52.50% | 48.31% |

> winogrande (50.59%) and piqa (52.50%) are binary choice tasks with a 50% random baseline. hellaswag/arc are 4-way choice with a 25% random baseline.

#### MMLU-EN (0-shot) — 61-Subject Average 25.81%

**Top 10 Subjects**:

| Subject | Accuracy |
|---------|----------|
| college_physics | 37.25% |
| college_computer_science | 34.00% |
| high_school_statistics | 33.80% |
| us_foreign_policy | 32.00% |
| security_studies | 31.43% |
| world_religions | 30.99% |
| professional_medicine | 30.88% |
| high_school_government_and_politics | 30.57% |
| jurisprudence | 30.56% |
| human_sexuality | 30.53% |

**Bottom 5 Subjects**:

| Subject | Accuracy |
|---------|----------|
| human_aging | 19.73% |
| college_biology | 19.44% |
| anatomy | 17.04% |
| global_facts | 17.00% |
| abstract_algebra | 15.00% |

### 10.5 Calibration

| Metric | Value |
|--------|-------|
| Top-1 Accuracy | 68.75% |
| Top-5 Accuracy | 81.64% |
| Top-10 Accuracy | 85.93% |
| Mean Correct Prob | 0.6152 |
| Mean Entropy | 1.5682 |

**Token NLL Distribution**:

| Statistic | Value |
|-----------|-------|
| Mean NLL | 1.5561 |
| Std Dev | 2.4926 |
| Median | 0.1221 |
| p95 | 7.0312 |
| p99 | 10.3125 |
| NLL > 5 ratio | 10.86% |
| NLL > 10 ratio | 1.18% |

> Top-1 at 68.75% means the model's most confident prediction is correct ~69% of the time. A median NLL of 0.12 (approx. e^0.12 = 1.13 PPL) indicates that most tokens are predicted with very high confidence, while a small number of high-difficulty tokens pull up the mean NLL — a typical distribution.

### 10.6 0-shot vs 5-shot Comparison

0-shot and 5-shot performance were compared across 18 Korean tasks.

| Task | 0-shot | 5-shot | Change |
|------|--------|--------|--------|
| global_mmlu_ko | 22.75% | 26.75% | **+4.00pp** |
| global_mmlu_ko_business | 24.14% | 31.03% | **+6.90pp** |
| global_mmlu_ko_humanities | 24.51% | 28.43% | +3.92pp |
| global_mmlu_ko_medical | 30.56% | 36.11% | **+5.56pp** |
| global_mmlu_ko_other | 19.64% | 23.21% | +3.57pp |
| global_mmlu_ko_social_sciences | 20.59% | 23.53% | +2.94pp |
| global_mmlu_ko_stem | 19.57% | 21.74% | +2.17pp |
| haerae | 19.71% | 20.26% | +0.55pp |
| haerae_general_knowledge | 21.59% | 22.73% | +1.14pp |
| haerae_history | 23.40% | 14.89% | -8.51pp |
| haerae_loan_word | 21.30% | 24.26% | +2.96pp |
| haerae_rare_word | 18.77% | 18.02% | -0.74pp |
| haerae_standard_nomenclature | 13.73% | 25.49% | **+11.76pp** |
| kobest_boolq | 50.28% | 50.21% | -0.07pp |
| kobest_copa | 49.30% | 46.80% | -2.50pp |
| kobest_hellaswag | 21.60% | 20.80% | -0.80pp |
| kobest_sentineg | 48.61% | 47.86% | -0.76pp |
| kobest_wic | 48.65% | 48.97% | +0.32pp |

**Average change: +1.80pp** | Improved: 12 | Declined: 6

> MMLU-KO consistently improves with 5-shot (+2-7pp), confirming that in-context learning capability is functional. KoBEST shows little change or slight decline — the model already performs pattern matching well at 0-shot, and few-shot examples may actually interfere. The +11.76pp for haerae_standard_nomenclature reflects learning the task's specific format from few-shot examples.

### 10.7 Reference Model Comparison

| Model | Parameters | MMLU-KO | MMLU-EN | KoBEST Avg | PPL |
|-------|-----------|---------|---------|------------|-----|
| **FRANKENSTALLM 3B** | **3B** | **22.75%** | **25.81%** | **43.69%** | **5.2263** |
| Llama-3.2-3B | 3B | ~42% | ~58% | ~55% | — |
| Qwen2.5-3B | 3B | ~48% | ~65% | ~60% | — |
| EXAONE-3.5-2.4B | 2.4B | ~35% | ~50% | ~50% | — |

> Reference models are results of training on trillions of tokens with thousands of GPU-hours. FRANKENSTALLM 3B was trained on 41.12B tokens (~68% of Chinchilla optimal), over 63 hours on 8 GPUs — context that should be considered. The gap is expected to narrow after SFT + extended pretraining (80-100B tokens).

### 10.8 Generation Quality and Parameter Grid Search

#### Repetition Rate Summary

| Setting | 3-gram Rep Rate | 4-gram Rep Rate |
|---------|----------------|----------------|
| greedy (temp=0.0) | 60.99% | 57.02% |
| temp=0.5 | 60.12% | 58.68% |
| temp=0.7 | 47.69% | 43.40% |
| temp=1.0 | 3.58% | 2.81% |

> The initial v1 evaluation's greedy 71.1% repetition rate was measured with `no_repeat_ngram_size=3` applied. In v2, measurements are standardized without it (raw), recording 60.99%.

#### 12-Combination Parameter Grid Search Results

| Setting | Temp | Rep Pen | 3-gram | 4-gram | Notes |
|---------|------|---------|--------|--------|-------|
| **t0.7_rep1.3** | **0.70** | **1.30** | **0.00%** | **0.00%** | **Optimal** |
| t0.9_rep1.2 | 0.90 | 1.20 | 0.00% | 0.00% | Runner-up |
| t0.7_rep1.2 | 0.70 | 1.20 | 0.88% | 0.00% | |
| t0.9_rep1.1 | 0.90 | 1.10 | 0.94% | 0.13% | |
| t1.0_rep1.1 | 1.00 | 1.10 | 1.21% | 0.48% | |
| t0.5_rep1.1 | 0.50 | 1.10 | 1.92% | 1.19% | |
| t1.0 | 1.00 | 1.00 | 3.58% | 2.81% | |
| t0.9 | 0.90 | 1.00 | 8.39% | 4.64% | |
| t0.7_rep1.1 | 0.70 | 1.10 | 8.51% | 5.51% | |
| t0.7 | 0.70 | 1.00 | 47.69% | 43.40% | |
| t0.5 | 0.50 | 1.00 | 60.12% | 58.68% | |
| greedy | 0.00 | 1.00 | 60.99% | 57.02% | |

#### Recommended Inference Parameters (for base experiments)

```python
# v2 grid search optimal values
temp=0.7, repetition_penalty=1.3
# or (for more diverse generation)
temp=0.9, repetition_penalty=1.2
```

> Adjusted upward from the initial v1 recommendation (`temp=0.9, top_p=0.9, no_repeat_ngram=3, repetition_penalty=1.1`) to `repetition_penalty=1.3`. The grid search confirmed that `repetition_penalty` alone is sufficient to suppress repetition, making `no_repeat_ngram_size` unnecessary.

### 10.9 Evaluation Pipeline

The v2 re-evaluation was performed using a modular 8-GPU parallel pipeline (`eval/reeval_pipeline.py`).

#### Architecture

```
reeval_pipeline.py
├── Single model load (HF model on GPU 0)
├── Phase 1: PPL evaluation (19 datasets, sequential)
├── Phase 2: Calibration + Token NLL
├── Phase 3: Generation quality + Parameter grid search (12 combinations)
├── Phase 4: lm-evaluation-harness (0-shot, 8-GPU parallel)
├── Phase 5: lm-evaluation-harness (5-shot, 8-GPU parallel)
└── Phase 6: Auto-generated reports (5 individual + 1 comprehensive)
```

#### Pipeline Mode

The model is loaded once to run 0-shot and 5-shot sequentially. This cuts model loading time in half compared to the previous approach (two separate processes).

#### Per-GPU Task Distribution

| GPU | 0-shot Tasks | 5-shot Tasks |
|-----|-------------|-------------|
| 0 | kobest_boolq, kobest_copa, kobest_hellaswag | Same |
| 1 | kobest_sentineg, kobest_wic | Same |
| 2 | haerae (full + 5 subtasks) | Same |
| 3 | global_mmlu_ko (6 categories) | Same |
| 4 | hellaswag, arc_easy | Same |
| 5 | arc_challenge, winogrande | Same |
| 6 | piqa, global_mmlu_en (61 subjects) | Same |
| 7 | (spare — dedicated to PPL/calibration) | — |

NUMA affinity applied: GPUs 0-3 on NUMA node 0 (cores 0-35), GPUs 4-7 on NUMA node 1 (cores 36-71).

**Total elapsed time: 256.6 seconds** (including model loading)

### SFT Go/No-Go Decision

**Conclusion: Proceed with SFT** — Loss 1.466 is a healthy completion signal with no structural issues. → **Phase 2 SFT started (2026-03-05)**

Detailed reports:
- v2 comprehensive: `eval/outputs/3b_reeval_20260305_1451/reports/` (5 individual reports + comprehensive)
- v1 legacy: `reports/2026-03-05_3B_BASE_EVALUATION_REPORT.md`

---

## 11. Experiment Results — 3B SFT Comprehensive Evaluation

A 6-dimensional comprehensive evaluation conducted after Phase 2 SFT completed via early stopping.

### 11.1 SFT Training Results

| Item | Value |
|------|-------|
| Final Step | 25,500 / 33,000 (77.3%, early stopping) |
| Best val_loss | **1.8851** (step 23,000) |
| Training Duration | ~15 hours 41 minutes |
| Data | 24 sources → 2,439,397 samples (7.48 GB) |
| Configuration | LR=1e-5, eff_batch=64, NEFTune alpha=5.0 |

**Val Loss Trajectory**:
```
Step     500: 2.0732 (warmup complete)
Step   2,000: 1.9558 (rapid descent)
Step   5,000: 1.9107 (stable convergence)
Step  10,000: 1.8917 (marginal decrease)
Step  15,000: 1.8864 (entering plateau)
Step  20,000: 1.8853 (variation < 0.001)
Step  23,000: 1.8851 ← BEST (early stopping reference point)
Step  25,500: Early Stop (patience 5/5 exhausted)
```

### 11.2 6-Dimensional Evaluation Summary

| # | Dimension | Result | Key Metrics |
|---|-----------|--------|-------------|
| 1 | Perplexity (Knowledge Retention) | **PASS** | Max forgetting 0.9%, all 19 datasets PASS |
| 2 | Generation Quality | **FAIL** | Greedy repetition rate 72.97% (target <5%), EOS 60% (target >90%) |
| 3 | Korean Benchmarks | **FAIL** | KoBEST avg 43.26% (target >55%) |
| 4 | English Benchmarks | **PASS** | hellaswag 26.1%, winogrande 50.8%, piqa 52.6% (all above lower bound) |
| 5 | Calibration | **PASS** | Top-1 68.59%, Top-5 81.55%, Entropy 1.54 |
| 6 | SFT Chat Capability | **PASS** | EOS termination rate 0%→60%, Chat template response |

### 11.3 Base vs SFT Comparison

| Metric | Base | SFT | Change | Verdict |
|--------|------|-----|--------|---------|
| PPL (aggregate) | 5.2263 | 5.2529 | +0.5% forgetting | PASS |
| Greedy 3-gram repetition rate | 60.99% | 72.97% | +12pp (worse) | FAIL |
| EOS termination rate | 0% | 60% | +60pp (major improvement) | Partial PASS |
| KoBEST avg | 43.69% | 43.26% | -0.4pp | FAIL |
| MMLU-KO | 22.75% | 26.00% | +3.2pp | Partial improvement |
| English benchmarks | — | — | within ±0.3pp | PASS (maintained) |
| Calibration Top-1 | 68.75% | 68.59% | -0.2pp | PASS (maintained) |

**Repetition Parameter Search** (promising):

| Setting | Repetition Rate | EOS Rate |
|---------|----------------|----------|
| t0.7_rep1.2 | **0.00%** | **100%** |
| t1.0_rep1.1 | **0.00%** | **100%** |
| greedy (raw) | 72.97% | 60% |

> With rep_penalty 1.1~1.3 applied, repetition rate reaches 0% → the model inherently possesses the ability to avoid repetition. This can be internalized via ORPO.

### 11.4 Code Improvements

Major code changes made during this Phase:

| File | Change | Lines | Purpose |
|------|--------|-------|---------|
| `train/sft.py` | MixingDataLoader, DDP rank 0 tokenization | +238 | SFT+pretrain interleaving, 8x memory savings |
| `train/trainer.py` | DDP early stop broadcast | +17 | Prevent DDP hang, patience 5→10 |
| `train/orpo.py` | YAML config, 3B defaults | +30 | ORPO execution preparation |
| `eval/report_generator.py` | SFT comparison report auto-generation | +831 | Evaluation automation |
| `eval/sft_eval_pipeline.py` | 6-dimensional evaluation pipeline | New | SFT comprehensive evaluation |
| `eval/tasks/generation_task.py` | Chat template, diversity metrics | +75 | SFT evaluation support |

### 11.5 ORPO Go/No-Go Decision

**Decision: Proceed to Phase 3 ORPO**

| Rationale | Details |
|-----------|---------|
| Good knowledge retention | Forgetting 0.9% — SFT did not destroy base knowledge |
| Repetition unresolved | Greedy 72.97% — preference alignment is the direct resolution path |
| Promising signal | 0% with rep_penalty → ORPO can internalize this |
| Data ready | 795,468 preference pairs (7.9 GB) |
| Code/config ready | `train/orpo.py` + `configs/korean_3b_orpo.yaml` |

**Post-ORPO Decision Criteria**:
- Repetition rate < 5% AND KoBEST > 50% → GGUF + Ollama deployment
- Repetition rate 5~15% → Hyperparameter tuning and retry
- Repetition rate > 15% → SFT v2 (lr=5e-5, data mixing) then retry

Details: `reports/2026-03-06_3B_SFT_COMPLETION_AND_EVAL_SUMMARY.md`

---

## 12. Phase 3 — ORPO (Preference Alignment)

### 12.1 Why ORPO

The SFT 6-dimensional evaluation revealed critical issues: 72.97% greedy repetition rate and 0% EOS termination rate. Since SFT only learns to "imitate good responses" without any signal to "suppress bad responses," preference optimization is essential for resolving the repetition problem.

**ORPO vs DPO**:
| Item | ORPO | DPO |
|------|------|-----|
| Reference model | Not required | Required (2x VRAM) |
| Implementation complexity | Low | Medium |
| Memory efficiency | High (only loads one 3B model) | Low (loads two 3B models) |
| Training stability | Medium | High |

ORPO was selected as the primary approach, with DPO as Plan B.

### 12.2 Data

- **Original**: 683,181 preference pairs (7 sources combined)
- **After filtering**: ~630,000 pairs (NaN prevention filter applied)
- **Eval split**: 5% (~31,500 pairs, seed=42)
- **Effective batch**: 4 × 8 GPU × 4 accum = 128

### 12.3 HP Sweep Design (6 Configs)

6 configurations selected using a center-axis fixed approach across 3 axes (beta, LR, max_length):

| Run | Name | Beta | LR | Max Length | Purpose |
|-----|------|------|----|-----------|---------|
| 1 | baseline_b015 | 0.15 | 8e-6 | 1536 | Weak beta baseline |
| 2 | baseline_b025 | 0.25 | 8e-6 | 1536 | Medium beta baseline |
| 3 | strong_b035 | 0.35 | 8e-6 | 1536 | Strong beta — aggressive repetition suppression |
| 4 | fast_lr12e6 | 0.25 | 1.2e-5 | 1536 | High LR — fast convergence |
| 5 | conserv_lr5e6 | 0.25 | 5e-6 | 1536 | Conservative LR — stability |
| 6 | short_1024 | 0.25 | 8e-6 | 1024 | Short max_length — VRAM savings |

200 steps each, eval_steps=100, 8xB200 DDP.

### 12.4 Trial History — 5 Failures

| # | Problem | Root Cause | Fix |
|---|---------|-----------|-----|
| 1 | NCCL Timeout | Tokenization 30min > timeout 1800s | ddp_timeout=7200, num_proc=64 |
| 2 | Config conflict | save_steps not a multiple of eval_steps | --no_load_best --save_steps 200 |
| 3 | Port collision + QKV missing | Zombie processes + fused QKV not split | pkill + QKV split logic |
| 4 | TRL NaN bug | tokenize_row truncates both responses simultaneously | Triple patch (clamp, truncation) |
| 5 | Tokenizer compatibility | zip(strict=True) + Korean merge ops | 8 TRL source patches |

The most severe was the TRL NaN bug, which caused a chain of 0 response tokens → log(0) = -inf → NaN propagation. Details: `reports/2026-03-08_ORPO_TRAINING_JOURNEY.md`

### 12.5 Sweep Final Results

| Run | Name | Beta | LR | MaxLen | Train Loss | Eval Loss | Margin | Status |
|-----|------|------|----|--------|-----------|-----------|--------|--------|
| 1 | baseline_b015 | 0.15 | 8e-6 | 1536 | 1.811 | 1.827 | 0.004 | ✅ |
| 2 | baseline_b025 | 0.25 | 8e-6 | 1536 | 1.890 | 1.906 | 0.009 | ✅ |
| 3 | strong_b035 | 0.35 | 8e-6 | 1536 | 2.055 | 1.985 | 0.007 | ✅ |
| **4** | **fast_lr12e6** | **0.25** | **1.2e-5** | **1536** | **1.917** | **1.862** | **0.009** | **Best** |
| 5 | conserv_lr5e6 | 0.25 | 5e-6 | 1536 | 1.833 | 1.910 | 0.004 | ✅ |
| 6 | short_1024 | 0.25 | 8e-6 | 1024 | 1.664 | 1.695 | 0.007 | ✅ |

**Best config: Run 4** (lowest eval_loss 1.862, highest margin 0.009, fast convergence).

### 12.6 Throughput Benchmark → Production Training Configuration

Before production training, throughput was measured across batch/grad_accum combinations to determine the optimal setting:

| batch_size | grad_accum | eff_batch | Throughput | Notes |
|-----------|-----------|----------|-----------|-------|
| **4** | **4** | **128** | **80.63 samples/s** | **Selected** |
| 2 | 8 | 128 | 73.14 samples/s | Previous setting |
| 8 | 2 | 128 | OOM | |

### 12.7 ORPO Production Training (In Progress, 2026-03-09)

| Parameter | Value |
|-----------|-------|
| Beta / LR | 0.25 / 1.2e-5 (Sweep Run 4) |
| Batch / Accum / Eff | 4 / 4 / 128 (benchmark optimal) |
| Max length | 1536 |
| Epochs | 2 (~9,840 steps) |
| GPU VRAM | ~52GB / 183GB (28%) |
| Speed | ~1.75 s/step |
| Estimated time | ~4.8 hours |

**Training Metrics Trajectory (as of step ~1,660)**:

| Step | Eval Loss | Pref Accuracy | Reward Margin | NLL Loss |
|-----:|----------:|--------------:|--------------:|---------:|
| ~1,000 | 1.791 | 66.8% | 0.107 | 1.647 |
| ~2,000 | 1.713 | 70.1% | 0.293 | 1.591 |
| ~3,000 | 1.681 | 71.9% | 0.372 | 1.567 |

- Train loss: 2.34 → **1.68** (-0.66)
- rewards/accuracies: 0.43 → **0.74** (rapid improvement in chosen/rejected discrimination)
- rewards/margins: -0.005 → **0.387** (confirmed preference signal learning)
- Speed ~1.76 s/step, GPU 92~100% utilization, stable progress

**Automatic evaluation after training completion**: `scripts/orpo_eval_watchdog.sh` monitors the training process and automatically triggers the 10-dimensional comprehensive evaluation pipeline upon completion

### 12.8 ORPO Comprehensive Evaluation Pipeline

A **10-dimensional comprehensive evaluation** that extends the SFT v2 6-dimensional evaluation with 4 ORPO-specific dimensions.
Upon training completion, `eval/orpo_eval_pipeline.py` automatically runs and generates a Base vs SFT vs ORPO 3-way comparison report.

**Evaluation Structure**:

| Phase | Content | GPU | Estimated Time |
|-------|---------|-----|---------------|
| Pre-phase | Extract training curves from train.log | - | ~1 sec |
| Phase 1 | Internal evaluation (PPL 19 datasets, Calibration, Generation, Repetition Grid) | 8 GPU parallel | ~30 min |
| Phase 2 | Benchmarks (KoBEST, HAE-RAE, MMLU-KO/EN, hellaswag, arc, piqa) | 8 GPU parallel | ~1 hour |
| Phase 3 | 3-way comparison report auto-generation | - | ~10 sec |

**10-Dimensional Evaluation Items**:

| # | Dimension | Criteria | SFT v2 Result | ORPO Target |
|---|-----------|----------|---------------|-------------|
| 1 | Knowledge Retention (PPL) | forgetting < 15% | 0.9% | < 5% |
| 2 | Generation Quality | greedy repetition < 5%, EOS > 90% | **72.97% / 60%** | **< 5% / > 90%** |
| 3 | Korean Benchmarks | KoBEST avg > 55% | 43.26% | >= 43% |
| 4 | English Benchmarks | Above lower bound | PASS | Maintain |
| 5 | Calibration | Top-1 >= 65% | 68.59% | >= 65% |
| 6 | Chat Capability | EOS termination rate | 60% | > 90% |
| 7 | Preference Accuracy | > 65% | — | > 65% |
| 8 | Reward Margins | > 0.1 | — | > 0.1 |
| 9 | Repetition Parameter Sensitivity | < 5% even at rep_penalty=1.0 | — | PASS |
| 10 | SFT→ORPO Improvement | repetition↓ + EOS↑ | — | PASS |

**Key Files**:
- `eval/orpo_eval_pipeline.py` — ORPO evaluation orchestrator
- `eval/report_generator.py` — 3-way comparison report generator (`generate_three_way_report()`)
- `scripts/orpo_eval_watchdog.sh` — Training completion detection + automatic evaluation trigger

**Deployment Criteria**: greedy repetition rate < 5% AND EOS > 90% AND forgetting < 5% AND KoBEST >= 43% → **DEPLOY**

---

## 13. How to Run

### Prerequisites

```bash
# Do NOT reinstall PyTorch (NVIDIA custom build)
# Only install the following additional packages
pip install transformers accelerate peft trl deepspeed \
            bitsandbytes sentencepiece wandb
```

### 3B Pretraining

```bash
# Run 8-GPU training with NCCL environment variables
bash scripts/launch_3b_pretrain.sh

# Manual execution (direct control)
torchrun --nproc_per_node=8 \
  --master_port=29500 \
  train/pretrain.py \
  --config configs/korean_3b_fp8.yaml
```

### SFT

```bash
bash scripts/launch_3b_sft.sh

# Or run directly
torchrun --nproc_per_node=8 \
  train/sft.py \
  --config configs/korean_3b_sft.yaml \
  --pretrain_ckpt checkpoints/3b_pretrain_best.pt
```

### ORPO (Preference Alignment)

```bash
# ORPO training
bash scripts/launch_3b_orpo.sh

# Automatic evaluation after training completion (watchdog)
nohup bash scripts/orpo_eval_watchdog.sh \
  > checkpoints/korean_3b_orpo_v1/watchdog.log 2>&1 &
```

### Evaluation

```bash
# Full base model evaluation (8 GPU parallel)
python eval/full_eval_pipeline.py

# SFT model evaluation (Base vs SFT 2-way comparison)
python eval/sft_eval_pipeline.py --skip-phase0 \
  --hf-model-path eval/outputs/hf_3b_sft_best

# ORPO model evaluation (Base vs SFT vs ORPO 3-way comparison)
python eval/orpo_eval_pipeline.py           # Automatically detects latest checkpoint
python eval/orpo_eval_pipeline.py --dry-run  # Preview execution plan only

# Quick evaluation (kobest_copa + PPL)
bash scripts/run_eval_quick.sh

# Generation parameter search
python eval/test_generation_params.py \
  --checkpoint checkpoints/3b_best.pt
```

### Deployment

```bash
# Step 1: GGUF conversion (llama.cpp format)
bash scripts/convert_3b_gguf.sh

# Step 2: Register and serve Ollama model
bash scripts/deploy_3b_ollama.sh

# Test with Ollama
ollama run frankenstallm-3b "Explain the steel industry in Korea."
```

### Training Monitoring

```bash
# Real-time monitor (tail -f style)
bash scripts/monitor_3b.sh

# Check process status
ps aux | grep pretrain

# GPU status
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv -l 5
```

### Single GPU Testing (Development/Debug)

```bash
python train/pretrain.py \
  --config configs/korean_3b_fp8.yaml \
  --device cuda:0 \
  --max_steps 100 \
  --debug
```

---

## 14. Roadmap

### Short-term (March 2026)

| Item | Status | Notes |
|------|--------|-------|
| Phase 1 (3B Pretrain) complete | ✅ Done | 57K steps, loss 1.466, 2026-03-05 |
| Phase 2 (SFT) complete | ✅ Done | 25.5K steps, val_loss 1.8851, 2026-03-06 |
| SFT 6-dimensional evaluation | ✅ Done | 4/6 PASS, ORPO decision |
| Phase 3 (ORPO Sweep) | ✅ Done | 6-config sweep complete, best config selected |
| **Phase 3 (ORPO Production Training)** | **In Progress** | **lr=1.2e-5, beta=0.25, 2 epochs, ~9,840 steps** |
| Phase 3.5 (ORPO Comprehensive Eval) | Pending | 10-dimensional eval (6 base + 4 ORPO-specific), 3-way comparison report |
| GGUF Conversion + Ollama Deployment | Pending | Phase 4 (if ORPO eval passes) |

### Mid-term (Q2 2026)

| Item | Notes |
|------|-------|
| Extended pretraining (80~100B tokens) | Reach Chinchilla-optimal point |
| QKV Fusion | +8~12% MFU expected |
| NUMA Affinity configuration | +4~9% expected |
| FA2 native RoPE | +3~5% expected |
| Context length extension (4096) | Based on RoPE theta=500K |

### Long-term (H2 2026)

| Item | Notes |
|------|-------|
| 7B experiment | Requires FSDP strategy |
| vLLM serving | Inference server based on PagedAttention |
| Domain-specific fine-tuning | Steel/manufacturing domain |
| Public release | Upload to HuggingFace Hub |

### Known Unapplied Optimizations

Optimizations discovered during Phase 0 analysis but not yet applied:

| Optimization | Expected Impact | Implementation Complexity |
|-------------|----------------|--------------------------|
| QKV Fusion | +8~12% MFU | Medium |
| NUMA Affinity | +4~9% | Low |
| FA2 Native RoPE | +3~5% | Low |
| HugePages | +1~3% (TLB optimization) | Low (sysctl) |

Applying all these optimizations could push MFU from the current 33.5% to 45~50%.

---

## 15. Reference Documents

| Document | Location | Content |
|----------|----------|---------|
| Full project journey | `docs/PROJECT_HISTORY.md` | Day-by-day detailed progress log |
| 3B work plan | `docs/3B_WORKPLAN.md` | Detailed phased work plan for 3B |
| Justice League debate | `eval/debate/justice_league_3b_case.md` | Full multi-agent debate transcript for 1B→3B transition |
| SFT restart verdict | `eval/decision/FINAL_DECISION_REPORT.md` | SFT v1 failure → v2 design verdict |
| 3B master plan | `eval/plan/3B_MASTER_PLAN.md` | Full training pipeline master plan |
| Phase 0 optimization report | `reports/2026-03-02_0200_FRANKENSTALLM_phase0_optimization_report.md` | Full VRAM/MFU optimization report |
| 3B Base evaluation report (v1) | `reports/2026-03-05_3B_BASE_EVALUATION_REPORT.md` | Initial PPL/benchmark/repetition evaluation |
| PPL evaluation report (v1) | `reports/2026-03-05_PPL_EVALUATION.md` | Detailed PPL across 4 validation sets |
| Benchmark results (v1) | `reports/2026-03-05_BENCHMARK_RESULTS.md` | Detailed belebele, MMLU results |
| Generation quality analysis (v1) | `reports/2026-03-05_GENERATION_QUALITY.md` | Repetition rate, decoding parameters |
| SFT training report | `reports/2026-03-05_3B_SFT_PROGRESS_REPORT.md` | Phase 2 SFT training process log |
| **SFT completion comprehensive report** | `reports/2026-03-06_3B_SFT_COMPLETION_AND_EVAL_SUMMARY.md` | **SFT completion + evaluation + code improvements + ORPO decision (latest)** |
| SFT evaluation plan | `reports/2026-03-06_3B_SFT_EVAL_PLAN.md` | 6-dimensional evaluation design |
| SFT evaluation results | `reports/2026-03-06_3B_SFT_EVALUATION_REPORT.md` | Detailed 6-dimensional evaluation results |
| 3B next steps reference | `reports/2026-03-05_3B_NEXT_STEPS_REFERENCE.md` | Post-SFT direction |
| Nemotron Nano feasibility | `reports/2026-03-05_NEMOTRON_NANO_FEASIBILITY_STUDY.md` | Hybrid architecture review |
| **v2 comprehensive evaluation report** | `eval/outputs/3b_reeval_20260305_1451/full_eval_report.md` | **13+ benchmarks comprehensive** |
| v2 PPL report | `eval/outputs/3b_reeval_20260305_1451/reports/01_perplexity_report.md` | Detailed PPL across 19 datasets |
| v2 Calibration report | `eval/outputs/3b_reeval_20260305_1451/reports/02_calibration_report.md` | Top-K accuracy, NLL distribution |
| v2 Generation quality report | `eval/outputs/3b_reeval_20260305_1451/reports/03_generation_quality.md` | 12-combination parameter grid search |
| v2 Benchmark report | `eval/outputs/3b_reeval_20260305_1451/reports/04_benchmark_report.md` | KoBEST, HAE-RAE, MMLU, 0/5-shot |
| Progress log | `PROGRESS.md` | Checkpoints, metrics, and decision log by date |
| **ORPO analysis and plan** | `reports/2026-03-07_ORPO_ANALYSIS_AND_PLAN.md` | **ORPO rationale, HP design, execution procedure** |
| **ORPO Sweep debug** | `reports/2026-03-08_ORPO_SWEEP_DEBUG_REPORT.md` | **QKV bug, NCCL timeout, TRL patches in detail** |
| **ORPO training journey** | `reports/2026-03-08_ORPO_TRAINING_JOURNEY.md` | **Full ORPO journey: 5 failures and HP sweep (latest)** |

---

## 16. Technology Stack Summary

| Area | Technology | Version |
|------|-----------|---------|
| Deep Learning Framework | PyTorch (NVIDIA custom build) | nv25.12 |
| Attention | FlashAttention-2 | 2.7.4.post1+25.12 |
| FP8 / Mixed Precision | TransformerEngine (MXFP8) | 2.10.0 |
| Distributed Training | DDP + NCCL (NVLS) | NCCL 2.28.9 |
| Kernel Compilation | Triton | 3.5.1 |
| Tokenizer | SentencePiece Unigram 64K | - |
| Monitoring | Telegram Bot (B200Bot) + cron watchdog | - |
| Inference Serving | GGUF + Ollama | - |
| GPU | 8x NVIDIA B200 (NVLink 5.0, NVSwitch) | CUDA 13.1 |
| CPU | 2x AMD EPYC 9365 (Zen 5) | - |

---

## Related Projects

### [EVAFRILL-Mo](https://github.com/pathcosmos/EVAFRILL-Mo) | [🤗 HuggingFace](https://huggingface.co/pathcosmos/EVAFRILL-Mo-3B)

**Hybrid Mamba-2 + Transformer Language Model** — A sister project of FRANKENSTALLM.

Inspired by the NVIDIA [Nemotron-H](https://arxiv.org/abs/2504.03624) architecture, this is a 3B hybrid model built from scratch. While FRANKENSTALLM is based on a pure Transformer architecture, EVAFRILL-Mo adopts a **Mamba-2 SSM + Sparse Transformer Attention** hybrid structure.

| Item | FRANKENSTALLM | EVAFRILL-Mo |
|------|:---:|:---:|
| Architecture | Pure Transformer (28L) | Mamba-2 24L + Attention 2L |
| Parameters | 3.17B | 2.94B |
| Core Technologies | GQA, FP8, FlashAttention-2 | Selective Scan, SwiGLU FFN in Mamba, GQA |
| Design Principle | Proven Transformer architecture | Nemotron-H segmented adoption |
| GPU | 8x B200 | 7x B200 |
| Training Strategy | Chinchilla-optimal | Targeting 93% of Chinchilla |

Both projects share the same tokenizer (64K SentencePiece), training data pipeline, and DDP/FP8 infrastructure. "Same ingredients, different recipes" — enabling comparative experiments on how architectural differences affect performance.

> *Name origin: Bride **Eva** (Bride of Frankenstein) + **FRI**DAY (Iron Man's AI assistant) + **LL**M + Nemotron's **Mo***

---

## 18. Next Optimization Plan — MFU 33.5% → 47% Target

> Detailed document: [`docs/NEXT_OPTIMIZATION_PLAN.md`](docs/NEXT_OPTIMIZATION_PLAN.md)

### Current Performance Diagnosis

Phase 1 pretraining measurements:
- **57,000 steps**, ~38.5B tokens, **approximately 63 hours**
- Throughput: 36~38K tok/s per rank → total **~292K tok/s** (8 GPUs)
- **MFU: ~33.5%**

### Key Bottleneck: NUMA Misalignment

```
AMD EPYC 9365 x 2 sockets:
  GPU 0~3 → NUMA node 0 (core 0-35)
  GPU 4~7 → NUMA node 1 (core 36-71)

During initial DDP launch, 5/8 ranks ran on the wrong NUMA node.
69% of DataLoader workers were cross-NUMA — causing ~2x latency.
```

### Expected Impact by Optimization

| Optimization | Expected MFU Gain | Difficulty |
|-------------|-------------------|-----------|
| NUMA affinity pinning | +4~9% | Low (launch script modification) |
| QKV fusion (TransformerEngine) | +8~12% | Medium (model code modification) |
| FA2 native RoPE | +3~5% | Medium (FA2 version dependent) |
| NCCL environment variable tuning | +1~2% | Low (one-line addition) |

### Expected Before/After Comparison

| Item | Current | After Optimization |
|------|---------|-------------------|
| MFU | 33.5% | ~45~47% |
| Throughput | 292K tok/s | ~390~410K tok/s |
| 50B token training | ~47 hours | ~34~36 hours |

### Ready-to-Apply Code

**NUMA affinity (launch script):**

```bash
numactl --cpunodebind=0 --membind=0 torchrun \
  --nproc_per_node=4 --node_rank=0 train/pretrain.py ... &
numactl --cpunodebind=1 --membind=1 torchrun \
  --nproc_per_node=4 --node_rank=1 train/pretrain.py ... &
```

**NCCL environment variables:**

```bash
export NCCL_MIN_NCHANNELS=4
export NCCL_SOCKET_NTHREADS=4
export CUDA_DEVICE_MAX_CONNECTIONS=1
```

> After Phase 3 ORPO completes, applying NUMA affinity before the next pretraining run can reduce training time by ~30%.

---

## 19. GPU Hardware & Cost Analysis — 3B x 60B Pretraining

> Detailed document: [`docs/GPU_COST_ANALYSIS.md`](docs/GPU_COST_ANALYSIS.md)

### Measured Baseline

```
FRANKENSTALLM Phase 1 measurements:
  B200 x 8, MFU 33.5%, 292K tok/s
  38.5B tokens → 63 hours
  Extrapolated for 60B tokens → approximately 98 hours
```

### Cloud Cost-Effectiveness Top 3 (60B tokens, after optimization)

| Rank | Configuration | Duration | Total Cost |
|------|---------------|----------|------------|
| 1 | H100x8 Cudo | 44.8hr | **$645** (~930K KRW) |
| 2 | H100x8 Vast.ai | 44.8hr | $670 (~970K KRW) |
| 3 | H100x8 RunPod | 44.8hr | $713 (~1.03M KRW) |

> B200 Blackwell is faster, but cloud pricing is 3x that of H100 → **H100 is 4.3x cheaper in total cost**

### Recommended Personal GPU Configurations

| Configuration | VRAM | NVLink | Price | Recommendation |
|---------------|------|--------|-------|----------------|
| A6000 Ada x 2 (Used) | 96GB (unified) | Yes | ~$7,000 | Best value |
| L40S x 2 | 96GB (unified) | Yes | ~$10,000 | Great |
| RTX Pro 6000 Blackwell | 96GB (single) | No | ~$8,500 | Good |

> Consumer GPUs (RTX 5090/4090) do not support NVLink. Professional-grade GPUs are required for 80GB+ unified memory.

### Recommended Strategy: Local + Cloud Hybrid

```
[Local] RTX 4090 x 4 (~$6,300) — Data preprocessing, experiments, SFT/ORPO
[Cloud] H100x8 (~$713/run) — Pretraining only
```

---

## 20. HuggingFace Deployment Status

> **Deployment URL**: https://huggingface.co/pathcosmos/frankenstallm

### Model File List

| File | Size | Description |
|------|------|-------------|
| `model.safetensors` | 4.76GB | v2 ORPO best (byte-fallback fix) — Direct loading via Transformers |
| `gguf/frankenstallm-3b-v2-Q4_K_M.gguf` | 757MB | **Recommended for Ollama** |
| `gguf/frankenstallm-3b-v2-Q8_0.gguf` | 1.2GB | High quality |
| `gguf/frankenstallm-3b-v2-f16.gguf` | 2.3GB | Highest quality |
| `gguf/frankenstallm-3b-Q4_K_M.gguf` | 1.9GB | v1 Q4_K_M |
| `gguf/frankenstallm-3b-Q8_0.gguf` | 3.2GB | v1 Q8_0 |
| `gguf/frankenstallm-3b-f16.gguf` | 6.0GB | v1 f16 |

Each GGUF file is accompanied by a corresponding `Modelfile.*` (with sampling config included).

---

## 21. Ollama Usage — Detailed Instructions and Caveats

### Quick Start (Recommended)

```bash
# 1. Download GGUF + Modelfile
huggingface-cli download pathcosmos/frankenstallm   gguf/frankenstallm-3b-v2-Q4_K_M.gguf   gguf/Modelfile.3b-v2-Q4_K_M   --local-dir ./frankenstallm

# 2. Update the FROM path in the Modelfile
# FROM ./outputs/gguf/frankenstallm-3b-v2-Q4_K_M.gguf
# → FROM ./frankenstallm/gguf/frankenstallm-3b-v2-Q4_K_M.gguf

# 3. Register the model with Ollama
ollama create frankenstallm-3b-v2 -f ./frankenstallm/gguf/Modelfile.3b-v2-Q4_K_M

# 4. Run
ollama run frankenstallm-3b-v2
```

### Validated Sampling Parameters

| Parameter | Recommended Value | Description |
|-----------|-------------------|-------------|
| `temperature` | **0.7** | Lower = more repetitive, higher = more random. 0.7 is optimal for Korean quality |
| `repeat_penalty` | **1.2** | **Required** — Without this, greedy decoding produces 30.89% repetition |
| `top_p` | **0.9** | Nucleus sampling, 0.9 or above recommended |
| `top_k` | **50** | Top 50 token candidates |
| `num_predict` | **512** | Maximum number of generated tokens |
| `num_ctx` | **4096** | Context window (max 4096) |

### Caveats

**1. repeat_penalty must be configured**
```
Even after ORPO training, greedy (temp=0) decoding still has 30.89% 3-gram repetition rate.
Setting repeat_penalty=1.2 completely suppresses repetition to 0%.
This is already configured in the Modelfile, so using the Modelfile applies it automatically.
```

**2. Caution with temperature=0 (greedy)**
```
Greedy decoding without repetition suppression results in 30.89% 3-gram repetition.
Always use temperature >= 0.5 combined with repeat_penalty >= 1.1.
```

**3. Performance degrades beyond num_ctx**
```
The model was trained with max_position_embeddings=4096.
Contexts exceeding 4096 tokens will significantly degrade performance.
```

**4. This is a Korean-centric model**
```
English capability: MMLU 42.0%, HellaSwag 27.9% — set low expectations for English tasks.
Korean: KoBEST 0-shot 52.75%, korean_nlu 100.0% (Ollama benchmark)
```

**5. Choosing between v2 and v1**
```
v2 is recommended: the byte-fallback fix makes inputs containing special characters like newlines safe.
v1 may crash llama.cpp when inputs contain newline characters.
```

### Direct Execution with Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "pathcosmos/frankenstallm"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# WARNING: do_sample=True + repetition_penalty must be set
inputs = tokenizer(
    "한국의 전통 음식 중 김치에 대해 설명해주세요.",  # "Explain about kimchi among Korean traditional foods."
    return_tensors="pt"
).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        do_sample=True,        # greedy decoding causes repetition
        temperature=0.7,       # key parameter
        repetition_penalty=1.2, # must be set
        top_p=0.9,
        top_k=50,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id,
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### API Serving (Ollama)

```bash
# Start background server (default port 11434)
ollama serve &

# REST API call
curl http://localhost:11434/api/generate -d '{
  "model": "frankenstallm-3b-v2",
  "prompt": "한국어로 자기소개를 해줘.",  // "Introduce yourself in Korean."
  "stream": false,
  "options": {
    "temperature": 0.7,
    "repeat_penalty": 1.2,
    "top_p": 0.9,
    "num_predict": 512
  }
}'
```

---

## 22. Model Performance Comparison — Base / SFT / ORPO / Ollama

### Key Metrics: 3-Way Comparison

| Metric | Base | SFT v2 | ORPO (v2) |
|--------|------|--------|-----------|
| **Greedy 3-gram repetition rate** | 60.99% | 72.97% | **30.89%** |
| **EOS termination rate (greedy)** | 0% | 60% | 67% |
| **Sampling repetition rate** (temp=0.7, rep=1.2) | — | — | **0%** |
| **KoBEST 0-shot average** | ~44% | 43.26%^1 | **52.75%** |
| **MMLU-KO 0-shot** | 38.8% | 42.0% | — |
| **HellaSwag EN** | 33.3% | — | 27.9% |
| **Calibration Top-1** | ~65% | 68.59% | 67.99% |
| **PPL forgetting** | 0% (baseline) | 0.9% | 4.1% |
| **Preference Accuracy** | — | — | **76.02%** |
| **Reward Margin** | — | — | **0.6100** |

> ^1 SFT initial evaluation: 43.26%; later re-evaluation: 52.75% (due to evaluation environment differences).

### KoBEST Detailed Comparison (0-shot)

| Task | Base | ORPO |
|------|------|------|
| BoolQ | ~48% | 54.3% |
| COPA | ~52% | 56.2% |
| WiC | ~49% | 51.8% |
| SentiNeg | ~44% | 51.4% |
| HellaSwag-KO | ~38% | 49.9% |
| **Average** | ~46% | **52.75%** |

### Ollama Benchmark (frankenstallm-3b-v2:Q4_K_M, 35 tests)

| Category | Score |
|----------|-------|
| korean_nlu | **100.0** |
| knowledge | 75.0 |
| instruction_following | 66.7 |
| reasoning | 50.0 |
| safety | 10.0 |
| repetition_resistance | 2.2 |
| **Auto-scored average** | **46.7** |
| Average TPS | 142.5 tok/s |
| Average TTFT | 16.7 ms |

> The repetition_resistance score of 2.2% was measured using Ollama default parameters (without sampling).
> With `repeat_penalty=1.2` + `temperature=0.7`, **repetition rate reaches 0%**.

### 3B-class Model Performance Comparison — Ollama Benchmark

Direct comparison with 3B-class open-source models in an identical environment (Ollama, 35 tests).

#### Overall Scores

| Model | Parameters | Auto-scored Average | Notes |
|-------|-----------|---------------------|-------|
| **Qwen 2.5 3B** | 3B | **63.4** | Overall 1st |
| **Phi-4 Mini** | 3.8B | **60.6** | Reasoning-focused |
| **FRANKENSTALLM 3B v2** | 3B | **46.7** | This model (ORPO) |
| FRANKENSTALLM 3B v1 | 3B | 37.9 | SFT only |

#### Category-level Detailed Comparison

| Category | FRANKENSTALLM v2 | Qwen 2.5 3B | Phi-4 Mini 3.8B | Notes |
|----------|:---:|:---:|:---:|-------|
| **Korean NLU** | **100.0** | **100.0** | 66.7 | On par for Korean comprehension |
| **Knowledge** | **75.0** | 20.8 | 29.2 | Dominant advantage in Korean knowledge |
| **Instruction Following** | **66.7** | 55.6 | 33.3 | Strong instruction following |
| **Reasoning** | 50.0 | 62.5 | **87.5** | Phi-4 specializes in reasoning |
| **Code** | 0.0 | **100.0** | 83.3 | Lacking in code capability |
| **Safety** | 10.0 | 35.0 | **70.0** | Safety is a weakness |
| **Repetition Resistance** | 2.2 | **75.0** | 58.9 | Based on default parameters^1 |

> ^1 FRANKENSTALLM's repetition resistance of 2.2% is based on Ollama default settings. With `repeat_penalty=1.2`, it achieves 0%.

#### Inference Speed Comparison

| Model | Avg TTFT (ms) | P95 TTFT (ms) | Avg TPS | Notes |
|-------|:---:|:---:|:---:|-------|
| **FRANKENSTALLM 3B v2** | **16.7** | **26.2** | **142.5** | Fastest |
| Phi-4 Mini 3.8B | 25.6 | 44.9 | 100.4 | |
| Qwen 2.5 3B | 28.2 | 46.5 | 93.8 | |

> FRANKENSTALLM is optimized for Ollama inference with the same architecture (LlamaForCausalLM) and 64K vocab.

#### lm-eval Benchmark Comparison (Base → SFT → ORPO)

| Benchmark | Base | SFT | ORPO | Change (Base→ORPO) |
|-----------|:---:|:---:|:---:|:---:|
| **KoBEST COPA** | 49.3% | 48.6% | **63.9%** | **+14.6pp** |
| **KoBEST HellaSwag** | 21.6% | 19.8% | **38.0%** | **+16.4pp** |
| **KoBEST SentiNeg** | 48.6% | 49.1% | **62.5%** | **+13.9pp** |
| **KoBEST BoolQ** | 50.3% | 50.1% | 50.6% | +0.3pp |
| **KoBEST WiC** | 48.7% | 48.7% | 48.8% | +0.2pp |
| **KoBEST Average** | 43.7% | 43.3% | **52.8%** | **+9.1pp** |
| **HAE-RAE** | 19.7% | 19.9% | 21.8% | +2.1pp |
| **PIQA** | 52.5% | 52.6% | **59.9%** | **+7.3pp** |
| **ARC-Easy** | 25.6% | 25.9% | **36.0%** | **+10.4pp** |
| **HellaSwag EN** | 26.2% | 26.1% | 29.2% | +3.0pp |
| **Winogrande** | 50.6% | 50.8% | 51.0% | +0.4pp |
| **PPL forgetting** | Baseline | 0.9% | 4.1% | Within 15% threshold |

#### Analysis and Positioning

**Strengths**:
- **Korean NLU 100%** — On par with Qwen 2.5 3B, far surpassing Phi-4
- **Korean Knowledge 75.0** — Highest among compared models (Qwen 20.8, Phi-4 29.2)
- **Fastest inference speed** — TTFT 16.7ms, TPS 142.5, advantageous for real-time serving
- **Maximized ORPO effect** — KoBEST +9.1pp, with 13-16pp gains in COPA/HellaSwag/SentiNeg

**Weaknesses and Improvement Directions**:
- **Code generation 0%** — Insufficient code SFT data. Code-specific data augmentation needed
- **Safety 10%** — Safety alignment data not incorporated. Consider additional RLHF/DPO training
- **Repetition issue** — Repetition occurs with default settings, but fully resolved with `repeat_penalty=1.2`
- **Overall score gap** — -16.7pp vs Qwen. Training data quality and scale differences are the primary cause

---

## 23. Reproduction Guide — Full-stage Configuration Details

> This section is for reference when reproducing in an identical environment.

### Environment Setup

```bash
# Do NOT reinstall NVIDIA custom PyTorch (breaks B200 optimization)
# Only install the following additional packages
pip install transformers==4.40.0 accelerate peft trl deepspeed \
            bitsandbytes sentencepiece wandb

# Full environment reproduction
pip install -r requirements.txt
```

**Software Versions (Measured)**:
```
torch          2.10.0a0+b4e4ee81d3.nv25.12
flash_attn     2.7.4.post1+25.12
transformers   4.40.x
datasets       4.4.1
tokenizers     0.22.1
huggingface_hub 1.2.3
trl            (with ORPO NaN bug patch applied — see scripts/trl_patch.py)
CUDA           13.1 / Driver 580.95.05
```

### Phase 1 — Pretrain Key Hyperparameters

```yaml
# Based on configs/korean_3b_fp8.yaml
model:
  hidden_size: 2048
  num_hidden_layers: 24
  num_attention_heads: 16
  num_key_value_heads: 4
  intermediate_size: 5632
  max_position_embeddings: 4096
  vocab_size: 64000

train:
  batch_size: 5
  gradient_accumulation_steps: 8
  # effective batch = 5 x 8 x 8GPU = 320
  learning_rate: 1.5e-4
  min_lr: 1.5e-5          # cosine decay floor (10% of max_lr)
  warmup_steps: 2000
  weight_decay: 0.1
  max_grad_norm: 1.0
  scheduler: cosine
  precision: bf16          # FP8 Tensor Core utilization (B200)
  total_tokens: ~38.5B
```

### Phase 2 — SFT v2 Key Hyperparameters

```yaml
# Based on configs/korean_3b_sft_v2.yaml
train:
  learning_rate: 5.0e-5
  batch_size: 4
  gradient_accumulation_steps: 8
  # effective batch = 4 x 8 x 8GPU = 256
  warmup_ratio: 0.03
  weight_decay: 0.01
  lr_scheduler_type: cosine
  max_steps: 33000          # early stop at 25,500 (patience=5)
  early_stopping_patience: 5
  fp16: false
  bf16: true

data:
  total_samples: 2439397    # train
  val_samples: 49801
  num_sources: 24
  total_size_gb: 7.48
  mixing: "70% instruction / 30% general"
```

### Phase 3 — ORPO Key Hyperparameters

```yaml
# Based on configs/korean_3b_orpo.yaml (main training)
train:
  beta: 0.25                # ORPO preference weight
  learning_rate: 1.2e-5    # Optimal value from HP sweep
  batch_size: 4
  gradient_accumulation_steps: 4
  # effective batch = 4 x 4 x 8GPU = 128
  warmup_ratio: 0.1
  weight_decay: 0.01
  lr_scheduler_type: cosine
  max_length: 1536          # max combined length of prompt + response
  max_prompt_length: 512
  num_train_epochs: 2       # actual early convergence at ~9,997 steps
  bf16: true
  optim: adamw_torch_fused

data:
  raw_pairs: 683181
  filtered_pairs: ~630000   # after NaN prevention filtering
  eval_split: 0.05          # seed=42
  eval_pairs: ~31500
```

### Phase 4 — GGUF Conversion Pipeline

```bash
# Step 1: Fix byte-fallback tokenizer
python scripts/fix_tokenizer_byte_fallback.py \
  --input  outputs/hf_checkpoint-best \
  --output outputs/hf_checkpoint-best-fixed

# Step 2: Convert to f16 GGUF (llama.cpp)
python outputs/llama.cpp/convert_hf_to_gguf.py \
  outputs/hf_checkpoint-best-fixed \
  --outfile outputs/gguf/frankenstallm-3b-v2-f16.gguf \
  --outtype f16

# Step 3: Quantization
QUANTIZE=outputs/llama.cpp/build/bin/llama-quantize
$QUANTIZE outputs/gguf/frankenstallm-3b-v2-f16.gguf \
          outputs/gguf/frankenstallm-3b-v2-Q4_K_M.gguf Q4_K_M
$QUANTIZE outputs/gguf/frankenstallm-3b-v2-f16.gguf \
          outputs/gguf/frankenstallm-3b-v2-Q8_0.gguf Q8_0

# Step 4: Register with Ollama
ollama create frankenstallm-3b-v2:Q4_K_M -f Modelfile.3b-v2-Q4
ollama create frankenstallm-3b-v2:Q8_0   -f Modelfile.3b-v2-Q8
ollama create frankenstallm-3b-v2:f16    -f Modelfile.3b-v2-f16
```

### Tokenizer Reproduction

```
Training method: SentencePiece Unigram
vocab_size: 64,000 (original) → 64,256 (after byte-fallback fix)
Training script: tokenizer/train_sp_tokenizer.py
Training data: C4 Korean + Namuwiki + Wikipedia Korean (mixed)
byte_fallback: True (added in v2)
Additional tokens: <0x00> ~ <0xFF> (256 tokens)
```

---

## Closing Remarks

This project has one motto:

> **"Document even the failures."**

The SFT v1 loss=0.0 disaster, the ineffectiveness of torch.compile, the frustration of 18% repetition rate — all of it is documented here. Phase 3 ORPO also went through **5 failures** — NCCL timeout, config conflicts, QKV conversion bugs, port collisions, TRL NaN bugs — before finally crossing the finish line.

Phase 1 pretraining completed at 57,000 steps with loss 1.466. Phase 2 SFT early-stopped at 25,500 steps (val_loss 1.8851). Phase 3 ORPO converged early at 9,997 steps — eval_loss 1.625, Preference Accuracy 76.02%. Phase 4 converted to GGUF and deployed to HuggingFace and Ollama.

**We made it**: greedy repetition rate 72.97% → 30.89% (ORPO), 0% with sampling + rep_penalty. TPS 142.5, TTFT 16.7ms. A 3B model that understands and speaks Korean, built from scratch.

Just as Frankenstein stitched together pieces to create life, FRANKENSTALLM was built the same way.

**Model Download**: https://huggingface.co/pathcosmos/frankenstallm

---

## Acknowledgment

This project was conducted using GPU computing resources provided through the **"Advanced GPU Utilization Support Program"** (MSIT Notice No. 2025-1068) by the **Ministry of Science and ICT (MSIT)** of the Republic of Korea.

> **National AI Computing Resource Support Portal**: https://aiinfrahub.kr
>
> - Organized by: Ministry of Science and ICT (MSIT), National IT Industry Promotion Agency (NIPA)
> - Operated by: Korea Association of Information & Telecommunication (KAIT)

We are deeply grateful for the national-level AI computing infrastructure support from the Korean government, which made it possible to train a Korean 3B LLM from scratch on 8× NVIDIA B200 GPUs.

---

*Last updated: 2026-03-26*
*Current status: **All phases complete** — Phase 1 Pretrain | Phase 2 SFT | Phase 3 ORPO | Phase 4 Deployment*
