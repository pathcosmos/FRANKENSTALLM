# AGENTS.md

## Cursor Cloud specific instructions

### Project overview
FRANKENSTALLM is a Korean 3B LLM project covering the full lifecycle: tokenizer training, pretraining, SFT, ORPO, GGUF conversion, and deployment. It is a pure Python/PyTorch ML training codebase with no web services or databases.

### Dependencies
- Install via `pip install -r requirements.txt`
- **Do NOT install PyTorch via pip** — the target hardware uses an NVIDIA custom build (`nv25.12`). In the Cloud VM, the standard PyTorch from pip is fine for code validation but not for actual GPU training.

### Key modules
| Directory | Purpose |
|-----------|---------|
| `model/` | Model architecture (`LMConfig`, `LLM`, `TransformerBlock`, `MultiHeadAttention`) |
| `train/` | Training scripts (pretrain, SFT, ORPO) |
| `eval/` | Evaluation pipelines (PPL, benchmarks, generation quality) |
| `configs/` | YAML training configs |
| `tokenizer/` | 64K SentencePiece tokenizer (loadable via `AutoTokenizer.from_pretrained('tokenizer/')`) |
| `scripts/` | Shell launch scripts, monitoring, conversion utilities |

### Running code on Cloud VM (no GPU)
- The model architecture can be instantiated and tested on CPU with `use_flash_attn=False` and `use_fp8=False` in `LMConfig`.
- Use `torch.bfloat16` dtype for the model to avoid dtype mismatches in the attention module (it internally casts to bfloat16).
- `model.forward()` returns a tuple `(logits, loss)`, not just logits.
- Actual distributed training (`torchrun`, 8×GPU) requires the B200 hardware.

### Lint
- No project-specific lint config exists. Use `ruff check` for quick Python linting: `python3 -m ruff check model/ train/ eval/ --select E,F`
- Existing code has lint issues (E501 line-too-long, F401 unused-import, etc.) — these are pre-existing.

### Configs
- `configs/small.yaml` — 125M model (good for CPU testing)
- `configs/korean_3b_fp8.yaml` — 3B pretrain config (GPU required)
- `configs/korean_3b_sft.yaml` / `korean_3b_sft_v2.yaml` — SFT configs
- `configs/korean_3b_orpo.yaml` — ORPO config

### Running commands
See `README.md` § 13 "실행 방법" for full training/eval/deploy commands.
