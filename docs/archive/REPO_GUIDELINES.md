# Repository Guidelines

## Project Structure & Modules
- `scu/`: Core library (control logic in `control.py`, dataset utils in `data.py`).
- `scripts/`: Training and analysis entry points (e.g., `train_scu.py`).
- `viz/`: Plotting utilities and CLI (`python -m viz.cli`).
- `assets/figures/`: Generated figures used in docs and the web folder.
- `data/`: Small sample `train.txt` and `val.txt` for quick runs.
- `configs/`: Defaults for training (`default.yaml`).
- `tools/`: Operational scripts (e.g., `hf_sync.py` to publish to Hugging Face).
- `web/`: Static site assets and pages.
- Tests: `test_model_loading.py`, `test_3b_models.py` at repo root.

## Build, Test, and Development
- Setup: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Quick training demo: `python scripts/train_scu.py --quickstart --adapter_out adapters/quickstart`
- Generate figures: `python -m viz.cli --root outputs --out assets/figures --which s_curve,lambda,validation`
- Validate 3B models: `scripts/run_3b_validation.sh` (auto-selects `cuda`/`mps`).
- Load test (CPU): `python test_model_loading.py` (downloads base + applies local adapters).

## Coding Style & Naming
- Python, PEP 8, 4-space indentation, type hints where useful.
- Modules and functions: `snake_case`; classes: `CapWords`.
- Docstrings: triple double quotes; prefer f-strings; avoid prints in library code (OK in CLIs).
- No enforced formatter in CI; keep lines ≤ 100 chars and follow existing patterns.

## Testing Guidelines
- No pytest suite; tests are executable scripts.
  - Model load: `python test_model_loading.py`
  - 3B validation (GPU/MPS recommended): `python test_3b_models.py --device cuda|mps`
- Large models require Hugging Face terms acceptance; CPU runs are slow.

## Commit & Pull Requests
- Prefer Conventional Commits: `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`.
- PRs must include: concise description, repro commands (env + CLI args), and screenshots if figures change.
- Keep `scripts/train_scu.py` CSV header consistent with CI (see `.github/workflows/lint.yml`), or update CI in the same PR.
- Avoid committing new large binaries; use HF Hub (`tools/hf_sync.py`) or Git LFS.

## Security & Configuration Tips
- GPU strongly recommended (CUDA); Mac `mps` supported. `bitsandbytes` is disabled on macOS.
- Data paths default to `data/train.txt` and `data/val.txt`; update via flags or `configs/default.yaml`.
- For HF publishing, run `huggingface-cli login` then `python tools/hf_sync.py`.
