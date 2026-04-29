# Repository Guidelines

## Project Structure and Module Organization
- `shannon_control/`: core SCU logic (control, data utilities, MLX engine).
- `scu_api/`: FastAPI server, job queue, and CLI entrypoints.
- `scripts/`: training, evaluation, and data prep helpers.
- `configs/`: YAML config presets for experiments and production runs.
- `docs/`: research narrative and technical notes (start with `docs/START_HERE.md`).
- `tests/`: pytest and unittest coverage for control and training logic.
- `labs-frontend/`: React client for the public site.
- `data/`, `adapters/`, `results/`, `logs/`, `experiments/`, `notebooks/`: local artifacts and research outputs.

## Build, Test, and Development Commands
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  pip install -e .[dev,server]
  ```
- PyTorch training (CUDA or MPS):
  ```bash
  python scripts/train_scu.py --base_model meta-llama/Llama-3.2-1B --train_data data/train.txt --steps 500
  ```
- MLX training (Apple Silicon only):
  ```bash
  python scripts/train_mlx_scu.py --model mlx-community/Llama-3.2-1B-Instruct-4bit --train-data data/wikitext_train.jsonl --steps 500
  ```
- API server and CLI:
  ```bash
  python -m scu_api.server
  scu train --base-model sshleifer/tiny-gpt2 --train-data data/train.txt --steps 5 --wait
  ```

## Coding Style and Naming Conventions
- Python: 4-space indentation, PEP8 style, `snake_case` for functions/vars, `PascalCase` for classes, `UPPER_CASE` for constants.
- Prefer type hints for public functions and configs.
- Frontend: `labs-frontend/client/src` uses `.tsx`, `PascalCase` React components, `camelCase` props.
- No repo-wide formatter is enforced; match surrounding file style.

## Testing Guidelines
- Tests live in `tests/` and are named `test_*.py`.
- Run all tests with:
  ```bash
  pytest
  ```
- Async tests use `pytest-asyncio`. There is no formal coverage threshold yet.

## Commit and Pull Request Guidelines
- Follow conventional commits (recent history uses `feat:`, `fix:`, `docs:`, `refactor:`).
- Keep commits focused; include a short summary and any relevant metrics (loss, BPT, S-ratio).
- PRs should describe the experiment or change, list commands run, and link issues or notes.

## Security and Data Notes
- Do not commit secrets (API keys, tokens). Use environment variables for remote services.
- Avoid committing large model weights or generated artifacts; prefer `adapters/` and `results/` locally.
