# Repository Guidelines

## Project Structure & Module Organization

SkyRL is organized as a monorepo. The unified package lives in `skyrl/`, with training code under `skyrl/train`, model and tensor utilities under `skyrl/tx`, backends under `skyrl/backends`, and Tinker API support under `skyrl/tinker`. Legacy or standalone packages remain in `skyrl-agent/`, `skyrl-gym/`, `skyrl-train/`, and `skyrl-tx/`. Tests mirror the main package in `tests/`, with focused suites such as `tests/train`, `tests/backends`, `tests/tinker`, and `tests/tx`. Examples and integration recipes are in `examples/`, CI scripts in `ci/`, Docker assets in `docker/`, and documentation in `docs/`.

## Build, Test, and Development Commands

- `uv sync --extra dev`: install the base development environment.
- `uv run --extra dev --extra tinker pytest -v`: run the standard test suite described by `tests/README.md`.
- `uv run --extra dev --extra tinker pytest -v -s tests/train/test_trainer.py`: run a focused test file.
- `uv run --extra fsdp -m examples.train_integrations.openenv.entrypoints.main_openenv ...`: launch a training integration; prefer editing the provided shell scripts in `examples/train_integrations/`.
- `pre-commit run --all-files`: run formatting, linting, and secret scanning before submitting.

Use package-local commands from `skyrl-agent/` or `skyrl-gym/` when working only inside those packages.

## Coding Style & Naming Conventions

Python code uses 4-space indentation and Black formatting. Ruff runs with auto-fix for most paths, while `skyrl-agent/` is excluded from the root Ruff hook. Prefer typed dataclasses for configuration, explicit module names, and descriptive snake_case for functions, variables, and test names. Keep imports grouped and avoid broad refactors in unrelated packages.

## Testing Guidelines

Tests use pytest. Name files `test_*.py` and place them near the matching subsystem under `tests/` or package-local test directories such as `skyrl-gym/tests/`. Add narrow tests for bug fixes and broaden coverage when touching shared trainer, backend, generator, or config behavior. GPU-heavy tests may require the relevant extras and hardware-specific environment.

## Commit & Pull Request Guidelines

Recent commits use bracketed scopes, for example `[bug] Fix ...`, `[qoc][trainer] ...`, or `[cleanup] ...`. Keep subjects imperative and concise. Pull requests should describe the change, list verification commands, link related issues, and call out GPU, data, or dependency requirements. Include screenshots only for documentation or UI-facing changes.
After each code or documentation modification, create a git commit so the work is checkpointed.

## Security & Configuration Tips

Do not commit API keys, model credentials, generated checkpoints, or local datasets. Use environment variables for secrets such as `WANDB_API_KEY`. Keep large artifacts under external storage or ignored paths, not in the repository.
