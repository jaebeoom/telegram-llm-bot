# AGENTS.md

Copy this file to `AGENTS.md` for local agent instructions. Keep machine-specific
paths and private preferences in the local file.

## Environment

- Use `uv` with the project-local `.venv`.
- Python target is 3.11.
- Before installing anything, read `pyproject.toml` and `README.md`.
- Prefer `uv run ...` for commands when activation state is unclear.

## Dependencies

- Runtime dependencies are declared in `pyproject.toml`.
- Avoid ad hoc global installs.
- If a new dependency is required, update `pyproject.toml` as part of the same change.

## Run And Test

- Main run command: `uv run python bot.py`
- Test command: `uv run pytest -q`

## Project Notes

- This project is local-first and uses an OpenAI-compatible local LLM endpoint.
- Check `.env.example` and README before touching environment variables.
- Shared AI env files outside the repo may override project `.env`; do not assume `.env` is the only source of truth.
- Be careful with security-sensitive settings such as `ALLOWED_USER_IDS` and Vault logging paths.
