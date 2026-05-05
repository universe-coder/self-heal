# self-heal

[![PyPI version](https://img.shields.io/pypi/v/self-heal-runtime?style=flat)](https://pypi.org/project/self-heal-runtime/) [![Python](https://img.shields.io/badge/python-3.11%2B-blue?style=flat)](https://pypi.org/project/self-heal-runtime/) [![License](https://img.shields.io/github/license/universe-coder/self-heal?style=flat)](https://github.com/universe-coder/self-heal/blob/main/LICENSE) [![Last commit](https://img.shields.io/github/last-commit/universe-coder/self-heal?style=flat)](https://github.com/universe-coder/self-heal/commits/main) [![Open issues](https://img.shields.io/github/issues/universe-coder/self-heal?style=flat)](https://github.com/universe-coder/self-heal/issues) [![Forks](https://img.shields.io/github/forks/universe-coder/self-heal?style=flat)](https://github.com/universe-coder/self-heal/network/members) [![Stars](https://img.shields.io/github/stars/universe-coder/self-heal?style=flat)](https://github.com/universe-coder/self-heal/stargazers)

Open-source **self-healing** helper for Python: index your repo (AST chunks + OpenAI-compatible **embeddings**), capture tracebacks (in-process hook / decorator or supervised subprocess), ask an **OpenAI-compatible** chat model for a **unified diff**, validate paths, optionally **apply** via `git apply` / `patch`, and **audit** proposals.

- **Python** ≥ 3.11  
- **Protocol**: OpenAI-compatible HTTP APIs (`base_url` + chat completions + embeddings)  
- **Safety**: no `exec` of model output; patches are unified diffs only; allow/deny paths; secrets masked in captured locals; API keys from env only.

## Install

```bash
pip install -e ".[dev]"   # from this repo
```

Set `OPENAI_API_KEY` (or the env name from `[llm].api_key_env` in `.self-heal.toml`).

## Quickstart

1. **Init config** in your project root:

   ```bash
   self-heal init
   ```

2. **Index** code (needs embeddings endpoint):

   ```bash
   self-heal index
   ```

3. **Supervised run** (captures stderr tracebacks and can propose/heal):

   ```bash
   PYTHONPATH=. self-heal run -- python -m examples.broken_app.main
   ```

   With auto-apply (still validates paths; use with care):

   ```bash
   PYTHONPATH=. self-heal run --auto --no-dry-run -- python -m examples.broken_app.main
   ```

4. **Offline heal** from a traceback file:

   ```bash
   python -m examples.broken_app.main 2> tb.txt   # or copy a traceback
   self-heal heal --tb tb.txt
   ```

5. **Library usage** (in-process):

   ```python
   from pathlib import Path
   from self_heal import install, self_heal

   install(project_root=Path(__file__).resolve().parents[1])

   @self_heal(mode="suggest")
   def risky():
       ...
   ```

## Configuration

See [.self-heal.example.toml](.self-heal.example.toml) or run `self-heal init` (template ships in `self_heal/templates/`).

Key sections: `[llm]`, `[index]`, `[heal]`, `[supervisor]`.

## Security

- Model output is **never executed** as Python; only **unified diffs** are accepted.
- Patches are checked against `allowed_paths` / `forbidden_paths`; defaults block `.git/`, `.env*`, secrets globs, `pyproject.toml`.
- Locals and tracebacks are **sanitized** before being sent to the model.
- Prefer **`dry-run`** (default in CLI) until you trust the workflow; use `--no-dry-run` with `--auto` only when appropriate.

## Limitations (MVP)

- Patch application: prefers **`git apply`** inside a git repo; otherwise tries system **`patch`**, then a small Python hunk applier.
- Supervised mode expects **Python-style** tracebacks on stderr.
- Embedding dimension is assumed compatible with Chroma’s stored vectors (default client setup targets OpenAI `text-embedding-3-small`-sized vectors).

## License

MIT — see [LICENSE](LICENSE).
