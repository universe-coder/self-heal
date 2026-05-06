# self-heal

[![PyPI version](https://img.shields.io/pypi/v/self-heal-runtime?style=flat)](https://pypi.org/project/self-heal-runtime/) [![Python](https://img.shields.io/badge/python-3.11%2B-blue?style=flat)](https://pypi.org/project/self-heal-runtime/) [![License](https://img.shields.io/github/license/universe-coder/self-heal?style=flat)](https://github.com/universe-coder/self-heal/blob/main/LICENSE) [![Last commit](https://img.shields.io/github/last-commit/universe-coder/self-heal?style=flat)](https://github.com/universe-coder/self-heal/commits/main) [![Open issues](https://img.shields.io/github/issues/universe-coder/self-heal?style=flat)](https://github.com/universe-coder/self-heal/issues) [![Forks](https://img.shields.io/github/forks/universe-coder/self-heal?style=flat)](https://github.com/universe-coder/self-heal/network/members) [![Stars](https://img.shields.io/github/stars/universe-coder/self-heal?style=flat)](https://github.com/universe-coder/self-heal/stargazers)

Open-source **self-healing** helper for Python.  
It indexes your repository (AST chunks + OpenAI-compatible **embeddings**), captures tracebacks (in-process hook / decorator or supervised subprocess), asks an **OpenAI-compatible** chat model for a **unified diff**, validates paths, optionally **applies** patches, and records an audit trail.

## Highlights

- **Python**: `>= 3.11`
- **Model protocol**: OpenAI-compatible HTTP APIs (`base_url`, chat completions, embeddings)
- **Safety first**: no `exec` of model output, unified diffs only, allow/deny path checks, masked secrets in locals, API keys from env only

## Contents

- [Install](#install)
- [Quickstart](#quickstart)
- [Configuration](#configuration)
- [Notifications](#notifications)
- [Security](#security)
- [Limitations (MVP)](#limitations-mvp)
- [License](#license)

## Install

```bash
pip install self-heal-runtime
# Optional: notifications integrations (Sentry SDK extra)
pip install "self-heal-runtime[notifications]"
```

Set `OPENAI_API_KEY` (or the env name from `[llm].api_key_env` in `.self-heal.toml`).

## Quickstart

1. **Initialize config** in your project root:

   ```bash
   self-heal init
   ```

2. **Index** code (requires an embeddings endpoint):

   ```bash
   self-heal index
   ```

3. **Run in supervised mode** (captures stderr tracebacks and can propose/heal):

   ```bash
   PYTHONPATH=. self-heal run -- python -m examples.broken_app.main
   ```

   Auto-apply variant (still validates paths; use with care):

   ```bash
   PYTHONPATH=. self-heal run --auto --no-dry-run -- python -m examples.broken_app.main
   ```

4. **Run offline heal** from a traceback file:

   ```bash
   python -m examples.broken_app.main 2> tb.txt   # or copy a traceback
   self-heal heal --tb tb.txt
   ```

5. **Use as a library** (in-process):

   ```python
   from pathlib import Path
   from self_heal import install, self_heal

   install(project_root=Path(__file__).resolve().parents[1])

   @self_heal(mode="suggest")
   def risky():
       ...
   ```

## Configuration

See [.self-heal.example.toml](.self-heal.example.toml) or run `self-heal init` (template is shipped in `self_heal/templates/`).

Key sections: `[llm]`, `[index]`, `[heal]`, `[supervisor]`, `[notifications]`.

## Notifications

Optional outbound reports when an error is captured and when a heal is proposed or applied. Enable `[notifications].enabled = true`, then turn on individual channels under `[notifications.telegram]`, `[notifications.slack]`, `[notifications.webhook]`, or `[notifications.sentry]`. Secrets stay in environment variables (e.g. `TELEGRAM_BOT_TOKEN`, `SLACK_WEBHOOK_URL`, `SELF_HEAL_WEBHOOK_URL`, `SENTRY_DSN`).

- **Payload**: by default, diffs are omitted and tracebacks are truncated (`include_diff`, `include_traceback`, `max_traceback_lines`)
- **Webhook**: JSON POST with optional HMAC (`X-SelfHeal-Signature: sha256=…`, `X-SelfHeal-Timestamp`); only `https` targets are allowed unless `allow_insecure = true`
- **Sentry**: install extras with `pip install 'self-heal-runtime[notifications]'` (pulls in `sentry-sdk`)

Delivery is asynchronous; successes and failures are also written to `.self-heal/audit.jsonl` as `notification_sent` / `notification_failed`.

## Security

- Model output is **never executed** as Python; only **unified diffs** are accepted.
- Patches are checked against `allowed_paths` / `forbidden_paths`; defaults block `.git/`, `.env*`, secret globs, `pyproject.toml`.
- Locals and tracebacks are **sanitized** before they are sent to the model.
- Prefer **`dry-run`** (CLI default) until you trust the workflow; use `--no-dry-run` with `--auto` only when appropriate.

## Limitations (MVP)

- Patch application prefers **`git apply`** inside a git repo; otherwise it tries system **`patch`**, then a small Python hunk applier.
- Supervised mode expects **Python-style** tracebacks on stderr.
- Embedding dimension is assumed compatible with Chroma stored vectors (default client setup targets OpenAI `text-embedding-3-small`-sized vectors).

## License

MIT — see [LICENSE](LICENSE).
