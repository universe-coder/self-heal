# self-heal

[![PyPI version](https://img.shields.io/pypi/v/self-heal-runtime?style=flat)](https://pypi.org/project/self-heal-runtime/) [![Python](https://img.shields.io/badge/python-3.11%2B-blue?style=flat)](https://pypi.org/project/self-heal-runtime/) [![CI](https://img.shields.io/github/actions/workflow/status/universe-coder/self-heal/ci.yml?branch=main&style=flat)](https://github.com/universe-coder/self-heal/actions/workflows/ci.yml) [![License](https://img.shields.io/github/license/universe-coder/self-heal?style=flat)](https://github.com/universe-coder/self-heal/blob/main/LICENSE) [![Last commit](https://img.shields.io/github/last-commit/universe-coder/self-heal?style=flat)](https://github.com/universe-coder/self-heal/commits/main) [![Open issues](https://img.shields.io/github/issues/universe-coder/self-heal?style=flat)](https://github.com/universe-coder/self-heal/issues) [![Stars](https://img.shields.io/github/stars/universe-coder/self-heal?style=flat)](https://github.com/universe-coder/self-heal/stargazers)

Open-source **self-healing** helper for Python.

It captures tracebacks (in-process hook / decorator, or a supervised subprocess), gathers code context (the source around the failing frames plus, optionally, similar chunks from an embeddings index), asks an LLM for a **unified diff**, validates it against a path policy, optionally **applies** it with a backup, and records an audit trail.

## Highlights

- **Python** `>= 3.11`; works with **OpenAI**, **Anthropic**, **Hugging Face** and **Ollama** (any OpenAI-compatible endpoint for chat and embeddings)
- **Works without an index**: the code around every traceback frame inside your project is always sent to the model; the vector index is an optional boost
- **Bounded retries with feedback**: an empty, rejected or unappliable patch is explained to the model and it gets another attempt (`heal.max_attempts`)
- **Safety first**: model output is never executed, unified diffs only, allow/deny path policy, file-type allowlist, dry-run by default, backups + `rollback`, secrets masked before anything leaves the process
- **Operations**: `doctor` for setup checks, JSONL audit log, notifications (Telegram, Slack, webhook, Sentry), supervised restarts that stop on repeated crashes

## Contents

- [Install](#install)
- [Quickstart](#quickstart)
- [CLI](#cli)
- [Library usage](#library-usage)
- [Configuration](#configuration)
- [Providers](#providers)
- [Notifications](#notifications)
- [Security](#security)
- [Limitations](#limitations)
- [Development](#development)
- [License](#license)

## Install

```bash
pip install self-heal-runtime
# Optional: Sentry notifications
pip install "self-heal-runtime[notifications]"
```

Set the API key environment variable for your provider (`OPENAI_API_KEY` by default, see `[llm].api_key_env`).

## Quickstart

1. **Initialize config** in your project root and check the setup:

   ```bash
   self-heal init
   self-heal doctor
   ```

2. **Index** your code (optional, requires an embeddings endpoint):

   ```bash
   self-heal index
   ```

3. **Heal from a traceback** (dry-run: the diff is shown, nothing is written):

   ```bash
   python -m examples.broken_app.main 2> tb.txt
   self-heal heal --tb tb.txt
   self-heal heal --tb tb.txt --apply      # write it (a backup is kept)
   self-heal rollback                      # undo the last applied patch
   ```

4. **Run under supervision** (captures stderr tracebacks, proposes or applies a fix, restarts):

   ```bash
   PYTHONPATH=. self-heal run -- python -m examples.broken_app.main
   PYTHONPATH=. self-heal run --auto -- python -m examples.broken_app.main   # apply automatically
   ```

## CLI

| Command | What it does |
| --- | --- |
| `self-heal init [--force]` | Write `.self-heal.toml` from the built-in template |
| `self-heal doctor` | Check config, API key, embeddings endpoint, index, git/patch, backups, notification env vars (exit 1 on failures) |
| `self-heal index [--force]` | Incrementally index Python files under `[index].roots` (`--force` rebuilds) |
| `self-heal heal [--tb FILE] [--apply] [--instruction TEXT]` | Generate a fix from a traceback (file or stdin); `--apply` writes it |
| `self-heal run [--auto] -- CMD...` | Supervise a command; heal on tracebacks; restart |
| `self-heal status [-n N] [--json]` | Recent audit entries |
| `self-heal rollback [--list] [--backup NAME]` | Restore files from a backup taken before a patch was applied |

`--apply` / `--auto` imply `--no-dry-run`; pass `--dry-run` explicitly to keep a dry run. Use `-v` for debug logging and `-q` for warnings only. `-p PATH` selects the project root (default: the directory containing `.self-heal.toml`, found by walking up from the current directory).

## Library usage

```python
from pathlib import Path
from self_heal import heal_context, install, self_heal

# Global hooks (sys.excepthook and threading.excepthook); previous hooks still run first.
install(project_root=Path(__file__).resolve().parent, mode="suggest")


@self_heal(retries=2, mode="suggest")  # sync or async functions
def risky() -> None:
    do_work()


with heal_context(mode="suggest"):
    risky()
```

Modes: `suggest` only proposes (audit + notifications), `apply` follows `[heal].auto_apply`, `auto` always applies. The same error (exception type + raise site) is not sent to the model again within `[heal].cooldown_s`.

Lower-level entry points: `heal_exception(ErrorEvent)`, `heal_from_traceback_text(str)`, `capture_from_exception(exc)`, `rollback(project_root)`.

## Configuration

See [.self-heal.example.toml](.self-heal.example.toml) (the same template is written by `self-heal init`). Unknown keys are rejected so typos never silently disable an option.

| Section | Key options |
| --- | --- |
| `[llm]` | `provider`, `base_url`, `model`, `api_key_env`, `embedding_model`, `embedding_base_url` / `embedding_api_key_env` (separate embeddings endpoint), `timeout_s`, `max_tokens`, `max_retries`, `temperature` |
| `[index]` | `roots`, `exclude`, `respect_gitignore`, `chunk_max_lines`, `chunk_overlap_lines` |
| `[heal]` | `allowed_paths`, `forbidden_paths`, `allowed_extensions`, `auto_apply`, `max_attempts`, `backup`, `top_k`, `context_lines`, `max_diff_lines`, `cooldown_s` |
| `[supervisor]` | `restart_on_exit_codes`, `max_restarts`, `healthcheck_grace_s`, `stop_on_repeated_traceback` |
| `[notifications]` | `enabled`, `events`, `include_diff`, `include_traceback`, `max_traceback_lines`, plus `[notifications.telegram|slack|webhook|sentry]` |

## Providers

`[llm].provider` selects how chat completions are issued. Embeddings always use an OpenAI-compatible endpoint: by default the same `base_url`, or `embedding_base_url` when set.

- **OpenAI** (default):

  ```toml
  [llm]
  provider = "openai"
  base_url = "https://api.openai.com/v1"
  model = "gpt-4o-mini"
  embedding_model = "text-embedding-3-small"
  api_key_env = "OPENAI_API_KEY"
  ```

- **Anthropic**: native Messages API. Anthropic has no embeddings API; healing still works with traceback context, and you can point embeddings elsewhere to enable the index:

  ```toml
  [llm]
  provider = "anthropic"
  base_url = "https://api.anthropic.com/v1"
  model = "claude-sonnet-4-5"
  api_key_env = "ANTHROPIC_API_KEY"
  embedding_base_url = "http://localhost:11434/v1"   # e.g. Ollama
  embedding_model = "nomic-embed-text"
  embedding_api_key_env = "OLLAMA_API_KEY"           # unset is fine for Ollama
  ```

- **Hugging Face** (Inference Providers router):

  ```toml
  [llm]
  provider = "huggingface"
  base_url = "https://router.huggingface.co/v1"
  model = "meta-llama/Llama-3.1-8B-Instruct"
  embedding_model = "intfloat/multilingual-e5-large"
  api_key_env = "HF_TOKEN"
  ```

- **Ollama** (local, no auth):

  ```toml
  [llm]
  provider = "ollama"
  base_url = "http://localhost:11434/v1"
  model = "llama3.1"
  embedding_model = "nomic-embed-text"
  ```

Changing `embedding_model` rebuilds the index on the next `self-heal index`.

## Notifications

Optional outbound reports when an error is captured and when a heal is proposed, applied, rejected or rolled back. Enable `[notifications].enabled = true`, then turn on channels under `[notifications.telegram]`, `[notifications.slack]`, `[notifications.webhook]` or `[notifications.sentry]`. Secrets stay in environment variables (`TELEGRAM_BOT_TOKEN`, `SLACK_WEBHOOK_URL`, `SELF_HEAL_WEBHOOK_URL`, `SENTRY_DSN`).

- **Payload**: diffs are omitted by default and tracebacks truncated (`include_diff`, `include_traceback`, `max_traceback_lines`)
- **Telegram**: `HTML`, `MarkdownV2` or plain text; messages are kept within Telegram's size limit
- **Webhook**: JSON POST with optional HMAC (`X-SelfHeal-Signature: sha256=…`, `X-SelfHeal-Timestamp`); only `https` and public hosts unless `allow_insecure = true`
- **Sentry**: `pip install 'self-heal-runtime[notifications]'`

Delivery is asynchronous; successes and failures are written to `.self-heal/audit.jsonl` as `notification_sent` / `notification_failed`.

## Security

- Model output is **never executed**; only **unified diffs** are accepted, and only for files matching `allowed_paths` / `allowed_extensions` and not `forbidden_paths` (defaults block `.git/`, `.env*`, secret globs, `pyproject.toml`). Paths are normalised and resolved; `..`, absolute paths and symlinks escaping the project are rejected. Deletions and binary patches are refused.
- Tracebacks and locals are **sanitized** before leaving the process: variables with secret-looking names, values of secret-looking environment variables, bearer tokens, API keys, JWTs and URL credentials are masked.
- The CLI is **dry-run by default**; every applied patch is preceded by a backup under `.self-heal/backups/` and can be undone with `self-heal rollback`.
- Supervised runs stop restarting when the same crash repeats without a patch; in-process healing has a cooldown so a crash loop cannot flood the model.

## Limitations

- Patch application prefers `git apply` (inside any git worktree), then the system `patch`, then a small atomic Python hunk applier.
- Only Python files are chunked and indexed; `allowed_extensions` controls what a patch may touch.
- Supervised mode expects Python-style tracebacks on stderr.
- Applied patches change files on disk; a running process keeps its already-imported code until restarted.

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
ruff check . && ruff format --check . && mypy && pytest
```

Tests do not need network access or API keys. See [CHANGELOG.md](CHANGELOG.md) for release notes.

## License

MIT — see [LICENSE](LICENSE).
