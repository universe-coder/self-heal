# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.4.0] - 2026-10-13

### Added
- `heal.max_attempts` is honoured: an empty, policy-rejected or unappliable diff is fed back to the model and another attempt is made.
- Traceback-anchored context: source around the frames named in the traceback is always sent to the model (`heal.context_lines`), so healing works without an index and with providers that have no embeddings API.
- Separate embeddings endpoint (`llm.embedding_base_url`, `llm.embedding_api_key_env`), e.g. Anthropic for chat with OpenAI or Ollama embeddings.
- `self-heal doctor`: checks config, credentials, index, tooling and notification environment.
- `self-heal rollback [--list] [--backup NAME]`: restore files from the backup taken before a patch; the pre-rollback state is backed up too.
- `self-heal index --force`, `self-heal status --json`, global `--verbose/--quiet/--version`, `heal --instruction`.
- In-process cooldown (`heal.cooldown_s`): the same error is not sent to the model again within the window.
- `@self_heal` supports `async def`; `install()` also hooks `threading.excepthook`.
- `index.respect_gitignore`, `heal.allowed_extensions`, `llm.temperature`, `llm.max_retries`, `supervisor.stop_on_repeated_traceback`.
- New audit kinds `heal_rejected`, `heal_skipped`, `heal_rolled_back`; audit entries record the attempt number and backup directory.
- GitHub Actions CI (lint, mypy, tests on 3.11–3.13, package build).

### Changed
- `self-heal heal --apply` and `self-heal run --auto` now imply `--no-dry-run` unless `--dry-run` is passed explicitly. Previously `--apply` without `--no-dry-run` did nothing.
- Unknown keys or sections in `.self-heal.toml` are an error instead of being silently ignored; numeric options are range-checked.
- Only throttling, timeouts, connection errors and 5xx/529 responses are retried; authentication and bad-request errors fail immediately. The OpenAI SDK's own retries are disabled so attempts no longer multiply.
- Chunking: methods are qualified (`Class.method`), decorators are part of the function chunk, long definitions are split into overlapping windows instead of being truncated, classes contribute their header only, and module-level code is indexed as `<module>` chunks.
- Indexing removes deleted files, embeds all changed files in one batched call, records the embedding model and rebuilds when it changes, prunes excluded directories during the walk, and stores vectors with cosine distance.
- The supervisor stops when the process dies with the same traceback again and nothing was applied; Ctrl-C terminates the child; chained tracebacks and header-less `SyntaxError` output are detected.
- Telegram messages are kept within the 4096-character limit and MarkdownV2 is escaped properly.
- `ErrorEvent`/`capture_from_exception` live in `self_heal.events`, the sanitizer in `self_heal.sanitize` (the `self_heal.runtime.*` paths still work).
- `run_index()` returns `IndexStats`; `DiffApplyResult` gained `diff`, `applied`, `backup_dir`, `attempts`, `policy_errors`.
- `pydantic-settings` is no longer a dependency.

### Fixed
- Path policy was evaluated on the wrong path for dot-directories: `normalize_diff_path(".github/x.py")` returned `github/x.py`, so a patch could target a path the policy never saw. Paths are now normalised correctly and `..`/absolute paths are rejected explicitly.
- Locals were captured from the outermost frame instead of the raise site; deep stacks dropped the innermost frames.
- `KeyboardInterrupt`/`SystemExit` triggered healing (and an LLM call) in the decorator, context manager and excepthook.
- `@self_heal(retries=N)` never retried.
- The pure-Python patch fallback could leave the tree half-patched when a later file failed; it now applies in memory first. `patch` runs `--dry-run` first.
- `git apply` was skipped for projects in a subdirectory of a repository or in worktrees.
- Importing `self_heal.healer.pipeline` directly raised a circular-import error.
- `__version__` was hard-coded to `0.1.0`.
- Chroma `DeprecationWarning` about the embedding function.
- Secret values of environment variables, API keys, tokens, JWTs and URL credentials are masked in tracebacks and locals (the previous check compared `repr()` strings and never matched).

## [0.3.0]
- Multiple LLM providers (OpenAI-compatible, Anthropic, Hugging Face, Ollama).

## [0.2.0]
- Notifications (Telegram, Slack, webhook, Sentry).

## [0.1.0]
- Initial release.
