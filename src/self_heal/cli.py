"""Typer CLI: init, index, heal, run, status, rollback, doctor."""

from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from self_heal import __version__
from self_heal.audit.log import read_recent
from self_heal.config import (
    DEFAULT_CONFIG_NAME,
    ConfigError,
    SelfHealConfig,
    find_config,
    get_api_key,
    get_embedding_api_key,
    load_config_for_root,
    requires_api_key,
)
from self_heal.healer.diff import list_backups, read_backup_manifest
from self_heal.healer.pipeline import heal_from_traceback_text, rollback
from self_heal.indexing.indexer import run_index
from self_heal.indexing.store import collection_count, load_index_manifest
from self_heal.llm.client import EMBEDDING_UNSUPPORTED_MSG, LLMClient
from self_heal.supervisor.runner import run_supervised

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help=(
        "Self-healing runtime for Python: capture tracebacks, ask an LLM for a diff, apply safely."
    ),
)
console = Console()
err_console = Console(stderr=True)

_NOISY_LOGGERS = ("httpx", "httpcore", "openai", "chromadb", "urllib3")

EXIT_CONFIG_ERROR = 2


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"self-heal {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Warnings and errors only"),
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True, help="Show version"
    ),
) -> None:
    level = logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    if not verbose:
        for name in _NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)


def _project_root(path: Path | None) -> Path:
    if path is not None:
        return path.resolve()
    found = find_config()
    if found is not None:
        return found.parent.resolve()
    return Path.cwd().resolve()


def _load(root: Path) -> SelfHealConfig:
    try:
        return load_config_for_root(root)
    except ConfigError as e:
        err_console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=EXIT_CONFIG_ERROR) from None


def _print_diff(diff: str) -> None:
    console.print(Syntax(diff.rstrip("\n"), "diff", theme="ansi_dark", word_wrap=False))


@app.command()
def init(
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing config"),
    project: Path | None = typer.Option(None, "--project", "-p", help="Project root"),
) -> None:
    """Create `.self-heal.toml` from the built-in template."""
    from importlib import resources

    root = _project_root(project)
    dest = root / DEFAULT_CONFIG_NAME
    if dest.exists() and not force:
        console.print(f"[yellow]{dest} already exists[/yellow] (use --force)")
        raise typer.Exit(code=1)
    try:
        text = (
            resources.files("self_heal.templates")
            .joinpath("self-heal.default.toml")
            .read_text(encoding="utf-8")
        )
    except (OSError, FileNotFoundError, ModuleNotFoundError):
        ex = Path(__file__).resolve().parent / "templates" / "self-heal.default.toml"
        text = ex.read_text(encoding="utf-8")
    dest.write_text(text, encoding="utf-8")
    console.print(f"Wrote [green]{dest}[/green]")


@app.command()
def index(
    project: Path | None = typer.Option(None, "--project", "-p"),
    force: bool = typer.Option(False, "--force", "-f", help="Drop the index and rebuild it"),
) -> None:
    """Index Python files under the configured roots (requires an embeddings endpoint)."""
    root = _project_root(project)
    cfg = _load(root)
    if not cfg.llm.embeddings_available:
        err_console.print(f"[red]{EMBEDDING_UNSUPPORTED_MSG}[/red]")
        raise typer.Exit(code=1)
    client = LLMClient(cfg)
    if not client.embeddings_available:
        err_console.print(
            f"[red]Set {cfg.llm.resolved_embedding_api_key_env} for embeddings.[/red]"
        )
        raise typer.Exit(code=1)
    stats = run_index(root, cfg, client, force=force)
    if stats.up_to_date:
        console.print(f"Index up to date ([green]{stats.files_scanned}[/green] files tracked).")
        return
    console.print(
        f"Indexed [green]{stats.files_indexed}[/green] files "
        f"([green]{stats.chunks_indexed}[/green] chunks), removed "
        f"[green]{stats.files_removed}[/green]; {stats.files_scanned} files tracked."
        + (" [yellow](full rebuild)[/yellow]" if stats.full_reindex else "")
    )


def _effective_dry_run(dry_run: bool | None, wants_apply: bool) -> bool:
    """``--apply``/``--auto`` imply ``--no-dry-run`` unless ``--dry-run`` is given explicitly."""
    return (not wants_apply) if dry_run is None else dry_run


@app.command("heal")
def heal_cmd(
    traceback_file: Path | None = typer.Option(
        None, "--tb", help="Read traceback from file (else stdin)"
    ),
    project: Path | None = typer.Option(None, "--project", "-p"),
    apply: bool = typer.Option(False, "--apply", help="Write the patch (a backup is kept)"),
    auto: bool = typer.Option(False, "--auto", hidden=True, help="Alias for --apply"),
    dry_run: bool | None = typer.Option(
        None,
        "--dry-run/--no-dry-run",
        help="Propose only / write. Default: dry-run unless --apply is given.",
    ),
    instruction: str = typer.Option(
        "", "--instruction", "-i", help="Extra instruction passed to the model"
    ),
    show_diff: bool = typer.Option(True, "--show-diff/--no-show-diff"),
) -> None:
    """Generate (and optionally apply) a fix from a traceback + code context."""
    root = _project_root(project)
    cfg = _load(root)

    if traceback_file is not None:
        tb = traceback_file.read_text(encoding="utf-8", errors="replace")
    else:
        if sys.stdin.isatty():
            err_console.print("[red]Provide --tb FILE or pipe a traceback on stdin[/red]")
            raise typer.Exit(code=1)
        tb = sys.stdin.read()
    if not tb.strip():
        err_console.print("[red]Traceback is empty[/red]")
        raise typer.Exit(code=1)

    wants_apply = apply or auto
    effective_dry_run = _effective_dry_run(dry_run, wants_apply)

    client = LLMClient(cfg)
    if not client.is_configured:
        err_console.print(f"[red]Set {cfg.llm.api_key_env}[/red]")
        raise typer.Exit(code=1)

    result = heal_from_traceback_text(
        tb,
        project_root=root,
        mode="apply" if wants_apply else "suggest",
        cfg=cfg,
        client=client,
        auto_apply_override=True if wants_apply else None,
        dry_run=effective_dry_run,
        extra_instruction=instruction,
    )
    if result is None:
        err_console.print("[yellow]Healing skipped (see logs / audit).[/yellow]")
        raise typer.Exit(code=1)

    if result.diff and show_diff:
        _print_diff(result.diff)
    style = "green" if result.ok else "red"
    console.print(f"[{style}]{result.message}[/{style}]")
    if result.paths_touched:
        console.print("Paths:", ", ".join(result.paths_touched))
    if result.backup_dir:
        console.print(f"Backup: {result.backup_dir}")
    if result.ok and not result.applied:
        console.print(
            "[dim]Not applied. Re-run with --apply to write it "
            "(backups go to .self-heal/backups; undo with `self-heal rollback`).[/dim]"
        )
    raise typer.Exit(code=0 if result.ok else 1)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def run(
    ctx: typer.Context,
    auto: bool = typer.Option(False, "--auto", help="Apply patches automatically when healing"),
    dry_run: bool | None = typer.Option(
        None,
        "--dry-run/--no-dry-run",
        help="Propose only / write. Default: dry-run unless --auto is given.",
    ),
    project: Path | None = typer.Option(None, "--project", "-p"),
) -> None:
    """Run a command under supervision: `self-heal run -- python -m myapp`."""
    root = _project_root(project)
    cfg = _load(root)
    args = list(ctx.args)
    if not args:
        err_console.print(
            "[red]Pass a command after `--`, e.g. self-heal run -- python app.py[/red]"
        )
        raise typer.Exit(code=1)
    code = run_supervised(
        args,
        project_root=root,
        cfg=cfg,
        auto_apply=auto,
        dry_run=_effective_dry_run(dry_run, auto),
    )
    raise typer.Exit(code=code)


def _summarize(entry: dict[str, Any]) -> str:
    data = entry.get("data") or {}
    bits: list[str] = []
    if data.get("exception"):
        bits.append(str(data["exception"]))
    if data.get("message"):
        bits.append(str(data["message"])[:80])
    if data.get("reason"):
        bits.append(str(data["reason"])[:80])
    if data.get("paths"):
        bits.append(", ".join(map(str, data["paths"]))[:80])
    if data.get("channel"):
        bits.append(str(data["channel"]))
    if data.get("attempt"):
        bits.append(f"attempt {data['attempt']}")
    if "applied" in data:
        bits.append("applied" if data["applied"] else "not applied")
    if data.get("backup_dir"):
        bits.append(f"backup {Path(str(data['backup_dir'])).name}")
    return " · ".join(bits)


@app.command()
def status(
    project: Path | None = typer.Option(None, "--project", "-p"),
    limit: int = typer.Option(15, "--limit", "-n"),
    as_json: bool = typer.Option(False, "--json", help="Print raw audit entries"),
) -> None:
    """Show recent audit log entries."""
    root = _project_root(project)
    entries = read_recent(root, limit=limit)
    if not entries:
        console.print("No audit entries yet.")
        return
    if as_json:
        for e in entries:
            console.print_json(data=e)
        return
    table = Table(title=f"self-heal audit ({root})", show_lines=False)
    table.add_column("time", style="dim", no_wrap=True)
    table.add_column("kind", no_wrap=True)
    table.add_column("details")
    for e in entries:
        ts = str(e.get("ts", ""))[:19].replace("T", " ")
        table.add_row(ts, str(e.get("kind", "")), _summarize(e))
    console.print(table)


@app.command("rollback")
def rollback_cmd(
    project: Path | None = typer.Option(None, "--project", "-p"),
    backup: str | None = typer.Option(
        None, "--backup", "-b", help="Backup name (see --list); default: the latest"
    ),
    list_only: bool = typer.Option(False, "--list", "-l", help="List available backups"),
) -> None:
    """Restore files from a backup taken before a patch was applied."""
    root = _project_root(project)
    cfg = _load(root)
    backups = list_backups(root)
    if list_only:
        if not backups:
            console.print("No backups.")
            return
        table = Table(title="self-heal backups")
        table.add_column("name", no_wrap=True)
        table.add_column("created", style="dim", no_wrap=True)
        table.add_column("files")
        for b in backups:
            m = read_backup_manifest(b)
            files = [*m.get("files", []), *m.get("created_by_patch", [])]
            table.add_row(b.name, str(m.get("created_at", ""))[:19], ", ".join(files))
        console.print(table)
        return
    target: Path | None = None
    if backup is not None:
        matches = [b for b in backups if b.name == backup]
        if not matches:
            err_console.print(f"[red]No backup named {backup!r}[/red] (see --list)")
            raise typer.Exit(code=1)
        target = matches[0]
    try:
        used, touched = rollback(root, target, cfg=cfg)
    except FileNotFoundError as e:
        err_console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from None
    console.print(f"Restored [green]{len(touched)}[/green] file(s) from {used.name}")
    for rel in touched:
        console.print(f"  {rel}")


def _check_rows(root: Path, cfg: SelfHealConfig) -> list[tuple[str, str, str]]:
    """``(status, check, detail)`` rows; status is OK / WARN / FAIL."""
    rows: list[tuple[str, str, str]] = []
    rows.append(("OK", "project root", str(root)))
    if cfg.source_path is not None:
        rows.append(("OK", "config", str(cfg.source_path)))
    else:
        rows.append(("WARN", "config", f"no {DEFAULT_CONFIG_NAME} found; using defaults"))

    llm = cfg.llm
    rows.append(("OK", "chat provider", f"{llm.provider} · {llm.model} · {llm.base_url}"))
    if requires_api_key(cfg):
        if get_api_key(cfg):
            rows.append(("OK", "api key", f"${llm.api_key_env} is set"))
        else:
            rows.append(("FAIL", "api key", f"${llm.api_key_env} is not set"))
    else:
        rows.append(("OK", "api key", f"not required for {llm.provider}"))

    if not llm.embeddings_available:
        rows.append(
            (
                "WARN",
                "embeddings",
                f"{llm.provider} has no embeddings API; set [llm].embedding_base_url to "
                "enable indexing (healing still works with traceback context)",
            )
        )
    else:
        detail = f"{llm.embedding_model} · {llm.resolved_embedding_base_url}"
        if llm.embedding_base_url and not get_embedding_api_key(cfg):
            detail += f" (no ${llm.resolved_embedding_api_key_env}; assuming no auth)"
        rows.append(("OK", "embeddings", detail))

    manifest = load_index_manifest(root)
    if manifest.files:
        try:
            n = collection_count(root)
        except Exception as e:  # noqa: BLE001
            rows.append(("FAIL", "index", f"cannot open vector store: {e}"))
        else:
            model_note = ""
            if manifest.embedding_model and manifest.embedding_model != llm.embedding_model:
                model_note = (
                    f"; indexed with {manifest.embedding_model}, config says "
                    f"{llm.embedding_model} (next `self-heal index` rebuilds)"
                )
            rows.append(("OK", "index", f"{len(manifest.files)} files, {n} chunks{model_note}"))
    else:
        rows.append(("WARN", "index", "not built; run `self-heal index`"))

    missing_roots = [r for r in cfg.index.roots if not (root / r).exists()]
    if missing_roots:
        rows.append(("WARN", "index roots", f"missing: {', '.join(missing_roots)}"))

    if shutil.which("git"):
        rows.append(("OK", "git", "available (patches applied with `git apply`)"))
    elif shutil.which("patch"):
        rows.append(("OK", "patch", "git missing; using `patch`"))
    else:
        rows.append(("WARN", "patch tools", "neither git nor patch found; pure-Python fallback"))

    rows.append(
        (
            "OK",
            "policy",
            f"allowed {cfg.heal.allowed_paths} · forbidden {cfg.heal.forbidden_paths} · "
            f"extensions {cfg.heal.allowed_extensions} · auto_apply={cfg.heal.auto_apply}",
        )
    )
    backups = list_backups(root)
    rows.append(("OK", "backups", f"{len(backups)} available" if backups else "none"))

    notif = cfg.notifications
    if notif.enabled:
        channels: list[str] = []
        for name, enabled, env in (
            ("telegram", notif.telegram.enabled, notif.telegram.bot_token_env),
            ("slack", notif.slack.enabled, notif.slack.webhook_url_env),
            ("webhook", notif.webhook.enabled, notif.webhook.url_env),
            ("sentry", notif.sentry.enabled, notif.sentry.dsn_env),
        ):
            if not enabled:
                continue
            channels.append(name)
            if not os.environ.get(env):
                rows.append(("FAIL", f"notifications.{name}", f"${env} is not set"))
        if notif.telegram.enabled and not notif.telegram.chat_id:
            rows.append(("FAIL", "notifications.telegram", "chat_id is empty"))
        rows.append(
            (
                "OK" if channels else "WARN",
                "notifications",
                ", ".join(channels) or "no channel enabled",
            )
        )
    else:
        rows.append(("OK", "notifications", "disabled"))
    return rows


@app.command()
def doctor(project: Path | None = typer.Option(None, "--project", "-p")) -> None:
    """Check configuration, credentials, index and tooling; exit 1 on failures."""
    root = _project_root(project)
    cfg = _load(root)
    rows = _check_rows(root, cfg)
    table = Table(title=f"self-heal {__version__} doctor")
    table.add_column("status", no_wrap=True)
    table.add_column("check", no_wrap=True)
    table.add_column("detail")
    styles = {"OK": "green", "WARN": "yellow", "FAIL": "red"}
    for status_, check, detail in rows:
        table.add_row(f"[{styles[status_]}]{status_}[/{styles[status_]}]", check, detail)
    console.print(table)
    if any(r[0] == "FAIL" for r in rows):
        raise typer.Exit(code=1)


@app.command()
def version() -> None:
    """Print the installed version."""
    console.print(f"self-heal {__version__}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
