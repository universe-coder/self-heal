"""Typer CLI: init, index, heal, run, status."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer
from rich.console import Console

from self_heal.audit.log import read_recent
from self_heal.config import DEFAULT_CONFIG_NAME, find_config, load_config_for_root
from self_heal.healer.pipeline import heal_from_traceback_text
from self_heal.indexing.indexer import run_index
from self_heal.llm.client import LLMClient
from self_heal.supervisor.runner import run_supervised

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


def _project_root(path: Path | None) -> Path:
    if path is not None:
        return path.resolve()
    found = find_config()
    if found is not None:
        return found.parent.resolve()
    return Path.cwd().resolve()


@app.command()
def init(
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing config"),
    project: Path | None = typer.Option(None, "--project", "-p", help="Project root"),
) -> None:
    """Create `.self-heal.toml` from the built-in example."""
    from importlib import resources

    root = _project_root(project)
    dest = root / DEFAULT_CONFIG_NAME
    if dest.exists() and not force:
        console.print(f"[yellow]{dest} already exists[/yellow] (use --force)")
        raise typer.Exit(code=1)
    try:
        text = resources.files("self_heal.templates").joinpath("self-heal.default.toml").read_text(
            encoding="utf-8"
        )
    except (OSError, FileNotFoundError, ModuleNotFoundError):
        ex = Path(__file__).resolve().parent / "templates" / "self-heal.default.toml"
        text = ex.read_text(encoding="utf-8")
    dest.write_text(text, encoding="utf-8")
    console.print(f"Wrote [green]{dest}[/green]")


@app.command()
def index(
    project: Path | None = typer.Option(None, "--project", "-p"),
) -> None:
    """Index Python files under configured roots (requires embeddings API key)."""
    logging.basicConfig(level=logging.INFO)
    root = _project_root(project)
    cfg = load_config_for_root(root)
    if cfg.llm.provider == "anthropic":
        console.print(
            "[red]Anthropic does not provide an embeddings API. "
            "Set [llm].provider to openai/huggingface/ollama for indexing.[/red]"
        )
        raise typer.Exit(code=1)
    client = LLMClient(cfg)
    if not client.is_configured:
        console.print(
            f"[red]Set {cfg.llm.api_key_env} for embeddings.[/red]",
        )
        raise typer.Exit(code=1)
    n_files, n_chunks = run_index(root, cfg, client)
    console.print(f"Indexed [green]{n_files}[/green] files, [green]{n_chunks}[/green] chunks.")


@app.command("heal")
def heal_cmd(
    traceback_file: Path | None = typer.Option(
        None,
        "--tb",
        help="Read traceback from file (else stdin)",
    ),
    project: Path | None = typer.Option(None, "--project", "-p"),
    apply: bool = typer.Option(False, "--apply", help="Apply patch (respects config policy)"),
    auto: bool = typer.Option(
        False,
        "--auto",
        help="Force auto-apply (still validates paths); implies --apply",
    ),
    dry_run: bool = typer.Option(
        True,
        "--dry-run/--no-dry-run",
        help=(
            "If dry-run, do not apply (audit only). "
            "Use --no-dry-run with --apply to write patches."
        ),
    ),
) -> None:
    """Generate (and optionally apply) a fix from a traceback + indexed codebase."""
    logging.basicConfig(level=logging.INFO)
    root = _project_root(project)
    cfg = load_config_for_root(root)

    if traceback_file is not None:
        tb = traceback_file.read_text(encoding="utf-8", errors="replace")
    else:
        if sys.stdin.isatty():
            console.print("[red]Provide --tb FILE or pipe traceback on stdin[/red]")
            raise typer.Exit(code=1)
        tb = sys.stdin.read()

    want_apply = bool(auto or (apply and not dry_run))

    client = LLMClient(cfg)
    if not client.is_configured:
        console.print(f"[red]Set {cfg.llm.api_key_env}[/red]")
        raise typer.Exit(code=1)

    result = heal_from_traceback_text(
        tb,
        project_root=root,
        mode="apply" if want_apply else "suggest",
        cfg=cfg,
        client=client,
        auto_apply_override=want_apply,
        dry_run=dry_run,
    )
    if result is None:
        raise typer.Exit(code=1)
    console.print(result.message)
    if result.paths_touched:
        console.print("Paths:", ", ".join(result.paths_touched))


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def run(
    ctx: typer.Context,
    auto: bool = typer.Option(False, "--auto", help="Auto-apply patches when healing"),
    dry_run: bool = typer.Option(
        True,
        "--dry-run/--no-dry-run",
        help="With --auto, use --no-dry-run to allow applying patches",
    ),
    project: Path | None = typer.Option(None, "--project", "-p"),
) -> None:
    """Run a command under supervision: `self-heal run -- python -m myapp`."""
    logging.basicConfig(level=logging.INFO)
    root = _project_root(project)
    args = list(ctx.args)
    if not args:
        console.print("[red]Pass command after `--`, e.g. self-heal run -- python app.py[/red]")
        raise typer.Exit(code=1)
    code = run_supervised(
        list(args),
        project_root=root,
        auto_apply=auto,
        dry_run=dry_run,
    )
    raise typer.Exit(code=code)


@app.command()
def status(
    project: Path | None = typer.Option(None, "--project", "-p"),
    limit: int = typer.Option(15, "--limit", "-n"),
) -> None:
    """Show recent audit log entries."""
    root = _project_root(project)
    entries = read_recent(root, limit=limit)
    if not entries:
        console.print("No audit entries yet.")
        return
    for e in entries:
        console.print_json(data=e)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
