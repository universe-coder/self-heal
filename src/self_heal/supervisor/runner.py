"""Supervise a subprocess: capture stderr, optionally heal and restart."""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import IO

from self_heal.config import SelfHealConfig, load_config_for_root
from self_heal.healer.pipeline import heal_from_traceback_text
from self_heal.llm.client import LLMClient
from self_heal.supervisor.tb_parser import last_traceback, normalize_traceback, should_restart_exit

log = logging.getLogger(__name__)

SIGINT_EXIT_CODE = 130


def _pump_stderr(pipe: IO[str], sink: list[str]) -> None:
    for line in iter(pipe.readline, ""):
        sink.append(line)
        sys.stderr.write(line)
        sys.stderr.flush()


def _terminate(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def run_supervised(
    argv: list[str],
    *,
    project_root: Path,
    cfg: SelfHealConfig | None = None,
    auto_apply: bool = False,
    dry_run: bool = True,
    client: LLMClient | None = None,
) -> int:
    """Run ``argv``; on a stderr traceback (or a configured exit code) heal and restart.

    Restarts stop after ``supervisor.max_restarts``, or earlier when the process
    dies with the same traceback again and no patch was applied in between.
    Ctrl-C terminates the child and returns 130.
    """
    root = project_root.resolve()
    conf = cfg or load_config_for_root(root)
    sup = conf.supervisor
    llm = client
    restarts = 0
    prev_tb: str | None = None

    while True:
        stderr_lines: list[str] = []
        proc: subprocess.Popen[str] = subprocess.Popen(
            argv,
            cwd=str(root),
            stdout=None,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert proc.stderr is not None
        pump = threading.Thread(target=_pump_stderr, args=(proc.stderr, stderr_lines), daemon=True)
        pump.start()
        try:
            exit_code = proc.wait()
        except KeyboardInterrupt:
            log.info("Interrupted; terminating supervised process")
            _terminate(proc)
            return SIGINT_EXIT_CODE
        pump.join(timeout=5)

        tb = last_traceback("".join(stderr_lines))
        if exit_code == 0 and tb is None:
            return 0
        if tb is None and not should_restart_exit(exit_code, sup.restart_on_exit_codes):
            return exit_code

        applied = False
        if tb is not None:
            if (
                sup.stop_on_repeated_traceback
                and prev_tb is not None
                and normalize_traceback(tb) == normalize_traceback(prev_tb)
            ):
                log.warning("Process died with the same traceback again; not restarting")
                return exit_code
            log.info(
                "Detected traceback; invoking heal (dry_run=%s, auto_apply=%s)", dry_run, auto_apply
            )
            if llm is None:
                llm = LLMClient(conf)
            try:
                result = heal_from_traceback_text(
                    tb,
                    project_root=root,
                    mode="apply" if auto_apply else "suggest",
                    cfg=conf,
                    client=llm,
                    auto_apply_override=auto_apply,
                    dry_run=dry_run,
                )
            except Exception as e:  # noqa: BLE001
                log.warning("heal failed: %s", e)
                result = None
            applied = bool(result is not None and result.applied)
            prev_tb = None if applied else tb

        if restarts >= sup.max_restarts:
            log.warning("Max restarts (%d) reached", sup.max_restarts)
            return exit_code

        restarts += 1
        log.info("Restarting supervised process (%d/%d)…", restarts, sup.max_restarts)
        if sup.healthcheck_grace_s > 0:
            time.sleep(sup.healthcheck_grace_s)
