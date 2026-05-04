"""Supervise a subprocess: capture stderr, optionally heal and restart."""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TextIO

from self_heal.config import SelfHealConfig, load_config_for_root
from self_heal.healer.pipeline import heal_from_traceback_text
from self_heal.llm.client import LLMClient
from self_heal.supervisor.tb_parser import last_traceback, should_restart_exit

log = logging.getLogger(__name__)


def run_supervised(
    argv: list[str],
    *,
    project_root: Path,
    cfg: SelfHealConfig | None = None,
    auto_apply: bool = False,
    dry_run: bool = True,
) -> int:
    """Run `argv`; on stderr traceback or matching exit code, optionally heal and restart."""
    root = project_root.resolve()
    conf = cfg or load_config_for_root(root)
    restarts = 0

    while restarts <= conf.supervisor.max_restarts:
        stderr_chunks: list[str] = []

        def reader(pipe: TextIO) -> None:
            assert pipe is not None
            for line in iter(pipe.readline, ""):
                stderr_chunks.append(line)
                print(line, end="", file=sys.stderr)

        proc = subprocess.Popen(
            argv,
            cwd=str(root),
            stdout=None,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert proc.stderr is not None
        t = threading.Thread(target=reader, args=(proc.stderr,))
        t.daemon = True
        t.start()

        exit_code = proc.wait()
        t.join(timeout=5)
        combined_err = "".join(stderr_chunks)

        tb = last_traceback(combined_err)

        if exit_code == 0 and tb is None:
            return 0

        codes = conf.supervisor.restart_on_exit_codes
        should_restart = (tb is not None) or should_restart_exit(exit_code, codes)

        if not should_restart:
            return exit_code if exit_code is not None else 1

        if tb is not None:
            log.info(
                "Detected traceback; invoking heal (dry_run=%s, auto_apply=%s)",
                dry_run,
                auto_apply,
            )
            client = LLMClient(conf)
            heal_from_traceback_text(
                tb,
                project_root=root,
                mode="apply" if auto_apply else "suggest",
                cfg=conf,
                client=client,
                auto_apply_override=auto_apply,
                dry_run=dry_run,
            )

        if restarts >= conf.supervisor.max_restarts:
            log.warning("Max restarts (%d) reached", conf.supervisor.max_restarts)
            return exit_code if exit_code is not None else 1

        restarts += 1
        log.info("Restarting supervised process (%d/%d)…", restarts, conf.supervisor.max_restarts)
        time.sleep(conf.supervisor.healthcheck_grace_s)

    return 1
