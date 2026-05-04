from self_heal.supervisor.tb_parser import last_traceback, should_restart_exit


def test_last_traceback() -> None:
    t = (
        "noise\nTraceback (most recent call last):\n"
        '  File "x.py", line 1\n'
        "    raise ValueError('bad')\n"
        "ValueError: bad\n"
    )
    tb = last_traceback(t)
    assert tb is not None
    assert "ValueError" in tb


def test_should_restart() -> None:
    assert should_restart_exit(1, [1]) is True
    assert should_restart_exit(0, [1]) is False
