from self_heal.supervisor.tb_parser import last_traceback, normalize_traceback, should_restart_exit


def test_last_traceback() -> None:
    t = (
        "noise\nTraceback (most recent call last):\n"
        '  File "x.py", line 1\n'
        "    raise ValueError('bad')\n"
        "ValueError: bad\n"
    )
    tb = last_traceback(t)
    assert tb is not None
    assert tb.startswith("Traceback") and "ValueError" in tb and "noise" not in tb


def test_last_traceback_keeps_chained_causes() -> None:
    t = (
        "Traceback (most recent call last):\n  File \"a.py\", line 1\nKeyError: 'k'\n\n"
        "During handling of the above exception, another exception occurred:\n\n"
        'Traceback (most recent call last):\n  File "a.py", line 3\nValueError: bad\n'
    )
    tb = last_traceback(
        'old\nTraceback (most recent call last):\n  File "z.py"\nTypeError: x\n' + t
    )
    assert tb is not None
    assert tb.startswith("Traceback")
    assert "KeyError" in tb and "ValueError" in tb
    assert "TypeError" not in tb  # an earlier, unrelated traceback is dropped


def test_last_traceback_handles_headerless_syntax_error() -> None:
    t = 'starting\n  File "app.py", line 3\n    def f(:\n          ^\nSyntaxError: invalid syntax\n'
    tb = last_traceback(t)
    assert tb is not None
    assert tb.startswith('File "app.py"') and tb.endswith("SyntaxError: invalid syntax")
    assert last_traceback("just logs\n") is None


def test_normalize_traceback_ignores_addresses_and_trailing_space() -> None:
    a = "ValueError: <obj at 0x7f00dead>   \n"
    b = "ValueError: <obj at 0x7f00beef>\n"
    assert normalize_traceback(a) == normalize_traceback(b)


def test_should_restart() -> None:
    assert should_restart_exit(1, [1]) is True
    assert should_restart_exit(0, [1]) is False
    assert should_restart_exit(None, []) is True
