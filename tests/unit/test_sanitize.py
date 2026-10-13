import pytest

from self_heal.runtime.sanitize import mask_secrets, sanitize_mapping, sanitize_traceback_text


def test_sanitize_secret_key() -> None:
    m = sanitize_mapping({"password": "secret123", "n": 1})
    assert m["password"] == "[redacted]"
    assert m["n"] == "1"


def test_sanitize_bearer() -> None:
    s = sanitize_traceback_text("Authorization: Bearer token123\n")
    assert "token123" not in s


def test_sanitize_masks_secret_env_values_everywhere(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_SERVICE_TOKEN", "hunter2hunter2")
    monkeypatch.setenv("HOME_DIR_NAME", "short")  # not secret-looking, must not be masked
    m = sanitize_mapping({"url": "https://x.test/?a=hunter2hunter2", "plain": "hunter2hunter2"})
    assert "hunter2hunter2" not in m["url"]
    assert m["plain"] == "[redacted]"
    tb = sanitize_traceback_text("ValueError: bad token hunter2hunter2 at short\n")
    assert "hunter2hunter2" not in tb
    assert "short" in tb


@pytest.mark.parametrize(
    "text",
    [
        "key sk-abcdefghijklmnopqrstuvwxyz0123",
        "token ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ1234",
        "slack xoxb-123456789012-abcdefghijkl",
        "aws AKIAIOSFODNN7EXAMPLE",
        "jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6",
        "db postgres://user:pa55word@db.internal/app",
        "query https://api.test/v1?api_key=abc123&x=1",
    ],
)
def test_mask_secrets_known_shapes(text: str) -> None:
    out = mask_secrets(text, env_values=[])
    assert "[redacted]" in out
    for leaked in ("sk-abcdef", "ghp_ABC", "xoxb-", "AKIAIOSF", "eyJhbGci", "pa55word", "abc123"):
        assert leaked not in out


def test_sanitize_mapping_skips_dunders_and_limits_length() -> None:
    m = sanitize_mapping({"__builtins__": {}, "big": "x" * 5000})
    assert "__builtins__" not in m
    assert len(m["big"]) <= 2000
