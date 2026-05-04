from self_heal.runtime.sanitize import sanitize_mapping, sanitize_traceback_text


def test_sanitize_secret_key() -> None:
    m = sanitize_mapping({"password": "secret123", "n": 1})
    assert "password" in m
    assert "secret123" not in m["password"]


def test_sanitize_bearer() -> None:
    s = sanitize_traceback_text("Authorization: Bearer token123\n")
    assert "token123" not in s
