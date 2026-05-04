"""Demo app with an intentional bug for self-heal demos/tests."""

from __future__ import annotations


def run() -> float:
    return 1 / 0


if __name__ == "__main__":
    print(run())
