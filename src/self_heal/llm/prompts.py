"""Prompts for unified-diff-only LLM output."""

SYSTEM_GENERATE_PATCH = """\
You are a senior Python engineer. Fix the bug described by the error and context.

Rules:
1. Output ONLY a valid unified diff (git format). No prose before or after.
2. Wrap the diff in a fenced code block using the language tag `diff`, like:

```diff
--- a/path/to/file.py
+++ b/path/to/file.py
...
```

3. Paths in the diff must be relative to the project root (use `a/` and `b/` prefixes as in git).
4. Only modify files that are necessary to fix the error. Keep changes minimal.
5. Do not add `eval`, `exec`, subprocess shell=True, or other unsafe patterns.
6. Preserve existing style and imports when possible.

If you cannot produce a safe patch, output an empty diff block:

```diff

```
"""


def user_message_error_and_context(
    *,
    traceback_text: str,
    retrieved_chunks: str,
    extra_instruction: str = "",
) -> str:
    parts = [
        "## Python traceback\n\n```\n" + traceback_text.strip() + "\n```\n",
        "## Relevant code chunks (from codebase index)\n\n" + retrieved_chunks.strip(),
    ]
    if extra_instruction.strip():
        parts.append("\n## Additional instruction\n\n" + extra_instruction.strip())
    parts.append(
        "\nProduce the minimal unified diff that fixes the error. "
        "Remember: only ```diff ... ``` block, nothing else."
    )
    return "\n".join(parts)
