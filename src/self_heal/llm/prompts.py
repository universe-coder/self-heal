"""Prompts for unified-diff-only LLM output."""

SYSTEM_GENERATE_PATCH = """\
You are a senior Python engineer. Fix the bug described by the error and context.

Rules:
1. Output ONLY a valid unified diff (git format). No prose before or after.
2. Wrap the diff in a fenced code block using the language tag `diff`, like:

```diff
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -10,7 +10,7 @@
...
```

3. Paths in the diff must be relative to the project root (use `a/` and `b/` prefixes as in git).
4. Only modify files that are necessary to fix the error. Keep changes minimal.
5. Context and removed lines must match the current file content exactly (whitespace included).
   Code excerpts marked "(from traceback)" are shown with a line-number prefix
   (`   42  code`); that prefix is NOT part of the file. Use the numbers for hunk headers.
6. Do not add `eval`, `exec`, subprocess shell=True, or other unsafe patterns.
7. Preserve existing style and imports when possible.

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
        "## Relevant code (traceback frames and indexed chunks)\n\n" + retrieved_chunks.strip(),
    ]
    if extra_instruction.strip():
        parts.append("\n## Additional instructions\n\n" + extra_instruction.strip())
    parts.append(
        "\nProduce the minimal unified diff that fixes the error. "
        "Remember: only a ```diff ... ``` block, nothing else."
    )
    return "\n".join(parts)


def feedback_empty_diff() -> str:
    return (
        "Your previous reply contained no unified diff. If a fix is possible, reply with a "
        "```diff``` block only; the diff must contain `---`/`+++` headers and at least one hunk."
    )


def feedback_rejected(reason: str) -> str:
    return (
        f"Your previous patch was rejected before being applied: {reason}. "
        "Only modify allowed source files inside the project, keep the diff small, "
        "and use paths relative to the project root."
    )


def feedback_apply_failed(reason: str) -> str:
    return (
        f"Applying your previous patch failed: {reason}. "
        "Regenerate the diff so that every context and removed line matches the current file "
        "content exactly, with correct @@ hunk line numbers."
    )
