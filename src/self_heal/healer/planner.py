"""Build LLM messages for healing."""

from __future__ import annotations

from self_heal.llm.prompts import SYSTEM_GENERATE_PATCH, user_message_error_and_context


def build_healing_prompt(
    traceback_text: str,
    retrieved_chunks: str,
    extra: str = "",
) -> tuple[str, str]:
    user = user_message_error_and_context(
        traceback_text=traceback_text,
        retrieved_chunks=retrieved_chunks,
        extra_instruction=extra,
    )
    return SYSTEM_GENERATE_PATCH, user
