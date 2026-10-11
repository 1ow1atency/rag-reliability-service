"""Shared Groq chat client with rate-limit (HTTP 429) handling.

Copied from the semantic-entropy-hallucination project (src/groq_chat.py), with prompt and
total token counts added to ChatResult. Settings are unchanged: max_completion_tokens=4096,
hidden low-effort reasoning, and a reply that is empty or hit the token limit is a failure.
"""

import logging
import os
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from groq import Groq, RateLimitError

MAX_RATE_LIMIT_WAITS = 10
MAX_WAIT_SECONDS = 300  # longer waits usually mean a daily quota; stop instead of hanging
DEFAULT_WAIT_SECONDS = 10.0
# Set explicitly: without it Groq's default cut gpt-oss-20b off at 2048 tokens, which hidden
# reasoning can use up before any answer text is written.
MAX_COMPLETION_TOKENS = 4096

logger = logging.getLogger(__name__)

load_dotenv()
_client = None

# Counters for reporting: every request sent to Groq, and how many of those got a 429.
api_calls = 0
rate_limited_calls = 0


@dataclass
class ChatResult:
    content: str | None  # raw reply text; None or partial if the model stopped early
    finish_reason: str | None  # "stop", or "length" if the token limit was hit
    completion_tokens: int | None  # includes hidden reasoning tokens
    prompt_tokens: int | None = None
    total_tokens: int | None = None


class IncompleteResponse(Exception):
    """Raised when a reply hit the token limit or came back with no text."""

    def __init__(self, finish_reason: str | None, message: str):
        super().__init__(message)
        self.finish_reason = finish_reason


def complete_text(result: ChatResult) -> str:
    """Return the stripped reply text, or raise IncompleteResponse if it is cut off or empty."""
    if result.finish_reason == "length":
        raise IncompleteResponse("length", f"hit the token limit after {result.completion_tokens} tokens")
    text = (result.content or "").strip()
    if not text:
        raise IncompleteResponse(result.finish_reason, "empty reply")
    return text


class RateLimitExhausted(Exception):
    """Raised when Groq keeps rate-limiting us, or asks us to wait longer than MAX_WAIT_SECONDS."""


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")
        # Retries are handled here and by callers, not by the SDK.
        _client = Groq(api_key=api_key, max_retries=0)
    return _client


def _retry_after_seconds(error: RateLimitError) -> float:
    try:
        return float(error.response.headers.get("retry-after"))
    except (TypeError, ValueError):
        return DEFAULT_WAIT_SECONDS


def chat(model: str, system_prompt: str, user_message: str, temperature: float) -> ChatResult:
    """Send one chat request and return the raw reply with its finish reason and token usage.

    Reasoning models run with low, hidden reasoning so only the final answer comes back.
    On a 429, waits for the server's retry-after time and tries again.
    """
    global api_calls, rate_limited_calls
    for _ in range(MAX_RATE_LIMIT_WAITS):
        api_calls += 1
        try:
            response = _get_client().chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                reasoning_effort="low",
                reasoning_format="hidden",
            )
            choice = response.choices[0]
            usage = response.usage
            return ChatResult(
                content=choice.message.content,
                finish_reason=choice.finish_reason,
                completion_tokens=usage.completion_tokens if usage else None,
                prompt_tokens=usage.prompt_tokens if usage else None,
                total_tokens=usage.total_tokens if usage else None,
            )
        except RateLimitError as e:
            rate_limited_calls += 1
            wait = _retry_after_seconds(e)
            if wait > MAX_WAIT_SECONDS:
                raise RateLimitExhausted(f"Groq asked to wait {wait:.0f}s ({e})") from e
            logger.warning("Rate limited by Groq; waiting %.1fs.", wait)
            time.sleep(wait)
    raise RateLimitExhausted(f"Still rate limited after {MAX_RATE_LIMIT_WAITS} waits.")
