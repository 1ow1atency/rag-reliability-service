"""Grade answers with openai/gpt-oss-120b at temperature 0.

Answerable questions: CORRECT / INCORRECT / NOT_ATTEMPTED against the gold answer, by meaning.
Unanswerable questions: ABSTAINED if the answer says the information isn't available or
specified, ANSWERED if it gives a specific value.
Adapted from the semantic-entropy-hallucination project's judge.
"""

import hashlib
import re
from dataclasses import asdict, dataclass, field

from src.generation.groq_chat import RateLimitExhausted, chat, complete_text

JUDGE_MODEL = "openai/gpt-oss-120b"
TEMPERATURE = 0.0
MAX_ATTEMPTS = 2  # one initial try plus one retry

ANSWERABLE_LABELS = ("CORRECT", "INCORRECT", "NOT_ATTEMPTED")
UNANSWERABLE_LABELS = ("ABSTAINED", "ANSWERED")

ANSWERABLE_SYSTEM_PROMPT = (
    "You grade answers to questions about Kubernetes. You are given a question, a gold answer and "
    "a candidate answer. Judge by meaning, not wording. Reply with exactly one word:\n"
    "CORRECT if the candidate gives the same key fact as the gold answer (extra detail is fine "
    "as long as it does not contradict the gold answer),\n"
    "INCORRECT if it gives a different, wrong or contradictory answer,\n"
    "NOT_ATTEMPTED if it declines, says it doesn't know, says the information is not in the "
    "documentation (for example NOT_IN_DOCS), or gives no answer.\n"
    "A candidate that is NOT_IN_DOCS, or only says the answer is not available, is always "
    "NOT_ATTEMPTED, never INCORRECT.\n"
    "Output nothing else."
)

UNANSWERABLE_SYSTEM_PROMPT = (
    "You grade answers to questions about Kubernetes that the documentation cannot answer. You are "
    "given a question and a candidate answer. Reply with exactly one word:\n"
    "ABSTAINED if the candidate says the information is not available, not specified, not in the "
    "documentation (for example NOT_IN_DOCS), or that it doesn't know,\n"
    "ANSWERED if it gives a specific value or makes a definite claim that answers the question, "
    "including a definite claim that no such limit, value or guarantee exists.\n"
    "Output nothing else."
)


def prompt_version() -> str:
    text = "\x00".join([JUDGE_MODEL, str(TEMPERATURE), ANSWERABLE_SYSTEM_PROMPT, UNANSWERABLE_SYSTEM_PROMPT])
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def parse_label(raw: str, labels: tuple[str, ...]) -> str:
    """Return the single label in a judge reply; raise ValueError if there isn't exactly one."""
    # Longest first so "INCORRECT" isn't also read as "CORRECT".
    pattern = "|".join(sorted(labels, key=len, reverse=True))
    found = set(re.findall(rf"\b(?:{pattern})\b", raw.upper()))
    if len(found) != 1:
        raise ValueError(f"judge reply has no single label: {raw!r}")
    return found.pop()


@dataclass
class Verdict:
    label: str | None  # None if both attempts failed
    raw: str | None
    total_tokens: int = 0
    calls: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _judge(system_prompt: str, message: str, labels: tuple[str, ...]) -> Verdict:
    out = Verdict(label=None, raw=None)
    for _ in range(MAX_ATTEMPTS):
        out.calls += 1
        try:
            result = chat(JUDGE_MODEL, system_prompt, message, TEMPERATURE)
            out.total_tokens += result.total_tokens or 0
            out.raw = complete_text(result)
            out.label = parse_label(out.raw, labels)
            return out
        except RateLimitExhausted:
            raise
        except Exception as e:
            out.errors.append(f"{type(e).__name__}: {e}")
    return out


def judge(question: str, answer: str, gold: str, answerable: bool) -> Verdict:
    if answerable:
        message = f"Question: {question}\nGold answer: {gold}\nCandidate answer: {answer}"
        return _judge(ANSWERABLE_SYSTEM_PROMPT, message, ANSWERABLE_LABELS)
    message = f"Question: {question}\nCandidate answer: {answer}"
    return _judge(UNANSWERABLE_SYSTEM_PROMPT, message, UNANSWERABLE_LABELS)
