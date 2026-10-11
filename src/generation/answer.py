"""Answer questions with openai/gpt-oss-20b at temperature 0, closed-book or with retrieved context.

closed_book: the question only.
rag:         the question plus the top chunks from the locked retrieval config, each labelled with
             its page path; the model must answer only from them or reply NOT_IN_DOCS.
"""

import hashlib
import logging
from dataclasses import asdict, dataclass, field

from src.generation.groq_chat import IncompleteResponse, RateLimitExhausted, chat, complete_text

MODEL = "openai/gpt-oss-20b"
TEMPERATURE = 0.0
MAX_ATTEMPTS = 2  # one initial try plus one retry
NOT_IN_DOCS = "NOT_IN_DOCS"
MODES = ("closed_book", "rag")

CLOSED_BOOK_SYSTEM_PROMPT = (
    "You answer questions about Kubernetes. Answer directly and concisely: give only the answer, "
    "with no explanation. If you do not know the answer, say that you don't know."
)

RAG_SYSTEM_PROMPT = (
    "You answer questions about Kubernetes using only the documentation excerpts in the user's "
    "message. Do not use any other knowledge. Answer directly and concisely: give only the answer, "
    "with no explanation. If the excerpts state the answer, or it follows directly from what they "
    "state (for example, a yes/no question settled by a stated fact), give it. Only if the excerpts "
    f"do not contain the information, reply with exactly {NOT_IN_DOCS} and nothing else."
)

logger = logging.getLogger(__name__)


def format_context(chunks: list[dict]) -> str:
    """Number each chunk and label it with its page path."""
    return "\n\n".join(
        f"[{i}] Page: {c['doc_path']}\n{c['text'].strip()}" for i, c in enumerate(chunks, 1)
    )


def build_prompt(mode: str, question: str, chunks: list[dict] | None = None) -> tuple[str, str]:
    """Return (system_prompt, user_message) for a mode."""
    if mode == "closed_book":
        return CLOSED_BOOK_SYSTEM_PROMPT, f"Question: {question}"
    if mode == "rag":
        if chunks is None:
            raise ValueError("rag mode needs retrieved chunks")
        return RAG_SYSTEM_PROMPT, f"Documentation excerpts:\n\n{format_context(chunks)}\n\nQuestion: {question}"
    raise ValueError(f"unknown mode {mode!r}")


def prompt_version() -> str:
    """Short hash of everything that changes an answer, used to invalidate cached results."""
    text = "\x00".join([MODEL, str(TEMPERATURE), CLOSED_BOOK_SYSTEM_PROMPT, RAG_SYSTEM_PROMPT,
                        format_context([{"doc_path": "P", "text": "T"}])])
    return hashlib.sha256(text.encode()).hexdigest()[:12]


@dataclass
class Answer:
    text: str | None  # None if both attempts failed
    finish_reason: str | None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0  # summed over attempts, failed ones included
    calls: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def answer(mode: str, question: str, chunks: list[dict] | None = None) -> Answer:
    """Ask the model once (retrying once on a failed call) and return the answer with token usage.

    An empty reply or one that hit the token limit counts as failed. Rate limits are waited
    out in chat(); RateLimitExhausted is raised to the caller.
    """
    system_prompt, user_message = build_prompt(mode, question, chunks)
    out = Answer(text=None, finish_reason=None)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            result = chat(MODEL, system_prompt, user_message, TEMPERATURE)
        except RateLimitExhausted:
            raise
        except Exception as e:
            out.calls += 1
            out.errors.append(f"{type(e).__name__}: {e}")
            continue
        out.calls += 1
        out.prompt_tokens += result.prompt_tokens or 0
        out.completion_tokens += result.completion_tokens or 0
        out.total_tokens += result.total_tokens or 0
        out.finish_reason = result.finish_reason
        try:
            out.text = complete_text(result)
            return out
        except IncompleteResponse as e:
            out.errors.append(f"attempt {attempt}: {e}")
            logger.warning("Answer attempt %d failed (%s).", attempt, e)
    return out
