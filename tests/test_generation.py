import json

import pytest

from src.evaluation.generation_eval import FAILED, load_done
from src.evaluation.judge import ANSWERABLE_LABELS, UNANSWERABLE_LABELS, parse_label
from src.generation.answer import NOT_IN_DOCS, build_prompt
from src.generation.groq_chat import ChatResult, IncompleteResponse, complete_text


def test_prompts_closed_book_and_rag_context():
    system, user = build_prompt("closed_book", "What is a Pod?")
    assert user == "Question: What is a Pod?" and "excerpts" not in system

    chunks = [{"doc_path": "concepts/a.md", "text": "Alpha text.\n"},
              {"doc_path": "tasks/b.md", "text": "Beta text."}]
    system, user = build_prompt("rag", "What is a Pod?", chunks)
    assert NOT_IN_DOCS in system and "only" in system
    assert user.startswith("Documentation excerpts:\n\n[1] Page: concepts/a.md\nAlpha text.\n\n[2] Page: tasks/b.md\nBeta text.")
    assert user.endswith("Question: What is a Pod?")
    with pytest.raises(ValueError):
        build_prompt("rag", "q")


def test_judge_label_parsing_and_failed_replies():
    assert parse_label("INCORRECT", ANSWERABLE_LABELS) == "INCORRECT"  # not also read as CORRECT
    assert parse_label("correct.", ANSWERABLE_LABELS) == "CORRECT"
    assert parse_label("NOT_ATTEMPTED", ANSWERABLE_LABELS) == "NOT_ATTEMPTED"
    assert parse_label("Abstained", UNANSWERABLE_LABELS) == "ABSTAINED"
    for raw in ["CORRECT or INCORRECT", "maybe", ""]:
        with pytest.raises(ValueError):
            parse_label(raw, ANSWERABLE_LABELS)

    # Empty replies and replies cut off at the token limit count as failed.
    with pytest.raises(IncompleteResponse):
        complete_text(ChatResult(content="  ", finish_reason="stop", completion_tokens=5))
    with pytest.raises(IncompleteResponse):
        complete_text(ChatResult(content="partial", finish_reason="length", completion_tokens=4096))
    assert complete_text(ChatResult(content=" 30 seconds ", finish_reason="stop", completion_tokens=9)) == "30 seconds"


def test_resume_skips_done_rows_for_current_version_only(tmp_path):
    path = tmp_path / "dev_rag.jsonl"
    rows = [
        {"id": "q001", "version": "v1", "label": "CORRECT"},
        {"id": "q002", "version": "v0", "label": "CORRECT"},   # older prompt version: redo
        {"id": "q003", "version": "v1", "label": FAILED},      # failed: retry
        {"id": "q004", "version": "v1", "label": FAILED},
        {"id": "q004", "version": "v1", "label": "ABSTAINED"},  # retried later and succeeded
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert sorted(load_done(path, "v1")) == ["q001", "q004"]
    assert load_done(tmp_path / "missing.jsonl", "v1") == {}
