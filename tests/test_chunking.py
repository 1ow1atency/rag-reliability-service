from collections import Counter

import pytest

from src.ingest.chunking import (
    MAX_CHUNK_TOKENS,
    Doc,
    Tokenizer,
    clean_markdown,
    heading_chunks,
    token_windows,
)


@pytest.fixture(scope="module")
def tok():
    return Tokenizer()


def filler(n_words: int, prefix: str = "item") -> str:
    return " ".join(f"{prefix}{i}" for i in range(n_words))


def test_clean_markdown_strips_front_matter_and_shortcodes():
    raw = (
        '---\ntitle: "Pods"\nweight: 10\n---\n<!-- overview -->\n'
        'A {{< glossary_tooltip text="Pod" term_id="pod" >}} runs on a '
        '{{< glossary_tooltip term_id="node" >}}.\n\n'
        "{{< note >}}\nBe careful.\n{{< /note >}}\n\n"
        '## {{% heading "prerequisites" %}}\n\n'
        "```shell\n## {{< not-a-heading >}}\n```\n\n"
        "## {{< some-widget >}}\n"
    )
    counts = Counter()
    title, text = clean_markdown(raw, counts)
    assert title == "Pods"
    assert text == (
        "A Pod runs on a node.\n\nBe careful.\n\n## Before you begin\n\n"
        "```shell\n##\n```\n\n## What's next"
    )
    # Two empty headings outside the code fence; one had no label and fell back.
    assert counts == {"empty_headings": 2, "empty_headings_fallback": 1}

    # Front-matter-only page without a trailing newline cleans to nothing.
    assert clean_markdown('---\ntitle: "Index"\n---') == ("Index", "")


def test_fixed_windows_overlap_and_word_boundaries(tok):
    text = filler(600)
    spans = token_windows(text, tok, 256)

    assert len(spans) > 1
    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    for (s1, e1), (s2, _) in zip(spans, spans[1:]):
        assert tok.count(text[s1:e1]) <= 256
        overlap = tok.count(text[s2:e1])
        assert 30 <= overlap <= 45  # ~15% of 256, adjusted to word boundaries
        assert text[s2 - 1] == " "  # never starts mid-word


def test_heading_chunks_merge_split_and_paths(tok):
    text = "\n\n".join([
        f"## Install\n\n{filler(40, 'inst')}",
        f"### Verify\n\nRun it.\n\n```shell\n## not a heading\nkubectl get pods\n```",
        f"## Configure\n\n{filler(800, 'conf')}",
    ])
    chunks = heading_chunks(Doc("tasks/example.md", "Example", text), tok)

    # Small "Verify" section (and its code fence) is merged into "Install".
    first = chunks[0]
    assert first.heading_path == ["Example", "Install"]
    assert "### Verify" in first.text and "## not a heading" in first.text

    # Oversized "Configure" section is split into <= 510-token windows.
    configure = chunks[1:]
    assert len(configure) >= 2
    assert all(c.heading_path == ["Example", "Configure"] for c in configure)
    assert all(c.n_tokens <= MAX_CHUNK_TOKENS for c in configure)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
