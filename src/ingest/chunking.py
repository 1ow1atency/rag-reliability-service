"""Chunk the cleaned Kubernetes docs three ways and write each to data/processed.

Chunkers:
  fixed_256 / fixed_512: sliding token windows with 15% overlap, snapped to word boundaries.
           fixed_512 keeps its name but uses MAX_CHUNK_TOKENS (510) windows.
  heading: split on ## / ### headings, merge sections under 100 tokens into the previous
           one, and split sections over 510 tokens into 510-token windows (15% overlap).

Token counts use the BAAI/bge-small-en-v1.5 tokenizer without special tokens; the 510 budget
leaves room for [CLS] and [SEP] within the model's 512-token limit.
A chunk's heading path is the title plus the ## / ### headings in effect where it starts.
"""

import bisect
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from src.ingest.stats import DOCS_ROOT, FRONT_MATTER_RE, SECTIONS

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "processed"

TOKENIZER_NAME = "BAAI/bge-small-en-v1.5"
OVERLAP = 0.15
MIN_SECTION_TOKENS = 100
MAX_CHUNK_TOKENS = 510  # 512 model limit minus [CLS] and [SEP]

TITLE_RE = re.compile(r"^title:\s*[\"']?(.*?)[\"']?\s*$", re.MULTILINE)
SHORTCODE_RE = re.compile(r"\{\{<(.*?)>\}\}|\{\{%(.*?)%\}\}", re.DOTALL)
SHORTCODE_ATTR_RE = re.compile(r'(\w+)="([^"]*)"')
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
HEADING_RE = re.compile(r"^(#{2,3})\s+(.+?)\s*(?:\{#[^}]*\})?\s*$")
ANY_HEADING_RE = re.compile(r"^( {0,3}#{2,6})(?:[ \t]+(.*?))?[ \t]*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")

EMPTY_HEADING_LABEL = "What's next"

# Labels Hugo renders for {{% heading "..." %}} (from the site's i18n strings).
HEADING_LABELS = {
    "prerequisites": "Before you begin",
    "whatsnext": "What's next",
    "cleanup": "Cleaning up",
    "objectives": "Objectives",
    "seealso": "See also",
}


@dataclass
class Doc:
    path: str
    title: str
    text: str


@dataclass
class Chunk:
    chunk_id: str
    doc_path: str
    heading_path: list[str]
    chunk_index: int
    text: str
    n_tokens: int


class Tokenizer:
    def __init__(self, name: str = TOKENIZER_NAME):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.tok.model_max_length = 10**9  # we window long docs ourselves

    def count(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def encode(self, text: str) -> tuple[list[str], list[tuple[int, int]]]:
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        return self.tok.convert_ids_to_tokens(enc["input_ids"]), enc["offset_mapping"]


# ---------- cleaning ----------


def _replace_shortcode(match: re.Match) -> str:
    body = (match.group(1) or match.group(2)).strip()
    name = body.split(maxsplit=1)[0] if body else ""
    if name == "glossary_tooltip":
        attrs = dict(SHORTCODE_ATTR_RE.findall(body))
        return attrs.get("text") or attrs.get("term_id", "")
    if name == "heading":
        key = body.removeprefix("heading").strip().strip('"')
        return HEADING_LABELS.get(key, key)
    return ""


def _resolve_empty_headings(text: str, counts: Counter) -> str:
    """Relabel headings (outside code fences) that are empty once shortcodes are stripped.

    A heading shortcode gets the label Hugo renders for it; anything still empty
    falls back to EMPTY_HEADING_LABEL.
    """
    lines, in_fence = [], False
    for line in text.split("\n"):
        if FENCE_RE.match(line):
            in_fence = not in_fence
        elif not in_fence and (m := ANY_HEADING_RE.match(line)):
            heading = m.group(2) or ""
            if SHORTCODE_RE.sub("", heading).strip():
                lines.append(line)
                continue
            counts["empty_headings"] += 1
            label = SHORTCODE_RE.sub(_replace_shortcode, heading).strip()
            if not label:
                label = EMPTY_HEADING_LABEL
                counts["empty_headings_fallback"] += 1
            line = f"{m.group(1)} {label}"
        lines.append(line)
    return "\n".join(lines)


def clean_markdown(raw: str, counts: Counter | None = None) -> tuple[str, str]:
    """Return (title, cleaned body) for a raw Hugo markdown file.

    If counts is given, it accumulates empty_headings / empty_headings_fallback.
    """
    counts = Counter() if counts is None else counts
    fm = FRONT_MATTER_RE.match(raw)
    title_match = TITLE_RE.search(fm.group(0)) if fm else None
    title = title_match.group(1).strip() if title_match else ""
    text = raw[fm.end():] if fm else raw
    text = HTML_COMMENT_RE.sub("", text)
    text = _resolve_empty_headings(text, counts)
    text = SHORTCODE_RE.sub(_replace_shortcode, text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return title, text.strip()


def load_docs(counts: Counter | None = None) -> list[Doc]:
    docs = []
    for section in SECTIONS:
        for path in sorted((DOCS_ROOT / section).rglob("*.md")):
            title, text = clean_markdown(path.read_text(encoding="utf-8"), counts)
            if text:  # skips the front-matter-only _index.md pages
                docs.append(Doc(str(path.relative_to(DOCS_ROOT)), title, text))
    return docs


# ---------- headings ----------


def find_headings(text: str) -> list[tuple[int, int, str]]:
    """(char offset, level, heading) for ## / ### lines outside code fences."""
    headings, in_fence, pos = [], False, 0
    for line in text.splitlines(keepends=True):
        if FENCE_RE.match(line):
            in_fence = not in_fence
        elif not in_fence and (m := HEADING_RE.match(line.rstrip("\n"))):
            headings.append((pos, len(m.group(1)), m.group(2)))
        pos += len(line)
    return headings


class HeadingIndex:
    """Looks up the heading path in effect at any character offset of a doc."""

    def __init__(self, title: str, headings: list[tuple[int, int, str]]):
        root = [title] if title else []
        self.offsets, self.paths = [0], [root]
        h2 = None
        for offset, level, name in headings:
            if level == 2:
                h2 = name
                path = [name]
            else:
                path = [h2, name] if h2 else [name]
            self.offsets.append(offset)
            self.paths.append(root + path)

    def at(self, offset: int) -> list[str]:
        return self.paths[bisect.bisect_right(self.offsets, offset) - 1]


# ---------- chunkers ----------


def token_windows(text: str, tok: Tokenizer, size: int) -> list[tuple[int, int]]:
    """Character spans of <= size-token windows with OVERLAP, never splitting a word."""
    tokens, offsets = tok.encode(text)
    n = len(tokens)
    if n == 0:
        return []
    stride = size - round(size * OVERLAP)
    spans, start = [], 0
    while True:
        while start < n and tokens[start].startswith("##"):
            start += 1
        end = min(start + size, n)
        while start + 1 < end < n and tokens[end].startswith("##"):
            end -= 1
        spans.append((offsets[start][0], offsets[end - 1][1]))
        if end >= n:
            return spans
        start += stride


def _make_chunks(doc: Doc, spans: list[tuple[int, int]], index: HeadingIndex, tok: Tokenizer) -> list[Chunk]:
    chunks = []
    for start, end in spans:
        text = doc.text[start:end].strip()
        if text:
            i = len(chunks)
            chunks.append(Chunk(f"{doc.path}#{i}", doc.path, index.at(start), i, text, tok.count(text)))
    return chunks


def fixed_chunks(doc: Doc, tok: Tokenizer, size: int) -> list[Chunk]:
    index = HeadingIndex(doc.title, find_headings(doc.text))
    return _make_chunks(doc, token_windows(doc.text, tok, size), index, tok)


def heading_chunks(doc: Doc, tok: Tokenizer) -> list[Chunk]:
    headings = find_headings(doc.text)
    index = HeadingIndex(doc.title, headings)

    bounds = [0] + [offset for offset, _, _ in headings] + [len(doc.text)]
    sections = [(s, e) for s, e in zip(bounds, bounds[1:]) if doc.text[s:e].strip()]

    # Merge small sections into the previous one (a small leading section absorbs the next).
    merged: list[list[int]] = []
    for start, end in sections:
        small = tok.count(doc.text[start:end]) < MIN_SECTION_TOKENS
        if merged and (small or tok.count(doc.text[slice(*merged[-1])]) < MIN_SECTION_TOKENS):
            merged[-1][1] = end
        else:
            merged.append([start, end])

    spans = []
    for start, end in merged:
        if tok.count(doc.text[start:end]) <= MAX_CHUNK_TOKENS:
            spans.append((start, end))
        else:
            spans += [(start + s, start + e) for s, e in token_windows(doc.text[start:end], tok, MAX_CHUNK_TOKENS)]
    return _make_chunks(doc, spans, index, tok)


CHUNKERS = {
    "fixed_256": lambda doc, tok: fixed_chunks(doc, tok, 256),
    "fixed_512": lambda doc, tok: fixed_chunks(doc, tok, MAX_CHUNK_TOKENS),
    "heading": heading_chunks,
}


def main() -> None:
    if not DOCS_ROOT.exists():
        raise SystemExit(f"{DOCS_ROOT} not found; run scripts/fetch_docs.py first")
    tok = Tokenizer()
    counts = Counter()
    docs = load_docs(counts)
    print(f"Documents (non-empty): {len(docs)}")
    print(
        f"Headings empty after stripping shortcodes: {counts['empty_headings']} "
        f"({counts['empty_headings'] - counts['empty_headings_fallback']} given their Hugo label, "
        f"{counts['empty_headings_fallback']} defaulted to {EMPTY_HEADING_LABEL!r})"
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, chunker in CHUNKERS.items():
        chunks = [c for doc in docs for c in chunker(doc, tok)]
        out = OUT_DIR / f"chunks_{name}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
        lengths = np.array([c.n_tokens for c in chunks])
        print(
            f"{name:<10} chunks={len(chunks):>5}  tokens min={lengths.min()} "
            f"median={int(np.median(lengths))} max={lengths.max()}  -> {out.relative_to(ROOT)}"
        )


if __name__ == "__main__":
    main()
