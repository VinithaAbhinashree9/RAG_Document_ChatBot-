"""Text normalisation.

Cleaning happens before chunking so that embeddings are computed over
consistent text. This is deliberately conservative: it removes noise that hurts
retrieval (control characters, ligatures, hyphenation artefacts, repeated
headers) but never rewrites the document's actual wording.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

# Zero-width and non-printing characters that PDFs frequently contain.
_INVISIBLE = re.compile(r"[\u200b\u200c\u200d\u200e\u200f\u2060\ufeff\xad]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Common PDF ligatures and typographic substitutions.
_REPLACEMENTS = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u2013": "-",
    "\u2014": " - ",
    "\u2026": "...",
    "\u00a0": " ",
    "\u2022": "* ",
    "\t": "    ",
}

# "hyphen-\nated" -> "hyphenated" (line-break hyphenation from PDFs)
_HYPHEN_BREAK = re.compile(r"(\w)-\s*\n\s*(\w)")
# Collapse 3+ blank lines to exactly one blank line.
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")
# Collapse runs of spaces/tabs but never newlines.
_EXCESS_SPACES = re.compile(r"[ \t]{2,}")
# Lines that are only punctuation/underscores (decorative rules).
_DECORATIVE = re.compile(r"^[\s._\-=*~#|+]{4,}$")
# Bare page-number lines, e.g. "12", "- 12 -", "Page 12 of 30".
_PAGE_NUMBER_LINE = re.compile(
    r"^(?:[-–—\s]*\d{1,4}[-–—\s]*|page\s+\d{1,4}(?:\s+of\s+\d{1,4})?)$",
    re.IGNORECASE,
)


def normalize_unicode(text: str) -> str:
    """NFKC-normalise and strip invisible characters."""
    text = unicodedata.normalize("NFKC", text)
    for source, target in _REPLACEMENTS.items():
        text = text.replace(source, target)
    text = _INVISIBLE.sub("", text)
    text = _CONTROL.sub(" ", text)
    return text


def clean_text(text: str, *, join_hyphenated: bool = True) -> str:
    """Normalise a single block of extracted text."""
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = normalize_unicode(text)

    if join_hyphenated:
        text = _HYPHEN_BREAK.sub(r"\1\2", text)

    lines: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        if _DECORATIVE.match(stripped):
            continue
        if _PAGE_NUMBER_LINE.match(stripped):
            continue
        lines.append(_EXCESS_SPACES.sub(" ", stripped))

    text = "\n".join(lines)
    text = _EXCESS_BLANK_LINES.sub("\n\n", text)
    return text.strip()


def find_repeated_lines(
    page_texts: list[str], *, min_pages: int = 3, ratio: float = 0.6
) -> set[str]:
    """Identify running headers/footers that repeat across most pages.

    Only applied when there are enough pages to be confident a repeated short
    line is boilerplate rather than meaningful content.
    """
    if len(page_texts) < min_pages:
        return set()

    counter: Counter[str] = Counter()
    for page in page_texts:
        candidate_lines = [line.strip() for line in page.split("\n") if line.strip()]
        # Headers/footers live at the edges of a page and are short.
        edges = candidate_lines[:3] + candidate_lines[-3:]
        for line in set(edges):
            if 3 <= len(line) <= 120:
                counter[line] += 1

    threshold = max(min_pages, int(len(page_texts) * ratio))
    return {line for line, count in counter.items() if count >= threshold}


def strip_lines(text: str, banned: set[str]) -> str:
    """Remove exact-match boilerplate lines."""
    if not banned:
        return text
    kept = [line for line in text.split("\n") if line.strip() not in banned]
    return _EXCESS_BLANK_LINES.sub("\n\n", "\n".join(kept)).strip()


def is_meaningful(text: str, *, min_chars: int = 20, min_alpha_ratio: float = 0.2) -> bool:
    """Reject blocks that are too short or mostly non-alphanumeric noise."""
    stripped = text.strip()
    if len(stripped) < min_chars:
        return False
    alnum = sum(1 for char in stripped if char.isalnum())
    return (alnum / len(stripped)) >= min_alpha_ratio
