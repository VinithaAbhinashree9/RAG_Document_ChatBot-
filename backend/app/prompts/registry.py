"""Prompt registry.

Prompts live in `prompts/templates/*.txt`, not inside business logic, so they can
be reviewed and version-bumped independently of the code that uses them.
"""

from __future__ import annotations

import logging
from enum import Enum
from functools import lru_cache
from pathlib import Path
from string import Formatter

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

# Bump when a template's contract changes; surfaced in logs and /health.
PROMPT_VERSION = "1.0.0"


class PromptName(str, Enum):
    """Every prompt used by the application."""

    SYSTEM = "system"
    QA_SINGLE = "qa_single"
    QA_MULTI = "qa_multi"
    FOLLOWUP_REWRITE = "followup_rewrite"
    SUMMARIZE_MAP = "summarize_map"
    SUMMARIZE_REDUCE = "summarize_reduce"
    SUMMARIZE_DIRECT = "summarize_direct"
    CONTEXT_VERIFICATION = "context_verification"
    SOURCE_ATTRIBUTION = "source_attribution"


@lru_cache(maxsize=None)
def load_template(name: PromptName | str) -> str:
    """Read a template from disk (cached)."""
    key = name.value if isinstance(name, PromptName) else str(name)
    path = TEMPLATE_DIR / f"{key}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def required_fields(name: PromptName | str) -> set[str]:
    """Placeholders a template expects — used by tests to catch drift."""
    template = load_template(name)
    return {
        field for _, field, _, _ in Formatter().parse(template) if field
    }


def render(name: PromptName | str, **kwargs: object) -> str:
    """Render a template, failing loudly on a missing placeholder."""
    template = load_template(name)
    needed = required_fields(name)
    missing = needed - set(kwargs)
    if missing:
        raise KeyError(
            f"Prompt '{name}' is missing required variables: {sorted(missing)}"
        )
    return template.format(**kwargs)


def system_prompt() -> str:
    """The shared anti-hallucination system prompt."""
    return load_template(PromptName.SYSTEM).strip()


# Canonical refusal string. The API, prompts and tests all reference this
# single constant so the contract can never drift.
REFUSAL_MESSAGE = "I couldn't find this information in the uploaded document."
