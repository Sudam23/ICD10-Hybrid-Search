"""How a code record is turned into the text that gets indexed.

The dense (embedding) side and the lexical (BM25) side want different things,
so they get different builders:

* The embedding model reads a short, natural sentence. Extra boilerplate
  dilutes the vector, so the variants here are deliberately minimal and are
  compared head-to-head in ``docs/EVALUATION.md``.
* BM25 only cares about which words appear, so it is fed every piece of
  vocabulary attached to the code - including synonyms the official title
  does not contain.
"""

from __future__ import annotations

from icd10_search.data.parse import CodeRecord

DOC_TEXT_VARIANTS = ("desc", "desc_incl", "desc_incl_parent", "desc_incl_parent_block")


def _synonyms(record: CodeRecord) -> list[str]:
    seen: list[str] = []
    for term in [*record.inclusion_terms, *record.inherited_terms]:
        if term not in seen and term.lower() != record.long_desc.lower():
            seen.append(term)
    return seen


def document_text(record: CodeRecord, variant: str = "desc_incl_parent") -> str:
    """Build the string that is embedded for one code."""
    if variant not in DOC_TEXT_VARIANTS:
        raise ValueError(f"unknown document text variant {variant!r}; expected {DOC_TEXT_VARIANTS}")

    head = record.long_desc.strip()
    if head and head[-1] not in ".?!":
        head += "."
    parts: list[str] = [head]

    if variant in ("desc_incl", "desc_incl_parent", "desc_incl_parent_block"):
        synonyms = _synonyms(record)
        if synonyms:
            parts.append("Also called: " + "; ".join(synonyms) + ".")

    if variant in ("desc_incl_parent", "desc_incl_parent_block"):
        parent = (record.parent_desc or "").strip()
        if parent and parent.lower() not in record.long_desc.lower():
            parts.append(f"Part of: {parent}.")

    if variant == "desc_incl_parent_block":
        if record.block_title:
            parts.append(f"Category: {record.block_title}.")

    return " ".join(part.strip() for part in parts if part.strip())


def lexical_text(record: CodeRecord) -> str:
    """Build the bag-of-words string for BM25: title, synonyms and parent title."""
    parts = [record.long_desc, *_synonyms(record)]
    if record.parent_desc and record.parent_desc.lower() not in record.long_desc.lower():
        parts.append(record.parent_desc)
    return " ".join(parts)
