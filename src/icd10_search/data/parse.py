"""Turn the raw CMS release into one normalised JSONL corpus.

Three sources are merged:

1. ``icd10cm_order_2026.txt`` - fixed-width, one row per code (98,186 rows).
   Columns: order number (0:5), code (6:13), billable flag (14:15),
   short title (16:76), long title (77:).
2. ``icd10cm_tabular_2026.xml`` - the hierarchy. Gives every code its chapter
   and block, plus "inclusion terms", which are the official clinical synonyms
   (for example "Classical cholera" for A00.0). Those synonyms are what make
   lexical search work on wording a clinician would actually use.
3. The same XML supplies parent descriptions, used to give terse leaf codes
   their missing context.

The XML only describes ~46k nodes while the order file has 98k codes, because
codes that are expanded by a 7th character (``S52.501A``) are not written out
individually. Those codes therefore inherit chapter, block and synonyms from
their nearest ancestor, which is exactly how the printed code book works.
"""

from __future__ import annotations

import gzip
import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from icd10_search.config import CORPUS_PATH, ICD10_RELEASE_YEAR

MAX_INHERITED_TERMS = 8


@dataclass(slots=True)
class CodeRecord:
    """One ICD-10-CM diagnosis code and everything known about it."""

    code: str
    code_dotted: str
    billable: bool
    short_desc: str
    long_desc: str
    chapter: int
    chapter_title: str
    block: str
    block_title: str
    parent_code: str | None = None
    parent_desc: str | None = None
    inclusion_terms: Sequence[str] = ()
    inherited_terms: Sequence[str] = ()

    def to_json(self) -> str:
        payload = asdict(self)
        payload["inclusion_terms"] = list(self.inclusion_terms)
        payload["inherited_terms"] = list(self.inherited_terms)
        return json.dumps(payload, ensure_ascii=False)


def dot_code(code: str) -> str:
    """``A000`` -> ``A00.0``. Codes of three characters are unchanged."""
    return code if len(code) <= 3 else f"{code[:3]}.{code[3:]}"


def undot_code(code: str) -> str:
    """``A00.0`` -> ``A000``; also upper-cases and strips whitespace."""
    return code.replace(".", "").replace(" ", "").upper()


def parse_order_file(path: Path) -> list[tuple[str, bool, str, str]]:
    """Return ``(code, billable, short_desc, long_desc)`` for every row."""
    rows: list[tuple[str, bool, str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if len(line) < 78:
                raise ValueError(f"{path.name}:{line_number}: row shorter than 78 chars")
            code = line[6:13].strip()
            flag = line[14:15]
            if flag not in ("0", "1"):
                raise ValueError(f"{path.name}:{line_number}: bad billable flag {flag!r}")
            rows.append((code, flag == "1", line[16:76].strip(), line[77:].strip()))
    return rows


@dataclass(slots=True)
class TabularFacts:
    """Lookups derived from the XML tabular list."""

    category_to_block: dict[str, str]
    block_titles: dict[str, str]
    block_to_chapter: dict[str, tuple[int, str]]
    descriptions: dict[str, str]
    inclusion_terms: dict[str, list[str]]


def _clean(text: str | None) -> str:
    return " ".join((text or "").split())


def _strip_trailing_range(title: str) -> str:
    """``Intestinal infectious diseases (A00-A09)`` -> ``Intestinal infectious diseases``."""
    if title.endswith(")") and "(" in title:
        head, _, tail = title.rpartition("(")
        if "-" in tail or tail.rstrip(")").isalnum():
            return head.strip()
    return title


def parse_tabular_xml(path: Path) -> TabularFacts:
    facts = TabularFacts({}, {}, {}, {}, {})
    tree = ET.parse(path)  # noqa: S314 - local file from a known government archive

    for chapter_el in tree.getroot().findall("chapter"):
        chapter_raw = _clean(chapter_el.findtext("name"))
        try:
            chapter_number = int(chapter_raw)
        except ValueError:
            continue
        chapter_title = _strip_trailing_range(_clean(chapter_el.findtext("desc")))

        for section_el in chapter_el.findall("section"):
            block_id = section_el.get("id", "").strip()
            if not block_id:
                continue
            facts.block_titles[block_id] = _strip_trailing_range(
                _clean(section_el.findtext("desc"))
            )
            facts.block_to_chapter[block_id] = (chapter_number, chapter_title)

            for diag_el in section_el.findall("diag"):
                category = undot_code(_clean(diag_el.findtext("name")))[:3]
                if category:
                    facts.category_to_block[category] = block_id
                _walk_diag(diag_el, facts)

    return facts


CODE_REFERENCE_RE = re.compile(r"\([A-Z][0-9]{2}|\b[A-Z][0-9]{2}[0-9A-Z]*\.?-")


def _synonym_notes(parent_el: ET.Element, tag: str) -> list[str]:
    notes: list[str] = []
    for container in parent_el.findall(tag):
        for note in container.findall("note"):
            text = _clean(note.text)
            if text and not CODE_REFERENCE_RE.search(text):
                notes.append(text)
    return notes


def _walk_diag(diag_el: ET.Element, facts: TabularFacts) -> None:
    """Record description and synonyms for a node, then recurse.

    Synonyms live in two different elements and both are needed:

    * ``<inclusionTerm>`` - the usual case, ~6.8k codes.
    * ``<includes>`` - used on a handful of high-traffic category codes. I10,
      the single most-used diagnosis code in US healthcare, keeps "high blood
      pressure" here and nowhere else, so skipping this element makes the most
      common lay phrasing in the whole code set unsearchable.
    """
    code = undot_code(_clean(diag_el.findtext("name")))
    if code:
        facts.descriptions[code] = _clean(diag_el.findtext("desc"))
        terms = [
            *_synonym_notes(diag_el, "inclusionTerm"),
            *_synonym_notes(diag_el, "includes"),
        ]
        deduped = list(dict.fromkeys(terms))
        if deduped:
            facts.inclusion_terms[code] = deduped
    for child in diag_el.findall("diag"):
        _walk_diag(child, facts)


def _ancestors(code: str) -> list[str]:
    """Candidate ancestor codes, longest (nearest) first: ``A0100`` -> A010, A01, A0."""
    return [code[:length] for length in range(len(code) - 1, 2, -1)]


def build_corpus(
    order_path: Path,
    tabular_path: Path,
) -> list[CodeRecord]:
    """Merge the order file with the XML hierarchy into one record per code."""
    rows = parse_order_file(order_path)
    facts = parse_tabular_xml(tabular_path)
    known_codes = {code for code, *_ in rows}
    descriptions = dict(facts.descriptions)
    for code, _billable, _short, long_desc in rows:
        descriptions.setdefault(code, long_desc)

    records: list[CodeRecord] = []
    unmapped_categories: set[str] = set()

    for code, billable, short_desc, long_desc in rows:
        category = code[:3]
        block = facts.category_to_block.get(category, "")
        if not block:
            unmapped_categories.add(category)
        chapter_number, chapter_title = facts.block_to_chapter.get(block, (0, "Unclassified"))

        parent_code = next((a for a in _ancestors(code) if a in known_codes), None)
        parent_desc = descriptions.get(parent_code) if parent_code else None

        inherited: list[str] = []
        for ancestor in _ancestors(code):
            for term in facts.inclusion_terms.get(ancestor, []):
                if term not in inherited:
                    inherited.append(term)
            if len(inherited) >= MAX_INHERITED_TERMS:
                break

        records.append(
            CodeRecord(
                code=code,
                code_dotted=dot_code(code),
                billable=billable,
                short_desc=short_desc,
                long_desc=long_desc,
                chapter=chapter_number,
                chapter_title=chapter_title,
                block=block,
                block_title=facts.block_titles.get(block, ""),
                parent_code=parent_code,
                parent_desc=parent_desc,
                inclusion_terms=tuple(facts.inclusion_terms.get(code, ())),
                inherited_terms=tuple(inherited[:MAX_INHERITED_TERMS]),
            )
        )

    if unmapped_categories:
        print(
            f"[parse] warning: {len(unmapped_categories)} categories had no XML block, "
            f"e.g. {sorted(unmapped_categories)[:5]}"
        )
    return records


def write_corpus(records: list[CodeRecord], path: Path = CORPUS_PATH) -> Path:
    """Write the corpus as gzipped JSONL (one record per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.to_json() + "\n")
    return path


_POOLED_FIELDS = ("chapter_title", "block", "block_title", "parent_code", "parent_desc")


def load_corpus(path: Path = CORPUS_PATH) -> list[CodeRecord]:
    """Read the corpus back in index order, pooling repeated strings."""
    if not path.exists():
        raise FileNotFoundError(
            f"corpus not found at {path} - run `make data` (or `python -m icd10_search.cli data`)"
        )

    pool: dict[str, str] = {}

    def shared(value: str | None) -> str | None:
        if value is None:
            return None
        return pool.setdefault(value, value)

    records: list[CodeRecord] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            for field_name in _POOLED_FIELDS:
                row[field_name] = shared(row[field_name])
            row["inclusion_terms"] = tuple(shared(term) for term in row["inclusion_terms"])
            row["inherited_terms"] = tuple(shared(term) for term in row["inherited_terms"])
            records.append(CodeRecord(**row))
    return records


def corpus_stats(records: list[CodeRecord]) -> dict[str, object]:
    return {
        "release_year": ICD10_RELEASE_YEAR,
        "total_codes": len(records),
        "billable_codes": sum(1 for r in records if r.billable),
        "chapters": len({r.chapter for r in records if r.chapter}),
        "blocks": len({r.block for r in records if r.block}),
        "codes_with_synonyms": sum(1 for r in records if r.inclusion_terms or r.inherited_terms),
    }
