"""Build the evaluation gold sets.

Two independent sets, because they test different things:

``gold_index_terms`` (derived, large)
    The CMS release ships the *Alphabetic Index to Diseases and Injuries* - the
    back-of-the-book index a human coder looks a term up in. Each entry is a
    path of terms ending in a code, e.g. ``Abdomen, abdominal > acute -> R10.0``.
    Reversing that path gives "acute abdomen", which is real medical phrasing
    that is *not* the code's official title. So this set measures: given the
    words a coder would look up, does the engine return the code the official
    index says is correct? The labels come from CMS, not from me.

``gold_lay_queries`` (hand-written, small)
    Plain-English phrases a non-coder would type ("chest pain when walking").
    I wrote these myself, which is a real limitation and is stated in the
    README: they are useful as a smoke test of lay phrasing, not as a
    statistically meaningful benchmark.

Entries whose target code is not a real code in the corpus are dropped, as are
cross-reference-only entries ("see Maltreatment"), which carry no code.
"""

from __future__ import annotations

import json
import random
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

from icd10_search.config import GOLD_INDEX_DEV_PATH, GOLD_INDEX_TEST_PATH, GOLD_LAY_PATH
from icd10_search.data.parse import undot_code

DEFAULT_SAMPLE_SIZE = 1000
MAX_PATH_DEPTH = 4
RANDOM_SEED = 20261123


@dataclass(slots=True)
class GoldQuery:
    query: str
    expected_code: str
    expected_code_dotted: str
    source: str
    path: list[str] | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def _clean_title(element: ET.Element) -> str:
    """Flatten a <title> node, dropping the parenthetical <nemod> qualifiers.

    ``<title>Abasia<nemod>(-astasia) (hysterical)</nemod></title>`` -> ``Abasia``.
    Non-essential modifiers are optional words a coder may ignore, so keeping
    them would make the query text unnaturally verbose.
    """
    text = element.text or ""
    for child in element:
        if child.tag != "nemod":
            text += child.text or ""
        text += child.tail or ""
    return " ".join(text.split()).strip(" ,-")


def _phrase_from_path(path: list[str]) -> str:
    """Turn an index path into a phrase close to how the term is spoken.

    The index is written head-term-first and then narrows:
    ``Deficiency > anti-hemophilic > factor > B``. Spoken English puts the
    qualifiers first and the head noun last ("anti-hemophilic factor B
    deficiency"), so the sub-terms are kept in their original order and the
    head term is moved to the end. The head term may itself be a
    comma-separated list of variants ("Abdomen, abdominal"); only the first
    variant is kept.
    """
    cleaned = [segment.split(",")[0].strip() for segment in path if segment.strip()]
    if not cleaned:
        return ""
    head, *subterms = cleaned
    return " ".join([*subterms, head]).strip()


def _walk_terms(
    element: ET.Element,
    path: list[str],
    out: list[tuple[list[str], str]],
) -> None:
    title_el = element.find("title")
    title = _clean_title(title_el) if title_el is not None else ""
    current = [*path, title] if title else list(path)

    code_el = element.find("code")
    if code_el is not None and (code_el.text or "").strip():
        code = (code_el.text or "").strip()
        code = code.split("[")[0].strip()
        if code and "-" not in code:
            out.append((current, code))

    for child in element.findall("term"):
        _walk_terms(child, current, out)


def extract_index_entries(index_xml: Path) -> list[tuple[list[str], str]]:
    """Return every ``(term path, code)`` pair in the Alphabetic Index."""
    tree = ET.parse(index_xml)  # noqa: S314 - local file from a known government archive
    entries: list[tuple[list[str], str]] = []
    for letter in tree.getroot().findall("letter"):
        for main_term in letter.findall("mainTerm"):
            _walk_terms(main_term, [], entries)
    return entries


def build_index_gold_set(
    index_xml: Path,
    valid_codes: set[str],
    *,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = RANDOM_SEED,
    min_words: int = 2,
    max_depth: int = MAX_PATH_DEPTH,
) -> list[GoldQuery]:
    """Sample a reproducible gold set from the CMS Alphabetic Index."""
    by_phrase: dict[str, dict[str, GoldQuery]] = {}

    for path, dotted in extract_index_entries(index_xml):
        code = undot_code(dotted)
        if code not in valid_codes:
            continue
        if len(path) > max_depth:
            continue
        phrase = _phrase_from_path(path)
        if len(phrase.split()) < min_words or len(phrase) > 120:
            continue
        by_phrase.setdefault(phrase.lower(), {}).setdefault(
            code,
            GoldQuery(
                query=phrase,
                expected_code=code,
                expected_code_dotted=dotted,
                source="cms_alphabetic_index",
                path=path,
            ),
        )

    ambiguous = sum(1 for codes in by_phrase.values() if len(codes) > 1)
    if ambiguous:
        print(f"[goldset] dropped {ambiguous:,} ambiguous phrases (multiple valid codes)")
    pool = sorted(
        (next(iter(codes.values())) for codes in by_phrase.values() if len(codes) == 1),
        key=lambda q: (q.query, q.expected_code),
    )
    rng = random.Random(seed)
    if sample_size and sample_size < len(pool):
        pool = rng.sample(pool, sample_size)
    return sorted(pool, key=lambda q: (q.query, q.expected_code))


def write_gold_set(queries: list[GoldQuery], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for query in queries:
            handle.write(query.to_json() + "\n")
    return path


def load_gold_set(path: Path) -> list[GoldQuery]:
    if not path.exists():
        raise FileNotFoundError(f"gold set not found at {path} - run `make data`")
    with path.open(encoding="utf-8") as handle:
        return [GoldQuery(**json.loads(line)) for line in handle if line.strip()]


def split_gold_set(
    pool: list[GoldQuery],
    *,
    dev_size: int,
    test_size: int,
    seed: int = RANDOM_SEED,
) -> tuple[list[GoldQuery], list[GoldQuery]]:
    """Draw a disjoint dev and test split from the pool, reproducibly.

    Tuning on the same queries that are reported on would turn the headline
    numbers into training accuracy. The dev split exists so that the fusion
    weights can be chosen without ever touching the test split.
    """
    needed = dev_size + test_size
    if needed > len(pool):
        raise ValueError(f"pool has {len(pool)} queries, need {needed}")
    drawn = random.Random(seed).sample(pool, needed)
    dev = sorted(drawn[:dev_size], key=lambda q: (q.query, q.expected_code))
    test = sorted(drawn[dev_size:], key=lambda q: (q.query, q.expected_code))
    return dev, test


def load_all_gold_sets() -> dict[str, list[GoldQuery]]:
    """Load every gold set. Only `cms_index_test` and `hand_written_lay` are reported."""
    return {
        "cms_index_dev": load_gold_set(GOLD_INDEX_DEV_PATH),
        "cms_index_test": load_gold_set(GOLD_INDEX_TEST_PATH),
        "hand_written_lay": load_gold_set(GOLD_LAY_PATH),
    }


LAY_SEED_PATH = Path(__file__).resolve().parents[3] / "data" / "seed" / "lay_queries.jsonl"


def build_lay_gold_set(
    valid_codes: set[str],
    seed_path: Path = LAY_SEED_PATH,
) -> list[GoldQuery]:
    """Load the hand-written lay queries, failing loudly on any unknown code.

    Validating against the real corpus is what stops this set from quietly
    rotting when CMS retires or re-numbers a code in a later release.
    """
    if not seed_path.exists():
        raise FileNotFoundError(f"lay query seed file missing: {seed_path}")

    queries: list[GoldQuery] = []
    unknown: list[str] = []
    with seed_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            code = undot_code(row["expected_code_dotted"])
            if code not in valid_codes:
                unknown.append(row["expected_code_dotted"])
                continue
            queries.append(
                GoldQuery(
                    query=row["query"],
                    expected_code=code,
                    expected_code_dotted=row["expected_code_dotted"],
                    source="hand_written",
                )
            )
    if unknown:
        raise ValueError(
            f"lay gold set references {len(unknown)} codes absent from the corpus: {unknown}"
        )
    return queries
