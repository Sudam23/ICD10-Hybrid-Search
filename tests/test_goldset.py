"""Gold-set construction - the labels the whole evaluation rests on."""

from __future__ import annotations

import json
import textwrap
import xml.etree.ElementTree as ET

import pytest

from icd10_search.data.goldset import (
    _clean_title,
    _phrase_from_path,
    build_index_gold_set,
    build_lay_gold_set,
    split_gold_set,
)

INDEX_FIXTURE = textwrap.dedent(
    """\
    <?xml version="1.0" encoding="utf-8"?>
    <ICD10CM.index>
      <letter>
        <title>A</title>
        <mainTerm><title>Aarskog's syndrome</title><code>Q87.19</code></mainTerm>
        <mainTerm><title>Abandonment</title><see>Maltreatment</see></mainTerm>
        <mainTerm>
          <title>Abdomen, abdominal</title>
          <term level="1"><title>acute</title><code>R10.0</code></term>
          <term level="1"><title>angina</title><code>K55.1</code></term>
        </mainTerm>
        <mainTerm>
          <title>Ambiguous</title>
          <term level="1"><title>thing</title><code>R10.0</code></term>
        </mainTerm>
        <mainTerm><title>Unknown disease</title><code>Z99.9</code></mainTerm>
      </letter>
    </ICD10CM.index>
    """
)

VALID = {"Q8719", "R100", "K551"}


@pytest.fixture()
def index_xml(tmp_path):
    path = tmp_path / "index.xml"
    path.write_text(INDEX_FIXTURE, encoding="utf-8")
    return path


def test_nemod_qualifiers_are_dropped():
    element = ET.fromstring("<title>Abasia<nemod>(-astasia) (hysterical)</nemod></title>")
    assert _clean_title(element) == "Abasia"


def test_phrase_puts_the_head_term_last():
    assert _phrase_from_path(["Abdomen, abdominal", "acute"]) == "acute Abdomen"
    assert (
        _phrase_from_path(["Deficiency, deficient", "anti-hemophilic", "factor", "B"])
        == "anti-hemophilic factor B Deficiency"
    )


def test_cross_reference_entries_carry_no_code(index_xml):
    queries = build_index_gold_set(index_xml, VALID, sample_size=0, min_words=1)
    assert all("Abandonment" not in query.query for query in queries)


def test_codes_outside_the_corpus_are_dropped(index_xml):
    queries = build_index_gold_set(index_xml, VALID, sample_size=0, min_words=1)
    assert all(query.expected_code != "Z999" for query in queries)


def test_one_phrase_mapping_to_several_codes_is_kept_unambiguous(index_xml):
    queries = build_index_gold_set(index_xml, VALID, sample_size=0, min_words=1)
    phrases = [query.query for query in queries]
    assert len(phrases) == len(set(phrases))


def test_sampling_is_deterministic(index_xml):
    first = build_index_gold_set(index_xml, VALID, sample_size=2, min_words=1)
    second = build_index_gold_set(index_xml, VALID, sample_size=2, min_words=1)
    assert [q.query for q in first] == [q.query for q in second]


def test_split_is_disjoint_and_reproducible(index_xml):
    pool = build_index_gold_set(index_xml, VALID, sample_size=0, min_words=1)
    dev, test = split_gold_set(pool, dev_size=1, test_size=1)
    assert not {q.query for q in dev} & {q.query for q in test}
    again_dev, again_test = split_gold_set(pool, dev_size=1, test_size=1)
    assert [q.query for q in dev] == [q.query for q in again_dev]
    assert [q.query for q in test] == [q.query for q in again_test]


def test_split_refuses_to_overdraw(index_xml):
    pool = build_index_gold_set(index_xml, VALID, sample_size=0, min_words=1)
    with pytest.raises(ValueError, match="need"):
        split_gold_set(pool, dev_size=999, test_size=999)


def test_lay_gold_set_rejects_a_code_that_no_longer_exists(tmp_path):
    seed = tmp_path / "lay.jsonl"
    seed.write_text(
        json.dumps({"query": "made up", "expected_code_dotted": "X99.9"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="absent from the corpus"):
        build_lay_gold_set({"R100"}, seed_path=seed)


def test_lay_gold_set_loads_valid_rows(tmp_path):
    seed = tmp_path / "lay.jsonl"
    seed.write_text(
        json.dumps({"query": "tummy ache", "expected_code_dotted": "R10.0"}) + "\n",
        encoding="utf-8",
    )
    queries = build_lay_gold_set({"R100"}, seed_path=seed)
    assert queries[0].expected_code == "R100"
    assert queries[0].source == "hand_written"
