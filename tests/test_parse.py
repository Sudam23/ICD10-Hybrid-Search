"""Parsing the CMS fixed-width order file and the XML tabular list."""

from __future__ import annotations

import textwrap

import pytest

from icd10_search.data.parse import (
    _ancestors,
    build_corpus,
    dot_code,
    parse_order_file,
    parse_tabular_xml,
    undot_code,
)

ORDER_FIXTURE = (
    "00001 A00     0 Cholera                                                      Cholera\n"
    "00002 A000    1 Cholera due to Vibrio cholerae 01                            "
    "Cholera due to Vibrio cholerae 01, biovar cholerae\n"
    "00003 I10     1 Essential (primary) hypertension                             "
    "Essential (primary) hypertension\n"
)

TABULAR_FIXTURE = textwrap.dedent(
    """\
    <?xml version="1.0" encoding="utf-8"?>
    <ICD10CM.tabular>
      <chapter>
        <name>1</name>
        <desc>Certain infectious and parasitic diseases (A00-B99)</desc>
        <section id="A00-A09">
          <desc>Intestinal infectious diseases (A00-A09)</desc>
          <diag>
            <name>A00</name><desc>Cholera</desc>
            <diag>
              <name>A00.0</name><desc>Cholera due to Vibrio cholerae 01</desc>
              <inclusionTerm><note>Classical cholera</note></inclusionTerm>
            </diag>
          </diag>
        </section>
      </chapter>
      <chapter>
        <name>9</name>
        <desc>Diseases of the circulatory system (I00-I99)</desc>
        <section id="I10-I1A">
          <desc>Hypertensive diseases (I10-I1A)</desc>
          <diag>
            <name>I10</name><desc>Essential (primary) hypertension</desc>
            <includes>
              <note>high blood pressure</note>
              <note>any condition in I50.- due to hypertension</note>
            </includes>
          </diag>
        </section>
      </chapter>
    </ICD10CM.tabular>
    """
)


@pytest.fixture()
def fixture_files(tmp_path):
    order = tmp_path / "order.txt"
    order.write_text(ORDER_FIXTURE, encoding="utf-8")
    tabular = tmp_path / "tabular.xml"
    tabular.write_text(TABULAR_FIXTURE, encoding="utf-8")
    return order, tabular


def test_dot_and_undot_round_trip():
    assert dot_code("A000") == "A00.0"
    assert dot_code("A00") == "A00"
    assert dot_code("S52501A") == "S52.501A"
    assert undot_code("S52.501A") == "S52501A"
    assert undot_code(" e11.9 ") == "E119"


def test_ancestors_are_longest_first():
    assert _ancestors("A0100") == ["A010", "A01"]
    assert _ancestors("A00") == []


def test_order_file_columns(fixture_files):
    rows = parse_order_file(fixture_files[0])
    assert len(rows) == 3
    code, billable, short_desc, long_desc = rows[1]
    assert code == "A000"
    assert billable is True
    assert short_desc == "Cholera due to Vibrio cholerae 01"
    assert long_desc == "Cholera due to Vibrio cholerae 01, biovar cholerae"
    assert rows[0][1] is False


def test_order_file_rejects_malformed_rows(tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_text("00001 A00     X Cholera" + " " * 60 + "Cholera\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bad billable flag"):
        parse_order_file(bad)


def test_tabular_xml_maps_categories_and_synonyms(fixture_files):
    facts = parse_tabular_xml(fixture_files[1])
    assert facts.category_to_block["A00"] == "A00-A09"
    assert facts.block_to_chapter["A00-A09"] == (1, "Certain infectious and parasitic diseases")
    assert facts.block_titles["A00-A09"] == "Intestinal infectious diseases"
    assert facts.inclusion_terms["A000"] == ["Classical cholera"]


def test_includes_notes_are_synonyms_but_code_references_are_not(fixture_files):
    """I10 keeps 'high blood pressure' in <includes>; the I50.- note is not a synonym."""
    facts = parse_tabular_xml(fixture_files[1])
    assert facts.inclusion_terms["I10"] == ["high blood pressure"]


def test_build_corpus_merges_sources(fixture_files):
    records = build_corpus(*fixture_files)
    by_code = {record.code: record for record in records}

    cholera = by_code["A000"]
    assert cholera.chapter == 1
    assert cholera.block == "A00-A09"
    assert cholera.parent_code == "A00"
    assert cholera.parent_desc == "Cholera"
    assert list(cholera.inclusion_terms) == ["Classical cholera"]

    hypertension = by_code["I10"]
    assert hypertension.chapter == 9
    assert "high blood pressure" in hypertension.inclusion_terms


def test_load_corpus_pools_repeated_strings(tmp_path, fixture_files):
    """Repeated values share one string object - the whole point of the pooling."""
    from icd10_search.data.parse import load_corpus, write_corpus

    records = build_corpus(*fixture_files)
    path = write_corpus(records, tmp_path / "corpus.jsonl.gz")
    reloaded = load_corpus(path)

    assert [r.code for r in reloaded] == [r.code for r in records]
    same_chapter = [r for r in reloaded if r.chapter == 1]
    assert len(same_chapter) > 1
    first, second = same_chapter[0], same_chapter[1]
    assert first.chapter_title is second.chapter_title
    assert first.block is second.block

    empty = [r for r in reloaded if not r.inherited_terms]
    assert len(empty) > 1
    assert empty[0].inherited_terms is empty[1].inherited_terms
