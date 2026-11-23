"""Document-text variants - what actually gets embedded."""

from __future__ import annotations

import pytest

from icd10_search.data.parse import CodeRecord
from icd10_search.data.text import DOC_TEXT_VARIANTS, document_text, lexical_text


@pytest.fixture()
def record() -> CodeRecord:
    return CodeRecord(
        code="A0100",
        code_dotted="A01.00",
        billable=True,
        short_desc="Typhoid fever, unspecified",
        long_desc="Typhoid fever, unspecified",
        chapter=1,
        chapter_title="Certain infectious and parasitic diseases",
        block="A00-A09",
        block_title="Intestinal infectious diseases",
        parent_code="A010",
        parent_desc="Typhoid fever",
        inclusion_terms=["Infection due to Salmonella typhi"],
        inherited_terms=["Enteric fever"],
    )


def test_desc_variant_is_the_title_alone(record):
    assert document_text(record, "desc") == "Typhoid fever, unspecified."


def test_synonyms_are_added_once_and_only_once(record):
    text = document_text(record, "desc_incl")
    assert "Infection due to Salmonella typhi" in text
    assert "Enteric fever" in text
    assert text.count("Also called") == 1


def test_parent_is_skipped_when_it_repeats_the_title(record):
    """'Typhoid fever' is already inside 'Typhoid fever, unspecified'."""
    assert "Part of:" not in document_text(record, "desc_incl_parent")


def test_parent_is_added_when_it_adds_information(record):
    record.long_desc = "Unspecified"
    assert "Part of: Typhoid fever." in document_text(record, "desc_incl_parent")


def test_block_variant_adds_the_category(record):
    assert "Intestinal infectious diseases" in document_text(record, "desc_incl_parent_block")


def test_unknown_variant_is_rejected(record):
    with pytest.raises(ValueError, match="unknown document text variant"):
        document_text(record, "nonsense")


def test_every_declared_variant_builds(record):
    for variant in DOC_TEXT_VARIANTS:
        assert document_text(record, variant)


def test_lexical_text_carries_the_vocabulary_without_labels(record):
    text = lexical_text(record)
    assert "Salmonella" in text
    assert "Also called" not in text
