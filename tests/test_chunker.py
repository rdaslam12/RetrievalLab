from src.chunker import chunk_sections
from src.types import SourceSection

from .helpers import WhitespaceTokenizer


def test_token_chunk_overlap_works_and_terminates() -> None:
    tokenizer = WhitespaceTokenizer()
    section = SourceSection("source", "doc.txt", "txt", None, "zero one two three four five six")
    passages = chunk_sections([section], tokenizer, chunk_size=4, overlap=2)

    assert [passage.text for passage in passages] == [
        "zero one two three",
        "two three four five",
        "four five six",
    ]
    assert all(passage.token_count <= 4 for passage in passages)
    assert len({passage.chunk_id for passage in passages}) == 3


def test_pdf_pages_are_chunked_independently() -> None:
    tokenizer = WhitespaceTokenizer()
    sections = [
        SourceSection("source", "doc.pdf", "pdf", 1, "a b c d e"),
        SourceSection("source", "doc.pdf", "pdf", 2, "f g h"),
    ]
    passages = chunk_sections(sections, tokenizer, chunk_size=4, overlap=1)
    assert [passage.page_number for passage in passages] == [1, 1, 2]
    assert [passage.passage_index for passage in passages] == [1, 2, 3]


def test_invalid_overlap_is_rejected() -> None:
    tokenizer = WhitespaceTokenizer()
    section = SourceSection("source", "doc.txt", "txt", None, "a b")
    try:
        chunk_sections([section], tokenizer, chunk_size=2, overlap=2)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("Expected invalid overlap to fail")


def test_jsonl_document_identity_is_preserved_in_every_chunk() -> None:
    tokenizer = WhitespaceTokenizer()
    section = SourceSection(
        "source",
        "corpus.jsonl",
        "jsonl",
        None,
        "one two three four five",
        document_id="beir-doc-7",
        title="Example title",
    )

    passages = chunk_sections([section], tokenizer, chunk_size=3, overlap=1)

    assert len(passages) == 2
    assert all(passage.document_id == "beir-doc-7" for passage in passages)
    assert all(passage.title == "Example title" for passage in passages)
    assert all(passage.page_number is None for passage in passages)


def test_passage_limit_remains_strict_by_default() -> None:
    tokenizer = WhitespaceTokenizer()
    sections = [
        SourceSection("one", "corpus.jsonl", "jsonl", None, "a b"),
        SourceSection("two", "corpus.jsonl", "jsonl", None, "c d"),
    ]

    try:
        chunk_sections(sections, tokenizer, chunk_size=2, overlap=0, max_passages=1)
    except ValueError as exc:
        assert "more than 1 passages" in str(exc)
        assert "bounded JSONL slice mode" in str(exc)
    else:
        raise AssertionError("Expected the strict passage limit to fail")


def test_bounded_mode_stops_before_splitting_a_document() -> None:
    tokenizer = WhitespaceTokenizer()
    sections = [
        SourceSection(
            "one",
            "corpus.jsonl",
            "jsonl",
            None,
            "a b",
            document_id="doc-1",
        ),
        SourceSection(
            "two",
            "corpus.jsonl",
            "jsonl",
            None,
            "c d e f g",
            document_id="doc-2",
        ),
        SourceSection(
            "three",
            "corpus.jsonl",
            "jsonl",
            None,
            "h i",
            document_id="doc-3",
        ),
    ]

    passages = chunk_sections(
        sections,
        tokenizer,
        chunk_size=2,
        overlap=0,
        max_passages=3,
        truncate_at_limit=True,
    )

    assert len(passages) == 1
    assert [passage.document_id for passage in passages] == ["doc-1"]
