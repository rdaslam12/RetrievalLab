"""Offline parsing, validation, and document-level retrieval metrics.

Metric aggregation is intentionally left to the evaluation orchestrator. Each
metric here evaluates one query, treats qrel scores greater than zero as
relevant, deduplicates retrieved document IDs before applying the cutoff, and
returns ``0.0`` when the query has no relevant documents.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from io import StringIO

from .types import Passage, RankedPassage


EVALUATION_K_VALUES = (1, 3, 5, 10)
Qrels = dict[str, dict[str, float]]


class EvaluationDataError(ValueError):
    """Raised when evaluation uploads are malformed or cannot be resolved."""


@dataclass(frozen=True, slots=True)
class EvaluationQuery:
    query_id: str
    text: str


def _decode_utf8(data: bytes, file_name: str, kind: str) -> str:
    if not data:
        raise EvaluationDataError(f"{file_name}: {kind} file is empty.")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise EvaluationDataError(f"{file_name}: {kind} file must be valid UTF-8.") from exc


def _identifier(value: object, *, field: str, file_name: str, line_number: int) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise EvaluationDataError(
            f"{file_name}: line {line_number} requires a non-empty string or integer "
            f"{field!r}."
        )
    identifier = str(value).strip()
    if not identifier:
        raise EvaluationDataError(
            f"{file_name}: line {line_number} requires a non-empty {field!r}."
        )
    return identifier


def parse_query_jsonl(data: bytes, *, file_name: str = "queries.jsonl") -> tuple[EvaluationQuery, ...]:
    """Parse strict JSONL records containing unique ``_id`` and ``text`` fields."""

    text = _decode_utf8(data, file_name, "query JSONL")
    queries: list[EvaluationQuery] = []
    seen_ids: set[str] = set()
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvaluationDataError(
                f"{file_name}: line {line_number} is not valid JSON: {exc.msg}."
            ) from exc
        if not isinstance(record, dict):
            raise EvaluationDataError(
                f"{file_name}: line {line_number} must contain a JSON object."
            )
        query_id = _identifier(
            record.get("_id"), field="_id", file_name=file_name, line_number=line_number
        )
        if query_id in seen_ids:
            raise EvaluationDataError(
                f"{file_name}: duplicate query _id {query_id!r} on line {line_number}."
            )
        if "text" not in record or not isinstance(record["text"], str):
            raise EvaluationDataError(
                f"{file_name}: line {line_number} field 'text' must be a non-empty string."
            )
        query_text = record["text"].strip()
        if not query_text:
            raise EvaluationDataError(
                f"{file_name}: line {line_number} field 'text' must be a non-empty string."
            )
        seen_ids.add(query_id)
        queries.append(EvaluationQuery(query_id=query_id, text=query_text))
    if not queries:
        raise EvaluationDataError(f"{file_name}: query JSONL contains no query records.")
    return tuple(queries)


def _normalized_header(value: str) -> str:
    return value.strip().casefold().replace("_", "-").replace(" ", "-")


def _is_qrels_header(fields: Sequence[str]) -> bool:
    if len(fields) != 3:
        return False
    query_field, document_field, score_field = map(_normalized_header, fields)
    return (
        query_field in {"query-id", "queryid", "qid"}
        and document_field in {"corpus-id", "document-id", "doc-id", "docid"}
        and score_field in {"score", "relevance", "rel"}
    )


def parse_qrels_tsv(data: bytes, *, file_name: str = "qrels.tsv") -> Qrels:
    """Parse three-column TSV qrels, accepting one conventional header row."""

    text = _decode_utf8(data, file_name, "qrels TSV")
    qrels: Qrels = {}
    saw_record = False
    first_nonempty = True
    for line_number, row in enumerate(csv.reader(StringIO(text), delimiter="\t"), start=1):
        if not row or all(not field.strip() for field in row):
            continue
        if first_nonempty and _is_qrels_header(row):
            first_nonempty = False
            continue
        first_nonempty = False
        if len(row) != 3:
            raise EvaluationDataError(
                f"{file_name}: line {line_number} must have exactly three tab-separated fields: "
                "query-id, corpus-id, score."
            )
        query_id = _identifier(
            row[0], field="query-id", file_name=file_name, line_number=line_number
        )
        document_id = _identifier(
            row[1], field="corpus-id", file_name=file_name, line_number=line_number
        )
        try:
            score = float(row[2].strip())
        except ValueError as exc:
            raise EvaluationDataError(
                f"{file_name}: line {line_number} score must be numeric."
            ) from exc
        if not math.isfinite(score) or score < 0.0:
            raise EvaluationDataError(
                f"{file_name}: line {line_number} score must be finite and non-negative."
            )
        query_qrels = qrels.setdefault(query_id, {})
        if document_id in query_qrels:
            raise EvaluationDataError(
                f"{file_name}: duplicate judgment for query {query_id!r} and document "
                f"{document_id!r} on line {line_number}."
            )
        query_qrels[document_id] = score
        saw_record = True
    if not saw_record:
        raise EvaluationDataError(f"{file_name}: qrels TSV contains no relevance judgments.")
    return qrels


def passage_document_id(passage: Passage) -> str:
    """Return the corpus evaluation identity for a passage's source document."""

    return passage.document_id or passage.source_id


def corpus_document_ids(passages: Sequence[Passage]) -> frozenset[str]:
    return frozenset(passage_document_id(passage) for passage in passages)


def validate_evaluation_data(
    queries: Sequence[EvaluationQuery],
    qrels: Mapping[str, Mapping[str, float]],
    passages: Sequence[Passage],
) -> tuple[str, ...]:
    """Resolve every qrel ID before retrieval and return non-fatal dataset warnings."""

    query_ids = {query.query_id for query in queries}
    unknown_query_ids = sorted(set(qrels) - query_ids)
    active_document_ids = corpus_document_ids(passages)
    unknown_document_ids = sorted(
        {
            document_id
            for judgments in qrels.values()
            for document_id in judgments
            if document_id not in active_document_ids
        }
    )
    errors: list[str] = []
    if unknown_query_ids:
        preview = ", ".join(repr(item) for item in unknown_query_ids[:10])
        suffix = "…" if len(unknown_query_ids) > 10 else ""
        errors.append(f"unknown query IDs in qrels: {preview}{suffix}")
    if unknown_document_ids:
        preview = ", ".join(repr(item) for item in unknown_document_ids[:10])
        suffix = "…" if len(unknown_document_ids) > 10 else ""
        errors.append(f"unknown corpus/document IDs in qrels: {preview}{suffix}")
    if errors:
        raise EvaluationDataError("Evaluation data could not be resolved: " + "; ".join(errors) + ".")

    unjudged = [query.query_id for query in queries if not qrels.get(query.query_id)]
    warnings: list[str] = []
    if unjudged:
        warnings.append(
            f"{len(unjudged)} of {len(queries)} queries have no relevance judgments; "
            "their per-query metrics are 0 and they remain in the aggregate mean."
        )
    no_positive = [
        query.query_id
        for query in queries
        if qrels.get(query.query_id)
        and not any(score > 0.0 for score in qrels[query.query_id].values())
    ]
    if no_positive:
        warnings.append(
            f"{len(no_positive)} judged queries have no positive relevance score; "
            "their Recall, Hit Rate, MRR, and NDCG values are 0."
        )
    return tuple(warnings)


def deduplicate_document_ids(document_ids: Sequence[str]) -> tuple[str, ...]:
    """Keep the first (highest-ranked) occurrence of each document identity."""

    seen: set[str] = set()
    unique: list[str] = []
    for document_id in document_ids:
        if document_id not in seen:
            seen.add(document_id)
            unique.append(document_id)
    return tuple(unique)


def deduplicate_ranked_documents(results: Sequence[RankedPassage]) -> tuple[str, ...]:
    return deduplicate_document_ids([passage_document_id(result.passage) for result in results])


def _relevant_documents(relevance: Mapping[str, float]) -> set[str]:
    return {document_id for document_id, score in relevance.items() if score > 0.0}


def _top_k_documents(retrieved: Sequence[str], k: int) -> tuple[str, ...]:
    if k <= 0:
        raise ValueError("Metric cutoff K must be positive.")
    return deduplicate_document_ids(retrieved)[:k]


def recall_at_k(retrieved: Sequence[str], relevance: Mapping[str, float], k: int) -> float:
    """Relevant documents in the deduplicated top K divided by all relevant documents."""

    top_k = _top_k_documents(retrieved, k)
    relevant = _relevant_documents(relevance)
    if not relevant:
        return 0.0
    return len(relevant.intersection(top_k)) / len(relevant)


def hit_rate_at_k(retrieved: Sequence[str], relevance: Mapping[str, float], k: int) -> float:
    """Return 1 when the deduplicated top K contains a relevant document, otherwise 0."""

    relevant = _relevant_documents(relevance)
    top_k = _top_k_documents(retrieved, k)
    return float(bool(relevant.intersection(top_k)))


def mrr_at_k(retrieved: Sequence[str], relevance: Mapping[str, float], k: int) -> float:
    """Reciprocal rank of the first relevant document in the deduplicated top K."""

    relevant = _relevant_documents(relevance)
    for rank, document_id in enumerate(_top_k_documents(retrieved, k), start=1):
        if document_id in relevant:
            return 1.0 / rank
    return 0.0


def _discounted_gain(grades: Sequence[float]) -> float:
    return sum((2.0**grade - 1.0) / math.log2(rank + 1) for rank, grade in enumerate(grades, 1))


def ndcg_at_k(retrieved: Sequence[str], relevance: Mapping[str, float], k: int) -> float:
    """Graded NDCG@K using exponential gain and log2 rank discount."""

    ranked = _top_k_documents(retrieved, k)
    gains = [max(0.0, relevance.get(document_id, 0.0)) for document_id in ranked]
    ideal = sorted((score for score in relevance.values() if score > 0.0), reverse=True)[:k]
    ideal_dcg = _discounted_gain(ideal)
    if ideal_dcg == 0.0:
        return 0.0
    return _discounted_gain(gains) / ideal_dcg


def metrics_at_k(
    retrieved: Sequence[str], relevance: Mapping[str, float], k: int
) -> dict[str, float]:
    """Compute all supported metrics for one query at one cutoff."""

    return {
        "Recall": recall_at_k(retrieved, relevance, k),
        "Hit Rate": hit_rate_at_k(retrieved, relevance, k),
        "MRR": mrr_at_k(retrieved, relevance, k),
        "NDCG": ndcg_at_k(retrieved, relevance, k),
    }
