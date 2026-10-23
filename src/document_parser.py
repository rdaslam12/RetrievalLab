"""In-memory extraction for PDF, DOCX, TXT, and BEIR JSONL uploads."""

from __future__ import annotations

import hashlib
import json
import logging
from io import BytesIO
from pathlib import Path
from typing import Iterable

from .types import ParsedCorpus, SourceSection


LOGGER = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".jsonl"}


class DocumentValidationError(ValueError):
    """Raised for an unsupported, malformed, or over-limit corpus."""


def make_source_id(file_name: str, data: bytes, duplicate_ordinal: int = 0) -> str:
    """Build a stable identifier while distinguishing duplicate filenames."""

    digest = hashlib.sha256()
    digest.update(file_name.encode("utf-8", errors="replace"))
    digest.update(b"\0")
    digest.update(str(duplicate_ordinal).encode("ascii"))
    digest.update(b"\0")
    digest.update(data)
    return digest.hexdigest()[:20]


def _parse_pdf(file_name: str, data: bytes, source_id: str) -> tuple[list[SourceSection], list[str]]:
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover - dependency integration path
        raise DocumentValidationError("PDF support requires PyMuPDF.") from exc

    sections: list[SourceSection] = []
    warnings: list[str] = []
    try:
        with fitz.open(stream=data, filetype="pdf") as document:
            for zero_based_page, page in enumerate(document):
                page_number = zero_based_page + 1
                text = page.get_text("text") or ""
                if not text.strip():
                    warnings.append(
                        f"{file_name}: page {page_number} contains no extractable text; OCR is not available."
                    )
                    continue
                sections.append(SourceSection(source_id, file_name, "pdf", page_number, text))
    except Exception as exc:
        raise DocumentValidationError(f"Could not parse PDF {file_name}: {exc}") from exc
    return sections, warnings


def _parse_docx(file_name: str, data: bytes, source_id: str) -> tuple[list[SourceSection], list[str]]:
    try:
        from docx import Document
    except ImportError as exc:  # pragma: no cover - dependency integration path
        raise DocumentValidationError("DOCX support requires python-docx.") from exc
    try:
        document = Document(BytesIO(data))
        text = "\n".join(paragraph.text for paragraph in document.paragraphs if paragraph.text.strip())
    except Exception as exc:
        raise DocumentValidationError(f"Could not parse DOCX {file_name}: {exc}") from exc
    sections = [SourceSection(source_id, file_name, "docx", None, text)] if text.strip() else []
    warnings = [] if sections else [f"{file_name}: document contains no extractable paragraph text."]
    return sections, warnings


def _parse_txt(file_name: str, data: bytes, source_id: str) -> tuple[list[SourceSection], list[str]]:
    warnings: list[str] = []
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="replace")
        warnings.append(f"{file_name}: invalid UTF-8 bytes were replaced during decoding.")
    sections = [SourceSection(source_id, file_name, "txt", None, text)] if text.strip() else []
    if not sections:
        warnings.append(f"{file_name}: text file is empty.")
    return sections, warnings


def _jsonl_record_source_id(container_source_id: str, document_id: str) -> str:
    digest = hashlib.sha256()
    digest.update(container_source_id.encode("ascii"))
    digest.update(b"\0")
    digest.update(document_id.encode("utf-8", errors="replace"))
    return digest.hexdigest()[:20]


def _parse_jsonl(
    file_name: str, data: bytes, source_id: str
) -> tuple[list[SourceSection], list[str]]:
    """Parse one BEIR-style corpus JSONL file into document-level sections."""

    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DocumentValidationError(f"{file_name}: JSONL must be valid UTF-8: {exc}") from exc

    sections: list[SourceSection] = []
    warnings: list[str] = []
    seen_document_ids: set[str] = set()
    record_count = 0
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        record_count += 1
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DocumentValidationError(
                f"{file_name}: line {line_number} is not valid JSON: {exc.msg}."
            ) from exc
        if not isinstance(record, dict):
            raise DocumentValidationError(
                f"{file_name}: line {line_number} must contain a JSON object."
            )

        raw_document_id = record.get("_id")
        if isinstance(raw_document_id, bool) or not isinstance(raw_document_id, (str, int)):
            raise DocumentValidationError(
                f"{file_name}: line {line_number} requires a non-empty string or integer '_id'."
            )
        document_id = str(raw_document_id).strip()
        if not document_id:
            raise DocumentValidationError(
                f"{file_name}: line {line_number} requires a non-empty '_id'."
            )
        if document_id in seen_document_ids:
            raise DocumentValidationError(
                f"{file_name}: duplicate document _id {document_id!r} on line {line_number}."
            )
        seen_document_ids.add(document_id)

        if "text" not in record:
            raise DocumentValidationError(
                f"{file_name}: line {line_number} is missing the required 'text' field."
            )
        body = record["text"]
        title = record.get("title", "")
        if not isinstance(body, str):
            raise DocumentValidationError(
                f"{file_name}: line {line_number} field 'text' must be a string."
            )
        if title is None:
            title = ""
        if not isinstance(title, str):
            raise DocumentValidationError(
                f"{file_name}: line {line_number} field 'title' must be a string when present."
            )

        clean_title = title.strip()
        clean_body = body.strip()
        searchable_text = "\n\n".join(part for part in (clean_title, clean_body) if part)
        if not searchable_text:
            warnings.append(
                f"{file_name}: document {document_id!r} on line {line_number} has no "
                "indexable title or text and was skipped."
            )
            continue
        sections.append(
            SourceSection(
                source_id=_jsonl_record_source_id(source_id, document_id),
                file_name=file_name,
                file_type="jsonl",
                page_number=None,
                text=searchable_text,
                document_id=document_id,
                title=clean_title or None,
            )
        )

    if record_count == 0:
        raise DocumentValidationError(f"{file_name}: JSONL file contains no JSON records.")
    return sections, warnings


def parse_document(
    file_name: str, data: bytes, *, duplicate_ordinal: int = 0
) -> tuple[list[SourceSection], list[str]]:
    """Parse one uploaded byte payload without writing it to disk."""

    safe_name = Path(file_name).name
    suffix = Path(safe_name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise DocumentValidationError(
            f"Unsupported file type for {safe_name}; use PDF, DOCX, TXT, or JSONL."
        )
    if not data:
        raise DocumentValidationError(f"{safe_name} is empty.")
    source_id = make_source_id(safe_name, data, duplicate_ordinal)
    parser = {
        ".pdf": _parse_pdf,
        ".docx": _parse_docx,
        ".txt": _parse_txt,
        ".jsonl": _parse_jsonl,
    }[suffix]
    LOGGER.info("Parsing upload name=%s type=%s bytes=%d", safe_name, suffix, len(data))
    return parser(safe_name, data, source_id)


def parse_corpus(
    files: Iterable[tuple[str, bytes]], *, max_files: int = 20, max_total_bytes: int = 50 * 1024 * 1024
) -> ParsedCorpus:
    """Validate and parse a complete corpus while keeping duplicate names distinct."""

    items = list(files)
    if not items:
        raise DocumentValidationError("Select at least one document.")
    if len(items) > max_files:
        raise DocumentValidationError(f"A corpus may contain at most {max_files} files.")
    total_bytes = sum(len(data) for _, data in items)
    if total_bytes > max_total_bytes:
        raise DocumentValidationError(
            f"Uploaded corpus is {total_bytes / (1024 * 1024):.1f} MB; the limit is "
            f"{max_total_bytes / (1024 * 1024):.0f} MB."
        )

    parsed = ParsedCorpus(total_bytes=total_bytes)
    occurrences: dict[str, int] = {}
    for file_name, data in items:
        safe_name = Path(file_name).name
        ordinal = occurrences.get(safe_name, 0)
        occurrences[safe_name] = ordinal + 1
        sections, warnings = parse_document(safe_name, data, duplicate_ordinal=ordinal)
        parsed.sections.extend(sections)
        parsed.warnings.extend(warnings)
        parsed.files_processed += 1
    if not parsed.sections:
        raise DocumentValidationError("No extractable text was found in the uploaded corpus.")
    return parsed
