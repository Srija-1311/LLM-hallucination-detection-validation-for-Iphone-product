import re
from pathlib import Path
from typing import Optional

STOP_HEADERS = ("dataset note", "synthetic data")
SECTION_RE = re.compile(r"(?m)^(\d+)\.\s+([A-Z][^\n]{2,80})\s*$")
HEADER_FIELDS = [
    "Document ID",
    "Product",
    "Domain",
    "Department",
    "Document Type",
    "Category",
    "Revision",
    "Status",
]


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_header_metadata(text: str, path: Path, bucket: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for field in HEADER_FIELDS:
        match = re.search(rf"(?im)^{re.escape(field)}:\s*(.+?)\s*$", text)
        if match:
            metadata[field.lower().replace(" ", "_")] = match.group(1).strip()

    # PDF-style lowercase keys
    for field in (
        "document_id",
        "product",
        "domain",
        "organization",
        "department",
        "document_type",
        "revision",
        "date",
        "topic",
        "file_path",
        "status",
    ):
        if field in metadata:
            continue
        match = re.search(rf"(?im)^{re.escape(field)}:\s*(.+?)\s*$", text)
        if match:
            metadata[field] = match.group(1).strip()

    metadata.setdefault("file_name", path.name)
    metadata.setdefault("relative_path", path.as_posix())
    metadata.setdefault("bucket", bucket)
    metadata.setdefault("source_type", "unstructured" if bucket == "unstructured" else bucket)
    if "document_id" not in metadata:
        metadata["document_id"] = path.stem
    return metadata


def parse_parameter_table(text: str) -> list[dict[str, str]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    start = None
    for index in range(len(lines) - 2):
        if (
            lines[index].lower() == "parameter"
            and lines[index + 1].lower() == "value"
            and lines[index + 2].lower() == "unit"
        ):
            start = index + 3
            break
    if start is None:
        return []

    rows: list[dict[str, str]] = []
    index = start
    while index + 2 < len(lines):
        parameter = lines[index]
        if parameter.lower().startswith(STOP_HEADERS):
            break
        value = lines[index + 1]
        unit = lines[index + 2]
        if value.lower().startswith(STOP_HEADERS) or unit.lower().startswith(STOP_HEADERS):
            break
        rows.append({"parameter": parameter, "value": value, "unit": unit})
        index += 3
    return rows


def header_chunk_text(metadata: dict[str, str], title_hint: Optional[str] = None) -> str:
    parts = [
        title_hint or "",
        f"Document {metadata.get('document_id')}",
        f"product {metadata.get('product')}",
        f"bucket {metadata.get('bucket')}",
        f"department {metadata.get('department')}",
        f"topic {metadata.get('topic')}",
        f"type {metadata.get('document_type')}",
    ]
    return " | ".join(part for part in parts if part and "None" not in part)


def spec_row_chunks(metadata: dict[str, str], rows: list[dict[str, str]]) -> list[dict[str, str]]:
    chunks = []
    document_id = metadata.get("document_id", "")
    product = metadata.get("product", "the product")
    for row in rows:
        unit = row["unit"]
        unit_text = ""
        if unit and unit not in {"-", "synthetic label"}:
            if unit.lower() not in row["value"].lower():
                unit_text = f" {unit}"
        content = (
            f"{product} {row['parameter']} is {row['value']}{unit_text} "
            f"[{document_id}]."
        )
        chunks.append({
            "content": content,
            "chunk_kind": "spec_row",
            "parameter": row["parameter"],
        })
    return chunks


def section_chunks(text: str, max_words: int = 350, overlap: int = 40) -> list[dict[str, str]]:
    matches = list(SECTION_RE.finditer(text))
    if not matches:
        return word_window_chunks(text, max_words=max_words, overlap=overlap)

    spans: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        title = match.group(2).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        spans.append((title, body))

    chunks: list[dict[str, str]] = []
    for title, body in spans:
        words = body.split()
        if len(words) <= max_words:
            chunks.append({
                "content": f"{title}. {body}".strip(),
                "chunk_kind": "section",
                "parameter": title,
            })
            continue
        start = 0
        while start < len(words):
            end = min(start + max_words, len(words))
            window = " ".join(words[start:end])
            chunks.append({
                "content": f"{title}. {window}".strip(),
                "chunk_kind": "section",
                "parameter": title,
            })
            if end == len(words):
                break
            start = max(end - overlap, start + 1)
    return chunks


def word_window_chunks(text: str, max_words: int = 300, overlap: int = 50) -> list[dict[str, str]]:
    words = text.split()
    if not words:
        return []
    chunks: list[dict[str, str]] = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append({
            "content": " ".join(words[start:end]),
            "chunk_kind": "word_window",
            "parameter": None,
        })
        if end == len(words):
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_document(
    text: str,
    metadata: dict[str, str],
    strategy: str = "table_section",
    word_size: int = 300,
    overlap: int = 50,
) -> list[dict[str, str]]:
    cleaned = clean_text(text)
    chunks: list[dict[str, str]] = []
    header = header_chunk_text(metadata)
    if header:
        chunks.append({"content": header, "chunk_kind": "header", "parameter": None})

    rows = parse_parameter_table(cleaned)

    if strategy in {"table_section", "table_row"} and rows:
        chunks.extend(spec_row_chunks(metadata, rows))
        if strategy == "table_row":
            return chunks

    if strategy in {"table_section", "section"} and not rows:
        chunks.extend(section_chunks(cleaned, max_words=word_size, overlap=overlap))
    elif strategy == "section":
        chunks.extend(section_chunks(cleaned, max_words=word_size, overlap=overlap))

    if strategy.startswith("word"):
        size = 400 if "400" in strategy else 200 if "200" in strategy else word_size
        chunks.extend(word_window_chunks(cleaned, max_words=size, overlap=overlap))

    if len(chunks) <= 1:
        chunks.extend(word_window_chunks(cleaned, max_words=word_size, overlap=overlap))
    return chunks
