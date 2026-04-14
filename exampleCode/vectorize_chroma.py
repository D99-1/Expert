from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import chromadb
from fastembed import TextEmbedding

INPUT_PATH = Path(__file__).with_name("data.json")
CHROMA_PATH = Path(__file__).with_name("chroma_db")
COLLECTION_NAME = "ysws_active"
CHUNK_SIZE = 900
OVERLAP = 150
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def _chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    normalized = " ".join((text or "").split())
    if not normalized:
        return []

    step = max(1, chunk_size - overlap)
    chunks: list[str] = []
    start = 0

    while start < len(normalized):
        end = min(len(normalized), start + chunk_size)
        piece = normalized[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(normalized):
            break
        start += step

    return chunks


def _iter_active_records(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    records: list[tuple[str, dict[str, Any]]] = []
    data = payload.get("data", {})
    if not isinstance(data, dict):
        return records

    for category_name, items in data.items():
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            if str(item.get("status", "")).lower() != "active":
                continue
            records.append((category_name, item))
    return records


def _build_documents(payload: dict[str, Any], chunk_size: int, overlap: int) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []

    for category_name, item in _iter_active_records(payload):
        name = str(item.get("name", "Untitled YSWS"))
        website = str(item.get("website", ""))
        description = str(item.get("description", ""))
        site_content = str(item.get("website_content", ""))
        page_title = str(item.get("website_page_title", ""))
        deadline = str(item.get("deadline", ""))
        slack_channel = str(item.get("slackChannel", ""))

        combined_text = "\n".join(
            part
            for part in [
                f"Name: {name}",
                f"Category: {category_name}",
                f"Description: {description}",
                f"Website title: {page_title}",
                f"Website: {website}",
                f"Slack channel: {slack_channel}",
                f"Deadline: {deadline}",
                f"Website content: {site_content}",
            ]
            if part
        )

        chunks = _chunk_text(combined_text, chunk_size=chunk_size, overlap=overlap)
        for chunk_index, chunk in enumerate(chunks):
            stable_id = hashlib.md5(
                f"{name}|{website}|{chunk_index}|{chunk}".encode("utf-8")
            ).hexdigest()
            documents.append(
                {
                    "id": stable_id,
                    "document": chunk,
                    "metadata": {
                        "name": name,
                        "status": str(item.get("status", "")),
                        "category": category_name,
                        "website": website,
                        "deadline": deadline,
                        "slack_channel": slack_channel,
                        "chunk_index": chunk_index,
                    },
                }
            )

    return documents


def vectorize_active_ysws() -> dict[str, int]:
    input_path = INPUT_PATH
    chroma_path = CHROMA_PATH
    collection_name = COLLECTION_NAME
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    docs = _build_documents(payload, chunk_size=CHUNK_SIZE, overlap=OVERLAP)

    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_or_create_collection(name=collection_name)
    if not docs:
        return {"active_programs": 0, "chunk_count": 0}

    model = TextEmbedding(model_name=EMBEDDING_MODEL)
    texts = [doc["document"] for doc in docs]
    embeddings = list(model.embed(texts))

    collection.upsert(
        ids=[doc["id"] for doc in docs],
        documents=texts,
        metadatas=[doc["metadata"] for doc in docs],
        embeddings=embeddings,
    )

    active_program_names = {
        doc["metadata"]["name"] for doc in docs if doc["metadata"].get("status") == "active"
    }
    return {
        "active_programs": len(active_program_names),
        "chunk_count": len(docs),
    }


def main() -> None:
    stats = vectorize_active_ysws()
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
