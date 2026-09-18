from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from rag_lab.chunker import chunk_policy
from rag_lab.embedder import Embedder
from rag_lab.generator import Generator
from rag_lab.models import AskResponse, Chunk
from rag_lab.retriever import Retriever
from rag_lab.store import ChunkStore


def ingest_policy(path: Path, embedder: Embedder, store: ChunkStore) -> list[Chunk]:
    markdown = path.read_text(encoding="utf-8")
    chunks = chunk_policy(markdown)
    if len(chunks) != 6:
        raise ValueError(f"expected 6 chunks, got {len(chunks)}")
    vectors = embedder.embed_many([chunk.text for chunk in chunks])
    embedded = [replace(chunk, embedding=vector) for chunk, vector in zip(chunks, vectors)]
    store.upsert_chunks(embedded)
    return embedded


def answer_question(
    question: str,
    retriever: Retriever,
    generator: Generator,
    limit: int = 3,
) -> AskResponse:
    retrieved = retriever.retrieve(question, limit=limit)
    return generator.generate(question, retrieved)
