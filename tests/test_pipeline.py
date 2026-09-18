from dataclasses import replace
from pathlib import Path

import pytest

from rag_lab.chunker import chunk_policy
from rag_lab.models import AskResponse, Chunk, Citation, RetrievedChunk, RetrievedChunkRef
from rag_lab.pipeline import ingest_policy


class FakeEmbedder:
    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [[float(i), 0.0] + [0.0] * 382 for i, _ in enumerate(texts)]


class FakeStore:
    def __init__(self) -> None:
        self.upserted: list[Chunk] = []

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        self.upserted = chunks


def test_ingest_policy_embeds_and_stores_exactly_six_chunks():
    path = Path(__file__).resolve().parents[1] / "policy.md"
    store = FakeStore()
    chunks = ingest_policy(path, FakeEmbedder(), store)
    assert len(chunks) == 6
    assert len(store.upserted) == 6
    assert all(chunk.embedding is not None and len(chunk.embedding) == 384 for chunk in chunks)
    assert [chunk.section for chunk in store.upserted] == ["1", "2", "3", "4", "5", "6"]


def test_ingest_policy_rejects_wrong_chunk_count(tmp_path: Path):
    policy = tmp_path / "policy.md"
    policy.write_text("# Employee Expense Policy — Version 2.0\n\n## 1. Meals\nOnly one.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected 6 chunks"):
        ingest_policy(policy, FakeEmbedder(), FakeStore())
