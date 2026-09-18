from __future__ import annotations

from rag_lab.embedder import Embedder
from rag_lab.models import RetrievedChunk
from rag_lab.store import ChunkStore


class Retriever:
    def __init__(self, embedder: Embedder, store: ChunkStore) -> None:
        self._embedder = embedder
        self._store = store

    def retrieve(self, question: str, limit: int = 3) -> list[RetrievedChunk]:
        query_embedding = self._embedder.embed(question)
        return self._store.search(query_embedding, limit=limit)
