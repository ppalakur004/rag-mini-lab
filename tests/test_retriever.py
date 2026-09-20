from rag_lab.models import Chunk, RetrievedChunk
from rag_lab.retriever import Retriever


class FakeEmbedder:
    def __init__(self, vector: list[float]) -> None:
        self.vector = vector

    def embed(self, text: str) -> list[float]:
        assert text == "Can I book first-class airfare?"
        return self.vector


class FakeStore:
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self.results = results
        self.seen: list[tuple[list[float], int]] = []

    def search(self, query_embedding: list[float], limit: int = 3) -> list[RetrievedChunk]:
        self.seen.append((query_embedding, limit))
        return self.results[:limit]


def test_retriever_embeds_question_and_returns_at_most_three_sorted_chunks():
    meals = Chunk(
        chunk_id="expense-policy:v2.0:section-1",
        document="Employee Expense Policy",
        version="2.0",
        section="1",
        section_title="Meals",
        text="meals",
    )
    airfare = Chunk(
        chunk_id="expense-policy:v2.0:section-3",
        document="Employee Expense Policy",
        version="2.0",
        section="3",
        section_title="Airfare",
        text="airfare",
    )
    hotels = Chunk(
        chunk_id="expense-policy:v2.0:section-2",
        document="Employee Expense Policy",
        version="2.0",
        section="2",
        section_title="Hotels",
        text="hotels",
    )
    query = [0.1] * 384
    store = FakeStore(
        [
            RetrievedChunk(chunk=airfare, distance=0.08),
            RetrievedChunk(chunk=hotels, distance=0.21),
            RetrievedChunk(chunk=meals, distance=0.33),
        ]
    )
    retriever = Retriever(embedder=FakeEmbedder(query), store=store)

    results = retriever.retrieve("Can I book first-class airfare?", limit=3)

    assert store.seen == [(query, 3)]
    assert len(results) == 3
    assert [item.chunk.section for item in results] == ["3", "2", "1"]
    assert [item.distance for item in results] == [0.08, 0.21, 0.33]
