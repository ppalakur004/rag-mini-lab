from pathlib import Path

import pytest

from rag_lab.config import get_settings
from rag_lab.embedder import Embedder
from rag_lab.models import section_label
from rag_lab.pipeline import ingest_policy
from rag_lab.questions import REQUIRED_QUESTIONS
from rag_lab.retriever import Retriever
from rag_lab.store import ChunkStore

POLICY_PATH = Path(__file__).resolve().parents[1] / "policy.md"


@pytest.fixture(scope="module")
def retriever(dsn: str) -> Retriever:
    psycopg = pytest.importorskip("psycopg")
    try:
        with psycopg.connect(dsn, connect_timeout=3) as conn:
            conn.execute("SELECT 1")
    except Exception as exc:
        pytest.skip(f"postgres unavailable at {dsn}: {exc}")

    settings = get_settings()
    store = ChunkStore(dsn)
    store.truncate()
    embedder = Embedder(settings.embedding_model)
    ingest_policy(POLICY_PATH, embedder, store)
    assert store.count() == 6
    return Retriever(embedder, store)


@pytest.mark.parametrize(
    "item",
    [q for q in REQUIRED_QUESTIONS if q["expected_section"] is not None],
    ids=lambda q: q["question"],
)
def test_expected_section_is_in_top_three(retriever: Retriever, item: dict):
    results = retriever.retrieve(item["question"], limit=3)
    assert len(results) <= 3
    distances = [hit.distance for hit in results]
    assert distances == sorted(distances)
    labels = [section_label(hit.chunk.section, hit.chunk.section_title) for hit in results]
    assert item["expected_section"] in labels


def test_gym_question_retrieves_at_most_three_chunks(retriever: Retriever):
    gym = next(q for q in REQUIRED_QUESTIONS if q["must_refuse"])
    results = retriever.retrieve(gym["question"], limit=3)
    assert len(results) <= 3
    distances = [hit.distance for hit in results]
    assert distances == sorted(distances)
