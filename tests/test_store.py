from rag_lab.models import Chunk


def unit_vector(hot_index: int, dim: int = 384) -> list[float]:
    vector = [0.0] * dim
    vector[hot_index] = 1.0
    return vector


def make_chunk(section: str, title: str, hot_index: int) -> Chunk:
    return Chunk(
        chunk_id=f"expense-policy:v2.0:section-{section}",
        document="Employee Expense Policy",
        version="2.0",
        section=section,
        section_title=title,
        text=f"{title} policy text.",
        embedding=unit_vector(hot_index),
    )


def test_upsert_persists_six_rows_and_search_sorts_cosine_distance_ascending(store):
    chunks = [
        make_chunk("1", "Meals", 0),
        make_chunk("2", "Hotels", 1),
        make_chunk("3", "Airfare", 2),
        make_chunk("4", "Ground Transportation", 3),
        make_chunk("5", "Receipts", 4),
        make_chunk("6", "Submission Deadline", 5),
    ]
    store.upsert_chunks(chunks)
    assert store.count() == 6

    query = unit_vector(2)
    query[1] = 0.2
    norm = sum(x * x for x in query) ** 0.5
    query = [x / norm for x in query]

    results = store.search(query, limit=3)
    assert len(results) <= 3
    assert [item.chunk.section for item in results] == ["3", "2", "1"]
    distances = [item.distance for item in results]
    assert distances == sorted(distances)
    assert all(isinstance(item.distance, float) for item in results)
    row = results[0]
    assert row.chunk.chunk_id == "expense-policy:v2.0:section-3"
    assert row.chunk.document == "Employee Expense Policy"
    assert row.chunk.version == "2.0"
    assert row.chunk.section_title == "Airfare"
    assert row.chunk.text == "Airfare policy text."
