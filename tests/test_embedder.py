from rag_lab.embedder import Embedder


def test_embed_returns_normalized_384d_vector():
    embedder = Embedder()
    vector = embedder.embed("Employees may claim up to $65 per day for meals.")
    assert len(vector) == 384
    assert all(isinstance(x, float) for x in vector)
    norm = sum(x * x for x in vector) ** 0.5
    assert abs(norm - 1.0) < 1e-5


def test_embed_many_preserves_order_and_differs_for_unrelated_text():
    embedder = Embedder()
    meals, gym = embedder.embed_many(
        [
            "Employees may claim up to $65 per day for meals while traveling overnight.",
            "Does the company reimburse gym memberships?",
        ]
    )
    assert len(meals) == len(gym) == 384
    assert meals != gym
