import json

from rag_lab.generator import REFUSE_ANSWER, Generator
from rag_lab.models import Chunk, RetrievedChunk, to_json_dict


def meals_hit() -> RetrievedChunk:
    chunk = Chunk(
        chunk_id="expense-policy:v2.0:section-1",
        document="Employee Expense Policy",
        version="2.0",
        section="1",
        section_title="Meals",
        text="Employees may claim up to $65 per day for meals while traveling overnight.",
    )
    return RetrievedChunk(chunk=chunk, distance=0.08)


def hotels_hit() -> RetrievedChunk:
    chunk = Chunk(
        chunk_id="expense-policy:v2.0:section-2",
        document="Employee Expense Policy",
        version="2.0",
        section="2",
        section_title="Hotels",
        text="Hotels are reimbursable up to $225 per night.",
    )
    return RetrievedChunk(chunk=chunk, distance=0.22)


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_generate_uses_retrieved_metadata_for_citation(monkeypatch):
    retrieved = [meals_hit(), hotels_hit()]

    def fake_post(url, json=None, timeout=None):
        assert url.endswith("/api/chat")
        assert json["model"] == "mistral:7b"
        assert json["format"] == "json"
        assert "Use only the provided policy excerpts" in json["messages"][0]["content"]
        return FakeResponse(
            {
                "message": {
                    "content": '{"answer":"Employees may claim up to $65 per day for meals.","section":"1"}'
                }
            }
        )

    monkeypatch.setattr("rag_lab.generator.httpx.post", fake_post)
    generator = Generator(ollama_url="http://127.0.0.1:11434", model="mistral:7b")
    response = generator.generate("How much can I spend on food each day?", retrieved)

    assert response.citation is not None
    assert response.citation.document == "Employee Expense Policy"
    assert response.citation.version == "2.0"
    assert response.citation.section == "1. Meals"
    payload = to_json_dict(response)
    assert payload["retrieved_chunks"][0] == {"section": "1. Meals", "distance": 0.08}
    assert isinstance(payload["retrieved_chunks"][0]["distance"], float)


def test_generate_refuses_without_citation_when_evidence_missing(monkeypatch):
    refuse_payload = json.dumps({"answer": REFUSE_ANSWER, "section": None})

    def fake_post(url, json=None, timeout=None):
        return FakeResponse({"message": {"content": refuse_payload}})

    monkeypatch.setattr("rag_lab.generator.httpx.post", fake_post)
    generator = Generator(ollama_url="http://127.0.0.1:11434", model="mistral:7b")
    response = generator.generate(
        "Does the company reimburse gym memberships?",
        [meals_hit(), hotels_hit()],
    )
    assert response.answer == REFUSE_ANSWER
    assert response.citation is None


def test_generate_refuses_if_model_cites_a_section_that_was_not_retrieved(monkeypatch):
    def fake_post(url, json=None, timeout=None):
        return FakeResponse(
            {"message": {"content": '{"answer":"Gym is covered.","section":"9"}'}}
        )

    monkeypatch.setattr("rag_lab.generator.httpx.post", fake_post)
    generator = Generator(ollama_url="http://127.0.0.1:11434", model="mistral:7b")
    response = generator.generate("gym?", [meals_hit()])
    assert response.answer == REFUSE_ANSWER
    assert response.citation is None
