from pathlib import Path

from rag_lab.chunker import chunk_policy

POLICY_PATH = Path(__file__).resolve().parents[1] / "policy.md"


def test_chunk_policy_produces_six_sections_with_metadata():
    markdown = POLICY_PATH.read_text(encoding="utf-8")
    chunks = chunk_policy(markdown)

    assert len(chunks) == 6
    assert [c.section for c in chunks] == ["1", "2", "3", "4", "5", "6"]
    assert [c.section_title for c in chunks] == [
        "Meals",
        "Hotels",
        "Airfare",
        "Ground Transportation",
        "Receipts",
        "Submission Deadline",
    ]
    assert {c.document for c in chunks} == {"Employee Expense Policy"}
    assert {c.version for c in chunks} == {"2.0"}
    assert [c.chunk_id for c in chunks] == [
        "expense-policy:v2.0:section-1",
        "expense-policy:v2.0:section-2",
        "expense-policy:v2.0:section-3",
        "expense-policy:v2.0:section-4",
        "expense-policy:v2.0:section-5",
        "expense-policy:v2.0:section-6",
    ]
    assert chunks[0].text.startswith("Employees may claim up to $65 per day")
    assert "Alcohol is not reimbursable." in chunks[0].text
    assert all(c.text.endswith(".") for c in chunks)
    assert all(c.embedding is None for c in chunks)
    assert "##" not in "".join(c.text for c in chunks)
