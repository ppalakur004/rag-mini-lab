from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document: str
    version: str
    section: str
    section_title: str
    text: str
    embedding: list[float] | None = field(default=None, hash=False)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    distance: float


@dataclass(frozen=True)
class Citation:
    document: str
    version: str
    section: str


@dataclass(frozen=True)
class RetrievedChunkRef:
    section: str
    distance: float


@dataclass(frozen=True)
class AskResponse:
    answer: str
    citation: Citation | None
    retrieved_chunks: list[RetrievedChunkRef]


def section_label(section: str, section_title: str) -> str:
    return f"{section}. {section_title}"


def to_json_dict(response: AskResponse) -> dict:
    citation = None
    if response.citation is not None:
        citation = {
            "document": response.citation.document,
            "version": response.citation.version,
            "section": response.citation.section,
        }
    return {
        "answer": response.answer,
        "citation": citation,
        "retrieved_chunks": [
            {"section": item.section, "distance": item.distance}
            for item in response.retrieved_chunks
        ],
    }
