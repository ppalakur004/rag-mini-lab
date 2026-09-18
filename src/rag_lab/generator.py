from __future__ import annotations

import json

import httpx

from rag_lab.models import (
    AskResponse,
    Citation,
    RetrievedChunk,
    RetrievedChunkRef,
    section_label,
)

REFUSE_ANSWER = "The provided policy does not answer this question."

SYSTEM_PROMPT = """Reply with JSON only, using this schema:
{"answer": "<string>", "section": "<section number like 1, or null>"}

Treat synonyms of excerpt rules as answered by that excerpt. Different wording is not a reason to refuse.
If an excerpt states a rule about a class of items (travel class, vehicle class, expense category), apply that rule to a more specific instance of the same class. Refuse only when the question is a different topic the excerpts never cover.

Refuse only when no excerpt states an applicable rule even after synonym matching. Then set answer to exactly "The provided policy does not answer this question." and section to null.

Use only the provided policy excerpts; do not add facts that are not in them.
If you answer, set section to the supporting excerpt's section number.
If any excerpt is on the same topic, you must answer and cite that excerpt. Do not refuse for synonyms, omitted conditions, or amounts compared with a stated limit.
"""


class Generator:
    def __init__(self, ollama_url: str, model: str = "mistral:7b") -> None:
        self._url = f"{ollama_url.rstrip('/')}/api/chat"
        self._model = model

    def generate(self, question: str, retrieved: list[RetrievedChunk]) -> AskResponse:
        refs = [
            RetrievedChunkRef(
                section=section_label(item.chunk.section, item.chunk.section_title),
                distance=float(item.distance),
            )
            for item in retrieved
        ]
        parsed = self._complete(question, retrieved)
        answer = str(parsed.get("answer", "")).strip()
        section = parsed.get("section")
        if answer == REFUSE_ANSWER or section in (None, "", "null"):
            return AskResponse(answer=REFUSE_ANSWER, citation=None, retrieved_chunks=refs)

        section_key = str(section).strip()
        match = next((item for item in retrieved if _section_matches(item, section_key)), None)
        if match is None:
            return AskResponse(answer=REFUSE_ANSWER, citation=None, retrieved_chunks=refs)

        citation = Citation(
            document=match.chunk.document,
            version=match.chunk.version,
            section=section_label(match.chunk.section, match.chunk.section_title),
        )
        return AskResponse(answer=answer, citation=citation, retrieved_chunks=refs)

    def _complete(self, question: str, retrieved: list[RetrievedChunk]) -> dict:
        excerpts = []
        for item in retrieved:
            label = section_label(item.chunk.section, item.chunk.section_title)
            excerpts.append(f"[Section {label}]\n{item.chunk.text}")
        user_prompt = (
            "Policy excerpts:\n"
            + "\n\n".join(excerpts)
            + f"\n\nQuestion: {question}\n\n"
            "If the question is not answered by these excerpts even after synonym matching "
            "(for example a benefit the excerpts never mention), refuse.\n"
            "Otherwise answer from the matching excerpt and set section to that excerpt's number.\n"
            "If a condition is in the excerpt but not the question, still state the excerpt's rule."
        )
        response = httpx.post(
            self._url,
            json={
                "model": self._model,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0},
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=120.0,
        )
        response.raise_for_status()
        content = response.json()["message"]["content"]
        return _parse_json_content(content)


def _section_matches(item: RetrievedChunk, section_key: str) -> bool:
    label = section_label(item.chunk.section, item.chunk.section_title)
    return section_key == item.chunk.section or section_key == label


def _parse_json_content(content: str) -> dict:
    text = content.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise
        return json.loads(text[start : end + 1])
