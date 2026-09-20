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

SYSTEM_PROMPT = """Answer the question using only the policy excerpts below.
Include the section that supports your answer.
If the excerpts do not contain the answer, respond:
"The provided policy does not answer this question."

Reply with JSON only, using this schema:
{"answer": "<string>", "section": "<section number like 1, or null>"}

Use only the provided policy excerpts.
- Answer from an applicable excerpt rule after synonym or class-to-instance matching (a more specific instance of a forbidden/required class still matches that class rule). Restate the answering excerpt's rule(s) in the answer.
- Cite the excerpt whose rule actually answers the question, not an excerpt that only mentions a related expense type. If two excerpts mention related items, cite only the one whose rule answers the asked requirement. Set section to exactly one number copied from that answering excerpt's header.
- When an excerpt states a required class, include that requirement in the answer if asked about a different class. Name the required class.
- Refuse with the exact sentence and section null when no applicable rule exists (a benefit the excerpts never mention).
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
            "Answer from an applicable excerpt rule after synonym or class-to-instance matching "
            "(a more specific instance of a forbidden/required class still matches that class rule). "
            "Restate the answering excerpt's rule(s) in the answer.\n"
            "Cite the excerpt whose rule actually answers the question, not an excerpt that only "
            "mentions a related expense type. If two excerpts mention related items, cite only the "
            "one whose rule answers the asked requirement. Set section to exactly one number copied "
            "from that answering excerpt's header.\n"
            "When an excerpt states a required class, include that requirement in the answer if "
            "asked about a different class. Name the required class.\n"
            "If no applicable rule exists (a benefit the excerpts never mention), refuse with the "
            "exact sentence and section null.\n"
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
