from __future__ import annotations

import re

from rag_lab.models import Chunk

H1_RE = re.compile(
    r"^#\s+(?P<title>.+?)\s+—\s+Version\s+(?P<version>[0-9.]+)\s*$",
    re.MULTILINE,
)
HEADING_RE = re.compile(
    r"^##\s+(?P<section>\d+)\.\s+(?P<title>.+?)\s*$",
    re.MULTILINE,
)


def chunk_policy(markdown: str) -> list[Chunk]:
    h1 = H1_RE.search(markdown)
    if h1 is None:
        raise ValueError("policy.md must have an H1 like '# Title — Version 2.0'")

    document = h1.group("title").strip()
    version = h1.group("version").strip()
    matches = list(HEADING_RE.finditer(markdown))
    if not matches:
        raise ValueError("policy.md has no numbered section headings")

    chunks: list[Chunk] = []
    for index, match in enumerate(matches):
        section = match.group("section")
        section_title = match.group("title").strip()
        body_start = match.end()
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        text = markdown[body_start:body_end].strip()
        if not text:
            raise ValueError(f"section {section} has an empty body")
        chunks.append(
            Chunk(
                chunk_id=f"expense-policy:v{version}:section-{section}",
                document=document,
                version=version,
                section=section,
                section_title=section_title,
                text=text,
            )
        )
    return chunks
