from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag_lab.config import get_settings
from rag_lab.embedder import Embedder
from rag_lab.generator import Generator
from rag_lab.models import to_json_dict
from rag_lab.pipeline import answer_question, ingest_policy
from rag_lab.questions import REQUIRED_QUESTIONS
from rag_lab.retriever import Retriever
from rag_lab.store import ChunkStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Grounded expense-policy assistant")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Chunk, embed, and store policy.md")
    ingest.add_argument("policy", nargs="?", default="policy.md", type=Path)

    ask = sub.add_parser("ask", help="Answer a question from retrieved policy chunks")
    ask.add_argument("question", help="User question")

    sub.add_parser("eval", help="Run the six required questions and write JSON output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    embedder = Embedder(settings.embedding_model)
    store = ChunkStore(settings.database_url)

    if args.command == "ingest":
        chunks = ingest_policy(args.policy, embedder, store)
        print(f"ingested {len(chunks)} chunks into {settings.database_url}")
        return 0

    retriever = Retriever(embedder, store)
    generator = Generator(settings.ollama_url, settings.ollama_model)

    if args.command == "ask":
        response = answer_question(args.question, retriever, generator, limit=settings.retrieve_k)
        json.dump(to_json_dict(response), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    if args.command == "eval":
        output_path = Path("tests/output/required_questions.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results = []
        for item in REQUIRED_QUESTIONS:
            response = answer_question(
                item["question"], retriever, generator, limit=settings.retrieve_k
            )
            results.append({"question": item["question"], "response": to_json_dict(response)})
        output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {output_path}")
        return 0

    return 1
