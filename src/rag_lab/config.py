from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    database_url: str = os.environ.get(
        "DATABASE_URL",
        "postgresql://rag:rag@127.0.0.1:5433/rag_lab",
    )
    ollama_url: str = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
    ollama_model: str = os.environ.get("OLLAMA_MODEL", "mistral:7b")
    embedding_model: str = os.environ.get(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    retrieve_k: int = 3


def get_settings() -> Settings:
    return Settings()
