from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "policy.md"


def postgres_dsn() -> str:
    return os.environ.get(
        "DATABASE_URL",
        "postgresql://rag:rag@127.0.0.1:5433/rag_lab",
    )


@pytest.fixture
def dsn() -> str:
    return postgres_dsn()


@pytest.fixture
def store(dsn: str):
    psycopg = pytest.importorskip("psycopg")
    try:
        with psycopg.connect(dsn, connect_timeout=3) as conn:
            conn.execute("SELECT 1")
    except Exception as exc:
        pytest.skip(f"postgres unavailable at {dsn}: {exc}")

    from rag_lab.store import ChunkStore

    chunk_store = ChunkStore(dsn)
    chunk_store.truncate()
    yield chunk_store
    chunk_store.truncate()
