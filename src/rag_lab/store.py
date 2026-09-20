from __future__ import annotations

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

from rag_lab.models import Chunk, RetrievedChunk


class ChunkStore:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def _connect(self) -> psycopg.Connection:
        conn = psycopg.connect(self._dsn)
        register_vector(conn)
        return conn

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        missing = [chunk.chunk_id for chunk in chunks if chunk.embedding is None]
        if missing:
            raise ValueError("chunks missing embeddings: {0}".format(missing))
        sql = """
            INSERT INTO chunks (
                chunk_id, document, version, section, section_title, text, embedding
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (chunk_id) DO UPDATE SET
                document = EXCLUDED.document,
                version = EXCLUDED.version,
                section = EXCLUDED.section,
                section_title = EXCLUDED.section_title,
                text = EXCLUDED.text,
                embedding = EXCLUDED.embedding
        """
        rows = [
            (
                chunk.chunk_id,
                chunk.document,
                chunk.version,
                chunk.section,
                chunk.section_title,
                chunk.text,
                chunk.embedding,
            )
            for chunk in chunks
        ]
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(sql, rows)
            conn.commit()

    def search(self, query_embedding: list[float], limit: int = 3) -> list[RetrievedChunk]:
        sql = """
            SELECT
                chunk_id,
                document,
                version,
                section,
                section_title,
                text,
                embedding <=> %s AS distance
            FROM chunks
            ORDER BY embedding <=> %s ASC
            LIMIT %s
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                query = Vector(query_embedding)
                cur.execute(sql, (query, query, limit))
                rows = cur.fetchall()
        results = []
        for row in rows:
            chunk = Chunk(
                chunk_id=row[0],
                document=row[1],
                version=row[2],
                section=row[3],
                section_title=row[4],
                text=row[5],
            )
            results.append(RetrievedChunk(chunk=chunk, distance=float(row[6])))
        return results

    def count(self) -> int:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM chunks")
                value = cur.fetchone()[0]
        return int(value)

    def truncate(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE chunks")
            conn.commit()
