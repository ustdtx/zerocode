"""Persistent hybrid code index for ZeroCode."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import struct
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable


def _pack(vector: list[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def _unpack(data: bytes) -> tuple[float, ...]:
    return struct.unpack(f"<{len(data) // 4}f", data)


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(value * value for value in left))
    norm_right = math.sqrt(sum(value * value for value in right))
    return dot / (norm_left * norm_right) if norm_left and norm_right else 0.0


class EmbeddingClient:
    def __init__(self, endpoint: str = "http://127.0.0.1:2248/v1", model: str | None = None, api_key: str = "sk-llama-local-key", timeout: int = 180):
        self.base = endpoint.rstrip("/")
        self.endpoint = self.base + "/embeddings"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def _model_id(self) -> str:
        if self.model:
            return self.model
        request = urllib.request.Request(self.base + "/models", headers={"Authorization": f"Bearer {self.api_key}"})
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                models = json.loads(response.read()).get("data", [])
            if models and models[0].get("id"):
                self.model = models[0]["id"]
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            pass
        return self.model or "embedding"

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = json.dumps({"model": self._model_id(), "input": texts, "encoding_format": "float"}).encode()
        request = urllib.request.Request(self.endpoint, data=payload, headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}",
        })
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:800]
            raise RuntimeError(f"Embedding server rejected request ({exc.code}): {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"Embedding server unavailable at {self.endpoint}: {exc}") from exc
        try:
            rows = sorted(body["data"], key=lambda row: row["index"])
            return [list(map(float, row["embedding"])) for row in rows]
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid embedding response: {json.dumps(body)[:800]}") from exc


class CodeIndex:
    def __init__(self, root: Path, embedding: EmbeddingClient | None = None):
        self.root = root.resolve()
        self.embedding = embedding
        self.directory = self.root / ".zerocode"
        self.database = self.directory / "index.sqlite3"

    def _connect(self) -> sqlite3.Connection:
        self.directory.mkdir(exist_ok=True)
        connection = sqlite3.connect(self.database)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("""
            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                digest TEXT NOT NULL,
                size INTEGER NOT NULL
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                path TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding BLOB
            )
        """)
        connection.execute("CREATE INDEX IF NOT EXISTS chunks_path ON chunks(path)")
        connection.execute("""
            CREATE TABLE IF NOT EXISTS symbols (
                id INTEGER PRIMARY KEY,
                path TEXT NOT NULL,
                name TEXT NOT NULL,
                kind TEXT NOT NULL,
                line INTEGER NOT NULL
            )
        """)
        connection.execute("CREATE INDEX IF NOT EXISTS symbols_name ON symbols(name)")
        return connection

    def _files(self) -> Iterable[Path]:
        ignored = {".git", ".venv", "venv", "node_modules", "bin", "obj", "dist", "build", "__pycache__", ".zerocode"}
        extensions = {".c", ".cc", ".cpp", ".cs", ".css", ".go", ".h", ".hpp", ".html", ".java", ".js", ".json", ".jsx", ".md", ".py", ".rs", ".scss", ".sh", ".sql", ".swift", ".toml", ".ts", ".tsx", ".txt", ".xml", ".yaml", ".yml"}
        for path in sorted(self.root.rglob("*")):
            if path.is_file() and not any(part in ignored for part in path.parts) and (path.suffix.lower() in extensions or not path.suffix):
                yield path

    def _digest(self, path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _symbols(self, path: str, lines: list[str]) -> list[tuple[str, str, int]]:
        import re
        pattern = re.compile(r"^\s*(?:class|interface|struct|enum|def|async\s+def|function|func|fn|const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)")
        found = []
        for number, line in enumerate(lines, 1):
            match = pattern.search(line)
            if match:
                found.append((match.group(1), line.strip().split()[0], number))
        return found

    def _chunks(self, lines: list[str], size: int = 32, overlap: int = 6) -> list[tuple[int, int, str]]:
        chunks = []
        step = max(1, size - overlap)
        for start in range(0, len(lines), step):
            end = min(len(lines), start + size)
            text = "\n".join(lines[start:end]).strip()
            if text:
                chunks.append((start + 1, end, text))
            if end == len(lines):
                break
        return chunks

    def build(self) -> str:
        connection = self._connect()
        indexed = 0
        embedded = 0
        pending: list[tuple[int, str]] = []
        try:
            known = {row[0]: row[1] for row in connection.execute("SELECT path, digest FROM files")}
            current: set[str] = set()
            for file_path in self._files():
                relative = file_path.relative_to(self.root).as_posix()
                current.add(relative)
                digest = self._digest(file_path)
                if known.get(relative) == digest:
                    continue
                try:
                    lines = file_path.read_text(encoding="utf-8").splitlines()
                except (OSError, UnicodeDecodeError):
                    continue
                connection.execute("DELETE FROM chunks WHERE path = ?", (relative,))
                connection.execute("DELETE FROM symbols WHERE path = ?", (relative,))
                connection.execute("INSERT OR REPLACE INTO files(path, digest, size) VALUES (?, ?, ?)", (relative, digest, file_path.stat().st_size))
                symbols = self._symbols(relative, lines)
                connection.executemany("INSERT INTO symbols(path, name, kind, line) VALUES (?, ?, ?, ?)", [(relative, name, kind, line) for name, kind, line in symbols])
                for start, end, text in self._chunks(lines):
                    cursor = connection.execute("INSERT INTO chunks(path, start_line, end_line, text) VALUES (?, ?, ?, ?)", (relative, start, end, text))
                    pending.append((cursor.lastrowid, text))
                indexed += 1
            stale = set(known) - current
            for path in stale:
                connection.execute("DELETE FROM files WHERE path = ?", (path,))
                connection.execute("DELETE FROM chunks WHERE path = ?", (path,))
                connection.execute("DELETE FROM symbols WHERE path = ?", (path,))
            if self.embedding:
                existing_ids = {chunk_id for chunk_id, _ in pending}
                pending.extend(
                    (chunk_id, text)
                    for chunk_id, text in connection.execute("SELECT id, text FROM chunks WHERE embedding IS NULL")
                    if chunk_id not in existing_ids
                )
            embedding_error = ""
            if self.embedding and pending:
                try:
                    offset = 0
                    while offset < len(pending):
                        batch: list[tuple[int, str]] = []
                        characters = 0
                        while offset < len(pending) and (not batch or characters + len(pending[offset][1]) <= 6000):
                            batch.append(pending[offset])
                            characters += len(pending[offset][1])
                            offset += 1
                        vectors = self.embedding.embed([text for _, text in batch])
                        if len(vectors) != len(batch):
                            raise RuntimeError("Embedding count did not match chunk count")
                        for (chunk_id, _), vector in zip(batch, vectors):
                            connection.execute("UPDATE chunks SET embedding = ? WHERE id = ?", (_pack(vector), chunk_id))
                            embedded += 1
                except RuntimeError as exc:
                    embedding_error = str(exc)
            connection.commit()
            total = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            suffix = f" embedding_error={embedding_error}" if embedding_error else ""
            return f"indexed_files={indexed} embedded_chunks={embedded} total_chunks={total} database={self.database}{suffix}"
        finally:
            connection.close()

    def retrieve(self, query: str, limit: int = 8) -> str:
        connection = self._connect()
        try:
            rows = connection.execute("SELECT id, path, start_line, end_line, text, embedding FROM chunks").fetchall()
            if not rows:
                return "(index is empty; run build_index)"
            query_terms = {term.lower() for term in query.split() if len(term) > 2}
            query_vector: tuple[float, ...] = ()
            if self.embedding:
                try:
                    vectors = self.embedding.embed([query])
                    query_vector = tuple(vectors[0]) if vectors else ()
                except RuntimeError:
                    pass
            scored = []
            for row in rows:
                text_lower = row[4].lower()
                lexical = sum(text_lower.count(term) for term in query_terms)
                semantic = _cosine(query_vector, _unpack(row[5])) if query_vector and row[5] else 0.0
                scored.append((semantic * 0.7 + min(1.0, lexical / 5.0) * 0.3, row))
            selected = sorted(scored, key=lambda item: item[0], reverse=True)[:limit]
            return "\n\n".join(
                f"{row[1]}:{row[2]}-{row[3]} score={score:.3f}\n{row[4]}" for score, row in selected
            )
        finally:
            connection.close()

    def symbols(self, name: str) -> str:
        connection = self._connect()
        try:
            rows = connection.execute("SELECT path, kind, line FROM symbols WHERE name = ? LIMIT 40", (name,)).fetchall()
            return "\n".join(f"{path}:{line}: {kind} {name}" for path, kind, line in rows) or "(symbol not found)"
        finally:
            connection.close()
