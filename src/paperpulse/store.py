"""
SQLite persistence: papers, cached embeddings, ingestion watermarks and runs.

Embeddings are stored as float32 blobs and compared in numpy. Candidate sets
are small (one time window, a few hundred papers), so brute force is faster
and simpler than a vector index, which also handles pre-filtering poorly.
"""

import json
import sqlite3
from collections.abc import Iterable
from datetime import date, datetime
from pathlib import Path

import numpy as np

from paperpulse.models import Assessment, Digest, Paper

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    id                TEXT PRIMARY KEY,
    doi               TEXT,
    pmid              TEXT,
    title             TEXT NOT NULL,
    abstract          TEXT NOT NULL,
    journal           TEXT NOT NULL,
    published         TEXT,
    added             TEXT NOT NULL,
    authors           TEXT NOT NULL,
    publication_types TEXT NOT NULL,
    url               TEXT NOT NULL,
    source            TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS papers_added ON papers(added);
CREATE INDEX IF NOT EXISTS papers_pmid ON papers(pmid);

-- Keyword index for BM25. Maintained by upsert_papers(); rebuilt on open if out of sync.
CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
    id UNINDEXED, title, abstract, tokenize = 'porter unicode61'
);

CREATE TABLE IF NOT EXISTS embeddings (
    paper_id TEXT NOT NULL REFERENCES papers(id) ON UPDATE CASCADE ON DELETE CASCADE,
    model    TEXT NOT NULL,
    vector   BLOB NOT NULL,
    PRIMARY KEY (paper_id, model)
);

CREATE TABLE IF NOT EXISTS assessments (
    paper_id   TEXT NOT NULL REFERENCES papers(id) ON UPDATE CASCADE ON DELETE CASCADE,
    key        TEXT NOT NULL,
    assessment TEXT NOT NULL,
    PRIMARY KEY (paper_id, key)
);

CREATE TABLE IF NOT EXISTS ingest_state (
    source   TEXT PRIMARY KEY,
    last_end TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id         TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    digest     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run_items (
    run_id   TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    paper_id TEXT NOT NULL REFERENCES papers(id) ON UPDATE CASCADE,
    rank     INTEGER NOT NULL,
    PRIMARY KEY (run_id, paper_id)
);
"""

PAPER_COLUMNS = (
    "id, doi, pmid, title, abstract, journal, published, added, "
    "authors, publication_types, url, source"
)


class Store:
    def __init__(self, path: Path | str):
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(SCHEMA)
        self._sync_keyword_index()

    def _sync_keyword_index(self) -> None:
        """Databases created before the keyword index existed get it backfilled once."""
        (papers,) = self.db.execute("SELECT count(*) FROM papers").fetchone()
        (indexed,) = self.db.execute("SELECT count(*) FROM papers_fts").fetchone()
        if papers != indexed:
            with self.db:
                self.db.execute("DELETE FROM papers_fts")
                self.db.execute(
                    "INSERT INTO papers_fts (id, title, abstract) "
                    "SELECT id, title, abstract FROM papers"
                )

    def close(self) -> None:
        self.db.close()

    # -- papers ------------------------------------------------------------

    def upsert_papers(self, papers: Iterable[Paper]) -> int:
        """Insert or refresh papers. Returns how many were new."""
        new = 0
        with self.db:
            for p in papers:
                self._migrate_pmid_id(p)
                old = self.db.execute(
                    "SELECT title, abstract FROM papers WHERE id = ?", (p.id,)
                ).fetchone()
                text_changed = old is not None and (old["title"], old["abstract"]) != (
                    p.title,
                    p.abstract,
                )
                if old is None:
                    new += 1
                elif text_changed:
                    self.db.execute("DELETE FROM embeddings WHERE paper_id = ?", (p.id,))
                    self.db.execute("DELETE FROM assessments WHERE paper_id = ?", (p.id,))
                    self.db.execute("DELETE FROM papers_fts WHERE id = ?", (p.id,))
                if old is None or text_changed:
                    self.db.execute(
                        "INSERT INTO papers_fts (id, title, abstract) VALUES (?, ?, ?)",
                        (p.id, p.title, p.abstract),
                    )
                self.db.execute(
                    f"INSERT INTO papers ({PAPER_COLUMNS}) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(id) DO UPDATE SET "
                    + ", ".join(f"{c} = excluded.{c}" for c in PAPER_COLUMNS.split(", ")[1:]),
                    (
                        p.id,
                        p.doi,
                        p.pmid,
                        p.title,
                        p.abstract,
                        p.journal,
                        p.published.isoformat() if p.published else None,
                        p.added.isoformat(),
                        json.dumps(p.authors),
                        json.dumps(p.publication_types),
                        p.url,
                        p.source,
                    ),
                )
        return new

    def _migrate_pmid_id(self, paper: Paper) -> None:
        """A paper first stored as `pmid:<n>` can gain a DOI later; re-key it so history follows."""
        if paper.doi and paper.pmid:
            old_id = f"pmid:{paper.pmid}"
            moved = self.db.execute(
                "UPDATE papers SET id = ? WHERE id = ? AND NOT EXISTS "
                "(SELECT 1 FROM papers WHERE id = ?)",
                (paper.id, old_id, paper.id),
            ).rowcount
            if moved:
                self.db.execute("UPDATE papers_fts SET id = ? WHERE id = ?", (paper.id, old_id))

    def papers_added_between(self, start: date, end: date) -> list[Paper]:
        rows = self.db.execute(
            f"SELECT {PAPER_COLUMNS} FROM papers WHERE added BETWEEN ? AND ? ORDER BY added, id",
            (start.isoformat(), end.isoformat()),
        )
        return [_row_to_paper(r) for r in rows]

    def keyword_search(self, query: str, within: list[str]) -> list[tuple[str, float]]:
        """
        BM25 search over titles and abstracts of the papers in `within`, best
        first. Title matches weigh double. Scores are sign-flipped from SQLite's
        convention so that higher is better. IDF statistics span the whole
        corpus, which only makes them more stable.
        """
        if not within:
            return []
        marks = ",".join("?" * len(within))
        rows = self.db.execute(
            "SELECT id, -bm25(papers_fts, 0.0, 2.0, 1.0) AS score FROM papers_fts "
            f"WHERE papers_fts MATCH ? AND id IN ({marks}) ORDER BY score DESC",
            (query, *within),
        )
        return [(r["id"], r["score"]) for r in rows]

    # -- embeddings --------------------------------------------------------

    def get_embeddings(self, paper_ids: list[str], model: str) -> dict[str, np.ndarray]:
        found = {}
        for chunk in _chunks(paper_ids, 500):
            marks = ",".join("?" * len(chunk))
            rows = self.db.execute(
                f"SELECT paper_id, vector FROM embeddings "
                f"WHERE model = ? AND paper_id IN ({marks})",
                (model, *chunk),
            )
            found.update(
                {r["paper_id"]: np.frombuffer(r["vector"], dtype=np.float32) for r in rows}
            )
        return found

    def put_embeddings(self, model: str, vectors: dict[str, np.ndarray]) -> None:
        with self.db:
            self.db.executemany(
                "INSERT OR REPLACE INTO embeddings (paper_id, model, vector) VALUES (?, ?, ?)",
                [(pid, model, v.astype(np.float32).tobytes()) for pid, v in vectors.items()],
            )

    # -- LLM assessments ---------------------------------------------------

    def get_assessments(self, paper_ids: list[str], key: str) -> dict[str, Assessment]:
        found = {}
        for chunk in _chunks(paper_ids, 500):
            marks = ",".join("?" * len(chunk))
            rows = self.db.execute(
                f"SELECT paper_id, assessment FROM assessments "
                f"WHERE key = ? AND paper_id IN ({marks})",
                (key, *chunk),
            )
            found.update(
                {r["paper_id"]: Assessment.model_validate_json(r["assessment"]) for r in rows}
            )
        return found

    def put_assessment(self, paper_id: str, key: str, assessment: Assessment) -> None:
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO assessments (paper_id, key, assessment) VALUES (?, ?, ?)",
                (paper_id, key, assessment.model_dump_json()),
            )

    # -- ingestion watermark -----------------------------------------------

    def last_ingested(self, source: str) -> date | None:
        row = self.db.execute(
            "SELECT last_end FROM ingest_state WHERE source = ?", (source,)
        ).fetchone()
        return date.fromisoformat(row["last_end"]) if row else None

    def set_last_ingested(self, source: str, end: date) -> None:
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO ingest_state (source, last_end) VALUES (?, ?)",
                (source, end.isoformat()),
            )

    # -- runs --------------------------------------------------------------

    def save_run(self, digest: Digest) -> None:
        with self.db:
            self.db.execute(
                "INSERT INTO runs (id, created_at, digest) VALUES (?, ?, ?)",
                (digest.run.id, digest.run.created_at.isoformat(), digest.model_dump_json()),
            )
            self.db.executemany(
                "INSERT INTO run_items (run_id, paper_id, rank) VALUES (?, ?, ?)",
                [(digest.run.id, p.id, p.rank) for p in digest.papers],
            )

    def selected_since(self, since: datetime) -> set[str]:
        rows = self.db.execute(
            "SELECT DISTINCT i.paper_id FROM run_items i JOIN runs r ON r.id = i.run_id "
            "WHERE r.created_at >= ?",
            (since.isoformat(),),
        )
        return {r["paper_id"] for r in rows}

    def latest_run(self) -> Digest | None:
        row = self.db.execute("SELECT digest FROM runs ORDER BY created_at DESC LIMIT 1").fetchone()
        return Digest.model_validate_json(row["digest"]) if row else None


def _row_to_paper(row: sqlite3.Row) -> Paper:
    data = dict(row)
    data["authors"] = json.loads(data["authors"])
    data["publication_types"] = json.loads(data["publication_types"])
    return Paper.model_validate(data)


def _chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]
