"""
PaperPulse domain models.

Defines the core data structures used throughout the pipeline.
"""

from dataclasses import dataclass, field
from datetime import date


# ---------------------------------------------------------------------------
# Paper
# ---------------------------------------------------------------------------

@dataclass
class Paper:
    """
    A single paper fetched from PubMed.

    Core fields are populated by the PubMed fetcher.
    Optional fields are added by downstream pipeline steps.
    """

    # Core fields — always populated by PubMed fetcher
    pmid: str
    title: str
    abstract: str
    journal: str
    pub_date: str
    authors: list[str]
    doi: str
    url: str

    # Set by the embedding pre-filter
    embedding_score: float = 0.0

    # Set by the LLM scoring step
    reason: str = ""

    # Set when reading back from the database
    rank: int = 0
    digest_date: str = ""

    def authors_str(self) -> str:
        """Return authors as a comma-separated string for database storage."""
        return ", ".join(self.authors)

    def __str__(self) -> str:
        return f"[{self.journal}] {self.title[:80]}"


# ---------------------------------------------------------------------------
# Digest
# ---------------------------------------------------------------------------

@dataclass
class Digest:
    """
    A daily digest — the top 3 papers selected for a given date.
    """

    digest_date: date
    papers: list[Paper] = field(default_factory=list)

    def __str__(self) -> str:
        return f"Digest({self.digest_date}, {len(self.papers)} papers)"