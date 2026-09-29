"""
Domain models.

Paper is what sources produce and the store persists. Digest is the
public output contract: it is serialised to JSON and described by a
versioned JSON schema (`paperpulse schema`).
"""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1"


class Paper(BaseModel):
    """A paper normalised from any source. `id` is the DOI when known, else `pmid:<n>`."""

    id: str
    title: str
    abstract: str = ""
    journal: str = ""
    published: date | None = None
    added: date = Field(description="When the paper appeared in the source; drives time windows")
    authors: list[str] = Field(default_factory=list)
    publication_types: list[str] = Field(default_factory=list)
    doi: str | None = None
    pmid: str | None = None
    url: str = ""
    source: str

    @property
    def text(self) -> str:
        """The text that gets embedded. The title comes first because it is the densest signal."""
        return f"{self.title}. {self.abstract}".strip()


def canonical_id(doi: str | None, pmid: str | None) -> str:
    """DOIs are case-insensitive, so they are lowercased to deduplicate across sources."""
    if doi:
        return doi.strip().lower()
    if pmid:
        return f"pmid:{pmid}"
    raise ValueError("a paper needs a DOI or a PMID")


StudyType = Literal[
    "original_research",
    "systematic_review",
    "narrative_review",
    "qualitative_or_survey",
    "perspective_or_commentary",
    "protocol",
    "other",
]


class Assessment(BaseModel):
    """
    An LLM's judgement of one paper. Field order is generation order,
    so the model states its reasoning before committing to a score.
    """

    study_type: StudyType
    rationale: str = Field(description="One sentence on why this paper does or does not matter")
    relevance: int = Field(ge=1, le=5)


class Score(BaseModel):
    """Why a paper ranked where it did."""

    total: float = Field(
        description="LLM relevance (1-5) plus dense similarity as tie-breaker; "
        "dense similarity alone when the LLM stage did not run"
    )
    dense: float = Field(description="Cosine similarity to the best-matching topic")
    matched_topic: str
    assessment: Assessment | None = None


class RankedPaper(BaseModel):
    rank: int
    id: str
    title: str
    journal: str
    published: date | None
    authors: list[str]
    doi: str | None
    pmid: str | None
    url: str
    score: Score


class RunInfo(BaseModel):
    """Everything needed to understand or reproduce a selection."""

    id: str
    created_at: datetime
    window_start: date
    window_end: date
    embedding_model: str
    llm_model: str | None = Field(default=None, description="None when the LLM stage was skipped")
    candidates: int = Field(description="Papers in the time window before filtering")
    after_filters: int = Field(description="Papers left after the hard filters")


class Digest(BaseModel):
    schema_version: str = SCHEMA_VERSION
    run: RunInfo
    papers: list[RankedPaper]
