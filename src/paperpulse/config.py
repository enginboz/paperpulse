"""
Configuration, loaded from a TOML file (default: ./paperpulse.toml).

Secrets and machine-specific values come from environment variables so
the TOML file can be shared or committed:

    PUBMED_EMAIL    contact email sent with NCBI requests (they ask for one)
    NCBI_API_KEY    optional, raises the NCBI rate limit from 3 to 10 req/s
"""

import os
import tomllib
from pathlib import Path

from pydantic import BaseModel, Field

# Publication types that are never research papers worth a slot in the digest.
DEFAULT_EXCLUDED_TYPES = [
    "Comment",
    "Editorial",
    "Erratum",
    "Letter",
    "News",
    "Published Erratum",
    "Retraction Notice",
    "Retraction of Publication",
    "Expression of Concern",
    "Biography",
    "Portrait",
]


class Profile(BaseModel):
    topics: list[str] = Field(min_length=1)


class FilteredJournals(BaseModel):
    """High-volume journals where only papers matching a keyword are ingested."""

    journals: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


class PubMedConfig(BaseModel):
    journals: list[str] = Field(default_factory=list)
    filtered: FilteredJournals = Field(default_factory=FilteredJournals)
    email: str = Field(default_factory=lambda: os.getenv("PUBMED_EMAIL", ""))
    api_key: str = Field(default_factory=lambda: os.getenv("NCBI_API_KEY", ""))


class Sources(BaseModel):
    pubmed: PubMedConfig = Field(default_factory=PubMedConfig)


class Selection(BaseModel):
    top: int = 3
    window_days: int = 7
    no_repeat_days: int = 7
    exclude_publication_types: list[str] = Field(
        default_factory=lambda: list(DEFAULT_EXCLUDED_TYPES)
    )
    require_abstract: bool = True


class Embedding(BaseModel):
    model: str = "pritamdeka/S-PubMedBert-MS-MARCO"


class Config(BaseModel):
    profile: Profile
    sources: Sources = Field(default_factory=Sources)
    selection: Selection = Field(default_factory=Selection)
    embedding: Embedding = Field(default_factory=Embedding)


def load_config(path: Path) -> Config:
    with path.open("rb") as f:
        return Config.model_validate(tomllib.load(f))
