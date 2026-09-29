import hashlib
import re
from datetime import date

import numpy as np
import pytest

from paperpulse.config import Config
from paperpulse.models import Paper, canonical_id
from paperpulse.store import Store


class FakeEmbedder:
    """Bag-of-words hashing: texts sharing words get similar vectors. Deterministic, no download."""

    name = "fake-bow"
    dim = 256

    def __init__(self):
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls.append(texts)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z]+", text.lower()):
                out[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


class FakeLLM:
    """Rates a paper by looking up its title in `ratings`; unknown titles get 1."""

    name = "fake-llm"

    def __init__(self, ratings: dict[str, int] | None = None, responses=None):
        self.ratings = ratings or {}
        self.responses = responses  # optional iterator of raw responses, overrides ratings
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, output):
        self.calls.append((system, user))
        if self.responses is not None:
            response = next(self.responses)
            if isinstance(response, Exception):
                raise response
            return output.model_validate(response)
        title = user.splitlines()[0].removeprefix("Title: ")
        return output(
            study_type="original_research",
            rationale=f"Rated {title}.",
            relevance=self.ratings.get(title, 1),
        )


def make_paper(n: int, title: str = "", abstract: str = "abstract", **kwargs) -> Paper:
    defaults = dict(
        id=canonical_id(None, str(n)),
        title=title or f"Paper {n}",
        abstract=abstract,
        journal="J Test",
        added=date(2026, 9, 28),
        pmid=str(n),
        url=f"https://pubmed.ncbi.nlm.nih.gov/{n}/",
        source="test",
    )
    return Paper(**{**defaults, **kwargs})


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def llm():
    return FakeLLM()


@pytest.fixture
def config():
    return Config.model_validate(
        {
            "profile": {"topics": ["fhir interoperability", "clinical notes nlp"]},
            "selection": {"top": 2},
        }
    )
