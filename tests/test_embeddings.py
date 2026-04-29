"""
Tests for the embedding-based pre-filtering module.

The sentence-transformers model is mocked to avoid downloading
and running a large model during testing.
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from paperpulse.models import Paper
from paperpulse.scoring.embeddings import (
    _cosine_similarity,
    _paper_text,
    shortlist_by_embedding,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_papers(n: int) -> list[Paper]:
    """Generate a list of n fake Paper objects for testing."""
    return [
        Paper(
            pmid=str(i),
            title=f"Paper {i} about clinical AI",
            abstract=f"This study evaluates an AI model for clinical use case {i}. " * 5,
            journal="npj Digital Medicine",
            pub_date="2026 Apr 21",
            authors=["John Smith"],
            doi=f"10.1038/test{i}",
            url=f"https://pubmed.ncbi.nlm.nih.gov/{i}/",
        )
        for i in range(1, n + 1)
    ]


def make_mock_model(n_papers: int, dims: int = 384):
    """
    Create a mock SentenceTransformer that returns deterministic vectors.
    First call (profile) returns a unit vector along dimension 0.
    Subsequent calls (papers) return vectors with varying similarity.
    """
    mock = MagicMock()

    call_count = [0]

    def fake_encode(texts, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            # Interest profile vector — unit vector along dim 0
            vec = np.zeros(dims)
            vec[0] = 1.0
            return vec
        else:
            # Paper vectors — first paper most similar, last least similar
            n = len(texts)
            vecs = np.zeros((n, dims))
            for i in range(n):
                vecs[i, 0] = 1.0 - (i * 0.05)  # decreasing similarity
                vecs[i, 1] = i * 0.05
            return vecs

    mock.encode.side_effect = fake_encode
    return mock


# ---------------------------------------------------------------------------
# _cosine_similarity tests
# ---------------------------------------------------------------------------

def test_cosine_similarity_identical_vectors():
    """Identical vectors should have similarity of 1.0."""
    vec = np.array([1.0, 0.0, 0.0])
    assert abs(_cosine_similarity(vec, vec) - 1.0) < 1e-6


def test_cosine_similarity_orthogonal_vectors():
    """Orthogonal vectors should have similarity of 0.0."""
    vec_a = np.array([1.0, 0.0, 0.0])
    vec_b = np.array([0.0, 1.0, 0.0])
    assert abs(_cosine_similarity(vec_a, vec_b)) < 1e-6


def test_cosine_similarity_opposite_vectors():
    """Opposite vectors should have similarity of -1.0."""
    vec_a = np.array([1.0, 0.0, 0.0])
    vec_b = np.array([-1.0, 0.0, 0.0])
    assert abs(_cosine_similarity(vec_a, vec_b) + 1.0) < 1e-6


def test_cosine_similarity_returns_float():
    """Result should be a Python float."""
    vec = np.array([1.0, 0.0, 0.0])
    result = _cosine_similarity(vec, vec)
    assert isinstance(result, float)


# ---------------------------------------------------------------------------
# _paper_text tests
# ---------------------------------------------------------------------------

def test_paper_text_combines_title_and_abstract():
    """Should combine title and abstract into a single string."""
    paper = Paper(pmid="1", title="AI in Medicine", abstract="This study evaluates AI.",
                  journal="", pub_date="", authors=[], doi="", url="")
    text = _paper_text(paper)
    assert "AI in Medicine" in text
    assert "This study evaluates AI." in text


def test_paper_text_title_comes_first():
    """Title should appear before abstract in the combined text."""
    paper = Paper(pmid="1", title="Title", abstract="Abstract",
                  journal="", pub_date="", authors=[], doi="", url="")
    text = _paper_text(paper)
    assert text.index("Title") < text.index("Abstract")


def test_paper_text_truncates_long_abstract():
    """Abstract should be truncated to 1000 characters."""
    paper = Paper(pmid="1", title="Title", abstract="x" * 2000,
                  journal="", pub_date="", authors=[], doi="", url="")
    text = _paper_text(paper)
    assert len(text) < 1100  # title + separator + 1000 chars


def test_paper_text_handles_missing_fields():
    """Should handle papers with empty title or abstract gracefully."""
    paper = Paper(pmid="1", title="", abstract="",
                  journal="", pub_date="", authors=[], doi="", url="")
    text = _paper_text(paper)
    assert isinstance(text, str)


# ---------------------------------------------------------------------------
# shortlist_by_embedding tests
# ---------------------------------------------------------------------------

def test_shortlist_returns_empty_for_empty_input():
    """Should return empty list for empty input without loading model."""
    result = shortlist_by_embedding([])
    assert result == []


@patch("paperpulse.scoring.embeddings._get_model")
def test_shortlist_returns_correct_number(mock_get_model):
    """Should return exactly n papers."""
    papers = make_papers(20)
    mock_get_model.return_value = make_mock_model(20)

    result = shortlist_by_embedding(papers, n=10)
    assert len(result) == 10


@patch("paperpulse.scoring.embeddings._get_model")
def test_shortlist_returns_all_if_fewer_than_n(mock_get_model):
    """Should return all papers if there are fewer than n."""
    papers = make_papers(5)
    mock_get_model.return_value = make_mock_model(5)

    result = shortlist_by_embedding(papers, n=15)
    assert len(result) == 5


@patch("paperpulse.scoring.embeddings._get_model")
def test_shortlist_sorted_by_score_descending(mock_get_model):
    """Papers should be sorted by embedding score, highest first."""
    papers = make_papers(10)
    mock_get_model.return_value = make_mock_model(10)

    result = shortlist_by_embedding(papers, n=10)
    scores = [p.embedding_score for p in result]
    assert scores == sorted(scores, reverse=True)


@patch("paperpulse.scoring.embeddings._get_model")
def test_shortlist_adds_embedding_score(mock_get_model):
    """Each returned paper should have an embedding_score field."""
    papers = make_papers(5)
    mock_get_model.return_value = make_mock_model(5)

    result = shortlist_by_embedding(papers, n=5)
    for paper in result:
        assert isinstance(paper.embedding_score, float)


@patch("paperpulse.scoring.embeddings._get_model")
def test_shortlist_preserves_original_fields(mock_get_model):
    """Original paper fields should be preserved in results."""
    papers = make_papers(5)
    mock_get_model.return_value = make_mock_model(5)

    result = shortlist_by_embedding(papers, n=5)
    for paper in result:
        assert paper.pmid != ""
        assert paper.title != ""
        assert paper.abstract != ""
        assert paper.journal != ""