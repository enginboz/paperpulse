"""
Tests for the LLM scoring module.

Ollama API calls are mocked — no running Ollama instance required.
Embedding stage is also mocked to isolate LLM behavior.
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from paperpulse.models import Paper
from paperpulse.scoring.llm import (
    _parse_selections,
    _stage2_select,
    select_top_papers,
    STAGE1_CANDIDATES,
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
            abstract=f"This is the abstract for paper {i}. " * 10,
            journal="npj Digital Medicine",
            pub_date="2026 Apr 21",
            authors=["John Smith"],
            doi=f"10.1038/test{i}",
            url=f"https://pubmed.ncbi.nlm.nih.gov/{i}/",
            embedding_score=0.9 - (i * 0.01),
        )
        for i in range(1, n + 1)
    ]


VALID_SELECTIONS_RESPONSE = json.dumps([
    {"index": 1, "reason": "Relevant to clinical AI decision support."},
    {"index": 2, "reason": "Relevant to NLP on clinical notes."},
    {"index": 3, "reason": "Relevant to wearable monitoring."},
])


# ---------------------------------------------------------------------------
# _parse_selections tests
# ---------------------------------------------------------------------------

def test_parse_selections_valid():
    """Should parse valid selections and set reason on papers."""
    papers = make_papers(5)
    selections = _parse_selections(VALID_SELECTIONS_RESPONSE, papers)
    assert len(selections) == 3
    assert selections[0].reason == "Relevant to clinical AI decision support."
    assert selections[1].reason == "Relevant to NLP on clinical notes."


def test_parse_selections_strips_markdown():
    """Should handle markdown code fences."""
    papers = make_papers(5)
    response = f"```json\n{VALID_SELECTIONS_RESPONSE}\n```"
    selections = _parse_selections(response, papers)
    assert len(selections) == 3


def test_parse_selections_skips_invalid_indices():
    """Selections with invalid indices should be skipped."""
    papers = make_papers(3)
    response = json.dumps([
        {"index": 1, "reason": "Valid."},
        {"index": 99, "reason": "Invalid index."},
    ])
    selections = _parse_selections(response, papers)
    assert len(selections) == 1
    assert selections[0].reason == "Valid."


def test_parse_selections_max_three():
    """Should return at most 3 selections even if more are provided."""
    papers = make_papers(10)
    response = json.dumps([
        {"index": i, "reason": f"Reason {i}."}
        for i in range(1, 6)
    ])
    selections = _parse_selections(response, papers)
    assert len(selections) == 3


def test_parse_selections_preserves_paper_fields():
    """Selected papers should retain all original fields plus reason."""
    papers = make_papers(3)
    selections = _parse_selections(VALID_SELECTIONS_RESPONSE, papers)
    assert selections[0].title != ""
    assert selections[0].abstract != ""
    assert selections[0].journal != ""
    assert selections[0].reason != ""


# ---------------------------------------------------------------------------
# _stage2_select tests
# ---------------------------------------------------------------------------

@patch("paperpulse.scoring.llm._call_ollama")
def test_stage2_returns_three_papers(mock_ollama):
    """Stage 2 should return exactly 3 papers."""
    mock_ollama.return_value = VALID_SELECTIONS_RESPONSE
    candidates = make_papers(15)
    selected = _stage2_select(candidates)
    assert len(selected) == 3


@patch("paperpulse.scoring.llm._call_ollama")
def test_stage2_papers_have_reason(mock_ollama):
    """Each selected paper should have a reason field."""
    mock_ollama.return_value = VALID_SELECTIONS_RESPONSE
    candidates = make_papers(15)
    selected = _stage2_select(candidates)
    for paper in selected:
        assert paper.reason != ""


# ---------------------------------------------------------------------------
# select_top_papers tests
# ---------------------------------------------------------------------------

def test_select_top_papers_empty_input():
    """Should return empty list for empty input without calling Ollama."""
    result = select_top_papers([])
    assert result == []


@patch("paperpulse.scoring.embeddings.shortlist_by_embedding")
@patch("paperpulse.scoring.llm._call_ollama")
def test_select_top_papers_full_pipeline(mock_ollama, mock_embed):
    """Full pipeline should return 3 papers with reasons."""
    mock_embed.return_value = make_papers(STAGE1_CANDIDATES)
    mock_ollama.return_value = VALID_SELECTIONS_RESPONSE

    papers = make_papers(52)
    result = select_top_papers(papers)

    assert len(result) == 3
    mock_embed.assert_called_once()
    mock_ollama.assert_called_once()


@patch("paperpulse.scoring.embeddings.shortlist_by_embedding")
@patch("paperpulse.scoring.llm._call_ollama")
def test_select_top_papers_returns_empty_on_connection_error(mock_ollama, mock_embed):
    """Should return empty list if Ollama is unreachable."""
    import requests
    mock_embed.return_value = make_papers(STAGE1_CANDIDATES)
    mock_ollama.side_effect = requests.exceptions.ConnectionError("refused")

    papers = make_papers(10)
    result = select_top_papers(papers)
    assert result == []


@patch("paperpulse.scoring.embeddings.shortlist_by_embedding")
@patch("paperpulse.scoring.llm._call_ollama")
def test_select_top_papers_calls_ollama_once(mock_ollama, mock_embed):
    """Pipeline should make exactly 1 call to Ollama (stage 2 only)."""
    mock_embed.return_value = make_papers(STAGE1_CANDIDATES)
    mock_ollama.return_value = VALID_SELECTIONS_RESPONSE

    papers = make_papers(52)
    select_top_papers(papers)
    assert mock_ollama.call_count == 1