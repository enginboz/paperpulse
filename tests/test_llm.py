"""
Tests for the LLM scoring module.

Ollama API calls are mocked — no running Ollama instance required.
"""

import json
import pytest
from unittest.mock import patch
from paperpulse.scoring.llm import (
    _parse_indices,
    _parse_selections,
    _stage1_shortlist,
    _stage2_select,
    select_top_papers,
    STAGE1_CANDIDATES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_papers(n: int) -> list[dict]:
    """Generate a list of n fake paper dicts for testing."""
    return [
        {
            "pmid": str(i),
            "title": f"Paper {i} about clinical AI",
            "abstract": f"This is the abstract for paper {i}. " * 10,
            "journal": "npj Digital Medicine",
            "pub_date": "2026 Apr 21",
            "authors": ["John Smith"],
            "doi": f"10.1038/test{i}",
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{i}/",
        }
        for i in range(1, n + 1)
    ]


VALID_INDICES_RESPONSE = json.dumps(list(range(1, STAGE1_CANDIDATES + 1)))

VALID_SELECTIONS_RESPONSE = json.dumps([
    {"index": 1, "reason": "Relevant to clinical AI decision support."},
    {"index": 2, "reason": "Relevant to NLP on clinical notes."},
    {"index": 3, "reason": "Relevant to wearable monitoring."},
])


# ---------------------------------------------------------------------------
# _parse_indices tests
# ---------------------------------------------------------------------------

def test_parse_indices_valid():
    """Should parse a valid JSON array of integers."""
    response = "[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]"
    indices = _parse_indices(response, max_index=52)
    assert len(indices) == 15
    assert 1 in indices
    assert 29 in indices


def test_parse_indices_strips_markdown():
    """Should handle markdown code fences the model might add."""
    response = "```json\n[1, 2, 3]\n```"
    indices = _parse_indices(response, max_index=10)
    assert indices == [1, 2, 3]


def test_parse_indices_filters_out_of_range():
    """Indices outside the valid range should be filtered out."""
    response = "[1, 2, 999]"
    indices = _parse_indices(response, max_index=52)
    assert 999 not in indices
    assert 1 in indices
    assert 2 in indices


def test_parse_indices_filters_invalid_types():
    """Non-integer values should be filtered out."""
    response = '[1, "two", 3]'
    indices = _parse_indices(response, max_index=10)
    assert "two" not in indices
    assert 1 in indices
    assert 3 in indices


# ---------------------------------------------------------------------------
# _parse_selections tests
# ---------------------------------------------------------------------------

def test_parse_selections_valid():
    """Should parse valid selections and enrich papers with reason."""
    papers = make_papers(5)
    selections = _parse_selections(VALID_SELECTIONS_RESPONSE, papers)
    assert len(selections) == 3
    assert selections[0]["reason"] == "Relevant to clinical AI decision support."
    assert selections[1]["reason"] == "Relevant to NLP on clinical notes."


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
    assert selections[0]["reason"] == "Valid."


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
    assert "title" in selections[0]
    assert "abstract" in selections[0]
    assert "journal" in selections[0]
    assert "reason" in selections[0]


# ---------------------------------------------------------------------------
# _stage1_shortlist tests
# ---------------------------------------------------------------------------

@patch("paperpulse.scoring.llm._call_ollama")
def test_stage1_returns_correct_number(mock_ollama):
    """Stage 1 should return exactly STAGE1_CANDIDATES papers."""
    mock_ollama.return_value = VALID_INDICES_RESPONSE
    papers = make_papers(52)
    candidates = _stage1_shortlist(papers)
    assert len(candidates) == STAGE1_CANDIDATES


@patch("paperpulse.scoring.llm._call_ollama")
def test_stage1_fallback_on_parse_failure(mock_ollama):
    """Stage 1 should fall back to first N papers if parsing fails."""
    mock_ollama.return_value = "this is not valid json"
    papers = make_papers(52)
    with pytest.raises(json.JSONDecodeError):
        # The fallback only triggers within select_top_papers
        # _stage1_shortlist itself raises on JSON error
        _stage1_shortlist(papers)


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
        assert "reason" in paper
        assert len(paper["reason"]) > 0


# ---------------------------------------------------------------------------
# select_top_papers tests
# ---------------------------------------------------------------------------

def test_select_top_papers_empty_input():
    """Should return empty list for empty input without calling Ollama."""
    result = select_top_papers([])
    assert result == []


@patch("paperpulse.scoring.llm._call_ollama")
def test_select_top_papers_full_pipeline(mock_ollama):
    """Full pipeline should return 3 papers with reasons."""
    # Stage 1 returns indices, stage 2 returns selections
    mock_ollama.side_effect = [
        VALID_INDICES_RESPONSE,
        VALID_SELECTIONS_RESPONSE,
    ]
    papers = make_papers(52)
    result = select_top_papers(papers)
    assert len(result) == 3
    assert mock_ollama.call_count == 2


@patch("paperpulse.scoring.llm._call_ollama")
def test_select_top_papers_returns_empty_on_connection_error(mock_ollama):
    """Should return empty list if Ollama is unreachable."""
    import requests
    mock_ollama.side_effect = requests.exceptions.ConnectionError("refused")
    papers = make_papers(10)
    result = select_top_papers(papers)
    assert result == []


@patch("paperpulse.scoring.llm._call_ollama")
def test_select_top_papers_calls_ollama_twice(mock_ollama):
    """Pipeline should make exactly 2 calls to Ollama."""
    mock_ollama.side_effect = [
        VALID_INDICES_RESPONSE,
        VALID_SELECTIONS_RESPONSE,
    ]
    papers = make_papers(52)
    select_top_papers(papers)
    assert mock_ollama.call_count == 2