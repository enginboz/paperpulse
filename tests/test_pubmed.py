"""
Tests for the PubMed fetcher module.

API calls are mocked — no network connection required.
"""

import pytest
from unittest.mock import patch, MagicMock
from paperpulse.fetchers.pubmed import (
    _build_focused_query,
    _build_broad_query,
    _parse_xml,
    fetch_recent_papers,
    FOCUSED_JOURNALS,
    BROAD_JOURNALS,
    BROAD_JOURNAL_KEYWORDS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_XML = """<?xml version="1.0" ?>
<!DOCTYPE PubmedArticleSet PUBLIC "-//NLM//DTD PubMedArticle, 1st January 2019//EN"
  "https://dtd.nlm.nih.gov/ncbi/pubmed/out/pubmed_190101.dtd">
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID Version="1">12345678</PMID>
      <Article>
        <Journal>
          <Title>npj Digital Medicine</Title>
        </Journal>
        <ArticleTitle>AI-assisted diagnosis in clinical practice.</ArticleTitle>
        <Abstract>
          <AbstractText>This study evaluates an AI model for clinical diagnosis.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author>
            <LastName>Smith</LastName>
            <ForeName>John</ForeName>
          </Author>
          <Author>
            <LastName>Doe</LastName>
            <ForeName>Jane</ForeName>
          </Author>
        </AuthorList>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <History>
        <PubMedPubDate PubStatus="pubmed">
          <Year>2026</Year>
          <Month>Apr</Month>
          <Day>21</Day>
        </PubMedPubDate>
      </History>
      <ArticleIdList>
        <ArticleId IdType="pubmed">12345678</ArticleId>
        <ArticleId IdType="doi">10.1038/s41746-026-00001-1</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID Version="1">99999999</PMID>
      <Article>
        <Journal>
          <Title>npj Digital Medicine</Title>
        </Journal>
        <ArticleTitle>A paper without an abstract.</ArticleTitle>
        <AuthorList>
          <Author>
            <LastName>Ghost</LastName>
            <ForeName>Author</ForeName>
          </Author>
        </AuthorList>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="pubmed">99999999</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>
"""

EMPTY_SEARCH_RESPONSE = {
    "esearchresult": {
        "idlist": []
    }
}

SAMPLE_SEARCH_RESPONSE = {
    "esearchresult": {
        "idlist": ["12345678"]
    }
}


# ---------------------------------------------------------------------------
# Query builder tests
# ---------------------------------------------------------------------------

def test_focused_query_contains_all_focused_journals():
    """All focused journals should appear in the focused query."""
    query = _build_focused_query(days=7)
    for journal in FOCUSED_JOURNALS:
        assert journal in query


def test_focused_query_excludes_broad_journals():
    """Broad journals should not appear in the focused query."""
    query = _build_focused_query(days=7)
    for journal in BROAD_JOURNALS:
        assert journal not in query


def test_focused_query_contains_date_filter():
    """Focused query should contain a PubMed date filter."""
    query = _build_focused_query(days=7)
    assert "[PDAT]" in query


def test_broad_query_contains_all_broad_journals():
    """All broad journals should appear in the broad query."""
    query = _build_broad_query(days=7)
    for journal in BROAD_JOURNALS:
        assert journal in query


def test_broad_query_contains_keywords():
    """Broad query should contain at least some topic keywords."""
    query = _build_broad_query(days=7)
    # At least one keyword should appear in the query
    assert any(kw in query for kw in BROAD_JOURNAL_KEYWORDS)


def test_broad_query_excludes_focused_journals():
    """Focused journals should not appear in the broad query."""
    query = _build_broad_query(days=7)
    for journal in FOCUSED_JOURNALS:
        assert journal not in query


# ---------------------------------------------------------------------------
# XML parser tests
# ---------------------------------------------------------------------------

def test_parse_xml_returns_correct_number_of_papers():
    """Parser should return only papers that have an abstract."""
    papers = _parse_xml(SAMPLE_XML)
    # SAMPLE_XML has 2 articles but only 1 has an abstract
    assert len(papers) == 1


def test_parse_xml_correct_title():
    papers = _parse_xml(SAMPLE_XML)
    assert papers[0].title == "AI-assisted diagnosis in clinical practice."


def test_parse_xml_correct_pmid():
    papers = _parse_xml(SAMPLE_XML)
    assert papers[0].pmid == "12345678"


def test_parse_xml_correct_doi():
    papers = _parse_xml(SAMPLE_XML)
    assert papers[0].doi == "10.1038/s41746-026-00001-1"


def test_parse_xml_correct_authors():
    papers = _parse_xml(SAMPLE_XML)
    assert "John Smith" in papers[0].authors
    assert "Jane Doe" in papers[0].authors


def test_parse_xml_correct_journal():
    papers = _parse_xml(SAMPLE_XML)
    assert papers[0].journal == "npj Digital Medicine"


def test_parse_xml_correct_url():
    papers = _parse_xml(SAMPLE_XML)
    assert papers[0].url == "https://pubmed.ncbi.nlm.nih.gov/12345678/"


def test_parse_xml_skips_papers_without_abstract():
    """Papers without an abstract should be silently skipped."""
    papers = _parse_xml(SAMPLE_XML)
    pmids = [p.pmid for p in papers]
    assert "99999999" not in pmids


def test_parse_xml_empty_input():
    """Parser should return an empty list for an empty article set."""
    empty_xml = """<?xml version="1.0" ?>
    <PubmedArticleSet></PubmedArticleSet>"""
    papers = _parse_xml(empty_xml)
    assert papers == []


# ---------------------------------------------------------------------------
# fetch_recent_papers tests
# ---------------------------------------------------------------------------

@patch("paperpulse.fetchers.pubmed._fetch_details")
@patch("paperpulse.fetchers.pubmed._search")
def test_fetch_recent_papers_returns_empty_on_no_results(mock_search, mock_fetch):
    """Should return an empty list when PubMed finds no papers."""
    mock_search.return_value = []
    papers = fetch_recent_papers()
    assert papers == []
    mock_fetch.assert_not_called()


@patch("paperpulse.fetchers.pubmed.sleep")
@patch("paperpulse.fetchers.pubmed._fetch_details")
@patch("paperpulse.fetchers.pubmed._search")
def test_fetch_recent_papers_deduplicates_ids(mock_search, mock_fetch, mock_sleep):
    """
    If a PMID appears in both focused and broad query results,
    it should only be fetched once.
    """
    # Same PMID returned by both queries
    mock_search.return_value = ["12345678"]
    mock_fetch.return_value = [{"pmid": "12345678", "title": "Test"}]

    fetch_recent_papers()

    # _fetch_details should be called with deduplicated IDs
    called_ids = mock_fetch.call_args[0][0]
    assert called_ids.count("12345678") == 1


@patch("paperpulse.fetchers.pubmed.sleep")
@patch("paperpulse.fetchers.pubmed._fetch_details")
@patch("paperpulse.fetchers.pubmed._search")
def test_fetch_recent_papers_runs_two_queries(mock_search, mock_fetch, mock_sleep):
    """fetch_recent_papers should always run exactly two PubMed queries."""
    mock_search.return_value = []
    fetch_recent_papers()
    assert mock_search.call_count == 2