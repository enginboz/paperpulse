from datetime import date
from pathlib import Path

import httpx

from paperpulse.config import FilteredJournals, PubMedConfig
from paperpulse.sources.pubmed import PubMedSource, parse_pubmed_xml

FIXTURE = (Path(__file__).parent / "fixtures" / "pubmed_efetch.xml").read_text()


def test_parses_research_article():
    paper = parse_pubmed_xml(FIXTURE)[0]
    assert paper.id == "10.1038/s41746-026-00001-x"
    assert paper.doi == "10.1038/S41746-026-00001-X"
    assert paper.pmid == "40000001"
    assert paper.title == (
        "Large language models for FHIR resource extraction from discharge summaries."
    )
    assert paper.abstract == (
        "Clinical notes are unstructured. HbA1c values were mapped to FHIR resources."
    )
    assert paper.authors == ["Jane Doe", "Max Roe"]
    assert paper.published == date(2026, 9, 24)
    assert paper.added == date(2026, 9, 25)
    assert paper.publication_types == ["Journal Article"]
    assert paper.url == "https://pubmed.ncbi.nlm.nih.gov/40000001/"


def test_parses_editorial_without_doi_or_abstract():
    paper = parse_pubmed_xml(FIXTURE)[1]
    assert paper.id == "pmid:40000002"
    assert paper.doi is None
    assert paper.abstract == ""
    assert paper.published == date(2026, 9, 1)
    assert paper.publication_types == ["Editorial"]


def test_ignores_book_articles():
    assert [p.pmid for p in parse_pubmed_xml(FIXTURE)] == ["40000001", "40000002"]


def test_builds_one_query_per_journal_group():
    source = PubMedSource(
        PubMedConfig(
            journals=["npj Digital Medicine"],
            filtered=FilteredJournals(journals=["JAMA"], keywords=["machine learning", "EHR"]),
        )
    )
    focused, filtered = source.build_queries(date(2026, 9, 22), date(2026, 9, 29))
    window = '("2026/09/22"[EDAT] : "2026/09/29"[EDAT])'
    assert focused == f'("npj Digital Medicine"[Journal]) AND {window}'
    assert filtered == (
        '("JAMA"[Journal]) AND ("machine learning"[Title/Abstract] OR "EHR"[Title/Abstract]) '
        f"AND {window}"
    )


def test_filtered_group_needs_keywords():
    source = PubMedSource(PubMedConfig(filtered=FilteredJournals(journals=["JAMA"])))
    assert source.build_queries(date(2026, 9, 22), date(2026, 9, 29)) == []


def test_fetch_searches_then_fetches_details():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("esearch.fcgi"):
            return httpx.Response(200, json={"esearchresult": {"idlist": ["40000001", "40000002"]}})
        return httpx.Response(200, text=FIXTURE)

    source = PubMedSource(
        PubMedConfig(journals=["npj Digital Medicine"], email="me@example.org", api_key="k"),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    papers = list(source.fetch(date(2026, 9, 22), date(2026, 9, 29)))

    assert len(papers) == 2
    assert requests[1].url.params["id"] == "40000001,40000002"
    assert requests[1].url.params["email"] == "me@example.org"
    assert requests[1].url.params["api_key"] == "k"


def test_fetch_retries_on_rate_limit(monkeypatch):
    monkeypatch.setattr("paperpulse.sources.pubmed.time.sleep", lambda _: None)
    responses = iter(
        [
            httpx.Response(429),
            httpx.Response(200, json={"esearchresult": {"idlist": []}}),
        ]
    )
    source = PubMedSource(
        PubMedConfig(journals=["JAMA"]),
        client=httpx.Client(transport=httpx.MockTransport(lambda _: next(responses))),
    )
    assert list(source.fetch(date(2026, 9, 22), date(2026, 9, 29))) == []
