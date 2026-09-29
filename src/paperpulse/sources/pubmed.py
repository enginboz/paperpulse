"""
PubMed source via the NCBI E-utilities API.

Queries by Entrez date (when a record entered PubMed) rather than by
publication date: publication dates are often weeks in the future or past,
so a publication-date window silently skips papers between runs.

API docs: https://www.ncbi.nlm.nih.gov/books/NBK25501/
"""

import logging
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from datetime import date

import httpx

from paperpulse.config import PubMedConfig
from paperpulse.models import Paper, canonical_id

logger = logging.getLogger(__name__)

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
FETCH_BATCH = 200
MAX_RETRIES = 3

MONTHS = {
    m: i
    for i, m in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1
    )
}


class PubMedSource:
    name = "pubmed"

    def __init__(self, config: PubMedConfig, client: httpx.Client | None = None):
        self.config = config
        self.client = client or httpx.Client(timeout=30)
        # NCBI allows 3 requests/s without an API key and 10 with one.
        self._min_interval = 0.11 if config.api_key else 0.34
        self._last_request = 0.0

    def fetch(self, start: date, end: date) -> Iterator[Paper]:
        if not self.config.email:
            logger.warning("PUBMED_EMAIL is not set; NCBI asks clients to identify themselves.")

        pmids: list[str] = []
        for query in self.build_queries(start, end):
            found = self._search(query)
            logger.info("PubMed query matched %d records", len(found))
            pmids.extend(found)
        pmids = list(dict.fromkeys(pmids))

        for i in range(0, len(pmids), FETCH_BATCH):
            yield from parse_pubmed_xml(self._efetch(pmids[i : i + FETCH_BATCH]))

    def fetch_pmids(self, pmids: list[str]) -> list[Paper]:
        """Fetch specific records, e.g. to rebuild papers behind an imported label set."""
        papers = []
        for i in range(0, len(pmids), FETCH_BATCH):
            papers.extend(parse_pubmed_xml(self._efetch(pmids[i : i + FETCH_BATCH])))
        return papers

    def build_queries(self, start: date, end: date) -> list[str]:
        window = f'("{start:%Y/%m/%d}"[EDAT] : "{end:%Y/%m/%d}"[EDAT])'
        queries = []
        if self.config.journals:
            queries.append(f"({_any_of(self.config.journals, 'Journal')}) AND {window}")
        filtered = self.config.filtered
        if filtered.journals and filtered.keywords:
            queries.append(
                f"({_any_of(filtered.journals, 'Journal')}) "
                f"AND ({_any_of(filtered.keywords, 'Title/Abstract')}) AND {window}"
            )
        return queries

    def _search(self, query: str) -> list[str]:
        data = self._get("esearch.fcgi", {"term": query, "retmax": 10000, "retmode": "json"})
        return data.json()["esearchresult"]["idlist"]

    def _efetch(self, pmids: list[str]) -> str:
        return self._get("efetch.fcgi", {"id": ",".join(pmids), "retmode": "xml"}).text

    def _get(self, endpoint: str, params: dict) -> httpx.Response:
        params = {"db": "pubmed", "tool": "paperpulse", **params}
        if self.config.email:
            params["email"] = self.config.email
        if self.config.api_key:
            params["api_key"] = self.config.api_key

        for attempt in range(MAX_RETRIES):
            wait = self._min_interval - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()
            try:
                response = self.client.get(f"{BASE_URL}/{endpoint}", params=params)
                if response.status_code != 429 and response.status_code < 500:
                    response.raise_for_status()
                    return response
                logger.warning("PubMed returned %d, retrying", response.status_code)
            except httpx.TransportError as e:
                logger.warning("PubMed request failed (%s), retrying", e)
            time.sleep(2**attempt)
        raise RuntimeError(f"PubMed {endpoint} failed after {MAX_RETRIES} attempts")


def _any_of(terms: list[str], field: str) -> str:
    return " OR ".join(f'"{t}"[{field}]' for t in terms)


def parse_pubmed_xml(xml_text: str) -> list[Paper]:
    papers = []
    for article in ET.fromstring(xml_text).iter("PubmedArticle"):
        pmid = article.findtext(".//MedlineCitation/PMID")
        try:
            papers.append(_parse_article(article))
        except Exception:
            logger.warning("Skipping unparseable PubMed record %s", pmid, exc_info=True)
    return papers


def _parse_article(article: ET.Element) -> Paper:
    pmid = article.findtext(".//MedlineCitation/PMID")
    doi = next(
        (
            el.text
            for el in article.iterfind(".//PubmedData/ArticleIdList/ArticleId")
            if el.get("IdType") == "doi" and el.text
        ),
        None,
    )
    authors = [
        f"{a.findtext('ForeName', '')} {a.findtext('LastName')}".strip()
        for a in article.iterfind(".//AuthorList/Author")
        if a.findtext("LastName")
    ]
    return Paper(
        id=canonical_id(doi, pmid),
        title=_text(article.find(".//ArticleTitle")),
        abstract=" ".join(_text(el) for el in article.iterfind(".//Abstract/AbstractText")),
        journal=article.findtext(".//Journal/Title", "").strip(),
        published=_published(article),
        added=_entrez_date(article) or date.today(),
        authors=authors,
        publication_types=[_text(el) for el in article.iterfind(".//PublicationType")],
        doi=doi,
        pmid=pmid,
        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        source="pubmed",
    )


def _text(el: ET.Element | None) -> str:
    """Element text including inline markup such as <i> or <sup>."""
    return " ".join("".join(el.itertext()).split()) if el is not None else ""


def _published(article: ET.Element) -> date | None:
    """Prefer the electronic publication date; journal issue dates are often coarse or future."""
    for path in (".//ArticleDate", ".//JournalIssue/PubDate"):
        if (parsed := _date(article.find(path))) is not None:
            return parsed
    return None


def _entrez_date(article: ET.Element) -> date | None:
    return _date(article.find(".//PubmedData/History/PubMedPubDate[@PubStatus='entrez']"))


def _date(el: ET.Element | None) -> date | None:
    if el is None or not el.findtext("Year"):
        return None
    month_raw = el.findtext("Month", "1")
    month = int(month_raw) if month_raw.isdigit() else MONTHS.get(month_raw[:3].lower(), 1)
    return date(int(el.findtext("Year")), month, int(el.findtext("Day", "1")))
