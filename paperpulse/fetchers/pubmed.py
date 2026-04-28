"""
PubMed fetcher module.

Queries the PubMed E-utilities API for recent papers from a curated
journal whitelist and returns structured paper data.

API docs: https://www.ncbi.nlm.nih.gov/books/NBK25501/
No API key required for low-volume usage (<3 requests/second).
"""

import os
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from time import sleep

import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PUBMED_EMAIL = os.getenv("PUBMED_EMAIL", "")
BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Focused journals — dedicated to digital medicine and clinical informatics.
# All papers from these journals are fetched without topic filtering.
FOCUSED_JOURNALS = [
    "npj Digital Medicine",
    "Lancet Digital Health",
    "Journal of Medical Internet Research",
    "Journal of the American Medical Informatics Association",
    "Journal of Biomedical Informatics",
    "Patterns",
    "PLOS Digital Health",
]

# Broad journals — high-impact but cover all of medicine.
# Only papers matching BROAD_JOURNAL_KEYWORDS are fetched from these.
BROAD_JOURNALS = [
    "Nature Medicine",
    "New England Journal of Medicine",
    "JAMA",
]

# Keywords used to filter papers from broad journals.
# A paper must match at least one of these terms to be included.
BROAD_JOURNAL_KEYWORDS = [
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "large language model",
    "natural language processing",
    "digital health",
    "clinical decision support",
    "wearable",
    "electronic health record",
    "EHR",
    "mHealth",
    "telemedicine",
    "digital therapeutics",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _date_range(days: int) -> tuple[str, str]:
    """Return (date_from, date_to) strings for the PubMed date filter."""
    date_from = (datetime.now() - timedelta(days=days)).strftime("%Y/%m/%d")
    date_to = datetime.now().strftime("%Y/%m/%d")
    return date_from, date_to


def _build_focused_query(days: int) -> str:
    """
    Build a query for focused journals — no topic filtering applied.
    All papers from these journals within the date range are included.
    """
    date_from, date_to = _date_range(days)
    journal_filter = " OR ".join(
        [f'"{j}"[Journal]' for j in FOCUSED_JOURNALS]
    )
    return f"({journal_filter}) AND ({date_from}[PDAT] : {date_to}[PDAT])"


def _build_broad_query(days: int) -> str:
    """
    Build a query for broad journals — only papers matching at least
    one topic keyword are included, to filter out off-topic content.
    """
    date_from, date_to = _date_range(days)
    journal_filter = " OR ".join(
        [f'"{j}"[Journal]' for j in BROAD_JOURNALS]
    )
    keyword_filter = " OR ".join(
        [f'"{kw}"[Title/Abstract]' for kw in BROAD_JOURNAL_KEYWORDS]
    )
    return f"({journal_filter}) AND ({keyword_filter}) AND ({date_from}[PDAT] : {date_to}[PDAT])"


def _search(query: str, max_results: int) -> list[str]:
    """
    Run an esearch query and return a list of PubMed IDs (PMIDs).
    """
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "tool": "paperpulse",
        "email": PUBMED_EMAIL,
    }

    response = requests.get(f"{BASE_URL}/esearch.fcgi", params=params, timeout=10)
    response.raise_for_status()

    ids = response.json().get("esearchresult", {}).get("idlist", [])
    return ids


def _fetch_details(pubmed_ids: list[str]) -> list[dict]:
    """
    Fetch full article details for a list of PMIDs via efetch
    and return parsed paper dicts.
    """
    params = {
        "db": "pubmed",
        "id": ",".join(pubmed_ids),
        "retmode": "xml",
        "rettype": "abstract",
        "tool": "paperpulse",
        "email": PUBMED_EMAIL,
    }

    response = requests.get(f"{BASE_URL}/efetch.fcgi", params=params, timeout=30)
    response.raise_for_status()

    return _parse_xml(response.text)


def _parse_xml(xml_text: str) -> list[dict]:
    """
    Parse PubMed XML response into a list of structured paper dicts.
    Papers without an abstract are skipped (editorials, letters, etc.).
    """
    root = ET.fromstring(xml_text)
    papers = []

    for article in root.findall(".//PubmedArticle"):
        try:
            # Title
            title_el = article.find(".//ArticleTitle")
            title = "".join(title_el.itertext()).strip() if title_el is not None else ""

            # Abstract — may have multiple labelled sections (background, methods, etc.)
            abstract_parts = article.findall(".//AbstractText")
            abstract = " ".join(
                "".join(el.itertext()).strip() for el in abstract_parts
            ).strip()

            # Skip papers without an abstract
            if not abstract:
                continue

            # Journal name
            journal_el = article.find(".//Journal/Title")
            journal = journal_el.text.strip() if journal_el is not None else ""

            # Publication date
            pub_date_el = article.find(".//PubDate")
            pub_date = ""
            if pub_date_el is not None:
                year = pub_date_el.findtext("Year", "")
                month = pub_date_el.findtext("Month", "")
                day = pub_date_el.findtext("Day", "")
                pub_date = " ".join(filter(None, [year, month, day]))

            # Authors — last name + fore name
            authors = []
            for author in article.findall(".//Author"):
                last = author.findtext("LastName", "")
                fore = author.findtext("ForeName", "")
                if last:
                    authors.append(f"{fore} {last}".strip())

            # PubMed ID
            pmid_el = article.find(".//PMID")
            pmid = pmid_el.text.strip() if pmid_el is not None else ""

            # DOI — needed for Altmetric enrichment later
            doi = ""
            for id_el in article.findall(".//ArticleId"):
                if id_el.get("IdType") == "doi":
                    doi = id_el.text.strip()
                    break

            papers.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "journal": journal,
                "pub_date": pub_date,
                "authors": authors,
                "doi": doi,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
            })

        except Exception as e:
            # Skip malformed entries without crashing the whole fetch
            print(f"Warning: could not parse article — {e}")
            continue

    return papers


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_recent_papers(days: int = 7, max_results: int = 100) -> list[dict]:
    """
    Fetch recent papers from whitelisted journals.

    Focused journals (npj Digital Medicine, Lancet Digital Health, etc.)
    are fetched without topic filtering. Broad journals (Nature Medicine,
    NEJM, JAMA) are filtered by topic keywords to exclude off-topic content.

    Args:
        days: How many days back to search (default: 7).
        max_results: Maximum number of PMIDs to fetch per query (default: 100).

    Returns:
        List of paper dicts with keys:
        pmid, title, abstract, journal, pub_date, authors, doi, url
    """
    if not PUBMED_EMAIL:
        print("Warning: PUBMED_EMAIL not set in .env — PubMed requests an email for polite usage.")

    # Query 1 — focused journals, no topic filter
    focused_query = _build_focused_query(days)
    print(f"Querying focused journals...")
    focused_ids = _search(focused_query, max_results)
    print(f"Found {len(focused_ids)} papers from focused journals.")

    sleep(0.4)  # Be polite to the PubMed API

    # Query 2 — broad journals, topic filtered
    broad_query = _build_broad_query(days)
    print(f"Querying broad journals with topic filter...")
    broad_ids = _search(broad_query, max_results)
    print(f"Found {len(broad_ids)} papers from broad journals.")

    # Merge and deduplicate by PMID
    all_ids = list(dict.fromkeys(focused_ids + broad_ids))
    print(f"Total unique papers to fetch: {len(all_ids)}")

    if not all_ids:
        return []

    sleep(0.4)

    papers = _fetch_details(all_ids)
    print(f"Parsed {len(papers)} papers with abstracts.")

    return papers