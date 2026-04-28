"""
Tests for the database module.

Uses an in-memory SQLite database to avoid needing a real PostgreSQL
instance during testing. SQLAlchemy's ORM works identically across both.
"""

import pytest
from datetime import date
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from paperpulse.db import (
    Base,
    Paper,
    DailyDigest,
    DigestPaper,
    upsert_paper,
    save_digest,
    get_latest_digest,
    get_digest_for_date,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    """Create an in-memory SQLite engine for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)


@pytest.fixture
def session(engine):
    """Provide a database session for each test."""
    with Session(engine) as session:
        yield session


@pytest.fixture(autouse=True)
def patch_engine(engine, monkeypatch):
    """
    Patch the module-level engine so save_digest and get_latest_digest
    use the in-memory test engine instead of the real PostgreSQL one.
    """
    import paperpulse.db as db_module
    monkeypatch.setattr(db_module, "engine", engine)


def make_paper_data(pmid: str = "12345678", title: str = "Test Paper") -> dict:
    """Return a minimal paper dict for testing."""
    return {
        "pmid": pmid,
        "title": title,
        "abstract": "This is a test abstract.",
        "journal": "npj Digital Medicine",
        "pub_date": "2026 Apr 21",
        "authors": ["John Smith", "Jane Doe"],
        "doi": f"10.1038/test-{pmid}",
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        "reason": "Relevant to clinical AI decision support.",
    }


def make_digest_papers(n: int = 3) -> list[dict]:
    """Return a list of n paper dicts for a digest."""
    return [
        make_paper_data(pmid=str(i), title=f"Paper {i}")
        for i in range(1, n + 1)
    ]


# ---------------------------------------------------------------------------
# upsert_paper tests
# ---------------------------------------------------------------------------

def test_upsert_paper_creates_new_paper(session):
    """Should create a new paper if PMID doesn't exist."""
    data = make_paper_data()
    paper = upsert_paper(session, data)
    session.flush()

    assert paper.pmid == "12345678"
    assert paper.title == "Test Paper"
    assert paper.journal == "npj Digital Medicine"


def test_upsert_paper_returns_existing_paper(session):
    """Should return the existing paper if PMID already exists."""
    data = make_paper_data()
    paper1 = upsert_paper(session, data)
    session.flush()

    paper2 = upsert_paper(session, data)
    session.flush()

    assert paper1.id == paper2.id
    assert session.query(Paper).count() == 1


def test_upsert_paper_joins_authors(session):
    """Authors list should be stored as a comma-separated string."""
    data = make_paper_data()
    paper = upsert_paper(session, data)
    session.flush()

    assert "John Smith" in paper.authors
    assert "Jane Doe" in paper.authors


def test_upsert_different_pmids_creates_two_papers(session):
    """Two papers with different PMIDs should both be stored."""
    data1 = make_paper_data(pmid="11111111")
    data2 = make_paper_data(pmid="22222222")

    upsert_paper(session, data1)
    upsert_paper(session, data2)
    session.flush()

    assert session.query(Paper).count() == 2


# ---------------------------------------------------------------------------
# save_digest tests
# ---------------------------------------------------------------------------

def test_save_digest_creates_digest(engine):
    """save_digest should create a DailyDigest row."""
    papers = make_digest_papers()
    save_digest(papers, digest_date=date(2026, 4, 28))

    with Session(engine) as session:
        assert session.query(DailyDigest).count() == 1


def test_save_digest_creates_three_digest_papers(engine):
    """save_digest should create 3 DigestPaper rows."""
    papers = make_digest_papers()
    save_digest(papers, digest_date=date(2026, 4, 28))

    with Session(engine) as session:
        assert session.query(DigestPaper).count() == 3


def test_save_digest_correct_date(engine):
    """Digest should be saved with the correct date."""
    papers = make_digest_papers()
    save_digest(papers, digest_date=date(2026, 4, 28))

    with Session(engine) as session:
        digest = session.query(DailyDigest).first()
        assert digest.digest_date == date(2026, 4, 28)


def test_save_digest_correct_ranks(engine):
    """Papers should be saved with ranks 1, 2, 3."""
    papers = make_digest_papers()
    save_digest(papers, digest_date=date(2026, 4, 28))

    with Session(engine) as session:
        ranks = [dp.rank for dp in session.query(DigestPaper).all()]
        assert sorted(ranks) == [1, 2, 3]


def test_save_digest_replaces_existing(engine):
    """Saving a second digest for the same date should replace the first."""
    papers1 = make_digest_papers()
    save_digest(papers1, digest_date=date(2026, 4, 28))

    papers2 = [make_paper_data(pmid="99", title="New Paper")]
    save_digest(papers2, digest_date=date(2026, 4, 28))

    with Session(engine) as session:
        assert session.query(DailyDigest).count() == 1
        assert session.query(DigestPaper).count() == 1


def test_save_digest_deduplicates_papers(engine):
    """Same paper appearing in two digests should only be stored once."""
    papers = make_digest_papers()
    save_digest(papers, digest_date=date(2026, 4, 27))
    save_digest(papers, digest_date=date(2026, 4, 28))

    with Session(engine) as session:
        # 3 unique papers, not 6
        assert session.query(Paper).count() == 3
        # But 6 digest_papers rows (3 per digest)
        assert session.query(DigestPaper).count() == 6


# ---------------------------------------------------------------------------
# get_latest_digest tests
# ---------------------------------------------------------------------------

def test_get_latest_digest_returns_none_when_empty():
    """Should return None if no digests exist."""
    result = get_latest_digest()
    assert result is None


def test_get_latest_digest_returns_three_papers(engine):
    """Should return exactly 3 papers."""
    papers = make_digest_papers()
    save_digest(papers, digest_date=date(2026, 4, 28))

    result = get_latest_digest()
    assert len(result) == 3


def test_get_latest_digest_returns_most_recent(engine):
    """Should return the most recent digest, not an older one."""
    old_papers = [make_paper_data(pmid="1", title="Old Paper")]
    new_papers = [make_paper_data(pmid="2", title="New Paper")]

    save_digest(old_papers, digest_date=date(2026, 4, 27))
    save_digest(new_papers, digest_date=date(2026, 4, 28))

    result = get_latest_digest()
    assert result[0]["title"] == "New Paper"


def test_get_latest_digest_has_reason(engine):
    """Each paper in the digest should have a reason field."""
    papers = make_digest_papers()
    save_digest(papers, digest_date=date(2026, 4, 28))

    result = get_latest_digest()
    for paper in result:
        assert "reason" in paper
        assert len(paper["reason"]) > 0


def test_get_latest_digest_has_digest_date(engine):
    """Each paper should include the digest date."""
    papers = make_digest_papers()
    save_digest(papers, digest_date=date(2026, 4, 28))

    result = get_latest_digest()
    for paper in result:
        assert paper["digest_date"] == "2026-04-28"


# ---------------------------------------------------------------------------
# get_recent_selected_pmids tests
# ---------------------------------------------------------------------------

def test_get_recent_selected_pmids_returns_correct_pmids(engine):
    """Should return PMIDs of papers selected in the last 7 days."""
    from paperpulse.db import get_recent_selected_pmids
    papers = make_digest_papers()
    save_digest(papers, digest_date=date.today())

    pmids = get_recent_selected_pmids(days=7)
    assert "1" in pmids
    assert "2" in pmids
    assert "3" in pmids


def test_get_recent_selected_pmids_excludes_old_digests(engine):
    """Should not return PMIDs from digests older than the cutoff."""
    from paperpulse.db import get_recent_selected_pmids
    from datetime import timedelta

    old_papers = [make_paper_data(pmid="99", title="Old Paper")]
    old_date = date.today() - timedelta(days=8)
    save_digest(old_papers, digest_date=old_date)

    pmids = get_recent_selected_pmids(days=7)
    assert "99" not in pmids


def test_get_recent_selected_pmids_returns_empty_when_no_digests():
    """Should return an empty set if no digests exist."""
    from paperpulse.db import get_recent_selected_pmids
    pmids = get_recent_selected_pmids(days=7)
    assert pmids == set()


# ---------------------------------------------------------------------------
# get_digest_for_date tests
# ---------------------------------------------------------------------------

def test_get_digest_for_date_returns_correct_digest(engine):
    """Should return the digest for the specified date."""
    papers = make_digest_papers()
    save_digest(papers, digest_date=date(2026, 4, 27))
    save_digest(papers, digest_date=date(2026, 4, 28))

    result = get_digest_for_date(date(2026, 4, 27))
    assert result is not None
    assert result[0]["digest_date"] == "2026-04-27"


def test_get_digest_for_date_returns_none_for_missing_date(engine):
    """Should return None if no digest exists for the given date."""
    result = get_digest_for_date(date(2026, 1, 1))
    assert result is None