"""
Tests for the database module.

Uses an in-memory SQLite database to avoid needing a real PostgreSQL
instance during testing. SQLAlchemy's ORM works identically across both.
"""

import pytest
from datetime import date, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from paperpulse.db import (
    Base,
    DBPaper,
    DBDailyDigest,
    DBDigestPaper,
    _upsert_paper,
    save_digest,
    get_latest_digest,
    get_digest_for_date,
    get_recent_selected_pmids,
)
from paperpulse.models import Paper, Digest


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


def make_paper(pmid: str = "12345678", title: str = "Test Paper") -> Paper:
    """Return a minimal Paper domain object for testing."""
    return Paper(
        pmid=pmid,
        title=title,
        abstract="This is a test abstract.",
        journal="npj Digital Medicine",
        pub_date="2026 Apr 21",
        authors=["John Smith", "Jane Doe"],
        doi=f"10.1038/test-{pmid}",
        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        reason="Relevant to clinical AI decision support.",
    )


def make_digest(digest_date: date = None, n: int = 3) -> Digest:
    """Return a Digest with n papers for testing."""
    if digest_date is None:
        digest_date = date(2026, 4, 28)
    papers = [make_paper(pmid=str(i), title=f"Paper {i}") for i in range(1, n + 1)]
    return Digest(digest_date=digest_date, papers=papers)


# ---------------------------------------------------------------------------
# _upsert_paper tests
# ---------------------------------------------------------------------------

def test_upsert_paper_creates_new_paper(session):
    """Should create a new DBPaper if PMID doesn't exist."""
    paper = make_paper()
    db_paper = _upsert_paper(session, paper)
    session.flush()

    assert db_paper.pmid == "12345678"
    assert db_paper.title == "Test Paper"
    assert db_paper.journal == "npj Digital Medicine"


def test_upsert_paper_returns_existing_paper(session):
    """Should return the existing DBPaper if PMID already exists."""
    paper = make_paper()
    db_paper1 = _upsert_paper(session, paper)
    session.flush()

    db_paper2 = _upsert_paper(session, paper)
    session.flush()

    assert db_paper1.id == db_paper2.id
    assert session.query(DBPaper).count() == 1


def test_upsert_paper_joins_authors(session):
    """Authors list should be stored as a comma-separated string."""
    paper = make_paper()
    db_paper = _upsert_paper(session, paper)
    session.flush()

    assert "John Smith" in db_paper.authors
    assert "Jane Doe" in db_paper.authors


def test_upsert_different_pmids_creates_two_papers(session):
    """Two papers with different PMIDs should both be stored."""
    _upsert_paper(session, make_paper(pmid="11111111"))
    _upsert_paper(session, make_paper(pmid="22222222"))
    session.flush()

    assert session.query(DBPaper).count() == 2


# ---------------------------------------------------------------------------
# save_digest tests
# ---------------------------------------------------------------------------

def test_save_digest_creates_digest(engine):
    """save_digest should create a DBDailyDigest row."""
    save_digest(make_digest())

    with Session(engine) as session:
        assert session.query(DBDailyDigest).count() == 1


def test_save_digest_creates_three_digest_papers(engine):
    """save_digest should create 3 DBDigestPaper rows."""
    save_digest(make_digest(n=3))

    with Session(engine) as session:
        assert session.query(DBDigestPaper).count() == 3


def test_save_digest_correct_date(engine):
    """Digest should be saved with the correct date."""
    save_digest(make_digest(digest_date=date(2026, 4, 28)))

    with Session(engine) as session:
        digest = session.query(DBDailyDigest).first()
        assert digest.digest_date == date(2026, 4, 28)


def test_save_digest_correct_ranks(engine):
    """Papers should be saved with ranks 1, 2, 3."""
    save_digest(make_digest(n=3))

    with Session(engine) as session:
        ranks = [dp.rank for dp in session.query(DBDigestPaper).all()]
        assert sorted(ranks) == [1, 2, 3]


def test_save_digest_replaces_existing(engine):
    """Saving a second digest for the same date should replace the first."""
    save_digest(make_digest(digest_date=date(2026, 4, 28), n=3))
    save_digest(Digest(
        digest_date=date(2026, 4, 28),
        papers=[make_paper(pmid="99", title="New Paper")]
    ))

    with Session(engine) as session:
        assert session.query(DBDailyDigest).count() == 1
        assert session.query(DBDigestPaper).count() == 1


def test_save_digest_deduplicates_papers(engine):
    """Same paper appearing in two digests should only be stored once."""
    digest1 = make_digest(digest_date=date(2026, 4, 27), n=3)
    digest2 = make_digest(digest_date=date(2026, 4, 28), n=3)
    save_digest(digest1)
    save_digest(digest2)

    with Session(engine) as session:
        assert session.query(DBPaper).count() == 3
        assert session.query(DBDigestPaper).count() == 6


# ---------------------------------------------------------------------------
# get_latest_digest tests
# ---------------------------------------------------------------------------

def test_get_latest_digest_returns_none_when_empty():
    """Should return None if no digests exist."""
    result = get_latest_digest()
    assert result is None


def test_get_latest_digest_returns_digest(engine):
    """Should return a Digest object."""
    save_digest(make_digest(n=3))
    result = get_latest_digest()
    assert isinstance(result, Digest)


def test_get_latest_digest_returns_three_papers(engine):
    """Should return exactly 3 papers."""
    save_digest(make_digest(n=3))
    result = get_latest_digest()
    assert len(result.papers) == 3


def test_get_latest_digest_returns_most_recent(engine):
    """Should return the most recent digest, not an older one."""
    old_digest = Digest(
        digest_date=date(2026, 4, 27),
        papers=[make_paper(pmid="1", title="Old Paper")]
    )
    new_digest = Digest(
        digest_date=date(2026, 4, 28),
        papers=[make_paper(pmid="2", title="New Paper")]
    )
    save_digest(old_digest)
    save_digest(new_digest)

    result = get_latest_digest()
    assert result.papers[0].title == "New Paper"


def test_get_latest_digest_papers_have_reason(engine):
    """Each paper in the digest should have a reason field."""
    save_digest(make_digest(n=3))
    result = get_latest_digest()
    for paper in result.papers:
        assert paper.reason != ""


def test_get_latest_digest_correct_date(engine):
    """Digest should have the correct date."""
    save_digest(make_digest(digest_date=date(2026, 4, 28)))
    result = get_latest_digest()
    assert result.digest_date == date(2026, 4, 28)


# ---------------------------------------------------------------------------
# get_recent_selected_pmids tests
# ---------------------------------------------------------------------------

def test_get_recent_selected_pmids_returns_correct_pmids(engine):
    """Should return PMIDs of papers selected in the last 7 days."""
    save_digest(make_digest(digest_date=date.today(), n=3))
    pmids = get_recent_selected_pmids(days=7)
    assert "1" in pmids
    assert "2" in pmids
    assert "3" in pmids


def test_get_recent_selected_pmids_excludes_old_digests(engine):
    """Should not return PMIDs from digests older than the cutoff."""
    old_digest = Digest(
        digest_date=date.today() - timedelta(days=8),
        papers=[make_paper(pmid="99", title="Old Paper")]
    )
    save_digest(old_digest)
    pmids = get_recent_selected_pmids(days=7)
    assert "99" not in pmids


def test_get_recent_selected_pmids_returns_empty_when_no_digests():
    """Should return an empty set if no digests exist."""
    pmids = get_recent_selected_pmids(days=7)
    assert pmids == set()


# ---------------------------------------------------------------------------
# get_digest_for_date tests
# ---------------------------------------------------------------------------

def test_get_digest_for_date_returns_correct_digest(engine):
    """Should return the digest for the specified date."""
    save_digest(make_digest(digest_date=date(2026, 4, 27)))
    save_digest(make_digest(digest_date=date(2026, 4, 28)))

    result = get_digest_for_date(date(2026, 4, 27))
    assert result is not None
    assert result.digest_date == date(2026, 4, 27)


def test_get_digest_for_date_returns_none_for_missing_date(engine):
    """Should return None if no digest exists for the given date."""
    result = get_digest_for_date(date(2026, 1, 1))
    assert result is None