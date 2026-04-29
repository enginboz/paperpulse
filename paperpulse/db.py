"""
Database module.

Defines the SQLAlchemy ORM models and database helper functions for PaperPulse.

Schema:
  - papers         — stores all fetched papers, deduplicated by PMID
  - daily_digests  — one row per day, tracks when a digest was created
  - digest_papers  — junction table linking a digest to its 3 selected papers
                     with the LLM's relevance reason for each

SQLAlchemy ORM models are prefixed with 'DB' to avoid naming conflicts
with the domain models in paperpulse.models.
"""

import os
from datetime import date, datetime, timedelta

from dotenv import load_dotenv
from sqlalchemy import (
    Column,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship

from paperpulse.models import Digest, Paper

load_dotenv()

# ---------------------------------------------------------------------------
# Engine setup
# ---------------------------------------------------------------------------

DATABASE_URL = os.getenv("DATABASE_URL", "")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set in .env")

engine = create_engine(DATABASE_URL)


# ---------------------------------------------------------------------------
# ORM models (prefixed with DB to avoid conflict with domain models)
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


class DBPaper(Base):
    """
    ORM model for a paper stored in the database.
    Deduplicated by PMID — each paper is stored only once.
    """
    __tablename__ = "papers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    pmid = Column(String(20), unique=True, nullable=False, index=True)
    title = Column(Text, nullable=False)
    abstract = Column(Text, nullable=False)
    journal = Column(String(255), nullable=False)
    pub_date = Column(String(50))
    authors = Column(Text)           # Stored as comma-separated string
    doi = Column(String(255))
    url = Column(String(500))
    created_at = Column(DateTime, default=datetime.now)

    digest_entries = relationship("DBDigestPaper", back_populates="paper")

    def __repr__(self):
        return f"<DBPaper pmid={self.pmid} title={self.title[:50]}>"


class DBDailyDigest(Base):
    """
    ORM model for a daily digest.
    One row per day — tracks when the daily pipeline ran.
    """
    __tablename__ = "daily_digests"

    id = Column(Integer, primary_key=True, autoincrement=True)
    digest_date = Column(Date, unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.now)

    papers = relationship(
        "DBDigestPaper",
        back_populates="digest",
        order_by="DBDigestPaper.rank",
        cascade="all, delete-orphan",
    )

    def __repr__(self):
        return f"<DBDailyDigest date={self.digest_date}>"


class DBDigestPaper(Base):
    """
    ORM model for the junction table linking a digest to a selected paper.
    """
    __tablename__ = "digest_papers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    digest_id = Column(Integer, ForeignKey("daily_digests.id"), nullable=False)
    paper_id = Column(Integer, ForeignKey("papers.id"), nullable=False)
    rank = Column(Integer, nullable=False)
    reason = Column(Text, nullable=False)

    digest = relationship("DBDailyDigest", back_populates="papers")
    paper = relationship("DBPaper", back_populates="digest_entries")

    def __repr__(self):
        return f"<DBDigestPaper digest={self.digest_id} rank={self.rank}>"


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def init_db() -> None:
    """
    Create all tables if they don't exist.
    Safe to run multiple times — existing tables are not affected.
    """
    Base.metadata.create_all(engine)
    print("Database tables created.")


def _upsert_paper(session: Session, paper: Paper) -> DBPaper:
    """
    Insert a Paper into the database if it doesn't exist,
    or return the existing DBPaper. Deduplicates by PMID.
    """
    existing = session.query(DBPaper).filter_by(pmid=paper.pmid).first()
    if existing:
        return existing

    db_paper = DBPaper(
        pmid=paper.pmid,
        title=paper.title,
        abstract=paper.abstract,
        journal=paper.journal,
        pub_date=paper.pub_date,
        authors=paper.authors_str(),
        doi=paper.doi,
        url=paper.url,
    )
    session.add(db_paper)
    return db_paper


def save_digest(digest: Digest) -> None:
    """
    Save a Digest and its selected papers to the database.
    If a digest already exists for the given date, it is replaced.

    Args:
        digest: A Digest object containing the date and selected papers.
    """
    with Session(engine) as session:
        # Remove existing digest for this date if it exists
        existing = session.query(DBDailyDigest).filter_by(
            digest_date=digest.digest_date
        ).first()
        if existing:
            print(f"Replacing existing digest for {digest.digest_date}.")
            session.delete(existing)
            session.flush()

        # Create the new digest
        db_digest = DBDailyDigest(digest_date=digest.digest_date)
        session.add(db_digest)
        session.flush()

        # Upsert each paper and link to the digest
        for rank, paper in enumerate(digest.papers, 1):
            db_paper = _upsert_paper(session, paper)
            session.flush()

            digest_paper = DBDigestPaper(
                digest_id=db_digest.id,
                paper_id=db_paper.id,
                rank=rank,
                reason=paper.reason,
            )
            session.add(digest_paper)

        session.commit()
        print(f"Digest for {digest.digest_date} saved with {len(digest.papers)} papers.")


def get_latest_digest() -> Digest | None:
    """
    Retrieve the most recent daily digest from the database.

    Returns:
        A Digest object with papers ordered by rank, or None if no digest exists.
    """
    with Session(engine) as session:
        db_digest = session.query(DBDailyDigest).order_by(
            DBDailyDigest.digest_date.desc()
        ).first()

        if not db_digest:
            return None

        return _db_digest_to_digest(db_digest)


def get_digest_for_date(target_date: date) -> Digest | None:
    """
    Retrieve the digest for a specific date.

    Returns:
        A Digest object, or None if not found.
    """
    with Session(engine) as session:
        db_digest = session.query(DBDailyDigest).filter_by(
            digest_date=target_date
        ).first()

        if not db_digest:
            return None

        return _db_digest_to_digest(db_digest)


def get_recent_selected_pmids(days: int = 7) -> set[str]:
    """
    Return the set of PMIDs that have been selected in the last N days.
    Used to exclude recently shown papers from the next selection.

    Args:
        days: How many days back to look (default: 7).

    Returns:
        Set of PMID strings.
    """
    cutoff = date.today() - timedelta(days=days)

    with Session(engine) as session:
        digests = session.query(DBDailyDigest).filter(
            DBDailyDigest.digest_date >= cutoff
        ).all()

        pmids = set()
        for digest in digests:
            for entry in digest.papers:
                pmids.add(entry.paper.pmid)

        return pmids


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _db_digest_to_digest(db_digest: DBDailyDigest) -> Digest:
    """Convert a DBDailyDigest ORM object to a Digest domain object."""
    papers = []
    for entry in db_digest.papers:
        p = entry.paper
        papers.append(Paper(
            pmid=p.pmid,
            title=p.title,
            abstract=p.abstract,
            journal=p.journal,
            pub_date=p.pub_date or "",
            authors=p.authors.split(", ") if p.authors else [],
            doi=p.doi or "",
            url=p.url or "",
            reason=entry.reason,
            rank=entry.rank,
            digest_date=db_digest.digest_date.isoformat(),
        ))

    return Digest(digest_date=db_digest.digest_date, papers=papers)