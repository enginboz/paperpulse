"""
Database module.

Defines the SQLAlchemy models and database helper functions for PaperPulse.

Schema:
  - papers         — stores all fetched papers, deduplicated by PMID
  - daily_digests  — one row per day, tracks when a digest was created
  - digest_papers  — junction table linking a digest to its 3 selected papers
                     with the LLM's relevance reason for each
"""

import os
from datetime import date, datetime

from dotenv import load_dotenv
from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship

load_dotenv()

# ---------------------------------------------------------------------------
# Engine setup
# ---------------------------------------------------------------------------

DATABASE_URL = os.getenv("DATABASE_URL", "")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set in .env")

engine = create_engine(DATABASE_URL)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


class Paper(Base):
    """
    A paper fetched from PubMed.
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
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship back to digest entries
    digest_entries = relationship("DigestPaper", back_populates="paper")

    def __repr__(self):
        return f"<Paper pmid={self.pmid} title={self.title[:50]}>"


class DailyDigest(Base):
    """
    One digest per day — tracks when the daily pipeline ran
    and links to the 3 selected papers.
    """
    __tablename__ = "daily_digests"

    id = Column(Integer, primary_key=True, autoincrement=True)
    digest_date = Column(Date, unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship to the 3 selected papers
    papers = relationship(
        "DigestPaper",
        back_populates="digest",
        order_by="DigestPaper.rank",
        cascade="all, delete-orphan",
    )

    def __repr__(self):
        return f"<DailyDigest date={self.digest_date}>"


class DigestPaper(Base):
    """
    Junction table linking a daily digest to one of its selected papers.
    Stores the LLM's relevance reason and the paper's rank (1, 2, or 3).
    """
    __tablename__ = "digest_papers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    digest_id = Column(Integer, ForeignKey("daily_digests.id"), nullable=False)
    paper_id = Column(Integer, ForeignKey("papers.id"), nullable=False)
    rank = Column(Integer, nullable=False)       # 1, 2, or 3
    reason = Column(Text, nullable=False)        # LLM's relevance reason

    digest = relationship("DailyDigest", back_populates="papers")
    paper = relationship("Paper", back_populates="digest_entries")

    def __repr__(self):
        return f"<DigestPaper digest={self.digest_id} rank={self.rank}>"


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


def upsert_paper(session: Session, paper_data: dict) -> Paper:
    """
    Insert a paper if it doesn't exist, or return the existing one.
    Deduplicates by PMID.
    """
    existing = session.query(Paper).filter_by(pmid=paper_data["pmid"]).first()
    if existing:
        return existing

    paper = Paper(
        pmid=paper_data["pmid"],
        title=paper_data["title"],
        abstract=paper_data["abstract"],
        journal=paper_data["journal"],
        pub_date=paper_data.get("pub_date", ""),
        authors=", ".join(paper_data.get("authors", [])),
        doi=paper_data.get("doi", ""),
        url=paper_data.get("url", ""),
    )
    session.add(paper)
    return paper


def save_digest(selected_papers: list[dict], digest_date: date = None) -> None:
    """
    Save a daily digest and its 3 selected papers to the database.

    If a digest already exists for the given date, it is replaced.

    Args:
        selected_papers: List of 3 paper dicts enriched with a 'reason' field.
        digest_date: The date for the digest (default: today).
    """
    if digest_date is None:
        digest_date = date.today()

    with Session(engine) as session:
        # Remove existing digest for this date if it exists
        existing = session.query(DailyDigest).filter_by(
            digest_date=digest_date
        ).first()
        if existing:
            print(f"Replacing existing digest for {digest_date}.")
            session.delete(existing)
            session.flush()

        # Create the new digest
        digest = DailyDigest(digest_date=digest_date)
        session.add(digest)
        session.flush()  # Get the digest ID before adding papers

        # Upsert each paper and link to the digest
        for rank, paper_data in enumerate(selected_papers, 1):
            paper = upsert_paper(session, paper_data)
            session.flush()  # Get the paper ID

            digest_paper = DigestPaper(
                digest_id=digest.id,
                paper_id=paper.id,
                rank=rank,
                reason=paper_data.get("reason", ""),
            )
            session.add(digest_paper)

        session.commit()
        print(f"Digest for {digest_date} saved with {len(selected_papers)} papers.")


def get_latest_digest() -> list[dict] | None:
    """
    Retrieve the most recent daily digest from the database.

    Returns:
        List of 3 paper dicts with a 'reason' field, ordered by rank.
        Returns None if no digest exists yet.
    """
    with Session(engine) as session:
        digest = session.query(DailyDigest).order_by(
            DailyDigest.digest_date.desc()
        ).first()

        if not digest:
            return None

        results = []
        for entry in digest.papers:
            p = entry.paper
            results.append({
                "pmid": p.pmid,
                "title": p.title,
                "abstract": p.abstract,
                "journal": p.journal,
                "pub_date": p.pub_date,
                "authors": p.authors,
                "doi": p.doi,
                "url": p.url,
                "reason": entry.reason,
                "rank": entry.rank,
                "digest_date": digest.digest_date.isoformat(),
            })

        return results


def get_recent_selected_pmids(days: int = 7) -> set[str]:
    """
    Return the set of PMIDs that have been selected in the last N days.
    Used to exclude recently shown papers from the next selection.

    Args:
        days: How many days back to look (default: 7).

    Returns:
        Set of PMID strings.
    """
    from datetime import timedelta
    cutoff = date.today() - timedelta(days=days)

    with Session(engine) as session:
        digests = session.query(DailyDigest).filter(
            DailyDigest.digest_date >= cutoff
        ).all()

        pmids = set()
        for digest in digests:
            for entry in digest.papers:
                pmids.add(entry.paper.pmid)

        return pmids


def get_digest_for_date(target_date: date) -> list[dict] | None:
    """
    Retrieve the digest for a specific date.

    Returns:
        List of 3 paper dicts with a 'reason' field, or None if not found.
    """
    with Session(engine) as session:
        digest = session.query(DailyDigest).filter_by(
            digest_date=target_date
        ).first()

        if not digest:
            return None

        results = []
        for entry in digest.papers:
            p = entry.paper
            results.append({
                "pmid": p.pmid,
                "title": p.title,
                "abstract": p.abstract,
                "journal": p.journal,
                "pub_date": p.pub_date,
                "authors": p.authors,
                "doi": p.doi,
                "url": p.url,
                "reason": entry.reason,
                "rank": entry.rank,
                "digest_date": digest.digest_date.isoformat(),
            })

        return results