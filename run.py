"""
PaperPulse daily pipeline entry point.

Fetches recent papers from PubMed, excludes papers selected in the
last 7 days, scores the remaining candidates with an LLM, and saves
the top 3 to the database.

Run manually or via cron:
    poetry run python run.py

Cron example (daily at 6am):
    0 6 * * * /path/to/venv/bin/python /path/to/paperpulse/run.py
"""

import logging
import os
import sys
from datetime import date
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv

from paperpulse.db import get_recent_selected_pmids, save_digest
from paperpulse.fetchers.pubmed import fetch_recent_papers
from paperpulse.models import Digest
from paperpulse.scoring.llm import select_top_papers

load_dotenv()

_log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
_log_fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

_log_dir = Path(__file__).parent / "logs"
_log_dir.mkdir(exist_ok=True)

_file_handler = TimedRotatingFileHandler(
    _log_dir / "paperpulse.log",
    when="midnight",
    backupCount=30,
    encoding="utf-8",
)
_file_handler.setFormatter(_log_fmt)

_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(_log_fmt)

logging.basicConfig(level=_log_level, handlers=[_file_handler, _stream_handler])

# Suppress transport-level noise — these loggers only produce useful output
# when debugging the HTTP libraries themselves, not application code.
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def run():
    logger.info("PaperPulse — %s", date.today())

    # Step 1 — fetch recent papers from PubMed
    papers = fetch_recent_papers(days=7)

    if not papers:
        logger.error("No papers fetched. Exiting.")
        sys.exit(1)

    # Step 2 — exclude papers already shown in the last 7 days
    recent_pmids = get_recent_selected_pmids(days=7)

    if recent_pmids:
        before = len(papers)
        papers = [p for p in papers if p.pmid not in recent_pmids]
        logger.info("Excluded %d recently shown papers. %d remaining.", before - len(papers), len(papers))

    if not papers:
        logger.error("All papers were recently shown. Exiting.")
        sys.exit(1)

    # Step 3 — score with LLM and select top 3
    top3 = select_top_papers(papers)

    if not top3:
        logger.error("LLM scoring returned no results. Exiting.")
        sys.exit(1)

    # Step 4 — save to database
    digest = Digest(digest_date=date.today(), papers=top3)
    save_digest(digest)

    # Step 5 — log summary
    logger.info("Today's top papers:")
    for i, p in enumerate(top3, 1):
        logger.info("  #%d [%s] %s", i, p.journal, p.title)
        logger.info("      Why: %s", p.reason)
        logger.info("      %s", p.url)

    logger.info("Done.")


if __name__ == "__main__":
    run()