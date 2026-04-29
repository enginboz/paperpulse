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

import sys
from datetime import date

from paperpulse.db import get_recent_selected_pmids, save_digest
from paperpulse.fetchers.pubmed import fetch_recent_papers
from paperpulse.models import Digest
from paperpulse.scoring.llm import select_top_papers


def run():
    print(f"=== PaperPulse — {date.today()} ===\n")

    # Step 1 — fetch recent papers from PubMed
    papers = fetch_recent_papers(days=7)

    if not papers:
        print("No papers fetched. Exiting.")
        sys.exit(1)

    # Step 2 — exclude papers already shown in the last 7 days
    recent_pmids = get_recent_selected_pmids(days=7)

    if recent_pmids:
        before = len(papers)
        papers = [p for p in papers if p.pmid not in recent_pmids]
        print(f"Excluded {before - len(papers)} recently shown papers. {len(papers)} remaining.\n")

    if not papers:
        print("All papers were recently shown. Exiting.")
        sys.exit(1)

    # Step 3 — score with LLM and select top 3
    top3 = select_top_papers(papers)

    if not top3:
        print("LLM scoring returned no results. Exiting.")
        sys.exit(1)

    # Step 4 — save to database
    digest = Digest(digest_date=date.today(), papers=top3)
    save_digest(digest)

    # Step 5 — print summary
    print("\n=== Today's top papers ===")
    for i, p in enumerate(top3, 1):
        print(f"\n#{i} {p.journal}")
        print(f"   {p.title}")
        print(f"   Why: {p.reason}")
        print(f"   {p.url}")

    print("\n=== Done ===")


if __name__ == "__main__":
    run()