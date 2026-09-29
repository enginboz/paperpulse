"""
The two halves of PaperPulse.

ingest()  pulls new papers from each source into the store. It is incremental
          and idempotent: each source resumes from its last watermark.
select()  ranks stored papers against the profile and returns a Digest. It
          never touches the network, so it can be re-run freely while tuning.
"""

import logging
import uuid
from datetime import UTC, date, datetime, timedelta

from paperpulse.config import Config
from paperpulse.embeddings import Embedder, embed_papers
from paperpulse.models import Digest, RankedPaper, RunInfo, Score
from paperpulse.ranking.dense import best_topic_matches
from paperpulse.ranking.filters import apply_filters
from paperpulse.sources import Source
from paperpulse.store import Store

logger = logging.getLogger(__name__)

# Entrez dates are day-granular and records keep arriving during the day,
# so each ingest re-reads the last day it saw. Upserts make that harmless.
INGEST_OVERLAP = timedelta(days=1)

RANKED_FIELDS = {"id", "title", "journal", "published", "authors", "doi", "pmid", "url"}


def ingest(store: Store, sources: list[Source], default_days: int, today: date) -> int:
    total_new = 0
    for source in sources:
        last = store.last_ingested(source.name)
        start = last - INGEST_OVERLAP if last else today - timedelta(days=default_days)
        logger.info("Ingesting %s from %s to %s", source.name, start, today)
        new = store.upsert_papers(source.fetch(start, today))
        store.set_last_ingested(source.name, today)
        logger.info("%s: %d new papers", source.name, new)
        total_new += new
    return total_new


def select(
    store: Store,
    config: Config,
    embedder: Embedder,
    now: datetime | None = None,
    record: bool = True,
) -> Digest:
    now = now or datetime.now(UTC)
    sel = config.selection
    window_end = now.date()
    window_start = window_end - timedelta(days=sel.window_days)

    candidates = store.papers_added_between(window_start, window_end)
    recent = store.selected_since(now - timedelta(days=sel.no_repeat_days))
    eligible = apply_filters(candidates, sel, exclude_ids=recent)
    logger.info("%d candidates, %d after filters", len(candidates), len(eligible))

    ranked: list[RankedPaper] = []
    if eligible:
        topics = config.profile.topics
        matches = best_topic_matches(
            embed_papers(store, embedder, eligible), embedder.encode(topics)
        )
        order = sorted(range(len(eligible)), key=lambda i: matches[i].score, reverse=True)
        for rank, i in enumerate(order[: sel.top], 1):
            p, m = eligible[i], matches[i]
            ranked.append(
                RankedPaper(
                    rank=rank,
                    **p.model_dump(include=RANKED_FIELDS),
                    score=Score(
                        total=round(m.score, 4),
                        dense=round(m.score, 4),
                        matched_topic=topics[m.topic_index],
                    ),
                )
            )

    digest = Digest(
        run=RunInfo(
            id=uuid.uuid4().hex,
            created_at=now,
            window_start=window_start,
            window_end=window_end,
            embedding_model=embedder.name,
            candidates=len(candidates),
            after_filters=len(eligible),
        ),
        papers=ranked,
    )
    if record and ranked:
        store.save_run(digest)
    return digest
