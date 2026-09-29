"""
The two halves of PaperPulse.

ingest()  pulls new papers from each source into the store. It is incremental
          and idempotent: each source resumes from its last watermark.
select()  ranks stored papers against the profile and returns a Digest. It
          only talks to the local LLM, and every model output is cached, so it
          can be re-run freely while tuning.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

from paperpulse.config import Config
from paperpulse.embeddings import Embedder, embed_papers
from paperpulse.llm import LLM, LLMUnavailable
from paperpulse.models import Digest, Paper, RankedPaper, RunInfo, Score
from paperpulse.ranking.assess import assess_papers
from paperpulse.ranking.dense import best_topic_matches
from paperpulse.ranking.filters import apply_filters
from paperpulse.ranking.fusion import reciprocal_rank_fusion
from paperpulse.ranking.keyword import best_keyword_matches
from paperpulse.sources import Source
from paperpulse.store import Store

logger = logging.getLogger(__name__)

# Entrez dates are day-granular and records keep arriving during the day,
# so each ingest re-reads the last day it saw. Upserts make that harmless.
INGEST_OVERLAP = timedelta(days=1)

RANKED_FIELDS = {"id", "title", "journal", "published", "authors", "doi", "pmid", "url"}


def window_start(end: date, days: int) -> date:
    """First day of a `days`-long window ending on `end`, both inclusive."""
    return end - timedelta(days=days - 1)


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
    llm: LLM | None = None,
    now: datetime | None = None,
    record: bool = True,
) -> Digest:
    now = now or datetime.now(UTC)
    sel = config.selection
    window_end = now.date()
    start = window_start(window_end, sel.window_days)

    candidates = store.papers_added_between(start, window_end)
    recent = store.selected_since(now - timedelta(days=sel.no_repeat_days))
    eligible = apply_filters(candidates, sel, exclude_ids=recent)
    logger.info("%d candidates, %d after filters", len(candidates), len(eligible))

    ranking = rank_papers(store, config, embedder, eligible, llm=llm)
    scored, llm_used = ranking.papers, ranking.llm_model

    ranked = [
        RankedPaper(
            rank=rank,
            **p.model_dump(include=RANKED_FIELDS),
            score=s.model_copy(
                update={f: round(getattr(s, f), 4) for f in ("total", "fusion", "dense")}
            ),
        )
        for rank, (p, s) in enumerate(scored[: sel.top], 1)
    ]

    digest = Digest(
        run=RunInfo(
            id=uuid.uuid4().hex,
            created_at=now,
            window_start=start,
            window_end=window_end,
            embedding_model=embedder.name,
            llm_model=llm_used,
            candidates=len(candidates),
            after_filters=len(eligible),
        ),
        papers=ranked,
    )
    if record and ranked:
        store.save_run(digest)
    return digest


@dataclass
class Ranking:
    papers: list[tuple[Paper, Score]]
    llm_model: str | None
    shortlist: list[str] = field(default_factory=list)
    """Paper ids that reached the LLM stage, in retrieval order."""


def rank_papers(
    store: Store,
    config: Config,
    embedder: Embedder,
    papers: list[Paper],
    llm: LLM | None = None,
    cached_assessments_only: bool = False,
) -> Ranking:
    """
    Order papers best first: hybrid retrieval, then (with an LLM) per-paper
    assessment of the shortlist. Shared by `select` and `eval`, so evaluation
    measures exactly what the digest would show.
    """
    if not papers:
        return Ranking([], None)
    scored = _retrieve(store, config, embedder, papers)
    if llm is None:
        return Ranking(scored, None)

    shortlist = scored[: config.llm.candidates]
    try:
        assessments = assess_papers(
            store,
            llm,
            config.profile,
            [p for p, _ in shortlist],
            cached_only=cached_assessments_only,
        )
    except LLMUnavailable as e:
        logger.warning("%s; ranking by retrieval scores only", e)
        return Ranking(scored, None)

    kept = [
        (p, s.model_copy(update={"total": a.relevance + s.fusion, "assessment": a}))
        for p, s in shortlist
        if (a := assessments.get(p.id)) and a.relevance >= config.llm.min_relevance
    ]
    logger.info(
        "%d of %d shortlisted papers rated relevance >= %d",
        len(kept),
        len(shortlist),
        config.llm.min_relevance,
    )
    kept.sort(key=lambda ps: ps[1].total, reverse=True)
    return Ranking(kept, llm.name, [p.id for p, _ in shortlist])


def _retrieve(
    store: Store, config: Config, embedder: Embedder, papers: list[Paper]
) -> list[tuple[Paper, Score]]:
    """Rank papers by fusing dense and keyword ranks, best first."""
    topics = config.profile.topics
    dense = best_topic_matches(embed_papers(store, embedder, papers), embedder.encode(topics))
    by_dense = sorted(range(len(papers)), key=lambda i: dense[i].score, reverse=True)
    dense_ranks = {papers[i].id: rank for rank, i in enumerate(by_dense, 1)}

    keyword = {}
    if config.retrieval.hybrid:
        keyword = best_keyword_matches(store, topics, [p.id for p in papers])
    fused = reciprocal_rank_fusion(
        [dense_ranks, {pid: m.rank for pid, m in keyword.items()}], k=config.retrieval.rrf_k
    )

    scored = []
    for paper, d in zip(papers, dense, strict=True):
        k = keyword.get(paper.id)
        scored.append(
            (
                paper,
                Score(
                    total=fused[paper.id],
                    fusion=fused[paper.id],
                    dense=d.score,
                    dense_rank=dense_ranks[paper.id],
                    matched_topic=topics[d.topic_index],
                    keyword_rank=k.rank if k else None,
                    keyword_topic=topics[k.topic_index] if k else None,
                ),
            )
        )
    scored.sort(key=lambda ps: ps[1].fusion, reverse=True)
    return scored
