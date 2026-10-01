"""
Offline evaluation of ranking variants against human relevance labels.

Labels come from two places: feedback on digests, and `paperpulse label`,
which asks about a *pool* of papers. The pool is the union of the top papers
of every ranking variant (as in TREC pooling), so each variant gets the chance
to show what it finds that the others miss. Judging only the papers a digest
already showed would make the current ranking look perfect by construction.
The pool goes as deep as the LLM shortlist, so every paper an LLM could
promote has a label; otherwise unjudged promotions would count against it.

For each time window, every variant ranks the same eligible papers:

    dense          embedding similarity only
    hybrid         dense + BM25 fused with reciprocal rank fusion
    hybrid+<llm>   hybrid, then that LLM assesses the shortlist (one row per
                   model, so models are compared on the same labels)

and is scored on

    precision@k  share of relevant papers among the (up to) k it would show
    recall@n     share of all relevant papers that reach its top n, where n is
                 the LLM shortlist size: what the LLM stage gets to see
    judged@k     share of the top k that carry a label; unlabelled papers
                 count as not relevant, so low coverage means label more
"""

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from importlib.metadata import version
from pathlib import Path
from statistics import mean

from paperpulse.config import Config
from paperpulse.embeddings import Embedder
from paperpulse.llm import LLM
from paperpulse.models import Label, Paper
from paperpulse.pipeline import rank_papers, window_start
from paperpulse.ranking.assess import PROMPT_VERSION, cache_key
from paperpulse.ranking.filters import apply_filters
from paperpulse.ranking.keyword import best_keyword_matches
from paperpulse.store import Store

logger = logging.getLogger(__name__)


def eligible_papers(store: Store, config: Config, start: date, end: date) -> list[Paper]:
    """Papers a digest for this window could pick; repeats are allowed, as in a fresh start."""
    papers = store.papers_added_between(start, end)
    return apply_filters(papers, config.selection, exclude_ids=set())


def _variant_config(config: Config, hybrid: bool) -> Config:
    variant = config.model_copy(deep=True)
    variant.retrieval.hybrid = hybrid
    return variant


# -- pooling ---------------------------------------------------------------


def labelling_pool(
    store: Store, config: Config, embedder: Embedder, papers: list[Paper], depth: int
) -> list[Paper]:
    """
    Union of the top `depth` papers by dense, keyword and hybrid ranking, minus
    papers already labelled. Interleaved round-robin, so stopping early still
    covers the head of every ranking.
    """
    rankings = [
        [p for p, _ in rank_papers(store, _variant_config(config, h), embedder, papers).papers]
        for h in (False, True)
    ]
    keyword = best_keyword_matches(store, config.profile.topics, [p.id for p in papers])
    by_id = {p.id: p for p in papers}
    rankings.append([by_id[pid] for pid in sorted(keyword, key=lambda pid: keyword[pid].rank)])

    labelled = store.get_labels()
    pool: dict[str, Paper] = {}
    for position in range(depth):
        for ranking in rankings:
            if position < len(ranking):
                paper = ranking[position]
                if paper.id not in labelled:
                    pool.setdefault(paper.id, paper)
    return list(pool.values())


# -- evaluation ------------------------------------------------------------


@dataclass
class WindowScore:
    start: date
    end: date
    relevant: int
    precision: float | None
    recall: float | None
    judged: float | None
    returned: int
    assessed: float | None = None
    top: list[str] = field(default_factory=list)
    """Ids of the papers this variant would show, so a result can be checked paper by paper."""
    shortlist: list[str] = field(default_factory=list)
    """Ids of its top n, the LLM shortlist; recall is computed over these."""


@dataclass
class VariantResult:
    name: str
    windows: list[WindowScore] = field(default_factory=list)
    unavailable: bool = False
    """The LLM could not be reached, so this variant was not measured."""

    def mean(self, attr: str) -> float | None:
        values = [v for w in self.windows if (v := getattr(w, attr)) is not None]
        return mean(values) if values else None


def evaluation_windows(
    labels: list[tuple[Label, Paper]], days: int, end: date | None = None
) -> list[tuple[date, date]]:
    """
    Consecutive, non-overlapping windows of `days` covering every labelled
    paper, newest last. They tile back from `end` (default: the newest
    labelled paper). Passing the labelling day keeps them aligned with the
    windows `paperpulse label` pooled from, so no window splits a pool.
    """
    if not labels:
        return []
    first = min(p.added for _, p in labels)
    end = end or max(p.added for _, p in labels)
    windows = []
    while end >= first:
        start = window_start(end, days)
        windows.append((start, end))
        end = start - timedelta(days=1)
    return windows[::-1]


def evaluate(
    store: Store,
    config: Config,
    embedder: Embedder,
    llms: list[LLM] = (),
    k: int | None = None,
    cached_only: bool = False,
    end: date | None = None,
) -> list[VariantResult]:
    k = k or config.selection.top
    n = config.llm.candidates
    labels = store.get_labels()
    variants: list[tuple[str, Config, LLM | None]] = [
        ("dense", _variant_config(config, hybrid=False), None),
        ("hybrid", _variant_config(config, hybrid=True), None),
    ]
    for llm in llms:
        variants.append((f"hybrid+{llm.name}", _variant_config(config, hybrid=True), llm))
    results = [VariantResult(name) for name, _, _ in variants]

    windows = evaluation_windows(store.labelled_papers(), config.selection.window_days, end)
    for start, end in windows:
        papers = eligible_papers(store, config, start, end)
        relevant = {p.id for p in papers if p.id in labels and labels[p.id].relevant}
        if not any(p.id in labels for p in papers):
            continue
        for result, (_, variant, variant_llm) in zip(results, variants, strict=True):
            ranking = rank_papers(
                store,
                variant,
                embedder,
                papers,
                llm=variant_llm,
                cached_assessments_only=cached_only,
            )
            if variant_llm is not None and ranking.llm_model is None:
                # rank_papers fell back to retrieval only; scoring that would
                # silently report hybrid's numbers under the LLM's name.
                result.unavailable = True
                continue
            top = [p.id for p, _ in ranking.papers[:k]]
            reached = ranking.shortlist_ids or [p.id for p, _ in ranking.papers[:n]]
            assessed = None
            if variant_llm is not None and ranking.shortlist:
                key = cache_key(variant_llm, config.profile)
                cached = store.get_assessments(ranking.shortlist_ids, key)
                assessed = len(cached) / len(ranking.shortlist)
            result.windows.append(
                WindowScore(
                    start=start,
                    end=end,
                    relevant=len(relevant),
                    precision=sum(pid in relevant for pid in top) / len(top) if top else None,
                    recall=len(relevant.intersection(reached)) / len(relevant)
                    if relevant
                    else None,
                    judged=sum(pid in labels for pid in top) / len(top) if top else None,
                    returned=len(top),
                    assessed=assessed,
                    top=top,
                    shortlist=reached,
                )
            )
    return results


def format_report(results: list[VariantResult], k: int, n: int) -> str:
    def pct(value: float | None) -> str:
        return "   -" if value is None else f"{value:4.0%}"

    windows = results[0].windows if results else []
    if not windows:
        return (
            "No labelled papers yet. Rate digest picks with `paperpulse feedback <rank> up|down`, "
            "or label a pool with `paperpulse label`."
        )
    width = max(len(r.name) for r in results) + 2
    lines = [
        f"{len(windows)} window(s), {sum(w.relevant for w in windows)} relevant labelled papers",
        "",
        f"{'variant':<{width}} {'P@' + str(k):>6} {'R@' + str(n):>6} {'judged':>7}",
    ]
    for r in results:
        if r.unavailable:
            lines.append(f"{r.name:<{width}} not measured: LLM unavailable (see log)")
            continue
        lines.append(
            f"{r.name:<{width}} {pct(r.mean('precision')):>6} {pct(r.mean('recall')):>6} "
            f"{pct(r.mean('judged')):>7}"
        )
    llm_rows = [r for r in results if r.mean("assessed") is not None]
    for r in llm_rows:
        if (cov := r.mean("assessed")) < 1:
            lines.append(
                f"\nNote: only {cov:.0%} of the {r.name} shortlist has cached assessments; "
                "unassessed papers are left out. Run without --cached-only to fill the cache."
            )
    if windows and (judged := results[0].mean("judged")) is not None and judged < 0.8:
        lines.append(
            "\nNote: many top-ranked papers are unlabelled and count as not relevant. "
            "Run `paperpulse label` to widen coverage."
        )
    return "\n".join(lines)


# -- saved results ---------------------------------------------------------

RESULT_SCHEMA_VERSION = "1"


def evaluation_record(
    results: list[VariantResult],
    config: Config,
    llms: list[LLM],
    labels_jsonl: str,
    k: int,
    cached_only: bool,
    now: datetime | None = None,
) -> dict:
    """
    Everything needed to trust or re-check a result: what was measured, with
    which setup, on which labels (by checksum of the exported label set), and
    which papers each variant picked.
    """
    incomplete = []
    if not llms:
        incomplete.append("no LLM variant evaluated")
    for r in results:
        if r.unavailable:
            incomplete.append(f"{r.name}: LLM unavailable")
        elif (cov := r.mean("assessed")) is not None and cov < 1:
            incomplete.append(f"{r.name}: only {cov:.0%} of shortlist assessed")

    profile_json = json.dumps(config.profile.model_dump(), sort_keys=True)
    label_lines = [json.loads(line) for line in labels_jsonl.splitlines() if line.strip()]
    windows = results[0].windows if results else []
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "created_at": (now or datetime.now(UTC)).isoformat(timespec="seconds"),
        "paperpulse_version": version("paperpulse"),
        "complete": not incomplete,
        "incomplete_reasons": incomplete,
        "setup": {
            "embedding_model": config.embedding.model,
            "llm_provider": config.llm.provider if llms else None,
            "llm_models": [llm.name for llm in llms],
            "prompt_version": PROMPT_VERSION,
            "profile_sha256": hashlib.sha256(profile_json.encode()).hexdigest()[:16],
            "k": k,
            "shortlist_size": config.llm.candidates,
            "min_relevance": config.llm.min_relevance,
            "window_days": config.selection.window_days,
            "rrf_k": config.retrieval.rrf_k,
            "cached_only": cached_only,
        },
        "labels": {
            "total": len(label_lines),
            "relevant": sum(1 for r in label_lines if r["relevant"]),
            "sha256": hashlib.sha256(labels_jsonl.encode()).hexdigest()[:16],
        },
        "summary": {
            r.name: None
            if r.unavailable
            else {m: r.mean(m) for m in ("precision", "recall", "judged")}
            for r in results
        },
        "windows": [
            {
                "start": w.start.isoformat(),
                "end": w.end.isoformat(),
                "relevant": w.relevant,
                "variants": {
                    r.name: {
                        key: value
                        for key, value in asdict(r.windows[i]).items()
                        if key not in ("start", "end", "relevant")
                    }
                    for r in results
                    if not r.unavailable
                },
            }
            for i, w in enumerate(windows)
        ],
    }


def save_evaluation(record: dict, labels_jsonl: str, directory: Path) -> Path:
    """
    Write the result as results/<timestamp>.json and refresh labels.jsonl next
    to it, so the label set that produced the newest result is always on disk.
    """
    results_dir = directory / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromisoformat(record["created_at"]).strftime("%Y-%m-%dT%H%M%S")
    path = results_dir / f"{stamp}.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    (directory / "labels.jsonl").write_text(labels_jsonl)
    return path
