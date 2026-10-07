"""
Human judgements, of two kinds kept strictly apart:

labels    from title and abstract, via `paperpulse label` or an imported set.
          The same information the ranking and the LLM see, so they are the
          reference standard for `eval`.
feedback  after reading the paper itself, via `paperpulse feedback`. A richer
          verdict ("was it worth reading?") that never enters the evaluation set.

Labels can be exported to and imported from JSONL so an evaluation set can be
versioned and shared.
"""

import json
from collections.abc import Callable, Iterable, Iterator
from datetime import UTC, datetime

from paperpulse.models import Feedback, Label, Paper
from paperpulse.ranking.assess import MAX_ABSTRACT_CHARS
from paperpulse.store import Store


def resolve_paper(store: Store, target: str) -> Paper | None:
    """A rank in the latest digest ("2"), a DOI, or a PubMed id ("pmid:123")."""
    if target.isdigit():
        digest = store.latest_run()
        ranked = {p.rank: p.id for p in digest.papers} if digest else {}
        return store.get_paper(ranked[int(target)]) if int(target) in ranked else None
    return store.get_paper(target.strip().lower())


def record_feedback(
    store: Store,
    paper: Paper,
    worth_reading: bool,
    note: str | None = None,
    now: datetime | None = None,
) -> Feedback:
    feedback = Feedback(
        paper_id=paper.id,
        worth_reading=worth_reading,
        note=note,
        given_at=now or datetime.now(UTC),
    )
    store.set_feedback(feedback)
    return feedback


def label_interactively(
    store: Store,
    papers: list[Paper],
    ask: Callable[[str], str] | None = None,
    show: Callable[[str], None] | None = None,
) -> int:
    """Ask about each paper until the pool is done or the user quits. Returns labels written."""
    # Resolved at call time, not as defaults, so input() can be replaced in tests.
    ask, show = ask or input, show or print
    written = 0
    for i, paper in enumerate(papers, 1):
        show(_describe(paper, i, len(papers)))
        while (answer := ask("Relevant? [y]es / [n]o / [s]kip / [q]uit: ").strip().lower()) not in {
            "y", "n", "s", "q",
        }:  # fmt: skip
            show("Please answer y, n, s or q.")
        if answer == "q":
            break
        if answer == "s":
            continue
        store.set_label(
            Label(
                paper_id=paper.id,
                relevant=answer == "y",
                source="pool",
                labelled_at=datetime.now(UTC),
            )
        )
        written += 1
    return written


def _describe(paper: Paper, i: int, total: int) -> str:
    meta = " · ".join(
        filter(
            None, [paper.journal, str(paper.published or ""), ", ".join(paper.publication_types)]
        )
    )
    abstract = paper.abstract
    # Show exactly as much as the LLM gets, so human and model judge the same text.
    if len(abstract) > MAX_ABSTRACT_CHARS:
        abstract = abstract[:MAX_ABSTRACT_CHARS] + " …"
    link = f"https://doi.org/{paper.doi}" if paper.doi else paper.url
    return f"\n[{i}/{total}] {meta}\n{paper.title}\n{link}\n\n{abstract}\n"


def export_labels(store: Store) -> Iterator[str]:
    """One JSON object per line, with enough bibliographic data to re-fetch each paper."""
    for label, paper in store.labelled_papers():
        yield json.dumps(
            {
                "id": paper.id,
                "doi": paper.doi,
                "pmid": paper.pmid,
                "title": paper.title,
                "added": paper.added.isoformat(),
                "relevant": label.relevant,
                "source": label.source,
                "note": label.note,
                "labelled_at": label.labelled_at.isoformat(),
            },
            ensure_ascii=False,
        )


def import_labels(
    store: Store,
    lines: Iterable[str],
    fetch_by_pmid: Callable[[list[str]], list[Paper]] | None = None,
) -> tuple[int, int]:
    """
    Import exported labels. Papers missing from the store are fetched by PMID
    when a fetcher is given. Returns (imported, skipped).
    """
    records = [json.loads(line) for line in lines if line.strip()]
    missing = [r["pmid"] for r in records if r.get("pmid") and store.get_paper(r["id"]) is None]
    if missing and fetch_by_pmid is not None:
        store.upsert_papers(fetch_by_pmid(missing))

    imported = skipped = 0
    for r in records:
        if store.get_paper(r["id"]) is None:
            skipped += 1
            continue
        store.set_label(
            Label(
                paper_id=r["id"],
                relevant=r["relevant"],
                source="import",
                note=r.get("note"),
                labelled_at=r.get("labelled_at") or datetime.now(UTC),
            )
        )
        imported += 1
    return imported, skipped
