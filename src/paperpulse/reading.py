"""
A digest as a reading list for the terminal: what to read, why, and where.
"""

import textwrap

from paperpulse.models import Candidate, Digest, RankedPaper

WIDTH = 88
INDENT = "   "
EXTRA_MIN_RELEVANCE = 4


def format_reading_list(digest: Digest, extras: bool = True) -> str:
    run = digest.run
    created = run.created_at.astimezone()
    lines = [
        f"PaperPulse · {created:%d %b %Y} · {len(digest.papers)} paper(s) · "
        f"{run.llm_model or 'no LLM'}",
        f"Papers added to PubMed {run.window_start} to {run.window_end}",
    ]
    if not digest.papers:
        lines.append("\nNothing in this digest. Quiet weeks yield fewer picks rather than filler.")

    for paper in digest.papers:
        lines.append("")
        lines.extend(_entry(f"{paper.rank}. ", paper))

    others = [
        c
        for c in digest.shortlist
        if not c.selected
        and c.score.assessment
        and c.score.assessment.relevance >= EXTRA_MIN_RELEVANCE
    ]
    if extras and others:
        lines.append(f"\nAlso rated {EXTRA_MIN_RELEVANCE}+ by the LLM, but not in the top picks:")
        for c in others:
            lines.append("")
            lines.extend(_entry("-  ", c, show_id=True))

    lines.append("\nAfter reading: paperpulse feedback <rank or id> up|down [--note ...]")
    return "\n".join(lines)


def _entry(prefix: str, paper: RankedPaper | Candidate, show_id: bool = False) -> list[str]:
    assessment = paper.score.assessment
    lines = textwrap.wrap(
        paper.title, WIDTH, initial_indent=prefix, subsequent_indent=" " * len(prefix)
    )
    meta = [paper.journal]
    if getattr(paper, "published", None):
        meta.append(str(paper.published))
    if assessment:
        meta.append(assessment.study_type.replace("_", " "))
        meta.append(f"relevance {assessment.relevance}/5")
    lines.append(INDENT + " · ".join(m for m in meta if m))
    if assessment:
        lines.extend(
            textwrap.wrap(
                assessment.rationale, WIDTH, initial_indent=INDENT, subsequent_indent=INDENT
            )
        )
    lines.append(INDENT + _link(paper))
    if show_id:
        lines.append(f"{INDENT}id: {paper.id}")
    return lines


def _link(paper: RankedPaper | Candidate) -> str:
    """DOI links survive publisher moves; fall back to PubMed."""
    doi = getattr(paper, "doi", None) or (None if paper.id.startswith("pmid:") else paper.id)
    return f"https://doi.org/{doi}" if doi else paper.url
