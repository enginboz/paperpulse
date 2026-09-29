"""
Hard filters. They run before any model so that cheap, certain rules remove
noise (editorials, errata, repeats) before it can compete for a slot.
"""

from paperpulse.config import Selection
from paperpulse.models import Paper


def apply_filters(papers: list[Paper], selection: Selection, exclude_ids: set[str]) -> list[Paper]:
    excluded_types = {t.lower() for t in selection.exclude_publication_types}
    # Journals often file correction notices as ordinary "Journal Article"s,
    # so the title is the only reliable signal.
    excluded_prefixes = tuple(t.lower() for t in selection.exclude_title_prefixes)
    return [
        p
        for p in papers
        if p.id not in exclude_ids
        and not (selection.require_abstract and not p.abstract)
        and not excluded_types.intersection(t.lower() for t in p.publication_types)
        and not p.title.lower().startswith(excluded_prefixes)
    ]
