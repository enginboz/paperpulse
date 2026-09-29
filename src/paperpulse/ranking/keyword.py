"""
Keyword retrieval: BM25 per profile topic over titles and abstracts.

Embeddings blur exact terms, and acronyms such as FHIR, HL7 or OMOP carry
little meaning for them. BM25 weighs rare terms by inverse document frequency,
so a paper that literally mentions FHIR ranks high on the FHIR topic.

Each paper keeps its best BM25 score across topics, and papers are ranked by
that score. Ranking by best per-topic *rank* instead would hand every topic's
top hit rank 1, however weak: with a dozen topics, a paper matching only
"clinical" would tie with one matching "FHIR". Raw BM25 scores are sums of
per-term IDF weights, so they stay roughly comparable across topic queries.
"""

import re
from dataclasses import dataclass

from paperpulse.store import Store

STOPWORDS = {
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "into", "of",
    "on", "or", "the", "to", "via", "with", "without",
}  # fmt: skip


@dataclass
class KeywordMatch:
    rank: int
    score: float
    topic_index: int


def topic_query(topic: str) -> str | None:
    """Turn a topic into an FTS5 OR-query of quoted terms, so no input can break the syntax."""
    terms = [t for t in re.findall(r"[a-z0-9]+", topic.lower()) if t not in STOPWORDS]
    terms = list(dict.fromkeys(terms))
    return " OR ".join(f'"{t}"' for t in terms) or None


def best_keyword_matches(
    store: Store, topics: list[str], paper_ids: list[str]
) -> dict[str, KeywordMatch]:
    """Rank papers by their best BM25 score across topics. Papers matching no term are absent."""
    best: dict[str, tuple[float, int]] = {}
    for topic_index, topic in enumerate(topics):
        query = topic_query(topic)
        if query is None:
            continue
        for paper_id, score in store.keyword_search(query, paper_ids):
            if paper_id not in best or score > best[paper_id][0]:
                best[paper_id] = (score, topic_index)
    ordered = sorted(best.items(), key=lambda item: item[1][0], reverse=True)
    return {
        paper_id: KeywordMatch(rank, score, topic_index)
        for rank, (paper_id, (score, topic_index)) in enumerate(ordered, 1)
    }
