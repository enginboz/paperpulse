"""
Reciprocal rank fusion (Cormack et al., 2009).

Combines rankings using positions only, so cosine similarities and BM25
scores never have to be made comparable. A paper ranked r in a list earns
1 / (k + r); the constant k dampens the advantage of the very top ranks.
"""


def reciprocal_rank_fusion(rankings: list[dict[str, int]], k: int = 60) -> dict[str, float]:
    """`rankings` map item -> 1-based rank; items missing from a ranking earn nothing from it."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for item, rank in ranking.items():
            fused[item] = fused.get(item, 0.0) + 1.0 / (k + rank)
    return fused
