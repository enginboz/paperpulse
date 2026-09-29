"""
Dense retrieval: each paper is scored by its best-matching profile topic.

Taking the max over topics, instead of comparing against one averaged
profile vector, keeps niche interests from being drowned out by broad ones.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class DenseMatch:
    score: float
    topic_index: int


def best_topic_matches(paper_vectors: np.ndarray, topic_vectors: np.ndarray) -> list[DenseMatch]:
    similarity = paper_vectors @ topic_vectors.T
    best = similarity.argmax(axis=1)
    return [DenseMatch(float(similarity[i, t]), int(t)) for i, t in enumerate(best)]
