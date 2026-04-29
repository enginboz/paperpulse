"""
Embedding-based paper pre-filtering module.

Uses sentence-transformers to compute semantic similarity between
each paper's title+abstract and the physician's interest profile
(stage 1).

The top N papers by cosine similarity are passed to the LLM for
final qualitative selection (stage 2).

Model: pritamdeka/S-PubMedBert-MS-MARCO
  - Fine-tuned on PubMed abstracts, best fit for biomedical text
  - ~400MB, cached locally after first download (~/.cache/huggingface/)

Alternative for faster/lighter usage:
  - all-MiniLM-L6-v2 (80MB, good general-purpose semantic similarity)
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from paperpulse.config import INTEREST_PROFILE
from paperpulse.models import Paper

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Model to use for embeddings.
# Change to 'all-MiniLM-L6-v2' for a smaller, faster alternative.
EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"

# Cache the model in memory after first load to avoid reloading between calls
_model: SentenceTransformer | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_model() -> SentenceTransformer:
    """
    Load the embedding model, using a cached instance if available.
    First call downloads the model and caches it locally.
    """
    global _model
    if _model is None:
        print(f"Loading embedding model '{EMBEDDING_MODEL}'...")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        print("Model loaded.")
    return _model


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(
        np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    )


def _paper_text(paper: Paper) -> str:
    """
    Combine title and abstract into a single string for embedding.
    Title is included first as it contains concentrated key terms.
    Abstract is truncated to keep embedding time reasonable.
    """
    return f"{paper.title}. {paper.abstract[:1000]}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def shortlist_by_embedding(papers: list[Paper], n: int = 15) -> list[Paper]:
    """
    Rank papers by semantic similarity to the interest profile and
    return the top N candidates.

    Uses cosine similarity between the interest profile embedding and
    each paper's title+abstract embedding.

    Args:
        papers: List of Paper objects from the PubMed fetcher.
        n: Number of top candidates to return (default: 15).

    Returns:
        List of up to N Paper objects, sorted by relevance score descending.
        Each Paper has its embedding_score field set.
    """
    if not papers:
        return []

    model = _get_model()

    print(f"Computing embeddings for {len(papers)} papers...")

    # Embed the interest profile
    profile_vec = model.encode(INTEREST_PROFILE, show_progress_bar=False, convert_to_numpy=True)

    # Embed all papers (batched for efficiency)
    paper_texts = [_paper_text(p) for p in papers]
    paper_vecs = model.encode(paper_texts, show_progress_bar=False, batch_size=32, convert_to_numpy=True)

    # Compute cosine similarity and set score on each paper
    for i, paper in enumerate(papers):
        paper.embedding_score = round(_cosine_similarity(paper_vecs[i], profile_vec), 4)

    # Sort by score descending and return top N
    sorted_papers = sorted(papers, key=lambda p: p.embedding_score, reverse=True)
    top_n = sorted_papers[:n]

    print(f"Top {len(top_n)} papers by embedding similarity:")
    for p in top_n:
        print(f"  {p.embedding_score:.3f} — {p.title[:70]}")

    return top_n