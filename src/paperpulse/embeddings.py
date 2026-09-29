"""
Text embeddings with a per-model cache in the store.

Vectors are L2-normalised, so cosine similarity is a plain dot product.
"""

import logging
from typing import Protocol

import numpy as np

from paperpulse.models import Paper
from paperpulse.store import Store

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    name: str

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) array of L2-normalised vectors."""
        ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        self.name = model_name
        self._model = None

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            # Imported lazily: torch takes seconds to load and ingest never needs it.
            from sentence_transformers import SentenceTransformer
            from transformers.utils import logging as hf_logging

            hf_logging.disable_progress_bar()

            logger.info("Loading embedding model %s", self.name)
            self._model = SentenceTransformer(self.name)
        return self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )


def embed_papers(store: Store, embedder: Embedder, papers: list[Paper]) -> np.ndarray:
    """Embed papers, computing only those missing from the cache. Rows follow `papers` order."""
    cached = store.get_embeddings([p.id for p in papers], embedder.name)
    missing = [p for p in papers if p.id not in cached]
    if missing:
        logger.info("Embedding %d new papers (%d cached)", len(missing), len(cached))
        fresh = embedder.encode([p.text for p in missing])
        computed = {p.id: v for p, v in zip(missing, fresh, strict=True)}
        store.put_embeddings(embedder.name, computed)
        cached.update(computed)
    return np.stack([cached[p.id] for p in papers])
