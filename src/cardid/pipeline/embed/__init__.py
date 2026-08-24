"""Visual embedding backends for parallel disambiguation."""

from .base import (
    EmbeddingBackend,
    NullEmbeddingBackend,
    cosine_similarity,
    rerank,
)

__all__ = [
    "EmbeddingBackend",
    "NullEmbeddingBackend",
    "cosine_similarity",
    "rerank",
]
