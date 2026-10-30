"""Embedding providers. The protocol does no I/O."""

from atlas.embeddings.base import Embedder
from atlas.embeddings.factory import build_embedder

__all__ = ["Embedder", "build_embedder"]
