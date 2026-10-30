"""Optional cross-encoder. Weights load only when a request asks for rerank."""

from __future__ import annotations

import logging
from typing import Protocol

from atlas.retrieval.hits import Hit

logger = logging.getLogger(__name__)

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


class Reranker(Protocol):
    def rerank(self, query: str, passages: list[Hit]) -> list[Hit]:
        """Return passages in the new order, with this reranker's scores."""


class RerankUnavailableError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


def load_cross_encoder() -> Reranker:
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RerankUnavailableError(
            "Cross-encoder reranking is unavailable because the rerank extra is not installed."
        ) from exc
    model = CrossEncoder(CROSS_ENCODER_MODEL)

    class _CrossEncoderReranker:
        def rerank(self, query: str, passages: list[Hit]) -> list[Hit]:
            if not passages:
                return []
            pairs = [(query, passage.text) for passage in passages]
            scores = model.predict(pairs)
            ordered = sorted(
                zip(passages, scores, strict=True),
                key=lambda item: float(item[1]),
                reverse=True,
            )
            reranked: list[Hit] = []
            for passage, score in ordered:
                passage.score = float(score)
                reranked.append(passage)
            return reranked

    return _CrossEncoderReranker()


def apply_rerank(
    reranker: Reranker, query: str, passages: list[Hit], *, top_k: int
) -> list[Hit]:
    """Rescore, then cut back to top-k. An empty result stays empty."""

    if not passages:
        return []
    return reranker.rerank(query, passages)[:top_k]
