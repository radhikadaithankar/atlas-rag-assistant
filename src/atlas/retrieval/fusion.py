"""Reciprocal Rank Fusion.

The score is the sum of ``1 / (k + rank)`` with 1-based ranks. A hit that is
missing from a list still scores from the lists that contain it, and the
missing rank stays null. Equal scores break by chunk id.
"""

from __future__ import annotations

import uuid
from dataclasses import replace

from atlas.retrieval.hits import Hit


def reciprocal_rank_fusion(ranked_lists: list[list[Hit]], *, k: int) -> list[Hit]:
    combined: dict[uuid.UUID, Hit] = {}
    scores: dict[uuid.UUID, float] = {}
    for hits in ranked_lists:
        for rank, hit in enumerate(hits, start=1):
            current = combined.get(hit.chunk_id)
            if current is None:
                current = replace(hit)
                combined[hit.chunk_id] = current
                scores[hit.chunk_id] = 0.0
            if hit.keyword_rank is not None:
                current.keyword_rank = hit.keyword_rank
            if hit.vector_rank is not None:
                current.vector_rank = hit.vector_rank
            scores[hit.chunk_id] += 1.0 / (k + rank)
    for chunk_id, hit in combined.items():
        hit.score = scores[chunk_id]
    return sorted(combined.values(), key=lambda item: (-item.score, str(item.chunk_id)))
