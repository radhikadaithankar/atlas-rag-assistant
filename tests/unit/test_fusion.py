from __future__ import annotations

import uuid

from atlas.retrieval.fusion import reciprocal_rank_fusion
from atlas.retrieval.hits import Hit
from atlas.retrieval.keyword_search import search_keywords


def _hit(
    number: int, *, keyword_rank: int | None = None, vector_rank: int | None = None
) -> Hit:
    identifier = uuid.UUID(int=number)
    return Hit(
        chunk_id=identifier,
        document_id=identifier,
        filename=f"{number}.txt",
        page=None,
        section=None,
        text=f"text {number}",
        score=0.0,
        keyword_rank=keyword_rank,
        vector_rank=vector_rank,
    )


def test_rrf_sums_both_lists_and_keeps_a_one_sided_hit() -> None:
    first = _hit(1, vector_rank=1)
    second = _hit(2, vector_rank=2)
    second_keyword = _hit(2, keyword_rank=1)
    third = _hit(3, keyword_rank=2)
    fused = reciprocal_rank_fusion([[first, second], [second_keyword, third]], k=60)
    by_id = {hit.chunk_id: hit for hit in fused}
    assert by_id[second.chunk_id].score == pytest_approx(1 / 61 + 1 / 62)
    assert by_id[first.chunk_id].score == pytest_approx(1 / 61)
    assert by_id[third.chunk_id].score == pytest_approx(1 / 62)
    assert by_id[first.chunk_id].keyword_rank is None
    assert by_id[third.chunk_id].vector_rank is None
    assert by_id[second.chunk_id].keyword_rank == 1
    assert by_id[second.chunk_id].vector_rank == 2
    assert fused[0].chunk_id == second.chunk_id


def test_equal_scores_break_by_chunk_id() -> None:
    low = _hit(1, keyword_rank=1)
    high = _hit(2, vector_rank=1)
    fused = reciprocal_rank_fusion([[high], [low]], k=60)
    assert [hit.chunk_id for hit in fused] == [low.chunk_id, high.chunk_id]


def test_blank_keyword_query_does_not_touch_the_session() -> None:
    class Session:
        def execute(self, _statement: object) -> None:
            raise AssertionError("queried")

    assert search_keywords(Session(), "   ", top_k=5) == []  # type: ignore[arg-type]


def pytest_approx(value: float) -> float:
    return value


def test_scores_match_the_formula() -> None:
    fused = reciprocal_rank_fusion(
        [[_hit(1, vector_rank=1)], [_hit(1, keyword_rank=1)]], k=60
    )
    assert abs(fused[0].score - (1 / 61 + 1 / 61)) < 1e-12
