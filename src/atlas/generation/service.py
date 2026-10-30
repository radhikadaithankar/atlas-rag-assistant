"""Ask the model, then drop citations that the retrieved passages do not support."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from atlas.generation.base import GenerationResult, Generator
from atlas.generation.citations import (
    KeptCitation,
    parse_model_json,
    validate_citations,
)
from atlas.generation.errors import GenerationProviderError
from atlas.generation.prompts import build_prompt
from atlas.retrieval.hits import Hit

logger = logging.getLogger(__name__)

INSUFFICIENT_EVIDENCE = (
    "The retrieved passages do not contain enough evidence to answer this question."
)


@dataclass(frozen=True)
class Answer:
    answer: str
    abstained: bool
    citations: list[KeptCitation]
    latency_ms: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    estimated_cost: None = None


def answer_from_passages(
    question: str, passages: list[Hit], generator: Generator
) -> Answer:
    if not passages:
        return _abstain(None)
    system, user = build_prompt(question, passages)
    try:
        result = generator.generate(system, user)
    except GenerationProviderError:
        raise
    except Exception as exc:
        logger.error("%s", type(exc).__name__)
        raise GenerationProviderError() from None
    parsed = parse_model_json(result.text)
    if parsed is None or parsed.abstained:
        _log_success(result, abstained=True)
        return _abstain(result)
    citations = validate_citations(parsed.citations, passages)
    if not citations:
        _log_success(result, abstained=True)
        return _abstain(result)
    _log_success(result, abstained=False)
    return Answer(
        answer=parsed.answer,
        abstained=False,
        citations=citations,
        latency_ms=result.latency_ms,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
    )


def _abstain(result: GenerationResult | None) -> Answer:
    if result is None:
        return Answer(INSUFFICIENT_EVIDENCE, True, [], None, None, None)
    return Answer(
        INSUFFICIENT_EVIDENCE,
        True,
        [],
        result.latency_ms,
        result.prompt_tokens,
        result.completion_tokens,
    )


def _log_success(result: GenerationResult, *, abstained: bool) -> None:
    logger.info(
        "latency_ms=%s prompt_tokens=%s completion_tokens=%s abstained=%s",
        result.latency_ms,
        result.prompt_tokens,
        result.completion_tokens,
        abstained,
    )
