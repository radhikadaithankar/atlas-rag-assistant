"""Run the saved dataset.

Provider-backed flags exit 2 when they cannot run, and they do not write a
score. A pending row cannot be given a numeric value.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

from atlas.config import Settings, load_settings, select_provider
from atlas.evaluation.answer_metrics import (
    FAITHFULNESS_NOT_GROUND_TRUTH,
    assign_metric,
    citation_form_ok,
    citation_precision,
    estimate_gemini_cost,
    linear_percentile,
)
from atlas.evaluation.dataset import EvalQuestion, load_questions, unique_quotes
from atlas.evaluation.retrieval_metrics import (
    containment,
    mean,
    recall_at_k,
    reciprocal_rank,
)
from atlas.generation.base import Generator
from atlas.generation.citations import parse_model_json
from atlas.generation.prompts import build_prompt
from atlas.ingestion.chunker import chunk_parsed
from atlas.ingestion.parser import parse_bytes
from atlas.retrieval.hits import Hit

CORPUS_SUFFIXES = {".md", ".txt", ".pdf"}


@dataclass
class RunFlags:
    vector: bool = False
    hybrid: bool = False
    rerank: bool = False
    answers: bool = False


def preflight(flags: RunFlags, settings: Settings) -> str | None:
    """Return a message when the process must exit 2 without writing a score."""

    needs_provider = flags.vector or flags.hybrid or flags.answers or flags.rerank
    if needs_provider and select_provider(settings) == "none":
        return "A provider key is required. No score was written."
    if flags.rerank and not _rerank_extra_installed():
        return "The rerank extra is not installed. No score was written."
    if not settings.database_url:
        return "DATABASE_URL is not set. No score was written."
    return None


def _rerank_extra_installed() -> bool:
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False
    return True


def chunk_corpus(corpus: Path, *, size: int, overlap: int) -> list[str]:
    texts: list[str] = []
    for path in sorted(corpus.iterdir()):
        if path.suffix.lower() not in CORPUS_SUFFIXES or not path.is_file():
            continue
        parsed = parse_bytes(path.read_bytes())
        for chunk in chunk_parsed(parsed, size=size, overlap=overlap):
            texts.append(chunk.text)
    return texts


def score_containment(corpus: Path, quotes: list[str]) -> dict[str, object]:
    wide, wide_found, total = containment(
        chunk_corpus(corpus, size=1000, overlap=200),
        quotes,
    )
    narrow, narrow_found, _total = containment(
        chunk_corpus(corpus, size=400, overlap=80),
        quotes,
    )
    return {
        "chunk_containment_1000_200": {
            "status": "measured",
            "value": wide,
            "detail": (
                f"{wide_found} of {total} unique gold quotes. This is not retrieval quality."
            ),
        },
        "chunk_containment_400_80": {
            "status": "measured",
            "value": narrow,
            "detail": (
                f"{narrow_found} of {total} unique gold quotes. This is not retrieval quality."
            ),
        },
    }


def retrieval_summary(rows: list[tuple[float, float]]) -> tuple[float, float]:
    recalls = [item[0] for item in rows]
    ranks = [item[1] for item in rows]
    return mean(recalls), mean(ranks)


def deterministic_abstention(hit_counts: list[int]) -> float:
    """1 when every unanswerable question retrieved nothing. The generator is not called."""

    if not hit_counts:
        raise ValueError("abstention requires unanswerable questions")
    return sum(1 for count in hit_counts if count == 0) / len(hit_counts)


def judge_supported(answer: str, passages: list[str], judge: Generator) -> bool:
    """Ask whether the answer is supported. The boolean is not ground truth."""

    system = (
        "Decide whether the answer is supported by the passages. "
        'Reply with JSON {"supported": true} or {"supported": false}. '
        + FAITHFULNESS_NOT_GROUND_TRUTH
    )
    fenced = build_prompt(
        "Judge the answer against these passages.", _judge_hits(passages)
    )
    user = f"Answer:\n{answer}\n\n{fenced[1]}"
    result = judge.generate(system, user)
    try:
        payload = json.loads(result.text)
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict):
        return False
    return payload.get("supported") is True


def _judge_hits(passages: list[str]) -> list[Hit]:
    hits: list[Hit] = []
    for index, text in enumerate(passages):
        hits.append(
            Hit(
                chunk_id=uuid.UUID(int=index + 1),
                document_id=uuid.UUID(int=index + 1),
                filename="passage.txt",
                page=None,
                section=None,
                text=text,
                score=0.0,
            )
        )
    return hits


def citation_form_rate(
    records: list[tuple[bool, list[tuple[str, str]], dict[str, str]]],
) -> float:
    if not records:
        raise ValueError("citation form requires records")
    good = sum(
        1
        for abstained, citations, retrieved in records
        if citation_form_ok(
            abstained=abstained, citations=citations, retrieved=retrieved
        )
    )
    return good / len(records)


def answer_precision_mean(
    records: list[tuple[bool, list[str], list[str]]],
) -> float:
    if not records:
        raise ValueError("citation precision requires records")
    scores = [
        citation_precision(
            abstained=abstained, citation_quotes=quotes, gold_quotes=gold
        )
        for abstained, quotes, gold in records
    ]
    return mean(scores)


def latency_summary(latencies_ms: list[float]) -> tuple[float, float]:
    return linear_percentile(latencies_ms, 0.5), linear_percentile(latencies_ms, 0.95)


def cost_from_tokens(prompt_tokens: int, completion_tokens: int) -> float:
    return estimate_gemini_cost(prompt_tokens, completion_tokens)


def refuse_pending(row: dict[str, object], value: float) -> None:
    assign_metric(row, value)


def pending_row(reason: str, command: str) -> dict[str, object]:
    return {"status": "pending", "value": None, "reason": reason, "command": command}


def measured_row(value: float, detail: str) -> dict[str, object]:
    row: dict[str, object] = {"status": "measured", "value": None, "detail": detail}
    assign_metric(row, value)
    return row


def parse_model_supported(text: str) -> bool | None:
    parsed = parse_model_json(text)
    if parsed is None:
        return None
    return not parsed.abstained


def execute(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the saved Atlas dataset.")
    parser.add_argument("--vector", action="store_true")
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--answers", action="store_true")
    parser.add_argument("--dataset", type=Path, default=Path("data/evaluation/v1"))
    parser.add_argument("--results", type=Path, default=None)
    args = parser.parse_args(argv)
    flags = RunFlags(args.vector, args.hybrid, args.rerank, args.answers)
    try:
        settings = load_settings()
    except Exception as exc:
        print(type(exc).__name__, file=sys.stderr)
        return 2
    message = preflight(flags, settings)
    if message is not None:
        print(message, file=sys.stderr)
        return 2
    destination = args.results or (args.dataset / "results.json")
    try:
        payload = run_saved_dataset(settings, flags, args.dataset)
    except Exception as exc:
        print(type(exc).__name__, file=sys.stderr)
        return 2
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report_path = Path("docs/evaluation_report.md")
    if report_path.parent.is_dir():
        from atlas.evaluation.report import render_report

        report_path.write_text(render_report(payload), encoding="utf-8")
    return 0


def run_saved_dataset(
    settings: Settings, flags: RunFlags, dataset: Path
) -> dict[str, object]:
    """Ingest the corpus, score it, and delete the rows this run inserted.

    This writes nothing. The caller stores the document after a successful return.
    """

    from atlas.database.repository import delete_document
    from atlas.database.session import session_scope
    from atlas.embeddings.factory import build_embedder
    from atlas.embeddings.vectors import require_width
    from atlas.ingestion.pipeline import prepare_upload, store_prepared
    from atlas.retrieval.keyword_search import search_keywords
    from atlas.retrieval.reranker import load_cross_encoder

    questions = load_questions(dataset / "questions.json")
    quotes = unique_quotes(questions)
    gold = [item for item in questions if item.has_gold]
    unanswerable = [item for item in questions if not item.has_gold]
    metrics: dict[str, object] = score_containment(dataset / "corpus", quotes)
    needs_embed = flags.vector or flags.hybrid or flags.answers or flags.rerank
    embedder = None
    if needs_embed:
        embedder = build_embedder(settings)
        require_width(embedder.embed(["atlas evaluation probe"]))
    created_ids: list[uuid.UUID] = []
    try:
        with session_scope(settings) as session:
            for path in sorted((dataset / "corpus").iterdir()):
                if path.suffix.lower() not in CORPUS_SUFFIXES or not path.is_file():
                    continue
                prepared = prepare_upload(path.read_bytes(), path.name, settings)
                document_id, created = store_prepared(session, prepared, None)
                if created:
                    created_ids.append(document_id)
                if embedder is not None:
                    from atlas.embeddings.apply import embed_document

                    embed_document(session, document_id, embedder, replace=False)
            keyword_rows: list[tuple[float, float]] = []
            for question in gold:
                hits = search_keywords(session, question.question, top_k=5)
                texts = [hit.text for hit in hits]
                keyword_rows.append(
                    (
                        recall_at_k(texts, question.relevant_quotes, k=5),
                        reciprocal_rank(texts, question.relevant_quotes),
                    )
                )
            empty_hits = [
                len(search_keywords(session, question.question, top_k=5))
                for question in unanswerable
            ]
            recall, mrr = retrieval_summary(keyword_rows)
            metrics["keyword_recall_at_5"] = measured_row(
                recall,
                "top_k 5, chunk size 1000, overlap 200. Tied ranks follow chunk id, "
                "so a new ingest can move a tied hit across the cut.",
            )
            metrics["keyword_mrr"] = measured_row(
                mrr, "Same keyword run as keyword_recall_at_5."
            )
            metrics["deterministic_abstention"] = measured_row(
                deterministic_abstention(empty_hits),
                "Empty keyword retrieval. The generator is not called. "
                "This is not model abstention quality.",
            )
            if flags.vector and embedder is not None:
                rows = _ranked(
                    session,
                    gold,
                    mode="vector",
                    embedder=embedder,
                    top_k=5,
                    settings=settings,
                )
                metrics["vector_recall_at_5"] = measured_row(
                    rows[0],
                    "cosine, top_k 5, output dimensionality 1536.",
                )
                metrics["vector_mrr"] = measured_row(
                    rows[1], "Same vector run as vector_recall_at_5."
                )
            else:
                metrics["vector_recall_at_5"] = pending_row(
                    "Query embedding needs a key.",
                    "uv run python scripts/evaluate.py --vector",
                )
                metrics["vector_mrr"] = pending_row(
                    "Query embedding needs a key.",
                    "uv run python scripts/evaluate.py --vector",
                )
            if (flags.hybrid or flags.answers or flags.rerank) and embedder is not None:
                rows = _ranked(
                    session,
                    gold,
                    mode="hybrid",
                    embedder=embedder,
                    top_k=5,
                    settings=settings,
                )
                metrics["hybrid_recall_at_5"] = measured_row(
                    rows[0],
                    "reciprocal rank fusion, top_k 5, RRF_K 60.",
                )
                metrics["hybrid_mrr"] = measured_row(
                    rows[1], "Same hybrid run as hybrid_recall_at_5."
                )
            else:
                metrics["hybrid_recall_at_5"] = pending_row(
                    "Hybrid embeds the query.",
                    "uv run python scripts/evaluate.py --hybrid",
                )
                metrics["hybrid_mrr"] = pending_row(
                    "Hybrid embeds the query.",
                    "uv run python scripts/evaluate.py --hybrid",
                )
            if flags.rerank and embedder is not None:
                reranker = load_cross_encoder()
                rows = _ranked(
                    session,
                    gold,
                    mode="hybrid",
                    embedder=embedder,
                    top_k=5,
                    settings=settings,
                    reranker=reranker,
                    pool=settings.rerank_pool,
                )
                metrics["rerank_recall_at_5"] = measured_row(
                    rows[0],
                    "hybrid pool of 20, then cross-encoder/ms-marco-MiniLM-L6-v2 cut to 5.",
                )
                metrics["rerank_mrr"] = measured_row(
                    rows[1], "Same rerank run as rerank_recall_at_5."
                )
            else:
                metrics["rerank_recall_at_5"] = pending_row(
                    "Cross-encoder weights were not downloaded.",
                    "uv run python scripts/evaluate.py --rerank",
                )
                metrics["rerank_mrr"] = pending_row(
                    "Cross-encoder weights were not downloaded.",
                    "uv run python scripts/evaluate.py --rerank",
                )
            if flags.answers and embedder is not None:
                _fill_answer_metrics(
                    metrics,
                    session=session,
                    questions=questions,
                    gold=gold,
                    settings=settings,
                    embedder=embedder,
                )
            else:
                _pending_answers(metrics)
            for document_id in created_ids:
                delete_document(session, document_id)
            created_ids.clear()
    finally:
        if created_ids:
            with session_scope(settings) as session:
                for document_id in created_ids:
                    delete_document(session, document_id)
    return {
        "version": "v1",
        "provenance": (
            "Measured by scripts/evaluate.py on this corpus. "
            "Replace the committed snapshot only when this run is the one you intend to publish."
        ),
        "questions": len(questions),
        "gold_quotes": len(quotes),
        "keyword_questions": len(gold),
        "embedding_model": settings.embedding_model,
        "generation_model": settings.generation_model,
        "output_dimensionality": settings.embedding_dimensions,
        "metrics": metrics,
        "notes": [FAITHFULNESS_NOT_GROUND_TRUTH],
    }


def _ranked(
    session: object,
    gold: list[EvalQuestion],
    *,
    mode: str,
    embedder: object,
    top_k: int,
    settings: Settings,
    reranker: object | None = None,
    pool: int | None = None,
) -> tuple[float, float]:
    from typing import cast

    from sqlalchemy.orm import Session

    from atlas.embeddings.base import Embedder
    from atlas.evaluation.dataset import EvalQuestion
    from atlas.retrieval.reranker import Reranker, apply_rerank
    from atlas.retrieval.service import retrieve

    if not isinstance(session, Session):
        raise TypeError("retrieval dependencies are missing")
    active_embedder = cast(Embedder, embedder)
    rows: list[tuple[float, float]] = []
    limit = pool if pool is not None else top_k
    for question in gold:
        if not isinstance(question, EvalQuestion):
            continue
        hits = retrieve(
            session,
            query=question.question,
            mode=mode,
            top_k=limit,
            document_id=None,
            embedder=active_embedder,
            rrf_k=settings.rrf_k,
        )
        if reranker is not None:
            hits = apply_rerank(
                cast(Reranker, reranker), question.question, hits, top_k=top_k
            )
        texts = [hit.text for hit in hits]
        rows.append(
            (
                recall_at_k(texts, question.relevant_quotes, k=top_k),
                reciprocal_rank(texts, question.relevant_quotes),
            )
        )
    return retrieval_summary(rows)


def _fill_answer_metrics(
    metrics: dict[str, object],
    *,
    session: object,
    questions: list[EvalQuestion],
    gold: list[EvalQuestion],
    settings: Settings,
    embedder: object,
) -> None:
    from typing import cast

    from sqlalchemy.orm import Session

    from atlas.embeddings.base import Embedder
    from atlas.evaluation.dataset import EvalQuestion
    from atlas.generation.factory import build_generator
    from atlas.generation.service import answer_from_passages
    from atlas.retrieval.service import retrieve

    if not isinstance(session, Session):
        raise TypeError("answer dependencies are missing")
    embedder = cast(Embedder, embedder)
    generator = build_generator(settings)
    judge = build_generator(settings)
    precisions: list[tuple[bool, list[str], list[str]]] = []
    forms: list[tuple[bool, list[tuple[str, str]], dict[str, str]]] = []
    latencies: list[float] = []
    prompt_tokens = 0
    completion_tokens = 0
    supported = 0
    judged = 0
    for question in questions:
        if not isinstance(question, EvalQuestion):
            continue
        hits = retrieve(
            session,
            query=question.question,
            mode="hybrid",
            top_k=5,
            document_id=None,
            embedder=embedder,
            rrf_k=settings.rrf_k,
        )
        answer = answer_from_passages(question.question, hits, generator)
        if answer.latency_ms is not None:
            latencies.append(answer.latency_ms)
        if answer.prompt_tokens is not None:
            prompt_tokens += answer.prompt_tokens
        if answer.completion_tokens is not None:
            completion_tokens += answer.completion_tokens
        retrieved = {str(hit.chunk_id): hit.text for hit in hits}
        citations = [(str(item.chunk_id), item.quote) for item in answer.citations]
        forms.append((answer.abstained, citations, retrieved))
        if question.has_gold:
            precisions.append(
                (
                    answer.abstained,
                    [item.quote for item in answer.citations],
                    question.relevant_quotes,
                )
            )
        passages = [hit.text for hit in hits]
        if judge_supported(answer.answer, passages, judge):
            supported += 1
        judged += 1
    if precisions:
        metrics["citation_correctness"] = measured_row(
            answer_precision_mean(precisions),
            "Mean precision over questions with gold. Abstaining on a gold question scores 0.",
        )
    metrics["deterministic_citation_form"] = measured_row(
        citation_form_rate(forms),
        f"{sum(1 for row in forms if citation_form_ok(abstained=row[0], citations=row[1], retrieved=row[2]))} "
        f"of {len(forms)}.",
    )
    if latencies:
        median, p95 = latency_summary(latencies)
        metrics["latency_median_ms"] = measured_row(
            median,
            "Median of generator calls, linear interpolation on the sorted latencies.",
        )
        metrics["latency_p95_ms"] = measured_row(
            p95,
            "P95 of generator calls, linear interpolation on the sorted latencies.",
        )
    metrics["prompt_tokens"] = measured_row(
        float(prompt_tokens), "Sum of prompt tokens."
    )
    metrics["completion_tokens"] = measured_row(
        float(completion_tokens),
        "candidates_token_count plus thoughts_token_count only when that field is an int.",
    )
    provider = select_provider(settings)
    if provider == "gemini":
        cost = cost_from_tokens(prompt_tokens, completion_tokens)
    elif provider == "openai":
        cost = prompt_tokens / 1_000_000 * 0.10 + completion_tokens / 1_000_000 * 0.50
    else:
        cost = None
    if cost is None:
        metrics["estimated_cost"] = pending_row(
            "Token counts were not measured, so the price was not multiplied.",
            "uv run python scripts/evaluate.py --answers",
        )
    else:
        metrics["estimated_cost"] = measured_row(
            cost,
            "Published price times measured tokens, not an invoice. "
            "The API field estimated_cost stays null.",
        )
    metrics["answer_faithfulness"] = measured_row(
        supported / judged if judged else 0.0,
        f"{supported} of {judged} stored answers were judged supported. {FAITHFULNESS_NOT_GROUND_TRUTH}",
    )
    _ = gold


def _pending_answers(metrics: dict[str, object]) -> None:
    command = "uv run python scripts/evaluate.py --answers"
    metrics["citation_correctness"] = pending_row(
        "No model answers, so citations were not checked.",
        command,
    )
    metrics["deterministic_citation_form"] = pending_row(
        "No model answers, so citation form was not checked.",
        command,
    )
    metrics["answer_faithfulness"] = pending_row(
        "Model answers were not generated. An LLM judge was not run. "
        "A judge score would not be ground truth.",
        command,
    )
    metrics["latency_median_ms"] = pending_row("The generator was not called.", command)
    metrics["latency_p95_ms"] = pending_row("The generator was not called.", command)
    metrics["prompt_tokens"] = pending_row("The provider was not called.", command)
    metrics["completion_tokens"] = pending_row("The provider was not called.", command)
    metrics["estimated_cost"] = pending_row(
        "Token counts were not measured, so the price was not multiplied.",
        command,
    )
