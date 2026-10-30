"""Versioned evaluation questions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    kind: str
    question: str
    criteria: str
    source_ids: list[str]
    relevant_quotes: list[str]
    expected_page: int | None = None

    @property
    def has_gold(self) -> bool:
        return bool(self.relevant_quotes)


def load_questions(path: Path) -> list[EvalQuestion]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_questions = payload["questions"]
    questions: list[EvalQuestion] = []
    for item in raw_questions:
        page = item.get("expected_page")
        questions.append(
            EvalQuestion(
                id=str(item["id"]),
                kind=str(item["kind"]),
                question=str(item["question"]),
                criteria=str(item["criteria"]),
                source_ids=[str(value) for value in item.get("source_ids", [])],
                relevant_quotes=[
                    str(value) for value in item.get("relevant_quotes", [])
                ],
                expected_page=int(page) if isinstance(page, int) else None,
            )
        )
    return questions


def unique_quotes(questions: list[EvalQuestion]) -> list[str]:
    seen: list[str] = []
    for question in questions:
        for quote in question.relevant_quotes:
            if quote not in seen:
                seen.append(quote)
    return seen
