"""Render ``docs/evaluation_report.md`` from a results document.

The renderer prints the values it is given. It does not compute a new score.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_results(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("results must be an object")
    return payload


def render_report(results: dict[str, object]) -> str:
    metrics = results.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("results.metrics must be an object")
    lines = [
        "# Atlas evaluation report",
        "",
        str(results.get("provenance", "")),
        "",
        (
            f"{results.get('questions')} questions, "
            f"{results.get('gold_quotes')} unique gold quotes, "
            f"{results.get('keyword_questions')} questions with gold. "
            f"Embedding `{results.get('embedding_model')}`, "
            f"generation `{results.get('generation_model')}`, "
            f"output dimensionality {results.get('output_dimensionality')}."
        ),
        "",
        "These figures are the measured run recorded in the project milestone reports. "
        "This file does not add scores from a later run.",
        "",
        "| Metric | Status | Value | Detail |",
        "| --- | --- | --- | --- |",
    ]
    for name, raw in metrics.items():
        if not isinstance(raw, dict):
            continue
        status = raw.get("status")
        value = raw.get("value")
        detail = raw.get("detail", "")
        shown = (
            "pending" if status == "pending" or value is None else _format_value(value)
        )
        lines.append(f"| `{name}` | {status} | {shown} | {detail} |")
    lines.extend(
        [
            "",
            "## Notes copied with the scores",
            "",
        ]
    )
    notes = results.get("notes")
    if isinstance(notes, list):
        for note in notes:
            lines.append(f"- {note}")
    lines.append("")
    faithfulness = metrics.get("answer_faithfulness")
    if isinstance(faithfulness, dict):
        lines.append(str(faithfulness.get("detail", "")))
        lines.append("")
        lines.append(
            "Faithfulness is not ground truth. The judge can agree with a fluent unsupported answer."
        )
        lines.append("")
    return "\n".join(lines)


def _format_value(value: object) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int | float):
        return json.dumps(value)
    return str(value)


def write_report(results_path: Path, report_path: Path) -> str:
    text = render_report(load_results(results_path))
    report_path.write_text(text, encoding="utf-8")
    return text
