"""Evaluate the versioned dataset. Provider flags exit 2 when they cannot run."""

from __future__ import annotations

from atlas.evaluation.runner import execute


def main(argv: list[str] | None = None) -> int:
    return execute(argv)


if __name__ == "__main__":
    raise SystemExit(main())
