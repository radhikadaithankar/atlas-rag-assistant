"""Grounded answers. Retrieved text is untrusted data."""

from atlas.generation.service import INSUFFICIENT_EVIDENCE, answer_from_passages

__all__ = ["INSUFFICIENT_EVIDENCE", "answer_from_passages"]
