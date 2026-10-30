"""Liveness. This handler does not open a database connection."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["health"])


class HealthBody(BaseModel):
    status: str


@router.get("/health", response_model=HealthBody)
def health() -> HealthBody:
    return HealthBody(status="ok")
