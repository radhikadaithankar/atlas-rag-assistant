"""Readiness. ``SELECT 1`` when a database URL is configured."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from atlas.config import Settings
from atlas.database.session import database_accepts_query

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ready"])


class ReadyBody(BaseModel):
    status: str


@router.get("/ready")
def ready(request: Request) -> JSONResponse:
    settings: Settings = request.app.state.settings
    try:
        ok = database_accepts_query(settings)
    except Exception as exc:
        logger.error("%s", type(exc).__name__)
        ok = False
    if not ok:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "reason": "database is unreachable"},
        )
    return JSONResponse(status_code=200, content={"status": "ready"})
