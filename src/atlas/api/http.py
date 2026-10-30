"""JSON errors. Bodies name a code and a short message, never a secret."""

from __future__ import annotations

from fastapi.responses import JSONResponse


def error_json(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": code, "message": message})
