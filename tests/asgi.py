"""In-process HTTP. Async routes use HTTPX's ASGI transport."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

import httpx
from fastapi import FastAPI


def call(
    app: FastAPI,
    method: str,
    url: str,
    *,
    json: object | None = None,
    files: Mapping[str, object] | None = None,
    params: Mapping[str, str] | None = None,
) -> httpx.Response:
    async def _run() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            return await client.request(
                method, url, json=json, files=files, params=params
            )

    return asyncio.run(_run())


class SyncASGITransport(httpx.BaseTransport):
    """Sync wrapper around the async ASGI transport, for the Streamlit client."""

    def __init__(self, app: FastAPI) -> None:
        self._app = app

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        async def _handle() -> httpx.Response:
            transport = httpx.ASGITransport(app=self._app)
            response = await transport.handle_async_request(request)
            content = await response.aread()
            await response.aclose()
            return httpx.Response(
                status_code=response.status_code,
                headers=response.headers,
                content=content,
                request=request,
            )

        return asyncio.run(_handle())
