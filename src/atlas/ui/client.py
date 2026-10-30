"""HTTP client for the Atlas API.

Timeout is 30 seconds. Transport failures log the exception class only.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx

from atlas.api.schemas.ask import AskResponse
from atlas.api.schemas.documents import DocumentList, DocumentSummary
from atlas.api.schemas.search import SearchResponse

logger = logging.getLogger(__name__)

DEFAULT_API_URL = "http://127.0.0.1:8000"


class AtlasClientError(Exception):
    def __init__(self, status: int, message: str) -> None:
        self.status = status
        self.message = message
        super().__init__(message)


class AtlasClient:
    def __init__(
        self,
        base_url: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        url = base_url or os.environ.get("ATLAS_API_URL", DEFAULT_API_URL)
        self._owns_client = client is None
        self._client = client or httpx.Client(base_url=url, timeout=30.0)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def health(self) -> dict[str, str]:
        payload = self._json("GET", "/health")
        status = payload.get("status")
        if not isinstance(status, str):
            raise AtlasClientError(0, "The API could not be reached.")
        return {"status": status}

    def upload(self, filename: str, data: bytes) -> DocumentSummary:
        try:
            response = self._client.post(
                "/documents",
                files={"file": (filename, data, "application/octet-stream")},
            )
        except httpx.HTTPError as exc:
            logger.error("%s", type(exc).__name__)
            raise AtlasClientError(0, "The API could not be reached.") from None
        self._raise_for_status(response)
        return DocumentSummary.model_validate(response.json())

    def list_documents(self) -> DocumentList:
        return DocumentList.model_validate(self._json("GET", "/documents"))

    def delete_document(self, document_id: str) -> None:
        try:
            response = self._client.delete(f"/documents/{document_id}")
        except httpx.HTTPError as exc:
            logger.error("%s", type(exc).__name__)
            raise AtlasClientError(0, "The API could not be reached.") from None
        if response.status_code == 204:
            return
        self._raise_for_status(response)

    def search(
        self,
        query: str,
        *,
        mode: str = "keyword",
        top_k: int = 5,
        rerank: bool = False,
    ) -> SearchResponse:
        payload = self._json(
            "POST",
            "/search",
            json={"query": query, "mode": mode, "top_k": top_k, "rerank": rerank},
        )
        return SearchResponse.model_validate(payload)

    def ask(
        self,
        question: str,
        *,
        mode: str = "keyword",
        top_k: int = 5,
        rerank: bool = False,
    ) -> AskResponse:
        payload = self._json(
            "POST",
            "/ask",
            json={"question": question, "mode": mode, "top_k": top_k, "rerank": rerank},
        )
        return AskResponse.model_validate(payload)

    def _json(self, method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = self._client.request(method, url, **kwargs)
        except httpx.HTTPError as exc:
            logger.error("%s", type(exc).__name__)
            raise AtlasClientError(0, "The API could not be reached.") from None
        self._raise_for_status(response)
        body = response.json()
        if not isinstance(body, dict):
            raise AtlasClientError(response.status_code, "The API request failed.")
        return body

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.is_success:
            return
        raise AtlasClientError(response.status_code, plain_message(response))


def plain_message(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except json.JSONDecodeError:
        return "The API request failed."
    if isinstance(payload, dict):
        message = payload.get("message")
        if isinstance(message, str):
            return message
    return "The API request failed."


def chunk_label(count: int) -> str:
    noun = "chunk" if count == 1 else "chunks"
    return f"{count} {noun}"


def format_place(filename: str, page: int | None, section: str | None) -> str:
    if page is not None:
        return f"{filename}, page {page}"
    if section:
        return f"{filename}, section {section}"
    return filename


def ask_error_text(status: int, message: str) -> str:
    if status == 503 and "OPENAI_API_KEY" in message:
        return (
            "The API is missing OPENAI_API_KEY (503). "
            "Ask needs a key. Keyword search does not."
        )
    if status == 409:
        return "This file is already stored."
    if status == 413:
        return "This file is larger than the upload limit."
    if status == 422 and "empty" in message.lower():
        return "The file is empty."
    return message
