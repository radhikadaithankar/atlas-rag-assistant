"""Ingestion failures that are safe to show to a caller."""

from __future__ import annotations


class IngestionError(Exception):
    """A file was rejected or a write could not be finished.

    ``code`` is a stable token such as ``empty`` or ``unreadable``. The message
    does not include file bytes or a database URL.
    """

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


class UploadTooLargeError(Exception):
    def __init__(self, message: str = "The file exceeds MAX_UPLOAD_BYTES.") -> None:
        self.code = "too_large"
        self.message = message
        super().__init__(message)
