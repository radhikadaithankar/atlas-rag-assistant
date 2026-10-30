"""Ingest one file. Exit 2 when the database URL is missing. Exit 0 on a duplicate."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from atlas.config import (
    AtlasConfigError,
    load_settings,
    require_database_url,
    select_provider,
)
from atlas.database.session import session_scope
from atlas.embeddings.apply import embed_document
from atlas.embeddings.factory import build_embedder
from atlas.ingestion.errors import IngestionError, UploadTooLargeError
from atlas.ingestion.pipeline import prepare_upload, store_prepared

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest one local file into Atlas.")
    parser.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    try:
        settings = load_settings()
        require_database_url(settings)
    except AtlasConfigError as exc:
        logger.error("%s", type(exc).__name__)
        print(exc, file=sys.stderr)
        return 2
    try:
        data = args.path.read_bytes()
        prepared = prepare_upload(data, args.path.name, settings)
    except UploadTooLargeError as exc:
        print(exc.message, file=sys.stderr)
        return 1
    except IngestionError as exc:
        print(exc.message, file=sys.stderr)
        return 1
    except OSError as exc:
        logger.error("%s", type(exc).__name__)
        print("The file could not be read.", file=sys.stderr)
        return 1
    try:
        with session_scope(settings) as session:
            document_id, created = store_prepared(session, prepared, None)
    except IngestionError as exc:
        print(exc.message, file=sys.stderr)
        return 1
    except AtlasConfigError as exc:
        print(exc, file=sys.stderr)
        return 2
    if select_provider(settings) != "none":
        try:
            embedder = build_embedder(settings)
            with session_scope(settings) as session:
                embed_document(session, document_id, embedder, replace=False)
        except Exception as exc:
            logger.error("%s", type(exc).__name__)
            print("The document was stored. Embeddings were not.", file=sys.stderr)
            return 1
    print(f"id={document_id} created={str(created).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
