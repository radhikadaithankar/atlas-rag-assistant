"""Process entrypoint used by the API container.

Boots when the provider key is absent. Refuses to boot when ``DATABASE_URL``
is missing or invalid, and the log line does not include the value.
"""

from __future__ import annotations

import logging

from atlas.config import (
    AtlasConfigError,
    load_settings,
    require_database_url,
    select_provider,
)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("atlas.serve")
    try:
        settings = load_settings()
        require_database_url(settings)
    except AtlasConfigError as exc:
        log.error("%s", exc)
        raise SystemExit(1) from None
    if select_provider(settings) == "none":
        log.info(
            "OPENAI_API_KEY is not set. Keyword search is available. "
            "Vector search, hybrid search, embed, and ask return 503."
        )
    import uvicorn

    uvicorn.run("atlas.main:app", host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
