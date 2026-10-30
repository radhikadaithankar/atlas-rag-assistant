"""Process settings.

Secrets stay in the environment. ``repr`` reports whether a secret is set and
does not include the value. A missing key is allowed at startup so keyword
search can boot. Callers that need a provider use :func:`require_provider`.
"""

from __future__ import annotations

from urllib.parse import urlparse

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

# Generation uses gemini-3.5-flash-lite when GEMINI_API_KEY is set and
# OPENAI_API_KEY is unset. gemini-2.5-flash and gemini-2.5-flash-lite return
# HTTP 404 for new users. gemini-3.6-flash accepts thinking_budget 0, but this
# project's free-tier generate_content quota was 20 requests per day.
# gemini-3.8-flash hit HTTP 504 inside the 30 second client timeout.
# gemini-3.5-flash is deprecated and its two pricing pages did not agree.
# gemini-3.5-flash-lite returns JSON. thinking_budget 0 is HTTP 400 on that
# model, so the Gemini client retries the same call once without thinking_config.
# Embeddings use gemini-embedding-001 with output_dimensionality 1536.
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
GEMINI_GENERATION_MODEL = "gemini-3.5-flash-lite"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_GENERATION_MODEL = "gpt-6-luna"
EMBEDDING_WIDTH = 1536

# Models whose published width is not the frozen column.
KNOWN_WIDTHS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-ada-002": 1536,
    "text-embedding-3-large": 3072,
    GEMINI_EMBEDDING_MODEL: 1536,
}

MISSING_PROVIDER_MESSAGE = "OPENAI_API_KEY is not set. It is optional for /health and required for OpenAI calls."
MISSING_DATABASE_MESSAGE = (
    "DATABASE_URL is not set. It is required to start the API. "
    "OPENAI_API_KEY is optional; keyword search boots without it."
)


class AtlasConfigError(Exception):
    """Configuration failed. The message names the variable and the rule."""


class Settings(BaseSettings):
    """Environment-backed settings. Secret fields are omitted from ``repr``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        hide_input_in_errors=True,
    )

    openai_api_key: str | None = Field(default=None, repr=False)
    gemini_api_key: str | None = Field(default=None, repr=False)
    database_url: str | None = Field(default=None, repr=False)
    embedding_dimensions: int = EMBEDDING_WIDTH
    embedding_model: str | None = None
    generation_model: str | None = None
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_upload_bytes: int = 10_485_760
    rrf_k: int = 60
    rerank_pool: int = 20
    reranker: str | None = None

    def __repr__(self) -> str:
        return (
            "Settings("
            f"openai_api_key={_flag(self.openai_api_key)}, "
            f"gemini_api_key={_flag(self.gemini_api_key)}, "
            f"database_url={_flag(self.database_url)})"
        )


def _flag(value: str | None) -> str:
    return "set" if value else "unset"


def load_settings() -> Settings:
    """Read settings. Validation errors do not include the raw value."""

    try:
        settings = Settings()
    except ValidationError as exc:
        raise AtlasConfigError(_validation_message(exc)) from None
    try:
        _normalize_secrets(settings)
        _validate(settings)
    except AtlasConfigError:
        raise
    except Exception:
        raise AtlasConfigError("Settings could not be loaded.") from None
    return settings


def _validation_message(exc: ValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return "Settings are invalid."
    loc = errors[0].get("loc", ())
    name = _env_name(str(loc[0])) if loc else "SETTINGS"
    return f"{name} is invalid."


def _env_name(field_name: str) -> str:
    return field_name.upper()


def _normalize_secrets(settings: Settings) -> None:
    settings.openai_api_key = _secret(settings.openai_api_key, "OPENAI_API_KEY")
    settings.gemini_api_key = _secret(settings.gemini_api_key, "GEMINI_API_KEY")
    settings.database_url = _blank_to_none(settings.database_url)
    settings.embedding_model = _blank_to_none(settings.embedding_model)
    settings.generation_model = _blank_to_none(settings.generation_model)
    settings.reranker = _blank_to_none(settings.reranker)


def _blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    if value == "":
        return None
    return value


def _secret(value: str | None, name: str) -> str | None:
    if value is None or value == "":
        return None
    if any(character.isspace() for character in value):
        raise AtlasConfigError(f"{name} must not contain whitespace.")
    return value


def _validate(settings: Settings) -> None:
    if settings.database_url is not None:
        _require_postgres_url(settings.database_url)
    if settings.embedding_dimensions != EMBEDDING_WIDTH:
        raise AtlasConfigError(
            "EMBEDDING_DIMENSIONS is 1536 in the applied migration. "
            "A new migration is required to change it."
        )
    model = settings.embedding_model
    if (
        model is not None
        and KNOWN_WIDTHS.get(model, EMBEDDING_WIDTH) != EMBEDDING_WIDTH
    ):
        raise AtlasConfigError(
            f"EMBEDDING_MODEL {model} returns {KNOWN_WIDTHS[model]} dimensions. "
            "A new migration is required."
        )
    if settings.chunk_size < 1:
        raise AtlasConfigError("CHUNK_SIZE must be at least 1.")
    if settings.chunk_overlap < 0:
        raise AtlasConfigError("CHUNK_OVERLAP must be zero or greater.")
    if settings.chunk_overlap >= settings.chunk_size:
        raise AtlasConfigError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
    if settings.max_upload_bytes < 1:
        raise AtlasConfigError("MAX_UPLOAD_BYTES must be at least 1.")
    if settings.rrf_k < 0:
        raise AtlasConfigError("RRF_K must be zero or greater.")
    if settings.rerank_pool < 1:
        raise AtlasConfigError("RERANK_POOL must be at least 1.")
    if settings.reranker not in {None, "cross_encoder"}:
        raise AtlasConfigError("RERANKER must be empty or cross_encoder.")


def _require_postgres_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"postgresql", "postgres"}:
        raise AtlasConfigError("DATABASE_URL must be a postgresql URL with a hostname.")
    if parsed.hostname is None or parsed.hostname == "":
        raise AtlasConfigError("DATABASE_URL must be a postgresql URL with a hostname.")


def require_database_url(settings: Settings) -> str:
    """Return the database URL or raise. The URL is not logged here."""

    if not settings.database_url:
        raise AtlasConfigError(MISSING_DATABASE_MESSAGE)
    return settings.database_url


def require_openai_api_key(settings: Settings) -> str:
    """Return the OpenAI key. Prefer :func:`select_provider` for routing."""

    if not settings.openai_api_key:
        raise AtlasConfigError(MISSING_PROVIDER_MESSAGE)
    return settings.openai_api_key


def select_provider(settings: Settings) -> str:
    """``openai`` wins when its key is set, else ``gemini``, else ``none``."""

    if settings.openai_api_key:
        return "openai"
    if settings.gemini_api_key:
        return "gemini"
    return "none"


def effective_embedding_model(settings: Settings) -> str:
    if settings.embedding_model:
        return settings.embedding_model
    if select_provider(settings) == "gemini":
        return GEMINI_EMBEDDING_MODEL
    return OPENAI_EMBEDDING_MODEL


def effective_generation_model(settings: Settings) -> str:
    if settings.generation_model:
        return settings.generation_model
    if select_provider(settings) == "gemini":
        return GEMINI_GENERATION_MODEL
    return OPENAI_GENERATION_MODEL


def to_psycopg_url(url: str) -> str:
    """Rewrite a libpq URL to the psycopg v3 SQLAlchemy dialect."""

    if url.startswith("postgresql+psycopg://"):
        return url
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://") :]
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url[len("postgres://") :]
    raise AtlasConfigError("DATABASE_URL must be a postgresql URL with a hostname.")
