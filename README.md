# Atlas

Atlas is a local retrieval-augmented assistant. It ingests PDF, text, and Markdown, stores chunks in Postgres with pgvector, and answers from retrieved passages with citations. It is a single-user local service. There is no authentication and no user isolation. Do not expose it with private documents.

## Stack

Python 3.12, FastAPI, Pydantic v2, SQLAlchemy 2, Alembic, PostgreSQL with pgvector, PyMuPDF 1.x, and Streamlit. Embeddings and generation use OpenAI when `OPENAI_API_KEY` is set. If only `GEMINI_API_KEY` is set, they use Gemini (`gemini-embedding-001` at output dimensionality 1536, and `gemini-3.5-flash-lite`). If neither key is set, the process still boots and keyword search works. Vector search, hybrid search, embed, and ask return 503.

The optional cross-encoder is the `rerank` dependency group (`sentence-transformers`). The API image does not install it. `uv sync` without that group does not install `sentence-transformers` or torch.

## Run the API on the host

Install [uv](https://docs.astral.sh/uv/) 0.13.5 and Python 3.12.

```bash
cp .env.example .env
uv sync --python 3.12
docker compose up -d postgres
uv run alembic upgrade head
uv run python -m atlas.serve
```

`GET /health` is liveness and does not touch Postgres. `GET /ready` runs `SELECT 1`.

In another shell:

```bash
uv run streamlit run src/atlas/ui/app.py --server.address 127.0.0.1 --server.port 8501
```

The UI default mode is keyword. Rerank is off. Keyword search does not need a key. Ask needs a provider key on the API, including keyword mode, because it generates an answer.

Ingest one file from the shell:

```bash
uv run python scripts/ingest.py data/sample_documents/note.txt
```

With no provider key the file is stored and embeddings stay null. A second ingest of the same bytes prints `created=false`.

## Run the stack with Compose

```bash
docker compose up --build
```

Compose starts Postgres, runs `alembic upgrade head`, then the API on port 8000 and Streamlit on port 8501. The UI calls `http://api:8000`. `OPENAI_API_KEY` is not set in the compose file. If a `.env` file exists it is passed (`required: false`) and `DATABASE_URL` is overridden to the `postgres` hostname. The image is built with `uv sync --frozen --no-dev --no-group rerank`. It does not copy `.env`.

The API process exits 1 when `DATABASE_URL` is missing or invalid. The log line names the variable and does not print the value.

## API

| Method | Path | Role |
| --- | --- | --- |
| GET | `/health` | Liveness. Does not open Postgres. |
| GET | `/ready` | `SELECT 1`. 503 when the database is unset or down. |
| POST | `/documents` | Upload. 201, 409 duplicate bytes, 413 too large, 422 empty or unreadable. |
| GET | `/documents` | List with `offset` and `limit`. |
| GET | `/documents/{id}` | One document, or 404. |
| DELETE | `/documents/{id}` | 204. Chunks are removed by `ON DELETE CASCADE`. |
| POST | `/documents/{id}/embed` | Fill null embeddings. `replace=true` recomputes them. |
| POST | `/search` | `mode` is `vector`, `keyword`, or `hybrid` (default `hybrid`). |
| POST | `/ask` | Grounded answer. The body does not include the system prompt. |

`top_k` is 1 to 50. `rerank` defaults to false. When it is true the retriever uses `RERANK_POOL` (default 20) and then cuts back to `top_k`. `top_k` larger than the pool is 422.

File type comes from the bytes. A leading `%PDF-` is a PDF. Other files must be UTF-8. UTF-8 with an ATX heading is Markdown. The filename and the client MIME type are not used. Upload names are stored as the base name only.

## Retrieval and answers

Vector search is an exact pgvector cosine scan and skips null embeddings. Keyword search is `plainto_tsquery('english', ...)` and `ts_rank_cd` on a stored generated `tsvector`. Hybrid runs both at the same top-k and fuses with reciprocal rank fusion: the sum of `1 / (k + rank)` with `k` defaulting to 60 and ranks starting at 1. A hit in only one list still scores. The other rank stays null. Equal scores break by chunk id.

Ask keeps a citation only when the chunk id was in that retrieval and the quote, after whitespace is collapsed, is a substring of that passage. The quote is not rewritten. If none remain, the answer is a fixed insufficient-evidence sentence. Passage text is untrusted and stays inside a fence. It is not copied into the system prompt.

`estimated_cost` on the API is always null. Token counts are copied when the provider sends them.

Gemini generation retries `thinking_budget` 0 once without `thinking_config`, because that model returns HTTP 400 for the budget. Short HTTP 429 responses retry up to 8 times. A delay written in hours, or any parsed delay over 60 seconds, is not retried.

## Evaluation

`data/evaluation/v1/` holds 53 synthetic questions. The numbers in `docs/evaluation_report.md` are copied from a measured run recorded before this rebuild. This rebuild did not call a provider and did not recompute them. Faithfulness is not ground truth: the judge can agree with a fluent unsupported answer.

| Metric | Value |
| --- | --- |
| Chunk containment 1000/200 | 1.0 (39 of 39). Not retrieval quality. |
| Chunk containment 400/80 | 0.9743589743589743 (38 of 39). Not retrieval quality. |
| Keyword Recall@5 | 1.0 |
| Keyword MRR | 1.0 |
| Deterministic abstention | 1.0 |
| Vector Recall@5 | 0.9888888888888889 |
| Vector MRR | 0.9333333333333333 |
| Hybrid Recall@5 | 1.0 |
| Hybrid MRR | 0.962962962962963 |
| Rerank Recall@5 | 0.9888888888888889 |
| Rerank MRR | 0.9703703703703703 |
| Citation correctness | 0.9111111111111111 |
| Deterministic citation form | 1.0 (53 of 53) |
| Latency median / P95 | 940.5741750015295 ms / 10390.92161160077 ms |
| Prompt / completion tokens | 46827 / 5345 |
| Estimated cost | 0.0274106 |
| Answer faithfulness | 1.0, not ground truth |

Keyword ties follow chunk id, so a new ingest can move a tied hit across the top-5 cut. The cost is the published Gemini paid-tier price times those token counts, not an invoice. See `docs/evaluation_report.md`.

Running `uv run python scripts/evaluate.py` needs Postgres and replaces `results.json` with a new measurement. `--vector`, `--hybrid`, `--rerank`, and `--answers` exit 2 when the provider or the rerank extra is missing, and they do not write a score.

## Tests

```bash
uv run ruff check
uv run ruff format --check
uv run mypy src/atlas
uv run pytest
```

Integration tests and the collection security tests skip when `DATABASE_URL` is unset. With the compose database they run against Postgres and do not call a paid API. Unit tests use fake embedders and generators.

## Security

There is no authentication. One local collection is visible to every caller who can reach the API. Secrets come from the environment. `.env` is gitignored. `.env.example` has placeholders only. SQL is parameterised. Logs record exception class names, latency, and token counts, not document text, prompts, or credentials. Upload size defaults to 10 MiB. See `docs/threat_model.md` and `docs/architecture.md`.

## Layout

`src/atlas/` holds config, the API, Postgres, ingestion, embeddings, retrieval, generation, evaluation, and the Streamlit client. Migrations live in `migrations/`. The embedding width is 1536 and is frozen in the first revision.
