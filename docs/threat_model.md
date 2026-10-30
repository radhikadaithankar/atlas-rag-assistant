# Threat model

Atlas is one local collection with no authentication and no user isolation. Any caller who can reach the API can list, read, search, and delete every document. Do not put the API or the Streamlit page on the public internet with private documents. The UI is another HTTP client, so it cannot be safer than the API.

## What the code enforces

- Secrets come from the environment. `.env` is gitignored. The image does not copy `.env` and the Dockerfile has no secret build arguments. `.env.example` is placeholders only.
- A database URL or API key is not written into logs, `/ready`, or configuration errors. Readiness logs the exception class name.
- Upload size defaults to 10 MiB. The route reads one byte past the limit and returns 413 before parsing. The same limit is checked by `scripts/ingest.py`.
- The file type is taken from the bytes, not the filename or the client MIME type. Names are stored as the base name. The upload is not written to a path from the client.
- SQL is parameterised through SQLAlchemy. The keyword query is a bound argument to `plainto_tsquery`.
- Passage text is fenced in the user message. The system string is constant. A citation is kept only when its chunk id was in that retrieval and the quote is a whitespace-collapsed substring of that passage. If none remain, the API abstains.
- Provider calls time out at 30 seconds. The OpenAI client does not retry. Gemini retries only a short 429, at most 8 times, and does not retry a delay over 60 seconds or a delay written in hours.
- The API field `estimated_cost` is always null.

## What this does not prove

The fence and the citation check stop a passage from rewriting the system prompt or citing a chunk the retriever did not return. They are not authentication. They do not stop a caller who can reach the API from reading the whole collection.

A quote can be a substring of the passage and still not support the claim. If a citation does survive, the answer string is returned even when the model copied instruction text into `answer`.

The faithfulness number in `docs/evaluation_report.md` is not ground truth. The judge can agree with a fluent unsupported answer.

Duplicate detection looks up the SHA-256 and also has a unique constraint. If the winning insert is not visible yet, the loser can get 422 instead of 409.

There is no approximate-index leak, because vector search is an exact cosine scan. That scan gets slower as the corpus grows. It is not a security boundary.

There is no rate limit beyond the upload size. One process serves the local collection. Documents are synthetic samples and the evaluation corpus, not private records.

The rerank extra downloads `cross-encoder/ms-marco-MiniLM-L6-v2` only when a request sets `rerank` and no test double is injected. The published API image does not install that extra.
