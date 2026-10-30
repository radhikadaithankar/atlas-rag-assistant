# Atlas evaluation report

Scores copied from the measured runs recorded in the Gemini evaluation and faithfulness milestone reports. This rebuild did not call a provider, did not download reranker weights, and did not recompute these scores.

53 questions, 39 unique gold quotes, 45 questions with gold. Embedding `gemini-embedding-001`, generation `gemini-3.5-flash-lite`, output dimensionality 1536.

These figures are the measured run recorded in the project milestone reports. This file does not add scores from a later run.

| Metric | Status | Value | Detail |
| --- | --- | --- | --- |
| `chunk_containment_1000_200` | measured | 1.0 | 39 of 39 unique gold quotes. This is not retrieval quality. |
| `chunk_containment_400_80` | measured | 0.9743589743589743 | 38 of 39 unique gold quotes. The quote that does not fit is the long clinic-linen gown paragraph. This is not retrieval quality. |
| `keyword_recall_at_5` | measured | 1.0 | top_k 5, chunk size 1000, overlap 200, 45 questions that have gold. Tied ranks follow chunk id, so a new ingest can move a tied hit across the cut. |
| `keyword_mrr` | measured | 1.0 | Same keyword run as keyword_recall_at_5. |
| `deterministic_abstention` | measured | 1.0 | All 8 unanswerable questions returned no keyword hits. Each one called answer_from_passages with an empty passage list. The generator was not called. This is not model abstention quality. |
| `vector_recall_at_5` | measured | 0.9888888888888889 | Cosine distance, gemini-embedding-001, output dimensionality 1536, top_k 5. |
| `vector_mrr` | measured | 0.9333333333333333 | Same vector run as vector_recall_at_5. |
| `hybrid_recall_at_5` | measured | 1.0 | Reciprocal rank fusion, top_k 5, RRF_K 60. |
| `hybrid_mrr` | measured | 0.962962962962963 | Same hybrid run as hybrid_recall_at_5. |
| `rerank_recall_at_5` | measured | 0.9888888888888889 | Hybrid pool of 20, then cross-encoder/ms-marco-MiniLM-L6-v2 cut to 5. |
| `rerank_mrr` | measured | 0.9703703703703703 | Same rerank run as rerank_recall_at_5. |
| `citation_correctness` | measured | 0.9111111111111111 | Mean precision over 45 questions with gold. 4 of those abstained and scored 0. 8 of 8 unanswerable questions abstained. 8 of those reached the generator and 0 had empty hybrid retrieval. |
| `deterministic_citation_form` | measured | 1.0 | 53 of 53. A kept citation's chunk id was retrieved and its quote occurs in that passage, or the answer abstained with no citations. |
| `latency_median_ms` | measured | 940.5741750015295 | Median of 53 generator calls, linear interpolation on the sorted latencies. |
| `latency_p95_ms` | measured | 10390.92161160077 | P95 of 53 generator calls, linear interpolation on the sorted latencies. |
| `prompt_tokens` | measured | 46827 | Sum of prompt tokens reported by the provider. |
| `completion_tokens` | measured | 5345 | candidates_token_count plus thoughts_token_count only when that field is an int. The thoughts field was missing and was not added. |
| `estimated_cost` | measured | 0.0274106 | Input $0.30 per 1M tokens and output, including thinking tokens, $2.50 per 1M tokens, for gemini-3.5-flash-lite paid standard tier. Source: Gemini Developer API pricing, page last updated 2026-10-22 UTC, checked 23 Oct 2026, https://ai.google.dev/gemini-api/docs/pricing. The free tier on that page is free of charge. 46827 / 1e6 * 0.30 + 5345 / 1e6 * 2.50 = 0.0274106. This is a published paid-tier price multiplied by the measured counts, not an invoice. The API response field estimated_cost stays null. |
| `answer_faithfulness` | measured | 1.0 | 53 of 53 stored answers were judged supported by gemini-3.5-flash-lite. This score is not ground truth. The judge can agree with a fluent unsupported answer. |

## Notes copied with the scores

- Faithfulness is not ground truth. The judge can agree with a fluent unsupported answer.
- Deterministic abstention is a separate score from model abstention. It uses empty keyword retrieval and does not call the generator.
- Chunk containment checks whether a gold quote fits inside a window. It is not retrieval quality.
- Keyword ties follow chunk id, so a new ingest can move a tied hit across the top-5 cut.
- The API field estimated_cost stays null. The cost above is the published price times the measured token counts.

53 of 53 stored answers were judged supported by gemini-3.5-flash-lite. This score is not ground truth. The judge can agree with a fluent unsupported answer.

Faithfulness is not ground truth. The judge can agree with a fluent unsupported answer.
