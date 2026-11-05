# Grounded Knowledge Assistant with Hybrid Retrieval

A FastAPI document-question-answering starter that combines lexical retrieval, TF-IDF vector-space retrieval, reciprocal-rank fusion, source-linked evidence, and evidence-based abstention.

> **Implementation status:** This repository contains a runnable local MVP and an evaluation baseline. The current implementation uses scikit-learn TF-IDF rather than a hosted embedding model; PostgreSQL/pgvector is provided as an extension schema. Cross-encoder reranking, LangChain orchestration, and LLM-generated answers are documented as next steps, not claimed as implemented.

## Problem

Document assistants can return plausible answers that are not supported by the source material. This project focuses on making retrieval inspectable and making insufficient evidence an explicit outcome.

## Current capabilities

- Ingests documents and splits them into overlapping text chunks with stable IDs.
- Retrieves candidate chunks using lexical term-frequency scores and TF-IDF cosine similarity.
- Combines the two ranked lists with Reciprocal Rank Fusion (RRF).
- Returns chunk IDs, document titles, evidence text, and normalized retrieval scores.
- Abstains when the evidence score is below a configurable threshold.
- Exposes health, ingestion, search, and answer endpoints through FastAPI.
- Includes a small labeled JSONL benchmark for Recall@5, MRR@5, top-hit evidence coverage proxy, latency, and estimated local inference cost.
- Includes a PostgreSQL/pgvector schema for a future persistent dense-vector backend.

## Architecture

```text
Documents
   |
   v
Chunking + source metadata ------> In-memory chunk index
                                      |             |
Question --------------------------> lexical       TF-IDF
                                      |             |
                                      +------ RRF --+
                                             |
                                      evidence threshold
                                        /          \
                                  cited excerpt   abstain
```

## Tech stack

- Python 3.11+
- FastAPI + Pydantic
- scikit-learn TF-IDF and cosine similarity
- PostgreSQL + pgvector schema (optional persistence extension)
- Docker
- Pytest + GitHub Actions

## Run locally

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open the interactive API docs at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### Try the API

Ingest sample documents:

```bash
curl -X POST http://127.0.0.1:8000/documents \
  -H "Content-Type: application/json" \
  -d '{"documents":[{"id":"policy-1","title":"Refund policy","text":"Customers may request a refund within 30 days of purchase."}]}'
```

Ask a question:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"How long do customers have to request a refund?","top_k":5}'
```

The current MVP returns the strongest supporting sentence with a source marker. It intentionally does not call an external LLM, so the demo works without API credentials and does not fabricate an answer beyond retrieved evidence.

## Run tests and benchmark

```bash
pytest -q
python scripts/run_benchmark.py --dataset data/sample_queries.jsonl --output reports/benchmark.json
```

The included sample benchmark has only three synthetic queries. A local run is useful as a smoke test, not as a statistically meaningful performance claim. Replace it with a manually labeled dataset before publishing comparative retrieval results.

## pgvector schema

The optional schema is in `sql/schema.sql`. It creates a chunk table with a vector column and HNSW cosine index. The current app does not yet connect to PostgreSQL; a production integration should add a repository layer, a versioned embedding model, tenant-aware filtering, and migration/configuration management.

## Roadmap

1. Add LangChain document loaders and retriever abstractions.
2. Add a versioned sentence-embedding model and pgvector-backed persistence.
3. Add cross-encoder reranking and compare vector-only vs hybrid vs hybrid-plus-reranking.
4. Add LLM answer synthesis with strict source-grounding prompts and a human-reviewed groundedness rubric.
5. Expand evaluation and record Recall@5, MRR, groundedness, p50/p95 latency, and cost/query across a larger labeled test set.
6. Add authentication, authorization, rate limiting, tracing, and tenant isolation before exposing the API to real users.

## Limitations

- In-memory storage resets when the process restarts.
- TF-IDF is a lightweight vector-space baseline, not a modern semantic embedding model.
- The evidence-coverage metric is a retrieval proxy, not a human groundedness score.
- No customer data or external production systems are connected.
