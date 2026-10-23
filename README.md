# RetrievalLab

**A modular workbench for comparing dense, lexical, and hybrid retrieval pipelines.**

RetrievalLab is a local Streamlit application for indexing your own documents, retrieving ranked passages, and inspecting why each result was returned. It is an information-retrieval workbench—not a full RAG system—and does not generate answers.

## Public pipelines

- **Dense** — semantic retrieval with normalized `Snowflake/snowflake-arctic-embed-s` embeddings and an exact inner-product index.
- **BM25** — Okapi BM25 lexical retrieval with raw score inspection.
- **Hybrid** — Dense and BM25 rankings combined with Reciprocal Rank Fusion (RRF), including each branch's rank and raw score.

All three pipelines use the same session-local passage corpus. Dense embeddings and the exact vector index are built once and reused across queries.

## Workflow

1. Add PDF, DOCX, TXT, or BEIR-style corpus JSONL files.
2. Index the replacement corpus.
3. Select Dense, BM25, or Hybrid.
4. Enter a natural-language query.
5. Inspect ranked passages, source metadata, scores, latency, and pipeline details.

PDF extraction remains page-aware, and an empty or image-only page is reported honestly; OCR is not included. DOCX, TXT, and JSONL passages do not claim page numbers. Corpus chunks use the Arctic-S tokenizer with 350-token windows and 50-token overlap, and PDF chunks never cross page boundaries.

## Local privacy and limits

Uploads, extracted passages, embeddings, and indexes remain in the active Streamlit session. RetrievalLab does not add persistence, analytics, tracking, remote document transmission, or answer generation.

The local workbench enforces:

- 20 uploaded files
- 50 MB total upload size
- 35,000 indexed passages
- 100 retrieval candidates, bounded by corpus size
- 10 displayed results

## Install and run

```bash
git clone https://github.com/rdaslam12/RetrievalLab.git
cd RetrievalLab
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

The first real-model run may download the configured embedding model into your local model cache. On supported platforms RetrievalLab uses FAISS `IndexFlatIP`; on macOS it intentionally uses an exact NumPy fallback to avoid a known native runtime conflict.

## Project structure

```text
.
├── app.py                       # Streamlit application
├── assets/                      # Local styles and font assets
├── config/retrievallab.json     # Public retrieval runtime profile
├── sample_data/                 # Small synthetic local examples
├── scripts/smoke_test.py        # Offline synthetic smoke test
├── src/
│   ├── bm25_retriever.py
│   ├── chunker.py
│   ├── config.py
│   ├── document_parser.py
│   ├── embedder.py
│   ├── hybrid_retriever.py
│   ├── interfaces.py
│   ├── search_engine.py
│   ├── services.py
│   ├── types.py
│   ├── ui_components.py
│   └── vector_index.py
└── tests/
```

The parsing, chunking, indexing, retrieval, orchestration, and presentation layers remain separate so additional public retrieval strategies can be added without rewriting the application.

## Testing

The ordinary suite is fast and offline. It uses deterministic fake embeddings while covering parsing, page-safe chunking, dense retrieval, vector indexing, BM25, Hybrid RRF, orchestration, and JSONL retrieval.

```bash
python -m compileall -q app.py src scripts tests
python -m pytest -q
python scripts/smoke_test.py
```

The real embedding-model smoke test is opt-in because it may require a model download:

```bash
RUN_MODEL_SMOKE=1 python -m pytest -q tests/test_real_model_smoke.py
```

## Current scope

RetrievalLab currently retrieves and inspects passages. It does not evaluate relevance, download benchmark datasets, train models, generate responses, or persist corpora between sessions.

RetrievalLab originated from deployment work around an ongoing retrieval and reranking research project. Unpublished research implementations, trained checkpoints, experimental configurations, and private research artifacts are intentionally not included in this public repository while the associated work is under review.

## Roadmap

- Additional public retrieval backends
- Generic, documented extension points for public post-retrieval components
- Fresh screenshots of the public-only interface
- Exportable retrieval diagnostics
- Expanded accessibility and cross-platform checks

Contributions should preserve local-session privacy, deterministic ranking behavior, transparent result provenance, and fast offline tests.
