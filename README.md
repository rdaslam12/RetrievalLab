# RetrievalLab

RetrievalLab is a modular workbench for running, evaluating, and comparing information-retrieval pipelines on user-provided corpora. It supports lexical, dense, hybrid, and reranked retrieval; compatible Hugging Face models can be swapped directly, while non-standard research architectures can integrate through a versioned remote adapter interface. RetrievalLab focuses on retrieval inference, evaluation, and comparison—not LLM answer generation.

## Key capabilities

- Dense retrieval with a local exact vector index
- Okapi BM25 lexical retrieval
- Hybrid Dense + BM25 retrieval using Reciprocal Rank Fusion (RRF)
- Optional CrossEncoder reranking over a fixed first-stage candidate set
- Compatible custom Hugging Face SentenceTransformer and CrossEncoder model IDs
- Remote adapters for custom dense encoders and rerankers
- PDF, DOCX, TXT, and supported corpus JSONL ingestion
- Passage chunking and session-local indexing
- Result provenance, component scores, rank changes, and retrieval/reranking/total latency diagnostics
- Document-level Recall@K, Hit Rate@K, MRR@K, and NDCG@K
- Multi-pipeline benchmarking with aggregate CSV and detailed JSON export

## How RetrievalLab works

1. Upload a corpus and build a shared passage collection.
2. Select a built-in, Hugging Face-compatible, or remote dense model.
3. Run Dense, BM25, or Hybrid retrieval against the same active corpus.
4. Optionally rerank the retrieved candidates with a local or remote reranker.
5. Inspect ranked passages and provenance, or evaluate them with query JSONL and qrels TSV files.
6. Benchmark selected configurations under the same corpus, queries, judgments, and candidate depth.

Corpus embeddings and indexes are reused across queries. Changing the dense model rebuilds only Dense and Hybrid indexes; changing the reranker reuses the existing first-stage indexes.

## Supported retrieval configurations

| Configuration | First stage | Optional second stage |
| --- | --- | --- |
| Dense | Selected dense encoder + exact inner-product search | Selected reranker |
| BM25 | Okapi BM25 | Selected reranker |
| Hybrid | Dense and BM25 rankings fused with RRF | Selected reranker |

The benchmark interface can compare Dense, BM25, Hybrid, Dense + reranker, BM25 + reranker, and Hybrid + reranker. These are controlled comparisons, not a claim that any one pipeline is universally best.

## Bring Your Own Retriever or Reranker

RetrievalLab supports three practical integration levels.

### Built-in

Use the default `Snowflake/snowflake-arctic-embed-s` dense encoder and `cross-encoder/ms-marco-MiniLM-L6-v2` CrossEncoder.

### Hugging Face-compatible

Enter a Hugging Face Hub model ID for a compatible SentenceTransformer embedding model or `sentence_transformers.CrossEncoder` model. RetrievalLab validates the ID, infers the embedding dimension, and loads the model with `trust_remote_code=False`. Compatibility with the relevant Sentence Transformers interface is required.

### Custom research architecture

Expose the model through the versioned `retrievallab-adapter/1` HTTP protocol. Researchers with non-standard architectures can implement a small adapter around their model rather than modifying RetrievalLab itself.

RetrievalLab remains responsible for corpus handling, chunking, local indexing, retrieval and reranking orchestration, evaluation, benchmarking, diagnostics, and export. The external service performs only compatible model inference: query encoding, document encoding, and/or passage reranking scores.

See the [remote adapter protocol and deployment guide](docs/adapter_protocol.md) and the [standalone adapter example](examples/remote_adapter/).

## Evaluation and benchmarking

Evaluation accepts:

- Query JSONL: one object per line with `_id` and `text`
- Qrels TSV: `query-id`, `corpus-id`, and a non-negative relevance score; the header is optional

Results are evaluated at the document level. Multiple passages from the same document are deduplicated before metric cutoffs. RetrievalLab reports Recall, Hit Rate, MRR, and graded NDCG at K = 1, 3, 5, and 10, together with mean retrieval, reranking, and total query latency where applicable.

Benchmarking runs any selected subset of the six supported configurations against the same active corpus, query set, qrels, metric implementation, and candidate depth. Aggregate results can be downloaded as CSV; JSON export also includes per-query metrics, document IDs, latency, model identity, warnings, and failed-configuration details. Evaluation cannot measure relevant documents outside the configured retrieval candidate depth.

## Architecture

```text
Corpus
  ↓
Parsing + Chunking
  ↓
Shared Passages
  ├── Dense Encoder Adapter ── Built-in / Hugging Face / Remote
  │        ↓
  │    Exact Vector Index
  └── BM25 Index
           ↓
Dense / BM25 / Hybrid (RRF) Retrieval
  ↓
Optional Reranker Adapter ── Built-in / Hugging Face / Remote
  ↓
Ranked Passages
  ↓
Evaluation + Benchmarking
  ↓
Metrics / Diagnostics / CSV + JSON Export
```

Parsing, chunking, model adapters, indexes, retrieval engines, reranking, evaluation, benchmarking, and presentation are separate modules. Dense embeddings use exact inner-product search through FAISS `IndexFlatIP` on supported platforms and an exact NumPy fallback on macOS.

## Corpus and input support

- PDF text extraction is page-aware; image-only pages are reported, and OCR is not included.
- DOCX ingestion extracts non-empty paragraph text.
- TXT ingestion accepts text files and reports replacement of invalid UTF-8 bytes.
- Corpus JSONL uses BEIR-style records with required `_id` and `text` fields and an optional `title`.
- The default profile uses 350-token chunks with 50-token overlap, a 100-passage candidate depth, and 10 displayed search results.

The local workbench accepts up to 20 files, 50 MB total upload size, and 35,000 indexed passages. A bounded slice mode is available for a single larger JSONL corpus; results then cover only the active slice.

## Installation and local usage

```bash
git clone https://github.com/rdaslam12/RetrievalLab.git
cd RetrievalLab
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the Streamlit URL, upload a corpus, choose a retrieval pipeline and model settings, then select **Index corpus**. The first use of a selected Hugging Face model may download it into the repository-local model cache. CUDA is preferred when available, followed by Apple MPS and CPU; model execution may fall back from MPS to CPU if needed.

## Remote adapter usage

Run the example adapter in a separate environment:

```bash
python -m venv /tmp/retrievallab-adapter-env
/tmp/retrievallab-adapter-env/bin/pip install -r examples/remote_adapter/requirements.txt
cd examples/remote_adapter
/tmp/retrievallab-adapter-env/bin/uvicorn server:app --host 127.0.0.1 --port 8000
```

Start RetrievalLab from the repository root with local-adapter development enabled:

```bash
RETRIEVALLAB_ALLOW_LOCAL_ADAPTERS=1 streamlit run app.py
```

In **Model settings**, choose **Remote Adapter**, enter `http://127.0.0.1:8000`, and test the connection. The included server is a deterministic development template, not a production model service. For protocol routes, schemas, validation rules, authentication guidance, deployment requirements, and transport limits, read [docs/adapter_protocol.md](docs/adapter_protocol.md).

## Testing

The default suite is offline and uses deterministic test doubles for model inference:

```bash
python -m compileall -q app.py src scripts tests examples
python -m pytest -q
python scripts/smoke_test.py
```

Real-model smoke tests are opt-in because they may download models:

```bash
RUN_MODEL_SMOKE=1 python -m pytest -q tests/test_real_model_smoke.py
RUN_RERANKER_SMOKE=1 python -m pytest -q tests/test_real_reranker_smoke.py
```

## Security and privacy boundaries

- RetrievalLab does not execute arbitrary uploaded Python code or load arbitrary user-uploaded checkpoints.
- Hugging Face models are selected by validated Hub ID; `trust_remote_code=True` is not used.
- Custom architectures integrate through remote adapters rather than executable uploads.
- Remote endpoints, metadata, response shapes, numeric values, sizes, redirects, and network targets are validated. These checks reduce risk but do not make an untrusted service safe.
- Adapter tokens remain in Streamlit session state, are not placed in model caches, and are excluded from benchmark exports and adapter identity metadata.
- Uploaded corpus data, passages, embeddings, indexes, evaluation inputs, and results are held in the active application session; RetrievalLab does not add corpus or index persistence.
- When a remote adapter is enabled, the document and query text required for its inference is sent to that endpoint. Use only an endpoint whose operator and data-handling policy you trust.

## Current limitations

- Remote adapters require a separately hosted endpoint implementing the compatible protocol.
- Arbitrary checkpoint, model-file, and Python-file uploads are intentionally unsupported.
- The remote protocol covers the implemented dense-encoding and reranking contracts, not every information-retrieval paradigm.
- Benchmark latency depends on hardware, model state, corpus size, and runtime conditions; remote-adapter latency also includes network overhead.
- Hugging Face model availability and downloads depend on the selected model and network/cache state.
- Evaluation is bounded by the fixed candidate depth, currently 100 or the corpus size when smaller.
- OCR, model training, benchmark-dataset downloads, durable corpus persistence, and answer generation are outside the current scope.

## Research origin and confidentiality

RetrievalLab originated from deployment and evaluation work around an ongoing retrieval and reranking research project. Unpublished research implementations, trained checkpoints, experimental configurations, and private research artifacts are intentionally excluded from this public repository while the associated work is under review.

## Project status

RetrievalLab is feature-complete for its current scope: retrieval inference, optional reranking, document-level evaluation, controlled multi-pipeline benchmarking, diagnostics, and export. It intentionally stops at ranked evidence and does not add an LLM answer-generation or chatbot layer.
