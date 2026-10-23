# RetrievalLab adapter protocol

`retrievallab-adapter/1` supports arbitrary dense retrieval and reranking architectures through a separately operated inference service. RetrievalLab sends JSON text and receives vectors or scores. It never imports adapter code or loads adapter checkpoints. No generation or answering endpoint exists.

## Endpoints

The endpoint setting is a base URL, optionally with a path prefix, such as `https://models.example.org/research-v2`. RetrievalLab appends `/metadata`, `/encode`, or `/rerank`. Query strings, fragments, URL credentials, and redirects are rejected. All responses must have `Content-Type: application/json` and no compression. When a token is supplied, every request sends `Authorization: Bearer <token>`.

### GET /metadata

```json
{
  "protocol": "retrievallab-adapter/1",
  "name": "My Research Retriever v2",
  "adapter_type": "dense+reranker",
  "dimension": 768,
  "similarity": "cosine",
  "max_batch_size": 32
}
```

The metadata JSON Schema (draft 2020-12) is:

```json
{
  "type": "object",
  "required": ["protocol", "name", "adapter_type"],
  "properties": {
    "protocol": {"const": "retrievallab-adapter/1"},
    "name": {"type": "string", "minLength": 1, "maxLength": 200},
    "adapter_type": {"enum": ["dense", "reranker", "dense+reranker"]},
    "dimension": {"type": "integer", "minimum": 1, "maximum": 16384},
    "similarity": {"enum": ["cosine", "dot"]},
    "max_batch_size": {"type": "integer", "minimum": 1, "maximum": 1000000}
  },
  "allOf": [{
    "if": {"properties": {"adapter_type": {"enum": ["dense", "dense+reranker"]}}},
    "then": {"required": ["dimension", "similarity"]}
  }]
}
```

Names must also contain non-whitespace text and no control characters. Extra fields are ignored and never exported. `max_batch_size` defaults to 32; RetrievalLab caps every inference batch at 128 regardless of a larger declaration. Use a revision-specific name: changing weights, preprocessing, or vector semantics behind an unchanged identity is not detectable. Reconnect and rebuild indexes after such changes.

### POST /encode

```json
{"role": "document", "texts": ["first passage", "second passage"]}
```

```json
{"vectors": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]}
```

This response illustrates a dimension-3 adapter. `role` is exactly `query` or `document`. Preserve input order and return one vector per text. Query and document inference may use different towers, prompts, or preprocessing, but must share a compatible vector space. A dense-only service need not implement `/rerank`.

Request schema:

```json
{"type":"object","required":["role","texts"],"properties":{"role":{"enum":["query","document"]},"texts":{"type":"array","items":{"type":"string"},"maxItems":128}}}
```

Response schema:

```json
{"type":"object","required":["vectors"],"properties":{"vectors":{"type":"array","items":{"type":"array","items":{"type":"number"}}}}}
```

Beyond the schema, the row count must equal the input count and each row must have exactly the declared dimension. Every value must be a finite JSON number representable as float32. Booleans and numeric strings are rejected. Cosine vectors must be nonzero. RetrievalLab normalizes cosine vectors locally; dot vectors retain their magnitude. Malformed vectors are never padded, truncated, reshaped, or substituted.

### POST /rerank

```json
{"query": "example query", "passages": ["first passage", "second passage"]}
```

```json
{"scores": [1.42, -0.31]}
```

Request schema:

```json
{"type":"object","required":["query","passages"],"properties":{"query":{"type":"string"},"passages":{"type":"array","items":{"type":"string"},"maxItems":128}}}
```

Response schema:

```json
{"type":"object","required":["scores"],"properties":{"scores":{"type":"array","items":{"type":"number"}}}}
```

Return exactly one finite numeric score per input passage, in input order; larger is better. Scores must be comparable across batches and must not depend on which other candidates happen to share a batch. RetrievalLab batches the entire configured candidate set and sorts only after all scores validate. Equal scores retain first-stage order. A reranker cannot provide passage IDs or add candidates; additional response fields do not affect ranking. A reranker-only service need not implement `/encode`.

## Validation and transport limits

- Unsupported protocol versions, roles, capabilities, similarities, or dimensions fail explicitly.
- Invalid UTF-8/JSON, duplicate JSON keys, NaN/Infinity, incorrect counts, nonnumeric values, and float32 overflow in vectors fail explicitly. Errors use locally authored messages; server error bodies and headers are never shown or logged.
- Each text/query is limited to 256 KiB UTF-8, each request to 2 MiB JSON, each response to 16 MiB. Batches shrink to satisfy both the declared batch size and request byte limit. A single oversized text fails rather than being truncated.
- Connections use a 5-second timeout, socket reads a 30-second timeout, and response-body reading a 30-second deadline. No automatic retry occurs. System DNS resolution uses the host resolver's timeout policy.
- Production endpoints require verified HTTPS. All DNS results are checked on every request; the connection is pinned to a validated numeric IP while TLS verifies the original hostname. Private, loopback, link-local, multicast, reserved, unspecified, and IPv4 transition addresses are rejected. Redirects and environment proxy configuration are not used.
- `RETRIEVALLAB_ALLOW_LOCAL_ADAPTERS=1` enables loopback development endpoints, including HTTP. It does not permit arbitrary private networks, cloud metadata, or public HTTP.
- Operator-managed egress restrictions remain useful defense in depth for public deployments. Transport design follows [OWASP SSRF guidance](https://cheatsheetseries.owasp.org/cheatsheets/Server_Side_Request_Forgery_Prevention_Cheat_Sheet.html).

## Internal contracts and index lifecycle

`DenseEncoderAdapter` defines `name`, `dimension`, `similarity`, `encode_queries(list[str])`, and `encode_documents(list[str])`. `RerankerAdapter` defines `name` and `score(query, list[str])`; `input_kind = "text"` marks this text-based contract for compatibility with older candidate-object rerankers. Both inference methods return NumPy arrays. Existing SentenceTransformer and CrossEncoder wrappers implement these contracts without duplicate model inference. Legacy dense `encode_passages`/`encode_query` and candidate-based rerankers remain supported.

RetrievalLab owns parsing, passages, the local exact vector index, nearest-neighbor search, BM25, Hybrid RRF, evaluation, and exports. Remote indexing sends existing passage text with `role=document`; searching sends one query with `role=query`. Newly uploaded corpora using a remote model use RetrievalLab's whitespace chunker. Switching models over an active corpus reuses its exact existing passages and BM25 index. Model changes discard only Dense/Hybrid indexes. Failed indexing never installs a partial index. Reranker changes reuse the dense index and preserve first-stage ranks, scores, and Hybrid provenance.

Identity contains source `remote`, endpoint host, an endpoint SHA-256 identifier, name, protocol, adapter type, dimension, and similarity. It excludes bearer tokens and raw URL paths. Evaluation reports and benchmark JSON/CSV carry this identity in model ID fields. Tokens remain in Streamlit session state and are read by session-owned clients on demand, never in global model caches or disk files. Do not place secrets in model names or URLs. Known token echoes in metadata are rejected.

Benchmarks compose the same Dense/BM25/Hybrid engines with the selected local or remote reranker. They reuse document embeddings and indexes and leave interactive search state unchanged. One benchmark run uses one selected dense adapter and one selected reranker, with any subset of the six pipeline combinations. To compare different model sources, switch the model, rebuild Dense when required, and export separate runs.

## Run the template locally

From the repository root, using a separate virtual environment:

```bash
python -m venv /tmp/retrievallab-adapter-env
/tmp/retrievallab-adapter-env/bin/pip install -r examples/remote_adapter/requirements.txt
cd examples/remote_adapter
/tmp/retrievallab-adapter-env/bin/uvicorn server:app --host 127.0.0.1 --port 8000
```

Start RetrievalLab in another terminal:

```bash
RETRIEVALLAB_ALLOW_LOCAL_ADAPTERS=1 streamlit run app.py
```

Inside **Model settings**, choose **Remote Adapter**, enter `http://127.0.0.1:8000`, leave the token empty for the local template, and select **Test connection**. Do this independently for Dense and/or reranking. Metadata appears after validation. Index a corpus (or rebuild Dense + Hybrid over the active corpus), then search or evaluate. Choosing a remote reranker also enables it in the search controls; choose Off there for a base-only run.

The deterministic default downloads nothing. Replace `encode_queries`, `encode_documents`, and/or `rerank`, plus model loading, in `examples/remote_adapter/server.py`. Declare the capabilities you actually implement and keep dimension/similarity accurate. For reranker-only use, set `adapter_type` to `reranker` and omit dimension/similarity.

## Deploy separately

Run the adapter on hardware you control, including your GPU server or a compatible cloud/Hugging Face hosting service that exposes these HTTP routes. Use a public HTTPS endpoint with a valid certificate, configure authentication at the service or reverse proxy, and forward the bearer header. Set concurrency, resource limits, and inference batch sizes for your model. The template has no built-in authentication and is a loopback demonstration; add authentication before exposing it publicly. Hosted services must return protocol JSON directly without interactive login or redirects.

Model-specific dependencies and checkpoints stay exclusively in your own deployment. RetrievalLab does not run a container, shell command, uploaded Python, pickle/joblib, or checkpoint on your behalf. Hugging Face model-ID support continues to use `trust_remote_code=False`. Passage and query text leave RetrievalLab when you opt into remote inference, so use a service whose data handling you accept.
