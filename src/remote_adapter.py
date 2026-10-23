"""retrievallab-adapter/1: validated, architecture-independent inference clients."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Protocol

import numpy as np

from .remote_transport import (
    AdapterError, Endpoint, HTTPTransport, MAX_REQUEST_BYTES, MAX_RESPONSE_BYTES, validate_endpoint,
)

PROTOCOL_VERSION = 'retrievallab-adapter/1'
MAX_BATCH_SIZE = 128
MAX_DIMENSION = 16384
MAX_TEXT_BYTES = 256 * 1024


class Transport(Protocol):
    def request(self, endpoint: Endpoint, method: str, route: str,
                body: bytes | None, token: str) -> bytes: ...


@dataclass(frozen=True)
class AdapterMetadata:
    protocol: str
    name: str
    adapter_type: str
    dimension: int | None = None
    similarity: str | None = None
    max_batch_size: int = 32


def validate_metadata(data: object) -> AdapterMetadata:
    if not isinstance(data, dict):
        raise AdapterError('Metadata must be a JSON object.')
    if data.get('protocol') != PROTOCOL_VERSION:
        raise AdapterError('Metadata protocol must be retrievallab-adapter/1.')
    kind = data.get('adapter_type')
    if kind not in ('dense', 'reranker', 'dense+reranker'):
        raise AdapterError('Metadata adapter_type must be dense, reranker, or dense+reranker.')
    name = data.get('name')
    if not isinstance(name, str) or not name.strip() or len(name) > 200 or any(ord(c) < 32 for c in name):
        raise AdapterError('Metadata name must be a non-empty string of at most 200 characters.')
    dimension, similarity = data.get('dimension'), data.get('similarity')
    if kind in ('dense', 'dense+reranker'):
        if type(dimension) is not int or not 1 <= dimension <= MAX_DIMENSION:
            raise AdapterError('Metadata dimension must be an integer between 1 and 16384.')
        if similarity not in ('cosine', 'dot'):
            raise AdapterError('Metadata similarity must be cosine or dot.')
    else:
        dimension = similarity = None
    batch = data.get('max_batch_size', 32)
    if type(batch) is not int or batch <= 0 or batch > 1_000_000:
        raise AdapterError('Metadata max_batch_size must be a positive integer no greater than 1000000.')
    return AdapterMetadata(PROTOCOL_VERSION, name, kind, dimension, similarity, batch)


def _reject_constant(_value: str) -> None:
    raise AdapterError('Adapter JSON contains NaN or Infinity; all values must be finite.')


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise AdapterError('Adapter JSON contains duplicate object keys.')
        result[key] = value
    return result


class RemoteAdapterClient:
    """Session-owned client. Read bearer tokens on demand; never store a token.

    The UI supplies a session-state lookup callable, never a cached resource.
    """
    def __init__(self, endpoint: str, *, token_provider: Callable[[], str] | None = None,
                 transport: Transport | None = None) -> None:
        self.endpoint = validate_endpoint(endpoint)
        self._token_provider = token_provider or (lambda: '')
        self._transport = transport or HTTPTransport()
        data = self.request('/metadata')
        self.metadata = validate_metadata(data)
        token = self._token_provider()
        if token and any(token in str(value) for value in asdict(self.metadata).values()):
            raise AdapterError('Adapter metadata must not contain authentication credentials.')
        identity = {
            'source': 'remote', 'endpoint_host': self.endpoint.host,
            'endpoint_id': hashlib.sha256(endpoint.rstrip('/').encode()).hexdigest(),
            'name': self.metadata.name, 'protocol': self.metadata.protocol,
            'adapter_type': self.metadata.adapter_type,
            'dimension': self.metadata.dimension, 'similarity': self.metadata.similarity,
        }
        self.identity = identity
        self.model_id = 'remote:' + json.dumps(identity, sort_keys=True, ensure_ascii=True)

    def request(self, route: str, payload: dict | None = None) -> dict:
        token = self._token_provider()
        if not isinstance(token, str) or len(token) > 8192 or any(ord(c) < 33 or ord(c) > 126 for c in token):
            raise AdapterError('Adapter bearer token must contain only printable non-space ASCII characters.')
        if token and token in self.endpoint.host:
            raise AdapterError('Adapter endpoint must not contain authentication credentials.')
        body = None if payload is None else json.dumps(payload, allow_nan=False).encode('utf-8')
        if body is not None and len(body) > MAX_REQUEST_BYTES:
            raise AdapterError('Adapter request exceeds the 2 MiB request limit.')
        try:
            raw = self._transport.request(self.endpoint, 'GET' if body is None else 'POST', route, body, token)
        except AdapterError:
            raise
        except Exception:
            raise AdapterError('Adapter transport failed; check the service and authentication.') from None
        if not isinstance(raw, bytes) or len(raw) > MAX_RESPONSE_BYTES:
            raise AdapterError('Adapter response exceeds the 16 MiB response limit.')
        try:
            result = json.loads(raw.decode('utf-8'), parse_constant=_reject_constant, object_pairs_hook=_unique_object)
        except AdapterError:
            raise
        except (ValueError, UnicodeError, RecursionError):
            raise AdapterError('Adapter response must be valid UTF-8 JSON.') from None
        if not isinstance(result, dict):
            raise AdapterError('Adapter response must be a JSON object.')
        return result

    def batches(self, texts: list[str], *, query: str | None = None):
        if not isinstance(texts, list) or any(not isinstance(text, str) for text in texts):
            raise AdapterError('Adapter inputs must be a list of strings.')
        if query is not None:
            _validate_text(query)
        limit = min(self.metadata.max_batch_size, MAX_BATCH_SIZE)
        batch: list[str] = []
        size = len(json.dumps(query).encode()) if query is not None else 0
        # Reserve room for keys, punctuation, and the role/query envelope.
        base_size = size + 128
        size = base_size
        for text in texts:
            _validate_text(text)
            text_size = len(json.dumps(text).encode()) + 2
            if batch and (len(batch) >= limit or size + text_size > MAX_REQUEST_BYTES):
                yield batch
                batch, size = [], base_size
            if size + text_size > MAX_REQUEST_BYTES:
                raise AdapterError('Adapter text exceeds the request size limit.')
            batch.append(text)
            size += text_size
        if batch:
            yield batch


def _validate_text(text: str) -> None:
    if not isinstance(text, str) or len(text.encode('utf-8')) > MAX_TEXT_BYTES:
        raise AdapterError('Each adapter input text must be a string of at most 256 KiB.')


def _number(value: object) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


class RemoteDenseAdapter:
    device = 'remote'

    def __init__(self, client: RemoteAdapterClient) -> None:
        if client.metadata.adapter_type not in ('dense', 'dense+reranker'):
            raise AdapterError('Selected adapter does not support dense encoding.')
        self.client = client
        self.name = client.metadata.name
        self.dimension = client.metadata.dimension
        self.similarity = client.metadata.similarity
        self.model_id = client.model_id
        self.identity = client.identity

    def _encode(self, texts: list[str], role: str) -> np.ndarray:
        matrices = []
        for batch in self.client.batches(texts):
            vectors = self.client.request('/encode', {'role': role, 'texts': batch}).get('vectors')
            if not isinstance(vectors, list) or len(vectors) != len(batch):
                raise AdapterError('Encode vector count must equal input text count.')
            for vector in vectors:
                if not isinstance(vector, list) or len(vector) != self.dimension:
                    raise AdapterError('Every encode vector must match the metadata dimension.')
                if not all(_number(value) for value in vector):
                    raise AdapterError('Encode vectors must contain only finite numbers.')
            with np.errstate(over='ignore', invalid='ignore'):
                matrix = np.asarray(vectors, dtype=np.float32)
            if not np.isfinite(matrix).all():
                raise AdapterError('Encode vectors must be finite in float32 representation.')
            if self.similarity == 'cosine' and np.any(np.linalg.norm(matrix.astype(np.float64), axis=1) == 0):
                raise AdapterError('Cosine encode vectors must have non-zero norms.')
            matrices.append(matrix)
        return np.concatenate(matrices) if matrices else np.empty((0, self.dimension), dtype=np.float32)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, 'query')

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, 'document')


class RemoteRerankerAdapter:
    input_kind = 'text'

    def __init__(self, client: RemoteAdapterClient) -> None:
        if client.metadata.adapter_type not in ('reranker', 'dense+reranker'):
            raise AdapterError('Selected adapter does not support reranking.')
        self.client = client
        self.name = client.metadata.name
        self.model_id = client.model_id
        self.identity = client.identity

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        result = []
        for batch in self.client.batches(passages, query=query):
            scores = self.client.request('/rerank', {'query': query, 'passages': batch}).get('scores')
            if not isinstance(scores, list) or len(scores) != len(batch):
                raise AdapterError('Rerank score count must equal candidate passage count.')
            if not all(_number(value) for value in scores):
                raise AdapterError('Rerank scores must contain only finite numbers.')
            result.extend(scores)
        return np.asarray(result, dtype=np.float64)
