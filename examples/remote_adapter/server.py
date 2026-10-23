"""Standalone dummy adapter. Replace only the three inference functions/model loading.

Run: uvicorn server:app --host 127.0.0.1 --port 8000
"""
import hashlib
import math
import re

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

DIMENSION = 32
METADATA = {
    'protocol': 'retrievallab-adapter/1',
    'name': 'Deterministic research template v1',
    'adapter_type': 'dense+reranker',
    'dimension': DIMENSION,
    'similarity': 'cosine',
    'max_batch_size': 32,
}


def _dummy_vectors(texts):
    result = []
    for text in texts:
        vector = [0.0] * DIMENSION
        for word in re.findall(r'\w+', text.casefold()):
            bucket = int.from_bytes(hashlib.sha256(word.encode()).digest()[:4], 'big') % DIMENSION
            vector[bucket] += 1.0
        if not any(vector):
            vector[0] = 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        result.append([value / norm for value in vector])
    return result


# REPLACE: Load your own model once in this separate service, on your own hardware.
# Keep metadata.dimension/similarity/name consistent with the loaded model revision.
def encode_queries(texts):
    """REPLACE with query inference; return N x dimension JSON-compatible numbers."""
    return _dummy_vectors(texts)


def encode_documents(texts):
    """REPLACE with document inference; same vector space as encode_queries."""
    return _dummy_vectors(texts)


def rerank(query, passages):
    """REPLACE with one finite, batch-independent relevance score per passage."""
    words = set(re.findall(r'\w+', query.casefold()))
    return [float(len(words & set(re.findall(r'\w+', text.casefold())))) for text in passages]


async def metadata(_request):
    return JSONResponse(METADATA)


async def inference(request: Request):
    # Bound streamed bodies before parsing, even without Content-Length.
    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw) > 2 * 1024 * 1024:
            return JSONResponse({'error': 'Request too large'}, status_code=413)
    try:
        import json
        payload = json.loads(raw)
        texts = payload.get('texts') if request.url.path == '/encode' else payload.get('passages')
        if (not isinstance(texts, list) or not 1 <= len(texts) <= METADATA['max_batch_size']
                or any(not isinstance(text, str) or len(text.encode()) > 256 * 1024 for text in texts)):
            raise ValueError
        if request.url.path == '/encode':
            role = payload.get('role')
            if role not in ('query', 'document'):
                raise ValueError
            function = encode_queries if role == 'query' else encode_documents
            return JSONResponse({'vectors': function(texts)})
        query = payload.get('query')
        if not isinstance(query, str) or len(query.encode()) > 256 * 1024:
            raise ValueError
        return JSONResponse({'scores': rerank(query, texts)})
    except (ValueError, TypeError, AttributeError):
        return JSONResponse({'error': 'Invalid adapter request'}, status_code=400)


app = Starlette(routes=[
    Route('/metadata', metadata),
    Route('/encode', inference, methods=['POST']),
    Route('/rerank', inference, methods=['POST']),
])
