"""All remote inference uses in-process byte responses; no network access."""
import json

import numpy as np
import pytest

from src.benchmark_service import ALL_BENCHMARK_CONFIGURATIONS, BenchmarkService, benchmark_to_csv, benchmark_to_json
from src.bm25_retriever import BM25SearchEngine
from src.evaluation import EvaluationQuery
from src.evaluation_service import EvaluationService
from src.hybrid_retriever import HybridSearchEngine
from src.model_selection import invalidate_dense_indexes_for_model
from src.remote_adapter import AdapterError, RemoteAdapterClient, RemoteDenseAdapter, RemoteRerankerAdapter, validate_metadata
from src.search_engine import DenseSearchEngine
from src.services import CorpusService, SearchService, build_dense_engines
from .helpers import FakeEmbedder, make_test_config
from .test_reranker import _passages

SECRET = 'test-only-bearer-credential'
METADATA = dict(protocol='retrievallab-adapter/1', name='Research v1', adapter_type='dense+reranker',
                dimension=3, similarity='cosine', max_batch_size=2)


class FakeTransport:
    def __init__(self, metadata=None):
        self.metadata = dict(METADATA if metadata is None else metadata)
        self.calls = []
        self.overrides = {}

    def request(self, endpoint, method, route, body, token):
        payload = json.loads(body) if body else None
        self.calls.append((route, payload))
        assert token in ('', SECRET)
        if route in self.overrides:
            return self.overrides[route]
        if route == '/metadata':
            return json.dumps(self.metadata).encode()
        if route == '/encode':
            vectors = [[1.0, float(len(text) % 3), 0.5] for text in payload['texts']]
            return json.dumps({'vectors': vectors}).encode()
        return json.dumps({'scores': [float(len(text)) for text in payload['passages']]}).encode()


def client(transport=None, endpoint='https://adapter.example.org/v1'):
    return RemoteAdapterClient(endpoint, token_provider=lambda: SECRET, transport=transport or FakeTransport())


@pytest.mark.parametrize('change', [
    {'protocol': 'retrievallab-adapter/2'}, {'adapter_type': 'other'}, {'dimension': 0},
    {'dimension': True}, {'dimension': 1.2}, {'dimension': 16385}, {'similarity': 'l2'},
    {'max_batch_size': 0}, {'max_batch_size': True}, {'name': ''}, {'name': '\nsecret'},
])
def test_bad_metadata_rejected(change):
    with pytest.raises(AdapterError):
        validate_metadata(METADATA | change)


def test_metadata_capabilities_and_dimension():
    dense = RemoteDenseAdapter(client())
    assert dense.dimension == 3 and dense.similarity == 'cosine' and dense.name == 'Research v1'
    with pytest.raises(AdapterError, match='dense'):
        RemoteDenseAdapter(client(FakeTransport(METADATA | {'adapter_type': 'reranker'})))
    with pytest.raises(AdapterError, match='reranking'):
        RemoteRerankerAdapter(client(FakeTransport(METADATA | {'adapter_type': 'dense'})))
    assert validate_metadata({'protocol': METADATA['protocol'], 'name': 'r', 'adapter_type': 'reranker'}).dimension is None


def test_roles_batching_and_empty_input():
    transport = FakeTransport()
    dense = RemoteDenseAdapter(client(transport))
    assert dense.encode_documents(['a', 'bb', 'ccc', 'dddd', 'e']).shape == (5, 3)
    assert dense.encode_queries(['query']).shape == (1, 3)
    assert dense.encode_documents([]).shape == (0, 3)
    calls = [payload for route, payload in transport.calls if route == '/encode']
    assert [len(item['texts']) for item in calls] == [2, 2, 1, 1]
    assert [item['role'] for item in calls] == ['document', 'document', 'document', 'query']


@pytest.mark.parametrize('vectors,match', [
    ([], 'count'), ([[1, 2]], 'dimension'), ([[1, 2, 3], [1, 2, 3]], 'count'),
    ([[True, 2, 3]], 'finite'), ([['1', 2, 3]], 'finite'), ([[float('nan'), 2, 3]], 'finite'),
    ([[float('inf'), 2, 3]], 'finite'), ([[1e100, 2, 3]], 'float32'), ([[0, 0, 0]], 'non-zero'),
])
def test_malformed_vectors(vectors, match):
    transport = FakeTransport()
    dense = RemoteDenseAdapter(client(transport))
    transport.overrides['/encode'] = json.dumps({'vectors': vectors}).encode()
    with pytest.raises(AdapterError, match=match):
        dense.encode_queries(['q'])


@pytest.mark.parametrize('scores,match', [([], 'count'), ([1, 2], 'count'),
    ([float('nan')], 'finite'), ([float('inf')], 'finite'), ([True], 'finite'), (['1'], 'finite')])
def test_malformed_scores(scores, match):
    transport = FakeTransport()
    reranker = RemoteRerankerAdapter(client(transport))
    transport.overrides['/rerank'] = json.dumps({'scores': scores}).encode()
    with pytest.raises(AdapterError, match=match):
        reranker.score('q', ['p'])


@pytest.mark.parametrize('raw', [b'<html>secret</html>', b'[]', b'\xff', b'{"scores":[],"scores":[]}'])
def test_invalid_json(raw):
    transport = FakeTransport()
    remote = client(transport)
    transport.overrides['/rerank'] = raw
    with pytest.raises(AdapterError):
        remote.request('/rerank', {'query': 'q', 'passages': ['p']})


def engines(count=6):
    transport = FakeTransport()
    remote = client(transport)
    dense = DenseSearchEngine(make_test_config(), RemoteDenseAdapter(remote))
    dense.index_passages(_passages(count))
    bm25 = BM25SearchEngine(make_test_config(), dense.passages)
    return {'Dense': dense, 'BM25': bm25, 'Hybrid': HybridSearchEngine(dense, bm25)}, remote, transport


@pytest.mark.parametrize('pipeline', ['Dense', 'BM25', 'Hybrid'])
def test_remote_reranking_fixed_candidates_and_provenance(pipeline):
    mapping, remote, transport = engines(105)
    engine = mapping[pipeline]
    originals = engine.search_candidates('passage 2').results
    response = SearchService(engine, RemoteRerankerAdapter(remote)).search('passage 2', result_limit=100)
    assert len(response.results) == len(originals) == 100
    assert {r.passage.chunk_id for r in response.results} == {r.passage.chunk_id for r in originals}
    original_map = {r.passage.chunk_id: r for r in originals}
    for result in response.results:
        original = original_map[result.passage.chunk_id]
        assert result.retrieval_rank == original.rank
        assert result.retrieval_score == original.score
        assert (result.dense_rank, result.bm25_rank, result.rrf_score) == (original.dense_rank, original.bm25_rank, original.rrf_score)
    supplied = [text for route, payload in transport.calls if route == '/rerank' for text in payload['passages']]
    assert supplied == [r.passage.text for r in originals]
    assert response.reranker_name.startswith('remote:')


def test_remote_dense_dot_and_cosine_rank_differ():
    def search(similarity):
        transport = FakeTransport(METADATA | {'similarity': similarity})
        dense = DenseSearchEngine(make_test_config(), RemoteDenseAdapter(client(transport)))
        transport.overrides['/encode'] = b'{"vectors":[[1,0,0],[2,2,0]]}'
        dense.index_passages(_passages(2))
        transport.overrides['/encode'] = b'{"vectors":[[1,0,0]]}'
        return dense.search('q')
    assert search('dot').results[0].passage.chunk_id == 'chunk-1'
    assert search('cosine').results[0].passage.chunk_id == 'chunk-0'


def test_failed_reindex_preserves_original_corpus():
    mapping, _, transport = engines()
    dense = mapping['Dense']
    old_index, old_embeddings = dense.vector_index, dense.embeddings
    transport.overrides['/encode'] = b'{"vectors":[]}'
    with pytest.raises(AdapterError):
        dense.index_passages(_passages(3))
    assert dense.vector_index is old_index and dense.embeddings is old_embeddings


def test_index_lifecycle_identity_and_bm25_independence():
    mapping, remote, _ = engines()
    dense = mapping['Dense']
    state = {'engine': dense, 'hybrid_engine': mapping['Hybrid'], 'bm25_engine': mapping['BM25']}
    assert not invalidate_dense_indexes_for_model(state, remote.model_id)
    SearchService(dense, RemoteRerankerAdapter(client(FakeTransport(METADATA | {'name': 'Other reranker'})))).search('q')
    assert state['engine'] is dense
    changed = client(FakeTransport(METADATA | {'name': 'Research v2'}))
    assert changed.model_id != remote.model_id
    assert invalidate_dense_indexes_for_model(state, changed.model_id)
    assert 'engine' not in state and 'hybrid_engine' not in state
    assert state['bm25_engine'] is mapping['BM25']


def test_benchmark_reuses_vectors_matches_local_metrics_and_excludes_secrets():
    mapping, remote, transport = engines()
    dense = mapping['Dense']
    query = EvaluationQuery('q', 'passage')
    qrels = {'q': {'document-0': 1.0}}
    interactive = SearchService(mapping['Hybrid']).search('passage')
    state = {'search_response': interactive, 'engine': dense}
    before_docs = sum(p['role'] == 'document' for r, p in transport.calls if r == '/encode')
    result = BenchmarkService(mapping, reranker_loader=lambda: RemoteRerankerAdapter(remote)).evaluate(
        ALL_BENCHMARK_CONFIGURATIONS, [query], qrels)
    assert len(result.successful_results) == 6
    assert state['search_response'] is interactive and state['engine'] is dense
    assert before_docs == sum(p['role'] == 'document' for r, p in transport.calls if r == '/encode')
    assert dense.top_results == 10
    for output in (benchmark_to_json(result), benchmark_to_csv(result), remote.model_id, repr(remote.identity)):
        assert SECRET not in output
        assert 'remote' in output and 'adapter.example.org' in output
    query_vector = dense.embedder.encode_queries(['passage'])[0]
    fake = FakeEmbedder(dense.embeddings, query_vector)
    local = DenseSearchEngine(make_test_config(), fake)
    local.index_passages(dense.passages)
    class LocalReranker:
        name = 'local equivalent'
        input_kind = 'text'
        def score(self, query, passages):
            return np.array([len(text) for text in passages], dtype=float)
    for key, remote_engine in mapping.items():
        local_engine = {'Dense': local, 'BM25': mapping['BM25'],
                        'Hybrid': HybridSearchEngine(local, mapping['BM25'])}[key]
        expected = EvaluationService(local_engine, LocalReranker()).evaluate([query], qrels)
        actual = EvaluationService(remote_engine, RemoteRerankerAdapter(remote)).evaluate([query], qrels)
        assert actual.metrics == expected.metrics
        assert actual.reranker_model_id == remote.model_id
        if key != 'BM25':
            assert actual.dense_model_id == remote.model_id


def test_remote_corpus_preparation_and_existing_passages():
    remote = client()
    prepared = CorpusService(make_test_config(), lambda: RemoteDenseAdapter(remote)).prepare([('a.txt', b'alpha beta gamma')])
    assert prepared.engine.search('alpha').results
    new_remote = client(FakeTransport(METADATA | {'name': 'new'}))
    dense, hybrid = build_dense_engines(make_test_config(), prepared.passage_records,
        lambda: RemoteDenseAdapter(new_remote), bm25_engine=prepared.bm25_engine)
    assert dense.passages == prepared.passage_records
    assert hybrid.bm25_engine is prepared.bm25_engine


def test_response_and_request_limits(monkeypatch):
    transport = FakeTransport()
    remote = client(transport)
    monkeypatch.setattr('src.remote_adapter.MAX_RESPONSE_BYTES', 8)
    with pytest.raises(AdapterError, match='response limit'):
        remote.request('/encode', {'texts': ['x']})
    with pytest.raises(AdapterError, match='256 KiB'):
        RemoteDenseAdapter(remote).encode_queries(['x' * (256 * 1024 + 1)])
    monkeypatch.setattr('src.remote_adapter.MAX_REQUEST_BYTES', 200)
    batches = list(remote.batches(['x' * 60, 'y' * 60]))
    assert len(batches) == 2


def test_secrets_not_reflected_in_metadata_or_errors(caplog):
    transport = FakeTransport(METADATA | {'name': SECRET})
    with pytest.raises(AdapterError) as caught:
        client(transport)
    assert SECRET not in str(caught.value)
    class FailingTransport:
        def request(self, *args):
            raise RuntimeError(SECRET)
    with pytest.raises(AdapterError) as caught:
        client(FailingTransport())
    assert SECRET not in str(caught.value) and SECRET not in caplog.text
    assert caught.value.__suppress_context__


def test_remote_dense_composes_with_existing_cross_encoder():
    from src.reranker import CrossEncoderReranker
    from .test_reranker import _FakeCrossEncoder
    mapping, _, _ = engines()
    reranker = CrossEncoderReranker(device='cpu')
    reranker._model = _FakeCrossEncoder()
    result = BenchmarkService(mapping, reranker_loader=lambda: reranker).evaluate(
        ALL_BENCHMARK_CONFIGURATIONS, [EvaluationQuery('q', 'passage')], {'q': {'document-0': 1}})
    assert len(result.successful_results) == 6


@pytest.mark.parametrize('change', [{'name': 'new'}, {'dimension': 4}, {'similarity': 'dot'}, {'adapter_type': 'dense'}])
def test_identity_tracks_model_contract(change):
    assert client().model_id != client(FakeTransport(METADATA | change)).model_id
    assert client().model_id != client(endpoint='https://adapter.example.org/v2').model_id


def test_secret_with_json_escape_cannot_be_echoed_in_identity():
    token = 'credential"with-quote'
    class Echo:
        def request(self, *args):
            return json.dumps(METADATA | {'name': token}).encode()
    with pytest.raises(AdapterError, match='credentials'):
        RemoteAdapterClient('https://adapter.example.org', token_provider=lambda: token, transport=Echo())


def test_dot_overflow_is_rejected():
    transport = FakeTransport(METADATA | {'similarity': 'dot'})
    dense = DenseSearchEngine(make_test_config(), RemoteDenseAdapter(client(transport)))
    transport.overrides['/encode'] = b'{"vectors":[[1e30,1e30,1e30]]}'
    dense.index_passages(_passages(1))
    with pytest.raises(ValueError, match='overflowed'):
        dense.search('q')
