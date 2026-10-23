from streamlit.testing.v1 import AppTest

from src.config import load_config
from src.reranker import DEFAULT_RERANKER_MODEL


def test_app_starts_with_default_models_and_conditional_custom_fields() -> None:
    app = AppTest.from_file("app.py", default_timeout=10).run()

    assert not app.exception
    assert app.radio("embedding_model_mode").value == "Default"
    assert app.radio("reranker_model_mode").value == "Default"
    assert app.radio("retrieval_pipeline").value == "Dense"
    assert app.radio("reranker").value == "Off"
    assert len(app.text_input) == 0
    assert load_config().model_id in app.caption[0].value
    assert DEFAULT_RERANKER_MODEL in app.caption[2].value

    app.radio("embedding_model_mode").set_value("Custom").run()
    assert not app.exception
    assert app.text_input("custom_embedding_model_id")
    assert len(app.text_input) == 1

    app.radio("reranker_model_mode").set_value("Custom").run()
    assert not app.exception
    assert app.text_input("custom_embedding_model_id")
    assert app.text_input("custom_reranker_model_id")
    assert len(app.text_input) == 2


def test_evaluation_workspace_starts_separate_and_disabled_without_a_corpus() -> None:
    app = AppTest.from_file("app.py", default_timeout=10).run()

    assert not app.exception
    assert app.radio("evaluation_mode").value == "Evaluate one pipeline"
    assert app.file_uploader("evaluation_queries").label == "Query JSONL"
    assert app.file_uploader("evaluation_qrels").label == "Qrels TSV"
    assert app.button("run_evaluation").disabled
    assert any("Evaluate retrieval" in item.value for item in app.markdown)


def test_benchmark_mode_has_default_pipelines_and_requires_indexes() -> None:
    app = AppTest.from_file("app.py", default_timeout=10).run()

    app.radio("evaluation_mode").set_value("Compare pipelines").run()

    assert not app.exception
    assert app.multiselect("benchmark_configurations").value == [
        "Dense",
        "BM25",
        "Hybrid",
    ]
    assert app.button("run_benchmark").disabled
    assert any("Dense index is unavailable or stale" in item.value for item in app.error)


def test_remote_settings_connect_and_clear_stale_metadata(monkeypatch) -> None:
    from src.remote_adapter import RemoteAdapterClient
    from tests.test_remote_adapter import FakeTransport, SECRET
    transport = FakeTransport()
    monkeypatch.setattr('src.remote_adapter.HTTPTransport', lambda: transport)
    app = AppTest.from_file('app.py', default_timeout=10).run()
    app.radio('embedding_model_mode').set_value('Remote Adapter').run()
    assert not app.exception
    assert app.text_input('remote_embedding_token').proto.type == 1
    app.text_input('remote_embedding_endpoint').set_value('https://adapter.example.org').run()
    app.text_input('remote_embedding_token').set_value(SECRET).run()
    app.button('test_embedding_adapter').click().run()
    assert not app.exception
    assert isinstance(app.session_state['remote_embedding_client'], RemoteAdapterClient)
    assert any('Research v1' in item.value and '3 dimensions' in item.value for item in app.text)
    assert SECRET not in '\n'.join(item.value for item in app.text)
    app.text_input('remote_embedding_endpoint').set_value('https://other.example.org').run()
    assert not app.exception
    assert 'remote_embedding_client' not in app.session_state
    app.radio('reranker_model_mode').set_value('Remote Adapter').run()
    assert not app.exception
    assert app.radio('reranker').value == 'Remote Adapter'
    app.text_input('remote_reranker_endpoint').set_value('https://adapter.example.org').run()
    app.button('test_reranker_adapter').click().run()
    assert not app.exception
    app.radio('evaluation_mode').set_value('Compare pipelines').run()
    assert 'Dense + Remote Reranker' in app.multiselect('benchmark_configurations').options
    app.radio('reranker_model_mode').set_value('Off').run()
    assert not app.exception
    assert app.radio('reranker').value == 'Off'


def test_remote_connection_error_is_safe_and_actionable(monkeypatch) -> None:
    from tests.test_remote_adapter import FakeTransport, METADATA, SECRET
    monkeypatch.setattr('src.remote_adapter.HTTPTransport', lambda: FakeTransport(METADATA | {'name': SECRET}))
    app = AppTest.from_file('app.py', default_timeout=10).run()
    app.radio('embedding_model_mode').set_value('Remote Adapter').run()
    app.text_input('remote_embedding_endpoint').set_value('https://adapter.example.org').run()
    app.text_input('remote_embedding_token').set_value(SECRET).run()
    app.button('test_embedding_adapter').click().run()
    assert not app.exception
    assert 'remote_embedding_client' not in app.session_state
    assert any('credentials' in item.value for item in app.error)
    assert all(SECRET not in item.value for item in app.error)


def test_remote_benchmark_preserves_interactive_search_and_dense_index(monkeypatch) -> None:
    from tests.test_remote_adapter import FakeTransport, SECRET
    transport = FakeTransport()
    monkeypatch.setattr('src.remote_adapter.HTTPTransport', lambda: transport)
    app = AppTest.from_file('app.py', default_timeout=10).run()
    for stage, mode_key in [('embedding', 'embedding_model_mode'), ('reranker', 'reranker_model_mode')]:
        app.radio(mode_key).set_value('Remote Adapter').run()
        app.text_input(f'remote_{stage}_endpoint').set_value('https://adapter.example.org').run()
        app.text_input(f'remote_{stage}_token').set_value(SECRET).run()
        app.button(f'test_{stage}_adapter').click().run()
    app.file_uploader('corpus_upload_0').set_value(('corpus.jsonl',
        b'{"_id":"d1","text":"passage dense"}\n{"_id":"d2","text":"passage lexical"}\n',
        'application/jsonl')).run()
    next(button for button in app.button if button.label == 'Index corpus').click().run()
    assert not app.exception
    engine = app.session_state['engine']
    index = engine.vector_index
    app.chat_input('retrieval_query').set_value('passage').run()
    assert not app.exception
    response = app.session_state['search_response']
    assert response.reranking_used
    document_calls = sum(p['role'] == 'document' for r, p in transport.calls if r == '/encode')
    app.radio('evaluation_mode').set_value('Compare pipelines').run()
    app.multiselect('benchmark_configurations').set_value([
        'Dense', 'BM25', 'Hybrid', 'Dense + Remote Reranker', 'BM25 + Remote Reranker', 'Hybrid + Remote Reranker'
    ]).run()
    app.file_uploader('evaluation_queries').set_value(('queries.jsonl', b'{"_id":"q1","text":"passage"}\n', 'application/jsonl')).run()
    app.file_uploader('evaluation_qrels').set_value(('qrels.tsv', b'query-id\tcorpus-id\tscore\nq1\td1\t1\n', 'text/tab-separated-values')).run()
    assert not app.button('run_benchmark').disabled
    app.button('run_benchmark').click().run()
    assert not app.exception
    assert len(app.session_state['benchmark_result'].result.successful_results) == 6
    assert app.session_state['search_response'] is response
    assert app.session_state['engine'] is engine and engine.vector_index is index
    assert document_calls == sum(p['role'] == 'document' for r, p in transport.calls if r == '/encode')
    app.radio('reranker_model_mode').set_value('Off').run()
    assert app.session_state['engine'] is engine and engine.vector_index is index
