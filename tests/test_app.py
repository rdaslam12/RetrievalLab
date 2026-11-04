from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.config import load_config
from src.reranker import DEFAULT_RERANKER_MODEL


APP_TEST_TIMEOUT_SECONDS = 30
WORKBENCH_TEST_SCRIPT = "from app import render_workbench\nrender_workbench()"


def _run_workbench() -> AppTest:
    return AppTest.from_string(
        WORKBENCH_TEST_SCRIPT,
        default_timeout=APP_TEST_TIMEOUT_SECONDS,
    ).run()


def test_landing_page_tells_the_product_story_without_workbench_widgets() -> None:
    app = AppTest.from_file("app.py", default_timeout=APP_TEST_TIMEOUT_SECONDS).run()

    assert not app.exception
    markdown_markup = [item.value for item in app.markdown]
    html_markup = [item.proto.body for item in app.get("html")]
    markup = "\n".join([*markdown_markup, *html_markup])
    assert 'class="landing-page-marker"' in markup
    assert [link.label for link in app.get("page_link")] == ["Home", "Workbench"]
    assert "Upload your documents." in markup
    assert "Search smarter." in markup
    assert "Your models. Your experiments." in markup
    assert "How does RetrievalLab work?" in markup
    assert "01</span>" in markup and "04</span>" in markup
    assert app.button("landing_explore").label == "Explore RetrievalLab →"
    assert app.button("landing_final_cta").label == "Open the workbench →"
    assert len(app.file_uploader) == 0
    assert len(app.chat_input) == 0
    assert len(app.radio) == 0


def test_landing_motion_is_fast_continuous_and_reduced_motion_safe() -> None:
    stylesheet = Path("assets/landing.css").read_text(encoding="utf-8")

    assert stylesheet.count("9.6s linear infinite") == 3
    assert "29.33%" in stylesheet and "33.33%" in stylesheet
    assert "62.66%" in stylesheet and "66.66%" in stylesheet
    assert "@supports (animation-timeline: view())" in stylesheet
    assert "@media (prefers-reduced-motion: reduce)" in stylesheet


def test_landing_halftone_is_scoped_responsive_and_resource_aware() -> None:
    effect = Path("assets/landing_halftone.html").read_text(encoding="utf-8")
    stylesheet = Path("assets/landing.css").read_text(encoding="utf-8")

    assert '<canvas class="landing-halftone" aria-hidden="true">' in effect
    assert "ResizeObserver" in effect
    assert "IntersectionObserver" in effect
    assert "MutationObserver" in effect
    assert "Math.min(window.devicePixelRatio || 1, 1.75)" in effect
    assert "maxFrameRate = 18" in effect
    assert 'document.addEventListener("visibilitychange"' in effect
    assert 'matchMedia("(prefers-reduced-motion: reduce)")' in effect
    assert "controller.abort()" in effect
    assert "pointer-events: none" in stylesheet
    assert "mask-image: radial-gradient" in stylesheet


def test_app_starts_with_default_models_and_conditional_custom_fields() -> None:
    app = _run_workbench()

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
    app = _run_workbench()

    assert not app.exception
    assert app.radio("evaluation_mode").value == "Evaluate one pipeline"
    assert app.file_uploader("evaluation_queries").label == "Query JSONL"
    assert app.file_uploader("evaluation_qrels").label == "Qrels TSV"
    assert app.button("run_evaluation").disabled
    assert any("3. Measure and compare search" in item.value for item in app.markdown)


def test_benchmark_mode_has_default_pipelines_and_requires_indexes() -> None:
    app = _run_workbench()

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
    app = _run_workbench()
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
    app = _run_workbench()
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
    app = _run_workbench()
    for stage, mode_key in [('embedding', 'embedding_model_mode'), ('reranker', 'reranker_model_mode')]:
        app.radio(mode_key).set_value('Remote Adapter').run()
        app.text_input(f'remote_{stage}_endpoint').set_value('https://adapter.example.org').run()
        app.text_input(f'remote_{stage}_token').set_value(SECRET).run()
        app.button(f'test_{stage}_adapter').click().run()
    app.file_uploader('corpus_upload_0').set_value(('corpus.jsonl',
        b'{"_id":"d1","text":"passage dense"}\n{"_id":"d2","text":"passage lexical"}\n',
        'application/jsonl')).run()
    next(button for button in app.button if button.label == 'Prepare for search').click().run()
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
