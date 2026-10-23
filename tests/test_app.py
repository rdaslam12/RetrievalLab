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
