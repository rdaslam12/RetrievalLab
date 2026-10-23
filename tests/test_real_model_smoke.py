import os

import pytest


@pytest.mark.skipif(os.getenv("RUN_MODEL_SMOKE") != "1", reason="real model smoke is opt-in")
def test_real_arctic_s_encoder_smoke() -> None:
    from src.config import load_config
    from src.embedder import ArcticEmbedder

    config = load_config()
    embedder = ArcticEmbedder(config.model_id, config.embedding_dimension, batch_size=2)
    passages = embedder.encode_passages(["A technical passage.", "A scientific observation."])
    query = embedder.encode_query("What was observed?")
    assert passages.shape == (2, 384)
    assert query.shape == (384,)
