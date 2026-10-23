import os

import pytest


@pytest.mark.skipif(
    os.getenv("RUN_RERANKER_SMOKE") != "1",
    reason="real cross-encoder smoke is opt-in",
)
def test_real_cross_encoder_reranker_smoke() -> None:
    from src.reranker import CrossEncoderReranker
    from src.types import Passage, RankedPassage

    passages = [
        Passage("a", "a", "a.txt", "txt", None, 1, "Paris is in France.", 4),
        Passage("b", "b", "b.txt", "txt", None, 1, "Whales live in oceans.", 4),
    ]
    candidates = [
        RankedPassage(passage=passage, rank=index, score=1.0 / index)
        for index, passage in enumerate(passages, start=1)
    ]
    scores = CrossEncoderReranker(device="cpu").score(
        "Where is Paris?",
        candidates,
    )

    assert len(scores) == 2
    assert scores[0] > scores[1]
