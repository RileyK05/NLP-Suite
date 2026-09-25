"""The profiler publishes every artifact the embedding panels consume."""

from __future__ import annotations

import pandas as pd

from core.analysis.word_embeddings import TrainedW2V
from core.conll.schema import Col
from core.profiler import executor
from core.profiler.executor import BatchContext
from core.result import Result


def test_gensim_adapter_matches_cli_controls_and_emits_explorable_outputs(monkeypatch) -> None:
    model = TrainedW2V(
        words=("government", "state", "nation", "policy", "public"),
        counts=(5, 4, 3, 2, 1),
        vectors=((1.0, 0.0), (0.9, 0.1), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)),
        vector_size=2,
        window=5,
        min_count=1,
        sg=1,
        seed=42,
        epochs=4,
    )
    seen: dict[str, object] = {}

    def train(frame: pd.DataFrame, **kwargs: object) -> Result[TrainedW2V]:
        seen.update(kwargs)
        return Result.success(model)

    monkeypatch.setattr(executor, "train_w2v", train)
    monkeypatch.setattr(
        executor,
        "project_tsne",
        lambda _model, seed: Result.success(pd.DataFrame({"Word": model.words, "X": range(5), "Y": range(5)})),
    )
    from core.viz import embeddings as embedding_viz

    monkeypatch.setattr(embedding_viz, "tsne_html", lambda *_args, **_kwargs: Result.success("<html></html>"))

    result = executor._adapt_word_embeddings(
        BatchContext(table=pd.DataFrame({"Form": ["X"], "Lemma": ["x"]})),
        {
            "field": "form",
            "remove-stopwords": True,
            "min-count": 1,
            "vector-size": 2,
            "window": 5,
            "seed": 42,
            "epochs": 4,
            "sg": 1,
            "query": "government",
            "top-n": 2,
        },
    )
    frames = result.unwrap()

    assert seen["field"] is Col.FORM
    assert seen["remove_stopwords"] is True
    assert set(frames) == {"vectors.csv", "tsne.csv", "tsne.html", "neighbours.csv"}
    # Word class is additive: the meaning figures keep to nouns or adjectives with it.
    assert frames["vectors.csv"].columns.tolist() == ["Word", "Count", "Vector", "Word class"]
    # Count is additive: the map needs it to show frequent words, not outliers;
    # Group (a meaning group, when the vocabulary has enough words) names its regions.
    assert frames["tsne.csv"].columns.tolist()[:4] == ["Word", "X", "Y", "Count"]
    assert set(frames["tsne.csv"].columns) <= {"Word", "X", "Y", "Count", "Group"}
    assert frames["neighbours.csv"].columns.tolist() == ["Word", "Neighbor", "Cosine"]
    assert frames["neighbours.csv"].iloc[0]["Neighbor"] == "state"
