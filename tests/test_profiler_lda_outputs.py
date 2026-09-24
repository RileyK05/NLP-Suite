"""Desktop LDA settings must reach the same tokenization choices as the CLI."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from core.conll.schema import Col
from core.profiler import executor
from core.profiler.executor import BatchContext
from core.result import Result


def test_gensim_lda_adapter_passes_selected_field_into_model_and_flow_tokens(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def token_frame(frame: pd.DataFrame, **kwargs: object) -> dict[str, list[str]]:
        seen["frame"] = frame
        seen["tokens_kwargs"] = kwargs
        return {"speech.txt": ["policy", "public"]}

    def fit(tokens: dict[str, list[str]], **kwargs: object) -> Result[SimpleNamespace]:
        seen["tokens"] = tokens
        seen["fit_kwargs"] = kwargs
        return Result.success(
            SimpleNamespace(
                topics=pd.DataFrame({"Topic": [0], "Word": ["policy"], "Weight": [0.5]}),
                dominant=pd.DataFrame(columns=["Document", "Dominant topic"]),
                relevance=None,
                intertopic=None,
                flow=None,
            )
        )

    monkeypatch.setattr(executor, "tokens_from_frame", token_frame)
    monkeypatch.setattr(executor, "fit_lda", fit)
    context = BatchContext(table=pd.DataFrame({"Form": ["policy"], "Lemma": ["policy"]}))

    result = executor._adapt_lda_gensim(context, {"field": "form", "topics": 2, "keep-stopwords": True})

    assert result.ok
    assert seen["tokens_kwargs"]["field"] is Col.FORM
    assert seen["tokens_kwargs"]["nouns_only"] is False
    assert seen["fit_kwargs"]["remove_stopwords"] is False
    assert set(result.unwrap()) == {"topics.csv", "topics_dominant.csv"}
