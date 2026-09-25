"""What the real exported models say, through the app's own runtime.

The tiny fixtures (test_models.py) prove the plumbing; these prove the
models. They need real exports and run only when
``NLP_SUITE_REAL_MODELS`` names a directory of them (``models/`` after
``scripts/export_models.py``, or an unpacked release)::

    NLP_SUITE_REAL_MODELS=models python -m pytest tests/test_models_real.py -m real_models
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import REAL_MODELS
from core.models import onnx_backend, registry
from core.models.locate import status

pytestmark = [
    pytest.mark.real_models,
    pytest.mark.skipif(not REAL_MODELS, reason="set NLP_SUITE_REAL_MODELS to a directory of exported models"),
]


def opened(model: str) -> object:
    spec = registry.get_model(model)
    assert spec is not None
    if status(spec) != "ready":
        pytest.skip(f"{model} is not in {REAL_MODELS}")
    return onnx_backend.open_model(model).unwrap()


def frame(sentences: list[str]) -> pd.DataFrame:
    rows = []
    for sent_id, sentence in enumerate(sentences, 1):
        for index, form in enumerate(sentence.split(), 1):
            rows.append(
                {
                    "ID": index,
                    "Form": form,
                    "Lemma": form.lower(),
                    "POS": "NN",
                    "NER": "O",
                    "Head": 0,
                    "DepRel": "dep",
                    "Sentence ID": sent_id,
                    "Document ID": 1,
                    "Document": "d.txt",
                }
            )
    return pd.DataFrame(rows)


MONEY = ["the bank raised interest rates again", "the bank approved the loan", "the central bank printed money"]
RIVER = ["we sat on the river bank", "fish swam near the muddy bank", "the flood covered the bank of the river"]


def test_sentiment_reads_obvious_sentences() -> None:
    classifier = opened("distilbert-sst2")
    verdicts = classifier.classify(["I love this movie, it is wonderful.", "This was the worst meal of my life."])
    assert verdicts[0][0] == "POSITIVE" and verdicts[0][1] > 0.9
    assert verdicts[1][0] == "NEGATIVE" and verdicts[1][1] > 0.9


def test_bert_separates_the_two_banks() -> None:
    backend = opened("bert-base-uncased")
    vectors = np.array(backend.embed(["bank"] * 6, MONEY + RIVER))
    similarity = vectors @ vectors.T
    within = (similarity[:3, :3].sum() - 3 + similarity[3:, 3:].sum() - 3) / 12
    across = similarity[:3, 3:].mean()
    assert within > across + 0.05


def test_word_senses_split_bank_by_meaning() -> None:
    from core.analysis.contextual import contextual_vectors, wsi_senses
    from core.conll.schema import Col

    opened("bert-base-uncased")
    table = frame([*MONEY, "a bank manager closed the account", *RIVER, "the grassy bank of the stream"])
    vectors = contextual_vectors(table, field=Col.FORM, model="bert-base-uncased").unwrap()
    senses = wsi_senses(table, vectors=vectors).unwrap()
    bank = senses[senses["Lemma"] == "bank"].sort_values("Sentence ID")["Sense"].tolist()
    assert len(set(bank[:4])) == 1 and len(set(bank[4:])) == 1 and bank[0] != bank[4]


@pytest.mark.parametrize("model", ["granite-embedding-english-r2", "qwen3-embedding-0.6b"])
def test_sentence_models_rank_paraphrases_first(model: str) -> None:
    backend = opened(model)
    texts = [
        "The central bank raised interest rates.",
        "Borrowing costs went up after the monetary authority's decision.",
        "The children played football in the park.",
    ]
    vectors = backend.embed_texts(texts)
    assert vectors[0] @ vectors[1] > vectors[0] @ vectors[2] + 0.1


def test_qwen_search_uses_its_query_prompt() -> None:
    backend = opened("qwen3-embedding-0.6b")
    documents = backend.embed_texts(MONEY + RIVER)
    query = backend.embed_texts(["who sets interest rates?"], query=True)[0]
    assert int(np.argmax(documents @ query)) < 3
