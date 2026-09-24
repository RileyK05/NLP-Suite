"""The one noun-tag rule, tested on both tagsets.

Seven modules used to carry their own copy of this predicate; two of the
copies disagreed with the fix when it was made (one accepted Universal tags
only, which kept nothing on a Penn parse). Consolidation exists so a change
to the rule is one edit, and so a test here covers every caller.
"""

from __future__ import annotations

import pandas as pd

from core.analysis.postags import is_noun_tag


class TestTheRule:
    def test_penn_nouns_in_all_their_forms(self) -> None:
        for tag in ("NN", "NNS", "NNP", "NNPS", "nn", " nns "):
            assert is_noun_tag(tag), tag

    def test_universal_nouns(self) -> None:
        assert is_noun_tag("NOUN") and is_noun_tag("PROPN")

    def test_every_non_noun_tag_is_refused(self) -> None:
        for tag in ("VB", "VBZ", "JJ", "RB", "IN", "PRON", "VERB", "ADJ", "", "  "):
            assert not is_noun_tag(tag), tag

    def test_penn_prefixes_admit_only_what_penn_actually_tags(self) -> None:
        """The rule is a Penn prefix check on purpose -- the same rule every
        noun check in this codebase applied before it was consolidated, and
        the only one that works for both NN and NNPS without a whitelist.
        "NNSX" is not a Penn tag and is admitted by the prefix; that is the
        rule's known shape, not a defect: real parses never emit it, and a
        whitelist would drift from the one thing this predicate must agree
        with (every other module's historical behaviour)."""
        assert is_noun_tag("NNSX")

    def test_any_input_is_answered(self) -> None:
        """Cells hold None, numbers and whitespace; the helper answers for
        the normalised string rather than trusting the caller."""
        assert not is_noun_tag(None)
        assert not is_noun_tag(1934)
        assert is_noun_tag("NNP")  # after strip/upper, whatever arrived

    def test_the_tag_tuple_is_not_restated_in_the_callers(self) -> None:
        """The consolidation is the point: the rule lives here. Callers may
        import the tuple or the function; a second inline (\"NOUN\",
        \"PROPN\") pair beside a startswith(\"NN\") is a copy that drifted
        the moment it was written."""
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        suspects = (
            "core/analysis/conll_wordlist.py",
            "core/analysis/nominalization.py",
            "core/analysis/lda.py",
            "core/analysis/k_sentences.py",
            "core/viz/shapes.py",
        )
        offenders: list[str] = []
        for name in suspects:
            source = (root / name).read_text(encoding="utf-8")
            if '("NOUN", "PROPN")' in source or '("NOUN", "PROPN", "PRON")' in source:
                offenders.append(name)
        assert not offenders, (
            f"{offenders} restate the noun tag tuple instead of importing "
            "NOUN_TAGS/is_noun_tag from core/analysis/postags.py"
        )

    def test_the_helper_agrees_with_the_engine_vectorised_filter(self) -> None:
        """lda.py re-expresses the rule vectorised (str.startswith |
        isin) under nouns_only=True; the two forms must select the same
        rows."""

        from core.analysis.lda import tokens_from_frame

        frame = pd.DataFrame(
            {
                "Document": ["a"] * 6,
                "Lemma": ["cat", "dog", "run", "1934", "", "quickly"],
                "POS": ["NN", "NNS", "VB", "CD", "NN", None],
            }
        )
        tokens = tokens_from_frame(frame, nouns_only=True, remove_stopwords=False)
        assert tokens == {"a": ["cat", "dog"]}
        # The same rows through the helper: exactly the POS values the
        # vectorised filter kept.
        for tag in frame["POS"].dropna():
            vectorised_kept = tag in ("NN", "NNS")
            assert is_noun_tag(tag) is vectorised_kept, tag
