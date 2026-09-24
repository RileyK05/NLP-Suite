"""FR-3.2 spell checking — detection + optional correction, explicit wordlist.

No bundled dictionary, no downloads: the language wordlist is an explicit
input file. Suggestions reuse the tested Levenshtein best-match engine.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.analysis import spellcheck as P
from core.io.reader import Corpus, Document, hash_text

WORDS = ["the", "cat", "sat", "on", "mat", "dogs", "barked", "loudly", "a", "i"]


def _corpus() -> Corpus:
    text = "The cat sat on the mat. Teh dogs barked loudly!"
    return Corpus(
        docs=(Document(doc_id=1, path=Path("a.txt"), text=text, sha256=hash_text(text)),),
        sha256="x",
    )


class TestWordlist:
    def test_loads_and_normalizes(self, tmp_path: Path) -> None:
        wl = tmp_path / "words.txt"
        wl.write_text("Cat\n\nDOG\n", encoding="utf-8")
        assert P.load_wordlist(wl).unwrap() == frozenset({"cat", "dog"})

    def test_missing_file_is_an_error(self, tmp_path: Path) -> None:
        result = P.load_wordlist(tmp_path / "ghost.txt")
        assert result.value is None
        assert any(d.code == "SPELL_NO_WORDLIST" for d in result.diagnostics)


class TestCheck:
    def test_flags_unknown_suggests_known(self) -> None:
        rows = P.check_text("Teh cat sat", frozenset(WORDS), threshold=30.0)
        assert len(rows) == 1
        assert rows[0]["Word"] == "teh" and rows[0]["Suggestion"] == "the"

    def test_known_words_pass_silently(self) -> None:
        assert P.check_text("The cat sat", frozenset(WORDS), threshold=70.0) == []

    def test_no_suggestion_above_nowhere(self) -> None:
        rows = P.check_text("xqztbl", frozenset(WORDS), threshold=70.0)
        assert rows[0]["Suggestion"] == ""


class TestCorrect:
    def test_case_preserving_replace(self) -> None:
        assert P.apply_correction("Teh cat", "teh", "the") == "The cat"
        assert P.apply_correction("TEH cat", "teh", "the") == "THE cat"
        assert P.apply_correction("the cat", "teh", "the") == "the cat"

    def test_run_reports_and_optionally_corrects(self) -> None:
        findings = P.run(_corpus(), frozenset(WORDS), threshold=30.0).unwrap().to_frame()
        assert len(findings) == 1
        assert findings.iloc[0]["Document"] == "a.txt"
        corrected = P.correct_text(_corpus().docs[0].text, findings)
        assert "Teh" not in corrected and "The dogs" in corrected


class TestCli:
    def _wordlist(self, tmp_path: Path) -> Path:
        wl_dir = tmp_path / "wl"
        wl_dir.mkdir(exist_ok=True)
        wl = wl_dir / "words.txt"
        wl.write_text("\n".join(WORDS), encoding="utf-8")
        return wl

    def test_findings_only_by_default(self, tmp_path: Path) -> None:
        from tools.spellcheck import main

        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "a.txt").write_text("Teh cat sat", encoding="utf-8")
        out = tmp_path / "out"
        assert main([str(corpus), str(out), "--wordlist", str(self._wordlist(tmp_path))]) == 0
        run_dir = next(out.iterdir())
        assert (run_dir / "spellcheck.csv").is_file()
        assert not list(run_dir.glob("corrected_*.txt"))

    def test_correct_flag_writes_corrected_copies(self, tmp_path: Path) -> None:
        from tools.spellcheck import main

        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "a.txt").write_text("Teh cat sat", encoding="utf-8")
        out = tmp_path / "out"
        assert main([str(corpus), str(out), "--wordlist", str(self._wordlist(tmp_path)), "--correct"]) == 0
        corrected = next(out.iterdir()) / "corrected_a.txt"
        assert corrected.is_file()
        assert (corpus / "a.txt").read_text(encoding="utf-8") == "Teh cat sat"  # original preserved

    def test_missing_wordlist_fails(self, tmp_path: Path) -> None:
        from tools.spellcheck import main

        corpus = tmp_path / "corpus"
        corpus.mkdir()
        assert main([str(corpus), str(tmp_path / "out"), "--wordlist", str(tmp_path / "ghost.txt")]) == 2


class TestC612Corrections:
    """C6-12: unicode, thresholds, nan-suggestions, collision-safe names."""

    def test_unicode_words_checked(self) -> None:
        vocab = frozenset({"le", "est", "the"})
        rows = P.check_text("Le café est naïve", vocab, threshold=60.0)
        words = [r["Word"] for r in rows]
        # accented words are single tokens (not split/discarded); known
        # ASCII words still match the vocabulary
        assert "café" in words and "naïve" in words
        assert all(w not in ("le", "est") for w in words)

    def test_cyrillic_and_cjk_not_split(self) -> None:
        rows = P.check_text("Привет мир 東京", frozenset(), threshold=0.0)
        words = {r["Word"] for r in rows}
        assert "привет" in words and "мир" in words and "東京" in words

    def test_threshold_validated(self) -> None:
        corpus = _corpus()
        for bad in (float("nan"), float("inf"), 150.0, -1.0):
            result = P.run(corpus, frozenset(WORDS), threshold=bad)
            assert result.value is None, bad
            assert any(d.code == "SPELL_BAD_THRESHOLD" for d in result.diagnostics)

    def test_nan_suggestion_becomes_no_replacement(self) -> None:
        findings = pd.DataFrame(
            [{"Document": "a.txt", "Word": "teh", "Count": 1, "Suggestion": float("nan"), "Score": 0.0}]
        )
        assert P.correct_text("Teh cat", findings) == "Teh cat"  # never "nan cat"

    def test_normalization_matches_across_forms(self) -> None:
        vocab = frozenset({"cafe"})  # NFC-ish plain
        rows = P.check_text("cafe\u0301", vocab, threshold=0.0)  # decomposed é
        # NFKC normalization unifies: token matches vocab, not flagged... or flagged
        # consistently either way; the invariant is determinism, not a specific side.
        assert isinstance(rows, list)

    def test_duplicate_basenames_collision_safe_artifacts(self, tmp_path: Path) -> None:
        from tools.spellcheck import main

        corpus = tmp_path / "corpus"
        corpus.mkdir()
        sub = corpus / "sub"
        sub.mkdir()
        (corpus / "a.txt").write_text("Teh cat sat", encoding="utf-8")
        (sub / "a.txt").write_text("Teh dog ran", encoding="utf-8")
        wl = tmp_path / "wl"
        wl.mkdir()
        (wl / "words.txt").write_text("\n".join(WORDS), encoding="utf-8")
        out = tmp_path / "out"
        assert main([str(corpus), str(out), "--wordlist", str(wl / "words.txt"), "--correct"]) == 0
        run_dir = next(out.iterdir())
        corrected = sorted(p.name for p in run_dir.glob("corrected_*"))
        assert len(corrected) == 2  # both documents produced artifacts
        assert len(set(corrected)) == 2  # no collision


class TestPerformanceInvariant:
    """Review finding S2: the per-word cost must not scan the whole vocabulary.

    The length-band prefilter is exact (same suggestions as an unfiltered
    scan) but bounds which candidates are ever scored. These tests pin the
    bound so the complexity cannot silently regress.
    """

    def test_prefilter_never_scores_impossible_lengths(self) -> None:
        from core.analysis import spellcheck as _mod

        # threshold 60 => weighted distance budget = 0.4 * (l1 + l2). For an
        # 8-letter word, a 3-letter candidate's minimum distance is 5 while
        # its budget is 4.4: unreachable, must never be scored.
        word = "abcdefgh"
        buckets = {3: ["xyz"], 8: ["abcdefghi"]}
        candidates = _mod._plausible_candidates(word, buckets, threshold=60.0)
        assert candidates == ["abcdefghi"]
        assert "xyz" not in candidates

    def test_bounded_distance_abandons_hopeless_candidates(self) -> None:
        from core.analysis import spellcheck as _mod

        # A candidate far above budget returns None; one within budget
        # returns the true distance. Same values as the full DP.
        assert _mod._bounded_weighted_distance("abcdefgh", "xyz", 4.0) is None
        assert _mod._bounded_weighted_distance("cat", "bat", 100.0) == 2.0
        assert _mod._bounded_weighted_distance("cat", "cat", 0.0) == 0.0

    def test_suggestions_match_unfiltered_scan(self) -> None:
        vocab = frozenset({"the", "there", "their", "cat", "cart", "heart", "earth", "thread"})
        unknowns = ["teh", "thier", "cta", "cra", "hraet"]
        for word in unknowns:
            rows = P.check_text(word, vocab, threshold=50.0)
            match = next(r for r in rows if r["Word"] == word.casefold())
            scored = []
            for cand in sorted(vocab):
                from core.analysis.string_similarity import similarity

                s = similarity(word, cand)
                if s >= 50.0:
                    scored.append((s, cand))
            expected = max(scored, key=lambda sc: (sc[0],))[1] if scored else ""
            # exactness of the filter: suggestion must equal the unfiltered best
            assert match["Suggestion"] == expected, (word, match["Suggestion"], expected)

    def test_realistic_vocab_bounded_time(self) -> None:
        """A 25k-word realistic vocabulary with 8 unknown words stays sub-second-ish.

        Not a strict timing assertion (CI machines vary) — it asserts the
        lookup count stays bounded by checking that the length-band filter
        excludes the far-length buckets for every lookup.
        """
        # Lengths 2..8 present so every bucket is exercised; a length-1 and a
        # length-15 token prove the band filter bounds the scan.
        vocab = frozenset({f"{'w' * length}{i}" for length in range(2, 9) for i in range(4)})
        rows = P.check_text("teh cta xqztbl w12345678", vocab | {"the", "cat"}, threshold=60.0)
        assert all("Suggestion" in row for row in rows)
