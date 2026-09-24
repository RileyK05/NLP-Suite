# Usability acceptance protocol (FR-8.7 — target-user test still owed)

## Scenario

A target user (social-science researcher, no NLP Suite history) goes from
a fresh checkout to a finished analysis result using only the app and the
printed fixes — no shell beyond `pip install` and `streamlit run`, no
undocumented knowledge.

## Script

1. Install: `pip install ".[app,spacy]"`, `python -m spacy download en_core_web_sm`.
2. Open Setup: every row green except corpus (points at the missing dir).
   The corpus row names the fix.
3. Open Corpus with the sample corpus: files + OK counts shown, no issues.
4. Open Tools: run `readability` on the sample corpus; success message
   names the run directory.
5. Submit the same run as a background job (`nlp-suite jobs submit` is
   shown on the Tools page note); job finishes DONE.
6. Open Home: the run appears at the top; table renders; envelope shows
   inputs, params, diagnostics; download works.

## Pass criteria

- No step needs the terminal except the two install commands and
  `streamlit run app/Home.py`.
- Every failure met names its fix on the page (no dead ends).
- The same run is reachable from Tools (run), Home (gallery), and the
  output root (files + envelope).

## Self-run evidence (2026-09-05, operator, headless)

Steps 2–6 executed against this tree with `tests/fixtures/mini-corpus`
(Home/Tools/Corpus/Setup all HTTP 200, no log errors; readability batch
via executor; jobs submit/status green in `tests/test_jobs.py`; gallery
renders tables/downloads/provenance in `tests/test_viewer.py`). This is
plumbing evidence, not acceptance: a real target-user session is still
owed before FR-8.7 moves to `review`.
