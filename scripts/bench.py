"""bench — performance benchmark (FR-9.7).

Times intake, parsing, and a fixed set of analyses over any corpus
directory and writes a JSON record. Each analysis is attempted and its
outcome recorded (seconds or loud error code) — a missing optional
dependency is a measurement fact, not a crash. This script asserts no
budget; budgets live in ``docs/PERFORMANCE.md`` next to recorded runs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

__all__ = ["main"]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Time intake + parse + analyses over a corpus")
    parser.add_argument("corpus", type=Path, help="corpus directory of .txt files")
    parser.add_argument("--out", type=Path, default=None, help="write the JSON record here (default: stdout)")
    parser.add_argument("--parser", choices=["spacy", "stanza"], default="spacy")
    parser.add_argument("--language", default="en")
    parser.add_argument(
        "--analyses",
        default="readability,lexical_diversity,sentence_complexity,coreference,narrative,lda_gensim,word2vec_gensim",
        help="comma-separated analysis names from the table below",
    )
    return parser.parse_args(argv)


def _timed(label: str, func, *args, **kwargs):  # type: ignore[no-untyped-def]
    start = time.perf_counter()
    try:
        result = func(*args, **kwargs)
    except Exception as exc:
        return label, {"seconds": round(time.perf_counter() - start, 3), "error": f"{type(exc).__name__}: {exc}"}
    seconds = round(time.perf_counter() - start, 3)
    if hasattr(result, "ok") and not result.ok:
        codes = [d.code for d in result.diagnostics]
        return label, {"seconds": seconds, "error": ";".join(codes)}
    return label, {"seconds": seconds}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    from core.io.reader import read_corpus
    from core.pipelines.cache import PipelineCache
    from tools._cli import get_pipeline, resolve_config

    stages: dict[str, float] = {}
    results: dict[str, object] = {}
    record: dict[str, object] = {
        "corpus": str(args.corpus),
        "parser": args.parser,
        "stages": stages,
        "analyses": results,
    }

    start = time.perf_counter()
    corpus_result = read_corpus(args.corpus)
    stages["read_corpus"] = round(time.perf_counter() - start, 3)
    if not corpus_result.ok:
        for diag in corpus_result.diagnostics:
            print(diag, file=sys.stderr)
        return 1
    corpus = corpus_result.unwrap()
    record["docs"] = len(corpus)

    try:
        config = resolve_config(args)
    except ValueError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    cache = PipelineCache()
    pipeline = get_pipeline(cache, config)
    start = time.perf_counter()
    table_result = pipeline.parse(corpus)
    stages["parse"] = round(time.perf_counter() - start, 3)
    if table_result.value is None:
        for diag in table_result.diagnostics:
            print(diag, file=sys.stderr)
        return 1
    table = table_result.unwrap()
    record["tokens"] = len(table)

    from core.analysis import (
        coreference,
        lexical_diversity,
        readability,
        sentence_complexity,
        topic_model,
        word_embeddings,
    )
    from core.narrative.arcs import emotion_arc

    analyses = {
        "readability": lambda: readability.run(corpus),
        "lexical_diversity": lambda: lexical_diversity.run(corpus),
        "sentence_complexity": lambda: sentence_complexity.run(table),
        "coreference": lambda: coreference.run(table),
        "narrative": lambda: emotion_arc(table),
        "topic_model": lambda: topic_model.run(table),
        "word2vec_gensim": lambda: word_embeddings.train(table),
    }
    wanted = [name.strip() for name in str(args.analyses).split(",") if name.strip()]
    for name in wanted:
        if name not in analyses:
            results[name] = {"seconds": 0.0, "error": "BENCH_UNKNOWN_ANALYSIS"}
            continue
        _, outcome = _timed(name, analyses[name])
        results[name] = outcome

    payload = json.dumps(record, indent=2)
    if args.out is not None:
        args.out.write_text(payload, encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
