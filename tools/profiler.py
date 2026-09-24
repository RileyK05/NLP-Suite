"""profiler — batch runner over a validated plan with linked envelopes."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any

from core.io.reader import read_corpus
from core.profiler.batch import BatchRequest, write_batch
from core.profiler.executor import BatchResult, execute
from core.profiler.plan import Plan, build_plan
from core.profiler.registry import get_tool, tool_names
from core.profiler.resume import find_reusable
from tools._cli import handle_result_states, build_common_parser, get_pipeline, resolve_config


def _default_selection() -> list[str]:
    return [
        name
        for name in tool_names()
        if (spec := get_tool(name)) is not None
        and spec.profiler_eligible
        and not any(param.required for param in spec.params)
    ]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = build_common_parser("Profiler (plan + batch execution with linked envelopes)")
    p.add_argument(
        "--analyses",
        default=None,
        help="comma-separated tool names (default: batch tools without required parameters)",
    )
    p.add_argument("--params", type=Path, help="JSON object mapping selected tools to their parameter objects")
    p.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="prior profiler run dir (same output root): reuse matching tool results",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        config = resolve_config(args)
    except ValueError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    corpus_result = read_corpus(args.corpus)
    # A bad document is a diagnostic, not an aborted run: the reader returns
    # the good documents with ERROR diagnostics; only a total failure (value
    # is None) stops the tool. Partial corpora flow into the four-state
    # writer policy, matching tools/_cli.load_corpus.
    if corpus_result.value is None:
        for d in corpus_result.diagnostics:
            print(d, file=sys.stderr)
        return 1
    if corpus_result.diagnostics:
        for d in corpus_result.diagnostics:
            print(d, file=sys.stderr)
    corpus = corpus_result.unwrap()

    if args.analyses:
        selection = [name.strip() for name in str(args.analyses).split(",") if name.strip()]
    else:
        selection = _default_selection()
    params: dict[str, Any] = {}
    if args.params is not None:
        try:
            params = json.loads(args.params.read_text(encoding="utf-8"))
            if not isinstance(params, dict) or not all(isinstance(value, dict) for value in params.values()):
                raise ValueError("params must map tool names to parameter objects")
        except (OSError, ValueError) as exc:
            print(f"params error: {exc}", file=sys.stderr)
            return 2
    planned = build_plan(selection, params)
    if planned.value is None:
        for d in planned.diagnostics:
            print(d, file=sys.stderr)
        return 2
    plan = planned.unwrap()

    reuse: dict[str, Any] = {}
    if args.resume_from is not None:
        reuse = find_reusable(
            args.resume_from, [t.name for t in plan.tools], plan.params, corpus, output_root=args.output
        )
    pending = Plan(
        tools=tuple(tool for tool in plan.tools if tool.name not in reuse),
        params={name: value for name, value in plan.params.items() if name not in reuse},
    )

    table = None
    parse_state = "ok"
    parse_diagnostics = ()
    if pending.needs_parse:
        from core.pipelines.cache import PipelineCache

        cache = PipelineCache()
        pipeline = get_pipeline(cache, config, args)
        table_result = pipeline.parse(corpus)
        parse_state = handle_result_states(table_result)
        parse_diagnostics = table_result.diagnostics
        table = table_result.unwrap()

    batch = execute(pending, corpus=corpus, table=table, parse_diagnostics=parse_diagnostics)
    batch = BatchResult(
        outcomes=tuple(
            replace(
                outcome,
                diagnostics=(
                    *corpus_result.diagnostics,
                    *outcome.diagnostics,
                ),
            )
            for outcome in batch.outcomes
        )
    )
    report = write_batch(
        args.output,
        BatchRequest(
            selection=tuple(t.name for t in plan.tools),
            params=plan.params,
            batch=batch,
            corpus=corpus,
            reuse=reuse,
        ),
    )
    if report.value is None:
        for d in report.diagnostics:
            print(d, file=sys.stderr)
        return 1
    info = report.unwrap()
    failed = [o.name for o in batch.outcomes if not o.ok]
    print(f"Wrote {len(batch.outcomes)} tool outcome(s) to {info.run_dir} ({len(failed)} failed)")
    return 1 if failed or parse_state == "partial" else 0


if __name__ == "__main__":
    raise SystemExit(main())
