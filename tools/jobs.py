"""jobs — thin CLI for background jobs (FR-8.4).

``submit`` starts any tool CLI in the background and prints the job id;
``status`` reads one job; ``list`` shows every job newest-first. Logs
live next to ``status.json`` in the job directory.
"""

from __future__ import annotations

import argparse
import sys

from core.jobs import DEFAULT_JOBS_ROOT, list_jobs, read_status, submit

__all__ = ["main"]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Background jobs for tool CLIs")
    parser.add_argument("--jobs-root", default=DEFAULT_JOBS_ROOT, help="job directory root")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("submit", help="start a tool in the background")
    run.add_argument("tool", help="tool CLI name, e.g. readability")
    run.add_argument("tool_args", nargs=argparse.REMAINDER, help="arguments for the tool")
    show = sub.add_parser("status", help="read one job's status")
    show.add_argument("job_id")
    sub.add_parser("list", help="list jobs newest-first")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "submit":
        submitted = submit(args.tool, list(args.tool_args), jobs_root=args.jobs_root)
        if not submitted.ok:
            for diag in submitted.diagnostics:
                print(diag, file=sys.stderr)
            return 1
        print(submitted.unwrap().status.id)
        return 0
    if args.command == "status":
        found = read_status(args.job_id, jobs_root=args.jobs_root)
        if found.value is None:
            for diag in found.diagnostics:
                print(diag, file=sys.stderr)
            return 1
        current = found.unwrap()
        print(f"{current.id} {current.tool} {current.state} exit={current.exit_code}")
        if current.error:
            print(current.error, file=sys.stderr)
        return 0
    listed = list_jobs(jobs_root=args.jobs_root)
    if listed.value is None:  # unreachable: list_jobs only fails on nothing
        return 1  # pragma: no cover
    for past in listed.unwrap():
        print(f"{past.id} {past.tool} {past.state} exit={past.exit_code}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
