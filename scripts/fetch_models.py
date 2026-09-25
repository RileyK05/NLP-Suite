"""Fetch published models from the models release into ``models/``.

The desktop build runs this before PyInstaller so the bundled models ship
inside the engine; a developer runs it to get real models without the
export environment. Same downloader as the app (``core.models.download``):
resumable, and every file checked against its SHA-256.

    python scripts/fetch_models.py --bundled          # what the installer ships
    python scripts/fetch_models.py qwen3-embedding-0.6b
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.models.download import download  # noqa: E402
from core.models.registry import MODELS, get_model  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("models", nargs="*", help="model ids (default with --bundled: every bundled model)")
    parser.add_argument("--bundled", action="store_true", help="fetch every model the installer ships")
    parser.add_argument("--out", type=Path, default=ROOT / "models")
    args = parser.parse_args(argv)
    names = list(args.models) or ([spec.id for spec in MODELS if spec.bundled] if args.bundled else [])
    if not names:
        parser.error("name models to fetch, or pass --bundled")
    failed = 0
    for name in names:
        spec = get_model(name)
        if spec is None:
            parser.error(f"unknown model {name!r}")
        last = [-1]

        def progress(done: int, total: int, spec_id: str = spec.id) -> None:
            percent = int(100 * done / total) if total else 100
            if percent >= last[0] + 10:
                last[0] = percent
                print(f"  {spec_id}: {percent}%", flush=True)

        print(f"{spec.id} ({spec.size_mb} MB)")
        result = download(spec, root=args.out, progress=progress)
        if not result.ok:
            failed += 1
            for diag in result.diagnostics:
                print(f"  {diag.code}: {diag.message}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
