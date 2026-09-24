"""Generate dependency notices from the isolated packaging environment."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path


def write_notices(destination: Path, license_path: Path | None = None) -> None:
    """Dependency notices, led by NLP Suite's own licence.

    The MIT terms require the copyright notice to travel with every copy of the
    software, so the installer has to carry it, not just the repository. This
    file is what the desktop serves at ``/api/notices``, which makes it the one
    place a recipient can read both our licence and every dependency's.
    """
    if license_path is None:
        license_path = Path(__file__).resolve().parents[1] / "LICENSE"
    sections = [
        "NLP Suite dependency notices\n\nGenerated from the packaging environment. Includes build dependencies as well as runtime dependencies. Third-party code retains its own license.\n"
    ]
    if license_path.is_file():
        sections.append(
            f"\n{'=' * 72}\nNLP Suite (this application)\n\n" + license_path.read_text(encoding="utf-8").rstrip() + "\n"
        )
    else:
        sections.append("\nThis inventory does not assign a license to NLP Suite's original code.\n")
    for distribution in sorted(importlib.metadata.distributions(), key=lambda d: d.metadata.get("Name", "")):
        name = distribution.metadata.get("Name", "unknown")
        sections.append(f"\n{'=' * 72}\n{name} {distribution.version}\n")
        sections.append(
            "License metadata: "
            + str(
                distribution.metadata.get("License-Expression")
                or distribution.metadata.get("License")
                or "See package notices"
            )
            + "\n"
        )
        for file in distribution.files or ():
            if any(part.endswith((".dist-info", ".egg-info")) for part in file.parts) and any(
                word in file.name.lower() for word in ("license", "licence", "notice", "copying")
            ):
                path = distribution.locate_file(file)
                if Path(path).is_file():
                    sections.append(f"\n--- {file} ---\n{Path(path).read_text(encoding='utf-8', errors='replace')}\n")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(sections), encoding="utf-8")
