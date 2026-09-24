"""A portable filename contract shared by imports and project archives."""

from pathlib import Path, PurePosixPath
import re
import unicodedata

MAX_COMPONENT_BYTES = 240
MAX_DOCUMENT_BYTES = 190
FIRST_PRINTABLE = 32
MIN_FILENAME_CHARACTERS = 16


def storage_filename(name: str, directory: Path) -> str:
    # Reserve a separator, collision prefix and conversion suffix below MAX_PATH.
    budget = 240 - len(str(directory.resolve())) - 15
    if budget < MIN_FILENAME_CHARACTERS:
        raise ValueError("Workspace path is too long. Move the workspace to a shorter location.")
    return document_filename(name, max_characters=budget)


_DEVICES = {
    "con",
    "prn",
    "aux",
    "nul",
    "conin$",
    "conout$",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
    "com¹",
    "com²",
    "com³",
    "lpt¹",
    "lpt²",
    "lpt³",
}


def portable_key(name: str) -> str:
    return unicodedata.normalize("NFC", name).casefold()


def safe_relative(name: str) -> str:
    if not isinstance(name, str) or not name or name == ".":
        raise ValueError("Unsafe archive member")
    path = PurePosixPath(name)
    if path.is_absolute() or path.as_posix() != name or ".." in path.parts:
        raise ValueError("Unsafe archive member")
    for part in path.parts:
        if (
            part[-1:] in (".", " ")
            or len(part.encode("utf-8")) > MAX_COMPONENT_BYTES
            or any(ord(char) < FIRST_PRINTABLE or char in '\\:*?"<>|' for char in part)
            or portable_key(part.split(".", 1)[0]) in _DEVICES
        ):
            raise ValueError("Filename is not portable across Windows, macOS and Linux")
    return name


def document_filename(name: str, *, max_characters: int = MAX_DOCUMENT_BYTES) -> str:
    name = unicodedata.normalize("NFC", re.sub(r"[^\w. -]", "_", name)).strip(" .") or "document.txt"
    if portable_key(name.split(".", 1)[0]) in _DEVICES:
        name = "document_" + name
    # Reserve room for a collision prefix and conversion suffix.
    while len(name.encode("utf-8")) > MAX_DOCUMENT_BYTES or len(name) > max_characters:
        stem, dot, suffix = name.rpartition(".")
        name = stem[:-1] + dot + suffix if dot and len(stem) > 1 else name[:-1]
    return name
