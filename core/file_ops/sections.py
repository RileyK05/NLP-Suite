"""A long text's own sections: chapters, and the parts and volumes above them (docs/PLAN_0.5.0.md 2.3).

A novel imported whole is one document, and every per-document figure then
has one point. Cut at its chapters, each chapter is a document on an order
axis and every tool works per chapter. This module finds the cuts; it never
writes anything. :func:`detect_sections` returns a plan -- where each section
starts and ends in the original text, what it is called, what was left out
-- for a person to confirm before the desktop stores it.

Three rules, tried in this order by ``auto``:

1. ``headings``: whole lines such as ``CHAPTER I.``, ``Chapter 3: The Ball``,
   ``BOOK II`` (a parent of the chapters after it), Markdown ``#`` headings,
   or -- when there are enough of them, far enough apart -- lines that are
   only a roman numeral or a number.
2. ``blank-gap``: three or more blank lines, or a form feed.
3. ``words``: blocks of N words. It always works, and it says so in every
   title ("Words 1-2,000") and in a warning, so nobody reads block 7 as
   the author's chapter 7.

The sections tile the kept text exactly: joined, they are the original from
the first section's start to the end of the body, with nothing lost or
repeated. What falls outside -- a Project Gutenberg licence, a preface before
the first chapter, a table of contents -- is reported with its size.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise
import re
from statistics import median

from core.result import Diagnostic, Result

__all__ = [
    "LEVELS",
    "PARENT_LEVELS",
    "RULES",
    "Section",
    "SectionPlan",
    "detect_sections",
    "front_and_back_matter",
    "roman_value",
]

RULES = ("auto", "headings", "blank-gap", "words", "pattern")
#: Heading words that name a section of their own.
LEVELS = ("chapter", "scene", "section")
#: Heading words that group the sections after them.
PARENT_LEVELS = ("volume", "book", "part", "act")

#: ``auto`` keeps a rule when its sections look like an author's divisions.
MIN_SECTIONS = 3
MIN_MEDIAN_WORDS = 300
MAX_SHARE = 0.6
#: Per-section warnings in the preview.
SHORT_WORDS = 300
LONG_WORDS = 20_000
#: Roman-only or number-only heading lines are believed only in numbers.
MIN_BARE_HEADINGS = 5
#: Headings closer than this many words are a table of contents, not chapters.
TOC_GAP_WORDS = 20
TOC_MIN_RUN = 3
#: A heading line longer than this is prose that happens to start with "Chapter".
MAX_HEADING_CHARS = 80
#: A title standing alone on the line after its heading is at most this long.
MAX_TITLE_CHARS = 60
MAX_TITLE_WORDS = 10
DEFAULT_BLOCK_WORDS = 2000

_ROMAN = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
_NUMBER_WORDS = {
    word: index
    for index, words in enumerate(
        (
            ("one", "first"),
            ("two", "second"),
            ("three", "third"),
            ("four", "fourth"),
            ("five", "fifth"),
            ("six", "sixth"),
            ("seven", "seventh"),
            ("eight", "eighth"),
            ("nine", "ninth"),
            ("ten", "tenth"),
            ("eleven", "eleventh"),
            ("twelve", "twelfth"),
            ("thirteen", "thirteenth"),
            ("fourteen", "fourteenth"),
            ("fifteen", "fifteenth"),
            ("sixteen", "sixteenth"),
            ("seventeen", "seventeenth"),
            ("eighteen", "eighteenth"),
            ("nineteen", "nineteenth"),
            ("twenty", "twentieth"),
        ),
        start=1,
    )
    for word in words
}
#: En and em dash, which separate a heading's number from its title.
_DASHES = chr(0x2013) + chr(0x2014)
_NUMBER = r"(?:\d{1,4}|[ivxlcdm]{1,8}|(?:the\s+)?[a-z]{3,12})"
# A heading: the level word, its number, then optionally a title after
# punctuation or a wide gap. Trailing ".", "]" and ")" are allowed because
# illustrated editions print "Chapter I.]" at the end of a picture caption.
_HEADING = re.compile(
    rf"^\s*(?P<level>{'|'.join(LEVELS + PARENT_LEVELS)})\s+(?P<number>{_NUMBER})"
    rf"(?:\s*[.:\-{_DASHES}]\s*|\s{{2,}})?(?P<title>[^\n]*?)[\s.\])]*$",
    re.IGNORECASE,
)
_BARE = re.compile(r"^\s*(?P<number>\d{1,3}|[IVXLC]{1,7})\s*\.?\s*$")
_MARKDOWN = re.compile(r"^\s{0,3}(?P<hashes>#{1,2})\s+(?P<title>\S.*?)\s*#*\s*$")
_GUTENBERG_START = re.compile(r"^\*{3}\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*$", re.IGNORECASE | re.MULTILINE)
_GUTENBERG_END = re.compile(r"^\*{3}\s*END OF (?:THE|THIS) PROJECT GUTENBERG.*$", re.IGNORECASE | re.MULTILINE)
_GAP = re.compile(r"(?:\r?\n[ \t]*){4,}|\f")


@dataclass(frozen=True, slots=True)
class Section:
    """One section: where it is in the original text, and what it is called."""

    #: 1-based place in the book.
    order: int
    #: What a reader is shown: "Chapter I: Down the Rabbit-Hole", "Words 1-2,000".
    title: str
    #: "chapter", "scene", "section", or a parent level when a book has no chapters.
    level: str
    #: Character offsets in the source text: ``text[start:end]`` is the section.
    start: int
    end: int
    #: The groups it sits in, outermost first: (("Volume", "II"),).
    parents: tuple[tuple[str, str], ...] = ()
    #: The number its heading gave, when it gave one (roman numerals read).
    number: int | None = None
    #: The heading's own title ("Down the Rabbit-Hole"), when it had one.
    name: str = ""
    words: int = 0


@dataclass(frozen=True, slots=True)
class SectionPlan:
    """What a split would do, for the preview."""

    sections: tuple[Section, ...]
    rule: str
    #: The body the rules read: after any licence header, before any footer.
    body_start: int
    body_end: int
    #: What was left out and how big it was: (("Project Gutenberg header", 312), ...), in words.
    left_out: tuple[tuple[str, int], ...] = field(default=())


# ---------------------------------------------------------------- helpers --


def roman_value(text: str) -> int | None:
    """ "XIV" -> 14; None for anything that is not a well-formed roman numeral."""
    letters = text.strip().lower()
    if not letters or any(letter not in _ROMAN for letter in letters):
        return None
    total = 0
    for index, letter in enumerate(letters):
        value = _ROMAN[letter]
        following = _ROMAN[letters[index + 1]] if index + 1 < len(letters) else 0
        total += -value if value < following else value
    return total if total > 0 and _to_roman(total) == letters.upper() else None


def _to_roman(value: int) -> str:
    out = ""
    for amount, symbol in (
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ):
        while value >= amount:
            out += symbol
            value -= amount
    return out


def _number_of(text: str) -> int | None:
    """A heading's number as written: "12", "XII", "twelve", "the Twelfth"."""
    word = re.sub(r"^the\s+", "", text.strip(), flags=re.IGNORECASE).lower()
    if word.isdigit():
        return int(word)
    return roman_value(word) if roman_value(word) is not None else _NUMBER_WORDS.get(word)


def _words(text: str) -> int:
    return len(text.split())


def _lines(text: str, start: int, end: int) -> list[tuple[int, int, str]]:
    """(line start, line end, line) for each line of ``text[start:end]``."""
    found = []
    at = start
    for line in text[start:end].splitlines(keepends=True):
        found.append((at, at + len(line), line.rstrip("\r\n")))
        at += len(line)
    return found


def front_and_back_matter(text: str) -> tuple[int, int]:
    """The body's bounds: after a Project Gutenberg header line, before its footer line.

    ``(0, len(text))`` when the text has neither. The licence is not the
    author's; left in, its 3,000 words would be the last "chapter".
    """
    start_match = _GUTENBERG_START.search(text)
    end_match = _GUTENBERG_END.search(text, start_match.end() if start_match else 0)
    start = start_match.end() if start_match else 0
    if start_match and start < len(text) and text[start] == "\r":
        start += 1
    if start_match and start < len(text) and text[start] == "\n":
        start += 1
    return start, end_match.start() if end_match else len(text)


# --------------------------------------------------------------- headings --


@dataclass(frozen=True, slots=True)
class _Heading:
    start: int
    end: int
    level: str
    written: str
    number: int | None
    name: str


def _heading_lines(text: str, start: int, end: int, pattern: re.Pattern[str] | None) -> list[_Heading]:
    """Every whole line in the body that reads as a heading."""
    found: list[_Heading] = []
    for line_start, line_end, line in _lines(text, start, end):
        if not line.strip() or len(line.strip()) > MAX_HEADING_CHARS:
            continue
        if pattern is not None:
            if pattern.match(line.strip()):
                found.append(_Heading(line_start, line_end, "section", line.strip(), None, ""))
            continue
        match = _HEADING.match(line)
        if match is not None:
            number = _number_of(match.group("number"))
            if number is None:
                continue
            found.append(
                _Heading(
                    line_start,
                    line_end,
                    match.group("level").lower(),
                    match.group("number").strip().rstrip("."),
                    number,
                    match.group("title").strip(" .:-" + _DASHES),
                )
            )
    return found


def _bare_lines(text: str, start: int, end: int) -> list[_Heading]:
    """Lines that are only "IV" or "12": chapter marks in many older novels."""
    found = []
    for line_start, line_end, line in _lines(text, start, end):
        match = _BARE.match(line)
        if match is None:
            continue
        written = match.group("number")
        number = int(written) if written.isdigit() else roman_value(written)
        if number is not None:
            found.append(_Heading(line_start, line_end, "chapter", written, number, ""))
    return found


def _markdown_lines(text: str, start: int, end: int) -> list[_Heading]:
    found = []
    for line_start, line_end, line in _lines(text, start, end):
        match = _MARKDOWN.match(line)
        if match is not None:
            level = "part" if len(match.group("hashes")) == 1 else "chapter"
            found.append(_Heading(line_start, line_end, level, "", None, match.group("title")))
    return found


def _drop_contents(text: str, headings: list[_Heading]) -> tuple[list[_Heading], int]:
    """Leave out a table of contents: a run of headings with (almost) no words between them.

    Left in, "Chapter I" would be matched twice and the first match would be
    a two-word section. Returns the headings kept and how many lines were
    dropped as contents.
    """
    dropped: set[int] = set()
    run: list[int] = []
    for index in range(len(headings)):
        run.append(index)
        after = headings[index + 1].start if index + 1 < len(headings) else None
        if after is not None and _words(text[headings[index].end : after]) < TOC_GAP_WORDS:
            continue
        # The run's last heading is followed by prose, so it is a real one
        # (the book's first chapter right after its contents); the lines
        # before it are the contents.
        listed = run[:-1]
        if len(listed) >= TOC_MIN_RUN:
            dropped.update(listed)
            # Contents followed by a preface: the last contents line is then
            # followed by prose too. It is contents when a later heading
            # repeats it (chapter numbers may restart per volume, so only
            # this one line is checked).
            last = headings[run[-1]]
            if any((h.level, h.number) == (last.level, last.number) for h in headings[run[-1] + 1 :]):
                dropped.add(run[-1])
        run = []
    kept = [heading for index, heading in enumerate(headings) if index not in dropped]
    return kept, len(dropped)


def _title_below(text: str, heading: _Heading, limit: int) -> str:
    """A title standing alone on a line just under its heading ("Down the Rabbit-Hole").

    Picture captions ("[Illustration]") are skipped. A line of prose is never
    taken: a title is short and stands alone, with a blank line after it.
    """
    lines = _lines(text, heading.end, min(limit, heading.end + 2000))
    for index, (_start, _end, line) in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("["):
            continue
        alone = index + 1 >= len(lines) or not lines[index + 1][2].strip()
        short = len(stripped) <= MAX_TITLE_CHARS and len(stripped.split()) <= MAX_TITLE_WORDS
        if alone and short and not _HEADING.match(stripped):
            return stripped
        return ""
    return ""


def _from_headings(text: str, headings: list[_Heading], body_end: int, rule: str) -> list[Section]:
    """Sections from heading lines: parents group the chapters after them."""
    has_sections = any(heading.level in LEVELS for heading in headings)
    cut_at = [heading for heading in headings if heading.level in LEVELS or not has_sections]
    if not cut_at:
        return []
    parents: dict[str, str] = {}
    parent_of: dict[int, tuple[tuple[str, str], ...]] = {}
    # A parent heading's own lines ("VOLUME II") open the chapter after it.
    opens_at: dict[int, int] = {}
    pending_start: int | None = None
    for heading in headings:
        if heading in cut_at:
            parent_of[heading.start] = tuple(
                (level.capitalize(), parents[level]) for level in PARENT_LEVELS if level in parents
            )
            opens_at[heading.start] = pending_start if pending_start is not None else heading.start
            pending_start = None
        else:
            outer = PARENT_LEVELS.index(heading.level)
            for level in PARENT_LEVELS[outer + 1 :]:
                parents.pop(level, None)
            parents[heading.level] = heading.written.upper() if roman_value(heading.written) else heading.written
            if pending_start is None:
                pending_start = heading.start
    sections: list[Section] = []
    starts = [opens_at[heading.start] for heading in cut_at]
    for order, heading in enumerate(cut_at, start=1):
        start = starts[order - 1]
        end = starts[order] if order < len(cut_at) else body_end
        name = heading.name or (_title_below(text, heading, end) if rule != "pattern" else "")
        label = f"{heading.level.capitalize()} {heading.written}".strip() if heading.written else ""
        title = ": ".join(part for part in (label, name) if part) or f"Section {order}"
        sections.append(
            Section(
                order=order,
                title=title,
                level=heading.level,
                start=start,
                end=end,
                parents=parent_of.get(heading.start, ()),
                number=heading.number,
                name=name,
                words=_words(text[start:end]),
            )
        )
    return sections


# ------------------------------------------------------------ other rules --


def _from_gaps(text: str, start: int, end: int) -> list[Section]:
    cuts = [start] + [match.end() for match in _GAP.finditer(text, start, end)]
    bounds = [(a, b) for a, b in zip(cuts, [*cuts[1:], end], strict=True) if text[a:b].strip()]
    # Blank-only stretches between gaps join the section before them, so the
    # sections still tile the body.
    starts = [a for a, _b in bounds]
    tiled = [(a, starts[i + 1] if i + 1 < len(starts) else end) for i, a in enumerate(starts)]
    return [
        Section(order=i, title=f"Section {i}", level="section", start=a, end=b, words=_words(text[a:b]))
        for i, (a, b) in enumerate(tiled, start=1)
    ]


def _from_words(text: str, start: int, end: int, block: int) -> list[Section]:
    """Blocks of *block* words, cut between words so nothing is lost."""
    spans = [match.start() for match in re.finditer(r"\S+", text[start:end])]
    sections: list[Section] = []
    for index, first in enumerate(range(0, len(spans), block), start=1):
        a = start if index == 1 else start + spans[first]
        nxt = first + block
        b = start + spans[nxt] if nxt < len(spans) else end
        last = min(first + block, len(spans))
        sections.append(
            Section(
                order=index,
                title=f"Words {first + 1:,}{chr(0x2013)}{last:,}",
                level="section",
                start=a,
                end=b,
                words=last - first,
            )
        )
    return sections


# ------------------------------------------------------------------ plan --


def _plausible(sections: Sequence[Section], body_words: int) -> bool:
    if len(sections) < MIN_SECTIONS:
        return False
    counts = [section.words for section in sections]
    return median(counts) >= MIN_MEDIAN_WORDS and max(counts) <= MAX_SHARE * max(body_words, 1)


def _heading_candidates(text: str, start: int, end: int) -> tuple[list[_Heading], int]:
    """The heading lines of the first kind that exists: named, then bare numbers, then Markdown."""
    named, contents = _drop_contents(text, _heading_lines(text, start, end, None))
    if named:
        return named, contents
    bare = _bare_lines(text, start, end)
    far_apart = all(_words(text[a.end : b.start]) >= MIN_MEDIAN_WORDS for a, b in pairwise(bare))
    if len(bare) >= MIN_BARE_HEADINGS and far_apart:
        return bare, 0
    return _drop_contents(text, _markdown_lines(text, start, end))


def detect_sections(
    text: str,
    *,
    rule: str = "auto",
    pattern: str = "",
    regex: bool = False,
    block_words: int = DEFAULT_BLOCK_WORDS,
    keep_front_matter: bool = False,
) -> Result[SectionPlan]:
    """Where to cut *text* into its sections, for a person to confirm.

    *rule* is one of :data:`RULES`. ``pattern`` names the heading lines when
    the rule is ``pattern``: a line starting with it (any case), or matching
    it as a regular expression when *regex*. *keep_front_matter* keeps a
    Project Gutenberg header and footer in the body.
    """
    if rule not in RULES:
        return Result.failure(Diagnostic.error("SECTIONS_BAD_RULE", f"rule must be one of {RULES}, got {rule!r}"))
    if not text.strip():
        return Result.failure(Diagnostic.error("SECTIONS_EMPTY", "The document is empty."))
    if block_words < 1:
        return Result.failure(Diagnostic.error("SECTIONS_BAD_BLOCK", "A block needs at least one word."))
    compiled: re.Pattern[str] | None = None
    if rule == "pattern":
        if not pattern.strip():
            return Result.failure(Diagnostic.error("SECTIONS_NO_PATTERN", "Say what a heading line starts with."))
        try:
            compiled = re.compile(pattern if regex else re.escape(pattern.strip()), re.IGNORECASE)
        except re.error as exc:
            return Result.failure(Diagnostic.error("SECTIONS_BAD_REGEX", f"That regular expression is invalid: {exc}"))

    body_start, body_end = (0, len(text)) if keep_front_matter else front_and_back_matter(text)
    left_out: list[tuple[str, int]] = []
    if body_start > 0:
        left_out.append(("Project Gutenberg header", _words(text[:body_start])))
    if body_end < len(text):
        left_out.append(("Project Gutenberg licence at the end", _words(text[body_end:])))
    body_words = _words(text[body_start:body_end])
    notes: list[Diagnostic] = []

    sections: list[Section] = []
    used = rule
    contents = 0
    if rule in ("auto", "headings", "pattern"):
        if compiled is not None:
            headings = _heading_lines(text, body_start, body_end, compiled)
        else:
            headings, contents = _heading_candidates(text, body_start, body_end)
        sections = _from_headings(text, headings, body_end, rule)
        used = "pattern" if rule == "pattern" else "headings"
        if rule == "auto" and not _plausible(sections, body_words):
            sections = []
        if rule != "auto" and not sections:
            notes.append(
                Diagnostic.warning(
                    "SECTIONS_NO_HEADINGS",
                    "No line reads as a heading"
                    + (f" starting {pattern!r}" if rule == "pattern" else "")
                    + "; the text is cut into blocks of words instead.",
                )
            )
    if not sections and rule in ("auto", "blank-gap"):
        sections = _from_gaps(text, body_start, body_end)
        used = "blank-gap"
        if rule == "auto" and not _plausible(sections, body_words):
            sections = []
    if not sections:
        sections = _from_words(text, body_start, body_end, block_words)
        used = "words"
        notes.append(
            Diagnostic.warning(
                "SECTIONS_NOT_CHAPTERS",
                f"These are blocks of {block_words:,} words, not the author's chapters: no headings were found. "
                "Each is titled by its words so no block is read as a chapter.",
            )
        )

    first = sections[0].start
    if first > body_start and text[body_start:first].strip():
        left_out.append(("front matter before the first heading", _words(text[body_start:first])))
    if contents:
        left_out.append(("table of contents lines", contents))
    for section in sections:
        if section.words < SHORT_WORDS:
            notes.append(
                Diagnostic.info(
                    "SECTIONS_SHORT", f"{section.title} is only {section.words} words.", order=section.order
                )
            )
        elif section.words > LONG_WORDS:
            notes.append(
                Diagnostic.warning(
                    "SECTIONS_LONG",
                    f"{section.title} is {section.words:,} words; a heading may have been missed.",
                    order=section.order,
                )
            )
    return Result.success(
        SectionPlan(
            sections=tuple(sections),
            rule=used,
            body_start=body_start,
            body_end=body_end,
            left_out=tuple(left_out),
        ),
        *notes,
    )


# ------------------------------------------------------------ transcripts --
#
# Interviews, hearings and debates interleave speakers. Comparing one
# speaker's debate rhetoric with their speeches means keeping only their
# turns, so a transcript is cut by speaker rather than by chapter (plan 2.5).

#: "MAYOR HALE:", "COUNCILLOR (DISTRICT 2):" -- an upper-case name of one to
#: four words at the start of a line, an optional role, then a colon.
_SPEAKER = re.compile(
    r"^[ \t]*(?P<name>[A-Z][A-Z'.\-]*(?:[ \t]+[A-Z][A-Z'.\-]*){0,3})"
    r"(?:[ \t]*\((?P<role>[^)\n]{1,40})\))?[ \t]*:[ \t]*",
    re.MULTILINE,
)
#: Stage directions a transcriber adds: (APPLAUSE), [CROSSTALK], (inaudible).
#: Bracketed text is always one; parenthesised text only when it is one of
#: these words, so "(district 2)" in a sentence stays.
_DIRECTION = re.compile(
    r"\[[^\]\n]{1,60}\]|\((?:applause|laughter|crosstalk|inaudible|booing|boos|cheers|cheering|"
    r"music|silence|pause|unintelligible|laughs|audience)[^)\n]{0,40}\)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class Turn:
    """One stretch of one speaker, with the stage directions taken out."""

    speaker: str
    role: str
    start: int
    end: int
    text: str
    words: int


@dataclass(frozen=True, slots=True)
class TurnPlan:
    """What cutting a transcript by speaker would do, for the preview."""

    turns: tuple[Turn, ...]
    #: speaker -> (turns, words), in order of first appearance.
    speakers: tuple[tuple[str, int, int], ...]
    #: Each stage direction as written, and how often it was left out.
    directions: tuple[tuple[str, int], ...]
    #: Words before the first speaker label (a title, a clerk's note).
    preamble_words: int


def _speaker_name(label: str) -> str:
    """ "MAYOR HALE" -> "Mayor Hale": how a Speaker detail reads elsewhere."""
    return " ".join(part.capitalize() for part in label.split())


def detect_turns(text: str) -> Result[TurnPlan]:
    """Each speaker's turns in a transcript, with stage directions counted and left out."""
    labels = list(_SPEAKER.finditer(text))
    if len({_speaker_name(m.group("name")) for m in labels}) < 2:  # noqa: PLR2004 - a transcript has two voices
        return Result.failure(
            Diagnostic.error(
                "TURNS_NO_SPEAKERS",
                "No two speakers were found. A transcript names each speaker in capitals at the start of a line, "
                "like 'MODERATOR:'.",
            )
        )
    directions: dict[str, int] = {}
    raw: list[Turn] = []
    for index, label in enumerate(labels):
        end = labels[index + 1].start() if index + 1 < len(labels) else len(text)
        body = text[label.end() : end]
        for found in _DIRECTION.findall(body):
            directions[found] = directions.get(found, 0) + 1
        clean = re.sub(r"[ \t]{2,}", " ", _DIRECTION.sub("", body))
        clean = "\n".join(line.strip() for line in clean.strip().splitlines())
        raw.append(
            Turn(
                speaker=_speaker_name(label.group("name")),
                role=(label.group("role") or "").strip(),
                start=label.start(),
                end=end,
                text=clean,
                words=_words(clean),
            )
        )
    # Consecutive turns by one speaker are one turn: a transcriber's line
    # break is not the other person speaking.
    turns: list[Turn] = []
    for turn in raw:
        if turns and turns[-1].speaker == turn.speaker:
            last = turns[-1]
            joined = f"{last.text}\n{turn.text}".strip()
            turns[-1] = Turn(last.speaker, last.role, last.start, turn.end, joined, _words(joined))
        else:
            turns.append(turn)
    counts: dict[str, list[int]] = {}
    for turn in turns:
        entry = counts.setdefault(turn.speaker, [0, 0])
        entry[0] += 1
        entry[1] += turn.words
    return Result.success(
        TurnPlan(
            turns=tuple(turns),
            speakers=tuple((name, n_turns, words) for name, (n_turns, words) in counts.items()),
            directions=tuple(sorted(directions.items(), key=lambda pair: (-pair[1], pair[0]))),
            preamble_words=_words(text[: labels[0].start()]),
        )
    )
