"""The stylesheet has to mean what it says.

Written after four rules in `styles.css` asked for `var(--border)`, which no
one had ever defined. A `var()` with no fallback that names nothing makes the
whole declaration invalid, and the browser drops it in silence -- so
`.insight-panel` was drawn with no border at all, and three separator rules
drew no separator. Nothing failed, nothing warned, and the panel had looked
wrong for as long as it had existed.

The same file held 214 distinct colours, dozens of them one hex digit apart:
`#4d6542` and `#4d6543`, `#fbfcf8` and `#fbfcf9`. Nobody can see the
difference, and everybody has to keep them in step.

These tests read the real stylesheet and the real components.
"""

from __future__ import annotations

from pathlib import Path
import re

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "desktop/src"
STYLES = SOURCE / "styles.css"

# Classes built from a template literal (`viz-origin-${item.origin}`) or handed
# to a third-party element. Each is spelled out so adding one is a decision.
DYNAMIC_CLASSES = {
    "family-",  # ToolCard: one per tool family
    "viz-origin-",  # VisualizationCatalog: one per provenance
    "lucide",  # icon library's own class
}

# Wrappers that deliberately carry no style: their children space themselves,
# and an empty rule would only be something else to keep.
UNSTYLED_ON_PURPOSE = {
    "run-list",  # .run-row draws its own separator
    "insight-charts",  # .insight-panel h4 supplies the spacing
}


def css() -> str:
    return STYLES.read_text(encoding="utf-8")


def defined_properties() -> set[str]:
    return set(re.findall(r"^\s*(--[\w-]+)\s*:", css(), flags=re.MULTILINE))


def used_properties() -> list[tuple[str, bool]]:
    """Every ``var(--x)`` in the sheet, and whether it supplied a fallback."""
    return [(m.group(1), bool(m.group(2))) for m in re.finditer(r"var\(\s*(--[\w-]+)\s*(,)?", css())]


def test_the_readers_of_this_file_are_not_reading_an_empty_string() -> None:
    assert len(css()) > 10_000
    assert len(defined_properties()) >= 6, defined_properties()
    assert len(used_properties()) >= 10


def test_every_custom_property_used_is_one_that_exists() -> None:
    """The bug this file was written for.

    A `var(--border)` that names nothing is not a fallback to some default --
    it invalidates the declaration it sits in, so the border is not drawn at
    all. Four rules did this and the app had been shipping without them.
    """
    defined = defined_properties()
    missing = sorted({name for name, fallback in used_properties() if name not in defined and not fallback})
    assert not missing, (
        f"{missing} are used with no fallback and never defined. Every rule using one "
        "is silently dropped by the browser. Define them in :root, or give a fallback."
    )


def test_no_custom_property_is_defined_and_never_used() -> None:
    used = {name for name, _ in used_properties()}
    unused = sorted(defined_properties() - used)
    assert not unused, f"{unused} are defined but nothing reads them."


def test_a_fallback_does_not_name_a_colour_the_app_does_not_have() -> None:
    """`var(--accent, #2563eb)` promised a blue this palette never contained.

    The token resolves, so the blue never appeared -- which is worse than if it
    had, because the next reader believes the app has a blue in it.
    """
    defined = defined_properties()
    for name, fallback in used_properties():
        if fallback and name in defined:
            pytest.fail(f"var({name}, ...) has a fallback but {name} is defined; the fallback is dead code.")


# ------------------------------------------------------------- the palette --


def colours() -> dict[str, int]:
    found: dict[str, int] = {}
    for raw in re.findall(r"#[0-9a-fA-F]{3,6}\b", css()):
        value = raw.lower().lstrip("#")
        if len(value) == 3:
            value = "".join(c * 2 for c in value)
        if len(value) == 6:
            found[value] = found.get(value, 0) + 1
    return found


def channels(value: str) -> tuple[int, int, int]:
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


def apart(a: str, b: str) -> int:
    return sum(abs(x - y) for x, y in zip(channels(a), channels(b), strict=True))


def test_no_two_colours_are_the_same_colour_spelled_twice() -> None:
    """Two hexes within four of each other are one colour and two maintainers.

    The threshold is deliberately tiny. This is not a rule about taste -- a
    palette may hold two greens that are genuinely different. It is a rule
    about `#4d6542` and `#4d6543`, which differ by one unit of blue and cannot
    be told apart by anybody, including the person who has to change both.
    """
    palette = sorted(colours())
    twins = [
        (f"#{a} ({colours()[a]}x)", f"#{b} ({colours()[b]}x)", apart(a, b))
        for index, a in enumerate(palette)
        for b in palette[index + 1 :]
        if apart(a, b) <= 4
    ]
    assert not twins, f"{len(twins)} pairs are the same colour twice: {twins[:6]}"


def test_the_palette_has_not_quietly_doubled() -> None:
    """A ceiling, not a target. It exists so that adding a hundred one-off
    colours is a decision somebody makes rather than a thing that happens."""
    assert len(colours()) <= 160, (
        f"{len(colours())} distinct colours. Reach for an existing one, or add a token to :root."
    )


# ------------------------------------------------- classes that exist twice --


def component_sources() -> list[Path]:
    return [p for p in sorted(SOURCE.glob("*.tsx")) if ".test." not in p.name]


# Two things inside a className expression look like class names and are not.
# A comparison operand -- `page === "setup" ? ... : ...` -- is a value. And a
# template literal's `${...}` holds code, not classes: reading
# `` `secondary ${filter === value ? "selected" : ""}` `` word by word offered
# up `value` as a class. Both produced phantom orphans the first time this ran,
# which is the failure mode these helpers exist to avoid: a checker that
# reports things that are fine teaches people to ignore it.
_COMPARED = re.compile(r"[!=]==?\s*$")
_INTERPOLATION = re.compile(r"\$\{[^{}]*\}")


def classes_used() -> dict[str, set[str]]:
    """Class names the components actually put on an element."""
    used: dict[str, set[str]] = {}
    for path in component_sources():
        text = path.read_text(encoding="utf-8")
        for expression in re.finditer(r'className=(\{(?:[^{}]|\{[^{}]*\})*\}|"[^"]*")', text):
            body = expression.group(1)
            for quoted in re.finditer(r'"([^"]*)"|`([^`]*)`', body):
                if _COMPARED.search(body[: quoted.start()].rstrip() + " "):
                    continue
                literal = quoted.group(1)
                if literal is None:
                    literal = _INTERPOLATION.sub(" ", quoted.group(2) or "")
                for name in literal.split():
                    if re.fullmatch(r"[a-z][\w-]*", name):
                        used.setdefault(name, set()).add(path.name)
    return used


def test_the_class_reader_is_not_reading_an_empty_string() -> None:
    found = classes_used()
    assert len(found) >= 60, sorted(found)
    assert "panel" in found and "workbench" in found


def test_every_class_a_component_renders_is_one_the_stylesheet_knows() -> None:
    """A class with no rule is an element the reader sees unstyled.

    `.view-bar` and its parts are the newest of these, and the reason this is
    checked: the markup and the stylesheet were written minutes apart in two
    files, and only one of them has to be forgotten.
    """
    styled = set(re.findall(r"\.([a-zA-Z][\w-]*)", css()))
    orphans = sorted(
        name
        for name, files in classes_used().items()
        if name not in styled
        and name not in UNSTYLED_ON_PURPOSE
        and not any(name.startswith(prefix) for prefix in DYNAMIC_CLASSES)
    )
    assert not orphans, f"{orphans} are rendered but have no rule in styles.css."


def test_the_saved_view_bar_is_actually_styled() -> None:
    """The general rule above can be satisfied by deleting the markup it was
    written for, so the specific case keeps its own test."""
    sheet = css()
    for name in (".view-bar", ".view-bar-name", ".view-bar-actions", ".secondary.danger"):
        assert name in sheet, name
    # Delete sits next to Duplicate in that bar; it must not read like them.
    assert ".secondary.danger" in sheet


def test_the_new_controls_survive_a_narrow_window() -> None:
    """Five controls in a row is a row that has to be allowed to wrap."""
    sheet = css()
    bar = sheet[sheet.index(".view-bar {") : sheet.index(".view-bar-name {")]
    assert "flex-wrap: wrap" in bar
    assert "@media (max-width: 720px)" in sheet


# ------------------------------------------------- text somebody can read --


def relative_luminance(value: str) -> float:
    def channel(pair: str) -> float:
        level = int(pair, 16) / 255
        return level / 12.92 if level <= 0.03928 else ((level + 0.055) / 1.055) ** 2.4

    red, green, blue = channel(value[0:2]), channel(value[2:4]), channel(value[4:6])
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def contrast(front: str, back: str) -> float:
    high, low = sorted((relative_luminance(front), relative_luminance(back)), reverse=True)
    return (high + 0.05) / (low + 0.05)


PAGE_BACKGROUND = "f8f9f6"


def token_values() -> dict[str, str]:
    return {name: value for name, value in re.findall(r"(--[\w-]+)\s*:\s*#([0-9a-fA-F]{3,6})", css())}


def as_six(value: str) -> str:
    value = value.lower().lstrip("#")
    return "".join(c * 2 for c in value) if len(value) == 3 else value


def coloured_rules() -> list[tuple[str, str, str]]:
    """(selector, text colour, background) for every rule that sets a colour."""
    tokens = {name: as_six(value) for name, value in token_values().items()}

    def resolve(raw: str) -> str | None:
        raw = raw.strip()
        named = re.fullmatch(r"var\((--[\w-]+)\)", raw)
        if named:
            return tokens.get(named.group(1))
        return as_six(raw) if re.fullmatch(r"#[0-9a-fA-F]{3,6}", raw) else None

    rules = []
    for selector, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css()):
        declarations = {
            key.strip(): value.strip()
            for key, _, value in (part.partition(":") for part in body.split(";"))
            if key.strip()
        }
        front = resolve(declarations.get("color", ""))
        if front is None:
            continue
        back = (
            resolve(declarations.get("background", ""))
            or resolve(declarations.get("background-color", ""))
            or PAGE_BACKGROUND
        )
        rules.append((selector.strip().splitlines()[-1], front, back))
    return rules


def test_the_contrast_reader_is_not_reading_an_empty_string() -> None:
    assert len(coloured_rules()) >= 60
    assert round(contrast("000000", "ffffff"), 1) == 21.0


def test_every_piece_of_text_meets_the_contrast_a_reader_needs() -> None:
    """98 rules did not, and the worst of them were the smallest type.

    Secondary text was drawn in sixty-seven near-identical greens, most around
    2.4:1 against the page -- readable if you already knew what it said. This
    is legibility, not taste, which is why it is a test and not a preference.
    """
    failures = [
        (selector, f"#{front}", f"#{back}", round(contrast(front, back), 2))
        for selector, front, back in coloured_rules()
        # An icon carries no words, so it answers to the 3:1 asked of
        # non-text content rather than the 4.5:1 asked of body copy.
        if contrast(front, back) < (3.0 if "svg" in selector else 4.5)
    ]
    assert not failures, f"{len(failures)} rules below WCAG AA: {failures[:5]}"


def test_the_secondary_text_token_is_itself_readable() -> None:
    """It was #758077 — 3.89:1 — so the token the whole app reaches for when it
    wants quiet text was the one handing out unreadable text."""
    muted = as_six(token_values()["--muted"])
    assert contrast(muted, PAGE_BACKGROUND) >= 4.5
    assert contrast(muted, "ffffff") >= 4.5


# Lettering inside the overview's drawing of a document. It is 5px and 6px
# because it is a picture of text at the scale of a thumbnail, not text: the
# element carrying it is aria-hidden, so no reader is ever asked to read it.
# test_the_tiny_type_is_only_ever_decoration keeps that true.
ILLUSTRATION_ONLY = {".paper-kicker", ".word-tags > span"}


def test_no_text_is_too_small_to_read_at_any_contrast() -> None:
    """Below about 8px, type stops being small and starts being a texture."""
    too_small = []
    for selector, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css()):
        name = selector.strip().splitlines()[-1].strip()
        if name in ILLUSTRATION_ONLY:
            continue
        for size in re.findall(r"font-size:\s*([\d.]+)px", body):
            if float(size) < 8:
                too_small.append((name, f"{size}px"))
    assert not too_small, f"type below 8px outside the illustration: {too_small}"


def test_the_tiny_type_is_only_ever_decoration() -> None:
    """The exemption above is earned by being hidden, not asserted.

    If the hero illustration ever stops being aria-hidden, its 5px lettering
    becomes text a reader is expected to read, and the exemption has to go
    with it.
    """
    app = (SOURCE / "App.tsx").read_text(encoding="utf-8")
    art = app.index('className="hero-art"')
    opening = app.rindex("<div", 0, art)
    assert 'aria-hidden="true"' in app[opening : art + 80], (
        "the hero illustration is no longer aria-hidden, so .paper-kicker and "
        ".word-tags are now text somebody is meant to read at 5px"
    )
    for name in ILLUSTRATION_ONLY:
        assert name in css(), name


def outside_media_queries() -> list[tuple[str, str]]:
    """(selector, body) for every rule not inside an ``@media`` block.

    A selector repeated inside a media query is a responsive override, which is
    the point of the block; a selector repeated outside one is two rules for the
    same elements, and only the later one counts.
    """
    sheet = css()
    spans = []
    for opening in re.finditer(r"@media[^{]*\{", sheet):
        depth, index = 1, opening.end()
        while index < len(sheet) and depth:
            depth += (sheet[index] == "{") - (sheet[index] == "}")
            index += 1
        spans.append((opening.start(), index))

    rules = []
    for rule in re.finditer(r"([^{}]+)\{([^{}]*)\}", sheet):
        if any(start <= rule.start() < end for start, end in spans):
            continue
        selector = " ".join(rule.group(1).split())
        if selector and not selector.startswith("@"):
            rules.append((selector, rule.group(2)))
    return rules


def test_the_rule_reader_is_not_reading_an_empty_string() -> None:
    assert len(outside_media_queries()) >= 200


def test_no_declaration_is_overridden_by_a_copy_of_its_own_rule() -> None:
    """`.badge` set 8px on line 649 and 10px on line 1686.

    A later block had been appended to make the small type readable rather than
    editing the rules it was correcting, so a quarter of the sheet's smallest
    sizes were dead text that a maintainer would read, believe, and change
    without effect. Removing a declaration the cascade was already discarding
    cannot alter what is drawn — and it did not: every selector's final set of
    declarations is identical to what it was.
    """
    seen: dict[str, set[str]] = {}
    shadowed: list[str] = []
    for selector, body in outside_media_queries():
        declared = {key.strip() for key, _, value in (part.partition(":") for part in body.split(";")) if key.strip()}
        for name in declared & seen.get(selector, set()):
            shadowed.append(f"{selector} sets {name} twice")
        seen.setdefault(selector, set()).update(declared)
    assert not shadowed, f"{len(shadowed)} dead declarations: {shadowed[:6]}"


# --------------------------------------------- headings somebody can follow --


def test_no_page_skips_a_heading_level() -> None:
    """Explore went `h1` straight to the workbench's `h3`.

    Under Runs & results the same panel sits below the run's own `h2`, so `h3`
    was right there and wrong on Explore. One component, two depths — which is
    why the level is the caller's to say.
    """
    workbench = (SOURCE / "Workbench.tsx").read_text(encoding="utf-8")
    assert "<h3>" not in workbench, "the panel hardcodes a depth again"
    assert "headingLevel" in workbench

    explore = (SOURCE / "Explore.tsx").read_text(encoding="utf-8")
    # The words in the `h1` now depend on which half the page asked for, so the
    # check is that there is one, not what it says.
    assert "<h1>" in explore
    assert "headingLevel={embedded ? 3 : 2}" in explore, (
        "Explore's workbench is an h2 under its own h1, and an h3 when Explore is itself a section of another page"
    )
    # Embedded under Past runs it must not open a second h1. A page with two
    # top-level headings cannot be navigated by its heading list.
    assert "embedded ? (" in explore, "Explore does not adapt its heading to being embedded"

    app = (SOURCE / "App.tsx").read_text(encoding="utf-8")
    assert "headingLevel" not in app, "the Runs pane's workbench sits under an h2, so the default h3 is right"
