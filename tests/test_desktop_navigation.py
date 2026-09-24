"""Every page the desktop can show has a way to get to it.

Written after the interactive workbench shipped with no front door. It was
built, tested, bundled and served, and it worked -- but the only route to it
was Runs & results, then a finished job, then the right CSV artifact. From
the outside that is indistinguishable from a feature that does not exist,
and that is exactly what it was reported as.

A page unreachable from the interface is not a page. These tests read the
real App.tsx rather than trusting a list restated in Python.
"""

from __future__ import annotations

from pathlib import Path
import re

import pytest

APP = Path(__file__).resolve().parents[1] / "desktop/src/App.tsx"

# Pages deliberately kept out of the sidebar. Each needs a reason, and each
# reason is checked below -- an entry here must still be reachable, just by a
# different control.
OFF_THE_SIDEBAR = {"setup": "Settings & backups opens from the gear in the sidebar footer."}


def source() -> str:
    return APP.read_text(encoding="utf-8")


def declared_pages() -> set[str]:
    """The ``Page`` union: every value the app's router can hold."""
    text = source()
    start = text.index("type Page =")
    body = text[start : text.index(";", start)]
    return set(re.findall(r'"([a-z]+)"', body))


def sidebar_pages() -> set[str]:
    """The keys of the navigation array the sidebar renders."""
    text = source()
    start = text.index("const navigation = [")
    body = text[start : text.index("] as const;", start)]
    return set(re.findall(r'key:\s*"([a-z]+)"', body))


def rendered_pages() -> set[str]:
    """Every page with a branch that actually draws something."""
    return set(re.findall(r'page === "([a-z]+)"', source()))


def test_the_readers_of_this_file_are_not_reading_an_empty_string() -> None:
    """Guard against a regex that silently matches nothing and passes."""
    assert len(declared_pages()) >= 7, declared_pages()
    assert len(sidebar_pages()) >= 6, sidebar_pages()
    assert "overview" in declared_pages()
    assert "overview" in sidebar_pages()
    assert "overview" in rendered_pages()


def test_every_page_is_reachable() -> None:
    unreachable = declared_pages() - sidebar_pages() - set(OFF_THE_SIDEBAR)
    assert not unreachable, (
        f"No way to reach {sorted(unreachable)}. Add a sidebar entry, or add it to "
        "OFF_THE_SIDEBAR with the control that opens it."
    )


@pytest.mark.parametrize("page", sorted(OFF_THE_SIDEBAR))
def test_a_page_kept_off_the_sidebar_still_has_a_control(page: str) -> None:
    assert f'setPage("{page}")' in source(), OFF_THE_SIDEBAR[page]


def test_every_page_draws_something() -> None:
    """A nav entry pointing at a blank main area is a dead link with an icon."""
    silent = sidebar_pages() - rendered_pages()
    assert not silent, f"{sorted(silent)} can be clicked but render nothing."


def test_no_page_is_drawn_that_cannot_be_selected() -> None:
    """The reverse: a branch for a page value the router can never hold."""
    orphaned = rendered_pages() - declared_pages()
    assert not orphaned, f"{sorted(orphaned)} is drawn but is not a Page."


def test_the_interaction_lab_is_one_of_those_pages() -> None:
    """The specific regression: live charting reachable in one click.

    Keeping this as its own test means the general rules above cannot be
    satisfied by quietly deleting the page they were written for. The page was
    called "explore" when it also held the archive of finished tables; it is
    called "interactive" now that it holds only the live bench. The name is
    allowed to change. Being one click from the sidebar is not.
    """
    assert "interactive" in sidebar_pages()
    assert "interactive" in rendered_pages()
    assert "<Explore" in source()


def sidebar_order() -> list[str]:
    """The navigation keys in the order the sidebar draws them."""
    text = source()
    start = text.index("const navigation = [")
    body = text[start : text.index("] as const;", start)]
    return re.findall(r'key:\s*"([a-z]+)"', body)


def test_the_research_pages_read_in_the_order_the_work_happens() -> None:
    """Interactive, then the studio, then what you already ran.

    Asked for directly, and the reason is that the three are one sequence: ask
    something small and watch it answer, run the heavy version over everything,
    come back to the results. Ordered any other way the sidebar is three nouns
    that happen to be near each other.
    """
    order = sidebar_order()
    positions = [order.index(key) for key in ("interactive", "studio", "runs")]
    assert positions == sorted(positions), f"sidebar reads {order}"


def test_analysis_and_visualization_are_one_page() -> None:
    """They were two pages for one action, which is the clutter complaint.

    Both did the same thing -- choose a workflow, fill its parameters, queue a
    job -- and differed only in which half of the tool list they filtered to.
    """
    text = source()
    assert "analyses" not in sidebar_pages(), "the split page came back"
    assert "visualize" not in sidebar_pages(), "the split page came back"
    # Grouping is by syllabus family (the "weird chunking" fix), not by the
    # old analysis/visualization input-format guess.
    assert "FAMILY_ORDER" in text, "the studio lost its family grouping"
    assert "tool.family" in text or "(tool.family ?? '')" in text, "the studio stopped reading the engine's family"


def test_the_archive_is_fetched_on_the_page_that_draws_it() -> None:
    """Published tables and saved views must be asked for where they are shown.

    The regression this is written from: the archive moved from Explore to Past
    runs, and the two fetches that fill it stayed gated on the page it had
    left. Nothing failed. The section drew itself on the right page, empty,
    with no error -- because it never asked for anything. A feature that is
    present and permanently blank is indistinguishable from one that was
    deleted, and that is how it was reported.

    So the page name is written once and read by both sides, and this checks
    that both sides really do read it.
    """
    text = source()
    match = re.search(r'const ARCHIVE_PAGE = "([a-z]+)" as const;', text)
    assert match, "ARCHIVE_PAGE is gone; the fetch and the render can drift again"
    page = match.group(1)

    assert page in sidebar_pages(), f"the archive lives on {page!r}, which is not in the sidebar"

    # Both fetches read the name rather than repeating a page string.
    gated = text.count("page !== ARCHIVE_PAGE")
    assert gated == 2, f"expected the table and view fetches to be gated on ARCHIVE_PAGE, found {gated}"

    # And the section really is drawn inside that page's branch: from where the
    # branch opens to where the next page's branch does.
    branch = text.index(f'page === "{page}" && (')
    others = [
        text.index(f'page === "{name}" && (', branch)
        for name in rendered_pages()
        if f'page === "{name}" && (' in text[branch + 1 :]
    ]
    end = min(others) if others else len(text)
    assert 'exploreView("results")' in text[branch:end], f"the archive is not drawn under page {page!r}"


def test_the_live_bench_is_still_on_the_interactive_page() -> None:
    """The other half of the same move: Interactive keeps the live bench.

    Splitting Explore's two tabs into two sidebar entries is only correct if
    each page ends up with its half. This checks the live one did.
    """
    text = source()
    assert 'exploreView("live")' in text
    assert "<LiveBench" in text, "the live bench element is gone"
    assert 'page === "interactive" && exploreView("live")' in text


# State that belongs to one project and would be wrong if it survived a switch
# to another. Each name is the setter the reset effect must call.
PER_PROJECT_STATE = (
    "setDocuments",
    "setJobs",
    "setSelectedJob",
    "setEnvelope",
    "setTable",
    "setPreview",
    "setSources",
    "setSource",
    "setSourceTable",
    # Saved views name a job id. Carried into another project, the view list
    # would offer arrangements of runs this project does not have.
    "setViews",
)


def project_change_effect() -> str:
    """The body of the effect that runs when projectId changes."""
    text = source()
    start = text.index("    setDocuments([]);")
    return text[start : text.index("if (!projectId) return;", start)]


def test_switching_projects_clears_what_belonged_to_the_old_one() -> None:
    """A left-behind selection asks the new project for another project's run.

    Explore holds a job id and an artifact index. Carried across a project
    switch, the next fetch addresses a run that this project does not have,
    and the reader is shown an error instead of their tables.
    """
    effect = project_change_effect()
    missing = [setter for setter in PER_PROJECT_STATE if setter not in effect]
    assert not missing, f"{missing} survive a project switch"


def test_that_effect_is_the_one_it_claims_to_be() -> None:
    effect = project_change_effect()
    assert len(effect) < 2000, "matched far more than one effect body"
    assert "setDocuments([]);" in effect
