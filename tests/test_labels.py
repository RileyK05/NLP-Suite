"""Every tool and parameter a reader can see must have a written label.

The desktop used to keep its own map of tool names and to derive parameter
labels by replacing underscores with spaces. Both failed quietly: three tools
added to ``CORPUS_TOOLS`` after the map was written showed as ``collocations``,
``tfidf`` and ``dispersion`` in lower case, three entries in the map named
tools the desktop does not publish, and every parameter label was the flag
name -- "sg", "op", "col x", "no normalize".

Nothing failed, because nothing checked. These tests check.
"""

from __future__ import annotations

import re

import pytest

from core.profiler.labels import (
    PARAM_LABEL_OVERRIDES,
    PARAM_LABELS,
    TOOL_DESCRIPTIONS,
    TOOL_LABELS,
    humanize,
    param_label,
    tool_label,
)
from core.profiler.registry import TOOL_REGISTRY, ToolSpec, get_tool
from desktop_backend.catalog import desktop_spec
from desktop_backend.runner import DESKTOP_TOOLS
from desktop_backend.tables import TABLE_TOOLS


def _spec(name: str) -> ToolSpec:
    """The spec behind a desktop tool name, whichever registry holds it."""
    found = TABLE_TOOLS.get(name)
    return found if found is not None else get_tool(name)


ALL_SPECS = (*TOOL_REGISTRY, *TABLE_TOOLS.values())


class TestEveryToolIsNamed:
    def test_every_registered_tool_has_a_label(self) -> None:
        missing = sorted(spec.name for spec in ALL_SPECS if spec.name not in TOOL_LABELS)
        assert not missing, (
            f"no label for {missing}; they would appear to the reader as raw identifiers. "
            "Add them to TOOL_LABELS in core/profiler/labels.py"
        )

    def test_no_label_names_a_tool_that_does_not_exist(self) -> None:
        """A label for a removed tool is dead weight that reads as coverage."""
        known = {spec.name for spec in ALL_SPECS}
        orphans = sorted(set(TOOL_LABELS) - known)
        assert not orphans, f"TOOL_LABELS names {orphans}, which are not in any registry"

    @pytest.mark.parametrize("name", sorted(DESKTOP_TOOLS))
    def test_each_desktop_tool_reads_as_a_name(self, name: str) -> None:
        label = tool_label(name)
        assert label and label[0].isupper(), f"{name} -> {label!r}"
        assert "_" not in label, f"{name} -> {label!r} still contains an identifier"

    @pytest.mark.parametrize("name", sorted(spec.name for spec in ALL_SPECS))
    def test_labels_are_specific_enough_to_choose_by(self, name: str) -> None:
        """The naming standard: a label says what the tool does in plain
        words ("N-grams over time (culturomics)"), so a reader can pick
        between two analyses from the gallery without opening the guide.

        Two failures this exists to prevent. A bare noun ("Knowledge graph",
        "Semantic tags") names a topic, not a thing the reader can do, and
        cannot be told apart from its neighbours. And an identifier with its
        separators stripped ("Nrc", "Kwic") is a flag wearing a capital
        letter: it describes the tool's implementation, not its question.
        Method names (Gensim LDA, Mann-Whitney) are allowed in parentheses,
        because the method is often the reason a reader wants that tool --
        but the words outside the parentheses must carry the meaning.
        """
        words = [w for w in re.split(r"[\s()]+", TOOL_LABELS[name]) if w]
        bare_nouns = {
            "graph",
            "statistics",
            "search",
            "senses",
            "entities",
            "frames",
            "classes",
            "tags",
            "filenames",
        }
        # A label made of nothing but a bare noun tells a reader nothing they
        # could act on. ("Named entities" names a *kind of finding*, which is
        # the exception: the tool's own name in the field.)
        assert not (len(words) <= 2 and set(w.lower().strip("()") for w in words) <= bare_nouns), (
            f"{name} -> {TOOL_LABELS[name]!r} names a topic, not an action; "
            "say what the tool does (see 'N-grams over time (culturomics)')"
        )


class TestEveryParameterIsNamed:
    def test_every_desktop_parameter_has_a_label(self) -> None:
        missing = sorted(
            {
                param.name
                for name in DESKTOP_TOOLS
                for param in desktop_spec(_spec(name)).params
                if param.name not in PARAM_LABELS
            }
        )
        assert not missing, (
            f"no label for {missing}; the form would show the flag name. "
            "Add them to PARAM_LABELS in core/profiler/labels.py"
        )

    def test_no_parameter_label_is_unused(self) -> None:
        used = {param.name for spec in ALL_SPECS for param in spec.params}
        orphans = sorted(set(PARAM_LABELS) - used)
        assert not orphans, f"PARAM_LABELS names {orphans}, which no tool declares"

    def test_every_override_points_at_a_real_parameter(self) -> None:
        pairs = {(spec.name, param.name) for spec in ALL_SPECS for param in spec.params}
        orphans = sorted(set(PARAM_LABEL_OVERRIDES) - pairs)
        assert not orphans, f"PARAM_LABEL_OVERRIDES names {orphans}, which do not exist"

    def test_an_override_actually_differs_from_the_shared_label(self) -> None:
        """An override that repeats the default is a copy waiting to drift."""
        same = sorted(key for key, value in PARAM_LABEL_OVERRIDES.items() if PARAM_LABELS.get(key[1]) == value)
        assert not same, f"these overrides repeat the shared label: {same}"

    # Abbreviations that only mean something to someone reading the command
    # line. A label is allowed to contain them inside a word ("Sankey"), never
    # as a word of its own.
    JARGON = frozenset({"col", "cols", "df", "tf", "sd", "n", "sg", "op", "agg", "idf", "ttr"})

    @pytest.mark.parametrize("name", sorted(DESKTOP_TOOLS))
    def test_no_desktop_form_shows_a_flag_name(self, name: str) -> None:
        spec = desktop_spec(_spec(name))
        for param in spec.params:
            label = param_label(name, param.name)
            assert label and label[0].isupper(), f"{name}.{param.name} -> {label!r}"
            assert label != param.name, f"{name}.{param.name} is labelled with its own flag name"
            assert "_" not in label, f"{name}.{param.name} -> {label!r} leaks an identifier"
            words = {word.strip("()–-,.").lower() for word in label.split()}
            assert not words & self.JARGON, (
                f"{name}.{param.name} -> {label!r} still carries command-line jargon ({sorted(words & self.JARGON)})"
            )

    def test_opaque_flags_are_actually_explained(self) -> None:
        """The abbreviations that started this: none of them may survive."""
        for tool, flag in (("word2vec_gensim", "sg"), ("table_search", "op"), ("table_charts", "agg")):
            label = param_label(tool, flag)
            assert len(label) > len(flag) + 3, f"{tool}.{flag} -> {label!r} is still an abbreviation"


class TestEveryToolIsDescribedToItsReader:
    """A card in the desktop shows a sentence. It should not be a flag list."""

    def test_no_description_names_a_tool_that_does_not_exist(self) -> None:
        known = {spec.name for spec in ALL_SPECS}
        orphans = sorted(set(TOOL_DESCRIPTIONS) - known)
        assert not orphans, f"TOOL_DESCRIPTIONS names {orphans}, which are not in any registry"

    @pytest.mark.parametrize("name", sorted(DESKTOP_TOOLS))
    def test_each_desktop_tool_reads_as_a_sentence(self, name: str) -> None:
        description = desktop_spec(_spec(name)).description
        assert description.endswith("."), f"{name}: {description!r}"
        assert description[0].isupper(), f"{name}: {description!r}"
        assert "|" not in description and "(needs a parse)" not in description, (
            f"{name} still shows its registry description: {description!r}"
        )

    def test_the_three_specs_that_say_more_keep_their_own_wording(self) -> None:
        """desktop_spec changes these alongside their parameters; do not clobber."""
        for name in ("sentiment_vader_anew", "search", "spellcheck"):
            assert name not in TOOL_DESCRIPTIONS, (
                f"{name} is described inside desktop_spec, next to the parameter "
                "changes that make the description true; a second copy here would drift"
            )
            assert desktop_spec(_spec(name)).description != _spec(name).description


class TestTheFallbackIsOnlyAFallback:
    def test_humanize_capitalises_and_unpacks(self) -> None:
        assert humanize("max-df-ratio") == "Max df ratio"
        assert humanize("topic_model") == "Topic model"

    def test_an_unknown_name_still_renders(self) -> None:
        assert tool_label("brand_new_tool") == "Brand new tool"
        assert param_label("brand_new_tool", "some-flag") == "Some flag"


class TestEveryInstalledComponentIsNamed:
    """Settings names what is installed; the scan answers in import names.

    The setup endpoint used to send the raw scan and the desktop showed none
    of it, so a reader could see that an analysis was unavailable but not what
    was missing or what it was for.
    """

    def test_every_scanned_name_is_described(self) -> None:
        from desktop_backend.environment import COMPONENT_LABELS, inventory

        missing = sorted(set(inventory()) - set(COMPONENT_LABELS))
        assert not missing, (
            f"the scan reports {missing} with no label or purpose; "
            "add them to COMPONENT_LABELS in desktop_backend/environment.py"
        )

    def test_no_description_names_something_never_scanned(self) -> None:
        from desktop_backend.environment import COMPONENT_LABELS, inventory

        orphans = sorted(set(COMPONENT_LABELS) - set(inventory()))
        assert not orphans, f"COMPONENT_LABELS names {orphans}, which the scan never reports"

    def test_each_component_says_what_it_is_for(self) -> None:
        from desktop_backend.environment import components

        for component in components():
            assert component["label"], component["name"]
            assert str(component["purpose"]).endswith("."), component["name"]
            assert isinstance(component["present"], bool)

    def test_every_component_a_tool_can_require_is_named(self) -> None:
        """A missing component is shown by name, so every one must have a name."""
        from desktop_backend.environment import COMPONENT_LABELS, availability, inventory

        found = dict.fromkeys(inventory(), False)  # pretend nothing is installed
        required = {
            name for tool in DESKTOP_TOOLS for name in availability(desktop_spec(_spec(tool)), found)["missing"]
        }
        assert required, "with nothing installed some tool should report a missing component"
        assert not sorted(required - set(COMPONENT_LABELS)), (
            f"tools can require {sorted(required - set(COMPONENT_LABELS))}, which Settings cannot name"
        )


class TestTheNamingStandard:
    """Every tool is named "Task: what it does exactly (method)".

    Written after the gallery offered four tools ending "(BERT)" with no way
    to tell sentiment from embeddings, and collocations and the map tools
    hid behind friendly names. The task comes first and is one of a fixed
    vocabulary, so every sentiment tool reads "Sentiment: ..." and a search
    for "collocations" or "geography" finds every tool that does it.
    """

    TASKS = frozenset(
        {
            "Batch",
            "Charts",
            "Cleaning",
            "Collocations",
            "Counts",
            "Dates",
            "Embeddings",
            "Emotion",
            "Entities",
            "Frequencies",
            "Gender",
            "Geography",
            "Grammar",
            "Intake",
            "Keywords",
            "Lexicons",
            "N-grams",
            "Quotes",
            "Readability",
            "Search",
            "Sentiment",
            "Similarity",
            "Statistics",
            "Story shape",
            "Summary",
            "Topics",
            "Vocabulary",
            "Word norms",
        }
    )

    @pytest.mark.parametrize("name", sorted(TOOL_LABELS))
    def test_a_label_starts_with_its_task(self, name: str) -> None:
        label = TOOL_LABELS[name]
        task, _, rest = label.partition(": ")
        assert rest, f"{name} -> {label!r}: name it 'Task: what it does'"
        assert task in self.TASKS, f"{name} -> {label!r}: {task!r} is not one of the tasks {sorted(self.TASKS)}"

    def test_tools_that_do_one_thing_share_its_task(self) -> None:
        by_task: dict[str, set[str]] = {}
        for name, label in TOOL_LABELS.items():
            by_task.setdefault(label.partition(": ")[0], set()).add(name)
        assert {"sentiment_neural_bert", "sentiment_vader_anew", "sentiment_neural_spacy"} <= by_task["Sentiment"]
        assert {"word2vec_bert", "word2vec_gensim", "word_sense_induction", "doc_embeddings"} <= by_task["Embeddings"]
        assert {"collocations", "ngram_cooccurrence"} <= by_task["Collocations"]
        assert {"geocode", "gis_map", "svo_map"} <= by_task["Geography"]
