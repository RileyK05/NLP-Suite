"""Notebooks the app offers to start from, which are also the guide's worked examples.

Each template answers one request a researcher would type, and the chatbot
guide (:mod:`core.script.reference`) shows it under that request. A template
must run on any dated corpus, not only the State of the Union:
tests/test_script_templates.py runs every one of them.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["TEMPLATES", "Template"]


@dataclass(frozen=True)
class Template:
    id: str
    name: str
    description: str
    #: The request, in a researcher's words, this notebook answers.
    request: str
    cells: tuple[tuple[str, str], ...]


_WORDS = """import nlpsuite as nlp

corpus = nlp.corpus()
WORDS = {
    "immigration": ["immigration", "immigrant", "immigrants", "migrant", "migrants",
                    "refugee", "refugees", "asylum", "border", "borders",
                    "deportation", "deportations", "citizenship", "alien", "aliens"],
}
# exact-lowercase: runs of letters in the raw text, as hand-written scripts count.
# Every inflection you want counted has to be in the list.
rates = nlp.term_rates(corpus, WORDS, per=1000, match="exact-lowercase")
nlp.show(rates, title="Rate in each document")"""

_YEARLY = """rate = "immigration per 1000"
yearly = rates.groupby("Year", as_index=False)[rate].mean()   # two documents in a year: averaged, not summed
yearly["trend"] = yearly[rate].rolling(5, center=True, min_periods=1).mean()
nlp.show(yearly, x="Year", y=[rate, "trend"], kind="line", title="Yearly mean and five-year trend")"""

_FIGURE = """import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8.6, 4.2))
ax.plot(rates["Year"], rates[rate], "o", ms=3.5, alpha=0.55, label="One document")
ax.plot(yearly["Year"], yearly["trend"], "-", lw=1.8, label="Five-observation trend")
ax.set(xlabel="Year", ylabel="Occurrences per 1,000 words", ylim=(0, None))
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False)
nlp.figure(fig, "word_group_over_time")"""

_PASSAGES = """passages = nlp.passages(corpus, WORDS, context=1, match="exact-lowercase")
nlp.show(passages, title="Every sentence that uses the words")
nlp.save(rates, "rates_per_document")"""

_PERIODS = """import nlpsuite as nlp

corpus = nlp.corpus()
middle = int(corpus.documents["Year"].median())   # or type the year you want to split at
early = corpus.filter(Year=lambda year: year < middle)
late = corpus.filter(Year=lambda year: year >= middle)
print(len(early), "documents before", middle, "and", len(late), "from", middle, "on")"""

_KEY = """words = nlp.keyness(early, late, top_n=40)
nlp.show(words, title="Words that set the earlier period apart")"""

_STYLE = """style = nlp.measures(corpus, which=("readability",))
style["Period"] = style["Year"].map(lambda year: "earlier" if year < middle else "later")
nlp.show(style, x="Period", y="Flesch Reading Ease", kind="box", title="Readability in each period")"""

TEMPLATES: tuple[Template, ...] = (
    Template(
        "word-group-over-time",
        "Word group over time",
        "How often a list of words is used in each document, by year, with a trend line, a figure and the passages.",
        "Graph how often immigration is mentioned in each speech over time, with a trend line.",
        (
            (
                "markdown",
                "# A word group over time\n\nThe rate of a fixed list of words per 1,000 words, yearly means "
                "(documents in the same year are averaged), and a five-year trend. Edit the list in the first cell.",
            ),
            ("code", _WORDS),
            ("code", _YEARLY),
            ("code", _FIGURE),
            ("code", _PASSAGES),
        ),
    ),
    Template(
        "compare-two-periods",
        "Compare two periods",
        "The words that set one period apart from another, and how their readability differs.",
        "Which words set the earlier speeches apart from the later ones, and did readability change?",
        (
            ("markdown", "# Two periods compared\n\nThe split is the median year; change `middle` to choose your own."),
            ("code", _PERIODS),
            ("code", _KEY),
            ("code", _STYLE),
        ),
    ),
    Template(
        "tone-over-time",
        "Tone over time",
        "VADER tone for each document, drawn over the years.",
        "Show the tone of each speech over time.",
        (("code", 'import nlpsuite as nlp\n\ntone = nlp.sentiment(nlp.corpus(), model="vader")\ntone'),),
    ),
    Template(
        "find-passages",
        "Find passages",
        "Every sentence that uses one of your words, with the sentences around it.",
        "Find every sentence that mentions war or peace, with some context.",
        (
            (
                "code",
                'import nlpsuite as nlp\n\nfound = nlp.passages(nlp.corpus(), ["war", "peace"], context=1)\nfound',
            ),
        ),
    ),
    Template(
        "start-from-a-run",
        "Start from a finished run",
        "Load a table an earlier analysis produced and chart it your way.",
        "Take the readability results I already ran and chart them my own way.",
        (
            ("code", "import nlpsuite as nlp\n\nnlp.tables()   # every table earlier runs produced"),
            (
                "code",
                '# Put a tool name (or a run id from the list above) here:\n# table = nlp.load("readability")\n# nlp.show(table)',
            ),
        ),
    ),
)
