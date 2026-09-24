"""Insight — help the analyst read their own output.

Two problems this package exists to solve, both of which show up the moment a
tool finishes and hands someone a CSV.

*Which chart is worth drawing?* Offering every column against every other
column invites the degenerate chart: plot corpus size against corpus size and
you get a perfect line, which looks like a dramatic finding and says nothing.
:mod:`core.insight.recommend` proposes a short list of charts that answer real
questions about a given table, and names the pairings that cannot be
informative and why.

*What does this table actually say?* :mod:`core.insight.readout` writes a
plain-language reading of a result: its shape, what dominates it, and the
specific traps that table is prone to. It describes what is in front of it and
never infers significance, causation or a recommendation the numbers do not
support.

Both are built on :mod:`core.insight.profile`, which decides what kind of
thing each column is.
"""

from __future__ import annotations

from core.insight.profile import ColumnProfile, ColumnRole, profile_frame
from core.insight.readout import Readout, readout
from core.insight.recommend import Recommendation, degenerate_reasons, recommend_charts

__all__ = [
    "ColumnProfile",
    "ColumnRole",
    "Readout",
    "Recommendation",
    "degenerate_reasons",
    "profile_frame",
    "readout",
    "recommend_charts",
]
