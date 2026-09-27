"""Smart scripts: the library a notebook imports as ``nlpsuite``.

A notebook asks for a corpus, runs the suite's own tools over it, and shows,
charts and saves what comes back. Everything a script can reach goes through
:mod:`core.script.api`; this package's other modules are the machinery behind
it (the session a kernel holds, where documents come from, what a cell's
outputs are, and the reference text generated from all of it).
"""
