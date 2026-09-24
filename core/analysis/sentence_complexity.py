"""Sentence complexity (FR-2.6) — dependency distance, depth, subordination.

Plus Yngve/Frazier over constituency trees, ported verbatim from legacy
``tree.py`` + ``sentence_complexity_node_util.py`` (both stdlib-only).

Parser requirements, documented (packet term):

* Dependency distance, depth, and subordination need a dependency parse —
  any suite backend (spaCy, Stanza) supplies ID/Head/DepRel columns.
* Yngve/Frazier need a constituency (phrase-structure) tree, which only the
  CoreNLP backend can supply (FR-5.2, SPEC_ONLY until then). They are pure
  functions over bracketed tree strings so they stay testable now; ``run()``
  does not call them.
* Subordination = tokens whose DepRel base is advcl/ccomp/xcomp/acl. That set
  is a documented choice, not a universal truth.

Legacy parity: ``compute_dependency_distance`` averaged |id - head| over
head > 0 words per sentence (single-token sentences score 0), 2dp.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from core.conll.schema import Col, validate_columns
from core.result import Diagnostic, Result

__all__ = [
    "ComplexityResult",
    "depth_of",
    "frazier_leaf_scores",
    "mean_dependency_distance",
    "mean_frazier",
    "mean_yngve",
    "run",
    "subordinate_count",
    "sum_frazier",
    "sum_yngve",
    "yngve_leaf_scores",
]

_SUBORDINATE_RELS = frozenset({"advcl", "ccomp", "xcomp", "acl"})


def mean_dependency_distance(ids: list[int], heads: list[int]) -> float:
    """Mean |id - head| over headed words; 0 when there is nothing to average."""
    distances = [abs(i - h) for i, h in zip(ids, heads, strict=True) if h > 0]
    if not distances:
        return 0.0
    return sum(distances) / len(distances)


def depth_of(heads: dict[int, int], token_id: int) -> int:
    """Edges from *token_id* up to the root (head <= 0), cycle-guarded."""
    depth = 0
    seen = {token_id}
    head = heads.get(token_id, 0)
    while head > 0 and head not in seen:
        seen.add(head)
        depth += 1
        head = heads.get(head, 0)
    return depth


def subordinate_count(deprels: list[str]) -> int:
    """Tokens whose DepRel base (before ':') marks a subordinate clause."""
    return sum(1 for rel in deprels if str(rel).split(":")[0] in _SUBORDINATE_RELS)


@dataclass
class _TreeNode:
    """Bracketed-tree node (legacy ``tree.Node`` shape, renamed)."""

    children: list[_TreeNode] = field(default_factory=list)


def _make_tree(tree_string: str) -> _TreeNode | None:
    stack: list[_TreeNode] = []
    root: _TreeNode | None = None
    for char in tree_string:
        if char == "(":
            node = _TreeNode()
            if stack:
                stack[-1].children.append(node)
            stack.append(node)
            if root is None:
                root = node
        elif char == ")" and stack:
            stack.pop()
    return root


@dataclass
class _ScoredNode:
    yngve: float = 0.0
    frazier: float = 0.0
    leaf: bool = False
    root: bool = True
    children: list[_ScoredNode] = field(default_factory=list)


def _score_tree(tree: _TreeNode, *, is_leaf: bool = False) -> _ScoredNode:
    """Legacy ``Node`` shape, verbatim: word brackets are always leaves.

    ``getChildrenAsList`` returns ``[node]`` itself when childless, so a
    childless root still gains one leaf child; every other childless bracket
    (``(DT The)``) is a leaf outright — there are no extra wrapper levels.
    """
    node = _ScoredNode(leaf=is_leaf)
    if is_leaf:
        return node
    kids = tree.children if tree.children else [tree]
    for kid in kids:
        if kid.children and kid is not tree:
            child = _score_tree(kid)
            child.root = False
            node.children.append(child)
        else:
            node.children.append(_ScoredNode(leaf=True, root=False))
    if not node.children:
        node.leaf = True
    return node


def _cal_yngve(node: _ScoredNode) -> None:
    for i, child in enumerate(node.children):
        child.yngve = node.yngve + len(node.children) - 1.0 - i
        if not child.leaf:
            _cal_yngve(child)


def _sum_yngve(node: _ScoredNode) -> float:
    if node.leaf:
        return node.yngve
    return sum(_sum_yngve(child) for child in node.children)


def _leaves(node: _ScoredNode) -> list[_ScoredNode]:
    if node.leaf:
        return [node]
    out: list[_ScoredNode] = []
    for child in node.children:
        out.extend(_leaves(child))
    return out


def yngve_leaf_scores(tree_string: str) -> Result[list[float]]:
    """Per-word Yngve scores, left to right (C6-7: malformed tree = diagnostic)."""
    if not tree_string.strip():
        return Result.success([])
    problem = _validate_tree_string(tree_string)
    if problem is not None:
        return Result.failure(problem)
    tree = _make_tree(tree_string)
    if tree is None:  # unreachable: validate_tree_string guarantees a root
        return Result.failure(Diagnostic.error("COMPLEXITY_BAD_TREE", "no root node found"))
    root = _score_tree(tree)
    _cal_yngve(root)
    leaves = _leaves(root)
    return Result.success([leaf.yngve for leaf in leaves])


def sum_yngve(tree_string: str) -> Result[float]:
    scored = yngve_leaf_scores(tree_string)
    if scored.value is None:
        return Result.failure(*scored.diagnostics)
    return Result.success(float(sum(scored.unwrap())))


def mean_yngve(tree_string: str) -> Result[float]:
    scored = yngve_leaf_scores(tree_string)
    if scored.value is None:
        return Result.failure(*scored.diagnostics)
    scores = scored.unwrap()
    return Result.success(sum(scores) / len(scores) if scores else 0.0)


def _cal_frazier(node: _ScoredNode) -> None:
    if not node.children:
        return
    first = node.children[0]
    first.frazier = node.frazier + 1.0 if not node.root else node.frazier + 1.5
    for i, child in enumerate(node.children):
        if i != 0:
            child.frazier = 0.0
        if not child.leaf:
            _cal_frazier(child)


def frazier_leaf_scores(tree_string: str) -> Result[list[float]]:
    """Per-word Frazier scores, left to right (C6-7: malformed tree = diagnostic)."""
    if not tree_string.strip():
        return Result.success([])
    problem = _validate_tree_string(tree_string)
    if problem is not None:
        return Result.failure(problem)
    tree = _make_tree(tree_string)
    if tree is None:  # unreachable: validate_tree_string guarantees a root
        return Result.failure(Diagnostic.error("COMPLEXITY_BAD_TREE", "no root node found"))
    root = _score_tree(tree)
    _cal_frazier(root)
    return Result.success([leaf.frazier for leaf in _leaves(root)])


def sum_frazier(tree_string: str) -> Result[float]:
    scored = frazier_leaf_scores(tree_string)
    if scored.value is None:
        return Result.failure(*scored.diagnostics)
    return Result.success(float(sum(scored.unwrap())))


def mean_frazier(tree_string: str) -> Result[float]:
    scored = frazier_leaf_scores(tree_string)
    if scored.value is None:
        return Result.failure(*scored.diagnostics)
    scores = scored.unwrap()
    return Result.success(sum(scores) / len(scores) if scores else 0.0)


@dataclass(frozen=True, slots=True)
class ComplexityResult:
    frame: pd.DataFrame

    def to_frame(self) -> pd.DataFrame:
        return self.frame.copy()


_OUTPUT_COLUMNS = [
    "Sentence ID",
    "Document ID",
    "Document",
    "Sentence",
    "Tokens",
    "Mean Dependency Distance",
    "Max Dependency Distance",
    "Depth",
    "Subordinate Clauses",
]


def _safe_int(value: object, *, coord: str) -> tuple[int | None, Diagnostic | None]:
    """Strict integer parse with coordinates; rejects non-int junk."""
    if isinstance(value, bool):
        return None, Diagnostic.error(
            "COMPLEXITY_BAD_INT", "boolean is not a valid ID/head", value=str(value), **{"_coord": coord}
        )
    try:
        if isinstance(value, float) and not float.is_integer(value):
            return None, Diagnostic.error(
                "COMPLEXITY_BAD_INT", f"fractional value {value!r}", value=str(value), **{"_coord": coord}
            )
        return int(value), None  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return None, Diagnostic.error(
            "COMPLEXITY_BAD_INT", f"value {value!r} is not an integer", value=str(value), **{"_coord": coord}
        )


def _validate_tree_string(tree_string: str) -> Diagnostic | None:
    """Balanced parentheses and exactly one root, or a diagnostic (no zeroes)."""
    depth = 0
    roots = 0
    depth_at_root = 0
    for i, char in enumerate(tree_string):
        if char == "(":
            if depth == 0:
                roots += 1
                depth_at_root = i
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return Diagnostic.error("COMPLEXITY_BAD_TREE", "unbalanced parentheses: close before open", position=i)
    if depth != 0:
        return Diagnostic.error(
            "COMPLEXITY_BAD_TREE", f"unbalanced parentheses: {depth} unclosed", length=len(tree_string)
        )
    if roots == 0:
        return Diagnostic.error("COMPLEXITY_BAD_TREE", "no root node found", length=len(tree_string))
    if roots > 1:
        return Diagnostic.error("COMPLEXITY_BAD_TREE", f"multiple roots ({roots})", roots=roots)
    return None


def _dependency_health(heads: list[int]) -> list[Diagnostic]:
    """Structural checks: self-heads, cycles, missing/multiple roots."""
    problems: list[Diagnostic] = []
    n = len(heads)
    roots = [i for i, h in enumerate(heads, start=1) if h == 0]
    if len(roots) == 0:
        problems.append(Diagnostic.error("COMPLEXITY_NO_ROOT", "sentence has no root token", n=n))
    elif len(roots) > 1:
        problems.append(Diagnostic.warning("COMPLEXITY_MULTI_ROOT", f"sentence has {len(roots)} roots", roots=roots))
    for token_id, head in enumerate(heads, start=1):
        if head == token_id:
            problems.append(
                Diagnostic.error("COMPLEXITY_SELF_HEAD", f"token {token_id} is its own head", token=token_id)
            )
    # cycle detection
    for start in range(1, n + 1):
        seen: set[int] = set()
        current = start
        while 0 < heads[current - 1] <= n and heads[current - 1] != current:
            if current in seen:
                problems.append(Diagnostic.error("COMPLEXITY_CYCLE", "dependency cycle detected", start=start))
                break
            seen.add(current)
            current = heads[current - 1]
        else:
            continue
        break  # only report the first cycle per sentence group
    return problems


def _dependency_health_with_ids(ids: list[int], heads: dict[int, int]) -> list[Diagnostic]:
    """Dependency health for sentence-local IDs (which need not be dense)."""
    problems: list[Diagnostic] = []
    roots = [token_id for token_id in ids if heads.get(token_id, 0) == 0]
    if not roots:
        problems.append(Diagnostic.error("COMPLEXITY_NO_ROOT", "sentence has no root token", n=len(ids)))
    elif len(roots) > 1:
        problems.append(Diagnostic.warning("COMPLEXITY_MULTI_ROOT", f"sentence has {len(roots)} roots", roots=roots))
    id_set = set(ids)
    for token_id in ids:
        head = heads.get(token_id, 0)
        if head == token_id:
            problems.append(
                Diagnostic.error("COMPLEXITY_SELF_HEAD", f"token {token_id} is its own head", token=token_id)
            )
    for start in ids:
        seen: set[int] = set()
        current = start
        while current in id_set and heads.get(current, 0) > 0:
            if current in seen:
                problems.append(Diagnostic.error("COMPLEXITY_CYCLE", "dependency cycle detected", start=start))
                return problems
            seen.add(current)
            current = heads[current]
    return problems


def run(frame: pd.DataFrame) -> Result[ComplexityResult]:
    """Per-sentence dependency complexity over a canonical CoNLL frame.

    C6-7: Head/ID values are validated (strings, NaN, booleans, fractions
    rejected with COMPLEXITY_BAD_INT naming doc/sentence/row); heads are
    validated as sentence-local 1..n with negative/out-of-range links
    excluded via COMPLEXITY_BAD_HEAD; self-heads, cycles, and missing/
    multiple roots are reported with document+sentence coordinates.
    Yngve/Frazier are NOT claimed as workflow outputs — they are pure
    functions pending FR-5.2 constituency trees (packet split recorded).
    """
    checked = validate_columns([str(c) for c in frame.columns])
    if not checked.ok:
        return Result[ComplexityResult](None, checked.diagnostics)
    required = [
        Col.ID.value,
        Col.FORM.value,
        Col.HEAD.value,
        Col.DEPREL.value,
        Col.SENTENCE_ID.value,
        Col.DOCUMENT_ID.value,
    ]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        return Result.failure(
            Diagnostic.error("COMPLEXITY_MISSING_COLUMN", f"missing column(s): {missing}", missing=missing),
        )
    doc_col = Col.DOCUMENT.value if Col.DOCUMENT.value in frame.columns else None
    rows: list[dict[str, object]] = []
    diags: list[Diagnostic] = []
    bad_head_examples: list[str] = []
    bad_head_count = 0
    if not frame.empty:
        grouped = frame.groupby([Col.DOCUMENT_ID.value, Col.SENTENCE_ID.value], sort=False)
        for (doc_id, sent_id), group in grouped:
            coord = f"doc {doc_id} sent {sent_id}"
            count = len(group)
            ids: list[int | None] = []
            raw_ids = group[Col.ID.value].tolist()
            int_diags: list[Diagnostic] = []
            for i, raw in enumerate(raw_ids):
                value, problem = _safe_int(raw, coord=f"{coord} row {i + 1} ID")
                if problem is not None:
                    int_diags.append(problem)
                ids.append(value)
            raw_heads = group[Col.HEAD.value].tolist()
            heads: list[int] = []
            for i, raw in enumerate(raw_heads):
                value, problem = _safe_int(raw, coord=f"{coord} row {i + 1} head")
                if problem is not None:
                    int_diags.append(problem)
                heads.append(value if value is not None else 0)
            if int_diags:
                diags.extend(int_diags[:3])
                if len(int_diags) > 3:
                    diags.append(
                        Diagnostic.warning(
                            "COMPLEXITY_BAD_INT_MORE", f"{len(int_diags) - 3} more invalid ID/head value(s) in {coord}"
                        )
                    )
                continue  # unusable sentence: never averaged with fabricated positions
            if any(i is None for i in ids):
                continue
            usable_ids = [i for i in ids if i is not None]
            # IDs may be non-dense (for example after parser filtering). Keep
            # them as the coordinate system; substituting 1..n changes both
            # dependency distance and depth and can silently corrupt output.
            dense_local = usable_ids == list(range(1, count + 1))
            if not dense_local:
                diags.append(
                    Diagnostic.warning(
                        "COMPLEXITY_NON_DENSE_IDS",
                        f"IDs in {coord} are not dense 1..n; raw IDs retained for distances",
                        coord=coord,
                        ids=usable_ids[:8],
                    )
                )
            if len(set(usable_ids)) != len(usable_ids):
                diags.append(
                    Diagnostic.error("COMPLEXITY_DUPLICATE_ID", f"duplicate token ID(s) in {coord}", coord=coord)
                )
                continue
            deprels = [str(v) for v in group[Col.DEPREL.value].tolist()]
            if any(h < 0 for h in heads):
                diags.append(
                    Diagnostic.error("COMPLEXITY_NEGATIVE_HEAD", f"negative head value(s) in {coord}", coord=coord)
                )
                continue
            id_set = set(usable_ids)
            out_of_range = [
                (token_id, h) for token_id, h in zip(usable_ids, heads, strict=True) if h > 0 and h not in id_set
            ]
            if out_of_range:
                bad_head_count += len(out_of_range)
                if len(bad_head_examples) < 3:
                    bad_head_examples.extend(
                        f"{coord} token {p} head {h}" for p, h in out_of_range[: 3 - len(bad_head_examples)]
                    )
                heads = [0 if h > 0 and h not in id_set else h for h in heads]
            head_map = dict(zip(usable_ids, heads, strict=True))
            diags.extend(_dependency_health_with_ids(usable_ids, head_map)[:2])
            distances = [abs(token_id - head) for token_id, head in zip(usable_ids, heads, strict=True) if head > 0]
            depth = max((depth_of(head_map, token_id) for token_id in usable_ids), default=0)
            rows.append(
                {
                    "Sentence ID": sent_id,
                    "Document ID": str(doc_id),
                    "Document": str(group[doc_col].iloc[0]) if doc_col is not None else "",
                    "Sentence": " ".join(str(v) for v in group[Col.FORM.value].tolist()),
                    "Tokens": count,
                    "Mean Dependency Distance": round(mean_dependency_distance(usable_ids, heads), 2),
                    "Max Dependency Distance": max(distances, default=0),
                    "Depth": depth,
                    "Subordinate Clauses": subordinate_count(deprels),
                }
            )
    if bad_head_count:
        diags.append(
            Diagnostic.warning(
                "COMPLEXITY_BAD_HEAD",
                f"{bad_head_count} head(s) point outside their sentence; excluded",
                count=bad_head_count,
                examples=bad_head_examples,
            )
        )
    out = pd.DataFrame(rows, columns=_OUTPUT_COLUMNS)
    return Result.success(ComplexityResult(frame=out), *diags)
