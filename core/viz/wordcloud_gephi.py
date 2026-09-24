"""Wordclouds and Gephi GEXF.

The raster wordcloud (``wordcloud_image``) is the port of the legacy
``wordclouds_util.py`` (Andreas Mueller's ``wordcloud`` package): spiral
layout, collision detection, optional image mask with contour, per-group
recoloring. The package is the optional ``[wordcloud]`` extra and loads
lazily — a missing package fails loudly with the fix command, never with
a stack trace. Offline tests inject a fake via ``WordcloudBackend``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import html as html_lib
import math
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd

from core.result import Diagnostic, Result

__all__ = [
    "WordcloudBackend",
    "gephi_gexf",
    "load_mask",
    "wordcloud_html",
    "wordcloud_image",
]


def _finite_weight(value: object) -> float | None:
    """Coerce *value* to a usable finite weight, or None if it cannot be one.

    ``float(nan)`` does not raise, so a bare try/except let NaN through and
    GEXF files ended up with ``weight="nan"``, which is not valid XML data.
    """
    try:
        w = float(value)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return None
    if not math.isfinite(w):
        return None
    return w


def wordcloud_html(
    frame: pd.DataFrame,
    word_col: str,
    weight_col: str,
    title: str = "",
    max_words: int = 100,
) -> Result[str]:
    """HTML wordcloud where font-size ∝ weight."""
    if word_col not in frame.columns or weight_col not in frame.columns:
        return Result.failure(
            Diagnostic.error("WC_BAD_COLUMN", f"columns {word_col!r}, {weight_col!r} must be in frame")
        )
    if frame.empty:
        return Result.failure(Diagnostic.error("WC_EMPTY", "frame is empty"))
    if max_words < 1:
        return Result.failure(Diagnostic.error("WC_BAD_MAX", f"max_words must be >=1, got {max_words}"))

    sub = frame[[word_col, weight_col]].copy()
    sub[weight_col] = pd.to_numeric(sub[weight_col], errors="coerce").fillna(0)
    sub = sub[sub[weight_col] > 0].sort_values(weight_col, ascending=False).head(max_words)
    if sub.empty:
        return Result.failure(Diagnostic.error("WC_NO_WEIGHT", "no positive weights"))

    max_w = float(sub[weight_col].max()) or 1.0
    min_w = float(sub[weight_col].min()) or 0.0
    span = max_w - min_w or 1.0

    parts: list[str] = []
    for _, row in sub.iterrows():
        w = html_lib.escape(str(row[word_col]))
        weight = float(row[weight_col])
        # Font size 12..48
        size = 12 + (weight - min_w) / span * 36
        # Simple color interpolation by weight
        alpha = (weight - min_w) / span
        # blue -> red
        r = int(55 + alpha * 180)
        b = int(200 - alpha * 150)
        parts.append(
            f'<span style="font-size:{size:.1f}px;color:rgb({r},60,{b});margin:4px;display:inline-block">{w}</span>'
        )

    title_html = f"<h3>{html_lib.escape(title)}</h3>" if title else ""
    html_str = (
        f"{title_html}<div style='line-height:1.8;text-align:center;padding:12px;border:1px solid #ddd'>\n"
        + " ".join(parts)
        + "\n</div>"
    )
    return Result.success(html_str)


# ---------------------------------------------------------------------------
# Raster wordcloud (the legacy wordclouds_util.py port)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Procedural shape masks (no image files, no licensed bytes)
# ---------------------------------------------------------------------------

# Fay's butterfly curve (Temple H. Fay, 1989): the classic published
# parametric form. One period (t in [0, 2*pi]) traces the full butterfly
# exactly once; sampling it beyond that retraces the same shape and makes
# the fill self-intersect. Pure math — nothing here is vendored or licensed.
_BUTTERFLY_T_RANGE = 2 * math.pi
_BUTTERFLY_SAMPLES = 2000
_SHAPE_MIN_SIZE = 64  # below this the mask cannot hold a word at all


def butterfly_mask(size: int = 800, *, thickness: int = 6, padding: int = 16) -> Result[Any]:
    """A butterfly-shaped mask as the numpy array ``WordCloud`` expects.

    White (255) is "no words here"; black strokes delimit the usable
    region, matching :func:`load_mask`'s convention. The curve is drawn
    scaled and centered on a white ``size x size`` canvas.
    """
    if size < _SHAPE_MIN_SIZE:
        return Result.failure(Diagnostic.error("WC_SHAPE_TOO_SMALL", f"mask size must be >=64, got {size}"))
    if thickness < 1 or padding < 0:
        return Result.failure(
            Diagnostic.error("WC_SHAPE_BAD_ARGS", f"thickness must be >=1 and padding >=0, got {thickness}/{padding}")
        )
    try:
        import numpy as np
        from PIL import Image, ImageDraw
    except ImportError as exc:
        return Result.failure(
            Diagnostic.error("WC_SHAPE_NO_PIL", f"shape masks need Pillow: {exc}", fix="pip install pillow")
        )
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    inner = size - 2 * padding
    points: list[tuple[float, float]] = []
    for step in range(_BUTTERFLY_SAMPLES):
        t = _BUTTERFLY_T_RANGE * step / _BUTTERFLY_SAMPLES
        # Fay's curve: r = e^{sin t} - 2cos(4t) + sin^5((2t - pi)/24)
        radius = math.e ** math.sin(t) - 2 * math.cos(4 * t) + (math.sin((2 * t - math.pi) / 24)) ** 5
        # The curve already sits upright in math space (y up); image space
        # grows y downward, so flip y only. No rotation: rotating by pi/2
        # collapses the curve's mirror symmetry (x is invariant under it,
        # which retraces one wing instead of drawing both).
        points.append((radius * math.cos(t), -radius * math.sin(t)))
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    # Fit the true bounding box: Fay's curve is y-symmetric but its x
    # extent is NOT centered (the body/tail hangs below the wings), so
    # scaling by max|x| would clip one wing while padding the other.
    width_raw = max(xs) - min(xs) or 1.0
    height_raw = max(ys) - min(ys) or 1.0
    scale = min(inner / width_raw, inner / height_raw)
    mid_x = (max(xs) + min(xs)) / 2
    mid_y = (max(ys) + min(ys)) / 2
    center = size / 2
    scaled = [(center + (x - mid_x) * scale, center + (y - mid_y) * scale) for x, y in points]
    # One closed polygon over a single curve period fills the wing body
    # cleanly (no self-crossing retraces). A thick outline pass keeps the
    # wing edges solid for word placement; outline drawing on a closed
    # path is safe where sequential line segments were not.
    draw.polygon(scaled, fill=(0, 0, 0), outline=(0, 0, 0), width=thickness)
    mask = np.asarray(canvas)
    return Result.success(mask)


_SHAPE_GENERATORS: dict[str, Callable[..., Result[Any]]] = {
    "butterfly": butterfly_mask,
}


def shape_mask(shape: str, size: int = 800) -> Result[Any]:
    """A named procedural mask (``--shape butterfly``). Loud on unknowns."""
    generator = _SHAPE_GENERATORS.get(shape.lower())
    if generator is None:
        return Result.failure(
            Diagnostic.error(
                "WC_SHAPE_UNKNOWN",
                f"unknown shape {shape!r} (available: {', '.join(sorted(_SHAPE_GENERATORS))})",
                shape=shape,
            )
        )
    return generator(size)


class WordcloudBackend(Protocol):
    """The surface of ``wordcloud.WordCloud`` this module needs.

    Production resolves it to the real class lazily (the optional extra
    must not gate imports); offline tests inject a deterministic fake.
    """

    def __call__(self, **kwargs: Any) -> Any: ...

    def generate_from_frequencies(self, frequencies: dict[str, float]) -> Any: ...

    def to_file(self, path: str) -> None: ...


def _real_backend() -> WordcloudBackend | None:
    try:
        from wordcloud import WordCloud
    except ImportError:
        return None
    return cast("WordcloudBackend", WordCloud)


_MASK_MIN_SIDE = 8  # below this a mask cannot hold a word at all
_MASK_WHITE_TOLERANCE = 245  # channels above this read as "white"
_MASK_RGB_DIMS = 3  # (height, width, bands)
_MASK_RGB_BANDS = 3


def load_mask(mask_path: Path) -> Result[Any]:
    """Load an image mask as the numpy array ``WordCloud`` expects.

    Non-white pixels (r,g,b > tolerance rejected) delimit where words may
    render; a fully-white mask is the legacy's "empty mask" mistake and
    fails loudly here rather than producing an invisible cloud.
    """
    try:
        from PIL import Image
    except ImportError as exc:
        return Result.failure(
            Diagnostic.error("WC_MASK_NO_PIL", f"image masks need Pillow: {exc}", fix="pip install pillow")
        )
    path = Path(mask_path)
    if not path.is_file():
        return Result.failure(Diagnostic.error("WC_MASK_MISSING", f"mask image not found: {path}", path=str(path)))
    try:
        with Image.open(path) as img:
            mask = np.asarray(img.convert("RGB"))
    except OSError as exc:
        return Result.failure(
            Diagnostic.error("WC_MASK_UNREADABLE", f"could not read mask {path}: {exc}", path=str(path))
        )
    if mask.ndim < _MASK_RGB_DIMS or mask.shape[2] < _MASK_RGB_BANDS:
        return Result.failure(Diagnostic.error("WC_MASK_BAD_MODE", f"mask must decode to RGB, got shape {mask.shape}"))
    height, width = mask.shape[:2]
    if width < _MASK_MIN_SIDE or height < _MASK_MIN_SIDE:
        return Result.failure(Diagnostic.error("WC_MASK_TOO_SMALL", f"mask must be at least 8x8, got {width}x{height}"))
    # Non-white region check over band slices (PIL's getdata is deprecated
    # in Pillow 14; numpy slicing avoids it).
    bands = [np.asarray(mask[:, :, band] <= _MASK_WHITE_TOLERANCE) for band in range(_MASK_RGB_BANDS)]
    usable = int(np.logical_or.reduce(bands).sum())
    if usable == 0:
        return Result.failure(
            Diagnostic.error(
                "WC_MASK_ALL_WHITE",
                "mask is entirely white — no usable non-white region for words",
                path=str(path),
            )
        )
    return Result.success(mask)


def _color_to_words(words: pd.Series, group_col: pd.Series) -> dict[str, list[str]]:
    """Grouped recolor map ({color string -> [words]}) from a group column."""
    mapping: dict[str, list[str]] = {}
    for word, group in zip(words.astype(str), group_col.astype(str), strict=True):
        mapping.setdefault(group, []).append(word)
    return mapping


def wordcloud_image(
    frame: pd.DataFrame,
    word_col: str,
    weight_col: str,
    *,
    output_path: Path,
    width: int = 800,
    height: int = 800,
    max_words: int = 200,
    prefer_horizontal: float = 0.9,
    background_color: str = "white",
    colormap: str = "viridis",
    mask_path: Path | None = None,
    shape: str | None = None,
    contour_width: int = 3,
    contour_color: str = "firebrick",
    group_col: str | None = None,
    group_colors: Mapping[str, str] | None = None,
    seed: int | None = None,
    backend: WordcloudBackend | None = None,
) -> Result[Path]:
    """Render a raster wordcloud PNG with the legacy layout engine.

    Legacy parity (``wordclouds_util.py``): ``generate_from_frequencies``
    over pre-counted words, optional image mask with a firebrick contour,
    per-group recoloring (SVO-style: one color per word group), PNG output.
    Divergences are deliberate: the RNG is seeded by default (the legacy
    drew unseeded, so no two of its clouds agreed), and the missing extra
    fails with ``WC_PACKAGE_MISSING`` plus the fix command instead of a
    GUI-less traceback.
    """
    if word_col not in frame.columns or weight_col not in frame.columns:
        return Result.failure(
            Diagnostic.error("WC_BAD_COLUMN", f"columns {word_col!r}, {weight_col!r} must be in frame")
        )
    if group_col is not None and group_col not in frame.columns:
        return Result.failure(Diagnostic.error("WC_BAD_GROUP_COLUMN", f"group column {group_col!r} not in frame"))
    if frame.empty:
        return Result.failure(Diagnostic.error("WC_EMPTY", "frame is empty"))
    if max_words < 1:
        return Result.failure(Diagnostic.error("WC_BAD_MAX", f"max_words must be >=1, got {max_words}"))

    columns = [word_col, weight_col] + ([group_col] if group_col is not None else [])
    sub = frame[columns].copy()
    sub[weight_col] = pd.to_numeric(sub[weight_col], errors="coerce").fillna(0)
    sub = sub[sub[weight_col] > 0].sort_values(weight_col, ascending=False).head(max_words)
    if sub.empty:
        return Result.failure(Diagnostic.error("WC_NO_WEIGHT", "no positive weights"))

    frequencies = {str(w): float(x) for w, x in zip(sub[word_col], sub[weight_col], strict=True)}

    mask: Any = None
    if mask_path is not None and shape is not None:
        return Result.failure(
            Diagnostic.error("WC_MASK_AND_SHAPE", "pass --mask or --shape, not both", mask=str(mask_path), shape=shape)
        )
    if mask_path is not None:
        mask_result = load_mask(mask_path)
        if mask_result.value is None:
            return Result[Path](None, mask_result.diagnostics)
        mask = mask_result.unwrap()
    elif shape is not None:
        side = max(width, height)
        mask_result = shape_mask(shape, size=side)
        if mask_result.value is None:
            return Result[Path](None, mask_result.diagnostics)
        mask = mask_result.unwrap()

    resolved = backend if backend is not None else _real_backend()
    if resolved is None:
        return Result.failure(
            Diagnostic.error(
                "WC_PACKAGE_MISSING",
                "the wordcloud package is not installed",
                fix='pip install "nlp-suite-ng[wordcloud]"',
            )
        )

    color_func: Callable[..., str] | None = None
    if group_col is not None:
        groups = _color_to_words(sub[word_col], sub[group_col])
        colors = dict(group_colors or {})
        try:
            from wordcloud import get_single_color_func
        except ImportError:
            return Result.failure(
                Diagnostic.error(
                    "WC_PACKAGE_MISSING",
                    "the wordcloud package is not installed",
                    fix='pip install "nlp-suite-ng[wordcloud]"',
                )
            )

        def color_func(word: str, **kwargs: Any) -> str:  # word is the wordcloud API shape
            for group, members in groups.items():
                if word in members:
                    return str(get_single_color_func(colors.get(group, "grey"))(word, **kwargs))
            return str(get_single_color_func("grey")(word, **kwargs))

    try:
        cloud = resolved(
            width=width,
            height=height,
            max_words=max_words,
            prefer_horizontal=prefer_horizontal,
            background_color=background_color,
            colormap=colormap,
            mask=mask,
            contour_width=contour_width if mask is not None else 0,
            contour_color=contour_color,
            color_func=color_func,
            collocations=False,
            mode="RGB",
            font_path=None,
            random_state=seed if seed is not None else 42,
        )
        cloud.generate_from_frequencies(frequencies)
    except (TypeError, ValueError, OSError) as exc:
        return Result.failure(Diagnostic.error("WC_RENDER_FAILED", f"wordcloud generation failed: {exc}"))

    output_path = Path(output_path)
    if not output_path.parent.is_dir():
        # R3: no directory creation in analysis/viz code — the caller (the
        # writer's run dir, or an explicit user path) must already exist.
        return Result.failure(
            Diagnostic.error(
                "WC_NO_OUTPUT_DIR", f"output directory does not exist: {output_path.parent}", path=str(output_path)
            )
        )
    try:
        cloud.to_file(str(output_path))
    except OSError as exc:
        return Result.failure(
            Diagnostic.error("WC_WRITE_FAILED", f"could not write {output_path}: {exc}", path=str(output_path))
        )
    return Result.success(output_path)


def gephi_gexf(
    frame: pd.DataFrame,
    source_col: str,
    target_col: str,
    weight_col: str | None = None,
) -> Result[str]:
    """GEXF 1.2 for a directed edge list."""
    if source_col not in frame.columns or target_col not in frame.columns:
        return Result.failure(
            Diagnostic.error("GEXF_BAD_COLUMN", f"columns {source_col!r}, {target_col!r} must be in frame")
        )
    if frame.empty:
        return Result.failure(Diagnostic.error("GEXF_EMPTY", "frame is empty"))

    # Collect nodes
    nodes: dict[str, int] = {}
    nid = 0
    for col in [source_col, target_col]:
        for val in frame[col].dropna().astype(str).unique():
            txt = val.strip()
            if not txt or txt.lower() in ("nan", "none"):
                continue
            if txt not in nodes:
                nodes[txt] = nid
                nid += 1

    if not nodes:
        return Result.failure(Diagnostic.error("GEXF_NO_NODES", "no nodes extracted"))

    # Build edges
    edge_lines: list[str] = []
    for eid, (_, row) in enumerate(frame.iterrows()):
        src = str(row[source_col]).strip()
        tgt = str(row[target_col]).strip()
        if src not in nodes or tgt not in nodes:
            continue
        w = 1.0
        if weight_col is not None:
            if weight_col not in frame.columns:
                return Result.failure(Diagnostic.error("GEXF_BAD_WEIGHT", f"weight column {weight_col!r} not in frame"))
            weight = _finite_weight(row[weight_col])
            w = 1.0 if weight is None else weight
        edge_lines.append(f'    <edge id="{eid}" source="{nodes[src]}" target="{nodes[tgt]}" weight="{w:.4f}" />')

    node_lines = "\n".join(
        f'    <node id="{nid}" label="{html_lib.escape(label)}" />'
        for label, nid in sorted(nodes.items(), key=lambda x: x[1])
    )
    edges = "\n".join(edge_lines)
    gexf = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<gexf xmlns="http://www.gexf.net/1.2draft" version="1.2">\n'
        '  <graph mode="static" defaultedgetype="directed">\n'
        "    <nodes>\n"
        f"{node_lines}\n"
        "    </nodes>\n"
        "    <edges>\n"
        f"{edges}\n"
        "    </edges>\n"
        "  </graph>\n"
        "</gexf>\n"
    )
    return Result.success(gexf)
