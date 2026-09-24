"""Raster wordcloud (wordcloud_image) — offline fake + real-package tests.

The legacy ``wordclouds_util.py`` port: frequencies layout, image mask,
per-group recoloring, PNG bytes. The heavy ``wordcloud`` package is an
optional extra, so offline tests inject a deterministic fake backend; the
real-package tests are marked and skip when the extra is absent.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from core.viz.wordcloud_gephi import butterfly_mask, load_mask, shape_mask, wordcloud_image


@pytest.fixture
def freq_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Word": ["america", "world", "peace", "war", "nation", "people"],
            "Count": [50, 40, 35, 30, 25, 20],
            "Group": ["A", "A", "B", "B", "C", "C"],
        }
    )


class _FakeCloud:
    """Deterministic stand-in for wordcloud.WordCloud (no bytes rendered)."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.frequencies: dict[str, float] | None = None

    def generate_from_frequencies(self, frequencies: dict[str, float]) -> None:
        self.frequencies = frequencies

    def to_file(self, path: str) -> None:
        Path(path).write_bytes(b"\x89PNG\r\n\x1a\nfake")


def _fake_backend(**kwargs: object) -> _FakeCloud:
    return _FakeCloud(**kwargs)


def test_fake_backend_writes_png_and_frequencies(freq_frame: pd.DataFrame, tmp_path: Path) -> None:
    out = tmp_path / "cloud.png"
    result = wordcloud_image(freq_frame, "Word", "Count", output_path=out, backend=_fake_backend)
    assert result.ok, result.diagnostics
    assert out.read_bytes().startswith(b"\x89PNG")
    assert result.unwrap() == out


def test_missing_backend_names_the_fix(freq_frame: pd.DataFrame, tmp_path: Path) -> None:
    result = wordcloud_image(freq_frame, "Word", "Count", output_path=tmp_path / "x.png", backend=None)
    # No injectable backend: the real package resolves or the failure names
    # the extra. On a machine with the extra this succeeds — assert both ways.
    if result.ok:
        pytest.skip("wordcloud extra installed — real backend answered")
    assert result.errors[0].code == "WC_PACKAGE_MISSING"
    assert "wordcloud" in result.errors[0].context["fix"]


def test_bad_columns_and_weights_fail_loudly(tmp_path: Path) -> None:
    empty = pd.DataFrame(columns=["Word", "Count"])
    r = wordcloud_image(empty, "Word", "Count", output_path=tmp_path / "x.png", backend=_fake_backend)
    assert r.errors[0].code == "WC_EMPTY"

    wrong = pd.DataFrame({"W": ["a"], "C": [1]})
    r2 = wordcloud_image(wrong, "Word", "Count", output_path=tmp_path / "x.png", backend=_fake_backend)
    assert r2.errors[0].code == "WC_BAD_COLUMN"

    zero = pd.DataFrame({"Word": ["a"], "Count": [0]})
    r3 = wordcloud_image(zero, "Word", "Count", output_path=tmp_path / "x.png", backend=_fake_backend)
    assert r3.errors[0].code == "WC_NO_WEIGHT"


def test_group_column_missing_fails_loudly(freq_frame: pd.DataFrame, tmp_path: Path) -> None:
    result = wordcloud_image(
        freq_frame, "Word", "Count", output_path=tmp_path / "x.png", group_col="Nope", backend=_fake_backend
    )
    assert result.errors[0].code == "WC_BAD_GROUP_COLUMN"


def test_register_artifact_contract(tmp_path: Path) -> None:
    from core.io.writer import OutputWriter

    writer = OutputWriter(tmp_path / "out", tool="test_register", params={})
    missing = writer.register_artifact("ghost.png", kind="image")
    assert missing.errors[0].code == "WRITER_REGISTER_MISSING"

    # write via the run dir the writer owns
    target = Path(writer.run_dir) / "wordcloud.png"
    target.write_bytes(b"\x89PNG")
    ok = writer.register_artifact("wordcloud.png", kind="image")
    assert ok.ok
    twice = writer.register_artifact("wordcloud.png", kind="image")
    assert twice.errors[0].code == "WRITER_REGISTERED_TWICE"

    escape = writer.register_artifact("..\\escape.png", kind="image")
    assert escape.errors[0].code in ("WRITER_ESCAPE", "WRITER_BAD_FILENAME")
    envelope = writer.finalize()
    assert envelope.ok
    kinds = [(a.kind, a.path) for a in envelope.unwrap().artifacts]
    assert ("image", "wordcloud.png") in kinds


class TestMask:
    def test_missing_mask_fails(self, tmp_path: Path) -> None:
        result = load_mask(tmp_path / "ghost.png")
        assert result.errors[0].code == "WC_MASK_MISSING"

    def test_all_white_mask_rejected(self, tmp_path: Path) -> None:
        pytest.importorskip("PIL")
        from PIL import Image

        mask_path = tmp_path / "white.png"
        Image.new("RGB", (40, 40), (255, 255, 255)).save(mask_path)
        result = load_mask(mask_path)
        assert result.errors[0].code == "WC_MASK_ALL_WHITE"

    def test_usable_mask_loads(self, tmp_path: Path) -> None:
        pytest.importorskip("PIL")
        from PIL import Image

        mask_path = tmp_path / "shape.png"
        img = Image.new("RGB", (40, 40), (255, 255, 255))
        for x in range(10, 30):
            for y in range(10, 30):
                img.putpixel((x, y), (0, 0, 0))
        img.save(mask_path)
        result = load_mask(mask_path)
        assert result.ok, result.diagnostics


@pytest.mark.model_integration
class TestRealPackage:
    """The real wordcloud package: layout, determinism, group recoloring."""

    @pytest.fixture(autouse=True)
    def _needs_extra(self) -> None:
        pytest.importorskip("wordcloud")

    def test_real_png_deterministic(self, freq_frame: pd.DataFrame, tmp_path: Path) -> None:
        first = tmp_path / "a.png"
        second = tmp_path / "b.png"
        r1 = wordcloud_image(freq_frame, "Word", "Count", output_path=first)
        r2 = wordcloud_image(freq_frame, "Word", "Count", output_path=second)
        assert r1.ok and r2.ok
        assert first.read_bytes() == second.read_bytes()  # seeded by default
        assert first.read_bytes().startswith(b"\x89PNG")

    def test_real_group_recolor(self, freq_frame: pd.DataFrame, tmp_path: Path) -> None:
        out = tmp_path / "grouped.png"
        result = wordcloud_image(
            freq_frame,
            "Word",
            "Count",
            output_path=out,
            group_col="Group",
            group_colors={"A": "red", "B": "blue", "C": "green"},
        )
        assert result.ok, result.diagnostics

    def test_real_mask_shape(self, freq_frame: pd.DataFrame, tmp_path: Path) -> None:
        from PIL import Image

        mask_path = tmp_path / "shape.png"
        img = Image.new("RGB", (200, 200), (255, 255, 255))
        for x in range(50, 150):
            for y in range(50, 150):
                img.putpixel((x, y), (0, 0, 0))
        img.save(mask_path)
        result = wordcloud_image(freq_frame, "Word", "Count", output_path=tmp_path / "m.png", mask_path=mask_path)
        assert result.ok, result.diagnostics


class TestViewerMapping:
    def test_image_kind_renders_inline(self) -> None:
        from app.scanner import artifact_view

        assert artifact_view("image", "wordcloud.png") == "image"
        assert artifact_view("image", "photo.svg") == "image"
        # unknown kind still degrades to download-only
        assert artifact_view("mystery", "thing.png") == "download"


class TestShapeMasks:
    def test_butterfly_mask_geometry(self) -> None:
        """The mask is a butterfly body: black usable region, white border."""
        pytest.importorskip("PIL")
        result = butterfly_mask(400)
        assert result.ok, result.diagnostics
        mask = result.unwrap()
        assert mask.shape == (400, 400, 3)
        usable = (mask <= 245).any(axis=2)
        usable_share = usable.mean()
        # The wing body is a real fill, not an outline: between 10% and 40%
        # of the canvas, with pure-white margins on all four borders.
        assert 0.10 < usable_share < 0.40, usable_share
        assert not usable[0, :].any() and not usable[-1, :].any()
        assert not usable[:, 0].any() and not usable[:, -1].any()
        # symmetric wings: left half and right half hold equal ink
        assert abs(usable[:, :200].sum() - usable[:, 200:].sum()) <= 400

    def test_butterfly_mask_rejects_bad_args(self) -> None:
        too_small = butterfly_mask(32)
        assert too_small.errors[0].code == "WC_SHAPE_TOO_SMALL"

    def test_shape_mask_unknown_is_loud(self) -> None:
        result = shape_mask("dragon")
        assert result.errors[0].code == "WC_SHAPE_UNKNOWN"
        assert "butterfly" in result.errors[0].message

    def test_mask_and_shape_are_mutually_exclusive(self, freq_frame: pd.DataFrame, tmp_path: Path) -> None:
        png = tmp_path / "m.png"
        png.write_bytes(b"\x89PNG")
        result = wordcloud_image(
            freq_frame,
            "Word",
            "Count",
            output_path=tmp_path / "x.png",
            mask_path=png,
            shape="butterfly",
        )
        assert result.errors[0].code == "WC_MASK_AND_SHAPE"

    def test_shape_reaches_the_render_kwargs(self, freq_frame: pd.DataFrame, tmp_path: Path) -> None:
        """--shape butterfly routes the generated mask into the backend."""

        captured: dict[str, object] = {}

        class _RecordingFake(_FakeCloud):
            def __init__(self, **kwargs: object) -> None:
                super().__init__(**kwargs)
                captured.update(kwargs)

        result = wordcloud_image(
            freq_frame,
            "Word",
            "Count",
            output_path=tmp_path / "x.png",
            shape="butterfly",
            backend=_RecordingFake,
        )
        assert result.ok, result.diagnostics
        assert captured["mask"] is not None

    def test_shape_unreachable_without_pil(self, freq_frame: pd.DataFrame, tmp_path: Path, monkeypatch) -> None:
        """No Pillow -> the shape path fails loudly, not with a traceback."""
        import builtins

        real_import = builtins.__import__

        def blocked(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
            if name == "PIL":
                raise ImportError("PIL blocked for test")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)
        result = wordcloud_image(freq_frame, "Word", "Count", output_path=tmp_path / "x.png", shape="butterfly")
        assert result.errors[0].code == "WC_SHAPE_NO_PIL"
