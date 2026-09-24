"""C47-C52 - PC-ACE analysis, GIS, profiler, narrative, install."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from conftest import has_spacy_model
from core.conll.schema import Col
from core.gis import geocode as geo_mod, mapping as map_mod
from core.install import check as install_mod
from core.io.reader import read_corpus
from core.narrative import arcs as arc_mod
from core.pcace import analysis as pcace_a_mod, core as pcace_mod
from core.pipelines.cache import PipelineCache
from core.pipelines.spacy_backend import build_spacy_pipeline, spacy_model_name
from core.profiler import profiler as prof_mod

FIXTURE = Path(__file__).parent / "fixtures" / "mini-corpus"


def _parsed_frame() -> pd.DataFrame:
    if not has_spacy_model():
        pytest.skip(f"spaCy model not installed — run: python -m spacy download {spacy_model_name('en')}")
    result = read_corpus(FIXTURE)
    assert result.ok
    corpus = result.unwrap()
    cache = PipelineCache()
    cache.register("spacy", build_spacy_pipeline)  # type: ignore[arg-type]
    return cache.get("spacy", "en").unwrap().parse(corpus).unwrap()


def test_pcace_analysis(tmp_path: Path) -> None:
    db = tmp_path / "pcace.db"
    assert pcace_mod.create_db(db, ["1.1", "1.2", "2.1"]).ok
    ana = pcace_a_mod.analyze(db)
    assert ana.ok
    adf = ana.unwrap()
    assert "Category" in adf.columns
    # Government appears once, etc.
    assert len(adf) >= 2

    val = pcace_a_mod.validate_codes(["1.1", "bad code !", ""])
    assert val.ok  # returns df with Status, but diagnostics for bad
    vdf = val.unwrap()
    assert len(vdf) == 3
    assert set(vdf["Status"]) == {"OK", "INVALID"}
    assert val.diagnostics  # has errors for bad

    empty = pcace_a_mod.validate_codes([])
    assert not empty.ok

    bad_db = pcace_a_mod.analyze(tmp_path / "no.db")
    assert not bad_db.ok


def test_geocode() -> None:
    res = geo_mod.geocode("Paris")
    assert res.ok
    lat, lon = res.unwrap()
    assert abs(lat - 48.8566) < 0.01
    assert abs(lon - 2.3522) < 0.01

    unknown = geo_mod.geocode("Atlantis")
    assert not unknown.ok

    many = geo_mod.geocode_many(["Paris", "London", "Atlantis"])
    assert many.ok
    mdf = many.unwrap()
    assert len(mdf) == 3
    assert many.diagnostics  # unknown

    dist = geo_mod.distance_places("Paris", "London")
    assert dist.ok
    # Approx 344 km
    assert 300 < dist.unwrap() < 400

    bad = geo_mod.distance_places("Paris", "Atlantis")
    assert not bad.ok

    empty = geo_mod.geocode_many([])
    assert not empty.ok


def test_mapping() -> None:
    df = pd.DataFrame({"Place": ["Paris", "London"], "Lat": [48.86, 51.51], "Lon": [2.35, -0.12]})
    k = map_mod.kml(df, lat_col="Lat", lon_col="Lon", name_col="Place")
    assert k.ok
    assert "<?xml" in k.unwrap()
    assert "Paris" in k.unwrap()

    h = map_mod.heatmap_html(df, lat_col="Lat", lon_col="Lon", title="Test")
    assert h.ok
    assert "48.86" in h.unwrap()

    empty = pd.DataFrame(columns=["Lat", "Lon", "Place"])
    assert not map_mod.kml(empty).ok
    assert not map_mod.heatmap_html(empty).ok


@pytest.mark.model_integration
def test_profiler() -> None:
    frame = _parsed_frame()
    res = prof_mod.profile(frame)
    assert res.ok
    pdf = res.unwrap()
    assert len(pdf) == 3  # three docs
    assert "Tokens" in pdf.columns or "Document ID" in pdf.columns

    empty = pd.DataFrame(
        columns=[
            "ID",
            "Form",
            "Lemma",
            "POS",
            "NER",
            "Head",
            "DepRel",
            "Sentence ID",
            "Document ID",
            "Document",
            "Deps",
            "Record ID",
            "Clause Tag",
        ]
    )
    assert not prof_mod.profile(empty).ok


@pytest.mark.model_integration
def test_narrative() -> None:
    frame = _parsed_frame()
    e = arc_mod.emotion_arc(frame, field=Col.FORM)
    assert e.ok
    edf = e.unwrap()
    # One row per sentence
    assert len(edf) >= 1
    assert "Compound" in edf.columns

    length_res = arc_mod.length_arc(frame)
    assert length_res.ok
    ldf = length_res.unwrap()
    assert "Tokens" in ldf.columns
    assert len(ldf) >= 1

    empty = pd.DataFrame(
        columns=[
            "ID",
            "Form",
            "Lemma",
            "POS",
            "NER",
            "Head",
            "DepRel",
            "Sentence ID",
            "Document ID",
            "Document",
            "Deps",
            "Record ID",
            "Clause Tag",
        ]
    )
    assert arc_mod.emotion_arc(empty).ok
    assert arc_mod.length_arc(empty).ok


def test_install() -> None:
    res = install_mod.check_environment()
    assert res.ok
    df = res.unwrap()
    assert len(df) > 0
    assert "Extra" in df.columns
    assert "Installed" in df.columns
    # At least pandas should be installed (we are running)
    # We don't assert specific extras, just shape
    assert df["Python"].iloc[0].count(".") >= 1
