"""NER movement tracks — person-location co-occurrence trails (CAP-NER-03).

Legacy reference: ``NER_location_tracking_util.py`` — sentences that mention
both a PERSON and a LOCATION produce (person, place) pairs; the output CSV
feeds an animated movement map. Ours builds on the F6 whole-mention spans in
:func:`core.analysis.ner._mentions_from_spans`: for each sentence, every
PERSON mention paired with every location mention of that sentence, keeping
the document, sentence id, and a stable Sentence column for provenance.

Geocoding the *unique* locations is optional (``geocode=True``): the offline
baked KB resolves well-known cities; unmatched places stay in the pairs
frame with empty coordinates and an INFO diagnostic, so the movement table
never fabricates coordinates.
"""

from __future__ import annotations

import pandas as pd

from core.conll.schema import Col, validate_columns
from core.result import Diagnostic, Result

__all__ = ["movement_tracks"]

_PERSON_TAGS = frozenset({"PERSON", "PER"})
_LOCATION_TAGS = frozenset({"LOC", "GPE", "LOCATION", "FAC", "FACILITY", "STATE_OR_PROVINCE", "COUNTRY", "CITY"})
_PAIR_COLUMNS = ["Entity", "Location", "Sentence ID", "Document ID", "Document", "Sentence"]
_SUMMARY_COLUMNS = ["Entity", "Location", "Count", "First Sentence", "Documents"]


def movement_tracks(
    frame: pd.DataFrame,
    *,
    geocode: bool = False,
) -> Result[pd.DataFrame]:
    """One row per (PERSON mention, location mention) within a sentence.

    Returns the pair table; a per-(entity, location) summary is derivable by
    grouping, and the geocoded variant carries ``Lat``/``Lon`` columns when
    the offline KB knows the place.
    """
    checked = validate_columns([str(c) for c in frame.columns])
    if not checked.ok:
        return Result[pd.DataFrame](None, checked.diagnostics)
    for need in (Col.FORM.value, Col.NER.value, Col.SENTENCE_ID.value, Col.DOCUMENT_ID.value):
        if need not in frame.columns:
            return Result.failure(Diagnostic.error("NER_MISSING_COLUMN", f"missing {need!r}"))
    if frame.empty:
        return Result.success(pd.DataFrame(columns=_PAIR_COLUMNS))

    from core.analysis.ner import _mentions_from_spans

    doc_col = Col.DOCUMENT.value if Col.DOCUMENT.value in frame.columns else None
    rows: list[dict[str, object]] = []
    diags: list[Diagnostic] = []
    for did, g in frame.groupby(Col.DOCUMENT_ID.value, sort=False):
        doc = str(g[doc_col].iloc[0]) if doc_col is not None else ""
        for sid, sent_group in g.groupby(Col.SENTENCE_ID.value, sort=False):
            # Mentions are spans; tag text for a sentence comes from its span
            # (first token's tag for a multiword span).
            mentions = _mentions_from_spans(sent_group)
            persons = [entity for (entity, tag) in mentions if tag in _PERSON_TAGS]
            places = [entity for (entity, tag) in mentions if tag in _LOCATION_TAGS]
            if not persons or not places:
                continue
            for person in persons:
                for place in places:
                    rows.append(
                        {
                            "Entity": person,
                            "Location": place,
                            "Sentence ID": sid,
                            "Document ID": str(did),
                            "Document": doc,
                            "Sentence": " ".join(str(v) for v in sent_group[Col.FORM.value].tolist()),
                        }
                    )
    if not rows:
        return Result.success(
            pd.DataFrame(columns=_PAIR_COLUMNS),
            Diagnostic.info(
                "NER_NO_MOVEMENT",
                "no sentence mentions both a person and a location; movement tracks empty",
            ),
        )
    pairs = pd.DataFrame(rows, columns=_PAIR_COLUMNS)

    if geocode:
        from core.gis.geocode import geocode_many

        unique_places = sorted({str(p) for p in pairs["Location"]})
        geo = geocode_many(unique_places)
        # Per-place failure is expected (offline KB); unknown -> empty coords.
        coords = {}
        if geo.value is not None:
            geo_frame = geo.unwrap()
            ok = geo_frame[geo_frame["Status"] == "OK"] if "Status" in geo_frame.columns else geo_frame
            for place, lat, lon in zip(ok["Place"], ok["Lat"], ok["Lon"], strict=True):
                coords[str(place).lower()] = (float(lat), float(lon))
            diags.extend(geo.diagnostics)
        lat_list: list[float] = []
        lon_list: list[float] = []
        unknown: list[str] = []
        for place in pairs["Location"].astype(str):
            hit = coords.get(place.lower())
            if hit is None:
                # NOT 0.0: (0, 0) is Null Island, a real point in the Gulf of
                # Guinea. Writing it would plot every unresolved place there on
                # any map or Earth tour built from this frame. NaN is skipped by
                # the KML writers and reads as missing everywhere else.
                lat_list.append(float("nan"))
                lon_list.append(float("nan"))
                if place not in unknown:
                    unknown.append(place)
            else:
                lat_list.append(hit[0])
                lon_list.append(hit[1])
        pairs["Lat"] = lat_list
        pairs["Lon"] = lon_list
        if unknown:
            diags.append(
                Diagnostic.info(
                    "NER_PLACES_UNGEOCODED",
                    f"{len(unknown)} location(s) not in the offline geocoder KB: {', '.join(unknown[:10])}"
                    + (", ..." if len(unknown) > 10 else ""),
                )
            )
    # Location completes the key: without it, two places in one sentence tie
    # and their order rides on frame construction rather than on the data.
    pairs = pairs.sort_values(["Document ID", "Entity", "Sentence ID", "Location"], kind="stable").reset_index(
        drop=True
    )
    return Result.success(pairs, *diags)


def movement_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    """Per-(entity, location) counts over a movement_tracks frame."""
    if pairs.empty:
        return pd.DataFrame(columns=_SUMMARY_COLUMNS)
    summary = (
        pairs.groupby(["Entity", "Location"], as_index=False)
        .agg(Count=("Sentence ID", "size"), First_Sentence=("Sentence ID", "min"))
        .sort_values(["Entity", "First_Sentence", "Location"], kind="stable")
        .reset_index(drop=True)
    )
    docs = pairs.groupby(["Entity", "Location"])["Document"].apply(lambda s: ", ".join(sorted(set(s.astype(str)))))
    summary["Documents"] = summary.set_index(["Entity", "Location"]).index.map(docs)
    summary = summary.rename(columns={"First_Sentence": "First Sentence"})
    return summary[_SUMMARY_COLUMNS]
