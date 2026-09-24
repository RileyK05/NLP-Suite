import { describe, expect, it } from "vitest";
import {
  barWidth,
  isDraftDifferent,
  percent,
  phraseRate,
  rateComparison,
  refined,
  type PhraseRequest,
} from "./phrase";

describe("phrase research formatting", () => {
  it("keeps no eligible text distinct from a zero rate", () => {
    expect(phraseRate(null)).toBe("No text");
    expect(phraseRate(0)).toBe("0");
  });

  it("does not draw rates outside the available bar", () => {
    expect(barWidth(null, 10)).toBe("0%");
    expect(barWidth(5, 10)).toBe("50%");
    expect(barWidth(20, 10)).toBe("100%");
  });

  it("labels document prevalence as a percentage", () => {
    expect(percent(null)).toBe("—");
    expect(percent(12.34)).toBe("12.3%");
  });

  it("describes a comparison without dividing by zero", () => {
    expect(rateComparison(20, 10)).toContain("2×");
    expect(rateComparison(5, 0)).toContain("ratio is undefined");
    expect(rateComparison(null, 2)).toContain("not enough eligible text");
  });
});

describe("the question on screen versus the question in the form", () => {
  const resolved: PhraseRequest = {
    text: "public health",
    snapshot_id: "snap-1",
    case_sensitive: false,
    normalize: false,
    match_lemma: false,
    match_nominalization: false,
    position_bins: 10,
    evidence_offset: 50,
    evidence_limit: 50,
    evidence_year: 1994,
    evidence_document_id: null,
  };
  const draft = {
    text: "public health",
    case_sensitive: false,
    normalize: false,
    match_lemma: false,
    match_nominalization: false,
    position_bins: 10,
  };

  it("sees no difference before anything is typed", () => {
    expect(isDraftDifferent(draft, "", resolved, null)).toBe(false);
  });

  it("notices a new phrase that has not been submitted", () => {
    // The case that made drilling dangerous: type a second phrase, click a
    // year, and the year request ran the phrase nobody had asked for.
    expect(
      isDraftDifferent(
        { ...draft, text: "private health" },
        "",
        resolved,
        null,
      ),
    ).toBe(true);
  });

  it("ignores whitespace nobody meant to type", () => {
    expect(
      isDraftDifferent(
        { ...draft, text: "  public health  " },
        "",
        resolved,
        null,
      ),
    ).toBe(false);
  });

  it("notices a changed setting, not only a changed phrase", () => {
    expect(
      isDraftDifferent({ ...draft, case_sensitive: true }, "", resolved, null),
    ).toBe(true);
    expect(
      isDraftDifferent({ ...draft, position_bins: 20 }, "", resolved, null),
    ).toBe(true);
  });

  it("notices a comparison phrase appearing or disappearing", () => {
    expect(isDraftDifferent(draft, "climate", resolved, null)).toBe(true);
    expect(isDraftDifferent(draft, "", resolved, "climate")).toBe(true);
    expect(isDraftDifferent(draft, "climate", resolved, "climate")).toBe(false);
  });

  it("has nothing to disagree with before a question is resolved", () => {
    expect(isDraftDifferent(draft, "", null, null)).toBe(false);
  });
});

describe("looking at different evidence for the same question", () => {
  const resolved: PhraseRequest = {
    text: "public health",
    snapshot_id: "snap-1",
    case_sensitive: true,
    normalize: false,
    match_lemma: false,
    match_nominalization: false,
    position_bins: 20,
    evidence_offset: 50,
    evidence_limit: 50,
    evidence_year: 1994,
    evidence_document_id: null,
  };

  it("moves the window and changes nothing else", () => {
    const next = refined(resolved, { evidence_offset: 100 });
    expect(next.evidence_offset).toBe(100);
    expect(next.text).toBe("public health");
    expect(next.case_sensitive).toBe(true);
    expect(next.position_bins).toBe(20);
  });

  it("keeps the snapshot, so a reload underneath is refused not answered", () => {
    expect(refined(resolved, { evidence_offset: 100 }).snapshot_id).toBe(
      "snap-1",
    );
  });

  it("can clear a filter without clearing the subject", () => {
    const cleared = refined(resolved, {
      evidence_offset: 0,
      evidence_year: null,
      evidence_document_id: null,
    });
    expect(cleared.evidence_year).toBeNull();
    expect(cleared.text).toBe("public health");
  });
});
