import { describe, expect, it } from "vitest";
import {
  FIELD_VALUE_PREFIX,
  detailChoiceValues,
  detailSuggestions,
} from "./ParamFields";

const details = {
  names: [
    { name: "Date", count: 3, distinct: 3, sources: [], sample_values: [] },
    { name: "Kind", count: 3, distinct: 2, sources: [], sample_values: [] },
    { name: "Speaker", count: 3, distinct: 2, sources: [], sample_values: [] },
  ],
  documents: {
    a: {
      Kind: { value: "sotu", source: "filename", overridden: [] },
      Speaker: { value: "Truman", source: "filename", overridden: [] },
    },
    b: {
      Kind: { value: "ina", source: "filename", overridden: [] },
      Speaker: { value: "Truman", source: "filename", overridden: [] },
    },
    c: { Kind: { value: "sotu", source: "user", overridden: [] } },
  },
} as Parameters<typeof detailSuggestions>[0];

describe("what the detail parameters offer", () => {
  it("offers the project's details, and the chosen detail's values", () => {
    const offered = detailSuggestions(details, { "group-field": "kind" });
    expect(offered["group-field"]).toEqual(["Kind", "Speaker"]);
    expect(offered["group-a"]).toEqual(["ina", "sotu"]);
    expect(offered["group-b"]).toEqual(["ina", "sotu"]);
  });

  it("offers no values before a detail is chosen, and nothing without details", () => {
    expect(detailSuggestions(details, {})["group-a"]).toEqual([]);
    expect(detailSuggestions(null, {})).toEqual({});
  });

  it("offers a detail-taking choice one value per detail, named field:<name>", () => {
    // "Count along" can say `field:Party` without this file knowing which
    // details a corpus has (0.5.0 plan 1.6). Date is left out: year and
    // decade already count along it.
    const param = {
      choices: ["year", "decade"],
      choice_labels: undefined,
      details: true,
    };
    expect(
      detailChoiceValues(param, details).map((option) => option.value),
    ).toEqual([
      "year",
      "decade",
      `${FIELD_VALUE_PREFIX}Kind`,
      `${FIELD_VALUE_PREFIX}Speaker`,
    ]);
    expect(
      detailChoiceValues(param, details).map((option) => option.label),
    ).toEqual(["year", "decade", "Kind", "Speaker"]);
    expect(
      detailChoiceValues(param, null).map((option) => option.value),
    ).toEqual(["year", "decade"]);
    expect(
      detailChoiceValues({ ...param, details: false }, details),
    ).toHaveLength(2);
  });
});
