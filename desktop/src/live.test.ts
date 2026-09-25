import { describe, expect, it } from "vitest";
import {
  COLD,
  REQUEST_TIMEOUT_MS,
  isCurrent,
  isDirty,
  liveTools,
  missingRequired,
  questionKey,
  readyNote,
  requestId,
  requestIsLost,
  warmingNote,
  type LiveState,
} from "./live";
import type { Tool } from "./api";

const tool = (over: Partial<Tool> = {}): Tool => ({
  name: "ngrams",
  label: "N-grams",
  description: "Repeated word sequences",
  requires_parse: true,
  params: [],
  input_kind: "text",
  outputs: ["ngrams.csv"],
  category: "analysis",
  availability: { state: "available", message: "", missing: [] },
  ...over,
});

const state = (over: Partial<LiveState> = {}): LiveState => ({
  ...COLD,
  ...over,
});

describe("what the live bench will run", () => {
  it("offers ordinary analyses", () => {
    expect(liveTools([tool()]).map((t) => t.name)).toEqual(["ngrams"]);
  });

  it("leaves out visualisations, which are a publishing step", () => {
    const chart = tool({ name: "charts", category: "visualization" });
    expect(liveTools([tool(), chart]).map((t) => t.name)).toEqual(["ngrams"]);
  });

  it("leaves out table tools, which need a table and not a corpus", () => {
    const table = tool({ name: "table_keyness" });
    expect(liveTools([tool(), table]).map((t) => t.name)).toEqual(["ngrams"]);
  });

  it("leaves out anything whose model is not installed", () => {
    // Offering it would mean choosing it, waiting, and being told no.
    const broken = tool({
      name: "topic_model",
      availability: {
        state: "needs_setup",
        message: "install it",
        missing: [],
      },
    });
    expect(liveTools([tool(), broken]).map((t) => t.name)).toEqual(["ngrams"]);
  });

  it("offers the BERT tools once their model is here, and not before", () => {
    // The models ship with the app now, so the engine reports them available.
    const ready = ["word_sense_induction", "bert_extract", "bert_topics", "sentiment_neural_bert", "word2vec_bert"].map(
      (name) => tool({ name, availability: { state: "available", message: "", missing: [] } }),
    );
    expect(liveTools(ready).map((t) => t.name)).toEqual(ready.map((t) => t.name));
    const waiting = tool({
      name: "bert_topics",
      availability: {
        state: "needs_model",
        message: "This needs BERT base (uncased) (105 MB). Download it from Models.",
        missing: [],
        model_id: "bert-base-uncased",
      },
    });
    expect(liveTools([tool(), waiting]).map((t) => t.name)).toEqual(["ngrams"]);
  });
});

describe("knowing which answer belongs to which question", () => {
  it("is the same question however the parameters are ordered", () => {
    // Otherwise re-rendering with a differently-ordered object would look like
    // a new question and cost a second request for the same answer.
    expect(questionKey("ngrams", { n: 2, field: "form" })).toBe(
      questionKey("ngrams", { field: "form", n: 2 }),
    );
  });

  it("changes when a parameter changes", () => {
    expect(questionKey("ngrams", { n: 2 })).not.toBe(
      questionKey("ngrams", { n: 3 }),
    );
  });

  it("changes when the analysis changes", () => {
    expect(questionKey("ngrams", { n: 2 })).not.toBe(
      questionKey("collocations", { n: 2 }),
    );
  });

  it("tells a stale answer from a current one", () => {
    // Responses do not arrive in the order they were sent. Letting go of a
    // slider at 30 must not leave the chart showing 12 because 12 was slower.
    const asked = questionKey("ngrams", { n: 30 });
    expect(isCurrent(questionKey("ngrams", { n: 30 }), asked)).toBe(true);
    expect(isCurrent(questionKey("ngrams", { n: 12 }), asked)).toBe(false);
  });
});

describe("whether the bench may ask without being asked (R-C1)", () => {
  it("marks the bench dirty when the controls leave the shown answer", () => {
    // "Dirty" is the whole of the iteration loop: change a setting, see that
    // it is not yet reflected, press Run when ready. Nothing is scheduled.
    const shown = questionKey("ngrams", { n: 2 });
    const asked = questionKey("ngrams", { n: 3 });
    expect(isDirty(shown, asked)).toBe(true);
  });

  it("is not dirty when the answer on show answers the controls", () => {
    // Re-running the same key is allowed and encouraged -- "run it twice and
    // check the same topics come back" is homework -- but it is not a change.
    const key = questionKey("ngrams", { n: 2 });
    expect(isDirty(key, key)).toBe(false);
  });

  it("is not dirty before anything can be asked", () => {
    // No analysis chosen, or a required field is empty: no question exists,
    // so there is nothing to run and nothing to promise.
    expect(isDirty("", "")).toBe(false);
    expect(isDirty(questionKey("ngrams", { n: 2 }), "")).toBe(false);
  });

  it("gives every request an identity the engine can echo (R-C2)", () => {
    // Two presses must not collide: a dropped answer is matched to its
    // question by this id, not by guessing which run it belonged to.
    expect(requestId()).not.toBe(requestId());
    expect(requestId().length).toBeGreaterThanOrEqual(8);
  });
});

describe("saying what is happening during the slow step", () => {
  it("says nothing when nothing is warming", () => {
    expect(warmingNote(state({ state: "ready" }))).toBe("");
  });

  it("estimates the wait from the size of the corpus", () => {
    const note = warmingNote(
      state({
        state: "warming",
        stage: "Parsing",
        selection: { documents: 20, words: 110_000, names: [], ids: [] },
      }),
    );
    expect(note).toContain("Parsing");
    expect(note).toMatch(/about \d+s/);
    expect(note).toContain("110,000 words");
    expect(note).toContain("once");
  });

  it("still says something before the size is known", () => {
    expect(warmingNote(state({ state: "warming" }))).toBeTruthy();
  });

  it("still counts the parser load on a corpus of a dozen words", () => {
    // Almost all of that wait is loading the model, so promising "about 2s"
    // for twelve words would be a smaller lie than the old one but a lie.
    const note = warmingNote(
      state({
        state: "warming",
        selection: { documents: 1, words: 12, names: [], ids: [] },
      }),
    );
    expect(note).toContain("about 5s");
  });

  it("estimates a large corpus from what a large corpus measured", () => {
    // 516,987 words parsed in 78.4 s. Treating the cost as pure proportion
    // predicted 141 s and made the one slow step look twice as expensive.
    const note = warmingNote(
      state({
        state: "warming",
        selection: { documents: 87, words: 516_987, names: [], ids: [] },
      }),
    );
    const seconds = Number(/about (\d+)s/.exec(note)?.[1]);
    expect(seconds).toBeGreaterThan(60);
    expect(seconds).toBeLessThan(100);
  });
});

describe("saying what the wait bought", () => {
  it("reports a fresh parse with the time it took", () => {
    const note = readyNote(
      state({
        state: "ready",
        documents: 20,
        tokens: 128731,
        source: "parsed",
        parse_ms: 30620,
      }),
    );
    expect(note).toContain("20 documents");
    expect(note).toContain("128,731 words");
    expect(note).toContain("30.6s");
  });

  it("says when the parse came back from an earlier one", () => {
    // The distinction people care about: this time it was instant.
    expect(
      readyNote(
        state({ state: "ready", documents: 2, tokens: 60, source: "disk" }),
      ),
    ).toContain("recovered");
  });

  it("says when it never left", () => {
    expect(
      readyNote(
        state({ state: "ready", documents: 2, tokens: 60, source: "memory" }),
      ),
    ).toContain("already in hand");
  });

  it("says nothing about a corpus that is not ready", () => {
    expect(readyNote(state({ state: "warming" }))).toBe("");
  });

  it("counts one document as one document", () => {
    expect(
      readyNote(
        state({ state: "ready", documents: 1, tokens: 5, source: "disk" }),
      ),
    ).toContain("1 document,");
  });
});

describe("a request that never answers is treated as lost", () => {
  it("keeps waiting well beyond the slowest real analysis", () => {
    // The slowest observed live analysis over the real corpus is the topic
    // model at about half a minute; three minutes is patience, not a timeout
    // race against real work.
    expect(REQUEST_TIMEOUT_MS).toBeGreaterThanOrEqual(120_000);
  });

  it("does not declare a normal analysis lost", () => {
    expect(requestIsLost(30_000)).toBe(false);
  });

  it("declares a wedged request lost, freeing the single slot", () => {
    // A dropped fetch used to hold the slot forever: the bench read
    // "Working…" to nobody, and no parameter change could revive it.
    expect(requestIsLost(REQUEST_TIMEOUT_MS + 1)).toBe(true);
  });
});

describe("analyses that cannot run yet", () => {
  const withParams = (over: Partial<Tool["params"][number]>[]) =>
    tool({
      params: over.map((p) => ({
        name: "terms",
        label: "Word groups",
        type: "str",
        default: null,
        required: true,
        help: "",
        choices: [],
        minimum: null,
        maximum: null,
        ...p,
      })),
    });

  it("names an empty required field instead of asking", () => {
    // Word groups with no words has nothing to compute. Sending it spends a
    // request to be told so and answers with an error about the very thing
    // the reader was on their way to type.
    expect(missingRequired(withParams([{}]), {})).toEqual(["Word groups"]);
  });

  it("treats whitespace as empty", () => {
    expect(missingRequired(withParams([{}]), { terms: "   " })).toEqual([
      "Word groups",
    ]);
  });

  it("is satisfied once the field has something in it", () => {
    expect(missingRequired(withParams([{}]), { terms: "Iraq: iraq" })).toEqual(
      [],
    );
  });

  it("ignores fields the analysis can default", () => {
    expect(
      missingRequired(withParams([{ name: "by", required: false }]), {}),
    ).toEqual([]);
  });

  it("says nothing about an analysis that needs nothing", () => {
    expect(missingRequired(tool(), {})).toEqual([]);
  });
});
