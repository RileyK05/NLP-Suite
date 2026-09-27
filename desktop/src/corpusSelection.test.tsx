import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import type { Document } from "./api";
import { CorpusScope } from "./CorpusScope";
import {
  allDocuments,
  scopeFilters,
  selectedDocuments,
  selectionError,
} from "./corpusSelection";

const docs: Document[] = [
  { id: "a", name: "a.txt", document_date: "2020-01-01", words: 10 },
  { id: "b", name: "b.txt", document_date: "2021-01-01", words: 20 },
  { id: "c", name: "c.txt", document_date: null, words: 30 },
].map((doc) => ({ stored_name: doc.name, bytes: 100, sha256: doc.id, ...doc }));

describe("the scope the user sees before running", () => {
  it("starts with all documents including undated ones", () => {
    expect(selectedDocuments(docs, allDocuments())).toEqual(docs);
    expect(selectionError(docs, allDocuments())).toBeNull();
  });
  it("intersects explicit choices with inclusive dates", () => {
    const selection = {
      ...allDocuments(),
      document_ids: ["a", "b"],
      date_from: "2021-01-01",
      date_to: "2021-01-01",
    };
    expect(selectedDocuments(docs, selection).map((doc) => doc.id)).toEqual([
      "b",
    ]);
  });
  it("requires an explicit choice to include undated files in a window", () => {
    const selection = { ...allDocuments(), date_to: "2020-12-31" };
    expect(selectedDocuments(docs, selection).map((doc) => doc.id)).toEqual([
      "a",
    ]);
    expect(
      selectedDocuments(docs, { ...selection, include_undated: true }).map(
        (doc) => doc.id,
      ),
    ).toEqual(["a", "c"]);
  });
  it("never turns an empty selection into the whole corpus", () => {
    expect(
      selectedDocuments(docs, { ...allDocuments(), document_ids: [] }),
    ).toEqual([]);
    expect(
      selectionError(docs, { ...allDocuments(), document_ids: [] }),
    ).toMatch(/No documents/);
  });
  it("keeps documents whose detail is one of the chosen values", () => {
    const detailed = docs.map((doc, index) => ({
      ...doc,
      fields: (index === 2
        ? {}
        : { Kind: index === 0 ? "sotu" : "ina" }) as Record<string, string>,
    }));
    const selection = {
      ...allDocuments(),
      fields: { Kind: ["sotu"] },
    };
    expect(selectedDocuments(detailed, selection).map((doc) => doc.id)).toEqual(
      ["a"],
    );
    expect(
      selectedDocuments(detailed, {
        ...allDocuments(),
        fields: { Kind: ["(empty)"] },
      }).map((doc) => doc.id),
    ).toEqual(["c"]);
  });
  it("windows an order axis the same way it windows dates", () => {
    const chapters = docs.map((doc, index) => ({
      ...doc,
      document_order: index === 2 ? null : index + 1,
    }));
    const window = { ...allDocuments(), order_from: 2, order_to: 3 };
    expect(selectedDocuments(chapters, window).map((doc) => doc.id)).toEqual([
      "b",
    ]);
    expect(
      selectedDocuments(chapters, { ...window, include_unordered: true }).map(
        (doc) => doc.id,
      ),
    ).toEqual(["b", "c"]);
    expect(
      selectionError(chapters, {
        ...allDocuments(),
        order_from: 3,
        order_to: 2,
      }),
    ).toMatch(/first order/);
    expect(
      selectionError(chapters, { ...allDocuments(), order_from: NaN }),
    ).toMatch(/real numbers/);
  });
  it("a run's record says which chapters and details it was narrowed to", () => {
    expect(
      scopeFilters({
        order_from: 2,
        order_to: 4,
        include_unordered: false,
        fields: { Kind: ["sotu", "ina"] },
      }),
    ).toBe(
      " Order 2 through 4, inclusive. Documents with no order excluded." +
        " Only documents whose Kind is sotu or ina.",
    );
    expect(scopeFilters({ order_from: 3 })).toContain("Order 3 and later.");
    // Runs recorded before these filters existed carry none of the keys.
    expect(scopeFilters({})).toBe("");
  });
  it("rejects reversed, invalid and stale selections", () => {
    expect(
      selectionError(docs, {
        ...allDocuments(),
        date_from: "2021-01-01",
        date_to: "2020-01-01",
      }),
    ).toMatch(/start date/);
    expect(
      selectionError(docs, { ...allDocuments(), date_from: "2021-02-29" }),
    ).toMatch(/valid calendar/);
    expect(
      selectionError(docs, { ...allDocuments(), document_ids: ["gone"] }),
    ).toMatch(/list changed/);
  });
  it("shows scope totals and explains the date source", () => {
    const markup = renderToStaticMarkup(
      <CorpusScope
        documents={docs}
        value={{ ...allDocuments(), date_to: "2020-12-31" }}
        onChange={() => {}}
      />,
    );
    expect(markup).toContain("1 of 3 documents");
    expect(markup).toContain("10 words");
    expect(markup).toContain("never the import date");
    expect(markup).toContain("Also include undated documents");
  });
  it("offers a detail filter and an order window when the corpus has them", () => {
    const detailed = docs.map((doc, index) => ({
      ...doc,
      fields: {
        Kind: index === 0 ? "sotu" : "ina",
        Party: index === 0 ? "Democratic" : "Republican",
        Date: "2020-01-01",
        Order: String(index + 1),
      },
      document_order: index === 2 ? null : index + 1,
    }));
    const markup = renderToStaticMarkup(
      <CorpusScope
        documents={detailed}
        value={{
          ...allDocuments(),
          fields: { Kind: ["sotu"] },
          order_from: 2,
          order_to: 3,
          include_unordered: true,
        }}
        onChange={() => {}}
      />,
    );
    // One chip per chosen detail, and a way to add the details not yet used.
    expect(markup).toContain("Remove the Kind filter");
    expect(markup).toContain("Keep only documents whose detail is");
    expect(markup).toContain("From order (inclusive)");
    expect(markup).toContain("Also include the");
    // Date and Order have their own windows, never a value checklist.
    expect(markup).toContain('<option value="Party">');
    expect(markup).not.toContain('<option value="Date">');
    expect(markup).not.toContain('<option value="Order">');
  });
});
