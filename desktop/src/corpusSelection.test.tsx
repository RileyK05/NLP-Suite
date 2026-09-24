import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import type { Document } from "./api";
import { CorpusScope } from "./CorpusScope";
import {
  allDocuments,
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
});
