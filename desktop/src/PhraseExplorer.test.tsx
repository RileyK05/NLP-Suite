import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { PhraseExplorer } from "./PhraseExplorer";
import type { SavedPhraseQuestion } from "./phrase";

const saved: SavedPhraseQuestion = {
  id: "question-1",
  name: "Public and private health",
  snapshot_id: "a".repeat(64),
  parser: "spacy",
  document_ids: ["document-1"],
  specification: {
    schema_version: 1,
    kind: "phrase_distribution",
    text: "public health",
    comparison_text: "private health",
    case_sensitive: false,
    normalize: false,
    match_lemma: false,
    match_nominalization: false,
    position_bins: 10,
    evidence_year: null,
    evidence_document_id: null,
  },
  matching: {
    version: 2,
    tokenizer: "spacy/en_core_web_sm",
    source: "snapshot",
    tokens: ["public", "health"],
    comparison_tokens: ["private", "health"],
  },
  revision: 2,
  created: "2026-09-20T00:00:00+00:00",
  updated: "2026-09-20T00:01:00+00:00",
};

describe("phrase research workspace", () => {
  it("offers direct comparison and persistent questions before an answer exists", () => {
    const html = renderToStaticMarkup(
      <PhraseExplorer
        ready
        onDuplicateQuestion={async () => null}
        onTrack={async () => null}
        onReadSource={async () => null}
        savedQuestions={[saved]}
        onSaveQuestion={async () => null}
        onOpenQuestion={async () => null}
        onDeleteQuestion={async () => false}
        onPublishQuestion={() => undefined}
        canPublish
      />,
    );
    expect(html).toContain("Compare with");
    expect(html).toContain("Public and private health · revision 2");
    expect(html).toContain("Track phrase");
    expect(html).toContain("Publish saved");
  });
});
