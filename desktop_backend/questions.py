"""Validated, persistent research questions for the interactive workspace."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator

MAX_QUESTION_NAME = 120
MAX_QUESTIONS_PER_PROJECT = 500
#: Mirrors core.research.phrase.MAX_PHRASE_TOKENS; a saved question cannot
#: record more resolved tokens than a question is allowed to have.
MAX_PHRASE_TOKENS = 32


class PhraseQuestionSpec(BaseModel):
    """Everything editable that changes or filters a phrase answer."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    kind: Literal["phrase_distribution"] = "phrase_distribution"
    text: StrictStr = Field(min_length=1, max_length=300)
    comparison_text: StrictStr | None = Field(default=None, min_length=1, max_length=300)
    case_sensitive: bool = False
    #: The three widenings, each off by default and each adding forms rather
    #: than replacing them. Specification, not provenance: ticking one changes
    #: what is being asked, so it changes the answer and belongs with the
    #: phrase rather than with the record of how the phrase was read.
    normalize: bool = False
    match_lemma: bool = False
    match_nominalization: bool = False
    position_bins: int = Field(default=10, ge=5, le=50)
    evidence_year: int | None = Field(default=None, ge=1, le=9999)
    evidence_document_id: StrictStr | None = Field(default=None, max_length=128)

    @field_validator("text", "comparison_text")
    @classmethod
    def _meaningful_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.strip():
            raise ValueError("Enter a phrase to save.")
        return value.strip()


class MatchingProfile(BaseModel):
    """How this question's text became the tokens that were matched.

    Saved beside the question because the text alone does not determine it.
    ``public health`` is two tokens under every parser, but ``U.S.`` is one
    under spaCy and ``COVID-19`` is one under spaCy and three under the
    regular expression the workspace used before. A question reopened or
    published under different rules is a different question, and without this
    record nothing in the answer would say so.

    The defaults describe a question saved before any of this was recorded:
    ``version`` 0 and ``source`` "unrecorded" mean the tokens are unknown, so
    reopening or publishing it resolves the text again with the snapshot's
    parser and says in the published record that it did.
    """

    model_config = ConfigDict(extra="forbid")

    version: int = Field(default=0, ge=0, le=1000)
    tokenizer: StrictStr = Field(default="", max_length=200)
    source: Literal["saved", "snapshot", "approximate", "unrecorded"] = "unrecorded"
    tokens: list[StrictStr] = Field(default_factory=list, max_length=MAX_PHRASE_TOKENS)
    comparison_tokens: list[StrictStr] = Field(default_factory=list, max_length=MAX_PHRASE_TOKENS)


class QuestionBody(BaseModel):
    """A reproducible question and the frozen corpus selection it belongs to."""

    model_config = ConfigDict(extra="forbid")

    name: StrictStr = Field(min_length=1, max_length=MAX_QUESTION_NAME)
    snapshot_id: StrictStr = Field(pattern=r"^[a-f0-9]{64}$")
    parser: Literal["spacy", "stanza"]
    document_ids: list[StrictStr] = Field(min_length=1, max_length=2000)
    specification: PhraseQuestionSpec
    #: Provenance, not specification: it says how the question was read, not
    #: what was asked, and editing the phrase is what changes the question.
    matching: MatchingProfile = Field(default_factory=MatchingProfile)
    #: The revision the editor believes it is replacing. An update that names
    #: a revision other than the stored one is refused, so two people -- or two
    #: windows -- editing the same saved question cannot silently overwrite
    #: each other and leave only the later one's numbers.
    expected_revision: int | None = Field(default=None, ge=1)

    @field_validator("name")
    @classmethod
    def _clean_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Give this question a name.")
        return cleaned

    @field_validator("document_ids")
    @classmethod
    def _unique_documents(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError("Document IDs cannot be empty.")
        if len(set(value)) != len(value):
            raise ValueError("Document IDs cannot repeat.")
        return value


__all__ = [
    "MAX_PHRASE_TOKENS",
    "MAX_QUESTIONS_PER_PROJECT",
    "MAX_QUESTION_NAME",
    "MatchingProfile",
    "PhraseQuestionSpec",
    "QuestionBody",
]
