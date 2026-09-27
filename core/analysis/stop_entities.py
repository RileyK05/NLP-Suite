"""Person entities that are probably not people: suggested, never applied (plan 5.2).

Transcripts and official speech are full of common nouns the tagger calls
people: "Speaker", "Chamber", "Boo", "Pell" (backlog 4d-5). The suite does
not guess them away. It names them -- "Most frequent PERSON entities that are
probably not people: Speaker, Chamber" -- and the reader decides, by adding
them to the ``ner`` tool's ``ignore`` parameter. The same list serves
``gender_guess`` and ``quote_annotator``, which inherit the same noise.

Two simple rules, both reviewable:

1. the entity's lowercase form is one of the common nouns people are mistaken
   for (the bundled list below, kept small and readable), or
2. the entity occurs in more than half of the documents (the same voice from
   the chair in every meeting is a role, not a recurring person).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

__all__ = ["COMMON_NOUNS", "suspect_persons"]

#: Common nouns NER tags as PERSON in official and everyday speech: offices
#: and roles, family and everyday people-words. Proper names of people never
#: appear here -- this list can only suggest, and a real "Chamber" (a person
#: named Chamber) is the reader's call, not the suite's.
COMMON_NOUNS: frozenset[str] = frozenset(
    {
        "ambassador",
        "anybody",
        "anyone",
        "audience",
        "boo",
        "boy",
        "brother",
        "captain",
        "chair",
        "chairman",
        "chairwoman",
        "chamber",
        "chief",
        "child",
        "children",
        "clerk",
        "colleague",
        "colleagues",
        "corporal",
        "crowd",
        "detective",
        "doctor",
        "driver",
        "enemy",
        "enemies",
        "everyone",
        "farmer",
        "father",
        "friend",
        "friends",
        "general",
        "gentleman",
        "gentlemen",
        "gentlewoman",
        "girl",
        "guard",
        "guest",
        "guests",
        "judge",
        "justice",
        "kid",
        "kids",
        "lawyer",
        "leader",
        "madam",
        "major",
        "man",
        "member",
        "members",
        "mother",
        "neighbor",
        "neighbors",
        "nurse",
        "officer",
        "people",
        "person",
        "pilot",
        "police",
        "president",
        "prisoner",
        "private",
        "professor",
        "reader",
        "readers",
        "sailor",
        "secretary",
        "sergeant",
        "sister",
        "soldier",
        "son",
        "somebody",
        "someone",
        "speaker",
        "student",
        "suspect",
        "teacher",
        "victim",
        "victims",
        "witness",
        "witnesses",
        "woman",
        "worker",
    }
)


def suspect_persons(
    entities: Iterable[str],
    document_shares: Mapping[str, float] | None = None,
    *,
    share_rule: float = 0.5,
) -> list[str]:
    """The entities worth suggesting as "probably not people", most useful first.

    *entities* are PERSON entity texts; *document_shares* maps each to the
    share of documents it occurs in (0..1), for rule 2. Names that occur in
    more than half the documents, or whose lowercase form is a common noun,
    come back sorted by share then name. Nothing is removed: the reader adds
    the names they agree with to the tool's ``ignore`` parameter.
    """
    shares = dict(document_shares or {})
    found: set[str] = set()
    for entity in entities:
        name = str(entity).strip()
        if not name:
            continue
        if name.casefold() in COMMON_NOUNS or shares.get(name, 0.0) > share_rule:
            found.add(name)
    return sorted(found, key=lambda name: (-shares.get(name, 0.0), name.casefold()))
