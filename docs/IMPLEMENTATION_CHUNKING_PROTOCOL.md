# Implementation Chunking and Prompt Protocol

This protocol turns one roadmap item from `FULL_REPLACEMENT_PLAN.md` into a
sequence of small, repeated prompts for an implementation model. It exists
because a model that can complete five bounded turns reliably may still fail
when asked to understand, design, implement, test, integrate, and audit a
feature in one turn.

The implementation model remains the author of all implementation code. The
professor supplies the assignment and evaluates the work; the professor does
not finish a weak submission.

## 1. Three levels of work

Do not use these terms interchangeably:

1. **Phase** (`FR-2`, for example): a program milestone containing related
   replacement work.
2. **Roadmap item** (`FR-2.1`): one capability or project outcome recorded in
   `REPLACEMENT_LEDGER.md`. Some roadmap items are still too large to give to a
   model directly.
3. **Execution chunk** (`FR-2.1/C2`): one bounded model turn with one expected
   output and a stop condition.

A roadmap item remains `implementing` across its execution chunks. It moves to
`review` only after its final verification chunk. Execution chunks are tracked
inside that item's dossier, not as extra rows in the high-level ledger.

## 2. Chunk-size test

An execution chunk is small enough when all of the following are true:

- it has one verb: inspect, specify, fixture, test, implement, integrate, or
  verify;
- its expected output can be described in one or two sentences;
- the allowed files are explicit and narrow;
- success can be checked with one focused command or inspection;
- the model can stop without starting adjacent work;
- a reviewer can understand the resulting diff without reconstructing the
  entire roadmap item.

Split a chunk again if it includes any two of these:

- a new external dependency or licensed asset;
- a network service, subprocess, model download, database migration, or UI
  background process;
- more than two production modules;
- more than one independently testable algorithm;
- more than one output schema;
- platform-specific behavior;
- a likely change larger than roughly 300 lines;
- a sentence containing “and” that joins two different user outcomes.

The line threshold is a warning, not a target. A 40-line statistical formula
and a 40-line model lifecycle are different levels of risk.

## 3. Standard repeated-prompt cycle

Use the same conversation for all chunks in a roadmap item so the model keeps
local context. Give only the next chunk after examining the previous response.
Do not paste all future prompts at once.

### C0 — reconnaissance (read-only)

The model reads the dossier, exact legacy references, current implementation,
tests, and contracts. It reports:

- its understanding of the user outcome;
- legacy inputs, outputs, options, and known defects;
- current new-suite behavior and gaps;
- files it expects later chunks to touch;
- unanswered questions or blockers.

It edits nothing. This turn catches hallucinated file names, misunderstood
legacy behavior, and missing prerequisites before code exists.

Stop condition: the report matches the dossier and every uncertainty is either
resolved by evidence or explicitly returned to the professor.

### C1 — oracle and fixture preparation

The model adds or identifies the smallest independent evidence needed for the
assignment: fixtures, accepted formula examples with hand-computed expected
values, direct calls to the backing library, recorded service responses,
spot-check legacy goldens (only where the dossier names a tier-2 glue
question — see Gate B in `FULL_REPLACEMENT_PLAN.md`), or `SPEC_ONLY` cases.

It does not implement production behavior. A missing legacy-run golden never
blocks: hand-computed definitional references are first-rank evidence for
documented formulas and package-backed analyses. If a required model,
licensed asset, service, or product decision is unavailable, it stops
instead of manufacturing expected values from the planned implementation.

Stop condition: each required behavior and edge case has a named fixture and
expected outcome whose origin is documented.

### C2 — contract tests

The model writes focused tests for public input/output shape, correct examples,
edge cases, failure diagnostics, and the named legacy defects. Tests should
fail for the missing production behavior for the expected reason.

It must show the focused test command and distinguish expected failures from
unrelated regressions. It does not weaken existing assertions or globally
skip the feature.

Stop condition: the tests express the dossier without embedding a second copy
of the intended implementation.

### C3 — core implementation

The model implements the smallest core behavior that satisfies C2. No CLI, UI,
packaging, opportunistic refactor, or adjacent feature unless the dossier says
it is part of this chunk.

It runs focused tests after the change and reports diagnostics or limitations.
It does not rename a stub and leave the stub algorithm intact.

Stop condition: focused core tests pass and the production diff stays within
the allowed core files.

### C4 — integration and artifacts

The model connects the verified core behavior to the required CLI, tool
registry, profiler eligibility, writer/envelope, or UI form. Split this into
`C4a`, `C4b`, and `C4c` when more than one interface is involved.

Stop condition: the same core API is exercised through every required surface,
and produced artifacts include the required provenance.

### C5 — self-audit and handoff

The model runs the focused, integration, and complete quality gates. It
compares the final diff with the dossier and inspects generated artifacts. It
then supplies the handoff from `FULL_REPLACEMENT_PLAN.md` and moves the roadmap
item to `review`, not `verified`.

This turn is not permission to make broad “cleanup” changes. Any discovered
unrelated problem becomes a follow-up note.

Stop condition: reproducible evidence is ready for independent review.

### C6 — review corrections

Use one correction chunk per coherent reviewer finding. The prompt quotes the
finding, names the violated contract, and gives an acceptance check. The model
must diagnose before editing and must not rewrite unrelated areas.

Repeat C6 as needed. After corrections, repeat C5. The professor alone records
`verified` after accepting the resubmission.

## 4. Complexity classes

Classify a roadmap item in its dossier before prompting implementation.

| Class | Typical work | Expected chunks |
|---|---|---:|
| S | one deterministic function, existing contracts/dependencies | C0, C1/C2, C3, C5 |
| M | several algorithms or one core function plus an interface | C0–C5, with C4 splits |
| L | model/service lifecycle, major UI flow, profiler, PC-ACE | decomposition only; create smaller roadmap items first |

Class L is not an instruction to give the model a longer prompt. Its first
assignment is read-only decomposition. The resulting child items each need
their own dossier and C0–C5 cycle.

## 5. State carried between prompts

Begin each follow-up prompt with a compact state block:

```text
Roadmap item: FR-x.y
Current execution chunk: Cn
Roadmap status: implementing
Accepted outputs from earlier chunks:
- ...
Files currently changed:
- ...
Known constraints/blockers:
- ...
This turn may edit:
- ...
This turn must not edit:
- ...
Stop after:
- ...
```

Ask the model to correct the state block before working if it is inaccurate.
This repeated restatement reduces context drift and makes accidental scope
expansion visible.

## 6. Prompt templates

### Reconnaissance prompt

```text
You are implementing roadmap item FR-x.y, but this is read-only chunk C0.

[state block]

Read the packet dossier and every named file completely. Do not edit files or
design beyond the stated outcome. Report the legacy behavior, current behavior,
contract, defects to avoid, likely files, and blockers. Cite file/function
evidence for every important claim. Stop after the reconnaissance report.
```

### Oracle/tests prompt

```text
Continue FR-x.y with chunk C1 (or C2).

[state block]

Produce only the fixtures/oracle evidence (or contract tests) named in the
dossier — hand-computed values with their origin documented, independent
library calls, recorded service fixtures, or the dossier's tier-2 spot-check
goldens. Do not implement production behavior. Show where every expected
value came from and run only the focused evidence/test command. Stop and
report if no independent expected value is obtainable (a missing legacy-run
golden is not a stop condition; a missing model, asset, service, or product
decision is).
```

### Core implementation prompt

```text
Continue FR-x.y with chunk C3.

[state block]

Implement only the core behavior required by the accepted C2 tests. Preserve
the project contracts and do not touch adapters, UI, packaging, or adjacent
algorithms. Run the focused tests and inspect the diff. Stop after reporting
the core result and remaining integration work.
```

### Integration prompt

```text
Continue FR-x.y with chunk C4[a/b/c].

[state block]

Connect the already-tested core API to only the named interface and artifact
contract. Do not duplicate analysis logic in the adapter. Run the focused
integration test and inspect the produced envelope/artifact. Stop there.
```

### Verification prompt

```text
Continue FR-x.y with chunk C5.

[state block]

Make no feature expansion. Run the packet acceptance commands and full quality
gate, inspect artifacts and the complete diff against the dossier, update only
the assigned ledger row to `review`, and provide the required handoff. Never
mark the item verified.
```

### Review-correction prompt

```text
Continue FR-x.y with correction chunk C6.n.

[state block]

Reviewer finding:
[quote exactly]

Contract/evidence violated:
[reference]

First explain the cause. Then make the smallest correction within the allowed
files and run the named regression check. Do not address other findings or
start new work. Stop with a correction handoff.
```

## 7. Decomposition examples

These examples demonstrate granularity; they are not substitutes for packet
dossiers.

### FR-2.1 — readability (Class M)

- C0: inspect legacy formulas, current token/sentence/syllable sources, and
  identify conflicting formula conventions;
- C1: capture accepted worked examples and edge fixtures;
- C2a: test shared count contract and undefined/short-text behavior;
- C2b: test one formula family at a time;
- C3a: implement shared validated counts or adapter to existing counts;
- C3b: implement Flesch/Flesch-Kincaid/ARI;
- C3c: implement Fog/Coleman-Liau/SMOG and remaining approved formulas;
- C4a: CLI and envelope artifacts;
- C4b: registry/profiler integration;
- C5: full comparison and handoff.

Do not ask for “all readability metrics, CLI, tests, and profiler integration”
in one prompt.

### FR-5.6 — contextual embeddings and WSI (Class L)

Split it into separate roadmap items before code:

1. transformer backend lifecycle and cache;
2. token/span alignment and batching;
3. contextual-vector artifact schema;
4. one explicitly selected clustering algorithm;
5. clustering evaluation/diagnostics;
6. CLI/registry integration;
7. optional GPU/device behavior and model integration tests.

Each child then receives its own C0–C5 sequence. A hash-vector fake may support
tests, but it must live in tests and cannot be the production backend.

### FR-6.9 — PC-ACE (Class L)

The first model turn is read-only inventory and decomposition. Create child
roadmap items for grammar validation, source import, relational schema,
queries, aggregation families, and visualizations. Do not let one prompt edit
the legacy 7,000-line workflow into one new module.

## 8. Operator cadence

After each model response:

1. compare its output with the current chunk's stop condition;
2. correct misunderstandings before issuing the next prompt;
3. update the state block with only accepted facts;
4. keep unresolved questions visible;
5. issue one next prompt, not the rest of the sequence;
6. request a concise diff/status summary every time files changed.

Do not reward scope expansion by accepting extra work merely because it passes
tests. Revert or separate it through the normal student workflow.

## 9. Failure recovery

- If C0 misunderstands the feature, repeat C0 with narrower references.
- If C1 lacks an independent oracle, stop the roadmap item as `blocked`;
  do not proceed to code on guessed behavior. A missing legacy-run golden
  is not an oracle gap — fall back to hand-computed definitional evidence;
  only a missing model, licensed asset, service, or product decision stops
  the item.
- If C2 tests implementation details, return them for behavior-level rewrite.
- If C3 becomes broad, stop and split the chunk before more code accumulates.
- If the full gate finds unrelated failures, record them separately; do not
  let the current item absorb them.
- If the model repeats the same mistake, provide the violated contract and a
  smaller acceptance example, but leave the correction to the model.
- If an item needs repeated architectural rescue, reclassify it as L and
  decompose it rather than increasing prompt length.

## 10. Professor boundary

The professor may write dossiers, clarify requirements, identify evidence,
review diffs, run checks, explain findings, and accept or reject work. The
professor does not patch production code, rewrite the student's tests, finish
an incomplete chunk, or quietly make the quality gate green.

This boundary is part of the process evidence: success means the student model
can implement the documented assignment after feedback, not that a stronger
reviewer can repair its submission.
