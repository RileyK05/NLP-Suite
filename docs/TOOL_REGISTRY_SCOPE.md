# Declarative tool registry scope

`core/profiler/registry.py` is the contract for tools that the profiler and
future UI may select without importing a tool module. It currently contains
the 14 retained, metadata-complete tools in `TOOL_REGISTRY`.

The remaining thin CLIs are not silently treated as covered. They are listed
in `TOOL_REGISTRY_EXCLUSIONS` with a reason (environment/reporting command,
conditional schema, parser/domain metadata, external service, database, or
optional model/asset dependency). A CLI may move into the registry only when
its input kind, conditional parameters, parser prerequisites, asset/version
requirements, and every emitted artifact are declared and tested.

Registry checks:

- `validate_specs(TOOL_REGISTRY)` must return no diagnostics.
- Every registered capability ID must occur in `docs/REPLACEMENT_LEDGER.md`.
- The exclusion list is reviewed whenever a new `tools/*.py` entry point is
  added; an omission is a scope error, not an implicit registration.
