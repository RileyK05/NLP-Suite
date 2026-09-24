# Desktop hardening, September 2026

The desktop now has an explicit release catalog in `desktop_backend/catalog.py`.
Adding a research adapter or command does not automatically expose it to end users.
The catalog is enforced on submission as well as in the interface.

## End-user scope

There are 35 reviewed desktop workflows. Experimental coreference, semantic
demonstrations, offline knowledge-graph stubs, and transformer research workflows
are excluded. NRC and SentiWordNet/hedonometer also remain in the developer
interfaces because their default resources are not supplied by the installer.
These engines are preserved in the source tree; hiding them does not certify
their scientific parity or remove them from the migration backlog.

The desktop uses the included English parser. Settings show analysis readiness,
project management, backups and notices, rather than Python package installation
commands. Tool names describe the analysis rather than exposing module names.
Required resources and unavailable components prevent submission in the form.
The backend independently validates every submitted request.

VADER runs with the included scorer by default. ANEW is an explicit optional
selection and requires a chosen lexicon file. Developer commands retain their
existing combined-analysis default; `--analysis vader` selects VADER alone.

The example importer reads three fictional texts from `assets/sample-corpus`,
never a developer's working corpus. The same texts are included in the frozen
runtime. No personal research corpus is bundled.

## Reliability and packaging

- Cancelled jobs are checked again after worker startup, before dispatch.
- Worker startup has a 90-second deadline and failed workers are reaped.
- Malformed non-object worker replies fail cleanly.
- Archived projects reject new imports and analysis submissions.
- Chart exports respect the requested top-N category limit. A requested PNG
  wordcloud that fails to render fails the job instead of reporting completion.
- N-gram document labels survive numeric document IDs; BIOES end/singleton
  boundaries are respected by the entity timeline.
- The frozen runtime excludes the `tools`, `app`, `scripts` and `tests` packages.
  Core analysis and desktop backend modules remain necessary runtime components.
  The frozen entrypoint is compiled by PyInstaller; developer scripts are not
  application navigation or separately shipped commands.
- Plotly, Wordcloud, Pillow and Openpyxl are declared packaging dependencies.
  Matplotlib is retained because Wordcloud imports it. Packaged chart exports
  offer HTML and Excel; formats requiring a separately installed rendering
  browser are not advertised. PNG wordclouds remain supported.

Regression coverage lives in `tests/test_production_boundary.py`, alongside the
existing desktop tests. The frozen-engine smoke test now also exercises VADER,
HTML/Excel chart exports, and PNG wordcloud generation.

## Release acceptance

Verification on the Windows development machine:

- The full standard Python suite passed: 1,395 tests, six skipped and 60
  model-integration tests deselected. Real bundled-model execution was checked
  separately by the frozen-engine smoke test.
- The frontend production build and all 73 frontend tests passed.
- All seven native Rust launcher/preview tests passed.
- Strict mypy passed for 201 Python source files; Ruff lint and formatting,
  the release audit, and whitespace checks passed.
- The rebuilt frozen engine passed 20 analysis/export cases, including VADER,
  HTML and Excel charts, and a PNG wordcloud whose file signature was checked.
  Input provenance, ZIP exports, backup/restore and graceful shutdown passed.
- The frozen module archive contains no first-party `tools`, `app`, `scripts`
  or `tests` packages. Its sample corpus contains exactly the three fictional
  example files.
- Browser checks covered project creation, example import, a real VADER run,
  results and the settings page. This is not a native installed-app UI check.

No installer was rebuilt or published during this pass. Existing installer
files still contain the earlier implementation.

Passing source tests is not a clean-machine installer certification. Before
publishing installers, run the native packaging workflow on each supported OS,
test installation and first launch on clean machines, and complete the target-user
protocol in `USABILITY_TEST.md`. The desktop remains labeled beta until that
acceptance evidence exists. This change does not publish a release or promise
full equivalence with NLP Suite 1.6.38.
