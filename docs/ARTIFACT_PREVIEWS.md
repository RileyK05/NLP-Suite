# Artifact preview fix (2026-09-15)

HTML previews no longer use blob URLs. Blob documents inherit the app's CSP,
which prevents inline Plotly scripts from running. The browser preview also
had a restrictive backend CSP, so this affected more than Tauri alone.
The original screenshot is consistent with frame blocking; its exact live
policy was not captured.

## Design and security

- Native desktop: an artifact-only `nlp-viz` protocol fetches through the Rust
  bridge. The backend bearer token never appears in a frame URL or response.
  The route accepts only project/job IDs and an artifact index, not arbitrary
  backend paths, file paths, queries or remote URLs. Redirects are disabled.
- Browser preview: an authenticated POST creates a five-minute, single-artifact
  ticket. The iframe URL contains that limited capability, not the session
  bearer token. Tickets are bounded in memory and disappear on server restart.
  Do not share preview URLs while their tickets remain valid.
- Both frame responses have their own CSP allowing inline chart scripts but
  blocking network fetches, external images/scripts, nested frames and forms.
  Both retain an opaque-origin sandbox: `allow-scripts`, not `allow-same-origin`.
- The main app still disallows inline scripts. Raster image previews still use
  authenticated blobs. Existing `?download=true` handling is unchanged.
- HTML previews must be self-contained and no larger than 32 MiB. External
  assets and sibling-file dependencies intentionally do not load. PDF support
  is webview-dependent; use Download if a sandboxed PDF cannot display.

On Windows, Tauri maps the protocol to `http://nlp-viz.localhost`; on macOS/Linux
it uses the custom scheme. The app CSP includes both forms. `convertFileSrc`
is used only to derive the platform-specific origin because it encodes slashes
when given a full route.

## Try the fixed source

Fully close the old app, then run `npm run desktop` from `desktop/` with the
documented Python/Rust development environment. This recompiles the native
protocol and restarts the source backend. A frontend refresh alone cannot
update native protocol registration or the packaged CSP.

For browser preview, run `npm run build` from `desktop/`, then restart
`python scripts/desktop_preview.py` from the project root, using your existing
`--data-dir` if applicable.

1. Open Runs & results and select an existing self-contained chart HTML.
2. Click the eye. Confirm the chart appears without a blocked page or blank box.
3. Hover the chart and resize the window; verify labels and tooltips work.
4. Close/reopen the preview and switch between HTML and PNG artifacts.
5. Download the HTML and confirm the downloaded artifact is still intact.
6. Repeat with a wordcloud. Jobs need not be rerun to change viewer behavior.

The frontend/build, backend preview tests and Rust protocol tests pass. The
rebuilt Windows development app launches, but native window inspection approval
timed out, so actual chart rendering still requires the manual check above.
macOS/Linux rendering is not yet verified.

The development launch also exposed a Windows `EBUSY` crash when Vite watched
Rust's generated executable. Vite now ignores `src-tauri` and `.toolchain`;
Tauri continues to watch and rebuild its own native sources.

## Installer caveat

The September 10 installer predates these changes and was **not rebuilt** for
this fix. Rebuilding only the frontend does not update that installer.
Before shipping the new visualization functionality, also revise the frozen
runtime recipe: `scripts/build_desktop_backend.py` currently excludes Plotly
and matplotlib. Bundle the visualization dependencies, rebuild the Python
runtime and Tauri installer, and test charts on a clean machine. This source
viewer fix is not a certification of that older packaged runtime.
