import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { convertFileSrc, invoke, isTauri } from "@tauri-apps/api/core";
import { artifactFrameUrl, connect } from "./api";

vi.mock("@tauri-apps/api/core", () => ({
  isTauri: vi.fn(),
  invoke: vi.fn(),
  convertFileSrc: vi.fn(),
}));
const fetchMock = vi.fn();
const path = `/projects/${"a".repeat(32)}/jobs/${"b".repeat(32)}/artifacts/0`;

beforeEach(async () => {
  vi.stubGlobal("fetch", fetchMock);
  vi.mocked(isTauri).mockReturnValue(true);
  vi.mocked(invoke).mockResolvedValue({
    baseUrl: "http://127.0.0.1:12345",
    token: "private-session-token",
  });
  fetchMock.mockResolvedValueOnce(Response.json({ ok: true }));
  await connect();
  fetchMock.mockClear();
});
afterEach(() => {
  vi.resetAllMocks();
  vi.unstubAllGlobals();
});

it("uses the platform-mapped protocol without a token or blob in native frames", async () => {
  vi.mocked(convertFileSrc).mockImplementation(
    (value, protocol) =>
      `http://${protocol}.localhost/${encodeURIComponent(value)}`,
  );
  expect(await artifactFrameUrl(path)).toBe(`http://nlp-viz.localhost${path}`);
  expect(convertFileSrc).toHaveBeenCalledWith("", "nlp-viz");
  expect(fetchMock).not.toHaveBeenCalled();
});
it("uses an authenticated scoped ticket request in browser preview", async () => {
  vi.mocked(isTauri).mockReturnValue(false);
  fetchMock.mockResolvedValueOnce(
    Response.json({ path: `/api/previews/${"x".repeat(43)}` }),
  );
  const url = await artifactFrameUrl(path);
  expect(url).toBe(`http://127.0.0.1:12345/api/previews/${"x".repeat(43)}`);
  expect(url).not.toContain("private-session-token");
  expect(fetchMock.mock.calls[0][1].headers.Authorization).toBe(
    "Bearer private-session-token",
  );
});
it.each([
  "/health",
  "https://example.com/",
  path + "?download=true",
  path + "/../../health",
])("rejects non-artifact frame paths: %s", async (invalid) => {
  await expect(artifactFrameUrl(invalid)).rejects.toThrow("Invalid artifact");
  expect(fetchMock).not.toHaveBeenCalled();
});
