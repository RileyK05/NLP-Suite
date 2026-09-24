import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { invoke } from "@tauri-apps/api/core";
import { api, connect, download, post, upload } from "./api";

vi.mock("@tauri-apps/api/core", () => ({
  isTauri: () => true,
  invoke: vi.fn(),
}));

describe("authenticated desktop transport", () => {
  const fetchMock = vi.fn();

  beforeEach(async () => {
    vi.stubGlobal("fetch", fetchMock);
    vi.mocked(invoke).mockResolvedValue({
      baseUrl: "http://127.0.0.1:12345",
      token: "fixture-token",
    });
    fetchMock.mockResolvedValueOnce(Response.json({ status: "ok" }));
    await connect();
    fetchMock.mockClear();
  });

  afterEach(() => {
    vi.resetAllMocks();
    vi.unstubAllGlobals();
  });

  it("retains authentication when posting JSON settings", async () => {
    fetchMock.mockResolvedValueOnce(Response.json({ id: "job" }));
    await post("/projects/p/jobs", { tool: "readability" });
    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:12345/api/projects/p/jobs",
      expect.objectContaining({
        method: "POST",
        headers: {
          Authorization: "Bearer fixture-token",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ tool: "readability" }),
      }),
    );
  });

  it("surfaces actionable server errors", async () => {
    fetchMock.mockResolvedValueOnce(
      Response.json({ detail: "Wait for active runs." }, { status: 409 }),
    );
    await expect(api("/projects/p/backup")).rejects.toThrow(
      "Wait for active runs.",
    );
  });

  it("forwards cancellation to fetch for superseded result requests", async () => {
    const controller = new AbortController();
    fetchMock.mockImplementationOnce(
      (_url, init) =>
        new Promise((_resolve, reject) => {
          init.signal.addEventListener("abort", () =>
            reject(new DOMException("Cancelled", "AbortError")),
          );
        }),
    );
    const result = api("/projects/p/jobs/j/artifacts/0", {
      signal: controller.signal,
    });
    controller.abort();
    await expect(result).rejects.toMatchObject({ name: "AbortError" });
    expect(fetchMock.mock.calls[0][1].signal).toBe(controller.signal);
  });

  it("handles non-JSON error responses", async () => {
    fetchMock.mockResolvedValueOnce(
      new Response("unavailable", { status: 503 }),
    );
    await expect(api("/health")).rejects.toThrow("Request failed (503)");
  });

  it("continues a multi-file import after an individual failure", async () => {
    fetchMock
      .mockResolvedValueOnce(Response.json({ duplicate: false }))
      .mockResolvedValueOnce(
        Response.json({ detail: "Unsupported file" }, { status: 400 }),
      )
      .mockResolvedValueOnce(Response.json({ duplicate: true }));
    const report = await upload("p", [
      new File(["first"], "a & b.txt"),
      new File(["bad"], "bad.exe"),
      new File(["existing"], "existing.txt"),
    ]);
    expect(report).toEqual({
      imported: 1,
      duplicates: 1,
      errors: [{ name: "bad.exe", message: "Unsupported file" }],
    });
    expect(fetchMock.mock.calls[0][0]).toContain("name=a%20%26%20b.txt");
  });

  it("delegates native exports without fetching the archive into JavaScript", async () => {
    vi.mocked(invoke).mockResolvedValueOnce(true);
    await download("/projects/p/backup", "project.nlpsuite");
    expect(invoke).toHaveBeenLastCalledWith("save_download", {
      path: "/projects/p/backup",
      name: "project.nlpsuite",
    });
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
