import { invoke, isTauri } from "@tauri-apps/api/core";
import { relaunch } from "@tauri-apps/plugin-process";
import { check, type Update } from "@tauri-apps/plugin-updater";
import { Download, TriangleAlert, X } from "lucide-react";
import { useEffect, useState } from "react";

/** The newest release, if it is newer than this app. Never throws: an
 * offline machine or an unreachable release page is not an error to show. */
export async function findUpdate(): Promise<Update | null> {
  if (!isTauri()) return null;
  try {
    return await check();
  } catch {
    return null;
  }
}

type Phase = "offered" | "downloading" | "installing" | "failed";

/**
 * Offers a newer release once, at the top of the workspace. Nothing happens
 * until the reader asks: the download runs while the app keeps working, then
 * the engine is stopped (Windows cannot replace a running runtime), the
 * update installs and the app restarts.
 */
export function UpdateBanner({ find = findUpdate }: { find?: () => Promise<Update | null> }) {
  const [update, setUpdate] = useState<Update | null>(null);
  const [phase, setPhase] = useState<Phase>("offered");
  const [percent, setPercent] = useState<number | null>(null);
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    let current = true;
    void find().then((found) => {
      if (current) setUpdate(found);
    });
    return () => {
      current = false;
    };
  }, [find]);

  if (!update || dismissed) return null;

  async function install(target: Update) {
    setPhase("downloading");
    let total = 0;
    let received = 0;
    try {
      await target.download((event) => {
        if (event.event === "Started") total = event.data.contentLength ?? 0;
        if (event.event === "Progress" && total > 0) {
          received += event.data.chunkLength;
          setPercent(Math.min(100, Math.round((received * 100) / total)));
        }
      });
      setPhase("installing");
      await invoke("stop_engine");
      await target.install();
      await relaunch();
    } catch {
      setPhase("failed");
    }
  }

  if (phase === "failed") {
    return (
      <div className="alert error" role="alert">
        <TriangleAlert size={18} />
        <span>
          NLP Suite {update.version} could not be installed. Close and reopen
          NLP Suite to try again, or download it from the Releases page.
        </span>
        <button aria-label="Dismiss update error" onClick={() => setDismissed(true)}>
          <X size={16} />
        </button>
      </div>
    );
  }

  return (
    <div className="alert notice" role="status">
      <Download size={18} />
      {phase === "offered" && (
        <>
          <span>NLP Suite {update.version} is available.</span>
          <button className="secondary" onClick={() => void install(update)}>
            Install and restart
          </button>
          <button aria-label="Dismiss update" onClick={() => setDismissed(true)}>
            <X size={16} />
          </button>
        </>
      )}
      {phase === "downloading" && (
        <span>
          Downloading NLP Suite {update.version}
          {percent === null ? "…" : ` (${percent}%)`}. You can keep working.
        </span>
      )}
      {phase === "installing" && (
        <span>Installing NLP Suite {update.version}. It will restart in a moment.</span>
      )}
    </div>
  );
}
