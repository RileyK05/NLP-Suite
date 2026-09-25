import { useCallback, useEffect, useRef, useState } from "react";
import { Check, Download, ExternalLink, Trash2, TriangleAlert, X } from "lucide-react";

import {
  cancelModelDownload,
  deleteModel,
  downloadModel,
  listModels,
  title,
  type ModelInfo,
} from "./api";

/** How often a running download's progress is read back from the engine. */
const POLL_MS = 1000;

function statusWords(model: ModelInfo): string {
  if (model.status === "downloading") return "Downloading";
  if (model.status === "ready") return model.removable ? "Downloaded" : model.bundled ? "Included with the app" : "Installed";
  if (model.status === "corrupt") return "Incomplete — download it again";
  if (model.status === "unpublished") return "Not available yet";
  return "Not downloaded";
}

function percent(model: ModelInfo): number {
  const job = model.download;
  if (!job || !job.total) return 0;
  return Math.min(100, Math.round((100 * job.done) / job.total));
}

function ModelCard({
  model,
  onAction,
  busy,
}: {
  model: ModelInfo;
  onAction: (action: "download" | "cancel" | "delete", model: ModelInfo) => void;
  busy: boolean;
}) {
  const ready = model.status === "ready";
  const downloading = model.status === "downloading";
  const failed = model.download && model.download.state !== "downloading" ? model.download : null;
  return (
    <article className="model-card" aria-label={model.name}>
      <header>
        <div>
          <h3>{model.name}</h3>
          <span className={`model-status status-${model.status}`}>
            {ready && <Check size={13} aria-hidden="true" />}
            {statusWords(model)}
          </span>
        </div>
        <span className="model-size">{model.sizeMb.toLocaleString()} MB</span>
      </header>
      <p>{model.description}</p>
      {model.usedBy.length > 0 && (
        <p className="muted model-used-by">
          Used by {model.usedBy.map((name) => title(name)).join(", ")}.
        </p>
      )}
      {downloading && (
        <div className="model-progress">
          <progress max={100} value={percent(model)} aria-label={`${model.name} download`} />
          <span>{percent(model)}%</span>
        </div>
      )}
      {failed && (
        <p className={failed.state === "failed" ? "model-error" : "muted"} role="status">
          {failed.state === "failed" && <TriangleAlert size={14} aria-hidden="true" />}
          {failed.message}
        </p>
      )}
      <footer>
        <a href={model.source} target="_blank" rel="noreferrer" className="model-source">
          {model.license} <ExternalLink size={12} aria-hidden="true" />
        </a>
        <div className="model-actions">
          {downloading && (
            <button className="secondary" disabled={busy} onClick={() => onAction("cancel", model)}>
              <X size={15} /> Cancel
            </button>
          )}
          {!ready && !downloading && model.status !== "unpublished" && (
            <button className="primary" disabled={busy} onClick={() => onAction("download", model)}>
              <Download size={15} /> {failed || model.status === "corrupt" ? "Try again" : "Download"}
            </button>
          )}
          {ready && model.removable && (
            <button className="secondary danger" disabled={busy} onClick={() => onAction("delete", model)}>
              <Trash2 size={15} /> Remove
            </button>
          )}
        </div>
      </footer>
    </article>
  );
}

/**
 * The Models page: every pretrained model the suite can run, whether it is
 * here, and a download button for the ones that are not.
 *
 * Small models ship inside the app; larger ones (and any added later) are
 * downloaded once and survive app updates. The list is the engine's registry
 * (core/models/registry.py), so a new model appears here without a UI change.
 */
export function Models({ onChanged }: { onChanged?: () => void }) {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [folder, setFolder] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const wasDownloading = useRef(false);
  // The parent passes a fresh function each render; read it through a ref so
  // it never re-triggers the fetch below.
  const changed = useRef(onChanged);
  changed.current = onChanged;

  const refresh = useCallback(async () => {
    try {
      const listing = await listModels();
      setModels(listing.models);
      setFolder(listing.folder);
      setError("");
      const downloading = listing.models.some((model) => model.status === "downloading");
      // A download that just finished changes which analyses can run.
      if (wasDownloading.current && !downloading) changed.current?.();
      wasDownloading.current = downloading;
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const downloading = models.some((model) => model.status === "downloading");
  useEffect(() => {
    if (!downloading) return;
    const timer = window.setInterval(() => void refresh(), POLL_MS);
    return () => window.clearInterval(timer);
  }, [downloading, refresh]);

  const act = async (action: "download" | "cancel" | "delete", model: ModelInfo) => {
    setBusy(true);
    try {
      if (action === "download") {
        await downloadModel(model.id);
        wasDownloading.current = true;
      } else if (action === "cancel") await cancelModelDownload(model.id);
      else {
        const result = await deleteModel(model.id);
        if (result.error) setError(result.error);
        changed.current?.();
      }
      await refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(false);
    }
  };

  const groups = new Map<string, ModelInfo[]>();
  for (const model of models) groups.set(model.kindLabel, [...(groups.get(model.kindLabel) ?? []), model]);

  return (
    <>
      <div className="page-heading">
        <div>
          <span className="eyebrow">LANGUAGE MODELS</span>
          <h1>Models</h1>
          <p>
            The pretrained models behind BERT, neural sentiment and document embeddings. Everything runs on this
            computer; nothing you analyse is sent anywhere.
          </p>
        </div>
      </div>
      {error && (
        <div className="alert error" role="alert">
          <TriangleAlert size={18} />
          <span>{error}</span>
        </div>
      )}
      {[...groups].map(([label, members]) => (
        <section className="panel model-group" key={label}>
          <h2>{label}</h2>
          <div className="model-grid">
            {members.map((model) => (
              <ModelCard key={model.id} model={model} busy={busy} onAction={(a, m) => void act(a, m)} />
            ))}
          </div>
        </section>
      ))}
      {folder && (
        <p className="muted model-folder">
          Downloaded models are kept in <code>{folder}</code> and stay when the app updates.
        </p>
      )}
    </>
  );
}
