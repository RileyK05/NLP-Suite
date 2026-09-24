import { useEffect, useState } from "react";
import { FileWarning, LoaderCircle } from "lucide-react";
import { artifactBlobUrl, artifactFrameUrl, artifactText } from "./api";
import { Markdown } from "./Markdown";

const IMAGES = new Set([".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"]);
// Markdown is the one artifact kind the app renders itself: served into an
// iframe it showed its own asterisks, which is a poor way to present the file
// whose whole job is to be read.
const MARKDOWN = new Set([".md"]);

export function viewableArtifact(path: string): boolean {
  const dot = path.lastIndexOf(".");
  const suffix = dot >= 0 ? path.slice(dot).toLowerCase() : "";
  return IMAGES.has(suffix) || /\.(html?|pdf|svg|txt|kml|md)$/.test(suffix);
}

export function ArtifactViewer({
  path,
  fetchUrl,
}: {
  path: string;
  fetchUrl: string;
}) {
  const [url, setUrl] = useState<string | null>(null);
  const [text, setText] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);
  const dot = path.lastIndexOf(".");
  const suffix = dot >= 0 ? path.slice(dot).toLowerCase() : "";
  const markdown = MARKDOWN.has(suffix);
  useEffect(() => {
    let alive = true;
    let created: string | null = null;
    setUrl(null);
    setText(null);
    setFailed(false);
    void (async () => {
      try {
        if (markdown) {
          const body = await artifactText(fetchUrl);
          if (alive) setText(body);
          return;
        }
        const blobUrl = await (IMAGES.has(suffix)
          ? artifactBlobUrl(fetchUrl)
          : artifactFrameUrl(fetchUrl));
        if (!alive) {
          if (blobUrl.startsWith("blob:")) URL.revokeObjectURL(blobUrl);
          return;
        }
        created = blobUrl;
        setUrl(blobUrl);
      } catch {
        if (alive) setFailed(true);
      }
    })();
    return () => {
      alive = false;
      if (created?.startsWith("blob:")) URL.revokeObjectURL(created);
    };
  }, [fetchUrl, suffix, markdown]);
  if (failed)
    return (
      <p className="muted">
        <FileWarning size={15} /> Could not load this artifact for viewing. Use
        the download button instead.
      </p>
    );
  if (markdown && text !== null)
    return (
      <div className="artifact-view artifact-reading">
        <Markdown source={text} />
      </div>
    );
  if (!url || (markdown && text === null))
    return (
      <p role="status" className="muted">
        <LoaderCircle size={15} className="spin" /> Loading preview…
      </p>
    );
  if (IMAGES.has(suffix))
    return (
      <div className="artifact-view">
        <img src={url} alt={path} />
      </div>
    );
  return (
    <iframe
      className="artifact-frame"
      title={path}
      src={url}
      sandbox="allow-scripts"
      referrerPolicy="no-referrer"
    />
  );
}
