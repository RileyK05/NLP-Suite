import { useEffect, useRef, useState } from "react";
import { Download } from "lucide-react";
import type { Diagnostic, FigureFormat, FigureImage } from "./api";
import { Diagnostics } from "./RunStatus";

/** Draws one publication figure in the format asked; the caller knows which
 *  panel, which parameters and which run or answer it belongs to. */
export type PublishFigure = (
  format: FigureFormat,
  dpi: number,
) => Promise<FigureImage>;

/** On-screen preview resolution; downloads use `DOWNLOAD_DPI`. */
const PREVIEW_DPI = 130;
const DOWNLOAD_DPI = 300;

/**
 * The same figure drawn for publication: matplotlib and seaborn, rendered by
 * the engine from the same prepared panel the interactive view draws.
 *
 * It shows what a static figure can add -- the spread around a trend,
 * violins under a box, a matrix reordered by clustering, labels placed so
 * none overprints another -- with the provenance caption under it, and
 * saves as PNG (300 dpi), SVG or PDF. Nothing here is clickable: that is
 * what the interactive view is for.
 */
export function PublicationFigure({
  publish,
  drawKey,
  fileBase,
}: {
  publish: PublishFigure;
  /** Changes whenever the drawn panel or its parameters change. */
  drawKey: string;
  fileBase: string;
}) {
  const [preview, setPreview] = useState<{
    url: string;
    warnings: number;
  } | null>(null);
  const [refusal, setRefusal] = useState<Diagnostic[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [saving, setSaving] = useState<FigureFormat | null>(null);
  const sequence = useRef(0);

  useEffect(() => {
    const mine = ++sequence.current;
    let url = "";
    setPreview(null);
    setRefusal(null);
    setBusy(true);
    publish("png", PREVIEW_DPI)
      .then((image) => {
        if (sequence.current !== mine) return;
        if (!image.ok) {
          setRefusal(image.diagnostics);
          return;
        }
        url = URL.createObjectURL(image.blob);
        setPreview({ url, warnings: image.warnings });
      })
      .catch((error: unknown) => {
        if (sequence.current !== mine) return;
        setRefusal([
          {
            severity: "ERROR",
            code: "FIGURE_REQUEST_FAILED",
            message: error instanceof Error ? error.message : String(error),
          },
        ]);
      })
      .finally(() => {
        if (sequence.current === mine) setBusy(false);
      });
    return () => {
      if (url) URL.revokeObjectURL(url);
    };
  }, [drawKey]); // eslint-disable-line react-hooks/exhaustive-deps

  const save = async (format: FigureFormat) => {
    setSaving(format);
    try {
      const image = await publish(format, DOWNLOAD_DPI);
      if (!image.ok) {
        setRefusal(image.diagnostics);
        return;
      }
      const url = URL.createObjectURL(image.blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `${fileBase}.${format}`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } finally {
      setSaving(null);
    }
  };

  return (
    <div className="publication-figure">
      <div className="publication-actions" role="group" aria-label="Save">
        {(["png", "svg", "pdf"] as const).map((format) => (
          <button
            key={format}
            type="button"
            className="secondary"
            disabled={!preview || saving !== null}
            onClick={() => void save(format)}
          >
            <Download size={14} aria-hidden="true" />{" "}
            {saving === format ? "Saving…" : format.toUpperCase()}
          </button>
        ))}
      </div>
      {busy && <p className="muted">Drawing the publication figure…</p>}
      {refusal && <Diagnostics items={refusal} />}
      {preview && (
        <>
          <img
            src={preview.url}
            alt="Publication figure: the same data, drawn with matplotlib and seaborn"
          />
          {preview.warnings > 0 && (
            <p className="muted">
              {preview.warnings} drawing problem(s) were detected (text printed
              over text); the figure is shown as drawn.
            </p>
          )}
        </>
      )}
    </div>
  );
}
