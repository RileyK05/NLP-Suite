import { useEffect, useState } from "react";
import { BookImage } from "lucide-react";
import { api, publicationFigure } from "./api";
import { PublicationFigure } from "./PublicationFigure";

/** A publication-only figure the engine can draw over a run's whole table
 *  (core/viz/static/bundles.py). */
export type BundleInfo = {
  name: string;
  tool: string;
  title: string;
  question: string;
};

/**
 * Figures that exist only for publication: a correlation matrix of every
 * measure a tool computes, a measure against length with both
 * distributions, the topics each decade leaned on, a map of the documents by
 * similarity. No interactive figure draws these; they answer questions about
 * the whole table at once, and they are written into the run's figures/
 * folder as well.
 *
 * Shown only when the run has some: an empty gallery reads as broken.
 */
export function BundleGallery({
  projectId,
  jobId,
}: {
  projectId: string;
  jobId: string;
}) {
  const [bundles, setBundles] = useState<BundleInfo[]>([]);
  const [chosen, setChosen] = useState<BundleInfo | null>(null);

  useEffect(() => {
    let alive = true;
    setBundles([]);
    setChosen(null);
    api<BundleInfo[]>(`/projects/${projectId}/jobs/${jobId}/bundles`)
      .then((list) => {
        if (alive) setBundles(list);
      })
      .catch(() => {
        // An offer on top of the run; if it cannot be listed, the run stands.
      });
    return () => {
      alive = false;
    };
  }, [projectId, jobId]);

  if (!bundles.length) return null;
  return (
    <section
      className="panel bundle-gallery"
      aria-label="Publication-only figures"
    >
      <div className="section-title">
        <div>
          <h2>
            <BookImage size={16} aria-hidden="true" /> Publication-only figures
          </h2>
          <p>
            Views of the whole table that only a static figure can give. They
            are also saved in this run's figures folder.
          </p>
        </div>
      </div>
      <div className="bundle-list" role="list">
        {bundles.map((bundle) => (
          <button
            key={bundle.name}
            type="button"
            role="listitem"
            className={
              chosen?.name === bundle.name ? "viz-kind selected" : "viz-kind"
            }
            aria-pressed={chosen?.name === bundle.name}
            onClick={() => setChosen(bundle)}
          >
            <strong>{bundle.title}</strong>
            <span>{bundle.question}</span>
          </button>
        ))}
      </div>
      {chosen && (
        <PublicationFigure
          publish={(format, dpi) =>
            publicationFigure(`/projects/${projectId}/jobs/${jobId}/bundles`, {
              bundle: chosen.name,
              format,
              dpi,
            })
          }
          drawKey={`${projectId}/${jobId}/${chosen.name}`}
          fileBase={chosen.name}
        />
      )}
    </section>
  );
}
