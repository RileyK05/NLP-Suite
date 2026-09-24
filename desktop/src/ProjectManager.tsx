import { useEffect, useState } from "react";
import { api, post, type Project } from "./api";

export function ProjectManager({
  project,
  busy,
  perform,
  refresh,
}: {
  project?: Project;
  busy: boolean;
  perform: (label: string, work: () => Promise<void>) => Promise<void>;
  refresh: (selectProjectId?: string) => Promise<void>;
}) {
  const [name, setName] = useState(project?.name || "");
  const [archived, setArchived] = useState<Project[]>([]);
  const [trashed, setTrashed] = useState<Project[]>([]);
  const [confirm, setConfirm] = useState(false);
  const [deleteConfirm, setDeleteConfirm] = useState(false);
  const [error, setError] = useState("");
  const load = async () =>
    setArchived(await api<Project[]>("/projects?archived=true"));
  const loadTrash = async () =>
    setTrashed(await api<Project[]>("/projects?trashed=true"));
  useEffect(() => {
    setName(project?.name || "");
    setConfirm(false);
    setDeleteConfirm(false);
    void load().catch((e) => setError(String(e)));
    void loadTrash().catch((e) => setError(String(e)));
  }, [project?.id, project?.name]);
  return (
    <section className="panel settings-panel">
      <h2>Manage projects</h2>
      {error && <p role="alert">{error}</p>}
      {project && (
        <>
          <label className="field-label">
            Project name
            <input
              value={name}
              maxLength={120}
              onChange={(e) => setName(e.target.value)}
            />
          </label>
          <div className="drop-actions">
            <button
              className="secondary"
              disabled={busy || !name.trim()}
              onClick={() =>
                void perform("Renaming project…", async () => {
                  await post(`/projects/${project.id}/rename`, { name });
                  await refresh();
                })
              }
            >
              Rename project
            </button>
            <button
              className="text-button"
              disabled={busy}
              onClick={() => setConfirm(!confirm)}
            >
              Archive project
            </button>
          </div>
          {confirm && (
            <div className="alert notice">
              <p>
                Hide this project from the workspace? Documents and results stay
                on disk and can be restored below.
              </p>
              <button
                className="secondary"
                disabled={busy}
                onClick={() =>
                  void perform("Archiving project…", async () => {
                    await post(`/projects/${project.id}/archive`, {
                      archived: true,
                    });
                    await refresh();
                    await load();
                    setConfirm(false);
                  })
                }
              >
                Confirm archive
              </button>
              <button className="text-button" onClick={() => setConfirm(false)}>
                Keep project
              </button>
            </div>
          )}
          <button
            className="text-button"
            disabled={busy}
            onClick={() => setDeleteConfirm(!deleteConfirm)}
          >
            Delete project
          </button>
          {deleteConfirm && (
            <div className="alert notice">
              <p>
                Move {project.name} and its documents, runs, saved views, and
                questions to Trash?
              </p>
              <button
                className="secondary"
                disabled={busy}
                onClick={() =>
                  void perform("Moving project to Trash…", async () => {
                    await post(`/projects/${project.id}/trash`, {});
                    await refresh();
                    await loadTrash();
                    setDeleteConfirm(false);
                  })
                }
              >
                Move project to Trash
              </button>
              <button
                className="text-button"
                onClick={() => setDeleteConfirm(false)}
              >
                Keep project
              </button>
            </div>
          )}
        </>
      )}
      <h3>Deleted projects</h3>
      {trashed.length ? (
        trashed.map((item) => (
          <div className="dependency-row" key={item.id}>
            <span>
              {item.name}
              <small>{item.documents} documents in Trash</small>
            </span>
            <button
              className="secondary"
              disabled={busy}
              onClick={() =>
                void perform("Restoring project…", async () => {
                  await post(`/projects/${item.id}/restore`, {});
                  if (item.archived)
                    await post(`/projects/${item.id}/archive`, {
                      archived: false,
                    });
                  await refresh(item.id);
                  await loadTrash();
                  await load();
                })
              }
            >
              Restore
            </button>
            <button
              className="text-button"
              disabled={busy}
              onClick={() => {
                if (
                  window.confirm(
                    `Permanently delete ${item.name} and all its data?`,
                  )
                )
                  void perform("Permanently deleting project…", async () => {
                    await post(`/projects/${item.id}/purge`, {});
                    await loadTrash();
                  });
              }}
            >
              Delete permanently
            </button>
          </div>
        ))
      ) : (
        <p className="muted">No deleted projects.</p>
      )}
      <h3>Archived projects</h3>
      {archived.length ? (
        archived.map((item) => (
          <div className="dependency-row" key={item.id}>
            <span>
              {item.name}
              <small>{item.documents} documents preserved</small>
            </span>
            <button
              className="secondary"
              disabled={busy}
              onClick={() =>
                void perform("Restoring archived project…", async () => {
                  await post(`/projects/${item.id}/archive`, {
                    archived: false,
                  });
                  await refresh();
                  await load();
                })
              }
            >
              Restore
            </button>
            <button
              className="text-button"
              disabled={busy}
              onClick={() => {
                if (
                  window.confirm(`Move archived project ${item.name} to Trash?`)
                ) {
                  void perform("Moving project to Trash…", async () => {
                    await post(`/projects/${item.id}/trash`, {});
                    await load();
                    await loadTrash();
                  });
                }
              }}
            >
              Delete project
            </button>
          </div>
        ))
      ) : (
        <p className="muted">
          No archived projects. Archiving never deletes source files or results.
        </p>
      )}
    </section>
  );
}
