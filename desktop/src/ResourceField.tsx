import { useRef, useState } from "react";
import { api } from "./api";

export function ResourceField({
  value,
  onChange,
  onBusyChange,
}: {
  value: string;
  onChange: (value: string) => void;
  onBusyChange: (busy: boolean) => void;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [name, setName] = useState("");
  return (
    <div>
      <div className="drop-actions">
        <button
          type="button"
          className="secondary"
          disabled={busy}
          onClick={() => input.current?.click()}
        >
          {busy
            ? "Importing…"
            : value
              ? "Replace file"
              : "Choose resource file"}
        </button>
        {value && (
          <button
            type="button"
            className="text-button"
            disabled={busy}
            onClick={() => {
              onChange("");
              setName("");
            }}
          >
            Clear
          </button>
        )}
      </div>
      <small>
        {name || (value ? "Resource selected" : "No resource selected")}
      </small>
      {error && <p role="alert">{error}</p>}
      <input
        ref={input}
        type="file"
        accept=".txt,.csv,.tsv,.json"
        hidden
        onChange={async (e) => {
          const file = e.target.files?.[0];
          e.target.value = "";
          if (!file) return;
          setBusy(true);
          onBusyChange(true);
          setError("");
          try {
            const result = await api<{ path: string; name: string }>(
              `/resources?name=${encodeURIComponent(file.name)}`,
              { method: "POST", body: file },
            );
            onChange(result.path);
            setName(result.name);
          } catch (err) {
            setError(String(err));
          } finally {
            setBusy(false);
            onBusyChange(false);
          }
        }}
      />
    </div>
  );
}
