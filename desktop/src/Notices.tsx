import { useState } from "react";
import { api } from "./api";

export function Notices() {
  const [text, setText] = useState("");
  return (
    <section className="panel settings-panel">
      <h2>Credits & licenses</h2>
      <p>
        NLP Suite uses open-source analysis libraries and language models. Each
        keeps its own license. Restricted research lexicons remain
        user-supplied; imported documents are not redistributed with the app.
      </p>
      <button
        className="secondary"
        onClick={() =>
          void api<{ text: string }>("/notices")
            .then((value) => setText(value.text))
            .catch((error) => setText(String(error)))
        }
      >
        Read dependency notices
      </button>
      {text && (
        <pre
          style={{
            whiteSpace: "pre-wrap",
            maxHeight: 360,
            overflow: "auto",
            fontSize: 12,
          }}
        >
          {text}
        </pre>
      )}
    </section>
  );
}
