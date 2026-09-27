import { useEffect, useRef } from "react";
import { EditorState, Prec, type Extension } from "@codemirror/state";
import {
  EditorView,
  drawSelection,
  highlightActiveLine,
  keymap,
  lineNumbers,
  placeholder as placeholderText,
} from "@codemirror/view";
import {
  defaultKeymap,
  history,
  historyKeymap,
  indentWithTab,
} from "@codemirror/commands";
import {
  bracketMatching,
  defaultHighlightStyle,
  indentOnInput,
  syntaxHighlighting,
} from "@codemirror/language";
import { python } from "@codemirror/lang-python";
import {
  autocompletion,
  closeBrackets,
  closeBracketsKeymap,
  completionKeymap,
  type CompletionContext,
  type CompletionResult,
} from "@codemirror/autocomplete";

import type { LibraryFunction } from "./notebooks";

/**
 * Completion without a running kernel: `nlp.` offers the library's functions
 * (from the same reference the side panel shows), and a quoted string inside
 * `filter(` / `where(` / `x=` / `y=` offers the corpus's column names -- the
 * two places a researcher most often has to go and look a name up.
 */
function completions(functions: LibraryFunction[], columns: string[]) {
  return (context: CompletionContext): CompletionResult | null => {
    const member = context.matchBefore(/\bnlp\.\w*/);
    if (member) {
      return {
        from: member.from + 4,
        options: functions.map((item) => ({
          label: item.name,
          type: item.kind === "class" ? "class" : "function",
          detail: item.signature
            .replace(/^nlp\./, "")
            .slice(item.name.length, 80),
          info: item.doc.split("\n\n")[0],
        })),
        validFor: /^\w*$/,
      };
    }
    const quoted = context.matchBefore(/["'][^"'\n]*/);
    if (quoted && columns.length) {
      const line = context.state.doc.lineAt(context.pos);
      const before = line.text.slice(0, quoted.from - line.from);
      if (/(filter\(|where\(|\b(x|y|group|by)=|\[)\s*$/.test(before)) {
        return {
          from: quoted.from + 1,
          options: columns.map((column) => ({
            label: column,
            type: "property",
          })),
        };
      }
    }
    return null;
  };
}

/** Readable in the app's palette rather than the editor's default greys. */
const theme = EditorView.theme({
  "&": {
    fontSize: "13px",
    backgroundColor: "var(--surface-muted)",
    border: "1px solid var(--line)",
    borderRadius: "7px",
  },
  "&.cm-focused": { outline: "2px solid #22473c55", outlineOffset: "-1px" },
  ".cm-content": {
    fontFamily: "ui-monospace, 'Cascadia Code', Consolas, monospace",
    padding: "9px 0",
    caretColor: "var(--ink)",
  },
  ".cm-gutters": {
    backgroundColor: "transparent",
    border: "none",
    color: "var(--icon-muted)",
  },
  ".cm-activeLine": { backgroundColor: "#22473c0a" },
  ".cm-tooltip-autocomplete": { fontSize: "12px" },
});

export function CodeEditor({
  value,
  onChange,
  language,
  onRun,
  functions = [],
  columns = [],
  label,
  placeholder = "",
  focusSignal = 0,
  insert,
}: {
  value: string;
  onChange: (value: string) => void;
  language: "python" | "markdown";
  /** Shift+Enter runs and moves on; Ctrl+Enter (Cmd+Enter) runs in place. */
  onRun?: (advance: boolean) => void;
  functions?: LibraryFunction[];
  columns?: string[];
  label: string;
  placeholder?: string;
  /** Bumped by the parent to put the cursor in this editor. */
  focusSignal?: number;
  /** Text to put at the cursor; a new object each time it should happen. */
  insert?: { text: string } | null;
}) {
  const host = useRef<HTMLDivElement>(null);
  const view = useRef<EditorView | null>(null);
  // The editor is built once; the callbacks it calls are read through refs so
  // a new function each render neither rebuilds it nor calls a stale one.
  const latest = useRef({ onChange, onRun, functions, columns });
  latest.current = { onChange, onRun, functions, columns };

  useEffect(() => {
    const run = (advance: boolean) => () => {
      latest.current.onRun?.(advance);
      return true;
    };
    const extensions: Extension[] = [
      lineNumbers(),
      history(),
      drawSelection(),
      indentOnInput(),
      bracketMatching(),
      closeBrackets(),
      highlightActiveLine(),
      syntaxHighlighting(defaultHighlightStyle, { fallback: true }),
      Prec.highest(
        keymap.of([
          { key: "Shift-Enter", run: run(true) },
          { key: "Mod-Enter", run: run(false) },
        ]),
      ),
      keymap.of([
        ...closeBracketsKeymap,
        ...defaultKeymap,
        ...historyKeymap,
        ...completionKeymap,
        indentWithTab,
      ]),
      // Markdown cells are plain wrapped text: @codemirror/lang-markdown
      // brings the HTML, CSS and JavaScript parsers with it, which more than
      // doubled the app's bundle for a colouring nobody needs to write notes.
      ...(language === "python" ? [python()] : []),
      EditorView.lineWrapping,
      EditorView.contentAttributes.of({ "aria-label": label }),
      EditorView.updateListener.of((update) => {
        if (update.docChanged)
          latest.current.onChange(update.state.doc.toString());
      }),
      theme,
    ];
    if (placeholder) extensions.push(placeholderText(placeholder));
    if (language === "python")
      extensions.push(
        autocompletion({
          override: [
            (context) =>
              completions(
                latest.current.functions,
                latest.current.columns,
              )(context),
          ],
        }),
      );
    const editor = new EditorView({
      state: EditorState.create({ doc: value, extensions }),
      parent: host.current!,
    });
    view.current = editor;
    return () => {
      editor.destroy();
      view.current = null;
    };
    // Rebuilt only when the language changes; value is synced below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [language, label]);

  // A change from outside (Insert from the reference, a template, a revert)
  // replaces the text; the editor's own typing already matches and is skipped.
  useEffect(() => {
    const editor = view.current;
    if (!editor) return;
    const current = editor.state.doc.toString();
    if (current !== value)
      editor.dispatch({
        changes: { from: 0, to: current.length, insert: value },
      });
  }, [value]);

  useEffect(() => {
    if (focusSignal) view.current?.focus();
  }, [focusSignal]);

  // "Insert" from the library reference: at the cursor, then selected, so the
  // researcher sees exactly what arrived and can type over the arguments.
  useEffect(() => {
    const editor = view.current;
    if (!editor || !insert) return;
    const { from, to } = editor.state.selection.main;
    editor.dispatch({
      changes: { from, to, insert: insert.text },
      selection: { anchor: from, head: from + insert.text.length },
      scrollIntoView: true,
    });
    editor.focus();
  }, [insert]);

  return <div className="code-editor" ref={host} />;
}
