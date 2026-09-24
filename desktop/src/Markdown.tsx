import { Fragment, type ReactNode } from "react";

/**
 * The Markdown this app publishes, rendered.
 *
 * `readout.md` — the plain-language reading of a result — was being served
 * into an iframe like any other artifact, so the most reader-facing file a run
 * produces was displayed with its asterisks and hyphens showing. It is the one
 * artifact whose whole purpose is to be read rather than opened elsewhere.
 *
 * This covers the Markdown `Readout.to_markdown` actually emits: a headline
 * paragraph, bold section headings on a line of their own, and bullet lists.
 * Headings and inline code are handled too, since a report written by hand
 * could contain them. Anything else is shown as its own literal text rather
 * than silently dropped — a renderer that quietly swallows what it does not
 * understand is worse than one that shows it.
 *
 * Everything is built as React elements, never as HTML, so nothing in an
 * artifact can become markup.
 */

/** Split on `**bold**` and `` `code` ``, keeping the surrounding text. */
function inline(text: string, keyPrefix: string): ReactNode[] {
  const parts: ReactNode[] = [];
  const pattern = /\*\*(.+?)\*\*|`([^`]+)`/g;
  let last = 0;
  let match: RegExpExecArray | null;
  let index = 0;
  while ((match = pattern.exec(text)) !== null) {
    if (match.index > last) parts.push(text.slice(last, match.index));
    const key = `${keyPrefix}-${index++}`;
    if (match[1] !== undefined)
      parts.push(<strong key={key}>{match[1]}</strong>);
    else parts.push(<code key={key}>{match[2]}</code>);
    last = match.index + match[0].length;
  }
  if (last < text.length) parts.push(text.slice(last));
  return parts;
}

/** A line that is nothing but bold text is a section heading, not a sentence. */
const BOLD_LINE = /^\*\*(.+)\*\*$/;
const HEADING = /^(#{1,4})\s+(.*)$/;
const BULLET = /^[-*]\s+(.*)$/;

export function Markdown({ source }: { source: string }) {
  const blocks: ReactNode[] = [];
  const lines = source.replace(/\r\n?/g, "\n").split("\n");
  let bullets: string[] = [];
  let paragraph: string[] = [];

  const flushBullets = () => {
    if (!bullets.length) return;
    const items = bullets;
    bullets = [];
    blocks.push(
      <ul key={`ul-${blocks.length}`}>
        {items.map((item, i) => (
          <li key={i}>{inline(item, `li-${blocks.length}-${i}`)}</li>
        ))}
      </ul>,
    );
  };
  const flushParagraph = () => {
    if (!paragraph.length) return;
    const text = paragraph.join(" ");
    paragraph = [];
    blocks.push(
      <p key={`p-${blocks.length}`}>{inline(text, `p-${blocks.length}`)}</p>,
    );
  };
  const flush = () => {
    flushBullets();
    flushParagraph();
  };

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) {
      flush();
      continue;
    }
    const heading = HEADING.exec(trimmed);
    if (heading) {
      flush();
      const level = Math.min(heading[1].length + 2, 6);
      const Tag = `h${level}` as "h3" | "h4" | "h5" | "h6";
      blocks.push(
        <Tag key={`h-${blocks.length}`}>
          {inline(heading[2], `h-${blocks.length}`)}
        </Tag>,
      );
      continue;
    }
    const bold = BOLD_LINE.exec(trimmed);
    if (bold) {
      flush();
      blocks.push(<h4 key={`b-${blocks.length}`}>{bold[1]}</h4>);
      continue;
    }
    const bullet = BULLET.exec(trimmed);
    if (bullet) {
      flushParagraph();
      bullets.push(bullet[1]);
      continue;
    }
    flushBullets();
    paragraph.push(trimmed);
  }
  flush();

  return (
    <div className="markdown-view">
      {blocks.map((block, i) => (
        <Fragment key={i}>{block}</Fragment>
      ))}
    </div>
  );
}
