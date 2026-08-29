/** Small formatting and rendering helpers. */

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function formatNumber(value: number): string {
  return value.toLocaleString('en-US');
}

export function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/**
 * Render the small Markdown subset the backend prompts emit (headings, bullets,
 * bold, italics, inline code) plus `[S1]` citation markers.
 *
 * Input is HTML-escaped *before* any tag is inserted, so model output cannot
 * inject markup. This avoids shipping a full Markdown parser for a known,
 * narrow format.
 */
export function renderMarkdown(source: string): string {
  const escaped = escapeHtml(source);
  const lines = escaped.split('\n');
  const html: string[] = [];
  let inList = false;

  const closeList = () => {
    if (inList) {
      html.push('</ul>');
      inList = false;
    }
  };

  const inline = (text: string): string =>
    text
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      .replace(/(^|[\s(])\*([^*\n]+)\*(?=[\s.,;:)]|$)/g, '$1<em>$2</em>')
      .replace(/(^|[\s(])_([^_\n]+)_(?=[\s.,;:)]|$)/g, '$1<em>$2</em>')
      // Citation markers, e.g. [S1] or [S1, S3]
      .replace(/\[\s*(S\d+(?:\s*,\s*S\d+)*)\s*\]/gi, (_m, group: string) =>
        group
          .split(',')
          .map((marker) => `<span class="cite-marker">${marker.trim().toUpperCase()}</span>`)
          .join(''),
      );

  for (const line of lines) {
    const trimmed = line.trim();

    if (!trimmed) {
      closeList();
      continue;
    }

    const heading = /^(#{1,6})\s+(.*)$/.exec(trimmed);
    if (heading) {
      closeList();
      const level = Math.min(heading[1].length + 1, 4);
      html.push(`<h${level}>${inline(heading[2])}</h${level}>`);
      continue;
    }

    const bullet = /^[-*+]\s+(.*)$/.exec(trimmed);
    if (bullet) {
      if (!inList) {
        html.push('<ul>');
        inList = true;
      }
      html.push(`<li>${inline(bullet[1])}</li>`);
      continue;
    }

    const numbered = /^\d+[.)]\s+(.*)$/.exec(trimmed);
    if (numbered) {
      if (!inList) {
        html.push('<ul>');
        inList = true;
      }
      html.push(`<li>${inline(numbered[1])}</li>`);
      continue;
    }

    closeList();
    html.push(`<p>${inline(trimmed)}</p>`);
  }

  closeList();
  return html.join('');
}

/** Trigger a client-side file download for generated text (e.g. a summary). */
export function downloadTextFile(filename: string, content: string): void {
  const blob = new Blob([content], { type: 'text/markdown;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}
