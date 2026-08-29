/** Automatic document summary, with download and regenerate actions. */

import type { DocumentInfo } from '../lib/types';
import { downloadTextFile, renderMarkdown } from '../lib/format';
import { IconDownload, IconRefresh, IconSpinner, IconSparkle } from './Icons';

interface SummaryPanelProps {
  document: DocumentInfo;
  regenerating: boolean;
  onRegenerate: () => void;
}

export function SummaryPanel({
  document: doc,
  regenerating,
  onRegenerate,
}: SummaryPanelProps) {
  const hasSummary = Boolean(doc.summary?.trim());

  return (
    <section className="panel">
      <div className="panel__header">
        <h2>
          <IconSparkle size={14} />
          Document summary
          {doc.page_count ? (
            <span className="badge">{doc.page_count} pages</span>
          ) : null}
          <span className="badge">{doc.chunk_count} chunks</span>
        </h2>

        {hasSummary && (
          <button
            type="button"
            className="icon-btn"
            title="Download summary as Markdown"
            aria-label="Download summary"
            onClick={() =>
              downloadTextFile(
                `${doc.filename.replace(/\.[^.]+$/, '')}-summary.md`,
                `# Summary — ${doc.filename}\n\n${doc.summary ?? ''}\n`,
              )
            }
          >
            <IconDownload size={14} />
          </button>
        )}
        <button
          type="button"
          className="icon-btn"
          title="Regenerate summary"
          aria-label="Regenerate summary"
          disabled={regenerating}
          onClick={onRegenerate}
        >
          {regenerating ? <IconSpinner size={14} /> : <IconRefresh size={14} />}
        </button>
      </div>

      <div className="panel__body">
        {regenerating && !hasSummary ? (
          <div className="typing" aria-label="Generating summary">
            <span />
            <span />
            <span />
          </div>
        ) : hasSummary ? (
          <div
            className="md"
            // Content is HTML-escaped by renderMarkdown before any tag is added.
            dangerouslySetInnerHTML={{ __html: renderMarkdown(doc.summary as string) }}
          />
        ) : (
          <p style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
            No summary is available for this document yet. Use the regenerate button
            to create one.
          </p>
        )}

        {doc.warnings.length > 0 && (
          <div style={{ marginTop: 14, fontSize: 11.5, color: 'var(--text-tertiary)' }}>
            {doc.warnings.map((warning, index) => (
              <div key={index}>· {warning}</div>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}
