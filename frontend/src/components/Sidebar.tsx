/** Left sidebar: upload, document list, selection, status and deletion. */

import { useCallback, useRef, useState } from 'react';
import type { DocumentInfo, HealthResponse } from '../lib/types';
import { formatBytes, formatNumber } from '../lib/format';
import {
  IconCheck,
  IconFile,
  IconLogo,
  IconMoon,
  IconSpinner,
  IconSun,
  IconTrash,
  IconUpload,
} from './Icons';

interface SidebarProps {
  documents: DocumentInfo[];
  selectedIds: string[];
  health: HealthResponse | null;
  uploading: boolean;
  open: boolean;
  theme: 'light' | 'dark';
  onUpload: (files: FileList) => void;
  onToggleSelect: (documentId: string) => void;
  onDelete: (documentId: string) => void;
  onToggleTheme: () => void;
}

const STATUS_TEXT: Record<string, string> = {
  uploaded: 'Queued',
  processing: 'Processing',
  indexed: 'Indexed',
  summarizing: 'Summarizing',
  ready: 'Ready',
  failed: 'Failed',
};

function StatusPill({ status }: { status: string }) {
  return (
    <span className="status" data-state={status}>
      <span className="status__dot" />
      {STATUS_TEXT[status] ?? status}
    </span>
  );
}

export function Sidebar({
  documents,
  selectedIds,
  health,
  uploading,
  open,
  theme,
  onUpload,
  onToggleSelect,
  onDelete,
  onToggleTheme,
}: SidebarProps) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const accept = health?.supported_extensions?.join(',') ?? undefined;

  const handleDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      setDragging(false);
      if (uploading) return;
      if (event.dataTransfer.files?.length) onUpload(event.dataTransfer.files);
    },
    [onUpload, uploading],
  );

  return (
    <aside className="sidebar" data-open={open}>
      <div className="sidebar__header">
        <div className="brand">
          <span className="brand__mark">
            <IconLogo size={17} />
          </span>
          <span>
            <span className="brand__name">DocuMind</span>
            <br />
            <span className="brand__tag">RAG document chat</span>
          </span>
        </div>
        <button
          type="button"
          className="icon-btn"
          onClick={onToggleTheme}
          title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
          aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
        >
          {theme === 'dark' ? <IconSun /> : <IconMoon />}
        </button>
      </div>

      <div className="sidebar__body">
        {/* ---------- Upload ---------- */}
        <div
          className="dropzone"
          data-dragging={dragging}
          data-busy={uploading}
          role="button"
          tabIndex={0}
          onClick={() => !uploading && inputRef.current?.click()}
          onKeyDown={(event) => {
            if ((event.key === 'Enter' || event.key === ' ') && !uploading) {
              event.preventDefault();
              inputRef.current?.click();
            }
          }}
          onDragOver={(event) => {
            event.preventDefault();
            if (!uploading) setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
        >
          {uploading ? <IconSpinner size={20} /> : <IconUpload size={20} />}
          <span className="dropzone__title">
            {uploading ? 'Processing document…' : 'Upload a document'}
          </span>
          <span className="dropzone__hint">
            {uploading
              ? 'Extracting, chunking and embedding'
              : 'Drag and drop, or click to browse'}
          </span>
          <input
            ref={inputRef}
            type="file"
            multiple
            accept={accept}
            onChange={(event) => {
              if (event.target.files?.length) onUpload(event.target.files);
              event.target.value = '';
            }}
          />
        </div>

        {/* ---------- Document list ---------- */}
        <div className="section-label">
          <span>Documents</span>
          {documents.length > 0 && (
            <span>
              {selectedIds.length}/{documents.length} selected
            </span>
          )}
        </div>

        {documents.length === 0 ? (
          <p className="empty-note">
            No documents yet.
            <br />
            Upload a file to start asking questions.
          </p>
        ) : (
          <ul className="doc-list">
            {documents.map((doc) => {
              const selected = selectedIds.includes(doc.document_id);
              const failed = doc.status === 'failed';
              return (
                <li key={doc.document_id}>
                  <div
                    className="doc-card"
                    data-selected={selected}
                    data-failed={failed}
                    role="button"
                    tabIndex={0}
                    aria-pressed={selected}
                    onClick={() => !failed && onToggleSelect(doc.document_id)}
                    onKeyDown={(event) => {
                      if ((event.key === 'Enter' || event.key === ' ') && !failed) {
                        event.preventDefault();
                        onToggleSelect(doc.document_id);
                      }
                    }}
                    title={failed ? doc.error ?? 'Processing failed' : doc.filename}
                  >
                    <span className="doc-card__check" aria-hidden="true">
                      {selected ? <IconCheck /> : null}
                    </span>

                    <div className="doc-card__body">
                      <div className="doc-card__name">{doc.filename}</div>
                      <div className="doc-card__meta">
                        <span className="type-badge">{doc.file_type}</span>
                        <StatusPill status={doc.status} />
                      </div>
                      <div className="doc-card__meta">
                        <span>{formatBytes(doc.size_bytes)}</span>
                        {doc.chunk_count > 0 && (
                          <>
                            <span>·</span>
                            <span>{formatNumber(doc.chunk_count)} chunks</span>
                          </>
                        )}
                        {doc.page_count ? (
                          <>
                            <span>·</span>
                            <span>{doc.page_count} pages</span>
                          </>
                        ) : null}
                      </div>
                    </div>

                    <div className="doc-card__actions">
                      <button
                        type="button"
                        className="icon-btn icon-btn--danger"
                        title="Delete document"
                        aria-label={`Delete ${doc.filename}`}
                        onClick={(event) => {
                          event.stopPropagation();
                          onDelete(doc.document_id);
                        }}
                      >
                        <IconTrash size={14} />
                      </button>
                    </div>
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </div>

      {/* ---------- Footer ---------- */}
      <div className="sidebar__footer">
        {health ? (
          <>
            <div className="sidebar__footer-row">
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}>
                <IconFile size={11} />
                {formatNumber(health.documents_indexed)} indexed
              </span>
              <span className="status" data-state={health.llm_configured ? 'ready' : 'failed'}>
                <span className="status__dot" />
                {health.llm_configured ? 'Groq connected' : 'No API key'}
              </span>
            </div>
            <div className="sidebar__footer-row">
              <span>Model</span>
              <code title={health.llm_model}>{health.llm_model}</code>
            </div>
            <div className="sidebar__footer-row">
              <span>Vector store</span>
              <code>{health.vector_store}</code>
            </div>
          </>
        ) : (
          <span>Connecting to the backend…</span>
        )}
      </div>
    </aside>
  );
}
