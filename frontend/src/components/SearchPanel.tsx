/** In-document semantic search over the exact indexed chunks. */

import { useState } from 'react';
import { api, ApiError } from '../lib/api';
import type { ChunkInfo } from '../lib/types';
import { IconClose, IconSearch, IconSpinner } from './Icons';

interface SearchPanelProps {
  documentId: string;
  filename: string;
  onClose: () => void;
}

export function SearchPanel({ documentId, filename, onClose }: SearchPanelProps) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<ChunkInfo[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runSearch = async () => {
    const trimmed = query.trim();
    if (!trimmed) return;
    setLoading(true);
    setError(null);
    try {
      const response = await api.searchDocument(documentId, trimmed, 15);
      setResults(response.chunks);
    } catch (err) {
      setError(
        err instanceof ApiError ? err.message : 'The search could not be completed.',
      );
      setResults(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="panel">
      <div className="panel__header">
        <h2>
          <IconSearch size={13} />
          Search in {filename}
        </h2>
        <button
          type="button"
          className="icon-btn"
          onClick={onClose}
          aria-label="Close search"
        >
          <IconClose />
        </button>
      </div>

      <div className="panel__body">
        <div className="search-field">
          <IconSearch size={14} />
          <input
            type="search"
            value={query}
            placeholder="Find passages by meaning, not just keywords…"
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter') runSearch();
            }}
            aria-label="Search query"
          />
          {loading && <IconSpinner size={14} />}
        </div>

        {error && (
          <p style={{ fontSize: 12, color: 'var(--danger)' }}>{error}</p>
        )}

        {results && results.length === 0 && (
          <p style={{ fontSize: 12.5, color: 'var(--text-tertiary)' }}>
            No matching passages were found.
          </p>
        )}

        {results && results.length > 0 && (
          <div className="search-results">
            {results.map((chunk) => (
              <div className="search-hit" key={chunk.chunk_id}>
                <div className="search-hit__head">
                  {chunk.page !== null && <span className="badge">Page {chunk.page}</span>}
                  {chunk.section && <span className="badge">{chunk.section}</span>}
                  <span>Chunk {chunk.chunk_index + 1}</span>
                </div>
                <p className="search-hit__text">
                  {chunk.text.length > 420
                    ? `${chunk.text.slice(0, 420).trimEnd()}…`
                    : chunk.text}
                </p>
              </div>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}
