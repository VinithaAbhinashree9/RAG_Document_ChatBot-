/** Application shell: state, data fetching and layout composition. */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api, ApiError, streamChat } from './lib/api';
import type { DocumentInfo, HealthResponse, Message } from './lib/types';
import { ChatMessage } from './components/ChatMessage';
import { Composer } from './components/Composer';
import { SearchPanel } from './components/SearchPanel';
import { Sidebar } from './components/Sidebar';
import { SummaryPanel } from './components/SummaryPanel';
import {
  IconAlert,
  IconClose,
  IconLogo,
  IconMenu,
  IconSearch,
  IconTrash,
} from './components/Icons';

const SESSION_KEY = 'documind.session';
const THEME_KEY = 'documind.theme';

const SUGGESTIONS = [
  'What is this document about?',
  'Summarize the main points.',
  'What are the key findings?',
  'What are the important dates?',
  'What are the recommendations?',
  'Who are the main people mentioned?',
];

function createId(): string {
  return `m_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}

function loadSessionId(): string {
  const existing = localStorage.getItem(SESSION_KEY);
  if (existing) return existing;
  const created = `s_${Math.random().toString(36).slice(2, 12)}`;
  localStorage.setItem(SESSION_KEY, created);
  return created;
}

function loadTheme(): 'light' | 'dark' {
  const stored = localStorage.getItem(THEME_KEY);
  if (stored === 'light' || stored === 'dark') return stored;
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

export default function App() {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [health, setHealth] = useState<HealthResponse | null>(null);

  const [input, setInput] = useState('');
  const [uploading, setUploading] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [regenerating, setRegenerating] = useState(false);
  const [notice, setNotice] = useState<{ kind: 'warning' | 'danger'; text: string } | null>(
    null,
  );
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const [theme, setTheme] = useState<'light' | 'dark'>(loadTheme);

  const sessionId = useRef(loadSessionId());
  const abortRef = useRef<(() => void) | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  // ---------------- Theme ----------------
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem(THEME_KEY, theme);
  }, [theme]);

  // ---------------- Initial load ----------------
  const refresh = useCallback(async () => {
    try {
      const [healthData, docsData] = await Promise.all([
        api.health(),
        api.listDocuments(),
      ]);
      setHealth(healthData);
      setDocuments(docsData.documents);

      // Default to selecting every usable document.
      setSelectedIds((current) => {
        const usable = docsData.documents
          .filter((doc) => doc.status !== 'failed')
          .map((doc) => doc.document_id);
        const stillValid = current.filter((id) => usable.includes(id));
        return stillValid.length > 0 ? stillValid : usable;
      });

      if (!healthData.llm_configured) {
        setNotice({
          kind: 'warning',
          text:
            'GROQ_API_KEY is not configured on the server. Documents will upload and index, but summaries and answers require a key. Add it to backend/.env and restart the backend.',
        });
      }
    } catch (error) {
      setNotice({
        kind: 'danger',
        text:
          error instanceof ApiError
            ? error.message
            : 'Could not reach the backend. Confirm it is running.',
      });
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // Keep the transcript pinned to the newest message.
  useEffect(() => {
    const element = scrollRef.current;
    if (element) element.scrollTop = element.scrollHeight;
  }, [messages]);

  // ---------------- Derived ----------------
  const activeDocuments = useMemo(
    () => documents.filter((doc) => selectedIds.includes(doc.document_id)),
    [documents, selectedIds],
  );

  const summaryDocument = activeDocuments.length === 1 ? activeDocuments[0] : null;

  const scopeLabel = useMemo(() => {
    if (activeDocuments.length === 0) return 'no document selected';
    if (activeDocuments.length === 1) return activeDocuments[0].filename;
    return `${activeDocuments.length} documents`;
  }, [activeDocuments]);

  const chatDisabled = activeDocuments.length === 0 || uploading;

  // ---------------- Uploads ----------------
  const handleUpload = useCallback(
    async (files: FileList) => {
      setUploading(true);
      setNotice(null);
      const failures: string[] = [];

      for (const file of Array.from(files)) {
        try {
          const response = await api.uploadDocument(file);
          setDocuments((current) => [response.document, ...current]);
          setSelectedIds([response.document.document_id]);
          if (response.summary_error) {
            failures.push(`${file.name}: ${response.summary_error}`);
          }
        } catch (error) {
          failures.push(
            `${file.name}: ${
              error instanceof ApiError ? error.message : 'Upload failed.'
            }`,
          );
        }
      }

      setUploading(false);
      setSidebarOpen(false);
      if (failures.length > 0) {
        setNotice({ kind: 'warning', text: failures.join(' · ') });
      }
      void refresh();
    },
    [refresh],
  );

  const handleDelete = useCallback(async (documentId: string) => {
    try {
      await api.deleteDocument(documentId);
      setDocuments((current) =>
        current.filter((doc) => doc.document_id !== documentId),
      );
      setSelectedIds((current) => current.filter((id) => id !== documentId));
      setSearchOpen(false);
    } catch (error) {
      setNotice({
        kind: 'danger',
        text:
          error instanceof ApiError ? error.message : 'The document could not be deleted.',
      });
    }
  }, []);

  const handleRegenerateSummary = useCallback(async () => {
    if (!summaryDocument) return;
    setRegenerating(true);
    try {
      const response = await api.summarize(summaryDocument.document_id, true);
      setDocuments((current) =>
        current.map((doc) =>
          doc.document_id === response.document_id
            ? { ...doc, summary: response.summary, status: 'ready' }
            : doc,
        ),
      );
    } catch (error) {
      setNotice({
        kind: 'danger',
        text:
          error instanceof ApiError
            ? error.message
            : 'The summary could not be regenerated.',
      });
    } finally {
      setRegenerating(false);
    }
  }, [summaryDocument]);

  // ---------------- Chat ----------------
  const send = useCallback(
    (question: string) => {
      const trimmed = question.trim();
      if (!trimmed || streaming || activeDocuments.length === 0) return;

      const documentIds = activeDocuments.map((doc) => doc.document_id);
      const assistantId = createId();

      setMessages((current) => [
        ...current,
        { id: createId(), role: 'user', content: trimmed, createdAt: new Date().toISOString() },
        {
          id: assistantId,
          role: 'assistant',
          content: '',
          streaming: true,
          question: trimmed,
          createdAt: new Date().toISOString(),
        },
      ]);
      setInput('');
      setStreaming(true);

      const update = (patch: Partial<Message>) =>
        setMessages((current) =>
          current.map((message) =>
            message.id === assistantId ? { ...message, ...patch } : message,
          ),
        );

      abortRef.current = streamChat(
        { message: trimmed, document_ids: documentIds, session_id: sessionId.current },
        {
          onToken: (token) =>
            setMessages((current) =>
              current.map((message) =>
                message.id === assistantId
                  ? { ...message, content: message.content + token }
                  : message,
              ),
            ),
          onSources: (sources) => update({ sources }),
          onDone: ({ grounded }) => {
            update({ streaming: false, grounded });
            setStreaming(false);
            abortRef.current = null;
          },
          onError: (message) => {
            setMessages((current) =>
              current.map((entry) =>
                entry.id === assistantId
                  ? {
                      ...entry,
                      content: entry.content || message,
                      streaming: false,
                      error: !entry.content,
                    }
                  : entry,
              ),
            );
            setStreaming(false);
            abortRef.current = null;
          },
        },
      );
    },
    [activeDocuments, streaming],
  );

  const stop = useCallback(() => {
    abortRef.current?.();
    abortRef.current = null;
    setStreaming(false);
    setMessages((current) =>
      current.map((message) =>
        message.streaming ? { ...message, streaming: false } : message,
      ),
    );
  }, []);

  const regenerate = useCallback(
    (message: Message) => {
      if (!message.question || streaming) return;
      // Drop the previous answer and its question, then re-ask.
      setMessages((current) => {
        const index = current.findIndex((entry) => entry.id === message.id);
        return index > 0 ? current.slice(0, index - 1) : current;
      });
      setTimeout(() => send(message.question as string), 0);
    },
    [send, streaming],
  );

  const clearConversation = useCallback(async () => {
    stop();
    try {
      await api.clearChat(sessionId.current);
    } catch {
      // Clearing server memory is best-effort; the UI resets regardless.
    }
    setMessages([]);
  }, [stop]);

  // ---------------- Render ----------------
  return (
    <div className="app">
      <div
        className="scrim"
        data-open={sidebarOpen}
        onClick={() => setSidebarOpen(false)}
      />

      <Sidebar
        documents={documents}
        selectedIds={selectedIds}
        health={health}
        uploading={uploading}
        open={sidebarOpen}
        theme={theme}
        onUpload={handleUpload}
        onToggleSelect={(id) =>
          setSelectedIds((current) =>
            current.includes(id)
              ? current.filter((value) => value !== id)
              : [...current, id],
          )
        }
        onDelete={handleDelete}
        onToggleTheme={() => setTheme((value) => (value === 'dark' ? 'light' : 'dark'))}
      />

      <main className="main">
        <header className="topbar">
          <button
            type="button"
            className="icon-btn menu-btn"
            onClick={() => setSidebarOpen(true)}
            aria-label="Open document sidebar"
          >
            <IconMenu />
          </button>

          <div className="topbar__title">
            <h1>{activeDocuments.length > 0 ? scopeLabel : 'DocuMind'}</h1>
            <p>
              {activeDocuments.length > 1
                ? 'Cross-document question answering'
                : activeDocuments.length === 1
                  ? `${activeDocuments[0].chunk_count} indexed chunks · answers cite this document`
                  : 'Upload a document to begin'}
            </p>
          </div>

          <div className="topbar__actions">
            {summaryDocument && (
              <button
                type="button"
                className="btn btn--ghost"
                onClick={() => setSearchOpen((value) => !value)}
                title="Search inside this document"
              >
                <IconSearch size={14} />
                Search
              </button>
            )}
            {messages.length > 0 && (
              <button
                type="button"
                className="btn btn--ghost"
                onClick={clearConversation}
                title="Clear the conversation"
              >
                <IconTrash size={14} />
                Clear chat
              </button>
            )}
          </div>
        </header>

        <div className="content" ref={scrollRef}>
          <div className="content__inner">
            {notice && (
              <div className={`banner banner--${notice.kind}`} role="status">
                <IconAlert />
                <p>{notice.text}</p>
                <button
                  type="button"
                  className="banner__close"
                  onClick={() => setNotice(null)}
                  aria-label="Dismiss message"
                >
                  <IconClose size={13} />
                </button>
              </div>
            )}

            {searchOpen && summaryDocument && (
              <SearchPanel
                documentId={summaryDocument.document_id}
                filename={summaryDocument.filename}
                onClose={() => setSearchOpen(false)}
              />
            )}

            {summaryDocument && messages.length === 0 && (
              <SummaryPanel
                document={summaryDocument}
                regenerating={regenerating}
                onRegenerate={handleRegenerateSummary}
              />
            )}

            {messages.length === 0 ? (
              <div className="hero">
                <span className="hero__mark">
                  <IconLogo size={24} />
                </span>
                <h2>
                  {activeDocuments.length > 0
                    ? 'Ask anything about your documents'
                    : 'Upload a document to get started'}
                </h2>
                <p>
                  {activeDocuments.length > 0
                    ? 'Every answer is generated only from the content of your selected documents, with citations back to the exact source passage.'
                    : 'Drag a PDF, Word, Excel, PowerPoint, CSV, Markdown or text file into the sidebar. You will get an automatic summary, then you can ask questions about it.'}
                </p>

                {activeDocuments.length > 0 && (
                  <div className="suggestions">
                    {SUGGESTIONS.map((suggestion) => (
                      <button
                        type="button"
                        className="suggestion"
                        key={suggestion}
                        onClick={() => send(suggestion)}
                      >
                        {suggestion}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <div className="messages">
                {messages.map((message, index) => (
                  <ChatMessage
                    key={message.id}
                    message={message}
                    canRegenerate={
                      message.role === 'assistant' &&
                      index === messages.length - 1 &&
                      !streaming
                    }
                    onRegenerate={() => regenerate(message)}
                  />
                ))}
              </div>
            )}
          </div>
        </div>

        <Composer
          value={input}
          disabled={chatDisabled}
          streaming={streaming}
          scopeLabel={scopeLabel}
          onChange={setInput}
          onSubmit={() => send(input)}
          onStop={stop}
        />
      </main>
    </div>
  );
}
