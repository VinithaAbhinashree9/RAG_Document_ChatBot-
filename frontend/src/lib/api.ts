/**
 * Typed API client.
 *
 * The API key is never held here: in development Vite proxies `/api` to the
 * backend, and in production the app is served from the same origin. Secrets
 * stay server-side.
 */

import type {
  ChatResponse,
  DocumentInfo,
  DocumentListResponse,
  HealthResponse,
  SourceCitation,
  SourcesResponse,
  SummarizeResponse,
  UploadResponse,
} from './types';

const BASE = import.meta.env.VITE_API_BASE_URL ?? '/api';

export class ApiError extends Error {
  code: string;
  status: number;
  requestId?: string;

  constructor(message: string, code: string, status: number, requestId?: string) {
    super(message);
    this.name = 'ApiError';
    this.code = code;
    this.status = status;
    this.requestId = requestId;
  }
}

async function parseError(response: Response): Promise<ApiError> {
  let message = `Request failed with status ${response.status}.`;
  let code = 'http_error';
  let requestId: string | undefined;
  try {
    const body = await response.json();
    if (body?.error) {
      message = body.error.message ?? message;
      code = body.error.code ?? code;
      requestId = body.error.request_id;
    }
  } catch {
    // Non-JSON error body (e.g. a proxy failure) — keep the generic message.
  }
  return new ApiError(message, code, response.status, requestId);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, {
      ...init,
      headers: {
        Accept: 'application/json',
        ...(init?.body instanceof FormData
          ? {}
          : { 'Content-Type': 'application/json' }),
        ...init?.headers,
      },
    });
  } catch {
    throw new ApiError(
      'Could not reach the server. Confirm the backend is running.',
      'network_error',
      0,
    );
  }

  if (!response.ok) throw await parseError(response);
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export const api = {
  health: () => request<HealthResponse>('/health'),

  listDocuments: () => request<DocumentListResponse>('/documents'),

  getDocument: (id: string) => request<DocumentInfo>(`/documents/${id}`),

  uploadDocument: (file: File) => {
    const form = new FormData();
    form.append('file', file);
    return request<UploadResponse>('/documents', { method: 'POST', body: form });
  },

  deleteDocument: (id: string) =>
    request<{ document_id: string; deleted: boolean; message: string }>(
      `/documents/${id}`,
      { method: 'DELETE' },
    ),

  summarize: (id: string, force = false) =>
    request<SummarizeResponse>(`/documents/${id}/summarize`, {
      method: 'POST',
      body: JSON.stringify({ force }),
    }),

  searchDocument: (id: string, query: string, limit = 20) =>
    request<SourcesResponse>(
      `/documents/${id}/sources?q=${encodeURIComponent(query)}&limit=${limit}`,
    ),

  listChunks: (id: string, limit = 50) =>
    request<SourcesResponse>(`/documents/${id}/sources?limit=${limit}`),

  chat: (payload: {
    message: string;
    document_ids: string[];
    session_id: string;
    top_k?: number;
  }) =>
    request<ChatResponse>('/chat', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  clearChat: (sessionId: string) =>
    request<{ session_id: string; cleared: boolean; message: string }>(
      '/chat/clear',
      { method: 'POST', body: JSON.stringify({ session_id: sessionId }) },
    ),
};

/** Server-sent-event callbacks for streaming chat. */
export interface StreamHandlers {
  onToken: (token: string) => void;
  onSources: (sources: SourceCitation[]) => void;
  onDone: (meta: { grounded: boolean; retrieved_chunks: number; model: string }) => void;
  onError: (message: string) => void;
}

/**
 * Stream a chat answer. Returns an abort function so the caller can cancel
 * an in-flight response.
 */
export function streamChat(
  payload: { message: string; document_ids: string[]; session_id: string; top_k?: number },
  handlers: StreamHandlers,
): () => void {
  const controller = new AbortController();

  (async () => {
    try {
      const response = await fetch(`${BASE}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
        body: JSON.stringify({ ...payload, stream: true }),
        signal: controller.signal,
      });

      if (!response.ok) {
        const error = await parseError(response);
        handlers.onError(error.message);
        return;
      }
      if (!response.body) {
        handlers.onError('The server returned an empty response stream.');
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // SSE frames are separated by a blank line.
        const frames = buffer.split('\n\n');
        buffer = frames.pop() ?? '';

        for (const frame of frames) {
          const line = frame.trim();
          if (!line.startsWith('data:')) continue;
          const raw = line.slice(5).trim();
          if (!raw || raw === '[DONE]') continue;

          try {
            const event = JSON.parse(raw);
            switch (event.type) {
              case 'token':
                handlers.onToken(event.content as string);
                break;
              case 'sources':
                handlers.onSources((event.sources ?? []) as SourceCitation[]);
                break;
              case 'done':
                handlers.onDone({
                  grounded: Boolean(event.grounded),
                  retrieved_chunks: Number(event.retrieved_chunks ?? 0),
                  model: String(event.model ?? ''),
                });
                break;
              case 'error':
                handlers.onError(String(event.message ?? 'The answer could not be generated.'));
                break;
            }
          } catch {
            // Ignore a partial frame; the next read will complete it.
          }
        }
      }
    } catch (error) {
      if ((error as Error)?.name === 'AbortError') return;
      handlers.onError('The connection to the server was interrupted.');
    }
  })();

  return () => controller.abort();
}
