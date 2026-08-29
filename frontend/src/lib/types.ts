/** API contract types — mirrors backend/app/schemas/api.py */

export type DocumentStatus =
  | 'uploaded'
  | 'processing'
  | 'indexed'
  | 'summarizing'
  | 'ready'
  | 'failed';

export interface DocumentInfo {
  document_id: string;
  filename: string;
  file_type: string;
  size_bytes: number;
  status: DocumentStatus;
  page_count: number | null;
  char_count: number;
  chunk_count: number;
  summary: string | null;
  error: string | null;
  extractor: string | null;
  warnings: string[];
  created_at: string;
  updated_at: string;
}

export interface SourceCitation {
  document_id: string;
  filename: string;
  chunk_id: string;
  chunk_index: number;
  page: number | null;
  section: string | null;
  score: number;
  excerpt: string;
  label: string;
}

export interface UploadResponse {
  document: DocumentInfo;
  summary: string | null;
  summary_error: string | null;
  message: string;
}

export interface DocumentListResponse {
  documents: DocumentInfo[];
  total: number;
}

export interface SummarizeResponse {
  document_id: string;
  summary: string;
  cached: boolean;
  strategy: 'direct' | 'map_reduce';
  map_calls: number;
}

export interface ChatResponse {
  answer: string;
  sources: SourceCitation[];
  session_id: string;
  document_ids: string[];
  grounded: boolean;
  retrieved_chunks: number;
  model: string;
}

export interface ChunkInfo {
  chunk_id: string;
  document_id: string;
  chunk_index: number;
  page: number | null;
  section: string | null;
  char_count: number;
  text: string;
}

export interface SourcesResponse {
  document_id: string;
  filename: string;
  total: number;
  chunks: ChunkInfo[];
}

export interface HealthResponse {
  status: string;
  app_env: string;
  llm_configured: boolean;
  llm_model: string;
  embedding_model: string;
  vector_store: string;
  documents_indexed: number;
  supported_extensions: string[];
}

/** UI-side chat message model. */
export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: SourceCitation[];
  grounded?: boolean;
  streaming?: boolean;
  error?: boolean;
  /** The question that produced this answer, enabling regeneration. */
  question?: string;
  createdAt: string;
}

export interface ApiErrorShape {
  code: string;
  message: string;
  details?: Record<string, unknown>;
  request_id?: string;
}
