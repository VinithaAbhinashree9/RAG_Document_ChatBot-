# RAG Document Chatbot

A retrieval-augmented Q&A chatbot: upload documents, get a grounded, cited
answer to any question about them. No hallucinated pages, no invented
citations — the model can only cite chunks it was actually given.

- **Backend:** FastAPI, ChromaDB (vector store), sentence-transformers
  (embeddings), Groq (LLM)
- **Frontend:** React + TypeScript + Vite

## Features

- Upload PDFs, Office docs, spreadsheets, presentations, markup, and plain
  text — see `GET /api/health` for the full supported-extension list
- Structure-aware chunking (paragraph → line → sentence → word) so citations
  stay accurate and sentences are never cut mid-way
- Query rewriting for follow-up questions ("what about *its* revenue?")
  using recent chat history
- Deterministic refusal when nothing relevant is retrieved — no LLM call, so
  no chance of an invented answer
- Per-document and multi-document chat modes
- Streaming responses (SSE) with citations delivered alongside the answer
- In-memory or ChromaDB vector store, swappable via config

## Project structure

```
rag-doc-chatbot/
├── backend/
│   ├── app/
│   │   ├── api/routes/       # chat, documents, health endpoints
│   │   ├── services/         # embeddings, chunker, retriever, chatbot,
│   │   │                     #   vector_store, groq_service, extraction/...
│   │   ├── models/           # internal document/chunk models
│   │   ├── schemas/          # public API request/response contracts
│   │   ├── prompts/          # prompt templates + registry
│   │   ├── config.py         # all settings, sourced from .env
│   │   └── main.py           # FastAPI app, CORS, error handling
│   ├── requirements.txt
│   └── .env                  # not committed — see Setup below
└── frontend/
    ├── src/
    │   ├── components/       # Sidebar, Composer, ChatMessage, SearchPanel...
    │   ├── lib/               # typed API client
    │   └── App.tsx
    └── package.json
```

## Prerequisites

- Python 3.11+
- Node.js 18+
- A free [Groq API key](https://console.groq.com/keys)

## Setup

### 1. Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create `backend/.env`:

```env
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

> Uploads and indexing work without a key; chat and summarization need it.

Run the API:

```bash
uvicorn app.main:app --reload --port 8000
```

- Health check: http://localhost:8000/api/health
- Interactive docs: http://localhost:8000/docs

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Visit http://localhost:5173. Both terminals need to stay running (backend on
`8000`, frontend on `5173`) — Vite proxies `/api` to the backend in dev.

## Configuration

All tunables live in `backend/app/config.py` and are overridable via
`backend/.env`. Notable ones:

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | *(empty)* | Required for chat/summarization |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Chat completion model |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |
| `EMBEDDING_OFFLINE_FALLBACK` | `false` | Use a dependency-free hashing embedder instead (lower quality, no network needed) |
| `VECTOR_STORE` | `chroma` | `chroma` or `memory` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1200` / `180` | Chunking granularity |
| `RETRIEVAL_TOP_K` | `6` | Chunks retrieved per query |
| `MAX_UPLOAD_MB` | `25` | Upload size limit |

## Building for production

```bash
cd frontend
npm run build      # outputs to frontend/dist
```

Serve `frontend/dist` from any static host, or point the backend at it, and
run the backend with a production ASGI server (e.g. `uvicorn` behind
`gunicorn`, or `uvicorn --workers N`).

## Security notes

- Never commit `backend/.env` — it holds your live Groq API key.
- `GROQ_API_KEY` is only ever reported as configured/not-configured via
  `/api/health`; the key itself is never returned by any endpoint.

## License

Add your license of choice here (MIT, Apache-2.0, etc.).
