# Gdrive Backend — AI-powered Google Drive Chat

FastAPI backend that lets a user manage their Google Drive entirely through natural language. Built with **LangChain + LangGraph** (agentic tool-use), **ChromaDB** (vector store for RAG), and a pluggable **multi-provider LLM layer** (OpenAI, Anthropic, OpenRouter).

## Flow

```
user → Google OAuth (Drive scopes) → JWT session
     │
     ├─ GET  /api/v1/rag/indexed     → user-visible list of docs in vectorDB
     ├─ POST /api/v1/drive/upload    → upload-only (NOT auto-indexed)
     │
     └─ POST /api/v1/chat { message, conversationId? }
            → ChatService persists user msg (per-conversation history)
            → LangGraph agent (ReAct loop, bounded iterations)
                 ├─ Drive ops →  listFiles, searchFiles, getFileMetadata,
                 │               createFolder, deleteFile, shareFile,
                 │               readFileContent
                 └─ Doc QnA →  qnaOverFiles
                       ├─ status="ok"          → answer with citations
                       └─ status="no_matches"  → HITL gate:
                              requestDriveSearchPermission(...)  ──┐
                              end turn, ask user for permission     │
                              ──── (next turn, user confirms) ──────┘
                              → searchFiles → indexFileForQna → qnaOverFiles
                              → answer with citations
            → ChatService persists assistant + tool msgs
            → returns { reply, messages (full trace), awaitingApproval? }
```

## Quick start

```bash
cp .env.example .env           # fill in GOOGLE_*, JWT_*, ENCRYPTION_KEY, OPENAI_API_KEY
docker compose up -d           # API on :8000, Postgres on :5432
# docs at http://localhost:8000/docs
```

### Local dev (no Docker)

```bash
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Keys to generate

```bash
# Fernet encryption key
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

## API surface

### Auth
| Method | Endpoint | Description |
|---|---|---|
| GET  | `/api/v1/auth/google/login`    | Get Google OAuth URL |
| GET  | `/api/v1/auth/google/callback` | OAuth callback → issues JWTs |
| POST | `/api/v1/auth/refresh`         | Refresh JWT tokens |
| POST | `/api/v1/auth/logout`          | Logout |

### Drive (direct REST)
| Method | Endpoint | Description |
|---|---|---|
| GET    | `/api/v1/drive/files`          | List files / list folder children |
| POST   | `/api/v1/drive/upload`         | Upload file (multipart) |
| GET    | `/api/v1/drive/download/{id}`  | Download file |
| GET    | `/api/v1/drive/download/{id}/stream` | Stream download |
| DELETE | `/api/v1/drive/file/{id}`      | Delete file |
| POST   | `/api/v1/drive/create-folder`  | Create folder |
| GET    | `/api/v1/drive/search?q=`      | Search files by name |
| POST   | `/api/v1/drive/share`          | Share file with email |

### RAG
| Method | Endpoint | Description |
|---|---|---|
| POST   | `/api/v1/rag/index`           | Index a Drive file into the vector store |
| GET    | `/api/v1/rag/indexed`         | List indexed files |
| DELETE | `/api/v1/rag/index/{fileId}`  | Remove a file from the index |

### Chat (LangGraph agent)
| Method | Endpoint | Description |
|---|---|---|
| POST   | `/api/v1/chat`                             | Send a message (new/existing conversation) |
| GET    | `/api/v1/chat/conversations`               | List user's conversations |
| GET    | `/api/v1/chat/conversations/{id}`          | Get conversation + all messages |
| PATCH  | `/api/v1/chat/conversations/{id}`          | Rename conversation |
| DELETE | `/api/v1/chat/conversations/{id}`          | Delete conversation |

### Chat request example

```json
POST /api/v1/chat
Authorization: Bearer <accessToken>
{
  "message": "summarise the contract PDF I uploaded yesterday",
  "provider": "openai",
  "model": "gpt-4o-mini"
}
```

Returns:

```json
{
  "conversationId": "…",
  "reply": "Here's a summary of Contract_v3.pdf …",
  "messages": [
    { "role": "assistant", "toolCalls": [{"name": "qnaOverFiles", "args": {"question": "summarise the contract"}}] },
    { "role": "tool", "toolName": "qnaOverFiles", "content": "{\"status\":\"ok\", \"chunks\":[…]}" },
    { "role": "assistant", "content": "Here's a summary of …" }
  ],
  "awaitingApproval": null
}
```

### HITL example — empty vector store

When the user asks a doc question but no relevant file is indexed, the turn ends with `awaitingApproval` set:

```json
{
  "conversationId": "…",
  "reply": "I don't have any indexed documents matching your question. Should I search your Drive for 'invoice'?",
  "messages": [
    { "role": "assistant", "toolCalls": [{"name": "qnaOverFiles", "args": {"question": "what's the total on my invoice?"}}] },
    { "role": "tool", "toolName": "qnaOverFiles", "content": "{\"status\":\"no_matches\", \"reason\":\"no_indexed_files\", \"indexedFileCount\":0}" },
    { "role": "assistant", "toolCalls": [{"name": "requestDriveSearchPermission", "args": {"reason": "No indexed documents matched the question", "suggestedQuery": "invoice"}}] },
    { "role": "tool", "toolName": "requestDriveSearchPermission", "content": "{\"status\":\"awaiting_user_approval\", …}" },
    { "role": "assistant", "content": "I don't have any indexed documents …", "metadata": {"awaitingApproval": {"scope": "drive_search", "suggestedQuery": "invoice"}} }
  ],
  "awaitingApproval": { "scope": "drive_search", "reason": "No indexed documents matched the question", "suggestedQuery": "invoice" }
}
```

The frontend can render an approve/deny button from `awaitingApproval`. On the user's next turn (e.g. "yes, search for invoice"), the agent proceeds with `searchFiles → indexFileForQna → qnaOverFiles` and replies with citations.

## Architecture

```
app/
├── api/           → FastAPI routers (auth, drive, rag, chat, health)
├── core/          → config, db, security, logging, deps, exceptions
├── middleware/    → request logging, error handling
├── models/        → SQLAlchemy ORM (User, Token, Conversation, Message, FileEmbedding, …)
├── schemas/       → Pydantic request/response
└── services/
    ├── google_drive_service.py
    ├── llm/              → multi-provider LLM factory
    ├── vector_store/     → Chroma + embeddings
    ├── rag/              → document loader, splitter, RAG service
    ├── tools/            → LangChain StructuredTool definitions (Drive + RAG)
    ├── agent/            → LangGraph ReAct agent + system prompt
    └── chat/             → ChatService + ConversationService
```

## Design principles

1. **Plug-and-play LLM** — add a new provider in `services/llm/llm_factory.py` (~10 lines).
2. **Per-user vector collections** — Chroma collection per userId; isolation is enforced in the retrieval layer.
3. **Agent = LangGraph ReAct loop** — `agent → tools → agent … → end`, with bounded iterations.
4. **Chat history** — persisted as `Message` rows with full tool-call/tool-message trace; replayed into the agent as `BaseMessage`s.
5. **Human-in-the-loop (HITL)** — when `qnaOverFiles` returns no matches, the agent calls the structured `requestDriveSearchPermission` tool to pause the turn. The response carries `awaitingApproval`, and `Message.metaData` flags the assistant ask so the frontend can render an approve/deny UI. The user's next message resumes the flow naturally via persisted history.
6. **Upload ≠ index** — files uploaded via `/api/v1/drive/upload` go straight to Drive only. Indexing happens on demand (REST `/api/v1/rag/index`, agent tool `indexFileForQna`, or after HITL approval).
7. **Safety** — system prompt forbids destructive ops without explicit confirmation; tool errors are surfaced as `ToolMessage` content rather than crashing the turn.
8. **Observability** — every tool call, agent step, LLM init, and HTTP request is logged via `structlog`.
