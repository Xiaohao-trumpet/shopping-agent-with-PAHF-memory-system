# Frontend Architecture

## Stack

- React 18
- Vite 5
- TypeScript

## Goals

- Keep frontend lightweight and local-dev friendly.
- Mirror core chat UX patterns without heavy platform dependencies.
- Integrate directly with backend HTTP APIs.

## Modules

- `src/App.tsx`
  - Main UI state and layout:
    - conversation sidebar
    - chat stream and composer
    - PAHF memory panel
    - trace drawer
- `src/api.ts`
  - HTTP client functions for backend endpoints
- `src/types.ts`
  - Frontend model types
- `src/styles.css`
  - Single-file layout and visual styling

## Data Flow

1. App bootstrap:
   - `GET /health`
   - `GET /api/v1/models` (fallback `/v1/models`)
   - `GET /api/v1/prompt-scenes`
2. User sends message:
   - POST `/api/v1/chat/completions`
3. Assistant response:
   - Render response content into current conversation
   - Save trace metadata for tools/trace drawer
4. Memory actions:
   - list/add/update/search/find-similar via `/api/v1/memory*`

## Persistence

Browser localStorage stores:

- conversations
- selected model
- selected scene
- system prompt text
- current user id

No server-side auth/session dependency is required for frontend state.

## Extension Points

- Add streaming UI when backend streaming implementation is ready.
- Add richer trace renderer for tool calls with typed payloads.
- Add optimistic updates and retry controls for network failures.
- Split App into components as UI grows (ConversationList, MessageList, MemoryPanel).
