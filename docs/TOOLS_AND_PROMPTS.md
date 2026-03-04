# Tools and Prompts (Phase 3)

## Architecture Overview

Phase 3 adds a local-first tool layer and a programmable prompt builder:

`memory_retrieval_node -> assistant_generation_node -> memory_extraction_node -> memory_update_node`

Core modules:

- `backend/tools/registry.py`: typed tool registration and schema validation.
- `backend/tools/planner.py`: intent routing and ReAct-style tool plan generation.
- `backend/tools/executor.py`: safe execution with allowlist, timeout, and rate limiting.
- `backend/tools/builtin.py`: built-in tools (`kb_search`, `create_ticket`, `get_ticket`, `list_tickets`).
- `backend/prompts/builder.py`: dynamic prompt assembly using runtime context.

## Built-in Tools

### 1) `kb_search`

Input schema (`KBSearchInput`):

- `query: str`
- `top_k: int` (1-10, default 3)

Output schema (`KBSearchOutput`):

- `query: str`
- `hits: list[dict]`

### 2) `create_ticket`

Input schema (`CreateTicketInput`):

- `user_id: str`
- `subject: str`
- `description: str`
- `priority: "low" | "medium" | "high" | "urgent"`
- `tags: list[str]`

Output schema (`CreateTicketOutput`):

- `ticket_id: str`
- `status: str`
- `user_id: str`
- `subject: str`
- `priority: str`
- `created_at: float`

### 3) `get_ticket`

Input schema (`GetTicketInput`):

- `ticket_id: str`

Output schema (`GetTicketOutput`):

- `found: bool`
- `ticket: dict | null`

### 4) `list_tickets`

Input schema (`ListTicketsInput`):

- `user_id: str`
- `limit: int` (1-50, default 10)

Output schema (`ListTicketsOutput`):

- `user_id: str`
- `tickets: list[dict]`

## How to Add a New Tool

1. Define input/output Pydantic models in `backend/tools/schemas.py`.
2. Implement tool logic in `backend/tools/builtin.py` (or a new module).
3. Register the tool in `register_builtin_tools(...)` using `ToolSpec`.
4. Add tool name to `TOOLS_ALLOWLIST` in `.env`.
5. Extend planner routing in `backend/tools/planner.py` if automatic invocation is needed.
6. Add at least one focused test in `tests/` for input validation and execution.

## Planner and ReAct-style Flow

Planner output structure:

```json
{
  "intent": "ticket_create",
  "needs_tools": true,
  "plan": [
    {
      "tool": "create_ticket",
      "arguments": {
        "user_id": "frontend_user",
        "subject": "I want to open a support ticket for my internet not working",
        "description": "I want to open a support ticket for my internet not working",
        "priority": "high",
        "tags": ["auto", "chat"]
      },
      "reason": "User requests support and likely needs a new ticket."
    }
  ]
}
```

Execution trace (response `trace` field):

- `intent`
- `tool_plan`
- `tool_results`
- `tool_errors`
- `retrieved_memories`
- `pahf_context_text`
- `clarification_question`
- `memory_candidate`
- `memory_update`

## Dynamic Prompt Assembly

Prompt builder (`backend/prompts/builder.py`) composes:

- scene-specific base system prompt
- PAHF memory context
- retrieved PAHF memories
- available tool metadata
- planner output
- tool execution results
- current user message

Templates live in:

- `backend/prompts/tool_system.txt`
- `backend/prompts/tool_output_format.txt`
- existing scene prompt files (`system.txt`, `it_helpdesk.txt`, ...)

## Configuration

Set in `.env`:

- `TOOLS_ENABLED=true|false`
- `TOOLS_ALLOWLIST=kb_search,create_ticket,get_ticket,list_tickets`
- `TOOL_MAX_CALLS_PER_TURN=3`
- `TOOL_TIMEOUT_SECONDS=3.0`
- `TOOL_RATE_LIMIT_PER_MINUTE=30`
- `KB_FILE_PATH=./data/kb/faq.json`
- `TICKET_DB_PATH=./data/tickets/tickets.db`

## Startup

```bash
python run_all.py
```

or backend only:

```bash
python run_backend.py
```

## Troubleshooting

- `Tool not allowed`: check `TOOLS_ALLOWLIST`.
- `Tool call timed out`: raise `TOOL_TIMEOUT_SECONDS` or reduce heavy tool logic.
- `No KB hits`: verify `KB_FILE_PATH` exists and has JSON array records.
- `Ticket tool returns not found`: verify `ticket_id` format and `TICKET_DB_PATH`.
- OpenWebUI stream mode: backend currently emits a single chunk then `[DONE]` (non-token streaming behavior).
