"""Integration-lite tests for planner -> executor flow."""

from pathlib import Path
import shutil
import tempfile
import json

from backend.tools import (
    ToolRegistry,
    FAQStore,
    TicketStore,
    register_builtin_tools,
    ToolPlanner,
    ToolExecutor,
)


def _make_tmp_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="planner_executor_"))


def test_planner_executor_creates_ticket_for_support_intent():
    tmp_dir = _make_tmp_dir()
    try:
        kb_path = tmp_dir / "faq.json"
        kb_path.write_text(json.dumps([]), encoding="utf-8")
        ticket_db = tmp_dir / "tickets.db"

        registry = ToolRegistry()
        register_builtin_tools(
            registry=registry,
            faq_store=FAQStore(kb_path=str(kb_path)),
            ticket_store=TicketStore(db_path=str(ticket_db)),
        )
        planner = ToolPlanner(tools_enabled=True, max_calls_per_turn=3)
        executor = ToolExecutor(
            registry=registry,
            allowlist=["kb_search", "create_ticket", "get_ticket", "list_tickets"],
            timeout_seconds=3.0,
            rate_limit_per_minute=30,
            max_calls_per_turn=3,
        )

        planned = planner.plan(
            user_id="user_ticket",
            user_message="I want to open a support ticket for my internet not working",
        )
        assert planned.needs_tools is True
        assert planned.plan
        assert planned.plan[0].tool == "create_ticket"

        results = executor.execute_plan(user_id="user_ticket", plan=planned.plan)
        assert results
        assert results[0].success is True
        assert results[0].output["ticket_id"].startswith("T")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
