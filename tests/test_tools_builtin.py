"""Tests for built-in tools."""

from pathlib import Path
import json
import shutil
import tempfile

from backend.tools import ToolRegistry, FAQStore, TicketStore, register_builtin_tools


def _make_tmp_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="tools_builtin_"))


def test_builtin_ticket_tools_create_get_list():
    tmp_dir = _make_tmp_dir()
    try:
        kb_path = tmp_dir / "faq.json"
        kb_path.write_text("[]", encoding="utf-8")
        ticket_db = tmp_dir / "tickets.db"

        registry = ToolRegistry()
        register_builtin_tools(
            registry=registry,
            faq_store=FAQStore(kb_path=str(kb_path)),
            ticket_store=TicketStore(db_path=str(ticket_db)),
        )

        created = registry.execute(
            "create_ticket",
            {
                "user_id": "u100",
                "subject": "Internet down",
                "description": "My home internet is not working.",
                "priority": "high",
                "tags": ["network"],
            },
        )
        assert created["ticket_id"].startswith("T")

        fetched = registry.execute("get_ticket", {"ticket_id": created["ticket_id"]})
        assert fetched["found"] is True
        assert fetched["ticket"]["user_id"] == "u100"

        listed = registry.execute("list_tickets", {"user_id": "u100", "limit": 5})
        assert len(listed["tickets"]) == 1
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_builtin_kb_search_returns_refund_hit():
    tmp_dir = _make_tmp_dir()
    try:
        kb_path = tmp_dir / "faq.json"
        kb_path.write_text(
            json.dumps(
                [
                    {
                        "id": "refund_policy",
                        "question": "What is your refund policy?",
                        "answer": "Refunds available within 7 days.",
                        "tags": ["refund"],
                    }
                ]
            ),
            encoding="utf-8",
        )
        ticket_db = tmp_dir / "tickets.db"

        registry = ToolRegistry()
        register_builtin_tools(
            registry=registry,
            faq_store=FAQStore(kb_path=str(kb_path)),
            ticket_store=TicketStore(db_path=str(ticket_db)),
        )

        result = registry.execute("kb_search", {"query": "refund policy", "top_k": 3})
        assert result["hits"]
        assert result["hits"][0]["id"] == "refund_policy"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
