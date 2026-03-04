"""Node implementations for the PAHF-based LangGraph workflow."""

from __future__ import annotations

from typing import Any, Dict, List

from ..models.universal_chat import UniversalChat
from ..tools.schemas import ToolCall


def memory_retrieval_node(
    state: Dict[str, Any],
    pahf_memory_service,
) -> Dict[str, Any]:
    """Retrieve PAHF memories and run pre-action clarification check."""
    user_id = state["user_id"]
    user_message = state["user_message"]

    hits = pahf_memory_service.retrieve_for_chat(person_id=user_id, user_message=user_message)
    retrieved_memories = [
        {
            "id": hit.memory.id,
            "person_id": hit.memory.person_id,
            "text": hit.memory.text,
            "score": hit.score,
        }
        for hit in hits
    ]
    pahf_context_text = pahf_memory_service.render_retrieval_context(hits)
    clarification_question = pahf_memory_service.maybe_generate_pre_clarification(
        user_message=user_message,
        hits=hits,
    )

    return {
        **state,
        "retrieved_memories": retrieved_memories,
        "pahf_context_text": pahf_context_text,
        "clarification_question": clarification_question,
    }


def assistant_generation_node(
    state: Dict[str, Any],
    model_client: UniversalChat,
    prompt_builder,
    prompt_scene: str,
    tool_planner,
    tool_executor,
    tool_registry,
    tools_enabled: bool,
) -> Dict[str, Any]:
    """Generate assistant response after PAHF retrieval and optional tool use."""
    user_id = state["user_id"]
    user_message = state["user_message"]
    temperature = state.get("temperature")
    max_tokens = state.get("max_tokens")

    clarification_question = state.get("clarification_question")
    if clarification_question:
        return {
            **state,
            "intent": "pahf_pre_action_clarification",
            "tool_plan": [],
            "tool_results": [],
            "tool_errors": [],
            "response": clarification_question,
        }

    planner_output = None
    tool_results: List[Dict[str, Any]] = []
    tool_errors: List[str] = []
    tool_plan: List[Dict[str, Any]] = []
    intent = "general_chat"

    if tools_enabled and tool_planner is not None:
        planner_output = tool_planner.plan(user_id=user_id, user_message=user_message)
        intent = planner_output.intent
        tool_plan = [call.model_dump() for call in planner_output.plan]
        if planner_output.plan and tool_executor is not None:
            plan = [ToolCall.model_validate(item) for item in tool_plan]
            executed = tool_executor.execute_plan(user_id=user_id, plan=plan)
            tool_results = [r.model_dump() for r in executed]
            tool_errors = [r.error for r in executed if not r.success and r.error]

    model_input = user_message
    if prompt_builder is not None:
        available_tools = tool_registry.list_tools() if tool_registry is not None else {}
        planner_payload = {
            "intent": intent,
            "needs_tools": bool(tool_plan),
            "plan": tool_plan,
        }
        model_input = prompt_builder.build_model_input(
            scene=prompt_scene,
            user_message=user_message,
            pahf_context_text=state.get("pahf_context_text", ""),
            retrieved_memories=state.get("retrieved_memories", []),
            available_tools=available_tools,
            planner_output=planner_payload,
            tool_results=tool_results,
        )

    response = model_client.chat(
        user_id=user_id,
        message=model_input,
        temperature=temperature,
        max_tokens=max_tokens,
        use_history=False,
    )
    return {
        **state,
        "intent": intent,
        "tool_plan": tool_plan,
        "tool_results": tool_results,
        "tool_errors": tool_errors,
        "response": response,
    }


def memory_extraction_node(state: Dict[str, Any], pahf_memory_service) -> Dict[str, Any]:
    """Extract memory candidate summary from the current turn."""
    user_id = state["user_id"]
    user_message = state.get("user_message", "")
    assistant_message = state.get("response", "")

    candidate = pahf_memory_service.extract_memory_candidate(
        person_id=user_id,
        user_message=user_message,
        assistant_message=assistant_message,
        hits=state.get("retrieved_memories", []),
    )
    return {
        **state,
        "memory_candidate": candidate,
    }


def memory_update_node(state: Dict[str, Any], pahf_memory_service) -> Dict[str, Any]:
    """Apply PAHF similarity-based memory add/update."""
    user_id = state["user_id"]
    candidate = state.get("memory_candidate")
    if not candidate:
        return {
            **state,
            "memory_update": {"updated": False, "action": "no_candidate"},
        }

    update_result = pahf_memory_service.apply_memory_update(
        person_id=user_id,
        candidate_summary=candidate,
    )
    return {
        **state,
        "memory_update": update_result,
    }
