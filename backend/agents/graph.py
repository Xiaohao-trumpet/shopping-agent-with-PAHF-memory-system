"""LangGraph definition for PAHF-only memory orchestration."""

from __future__ import annotations

from typing import Any, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from .node_calls import (
    assistant_generation_node,
    memory_extraction_node,
    memory_retrieval_node,
    memory_update_node,
)


class ChatState(TypedDict):
    """State definition for PAHF-first chat workflow."""

    user_id: str
    user_message: str
    response: Optional[str]
    temperature: Optional[float]
    max_tokens: Optional[int]
    session: Any
    retrieved_memories: List[dict]
    pahf_context_text: str
    clarification_question: Optional[str]
    intent: Optional[str]
    tool_plan: List[dict]
    tool_results: List[dict]
    tool_errors: List[str]
    memory_candidate: Optional[str]
    memory_update: Optional[dict]


def create_chat_graph(
    model_client,
    pahf_memory_service,
    tool_planner=None,
    tool_executor=None,
    tool_registry=None,
    prompt_builder=None,
    prompt_scene: str = "default",
    tools_enabled: bool = True,
):
    """Create and compile the PAHF-only chat graph."""
    workflow = StateGraph(ChatState)

    workflow.add_node(
        "memory_retrieval_node",
        lambda state: memory_retrieval_node(
            state=state,
            pahf_memory_service=pahf_memory_service,
        ),
    )
    workflow.add_node(
        "assistant_generation_node",
        lambda state: assistant_generation_node(
            state=state,
            model_client=model_client,
            prompt_builder=prompt_builder,
            prompt_scene=prompt_scene,
            tool_planner=tool_planner,
            tool_executor=tool_executor,
            tool_registry=tool_registry,
            tools_enabled=tools_enabled,
        ),
    )
    workflow.add_node(
        "memory_extraction_node",
        lambda state: memory_extraction_node(
            state=state,
            pahf_memory_service=pahf_memory_service,
        ),
    )
    workflow.add_node(
        "memory_update_node",
        lambda state: memory_update_node(
            state=state,
            pahf_memory_service=pahf_memory_service,
        ),
    )

    workflow.set_entry_point("memory_retrieval_node")
    workflow.add_edge("memory_retrieval_node", "assistant_generation_node")
    workflow.add_edge("assistant_generation_node", "memory_extraction_node")
    workflow.add_edge("memory_extraction_node", "memory_update_node")
    workflow.add_edge("memory_update_node", END)
    return workflow.compile()
