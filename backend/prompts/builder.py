"""Programmable prompt builder for memory + tools aware responses."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .prompt_factory import PromptFactory


class PromptBuilder:
    """Builds final model input from structured runtime context."""

    def __init__(self, prompt_factory: PromptFactory):
        self.prompt_factory = prompt_factory

    @staticmethod
    def _to_json_block(data: Any) -> str:
        return json.dumps(data, ensure_ascii=False, indent=2)

    def build_model_input(
        self,
        scene: str,
        user_message: str,
        pahf_context_text: str = "",
        retrieved_memories: Optional[List[Dict[str, Any]]] = None,
        available_tools: Optional[Dict[str, Dict[str, str]]] = None,
        planner_output: Optional[Dict[str, Any]] = None,
        tool_results: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        retrieved_memories = retrieved_memories or []
        available_tools = available_tools or {}
        planner_output = planner_output or {}
        tool_results = tool_results or []

        base_system = self.prompt_factory.get_system_prompt(scene)
        tool_system = self.prompt_factory.get_template("tool_system")
        tool_output_format = self.prompt_factory.get_template("tool_output_format")

        sections = [
            "### Base System Prompt",
            base_system,
            "",
            "### Tool Policy",
            tool_system,
            "",
        ]

        if pahf_context_text:
            sections.extend(["### PAHF Memory Context", pahf_context_text, ""])

        if retrieved_memories:
            sections.extend(
                [
                    "### Retrieved PAHF Memories",
                    self._to_json_block(retrieved_memories),
                    "",
                ]
            )

        if available_tools:
            sections.extend(
                [
                    "### Available Tools",
                    self._to_json_block(available_tools),
                    "",
                ]
            )

        if planner_output:
            sections.extend(
                [
                    "### Planner Output",
                    self._to_json_block(planner_output),
                    "",
                ]
            )

        if tool_results:
            sections.extend(
                [
                    "### Tool Execution Results",
                    self._to_json_block(tool_results),
                    "",
                ]
            )

        sections.extend(
            [
                "### Response Format Rules",
                tool_output_format,
                "",
                "### Current User Message",
                user_message,
            ]
        )

        return "\n".join(sections).strip()
