"""Harness configuration and the small set of local tools used by the sample."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from agent_framework import (
    AgentModeProvider,
    AgentSession,
    Content,
    ContextProvider,
    FunctionTool,
    SessionContext,
    TodoProvider,
    create_harness_agent,
    tool,
)
from agent_framework.openai import OpenAIChatClient, OpenAIChatCompletionClient

from harness_state import COMPLETION_SOURCE_ID, Settings


AGENT_INSTRUCTIONS = """\
You are a general-purpose working agent operating on the user's local workspace.

For every substantive task:
1. Inspect the request and create concrete todo items before doing multi-step work.
2. Work through every todo using the available tools. Do not stop merely because
   one model response is ending; the host will invoke you again while unfinished.
3. Mark each todo complete only after its result has been checked.
4. If a fact or choice is genuinely required from the user, call ask_user. Never
   merely print a question and wait.
5. File reads are autonomous. Every write or deletion requires host approval.
6. Call task_finish exactly once, and only after all todos are complete and the
   requested result has been verified. Include a useful completion summary.

When a tool fails, diagnose it and try a materially different approach. Never
claim a file was changed unless a tool result confirms it.
"""


class CompletionProvider(ContextProvider):
    """Expose a session-scoped task_finish tool to the agent."""

    def __init__(self) -> None:
        super().__init__(COMPLETION_SOURCE_ID)

    async def before_run(
        self,
        *,
        agent: Any,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        del agent, session

        @tool(name="task_finish", approval_mode="never_require")
        def task_finish(summary: str) -> str:
            """Mark the current task complete after all requested work is verified."""
            state["done"] = True
            state["summary"] = summary.strip()
            return "The host recorded the task as complete. Return the final result to the user."

        context.extend_instructions(
            self.source_id,
            [
                "The host continues invoking you until task_finish is called. "
                "Call task_finish only after all requested work and verification are complete."
            ],
        )
        context.extend_tools(self.source_id, [task_finish])


def make_workspace_tools(workspace: Path) -> list[Any]:
    """Return read tools plus explicitly approved text-file mutation tools."""
    workspace.mkdir(parents=True, exist_ok=True)
    root = workspace.resolve()

    def resolve_path(relative_path: str, *, allow_root: bool = False) -> Path:
        requested = Path(relative_path)
        if requested.is_absolute():
            raise ValueError("Absolute paths are not allowed.")
        resolved = (root / requested).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError("Path escapes the configured workspace.")
        if not allow_root and resolved == root:
            raise ValueError("A file path is required; the workspace root is not a file.")
        return resolved

    @tool(name="workspace_list", approval_mode="never_require")
    def workspace_list(directory: str = ".") -> str:
        """List files recursively under a workspace directory."""
        target = resolve_path(directory, allow_root=True)
        if not target.exists():
            return f"Directory not found: {directory}"
        if not target.is_dir():
            return f"Not a directory: {directory}"
        entries = sorted(
            str(path.relative_to(root)) + ("/" if path.is_dir() else "")
            for path in target.rglob("*")
            if not path.is_symlink()
        )
        return "\n".join(entries) if entries else "(workspace is empty)"

    @tool(name="workspace_read_text", approval_mode="never_require")
    def workspace_read_text(path: str) -> str:
        """Read a UTF-8 text file from the workspace."""
        target = resolve_path(path)
        if not target.exists():
            return f"File not found: {path}"
        if not target.is_file():
            return f"Not a file: {path}"
        return target.read_text(encoding="utf-8")

    @tool(name="workspace_write_text", approval_mode="always_require")
    def workspace_write_text(path: str, content: str) -> str:
        """Create or replace a UTF-8 text file; every call requires approval."""
        target = resolve_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, target)
        return f"Wrote {len(content.encode('utf-8'))} bytes to {target.relative_to(root)}"

    @tool(name="workspace_delete", approval_mode="always_require")
    def workspace_delete(path: str) -> str:
        """Delete one workspace file; every call requires approval."""
        target = resolve_path(path)
        if not target.exists():
            return f"File not found: {path}"
        if not target.is_file() or target.is_symlink():
            return "Only regular non-symlink files can be deleted."
        target.unlink()
        return f"Deleted {target.relative_to(root)}"

    # func=None makes this a declaration-only function. The host receives it
    # as a user-input request and supplies the answer as a function result.
    ask_user = FunctionTool(
        name="ask_user",
        description="Pause and ask the user for information. Provide short options when useful.",
        func=None,
        input_model={
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "One clear question for the user."},
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional short answer choices.",
                },
            },
            "required": ["question"],
            "additionalProperties": False,
        },
    )

    return [workspace_list, workspace_read_text, workspace_write_text, workspace_delete, ask_user]


def build_agent(settings: Settings) -> tuple[Any, TodoProvider, AgentModeProvider]:
    """Build a 1.17.0 Harness without the experimental agent loop."""
    client_type = OpenAIChatCompletionClient if settings.client_kind == "chat_completions" else OpenAIChatClient
    client = client_type(
        model=settings.model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        function_invocation_configuration={
            "enabled": True,
            "max_iterations": 80,
            "max_function_calls": 240,
            "max_consecutive_errors_per_request": 5,
            "include_detailed_errors": True,
        },
    )
    todo_provider = TodoProvider()
    mode_provider = AgentModeProvider(default_mode="execute")
    agent = create_harness_agent(
        client=client,
        name="ResilientHarnessAgent",
        description="A persistent working agent supervised by a small host loop.",
        agent_instructions=AGENT_INSTRUCTIONS,
        tools=make_workspace_tools(settings.workspace),
        max_context_window_tokens=settings.context_window_tokens,
        max_output_tokens=settings.max_output_tokens,
        todo_provider=todo_provider,
        mode_provider=mode_provider,
        disable_file_memory=True,
        context_providers=[CompletionProvider()],
        disable_web_search=True,
    )
    return agent, todo_provider, mode_provider
