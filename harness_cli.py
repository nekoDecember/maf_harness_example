"""Resilient interactive Harness Agent for Microsoft Agent Framework 1.17.0.

This version uses the released Harness ``loop_should_continue`` API for the
normal autonomous loop. A small host-side supervisor remains deliberately: it
handles approvals/questions, checkpoints the session, retries transient API
failures, and starts another bounded loop batch instead of silently abandoning
an unfinished task when ``loop_max_iterations`` is reached.

Features:
* automatic tool-call loop inside each Agent Framework run;
* official Harness looping until the agent explicitly marks the task complete;
* host-side supervision across bounded loop batches;
* interactive approval for all file mutations;
* structured user questions through a declaration-only tool;
* atomic session checkpoints and ``--resume`` recovery;
* streaming output and bounded retries.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv

from agent_framework import (
    AgentModeProvider,
    AgentSession,
    Content,
    ContextProvider,
    FunctionTool,
    Message,
    SessionContext,
    TodoProvider,
    create_harness_agent,
    get_agent_mode,
    set_agent_mode,
    tool,
    todos_remaining,
    todos_remaining_message,
)
from agent_framework.openai import OpenAIChatClient, OpenAIChatCompletionClient


REQUIRED_VERSIONS = {
    "agent-framework-core": "1.17.0",
    # Microsoft releases provider packages independently. 1.14.2 is the
    # OpenAI provider release paired with core 1.17.x and requires core>=1.17.
    "agent-framework-openai": "1.14.2",
}
COMPLETION_SOURCE_ID = "task_completion"
RUNNER_STATE_KEY = "resilient_runner"


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


@dataclass(frozen=True)
class Settings:
    model: str
    api_key: str
    base_url: str | None
    client_kind: str
    context_window_tokens: int
    max_output_tokens: int
    loop_iterations_per_batch: int
    auto_batches_before_prompt: int
    api_retries: int
    workspace: Path
    checkpoint: Path

    @classmethod
    def from_environment(cls) -> "Settings":
        model = os.getenv("OPENAI_MODEL", "").strip()
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not model:
            raise ValueError("OPENAI_MODEL is required. Copy .env.example to .env and set it.")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required (a LiteLLM dummy key is acceptable if configured there).")

        client_kind = os.getenv("MAF_CLIENT", "chat_completions").strip().lower()
        if client_kind not in {"chat_completions", "responses"}:
            raise ValueError("MAF_CLIENT must be 'chat_completions' or 'responses'.")

        context_window = _positive_int_env("MAF_CONTEXT_WINDOW_TOKENS", 128_000)
        max_output = _positive_int_env("MAF_MAX_OUTPUT_TOKENS", 16_384)
        if max_output >= context_window:
            raise ValueError("MAF_MAX_OUTPUT_TOKENS must be smaller than MAF_CONTEXT_WINDOW_TOKENS.")

        return cls(
            model=model,
            api_key=api_key,
            base_url=os.getenv("OPENAI_BASE_URL", "").strip() or None,
            client_kind=client_kind,
            context_window_tokens=context_window,
            max_output_tokens=max_output,
            loop_iterations_per_batch=_positive_int_env("MAF_LOOP_ITERATIONS_PER_BATCH", 20),
            auto_batches_before_prompt=_positive_int_env("MAF_AUTO_BATCHES_BEFORE_PROMPT", 4),
            api_retries=_positive_int_env("MAF_API_RETRIES", 3),
            workspace=Path(os.getenv("MAF_WORKSPACE", "./workspace")).expanduser().resolve(),
            checkpoint=Path(os.getenv("MAF_CHECKPOINT", "./.state/session.json")).expanduser().resolve(),
        )


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    value = int(raw) if raw else default
    if value < 1:
        raise ValueError(f"{name} must be at least 1.")
    return value


def verify_framework_version() -> None:
    """Fail early instead of accidentally running against a different API."""
    versions = {
        name: importlib.metadata.version(name)
        for name in REQUIRED_VERSIONS
    }
    mismatches = {
        name: version
        for name, version in versions.items()
        if version != REQUIRED_VERSIONS[name]
    }
    if mismatches:
        details = ", ".join(f"{name}={version}" for name, version in mismatches.items())
        expected = ", ".join(f"{name}={version}" for name, version in REQUIRED_VERSIONS.items())
        raise RuntimeError(f"This sample requires {expected}; found {details}.")


class CompletionProvider(ContextProvider):
    """Expose a session-scoped, tool-driven completion signal to the agent."""

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
        del agent

        @tool(name="task_finish", approval_mode="never_require")
        def task_finish(summary: str) -> str:
            """Mark the current user task complete after every todo is complete and verified."""
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
    """Create text-file tools confined below one resolved workspace root."""
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
        """Create or replace a UTF-8 text file. Every call requires user approval."""
        target = resolve_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, target)
        return f"Wrote {len(content.encode('utf-8'))} bytes to {target.relative_to(root)}"

    @tool(name="workspace_delete", approval_mode="always_require")
    def workspace_delete(path: str) -> str:
        """Delete one workspace file. Every call requires user approval."""
        target = resolve_path(path)
        if not target.exists():
            return f"File not found: {path}"
        if not target.is_file() or target.is_symlink():
            return "Only regular non-symlink files can be deleted."
        target.unlink()
        return f"Deleted {target.relative_to(root)}"

    # A FunctionTool with func=None is a declaration-only tool. The Harness
    # returns it to the host as a user_input_request instead of executing it.
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


def make_completion_loop(
    todo_provider: TodoProvider,
    mode_provider: AgentModeProvider,
) -> tuple[Any, Any]:
    """Build the official Harness loop predicate and its continuation message.

    ``todos_remaining`` alone stops when the model forgot to create todos. This
    sample also requires the session-scoped ``task_finish`` signal, so a plain
    text answer cannot accidentally end an unfinished execute-mode task.
    """
    has_open_todos = todos_remaining(looping_modes=["execute"])

    async def should_continue(
        *,
        session: AgentSession | None = None,
        agent: Any = None,
        **kwargs: Any,
    ) -> bool:
        if session is None or agent is None:
            return False
        # AgentLoopMiddleware already yields pending approval requests, but a
        # declaration-only function such as ask_user is a different kind of
        # user-input request. Stop explicitly so the model cannot continue
        # before the caller has supplied the answer.
        last_result = kwargs.get("last_result")
        if getattr(last_result, "user_input_requests", None):
            return False
        mode = get_agent_mode(
            session,
            source_id=mode_provider.source_id,
            default_mode=mode_provider.default_mode,
            available_modes=mode_provider.available_modes,
        )
        if mode.strip().lower() != "execute":
            return False
        todos_open = await has_open_todos(session=session, agent=agent, **kwargs)
        return bool(todos_open or not task_is_complete(session))

    async def next_message(
        *,
        session: AgentSession | None = None,
        agent: Any = None,
        **kwargs: Any,
    ) -> str:
        todo_message = await todos_remaining_message(session=session, agent=agent, **kwargs)
        if todo_message:
            return (
                f"{todo_message}\n\n"
                "Do not end the task until you have verified the result and called task_finish."
            )
        return (
            "The current task is still active because task_finish has not been called. "
            "Continue working autonomously. Create or update todos for multi-step work. "
            "If required information is unavailable, call ask_user. When everything is "
            "verified, call task_finish exactly once."
        )

    return should_continue, next_message


def build_agent(settings: Settings) -> tuple[Any, TodoProvider, AgentModeProvider]:
    """Build the released 1.17.0 Harness Agent plus inspectable providers."""
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
    completion_provider = CompletionProvider()
    should_continue, next_message = make_completion_loop(todo_provider, mode_provider)

    agent = create_harness_agent(
        client=client,
        name="ResilientHarnessAgent",
        description="A persistent working agent that continues until explicitly complete.",
        agent_instructions=AGENT_INSTRUCTIONS,
        tools=make_workspace_tools(settings.workspace),
        max_context_window_tokens=settings.context_window_tokens,
        max_output_tokens=settings.max_output_tokens,
        todo_provider=todo_provider,
        mode_provider=mode_provider,
        context_providers=[completion_provider],
        disable_web_search=True,
        loop_should_continue=should_continue,
        loop_next_message=next_message,
        loop_max_iterations=settings.loop_iterations_per_batch,
    )
    return agent, todo_provider, mode_provider


def save_session(session: AgentSession, checkpoint: Path) -> None:
    """Atomically persist the full Harness session (history, todos, and modes)."""
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_name(f".{checkpoint.name}.tmp")
    temporary.write_text(json.dumps(session.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, checkpoint)


def load_session(checkpoint: Path) -> AgentSession:
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    return AgentSession.from_dict(payload)


def task_is_complete(session: AgentSession) -> bool:
    state = session.state.get(COMPLETION_SOURCE_ID, {})
    return bool(isinstance(state, dict) and state.get("done"))


def begin_task(session: AgentSession, task: str) -> None:
    session.state[COMPLETION_SOURCE_ID] = {"done": False, "summary": ""}
    session.state[RUNNER_STATE_KEY] = {"active": True, "task": task}


def finish_task_state(session: AgentSession) -> None:
    runner_state = session.state.setdefault(RUNNER_STATE_KEY, {})
    runner_state["active"] = False


def active_task(session: AgentSession) -> str | None:
    state = session.state.get(RUNNER_STATE_KEY, {})
    if isinstance(state, dict) and state.get("active") and isinstance(state.get("task"), str):
        return state["task"]
    return None


async def stream_one_run(agent: Any, session: AgentSession, agent_input: str | Sequence[Message]) -> Any:
    stream = agent.run(agent_input, session=session, stream=True)
    async for update in stream:
        if update.text:
            print(update.text, end="", flush=True)
    response = await stream.get_final_response()
    if response.text:
        print()
    return response


def _parse_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


async def collect_user_responses(requests: Sequence[Content]) -> list[Message]:
    """Turn Framework user-input requests into terminal prompts and reply messages."""
    replies: list[Message] = []
    for request in requests:
        if request.type == "function_approval_request" and request.function_call is not None:
            call = request.function_call
            print(f"\n[承認が必要] {call.name}")
            print(json.dumps(_parse_arguments(call.arguments), ensure_ascii=False, indent=2))
            answer = await asyncio.to_thread(input, "実行を承認しますか？ [y/N]: ")
            approved = answer.strip().lower() in {"y", "yes"}
            replies.append(Message("user", [request.to_function_approval_response(approved)]))
            continue

        if request.type == "function_call" and request.name == "ask_user":
            arguments = _parse_arguments(request.arguments)
            question = str(arguments.get("question") or "追加情報を入力してください")
            options = arguments.get("options")
            print(f"\n[エージェントからの確認] {question}")
            if isinstance(options, list):
                for index, option in enumerate(options, start=1):
                    print(f"  {index}. {option}")
            answer = await asyncio.to_thread(input, "> ")
            replies.append(
                Message(
                    "tool",
                    [Content.from_function_result(call_id=request.call_id, result=answer)],
                )
            )
            continue

        print(f"\n[未対応の入力要求] type={request.type}")
        answer = await asyncio.to_thread(input, "応答: ")
        call_id = getattr(request, "call_id", None) or getattr(request, "id", None)
        replies.append(Message("tool", [Content.from_function_result(call_id=call_id, result=answer)]))
    return replies


async def remaining_todos(todo_provider: TodoProvider, session: AgentSession) -> list[Any]:
    items = await todo_provider.store.load_items(session, source_id=todo_provider.source_id)
    return [item for item in items if not item.is_complete]


async def run_task(
    *,
    agent: Any,
    todo_provider: TodoProvider,
    mode_provider: AgentModeProvider,
    session: AgentSession,
    settings: Settings,
    task: str,
    resume: bool = False,
) -> None:
    """Supervise official loop batches until task_finish is called."""
    if not resume:
        begin_task(session, task)
        next_input: str | Sequence[Message] = task
    else:
        next_input = (
            "The previous process stopped while this task was unfinished. Resume from the persisted todos and "
            "history. Inspect current state, continue the work, and call task_finish only when verified complete."
        )

    save_session(session, settings.checkpoint)
    auto_batches = 0

    while True:
        response = None
        for attempt in range(1, settings.api_retries + 1):
            try:
                print(f"\n--- Harness loop batch {auto_batches + 1} ---")
                response = await stream_one_run(agent, session, next_input)
                save_session(session, settings.checkpoint)
                break
            except (KeyboardInterrupt, asyncio.CancelledError):
                save_session(session, settings.checkpoint)
                raise
            except Exception as exc:  # provider/network exceptions vary by backend
                save_session(session, settings.checkpoint)
                print(f"\n[APIエラー {attempt}/{settings.api_retries}] {type(exc).__name__}: {exc}")
                if attempt < settings.api_retries:
                    await asyncio.sleep(min(2 ** (attempt - 1), 8))

        if response is None:
            action = (await asyncio.to_thread(input, "再試行(r) / 指示を追加(f) / 中断して保存(s): ")).strip().lower()
            if action == "s":
                print(f"未完了状態を保存しました。再開: python harness_cli.py --resume")
                return
            if action == "f":
                feedback = await asyncio.to_thread(input, "追加指示: ")
                next_input = feedback
            continue

        if response.user_input_requests:
            next_input = await collect_user_responses(response.user_input_requests)
            save_session(session, settings.checkpoint)
            continue

        if task_is_complete(session) and not await remaining_todos(todo_provider, session):
            break

        current_mode = get_agent_mode(
            session,
            source_id=mode_provider.source_id,
            default_mode=mode_provider.default_mode,
            available_modes=mode_provider.available_modes,
        )
        if current_mode.strip().lower() != "execute":
            print("\n[plan mode] 自律ループは停止中です。計画を確認してください。")
            action = (
                await asyncio.to_thread(
                    input,
                    "executeへ移行(e) / 追加指示(f) / 中断して保存(s) [e]: ",
                )
            ).strip().lower()
            if action == "s":
                print("未完了状態を保存しました。再開: python harness_cli.py --resume")
                return
            if action == "f":
                next_input = await asyncio.to_thread(input, "追加指示: ")
            else:
                set_agent_mode(
                    session,
                    "execute",
                    source_id=mode_provider.source_id,
                    available_modes=mode_provider.available_modes,
                )
                next_input = "The user approved the plan. Switch to execution and complete every todo now."
            save_session(session, settings.checkpoint)
            continue

        open_items = await remaining_todos(todo_provider, session)
        titles = ", ".join(item.title for item in open_items) if open_items else "todo未作成または全件完了"
        next_input = (
            "The bounded Harness loop batch ended, but task_finish has not been called. "
            "Resume the same task in a new loop batch and continue autonomously. "
            f"Current open todos: {titles}. If blocked on required user information, call ask_user."
        )
        auto_batches += 1

        if auto_batches >= settings.auto_batches_before_prompt:
            total_iterations = settings.loop_iterations_per_batch * settings.auto_batches_before_prompt
            print(f"\n[安全確認] 最大{total_iterations}回の自律iteration後もタスクが未完了です。")
            action = (await asyncio.to_thread(input, "続行(c) / 指示を追加(f) / 中断して保存(s) [c]: ")).strip().lower()
            if action == "s":
                print("未完了状態を保存しました。再開: python harness_cli.py --resume")
                return
            if action == "f":
                next_input = await asyncio.to_thread(input, "追加指示: ")
            auto_batches = 0

    finish_task_state(session)
    save_session(session, settings.checkpoint)
    completion = session.state.get(COMPLETION_SOURCE_ID, {})
    if isinstance(completion, dict) and completion.get("summary"):
        print(f"\n[完了] {completion['summary']}")


async def repl(settings: Settings, *, resume: bool) -> None:
    agent, todo_provider, mode_provider = build_agent(settings)
    if resume:
        if not settings.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {settings.checkpoint}")
        session = load_session(settings.checkpoint)
    else:
        session = agent.create_session()

    print("Microsoft Agent Framework core 1.17.0 / OpenAI provider 1.14.2 Harness CLI")
    print(f"workspace: {settings.workspace}")
    print("commands: /mode plan|execute, /todos, /new, /exit")

    resumed_task = active_task(session) if resume else None
    if resumed_task:
        print(f"\n[再開] {resumed_task}")
        await run_task(
            agent=agent,
            todo_provider=todo_provider,
            mode_provider=mode_provider,
            session=session,
            settings=settings,
            task=resumed_task,
            resume=True,
        )

    while True:
        raw = (await asyncio.to_thread(input, "\nuser> ")).strip()
        if not raw:
            continue
        if raw in {"/exit", "/quit"}:
            save_session(session, settings.checkpoint)
            return
        if raw == "/new":
            session = agent.create_session()
            save_session(session, settings.checkpoint)
            print("新しいセッションを開始しました。")
            continue
        if raw.startswith("/mode "):
            mode = raw.removeprefix("/mode ").strip()
            set_agent_mode(session, mode, source_id=mode_provider.source_id, available_modes=mode_provider.available_modes)
            save_session(session, settings.checkpoint)
            print(f"mode: {get_agent_mode(session, source_id=mode_provider.source_id)}")
            continue
        if raw == "/todos":
            items = await todo_provider.store.load_items(session, source_id=todo_provider.source_id)
            if not items:
                print("todo: none")
            for item in items:
                print(f"[{'x' if item.is_complete else ' '}] {item.id}: {item.title}")
            continue

        await run_task(
            agent=agent,
            todo_provider=todo_provider,
            mode_provider=mode_provider,
            session=session,
            settings=settings,
            task=raw,
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true", help="resume an unfinished checkpoint")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    load_dotenv()
    args = parse_args(argv)
    try:
        verify_framework_version()
        settings = Settings.from_environment()
        asyncio.run(repl(settings, resume=args.resume))
    except KeyboardInterrupt:
        print("\nStopped. The latest completed agent run remains checkpointed.")
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
