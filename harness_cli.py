"""Interactive CLI and host-side supervisor for the 1.17.0 Harness sample."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any, Sequence

from dotenv import load_dotenv

from agent_framework import AgentModeProvider, AgentSession, Content, Message, TodoProvider, get_agent_mode, set_agent_mode

from harness_agent import build_agent
from harness_state import (
    COMPLETION_SOURCE_ID,
    REQUIRED_VERSIONS,
    Settings,
    active_task,
    begin_task,
    finish_task_state,
    load_session,
    save_session,
    task_is_complete,
    verify_framework_version,
)


async def run_one_turn(agent: Any, session: AgentSession, agent_input: str | Sequence[Message]) -> Any:
    """Run one complete Framework turn and preserve any pending input requests.

    In core 1.17.0, the streaming wrapper does not reliably carry
    ``user_input_requests`` to its final response. A completed turn is a small
    and predictable boundary for this sample, so use the normal response path
    and let the host supervisor resume from there.
    """
    response = await agent.run(agent_input, session=session)
    if response.text:
        print(response.text)
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
    """Translate Framework approval and input requests into terminal replies."""
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
            replies.append(Message("tool", [Content.from_function_result(request.call_id, result=answer)]))
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
    """Keep calling the stable single-turn Harness until task_finish is seen."""
    if not resume:
        begin_task(session, task)
        next_input: str | Sequence[Message] = task
    else:
        next_input = (
            "The previous process stopped while this task was unfinished. Resume from the persisted todos and "
            "history. Inspect current state, continue the work, and call task_finish only when verified complete."
        )

    save_session(session, settings.checkpoint)
    runs_since_prompt = 0

    while not task_is_complete(session):
        response = None
        for attempt in range(1, settings.api_retries + 1):
            try:
                print(f"\n--- host-supervised run {runs_since_prompt + 1} ---")
                response = await run_one_turn(agent, session, next_input)
                save_session(session, settings.checkpoint)
                break
            except (KeyboardInterrupt, asyncio.CancelledError):
                save_session(session, settings.checkpoint)
                raise
            except Exception as exc:  # Provider/network exception types vary by backend.
                save_session(session, settings.checkpoint)
                print(f"\n[APIエラー {attempt}/{settings.api_retries}] {type(exc).__name__}: {exc}")
                if attempt < settings.api_retries:
                    await asyncio.sleep(min(2 ** (attempt - 1), 8))

        if response is None:
            action = (
                await asyncio.to_thread(input, "再試行(r) / 指示を追加(f) / 中断して保存(s): ")
            ).strip().lower()
            if action == "s":
                print("未完了状態を保存しました。再開: python harness_cli.py --resume")
                return
            if action == "f":
                next_input = await asyncio.to_thread(input, "追加指示: ")
            continue

        # The Framework returns here for declaration-only questions and tool approvals.
        if response.user_input_requests:
            next_input = await collect_user_responses(response.user_input_requests)
            save_session(session, settings.checkpoint)
            continue

        if task_is_complete(session):
            break

        current_mode = get_agent_mode(
            session,
            source_id=mode_provider.source_id,
            default_mode=mode_provider.default_mode,
            available_modes=mode_provider.available_modes,
        )
        if current_mode.strip().lower() != "execute":
            print("\n[plan mode] 自律実行は停止中です。計画を確認してください。")
            action = (
                await asyncio.to_thread(input, "executeへ移行(e) / 追加指示(f) / 保存して中断(s) [e]: ")
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
            "The previous Framework run ended, but task_finish has not been called. "
            "Continue the same task now; do not merely describe what remains. "
            f"Current open todos: {titles}. If blocked on required user information, call ask_user."
        )
        runs_since_prompt += 1

        if runs_since_prompt >= settings.supervisor_runs_before_prompt:
            print(f"\n[安全確認] {runs_since_prompt}回のhost-supervised run後も未完了です。")
            action = (
                await asyncio.to_thread(input, "続行(c) / 指示を追加(f) / 中断して保存(s) [c]: ")
            ).strip().lower()
            if action == "s":
                print("未完了状態を保存しました。再開: python harness_cli.py --resume")
                return
            if action == "f":
                next_input = await asyncio.to_thread(input, "追加指示: ")
            runs_since_prompt = 0

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

    print("Microsoft Agent Framework core 1.17.0 / OpenAI provider 1.14.2")
    print("stable host-supervised Harness CLI")
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
