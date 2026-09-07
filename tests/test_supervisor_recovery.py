import asyncio
from types import SimpleNamespace

import pytest
from agent_framework import AgentModeProvider, AgentSession, Content, Message, TodoProvider

import harness_cli
from harness_state import (
    COMPLETION_SOURCE_ID, RUNNER_STATE_KEY, Settings, active_task, begin_task,
    load_next_input, load_session, pending_requests, save_next_input, save_session,
    set_pending_requests,
)


def settings(tmp_path):
    return Settings(
        model="mock", api_key="test", base_url=None, client_kind="chat_completions",
        context_window_tokens=8192, max_output_tokens=1024, supervisor_runs_before_prompt=12,
        api_retries=3, workspace=tmp_path / "workspace", checkpoint=tmp_path / "state.json",
    )


@pytest.mark.asyncio
async def test_resume_answers_persisted_question_before_calling_agent(tmp_path, monkeypatch):
    configuration = settings(tmp_path)
    session = AgentSession()
    begin_task(session, "original task")
    request = Content.from_function_call(
        call_id="original-question", name="ask_user", arguments='{"question":"Which file?"}',
    )
    set_pending_requests(session, [request])
    save_session(session, configuration.checkpoint)
    restored = load_session(configuration.checkpoint)

    async def answer(requests):
        assert requests[0].call_id == "original-question"
        return [Message("tool", [Content.from_function_result("original-question", result="answer")])]

    class Agent:
        async def run(self, agent_input, *, session, **kwargs):
            assert agent_input[0].contents[0].call_id == "original-question"
            session.state[COMPLETION_SOURCE_ID] = {"done": True, "summary": "verified"}
            return SimpleNamespace(text="done", user_input_requests=[])

    monkeypatch.setattr(harness_cli, "collect_user_responses", answer)
    await harness_cli.run_task(
        agent=Agent(), todo_provider=TodoProvider(), mode_provider=AgentModeProvider(),
        session=restored, settings=configuration, task="original task", resume=True,
    )
    assert not pending_requests(restored)
    assert active_task(restored) is None


@pytest.mark.asyncio
async def test_premature_completion_requires_a_fresh_finish_call(tmp_path, monkeypatch):
    session = AgentSession()
    checks = 0

    async def todos(*args):
        nonlocal checks
        checks += 1
        return [SimpleNamespace(title="unfinished")] if checks == 1 else []

    class Agent:
        calls = 0

        async def run(self, agent_input, *, session, **kwargs):
            self.calls += 1
            if self.calls in {1, 3}:
                session.state[COMPLETION_SOURCE_ID] = {"done": True, "summary": "verified"}
            return SimpleNamespace(text="step", user_input_requests=[])

    agent = Agent()
    monkeypatch.setattr(harness_cli, "remaining_todos", todos)
    await harness_cli.run_task(
        agent=agent, todo_provider=TodoProvider(), mode_provider=AgentModeProvider(default_mode="execute"),
        session=session, settings=settings(tmp_path), task="work",
    )
    assert agent.calls == 3


@pytest.mark.asyncio
async def test_failed_turn_does_not_automatically_replay_approved_work(tmp_path, monkeypatch):
    class Agent:
        calls = 0

        async def run(self, *args, **kwargs):
            self.calls += 1
            raise RuntimeError("write may already have completed")

    agent = Agent()
    session = AgentSession()
    configuration = settings(tmp_path)
    monkeypatch.setattr("builtins.input", lambda _: "s")
    await harness_cli.run_task(
        agent=agent, todo_provider=TodoProvider(), mode_provider=AgentModeProvider(),
        session=session, settings=configuration, task="write file",
    )
    restored = load_session(configuration.checkpoint)
    assert agent.calls == 1
    assert "Do not replay" in load_next_input(restored)
    assert active_task(restored) == "write file"


def test_in_flight_checkpoint_never_replays_saved_approval(tmp_path):
    session = AgentSession()
    begin_task(session, "write file")
    save_next_input(session, [Message("user", [Content.from_text("old approval")])])
    session.state[RUNNER_STATE_KEY]["in_flight"] = True
    checkpoint = tmp_path / "state.json"
    save_session(session, checkpoint)
    restored = load_session(checkpoint)
    assert isinstance(load_next_input(restored), str)
    assert "Do not replay" in load_next_input(restored)


@pytest.mark.asyncio
async def test_question_remains_pending_when_user_input_is_interrupted(tmp_path, monkeypatch):
    class Agent:
        async def run(self, *args, **kwargs):
            return SimpleNamespace(text="", user_input_requests=[Content.from_function_call(
                call_id="pending", name="ask_user", arguments='{"question":"Which?"}',
            )])

    async def interrupted(requests):
        raise asyncio.CancelledError

    configuration = settings(tmp_path)
    monkeypatch.setattr(harness_cli, "collect_user_responses", interrupted)
    with pytest.raises(asyncio.CancelledError):
        await harness_cli.run_task(
            agent=Agent(), todo_provider=TodoProvider(), mode_provider=AgentModeProvider(),
            session=AgentSession(), settings=configuration, task="work",
        )
    assert pending_requests(load_session(configuration.checkpoint))[0].call_id == "pending"


@pytest.mark.asyncio
async def test_completed_checkpoint_does_not_invoke_model_again(tmp_path):
    class Agent:
        async def run(self, *args, **kwargs):
            raise AssertionError("Completed task must not run again")

    session = AgentSession()
    begin_task(session, "done task")
    session.state[COMPLETION_SOURCE_ID] = {"done": True, "summary": "already verified"}
    await harness_cli.run_task(
        agent=Agent(), todo_provider=TodoProvider(), mode_provider=AgentModeProvider(),
        session=session, settings=settings(tmp_path), task="done task", resume=True,
    )
    assert active_task(session) is None


def test_host_plan_mode_rejects_a_write_even_with_a_function_context(tmp_path):
    from harness_agent import make_workspace_tools

    write = next(tool for tool in make_workspace_tools(tmp_path) if tool.name == "workspace_write_text")
    context = SimpleNamespace(kwargs={"host_mode": "plan"})
    with pytest.raises(ValueError, match="plan mode"):
        write.func(context, path="should-not-exist.txt", content="blocked")
    assert not (tmp_path / "should-not-exist.txt").exists()
