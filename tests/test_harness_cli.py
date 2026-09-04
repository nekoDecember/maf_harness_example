from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from agent_framework import AgentSession, Content, Message

from harness_cli import (
    COMPLETION_SOURCE_ID,
    REQUIRED_VERSIONS,
    Settings,
    active_task,
    begin_task,
    build_agent,
    load_session,
    save_session,
    task_is_complete,
    verify_framework_version,
)


def make_settings(tmp_path: Path, base_url: str) -> Settings:
    return Settings(
        model="mock-model",
        api_key="test-key",
        base_url=base_url,
        client_kind="chat_completions",
        context_window_tokens=8_192,
        max_output_tokens=1_024,
        loop_iterations_per_batch=4,
        auto_batches_before_prompt=2,
        api_retries=1,
        workspace=tmp_path / "workspace",
        checkpoint=tmp_path / ".state" / "session.json",
    )


def test_exact_version_is_installed() -> None:
    assert REQUIRED_VERSIONS == {
        "agent-framework-core": "1.17.0",
        "agent-framework-openai": "1.14.2",
    }
    verify_framework_version()


def test_session_checkpoint_round_trip(tmp_path: Path) -> None:
    session = AgentSession(session_id="checkpoint-test")
    begin_task(session, "continue this work")
    session.state[COMPLETION_SOURCE_ID] = {"done": False, "summary": ""}
    checkpoint = tmp_path / "nested" / "session.json"

    save_session(session, checkpoint)
    restored = load_session(checkpoint)

    assert restored.session_id == "checkpoint-test"
    assert active_task(restored) == "continue this work"
    assert not task_is_complete(restored)


class MockChatHandler(BaseHTTPRequestHandler):
    request_count = 0

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        json.loads(self.rfile.read(length))
        type(self).request_count += 1

        if type(self).request_count == 1:
            # A normal assistant answer ends the chat client's tool loop. The
            # 1.17 Harness loop must autonomously invoke the agent again.
            message = {"role": "assistant", "content": "I am still working."}
            finish_reason = "stop"
        elif type(self).request_count == 2:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_question",
                        "type": "function",
                        "function": {
                            "name": "ask_user",
                            "arguments": json.dumps({"question": "What should the file contain?"}),
                        },
                    }
                ],
            }
            finish_reason = "tool_calls"
        elif type(self).request_count == 3:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_write",
                        "type": "function",
                        "function": {
                            "name": "workspace_write_text",
                            "arguments": json.dumps({"path": "result.txt", "content": "approved content"}),
                        },
                    }
                ],
            }
            finish_reason = "tool_calls"
        elif type(self).request_count == 4:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_finish",
                        "type": "function",
                        "function": {
                            "name": "task_finish",
                            "arguments": json.dumps({"summary": "mock task verified"}),
                        },
                    }
                ],
            }
            finish_reason = "tool_calls"
        else:
            message = {"role": "assistant", "content": "Mock task complete."}
            finish_reason = "stop"

        body = json.dumps(
            {
                "id": f"chatcmpl-{type(self).request_count}",
                "object": "chat.completion",
                "created": 1,
                "model": "mock-model",
                "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        del format, args


@pytest.mark.asyncio
async def test_real_117_harness_loops_and_executes_completion_tool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The mock endpoint is local; inherited CI proxy variables must not intercept it.
    for name in ("ALL_PROXY", "HTTPS_PROXY", "HTTP_PROXY", "all_proxy", "https_proxy", "http_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    MockChatHandler.request_count = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockChatHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        settings = make_settings(tmp_path, f"http://{host}:{port}/v1")
        agent, _, _ = build_agent(settings)
        session = agent.create_session()
        begin_task(session, "finish the mock task")

        response = await agent.run("finish the mock task", session=session)
        # Two HTTP requests prove loop_should_continue re-invoked the agent
        # after the first plain-text response.
        assert MockChatHandler.request_count == 2
        assert len(response.user_input_requests) == 1
        question = response.user_input_requests[0]
        assert question.type == "function_call"
        assert question.name == "ask_user"

        response = await agent.run(
            [Message("tool", [Content.from_function_result(question.call_id, result="approved content")])],
            session=session,
        )
        assert len(response.user_input_requests) == 1
        approval = response.user_input_requests[0]
        assert approval.type == "function_approval_request"
        assert approval.function_call.name == "workspace_write_text"

        response = await agent.run(
            [Message("user", [approval.to_function_approval_response(True)])],
            session=session,
        )

        assert response.text == "Mock task complete."
        assert task_is_complete(session)
        assert session.state[COMPLETION_SOURCE_ID]["summary"] == "mock task verified"
        assert (settings.workspace / "result.txt").read_text(encoding="utf-8") == "approved content"
        assert MockChatHandler.request_count == 5
    finally:
        server.shutdown()
        thread.join(timeout=2)
