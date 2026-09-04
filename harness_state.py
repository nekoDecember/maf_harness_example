"""Small, file-based state helpers for the Harness CLI."""

from __future__ import annotations

import importlib.metadata
import json
import os
from dataclasses import dataclass
from pathlib import Path

from agent_framework import AgentSession


REQUIRED_VERSIONS = {
    "agent-framework-core": "1.17.0",
    # Provider packages are released independently. This provider requires
    # agent-framework-core>=1.17.0,<2.
    "agent-framework-openai": "1.14.2",
}
COMPLETION_SOURCE_ID = "task_completion"
RUNNER_STATE_KEY = "resilient_runner"


@dataclass(frozen=True)
class Settings:
    model: str
    api_key: str
    base_url: str | None
    client_kind: str
    context_window_tokens: int
    max_output_tokens: int
    supervisor_runs_before_prompt: int
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
            raise ValueError(
                "OPENAI_API_KEY is required (a LiteLLM dummy key is acceptable if configured there)."
            )

        client_kind = os.getenv("MAF_CLIENT", "chat_completions").strip().lower()
        if client_kind not in {"chat_completions", "responses"}:
            raise ValueError("MAF_CLIENT must be 'chat_completions' or 'responses'.")

        context_window = positive_int_env("MAF_CONTEXT_WINDOW_TOKENS", 128_000)
        max_output = positive_int_env("MAF_MAX_OUTPUT_TOKENS", 16_384)
        if max_output >= context_window:
            raise ValueError("MAF_MAX_OUTPUT_TOKENS must be smaller than MAF_CONTEXT_WINDOW_TOKENS.")

        return cls(
            model=model,
            api_key=api_key,
            base_url=os.getenv("OPENAI_BASE_URL", "").strip() or None,
            client_kind=client_kind,
            context_window_tokens=context_window,
            max_output_tokens=max_output,
            supervisor_runs_before_prompt=positive_int_env("MAF_SUPERVISOR_RUNS_BEFORE_PROMPT", 12),
            api_retries=positive_int_env("MAF_API_RETRIES", 3),
            workspace=Path(os.getenv("MAF_WORKSPACE", "./workspace")).expanduser().resolve(),
            checkpoint=Path(os.getenv("MAF_CHECKPOINT", "./.state/session.json")).expanduser().resolve(),
        )


def positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    value = int(raw) if raw else default
    if value < 1:
        raise ValueError(f"{name} must be at least 1.")
    return value


def verify_framework_version() -> None:
    """Fail early instead of accidentally running against another API."""
    versions = {name: importlib.metadata.version(name) for name in REQUIRED_VERSIONS}
    mismatches = {
        name: version for name, version in versions.items() if version != REQUIRED_VERSIONS[name]
    }
    if mismatches:
        found = ", ".join(f"{name}={version}" for name, version in mismatches.items())
        expected = ", ".join(f"{name}={version}" for name, version in REQUIRED_VERSIONS.items())
        raise RuntimeError(f"This sample requires {expected}; found {found}.")


def save_session(session: AgentSession, checkpoint: Path) -> None:
    """Persist history and provider state with a same-directory atomic replace."""
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_name(f".{checkpoint.name}.tmp")
    temporary.write_text(json.dumps(session.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, checkpoint)


def load_session(checkpoint: Path) -> AgentSession:
    return AgentSession.from_dict(json.loads(checkpoint.read_text(encoding="utf-8")))


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
