"""tau-bench adapter: wrap Sierra's tau-bench as a wdp Benchmark.

tau-bench is the real target this controller is built for: each task is a
multi-turn customer-service conversation against an LLM-simulated user, where
success is graded by the environment's database state (did the right rows
change?), not by a text answer. That makes it a faithful stress test for
*compute allocation* -- when to open a fresh attempt (WIDER), keep pushing the
current conversation (DEEPER), or stop -- on tasks with real headroom, unlike
the local arithmetic suite.

Three pieces, mirroring the rest of wdp:

  - `TauBenchBenchmark`  supplies the tasks and the terminal verifier.
  - `TauReActExecutor`   is the agent: it drives a tau-bench Env via reset/step,
                         talking to the simulated user and the domain tools, and
                         returns a Trajectory carrying the env's ground-truth
                         reward. WIDER = a fresh env.reset (independent attempt);
                         DEEPER = continue stepping the same env.
  - `TauTerminalVerifier` is a thin stand-in: the reward already rides on the
                         Trajectory (env-graded), so the runner trusts that and
                         only falls back here, which reports 0.

The agent's own model calls go through the wdp OpenRouter client so they land in
the cost ledger; the user simulator runs on tau-bench's own litellm path and its
cost is tracked separately by the env (not the agent's spend).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

from wdp.executor.react import Step, Task, Trajectory, _parse_turn
from wdp.verifier.scorer import Score

# tau-bench is an optional heavy dependency; import lazily-friendly names here so
# importing this module is cheap and offline tests can inject a fake env.
from tau_bench.types import Action, RESPOND_ACTION_NAME


_SYSTEM_TEMPLATE = (
    "{wiki}\n\n"
    "You are a customer-service agent. Work step by step. On every turn respond "
    "with ONLY a single JSON object and nothing else:\n"
    '{{"thought": "...", "action": "<tool name or respond>", "action_input": {{...}}}}\n'
    "- Use action \"respond\" with action_input {{\"content\": \"...\"}} to send a "
    "message to the customer (ask a question, confirm, or give the final answer).\n"
    "- Otherwise use exactly one of the tools below, with its arguments in "
    "action_input.\n"
    "- Follow the policy above strictly; never invent ids or facts the customer "
    "did not give you.\n\n"
    "TOOLS:\n{catalog}\n"
    "- respond: send a message to the customer | content (string, required)"
)


def _catalog(tools_info: list[dict]) -> str:
    lines: list[str] = []
    for ti in tools_info:
        fn = ti.get("function", ti)
        params = fn.get("parameters", {}).get("properties", {})
        required = set(fn.get("parameters", {}).get("required", []))
        parts = []
        for name, spec in params.items():
            tag = "required" if name in required else "optional"
            parts.append(f"{name} ({spec.get('type', 'any')}, {tag})")
        sig = "; ".join(parts) if parts else "no arguments"
        lines.append(f"- {fn['name']}: {fn.get('description', '').strip()} | {sig}")
    return "\n".join(lines)


@dataclass
class TauReActExecutor:
    """A tau-bench agent that satisfies the wdp Executor surface (run / continue_from).

    Each `run` resets a *fresh* env for the task (an independent attempt, what the
    Allocator's WIDER action wants); `continue_from` resumes the same env and
    conversation (DEEPER). The ground-truth reward is read from the env at the
    moment the conversation ends and stashed on the Trajectory.
    """
    client: object
    model: str
    env_name: str = "retail"
    split: str = "test"
    user_model: str = "openai/gpt-4o-mini"
    user_provider: str = "openrouter"
    user_strategy: str = "llm"
    max_steps: int = 30
    temperature: float = 0.0
    # Injectable so offline tests can hand in a fake env without touching network.
    env_factory: Callable[[int], object] | None = None

    def _make_env(self, task_index: int):
        if self.env_factory is not None:
            return self.env_factory(task_index)
        from tau_bench.envs import get_env

        return get_env(
            self.env_name,
            user_strategy=self.user_strategy,
            user_model=self.user_model,
            user_provider=self.user_provider,
            task_split=self.split,
            task_index=task_index,
        )

    def run(self, task: Task, *, ledger=None, parallel_group: str | None = None) -> Trajectory:
        env = self._make_env(int(task.metadata["task_index"]))
        reset = env.reset(task_index=int(task.metadata["task_index"]))
        traj = Trajectory(task_id=task.id, parallel_group=parallel_group, env=env)
        system = _SYSTEM_TEMPLATE.format(wiki=env.wiki, catalog=_catalog(env.tools_info))
        traj._messages = [  # type: ignore[attr-defined]
            {"role": "system", "content": system},
            {"role": "user", "content": reset.observation},
        ]
        return self._loop(traj, env, ledger, parallel_group, budget=self.max_steps)

    def continue_from(self, task: Task, traj: Trajectory, *, ledger=None,
                      parallel_group: str | None = None,
                      extra_steps: int | None = None) -> Trajectory:
        env = traj.env
        traj.stalled = False
        budget = traj.depth + (extra_steps or self.max_steps)
        return self._loop(traj, env, ledger, parallel_group, budget=budget)

    def _loop(self, traj: Trajectory, env, ledger, parallel_group, *, budget: int) -> Trajectory:
        messages = traj._messages  # type: ignore[attr-defined]
        while traj.depth < budget and not traj.done:
            resp = self.client.chat(
                self.model, messages, ledger=ledger,
                parallel_group=parallel_group, temperature=self.temperature,
            )
            turn = _parse_turn(resp.text)
            action_name = (turn.get("action") or "").strip()
            ai = turn.get("action_input") or {}
            if not isinstance(ai, dict):
                ai = {}

            if action_name.lower() in ("respond", "finish", "answer", ""):
                content = ai.get("content") or ai.get("answer") or str(turn.get("thought", ""))
                tb_action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": str(content)})
            else:
                tb_action = Action(name=action_name, kwargs=ai)

            env_resp = env.step(tb_action)
            step = Step(thought=str(turn.get("thought", "")),
                        action=action_name or RESPOND_ACTION_NAME,
                        action_input=ai, observation=str(env_resp.observation))
            traj.steps.append(step)
            messages.append({"role": "assistant", "content": json.dumps(turn)})
            messages.append({"role": "user", "content": f"OBSERVATION: {env_resp.observation}"})

            if env_resp.done:
                traj.final_answer = str(env_resp.observation)
                traj.reward = float(env_resp.reward)
                break
        return traj


class TauTerminalVerifier:
    """Stand-in TerminalVerifier. The real reward is env-graded and rides on the
    Trajectory, so the runner uses that; this only answers the fallback path and
    the abstention probe. tau-bench tasks are all solvable, so abstaining is never
    the right call -> abstention reward 0."""

    def score_final(self, task: Task, answer: str) -> Score:
        return Score(value=0.0, rationale="env-graded; reward rides on the trajectory")

    def score_abstention(self, task: Task) -> Score:
        return Score(value=0.0, rationale="tau-bench tasks are solvable; STOP is never correct")


@dataclass
class TauBenchBenchmark:
    """wdp Benchmark over a tau-bench domain split.

    `task_indices` selects which tasks to use (default: a small prefix, since each
    task is a multi-turn live conversation and the self-improvement loop multiplies
    that cost). Tools are driven through the env, not a global tool dict, so
    `tools()` is empty and the executor talks to the env directly."""
    env_name: str = "retail"
    split: str = "test"
    task_indices: list[int] = field(default_factory=lambda: list(range(10)))
    name: str = "taubench"

    def tasks(self) -> list[Task]:
        # The agent must not see the gold instruction (that lives with the user
        # simulator); prompt is unused because the executor seeds the conversation
        # from env.reset(). We only carry the task_index the executor needs.
        return [
            Task(id=f"{self.env_name}-{self.split}-{i}", prompt="",
                 metadata={"task_index": i, "env_name": self.env_name, "split": self.split})
            for i in self.task_indices
        ]

    def tools(self) -> dict:
        return {}

    def terminal_verifier(self) -> TauTerminalVerifier:
        return TauTerminalVerifier()
