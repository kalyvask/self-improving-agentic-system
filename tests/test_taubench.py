"""Offline test for the tau-bench adapter (no network, no API key, no real env).

The real `get_env` path is intentionally avoided: tau-bench's LLM user simulator
makes a model call at construction, so any live test would spend credits and need
a key. Instead we inject a scripted `FakeEnv` via `TauReActExecutor.env_factory`
and a scripted `FakeClient`, then assert the adapter mechanics:

  - the agent's JSON turns are mapped to tau-bench `Action`s (tool call vs respond),
  - the env's ground-truth reward is captured onto `Trajectory.reward` at `done`,
  - WIDER (`run`) opens a fresh env (reset), DEEPER (`continue_from`) resumes the
    same env and pushes it to completion,
  - the agent's own model calls are billed to the shared ledger.
"""
from __future__ import annotations

import json

from tau_bench.types import Action, RESPOND_ACTION_NAME

from wdp.cost import CostLedger, Spend
from wdp.llm.openrouter import LLMResponse
from wdp.executor.react import Task
from wdp.benchmarks.taubench import (
    TauReActExecutor,
    TauTerminalVerifier,
    TauBenchBenchmark,
)


class _Resp:
    """Stand-in for tau-bench's EnvResetResponse / EnvResponse (only the fields
    the adapter reads)."""

    def __init__(self, observation, reward=0.0, done=False):
        self.observation = observation
        self.reward = reward
        self.done = done


class FakeEnv:
    """Scripted tau-bench Env. Records the actions it receives and becomes `done`
    on the 2nd step, returning a known ground-truth reward."""

    wiki = "RETAIL POLICY: be helpful."
    tools_info = [
        {
            "type": "function",
            "function": {
                "name": "get_order",
                "description": "Look up an order by id.",
                "parameters": {
                    "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"],
                },
            },
        }
    ]

    def __init__(self, task_index: int, reward: float = 1.0):
        self.task_index = task_index
        self.reward = reward
        self.actions: list[Action] = []
        self.reset_calls = 0

    def reset(self, task_index=None):
        self.reset_calls += 1
        return _Resp(observation="Hi, I need help with my order.")

    def step(self, action: Action) -> _Resp:
        self.actions.append(action)
        # Done once the agent has acted twice (tool lookup, then respond).
        if len(self.actions) >= 2:
            return _Resp(observation="###STOP### thanks", reward=self.reward, done=True)
        return _Resp(observation="order #1 is shipped", reward=0.0, done=False)


class FakeClient:
    """Emits a scripted JSON turn per call: first a tool call, then a respond.
    Folds a realistic Spend into the ledger like the real client does."""

    _TURNS = [
        {"thought": "look it up", "action": "get_order",
         "action_input": {"order_id": "1"}},
        {"thought": "tell the customer", "action": "respond",
         "action_input": {"content": "Your order is shipped."}},
    ]

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, model, messages, *, ledger=None, parallel_group=None,
             temperature=0.0, max_tokens=None, **kwargs) -> LLMResponse:
        turn = self._TURNS[min(self.calls, len(self._TURNS) - 1)]
        self.calls += 1
        spend = Spend(model=model, prompt_tokens=100, completion_tokens=20,
                      wall_seconds=0.5, dollars=0.001, parallel_group=parallel_group)
        if ledger is not None:
            ledger.add(spend)
        return LLMResponse(text=json.dumps(turn), model=model, spend=spend, raw={})


def _task(i: int = 0) -> Task:
    return Task(id=f"retail-test-{i}", prompt="",
                metadata={"task_index": i, "env_name": "retail", "split": "test"})


def _executor(client, **kw):
    envs: list[FakeEnv] = []

    def factory(task_index: int) -> FakeEnv:
        env = FakeEnv(task_index)
        envs.append(env)
        return env

    ex = TauReActExecutor(client=client, model="fake/model",
                          env_factory=factory, **kw)
    return ex, envs


def test_run_maps_actions_and_captures_reward():
    client = FakeClient()
    ex, envs = _executor(client)
    ledger = CostLedger()
    traj = ex.run(_task(0), ledger=ledger, parallel_group="g")

    # The env was reset once (a fresh attempt) and saw two actions.
    assert len(envs) == 1
    assert envs[0].reset_calls == 1
    names = [a.name for a in envs[0].actions]
    assert names == ["get_order", RESPOND_ACTION_NAME]
    # The tool call carried its kwargs; the respond carried its content.
    assert envs[0].actions[0].kwargs == {"order_id": "1"}
    assert envs[0].actions[1].kwargs == {"content": "Your order is shipped."}

    # Ground-truth env reward rode onto the trajectory; not from any verifier.
    assert traj.done
    assert traj.reward == 1.0
    assert traj.final_answer is not None
    assert traj.depth == 2
    # The agent's own calls were billed.
    assert ledger.amount("dollars") > 0.0


def test_wider_opens_a_fresh_env():
    client = FakeClient()
    ex, envs = _executor(client)
    ex.run(_task(0))
    ex.run(_task(1))
    # Two independent attempts => two distinct envs, each reset once.
    assert len(envs) == 2
    assert envs[0] is not envs[1]
    assert all(e.reset_calls == 1 for e in envs)


def test_deeper_resumes_same_env():
    client = FakeClient()
    # max_steps=1 so the first attempt stops before the env is done.
    ex, envs = _executor(client, max_steps=1)
    traj = ex.run(_task(0))
    assert not traj.done
    assert traj.depth == 1
    assert len(envs) == 1

    # DEEPER continues the SAME env (no new reset) and finishes it.
    resumed = ex.continue_from(_task(0), traj, extra_steps=5)
    assert resumed is traj
    assert len(envs) == 1               # no fresh env
    assert envs[0].reset_calls == 1     # still only the original reset
    assert resumed.done
    assert resumed.reward == 1.0
    assert resumed.depth == 2


def test_terminal_verifier_is_fallback_only():
    v = TauTerminalVerifier()
    assert v.score_final(_task(), "anything").value == 0.0
    # tau-bench tasks are solvable => abstaining is never correct.
    assert v.score_abstention(_task()).value == 0.0


def test_benchmark_tasks_carry_index_and_hide_goal():
    bench = TauBenchBenchmark(task_indices=[2, 5])
    tasks = bench.tasks()
    assert [t.metadata["task_index"] for t in tasks] == [2, 5]
    # The agent must not see the gold instruction; prompt stays empty.
    assert all(t.prompt == "" for t in tasks)
    assert bench.tools() == {}
    assert isinstance(bench.terminal_verifier(), TauTerminalVerifier)
