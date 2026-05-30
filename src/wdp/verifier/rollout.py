"""Rollout-grounded difficulty (Math-Shepherd style), using the FREE terminal verifier.

The cheap LLM process verifier fails the alternative-annotator test (it agrees with the
ground-truth outcome barely above a majority-class baseline), so the `difficulty` feature
derived from it (`1 - first_process_score`) is near-noise. Math-Shepherd (arXiv:2312.08935)
grounds a process/difficulty signal in *rollouts*: from a state, fork N continuations, score
each with the exact verifier, and use the success fraction. We apply that to task difficulty:

    difficulty(task) = 1 - (fraction of N fresh attempts the terminal verifier marks correct)

This needs no trained PRM and no LLM judge -- only the executor (which we already pay for) and
the terminal grader (exact and free on the arithmetic suite). It replaces a noisy judge-derived
feature with a calibrated, grounded one. Results are cached by (task_id, prompt-hash) so each
task is probed once; the forked attempts are the only spend.
"""
from __future__ import annotations

import hashlib

from wdp.cost import CostLedger
from wdp.verifier.scorer import Score


class RolloutProcessVerifier:
    """Estimates task difficulty from the success rate of N fresh forked attempts.

    Injected with the same executor and terminal verifier the loop uses. `difficulty`
    returns a value in [0,1] (1 = no fork solved it, 0 = all solved). Cached per task."""

    def __init__(self, executor, terminal, *, n_rollouts: int = 4,
                 solved_threshold: float = 0.99) -> None:
        if n_rollouts <= 0:
            raise ValueError("n_rollouts must be positive")
        self._executor = executor
        self._terminal = terminal
        self.n_rollouts = n_rollouts
        self.solved_threshold = solved_threshold
        self._cache: dict[str, float] = {}
        # Labeling rollouts are a TRAIN-TIME cost, separate from per-task solve/cost.
        # They bill here (not into the task ledger) so the cost is visible and can be
        # reported separately, instead of silently distorting the per-task cost metric
        # (and unevenly, since difficulty is cached and only the first task would pay).
        self.labeling_ledger = CostLedger()

    @staticmethod
    def _key(task) -> str:
        h = hashlib.sha1(str(getattr(task, "prompt", "")).encode()).hexdigest()[:12]
        return f"{getattr(task, 'id', '?')}:{h}"

    def _terminal_value(self, task, traj) -> float:
        # Env-graded benchmarks (tau-bench) carry reward on the trajectory; trust it.
        if getattr(traj, "reward", None) is not None:
            return float(traj.reward)
        if traj.final_answer is None:
            return 0.0
        return float(self._terminal.score_final(task, traj.final_answer).value)

    def solve_fraction(self, task, *, ledger: CostLedger | None = None) -> float:
        """Fraction of N fresh attempts that solve the task (cached)."""
        key = self._key(task)
        if key in self._cache:
            return self._cache[key]
        led = ledger if ledger is not None else self.labeling_ledger
        solved = 0
        for i in range(self.n_rollouts):
            traj = self._executor.run(task, ledger=led,
                                      parallel_group=f"diffprobe:{key}:{i}")
            if self._terminal_value(task, traj) >= self.solved_threshold:
                solved += 1
        frac = solved / self.n_rollouts
        self._cache[key] = frac
        return frac

    def difficulty(self, task, *, ledger: CostLedger | None = None) -> float:
        """Rollout-grounded difficulty in [0,1]; 1 - solve_fraction."""
        return 1.0 - self.solve_fraction(task, ledger=ledger)

    def score(self, task, *, ledger: CostLedger | None = None) -> Score:
        return Score(value=self.solve_fraction(task, ledger=ledger),
                     rationale=f"rollout solve fraction over {self.n_rollouts} fresh attempts")
