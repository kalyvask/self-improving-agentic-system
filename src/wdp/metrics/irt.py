"""Rasch (1PL) item-response model for tau-bench tasks.

From tinyBenchmarks (Maia Polo et al., ICML 2024) and the IRT scaling-laws
reading: treat each task as an *item* with a latent difficulty z, and each
policy/round that attempted it as a *respondent* with a latent ability theta, so

    P(task t solved by respondent i) = sigmoid(theta_i - z_t).

This buys two things the project needs:
  - a calibrated per-task difficulty z to replace the crude (1 - first_score)
    proxy in the WIDER-vs-DEEPER feature, and
  - Fisher item information I(theta) = p(1-p), which says which tasks actually
    discriminate near the agent's ability -- so a small eval can be chosen to
    have power instead of being a random underpowered draw.

Plain numpy joint-MLE with L2 anchoring and a mean-zero difficulty constraint for
identifiability. Tiny data, so this is a descriptive estimate, reported with that
caveat -- not a claim of a precisely identified latent trait.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class RaschFit:
    items: list[str]                 # task ids, aligned with difficulty
    respondents: list[str]           # policy/round labels, aligned with ability
    difficulty: np.ndarray           # z_t, higher = harder
    ability: np.ndarray              # theta_i, higher = stronger
    n_responses: int

    def solve_prob(self, ability: float) -> np.ndarray:
        return _sigmoid(ability - self.difficulty)

    def information(self, ability: float) -> np.ndarray:
        """Fisher information per item at a given ability: p(1-p). Peaks where the
        item difficulty matches the ability -- those tasks discriminate best."""
        p = self.solve_prob(ability)
        return p * (1.0 - p)

    def pass_at_k(self, ability: float, k: int) -> np.ndarray:
        """IRT pass@k = 1 - (1 - p)^k, the coverage the WIDER action chases."""
        p = self.solve_prob(ability)
        return 1.0 - (1.0 - p) ** k


def fit_rasch(resp_item: list[int], resp_user: list[int], success: list[float],
              n_items: int, n_users: int, *, l2: float = 1e-2,
              lr: float = 0.5, epochs: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Joint-MLE Rasch fit on long-format responses. Returns (difficulty, ability).

    Identifiability: difficulties are centered to mean zero each step (the model
    is invariant to a constant shift between theta and z)."""
    item = np.asarray(resp_item, dtype=int)
    user = np.asarray(resp_user, dtype=int)
    y = np.asarray(success, dtype=float)
    z = np.zeros(n_items)
    theta = np.zeros(n_users)
    n = len(y)
    for _ in range(epochs):
        p = _sigmoid(theta[user] - z[item])
        err = p - y                                  # dNLL/d(logit)
        gz = np.zeros(n_items)
        gt = np.zeros(n_users)
        # logit = theta - z, so dlogit/dz = -1 and dlogit/dtheta = +1.
        np.add.at(gz, item, -err)
        np.add.at(gt, user, +err)
        gz = gz / n + l2 * z
        gt = gt / n + l2 * theta
        z -= lr * gz
        theta -= lr * gt
        z -= z.mean()                                # anchor
    return z, theta


def fit_from_responses(responses: list[tuple[str, str, float]],
                       **kwargs) -> RaschFit:
    """Convenience: responses are (task_id, respondent_label, solved 0/1)."""
    items = sorted({t for t, _, _ in responses})
    users = sorted({u for _, u, _ in responses})
    i_idx = {t: i for i, t in enumerate(items)}
    u_idx = {u: i for i, u in enumerate(users)}
    z, theta = fit_rasch(
        [i_idx[t] for t, _, _ in responses],
        [u_idx[u] for _, u, _ in responses],
        [s for _, _, s in responses],
        n_items=len(items), n_users=len(users), **kwargs,
    )
    return RaschFit(items=items, respondents=users, difficulty=z,
                    ability=theta, n_responses=len(responses))
