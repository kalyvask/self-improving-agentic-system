"""Alternative-annotator test (Calderon et al., 2501.10970), adapted to a single
ground-truth annotator: a pass/fail verdict on whether a cheap LLM judge is good
enough to *act on*.

The controller leans on the cheap ProcessVerifier to choose WIDER/DEEPER/STOP. A
raw AUC of 0.57-0.71 is hard to act on -- is that "good enough"? The alt-test
reframes the question as a decision rather than a number: the judge is usable iff
its agreement with the ground-truth TerminalVerifier beats the trivial no-judge
baseline (always predict the majority outcome) by a margin epsilon, with
statistical significance.

Calderon's test tolerates an LLM annotator that is slightly *worse* than a human
because it is far cheaper; epsilon plays the same role here -- the cheap judge may
clear only a modest bar over the baseline and still be worth acting on, since it
costs a fraction of a real rollout. With a single ground-truth annotator the
multi-annotator "winning rate" omega collapses to one one-sided test, and we
report it honestly as such rather than dressing it up as the full procedure.

numpy/stdlib only; percentile bootstrap, assumption-light.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AltTestResult:
    n: int
    threshold: float            # judge score >= threshold => predicts "solved"
    epsilon: float              # margin over baseline the judge must clear
    judge_acc: float            # agreement of the thresholded judge with truth
    baseline_acc: float         # always-predict-majority-outcome accuracy
    advantage: float            # judge_acc - baseline_acc (point estimate)
    advantage_lo: float         # bootstrap CI on the paired advantage
    advantage_hi: float
    passed: bool                # advantage_lo > epsilon
    alpha: float

    def __str__(self) -> str:
        verdict = "PASS -- safe to act on" if self.passed else "FAIL -- not better enough than no judge"
        return (f"alt-test [{verdict}] n={self.n} thr={self.threshold:.2f} eps={self.epsilon:.2f}\n"
                f"  judge agreement {self.judge_acc:.3f} vs majority baseline {self.baseline_acc:.3f}\n"
                f"  advantage {self.advantage:+.3f} [{self.advantage_lo:+.3f}, {self.advantage_hi:+.3f}] "
                f"(needs lower bound > {self.epsilon:.2f})")


def alt_test(judge_scores: list[float], truth: list[float], *,
             threshold: float = 0.5, epsilon: float = 0.05,
             alpha: float = 0.05, n_boot: int = 10000, seed: int = 0) -> AltTestResult:
    """Decide whether a cheap judge is good enough to act on vs the no-judge baseline.

    judge_scores: the cheap verifier's success signal per item (e.g. process score).
    truth: ground-truth solved 0/1 per item (the TerminalVerifier).
    The judge predicts "solved" when its score >= threshold. The baseline predicts
    the majority outcome for every item. The judge passes iff the lower bootstrap
    bound on (judge_acc - baseline_acc) exceeds epsilon.
    """
    s = np.asarray(judge_scores, dtype=float)
    g = np.asarray(truth, dtype=float)
    n = len(g)
    if n == 0:
        return AltTestResult(0, threshold, epsilon, 0.0, 0.0, 0.0, 0.0, 0.0, False, alpha)

    pred = (s >= threshold).astype(float)
    majority = 1.0 if g.mean() >= 0.5 else 0.0       # the no-judge cheap alternative
    judge_agree = (pred == g).astype(float)          # per-item agreement, paired
    base_agree = (g == majority).astype(float)
    adv = judge_agree - base_agree                   # per-item paired advantage in {-1,0,1}

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = adv[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return AltTestResult(
        n=n, threshold=threshold, epsilon=epsilon,
        judge_acc=float(judge_agree.mean()), baseline_acc=float(base_agree.mean()),
        advantage=float(adv.mean()), advantage_lo=float(lo), advantage_hi=float(hi),
        passed=bool(lo > epsilon), alpha=alpha,
    )


def best_threshold(judge_scores: list[float], truth: list[float], *,
                   grid: int = 19, **kwargs) -> AltTestResult:
    """Run alt_test across candidate thresholds and return the most favorable one
    (highest agreement). Reports the judge at its best operating point so a FAIL is
    a verdict on the judge, not on a poorly-chosen cutoff."""
    best: AltTestResult | None = None
    for t in (np.linspace(0.05, 0.95, grid)):
        r = alt_test(judge_scores, truth, threshold=float(t), **kwargs)
        if best is None or r.judge_acc > best.judge_acc:
            best = r
    return best
