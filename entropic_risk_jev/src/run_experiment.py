"""Headline experiment: how entropic risk reshapes the optimal Jev-style policy.

Sweeps the risk parameter ``beta`` and, for each value, solves the RiskyCheckout
MDP exactly with risk-sensitive value iteration.  Reports the per-stage optimal
action and Monte-Carlo outcome statistics, showing the risk-neutral policy
(``beta=0``, i.e. ordinary Jev value learning) take the expensive-tail shortcuts
while risk-averse policies escalate / play safe on exactly the high-cost stages.

Run (from the repo root):
    python -m entropic_risk_jev.src.run_experiment
"""

from __future__ import annotations

import argparse

import numpy as np

from .env import ACTION_NAMES, RiskyCheckout, RiskyCheckoutConfig
from .rollout import policy_statistics
from .value_iteration import risk_sensitive_value_iteration


def _policy_string(env: RiskyCheckout, policy: np.ndarray) -> str:
    return " ".join(ACTION_NAMES[policy[i]] for i in range(env.n_stages))


def run(betas, n_episodes: int = 20_000, seed: int = 0, make_plot: bool = False):
    env = RiskyCheckout(RiskyCheckoutConfig(seed=seed))

    print("=" * 78)
    print("Entropic-Risk-Sensitive Jev  --  RiskyCheckout")
    print("=" * 78)
    print(f"stages           : {env.n_stages}")
    print(f"catastrophe prob : {list(env.cfg.p_cat)}")
    print(f"catastrophe cost : {list(env.cfg.cost_cat)}   (heterogeneous)")
    print(f"goal reward      : {env.cfg.goal_reward}")
    print()
    print(f"Per-stage optimal action (stages 1..{env.n_stages} left->right):")
    print("-" * 78)

    rows = []
    for beta in betas:
        sol = risk_sensitive_value_iteration(env, beta)
        stats = policy_statistics(env, sol.policy, n_episodes=n_episodes, seed=seed)
        rows.append((beta, sol, stats))
        print(
            f"beta={beta:<5.2f} | {_policy_string(env, sol.policy):<28} | "
            f"V0={sol.V[env.start_state]:+7.3f}"
        )

    print("-" * 78)
    print("Realised outcomes under each optimal policy (Monte-Carlo):")
    print("-" * 78)
    header = (
        f"{'beta':>5} | {'success':>8} | {'catastr':>8} | {'mean':>8} | "
        f"{'std':>7} | {'CVaR5%':>8} | {'worst':>8}"
    )
    print(header)
    for beta, _sol, st in rows:
        print(
            f"{beta:>5.2f} | {st.success_rate:>8.3f} | {st.catastrophe_rate:>8.3f} | "
            f"{st.mean_return:>8.3f} | {st.std_return:>7.3f} | {st.cvar_5:>8.3f} | "
            f"{st.worst_return:>8.2f}"
        )

    print("-" * 78)
    print("Reading: beta=0 is ordinary (risk-neutral) Jev value learning -- it")
    print("uses the FAST shortcut almost everywhere and eats a 6% catastrophe")
    print("rate. As beta grows the policy peels FAST off the MOST EXPENSIVE stage")
    print("first and climbs the ladder FAST->SAFE->ASK, trading mean return for a")
    print("large cut in catastrophe rate and tail risk (CVaR / worst case).")

    if make_plot:
        _plot(env, rows)
    return env, rows


def _plot(env, rows):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[plot skipped: {exc}]")
        return
    betas = [r[0] for r in rows]
    mean = [r[2].mean_return for r in rows]
    cat = [r[2].catastrophe_rate for r in rows]
    cvar = [r[2].cvar_5 for r in rows]

    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax1.plot(betas, mean, "o-", color="#2563eb", label="mean return")
    ax1.plot(betas, cvar, "s--", color="#7c3aed", label="CVaR 5% (tail)")
    ax1.set_xlabel("risk aversion  beta")
    ax1.set_ylabel("return")
    ax2 = ax1.twinx()
    ax2.plot(betas, cat, "^-", color="#dc2626", label="catastrophe rate")
    ax2.set_ylabel("catastrophe rate", color="#dc2626")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="center right", fontsize=8)
    ax1.set_title("Entropic risk trades mean return for tail safety")
    fig.tight_layout()
    import os

    out = os.path.join(os.path.dirname(__file__), "..", "results", "risk_sweep.png")
    out = os.path.abspath(out)
    fig.savefig(out, dpi=130)
    print(f"[plot saved: {out}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--betas",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.1, 0.2, 0.4, 0.8],
    )
    ap.add_argument("--episodes", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    run(args.betas, n_episodes=args.episodes, seed=args.seed, make_plot=args.plot)


if __name__ == "__main__":
    main()
