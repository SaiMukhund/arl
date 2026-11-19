"""Full risk-sensitive comparison on RiskyFlow, with yev0-4b and RL baselines.

Methods compared
----------------
Model-based references (use the true state; upper bounds per attitude):
  * oracle entropic (exact VI)
  * risk-neutral (distributional DP, mean-greedy)          <- "without entropic risk"
  * nested CVaR (distributional DP, cvar-greedy)

Model-free learners (learn from sampled experience on observations):
  * DQN neutral          -- standard DQN                    <- "without entropic risk"
  * DQN entropic         -- exponential-utility DQN
  * QR-DQN + CVaR        -- quantile DQN with CVaR selection
  * DQN entropic (yev)   -- same, on yev0-4b features
  * QR-DQN+CVaR (yev)    -- same, on yev0-4b features

Heuristics / zero-shot:
  * always FAST / SAFE / ASK, random, stock yev0-4b (zero-shot choice)

Every policy is scored EXACTLY (no Monte-Carlo noise) via distributional policy
evaluation of the true return distribution: mean, CVaR_alpha, entropic U_beta,
and minor/major failure probabilities, plus agreement with the entropic oracle.

Run:
    python -m entropic_risk_jev.src.experiment_v2 --beta 0.1 --device cuda
    python -m entropic_risk_jev.src.experiment_v2 --beta 0.1 --no-yev
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from .distributional import (
    distributional_policy_evaluation,
    distributional_risk_control,
    policy_terminal_probs,
)
from .dqn import ObsMatrix, mlp_obs_matrix, train_value_dqn
from .env import ASK, FAST, SAFE
from .env_v2 import RiskyFlow, RiskyFlowConfig
from .risk_measures import (
    cvar_categorical,
    entropic_categorical,
    make_risk_fn,
    mean_categorical,
)
from .value_iteration import risk_sensitive_value_iteration

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))


def standardize(M: np.ndarray) -> np.ndarray:
    mu = M.mean(axis=0, keepdims=True)
    sd = M.std(axis=0, keepdims=True) + 1e-6
    return ((M - mu) / sd).astype(np.float32)


def evaluate(env, policy, beta, alpha, oracle):
    p, z = distributional_policy_evaluation(env, policy, natoms=601)
    tp = policy_terminal_probs(env, policy)
    nonterm = [s for s in range(env.n_states) if not env.is_terminal(s)]
    agree = float(np.mean([policy[s] == oracle[s] for s in nonterm]))
    return {
        "mean": mean_categorical(z, p),
        "cvar": cvar_categorical(z, p, alpha),
        "U_beta": entropic_categorical(z, p, beta),
        "p_minor": tp[env.CAT_MINOR],
        "p_major": tp[env.CAT_MAJOR],
        "agree": agree,
    }


def run(beta, alpha, device, use_yev, yev_model, episodes, seed):
    torch.set_num_threads(8)
    env = RiskyFlow(RiskyFlowConfig(seed=seed))
    print(env.summary())
    print(f"beta={beta}  alpha(CVaR)={alpha}\n")

    policies = {}
    # ---- model-based references ----
    policies["oracle entropic (VI)"] = risk_sensitive_value_iteration(env, beta).policy
    policies["risk-neutral (DP)"] = distributional_risk_control(
        env, make_risk_fn("neutral"), natoms=251).policy
    policies["nested CVaR (DP)"] = distributional_risk_control(
        env, make_risk_fn("cvar", alpha=alpha), natoms=251).policy
    # ---- heuristics ----
    policies["always FAST"] = np.full(env.n_states, FAST)
    policies["always SAFE"] = np.full(env.n_states, SAFE)
    policies["always ASK"] = np.full(env.n_states, ASK)
    rng = np.random.default_rng(seed)
    policies["random"] = rng.integers(0, env.n_actions, size=env.n_states)

    # ---- model-free (MLP features) ----
    mlp = mlp_obs_matrix(env)
    for mode, name in [("neutral", "DQN neutral"), ("entropic", "DQN entropic"),
                       ("qr-cvar", "QR-DQN+CVaR")]:
        print(f"[train] {name} (MLP) ...")
        pol, _ = train_value_dqn(env, mlp, mode=mode, beta=beta, alpha=alpha,
                                 episodes=episodes, device="cpu", seed=seed, batch_size=256)
        policies[name] = pol

    # ---- yev0-4b ----
    if use_yev:
        from .baselines import stock_yev_policy
        from .yev_backbone import YevValueNet

        print(f"[load] {yev_model} on {device} ...")
        yev = YevValueNet(env, model_name=yev_model, device=device, freeze_base=True)
        print("[stock] yev0-4b zero-shot ...")
        policies["stock yev0-4b (0-shot)"] = stock_yev_policy(yev)
        print("[extract] yev0-4b features ...")
        yev_obs = ObsMatrix(standardize(yev.extract_features_all(verbose=False)))
        for mode, name in [("entropic", "DQN entropic (yev)"),
                           ("qr-cvar", "QR-DQN+CVaR (yev)")]:
            print(f"[train] {name} ...")
            pol, _ = train_value_dqn(env, yev_obs, mode=mode, beta=beta, alpha=alpha,
                                     episodes=episodes, device=device, seed=seed, batch_size=256)
            policies[name] = pol

    # ---- evaluate ----
    print("\n[eval] exact distributional policy evaluation ...")
    oracle = policies["oracle entropic (VI)"]
    rows = {n: evaluate(env, p, beta, alpha, oracle) for n, p in policies.items()}
    _print_table(rows, beta, alpha)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"riskyflow_beta{beta}.json")
    json.dump({"beta": beta, "alpha": alpha, "rows": rows}, open(out, "w"), indent=2)
    print(f"\n[saved] {out}")
    _plot(rows, beta, alpha)
    return env, rows


def _print_table(rows, beta, alpha):
    print("=" * 104)
    print(f"RiskyFlow comparison  (beta={beta}, CVaR alpha={alpha}).  "
          f"Sorted by entropic objective U_beta.")
    print("=" * 104)
    hdr = (f"{'policy':<24} | {'mean':>7} | {'CVaR':>8} | {'U_beta':>8} | "
           f"{'P(minor)':>8} | {'P(major)':>9} | {'agree':>5}")
    print(hdr); print("-" * 104)
    for n, r in sorted(rows.items(), key=lambda kv: -kv[1]["U_beta"]):
        print(f"{n:<24} | {r['mean']:>7.2f} | {r['cvar']:>8.2f} | {r['U_beta']:>8.2f} | "
              f"{r['p_minor']:>8.3f} | {r['p_major']:>9.4f} | {r['agree']:>5.2f}")
    print("-" * 104)
    print("U_beta = entropic objective; CVaR = mean of worst alpha tail (both higher=better).")
    print("mean is highest for risk-neutral methods, but their U_beta/CVaR are much worse.")


def _plot(rows, beta, alpha):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot skipped: {exc}]"); return
    names = sorted(rows, key=lambda n: rows[n]["U_beta"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    specs = [("U_beta", "entropic U_beta (higher=better)", "#16a34a"),
             ("cvar", f"CVaR {alpha} (higher=better)", "#7c3aed"),
             ("p_major", "P(major catastrophe) (lower=better)", "#dc2626")]
    for ax, (key, title, color) in zip(axes, specs):
        ax.barh(names, [rows[n][key] for n in names], color=color)
        ax.set_title(title, fontsize=10)
    fig.suptitle(f"RiskyFlow: risk-sensitive methods vs baselines (beta={beta})")
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, f"riskyflow_beta{beta}.png")
    fig.savefig(out, dpi=130)
    print(f"[plot saved: {out}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-yev", dest="use_yev", action="store_false")
    ap.add_argument("--yev-model", default="choyiny/yev0-4b")
    ap.add_argument("--episodes", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args.beta, args.alpha, args.device, args.use_yev, args.yev_model,
        args.episodes, args.seed)


if __name__ == "__main__":
    main()
