"""Full experiment: Entropic-Risk-Sensitive Jev on RiskyWeb, with baselines.

Pipeline
--------
1. Build the larger RiskyWeb environment.
2. Load yev0-4b, extract a frozen feature per state (one forward each).
3. Train a small risk-sensitive value head on those features via entropic fitted
   value iteration at the target beta  (==> "yev0-4b + risk head").
4. Train the same head on MLP features  (==> ablation: no Jev pretraining).
5. Build baselines: exact entropic oracle, risk-neutral Jev (beta=0), stock
   yev0-4b zero-shot choice, and fixed/random heuristics.
6. Evaluate every policy by Monte-Carlo and report mean return, catastrophe rate,
   tail risk (CVaR 5%, worst case), the entropic objective U_beta, and agreement
   with the oracle.

Run:
    python -m entropic_risk_jev.src.experiment_full --beta 0.15 --device cuda
    python -m entropic_risk_jev.src.experiment_full --beta 0.15 --no-yev   # skip the 4B model
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from .baselines import (
    fixed_action_policy,
    oracle_entropic_policy,
    random_policy,
    risk_neutral_policy,
    stock_yev_policy,
)
from .env import ASK, FAST, SAFE
from .env_web import RiskyWeb, RiskyWebConfig
from .models import FeatureMatrixValueNet, MLPValueNet
from .rollout import policy_statistics
from .train_fittedq import train_fitted_vi
from .value_iteration import entropic_policy_value, risk_sensitive_value_iteration

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))


def agreement(env, policy, oracle) -> float:
    non_term = [s for s in range(env.n_states) if not env.is_terminal(s)]
    return float(np.mean([policy[s] == oracle[s] for s in non_term]))


def evaluate_all(env, policies: dict, beta: float, n_episodes: int, seed: int):
    oracle = policies["oracle (entropic VI)"]
    rows = {}
    for name, pol in policies.items():
        st = policy_statistics(
            env, pol, n_episodes=n_episodes, seed=seed, betas=(0.0, beta)
        )
        rows[name] = {
            **st.as_row(),
            "U_beta": entropic_policy_value(env, pol, beta),  # exact, noise-free
            "U_beta_mc": st.entropic_values[beta],            # MC estimate (noisy)
            "agree_oracle": agreement(env, pol, oracle),
        }
    return rows


def print_table(rows: dict, beta: float):
    print("=" * 100)
    print(f"Policy comparison on RiskyWeb  (risk objective: U_beta at beta={beta})")
    print("=" * 100)
    hdr = (
        f"{'policy':<28} | {'mean':>7} | {'catastr':>7} | {'CVaR5%':>8} | "
        f"{'worst':>7} | {'U_beta':>8} | {'agree':>6}"
    )
    print(hdr)
    print("-" * 100)
    # Sort by the risk objective (what we actually care about).
    for name, r in sorted(rows.items(), key=lambda kv: -kv[1]["U_beta"]):
        print(
            f"{name:<28} | {r['mean_return']:>7.3f} | {r['catastrophe_rate']:>7.3f} | "
            f"{r['cvar_5']:>8.3f} | {r['worst_return']:>7.2f} | {r['U_beta']:>8.3f} | "
            f"{r['agree_oracle']:>6.2f}"
        )
    print("-" * 100)
    print("U_beta is the entropic risk objective (higher = better risk-adjusted).")
    print("'agree' = fraction of states whose action matches the exact oracle.")


def run(beta: float, device: str, use_yev: bool, yev_model: str,
        n_episodes: int, seed: int, iterations: int):
    env = RiskyWeb(RiskyWebConfig(seed=seed))
    print(f"RiskyWeb: {env.n_layers} layers x {env.width} = "
          f"{env.n_layers * env.width} decision states (+2 terminal)")
    print(f"page types: {env.danger_summary()}")
    print(f"catastrophe cost range: [{env.cost_cat[:env.n_layers*env.width].min():.1f}, "
          f"{env.cost_cat[:env.n_layers*env.width].max():.1f}]")
    print()

    policies: dict = {}

    # --- exact / heuristic baselines -------------------------------------
    policies["oracle (entropic VI)"] = oracle_entropic_policy(env, beta)
    policies["risk-neutral Jev (b=0)"] = risk_neutral_policy(env)
    policies["always FAST"] = fixed_action_policy(env, FAST)
    policies["always SAFE"] = fixed_action_policy(env, SAFE)
    policies["always ASK"] = fixed_action_policy(env, ASK)
    policies["random"] = random_policy(env, seed=seed)

    # --- learned: MLP ablation -------------------------------------------
    print("[train] MLP backbone + risk head ...")
    mlp = MLPValueNet(env)
    _, _, pol_mlp = train_fitted_vi(
        env, mlp, beta=beta, iterations=iterations, device="cpu", verbose=False
    )
    policies["MLP + risk head"] = pol_mlp

    # --- learned: yev0-4b ------------------------------------------------
    if use_yev:
        from .yev_backbone import YevValueNet

        print(f"[load] {yev_model} on {device} ...")
        yev = YevValueNet(env, model_name=yev_model, device=device, freeze_base=True)

        print("[stock] yev0-4b zero-shot choice policy ...")
        policies["stock yev0-4b (zero-shot)"] = stock_yev_policy(yev)

        print("[extract] yev0-4b features for all states ...")
        feats = yev.extract_features_all(verbose=True)
        head = FeatureMatrixValueNet(feats, env.n_actions)
        print("[train] yev0-4b features + risk head ...")
        _, _, pol_yev = train_fitted_vi(
            env, head, beta=beta, iterations=iterations, device=device, verbose=False
        )
        policies["yev0-4b + risk head"] = pol_yev

    # --- evaluate --------------------------------------------------------
    print("\n[eval] Monte-Carlo rollouts ...")
    rows = evaluate_all(env, policies, beta, n_episodes, seed)
    print()
    print_table(rows, beta)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"full_comparison_beta{beta}.json")
    with open(out, "w") as f:
        json.dump({"beta": beta, "rows": rows}, f, indent=2)
    print(f"\n[saved] {out}")
    _bar_plot(rows, beta)
    return env, rows


def _bar_plot(rows: dict, beta: float):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot skipped: {exc}]")
        return
    names = list(rows.keys())
    order = sorted(range(len(names)), key=lambda i: rows[names[i]]["U_beta"])
    names = [names[i] for i in order]
    u = [rows[n]["U_beta"] for n in names]
    cat = [rows[n]["catastrophe_rate"] for n in names]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    a1.barh(names, u, color="#2563eb")
    a1.set_title(f"Entropic risk objective U_beta (beta={beta})")
    a1.set_xlabel("U_beta (higher better)")
    a2.barh(names, cat, color="#dc2626")
    a2.set_title("Catastrophe rate")
    a2.set_xlabel("P(catastrophe)")
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, f"full_comparison_beta{beta}.png")
    fig.savefig(out, dpi=130)
    print(f"[plot saved: {out}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=0.15)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-yev", dest="use_yev", action="store_false")
    ap.add_argument("--yev-model", default="choyiny/yev0-4b")
    ap.add_argument("--episodes", type=int, default=30_000)
    ap.add_argument("--iterations", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args.beta, args.device, args.use_yev, args.yev_model,
        n_episodes=args.episodes, seed=args.seed, iterations=args.iterations)


if __name__ == "__main__":
    main()
