"""Beta-sweep on RiskyWeb: the risk-neutral policy collapses, the method tracks.

For each risk level beta we compare the *exact* entropic objective U_beta of:

* the risk-sensitive **oracle** at that beta (best achievable);
* the fixed **risk-neutral** policy (ordinary Jev, argmax at beta=0), evaluated
  under the risk objective -- this is what you get if you ignore failure costs;
* the **learned** risk-sensitive policy (MLP risk head by default; --yev to use
  yev0-4b features), which should track the oracle.

The gap between the oracle and the risk-neutral policy is the value of being
risk-aware, and it widens sharply as beta (how much you fear costly failures)
grows.

Run:
    python -m entropic_risk_jev.src.sweep_beta
    python -m entropic_risk_jev.src.sweep_beta --yev --device cuda
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from .env_web import RiskyWeb, RiskyWebConfig
from .models import FeatureMatrixValueNet, MLPValueNet
from .train_fittedq import train_fitted_vi
from .value_iteration import entropic_policy_value, risk_sensitive_value_iteration

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))


def run(betas, use_yev: bool, device: str, seed: int, iterations: int):
    env = RiskyWeb(RiskyWebConfig(seed=seed))
    nonterm = [s for s in range(env.n_states) if not env.is_terminal(s)]

    # Risk-neutral (vanilla Jev) policy is fixed across the sweep.
    rn_policy = risk_sensitive_value_iteration(env, 0.0).policy

    # Optional: extract yev features once; otherwise use the MLP backbone.
    yev_feats = None
    if use_yev:
        from .yev_backbone import YevValueNet

        print(f"[load] yev0-4b on {device}; extracting features once ...")
        yev = YevValueNet(env, device=device, freeze_base=True)
        yev_feats = yev.extract_features_all(verbose=False)

    print(f"\nRiskyWeb: {env.n_layers * env.width} states. Sweeping beta.\n")
    hdr = (f"{'beta':>5} | {'oracle U':>9} | {'method U':>9} | "
           f"{'risk-neutral U':>14} | {'method agree':>12}")
    print(hdr)
    print("-" * len(hdr))

    curves = {"beta": [], "oracle": [], "method": [], "risk_neutral": []}
    for beta in betas:
        oracle = risk_sensitive_value_iteration(env, beta)
        if use_yev:
            net = FeatureMatrixValueNet(yev_feats, env.n_actions)
            dev = device
        else:
            net = MLPValueNet(env)
            dev = "cpu"
        _, _, learned = train_fitted_vi(
            env, net, beta=beta, iterations=iterations, device=dev, verbose=False
        )
        u_oracle = entropic_policy_value(env, oracle.policy, beta)
        u_method = entropic_policy_value(env, learned, beta)
        u_rn = entropic_policy_value(env, rn_policy, beta)
        agree = float(np.mean([learned[s] == oracle.policy[s] for s in nonterm]))
        print(f"{beta:>5.2f} | {u_oracle:>9.3f} | {u_method:>9.3f} | "
              f"{u_rn:>14.3f} | {agree:>12.2f}")
        curves["beta"].append(beta)
        curves["oracle"].append(u_oracle)
        curves["method"].append(u_method)
        curves["risk_neutral"].append(u_rn)

    print("\nReading: 'risk-neutral U' is vanilla Jev's policy scored under the")
    print("risk objective -- it falls off a cliff as beta grows, while the learned")
    print("risk-sensitive policy tracks the oracle.")
    _plot(curves, use_yev)
    return curves


def _plot(curves, use_yev):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot skipped: {exc}]")
        return
    b = curves["beta"]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(b, curves["oracle"], "o-", color="#16a34a", label="oracle (entropic VI)")
    tag = "yev0-4b + risk head" if use_yev else "MLP + risk head"
    ax.plot(b, curves["method"], "s--", color="#2563eb", label=tag)
    ax.plot(b, curves["risk_neutral"], "^-", color="#dc2626",
            label="risk-neutral Jev policy")
    ax.set_xlabel("risk aversion  beta")
    ax.set_ylabel("exact entropic objective  U_beta")
    ax.set_title("Being risk-aware pays off more as failures get scarier")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "beta_sweep_web.png")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"[plot saved: {out}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--betas", type=float, nargs="+",
                    default=[0.0, 0.02, 0.05, 0.1, 0.15, 0.25, 0.4])
    ap.add_argument("--yev", dest="use_yev", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iterations", type=int, default=400)
    args = ap.parse_args()
    run(args.betas, args.use_yev, args.device, args.seed, args.iterations)


if __name__ == "__main__":
    main()
