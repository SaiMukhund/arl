"""Sample-efficiency study: do yev0-4b features help the risk-sensitive learner?

Trains the entropic DQN on two observation representations -- engineered MLP
features vs frozen yev0-4b hidden features -- across several interaction budgets
and seeds, and compares the exact entropic objective U_beta and the major-
catastrophe probability of the learned policies.

Run:
    python -m entropic_risk_jev.src.sample_efficiency --device cuda
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from .distributional import distributional_policy_evaluation, policy_terminal_probs
from .dqn import ObsMatrix, mlp_obs_matrix, train_value_dqn
from .env_v2 import RiskyFlow, RiskyFlowConfig
from .risk_measures import entropic_categorical

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))


def _standardize(M):
    return ((M - M.mean(0, keepdims=True)) / (M.std(0, keepdims=True) + 1e-6)).astype(np.float32)


def run(budgets, seeds, beta, device, yev_model):
    torch.set_num_threads(8)
    env = RiskyFlow(RiskyFlowConfig(seed=0))

    def ev(pol):
        p, z = distributional_policy_evaluation(env, pol, natoms=501)
        return (entropic_categorical(z, p, beta),
                policy_terminal_probs(env, pol)[env.CAT_MAJOR])

    reps = {"mlp": mlp_obs_matrix(env)}
    from .yev_backbone import YevValueNet
    print(f"[load] {yev_model} + extract features ...")
    yev = YevValueNet(env, model_name=yev_model, device=device, freeze_base=True)
    reps["yev"] = ObsMatrix(_standardize(yev.extract_features_all(verbose=False)))

    results = {rep: {"U": {}, "Pmaj": {}} for rep in reps}
    for rep, obs in reps.items():
        for B in budgets:
            Us, Ps = [], []
            for sd in seeds:
                dev = device if rep == "yev" else "cpu"
                pol, _ = train_value_dqn(env, obs, mode="entropic", beta=beta,
                                         episodes=B, device=dev, seed=sd, batch_size=256)
                u, pm = ev(pol)
                Us.append(u); Ps.append(pm)
            results[rep]["U"][B] = (float(np.mean(Us)), float(np.std(Us)))
            results[rep]["Pmaj"][B] = (float(np.mean(Ps)), float(np.std(Ps)))
            print(f"  {rep:4s} B={B:5d}: U_beta={np.mean(Us):8.2f}+-{np.std(Us):5.2f}  "
                  f"P(major)={np.mean(Ps):.4f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    json.dump({"beta": beta, "budgets": budgets, "results": results},
              open(os.path.join(RESULTS_DIR, "sample_efficiency.json"), "w"), indent=2)
    _plot(results, budgets)


def _plot(results, budgets):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot skipped: {exc}]"); return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    col = {"mlp": "#2563eb", "yev": "#f59e0b"}
    for rep in results:
        um = [results[rep]["U"][B][0] for B in budgets]
        us = [results[rep]["U"][B][1] for B in budgets]
        pm = [results[rep]["Pmaj"][B][0] for B in budgets]
        a1.errorbar(budgets, um, yerr=us, marker="o", label=rep, color=col[rep], capsize=3)
        a2.plot(budgets, pm, marker="o", label=rep, color=col[rep])
    a1.set_xlabel("training episodes"); a1.set_ylabel("entropic U_beta"); a1.legend()
    a1.set_title("Sample efficiency: entropic objective")
    a2.set_xlabel("training episodes"); a2.set_ylabel("P(major catastrophe)"); a2.legend()
    a2.set_title("Sample efficiency: catastrophe rate")
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "sample_efficiency.png")
    fig.savefig(out, dpi=130); print(f"[saved] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--yev-model", default="choyiny/yev0-4b")
    ap.add_argument("--budgets", type=int, nargs="+", default=[500, 1500, 3000])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    args = ap.parse_args()
    run(args.budgets, args.seeds, args.beta, args.device, args.yev_model)


if __name__ == "__main__":
    main()
