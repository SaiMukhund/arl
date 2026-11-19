"""Risk-sensitive fitted value iteration (the learnable 'training' pipeline).

This trains a value-network backbone (MLP surrogate or yev0-4b feature backbone)
to represent the *entropic risk-sensitive* action-value function, using targets
computed by the entropic Bellman operator over the environment's heterogeneous
catastrophe costs.

Algorithm (model-based fitted VI with entropic targets)
-------------------------------------------------------
Repeat:
  1. V(s)        = max_a  Q_theta(s, a)                       (current network)
  2. target Q(s,a) = -(1/beta) log sum_{s'} P(s'|s,a)
                         exp( -beta ( R(s,a,s') + V(s') ) )   (entropic backup)
  3. theta <- argmin  || Q_theta(s,a) - target(s,a) ||^2       (regression)

Step 2 is exactly where "how costly rare failures are" enters training: the
exponential weighting of the large negative catastrophe rewards R(s,a,CAT) is
what pushes Q down for risky actions as beta grows. For beta -> 0 the targets
become the ordinary expected-value backup and the network learns risk-neutral
Jev values.

Because the backbone is swappable, the identical loop fine-tunes a yev0-4b value
head (see ``yev_backbone.py``); only the feature extractor changes.

Run:
    python -m entropic_risk_jev.src.train_fittedq --beta 0.15
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from .env import ACTION_NAMES, RiskyCheckout, RiskyCheckoutConfig
from .risk import entropic_backup
from .value_iteration import risk_sensitive_value_iteration


def train_fitted_vi(
    env: RiskyCheckout,
    backbone: torch.nn.Module,
    beta: float,
    iterations: int = 300,
    inner_epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cpu",
    verbose: bool = True,
):
    """Fit ``backbone`` to the entropic risk-sensitive Q via fitted VI."""
    backbone = backbone.to(device)
    opt = torch.optim.Adam(backbone.parameters(), lr=lr)
    all_states = list(range(env.n_states))
    P = torch.tensor(env.P, dtype=torch.float64, device=device)
    R = torch.tensor(env.R, dtype=torch.float64, device=device)
    terminals = env.terminal_states

    for it in range(1, iterations + 1):
        # --- build entropic Bellman targets from the current network ---------
        with torch.no_grad():
            Q = backbone(all_states).double()            # (S, A)
            V = Q.max(dim=1).values                        # (S,)
            V_np = V.detach().cpu().numpy()
            V_np[list(terminals)] = 0.0
            target_np = entropic_backup(env.P, env.R, V_np, beta)  # (S, A)
            target = torch.tensor(target_np, dtype=torch.float32, device=device)

        # --- regress network onto targets ------------------------------------
        for _ in range(inner_epochs):
            opt.zero_grad()
            pred = backbone(all_states)
            loss = torch.mean((pred - target) ** 2)
            loss.backward()
            opt.step()

        if verbose and (it % 50 == 0 or it == 1):
            print(f"  iter {it:4d}  fit-loss={loss.item():.3e}")

    with torch.no_grad():
        Q_final = backbone(all_states).cpu().numpy()
    policy = Q_final.argmax(axis=1)
    return backbone, Q_final, policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=0.15)
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--backbone", choices=["mlp", "yev"], default="mlp")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--yev-model", default="choyiny/yev0-4b")
    args = ap.parse_args()

    env = RiskyCheckout(RiskyCheckoutConfig())

    if args.backbone == "mlp":
        from .models import MLPValueNet

        net = MLPValueNet(env)
    else:
        from .yev_backbone import YevValueNet

        net = YevValueNet(env, model_name=args.yev_model, device=args.device)

    print(f"Training risk-sensitive fitted VI  (backbone={args.backbone}, beta={args.beta})")
    net, Q_learned, policy_learned = train_fitted_vi(
        env, net, beta=args.beta, iterations=args.iterations, device=args.device
    )

    # Compare against exact value iteration.
    exact = risk_sensitive_value_iteration(env, args.beta)
    pol_str = lambda p: " ".join(ACTION_NAMES[p[i]] for i in range(env.n_stages))
    print("-" * 60)
    print(f"learned policy : {pol_str(policy_learned)}")
    print(f"exact   policy : {pol_str(exact.policy)}")
    match = np.array_equal(policy_learned[: env.n_stages], exact.policy[: env.n_stages])
    print(f"policies match : {match}")
    q_err = np.abs(Q_learned[: env.n_stages] - exact.Q[: env.n_stages]).max()
    print(f"max |Q| error  : {q_err:.4f}")


if __name__ == "__main__":
    main()
