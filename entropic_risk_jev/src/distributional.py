"""Categorical distributional dynamic programming with risk-greedy control.

Model-based (uses the known P, R over *true* states), so these policies are
strong, near-exact references for each risk attitude:

* risk ``neutral``  -> expected-return optimal (standard distributional VI);
* risk ``entropic`` -> nested entropic policy (cross-checks exact entropic VI);
* risk ``cvar``     -> nested/Markov CVaR policy (a tail-risk baseline).

We represent the return distribution of each state on a fixed categorical support
(C51-style) and iterate a risk-greedy distributional Bellman operator:

    Z(s,a) = project_{s'~P}( R(s,a,s') + gamma * Z(s') )
    pi(s)  = argmax_a  rho( Z(s,a) )          # rho = chosen risk measure
    Z(s)   = Z(s, pi(s))

For ``rho = mean`` this is exactly distributional value iteration. For a general
coherent ``rho`` it is the standard "nested"/dynamic risk construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _project(values: np.ndarray, probs: np.ndarray, z: np.ndarray) -> np.ndarray:
    """C51 projection of atoms (values, probs) onto fixed support z (ascending)."""
    Vmin, Vmax = z[0], z[-1]
    natoms = z.shape[0]
    dz = (Vmax - Vmin) / (natoms - 1)
    v = np.clip(values, Vmin, Vmax)
    b = (v - Vmin) / dz
    lo = np.clip(np.floor(b).astype(int), 0, natoms - 1)
    hi = np.clip(np.ceil(b).astype(int), 0, natoms - 1)
    m = np.zeros(natoms, dtype=np.float64)
    # exact-hit atoms
    eq = lo == hi
    np.add.at(m, lo[eq], probs[eq])
    ne = ~eq
    np.add.at(m, lo[ne], probs[ne] * (hi[ne] - b[ne]))
    np.add.at(m, hi[ne], probs[ne] * (b[ne] - lo[ne]))
    return m


@dataclass
class DistSolution:
    policy: np.ndarray
    Z: np.ndarray            # (n_states, natoms) return distributions
    z: np.ndarray            # support
    iterations: int


def distributional_risk_control(
    env,
    risk_fn,
    gamma: float = 1.0,
    natoms: int = 301,
    v_margin: float = 5.0,
    tol: float = 1e-7,
    max_iter: int = 2000,
) -> DistSolution:
    """Risk-greedy categorical distributional value iteration.

    ``risk_fn(z, p) -> float`` is a categorical risk functional (higher better),
    e.g. from ``risk_measures.make_risk_fn``.
    """
    S, A = env.n_states, env.n_actions
    P, R = env.P, env.R
    pages = [s for s in range(S) if not env.is_terminal(s)]

    vmax = float(R.max()) + v_margin
    vmin = float(R.min()) + v_margin * -1.0  # R.min() is a big negative
    vmin = float(R.min()) - v_margin
    z = np.linspace(vmin, vmax, natoms)

    # Init: terminal states are delta at 0; others delta at 0 too.
    Z = np.zeros((S, natoms), dtype=np.float64)
    zero_idx = int(np.argmin(np.abs(z - 0.0)))
    Z[:, zero_idx] = 1.0

    # Precompute, per (s,a), the nonzero successors.
    succ = {(s, a): np.nonzero(P[s, a] > 0)[0] for s in pages for a in range(A)}

    iters = 0
    for iters in range(1, max_iter + 1):
        Z_new = Z.copy()
        max_delta = 0.0
        for s in pages:
            best_rho, best_dist = -np.inf, None
            for a in range(A):
                sp = succ[(s, a)]
                # atoms: value = R[s,a,s'] + gamma*z_j ; prob = P[s,a,s']*Z[s',j]
                vals = (R[s, a, sp][:, None] + gamma * z[None, :]).ravel()
                prbs = (P[s, a, sp][:, None] * Z[sp, :]).ravel()
                dist = _project(vals, prbs, z)
                rho = risk_fn(z, dist)
                if rho > best_rho:
                    best_rho, best_dist = rho, dist
            Z_new[s] = best_dist
            max_delta = max(max_delta, np.abs(best_dist - Z[s]).sum())
        Z = Z_new
        if max_delta < tol:
            break

    # Extract greedy policy.
    policy = np.zeros(S, dtype=int)
    for s in pages:
        rhos = []
        for a in range(A):
            sp = succ[(s, a)]
            vals = (R[s, a, sp][:, None] + gamma * z[None, :]).ravel()
            prbs = (P[s, a, sp][:, None] * Z[sp, :]).ravel()
            rhos.append(risk_fn(z, _project(vals, prbs, z)))
        policy[s] = int(np.argmax(rhos))
    return DistSolution(policy=policy, Z=Z, z=z, iterations=iters)


def distributional_policy_evaluation(
    env, policy: np.ndarray, gamma: float = 1.0, natoms: int = 401,
    v_margin: float = 5.0, tol: float = 1e-9, max_iter: int = 5000,
):
    """Exact (grid) return distribution of a fixed policy via the *risk-neutral*
    distributional Bellman. Returns (Z[start], z) -- the true distribution of
    total return from the start state, from which any risk measure is exact.
    """
    S = env.n_states
    P, R = env.P, env.R
    pages = [s for s in range(S) if not env.is_terminal(s)]
    z = np.linspace(float(R.min()) - v_margin, float(R.max()) + v_margin, natoms)
    Z = np.zeros((S, natoms)); Z[:, int(np.argmin(np.abs(z)))] = 1.0
    succ = {s: np.nonzero(P[s, policy[s]] > 0)[0] for s in pages}
    for _ in range(max_iter):
        Z_new = Z.copy(); delta = 0.0
        for s in pages:
            a = policy[s]; sp = succ[s]
            vals = (R[s, a, sp][:, None] + gamma * z[None, :]).ravel()
            prbs = (P[s, a, sp][:, None] * Z[sp, :]).ravel()
            d = _project(vals, prbs, z)
            delta = max(delta, np.abs(d - Z[s]).sum()); Z_new[s] = d
        Z = Z_new
        if delta < tol:
            break
    return Z[env.start_state], z


def policy_terminal_probs(env, policy: np.ndarray, max_iter: int = 5000):
    """Exact probabilities of reaching each terminal state under a policy."""
    S = env.n_states
    P = env.P
    pages = [s for s in range(S) if not env.is_terminal(s)]
    out = {}
    for term in env.terminal_states:
        reach = np.zeros(S)
        reach[term] = 1.0
        for _ in range(max_iter):
            new = reach.copy()
            for s in pages:
                new[s] = P[s, policy[s]] @ reach
            if np.max(np.abs(new - reach)) < 1e-12:
                reach = new
                break
            reach = new
        out[term] = float(reach[env.start_state])
    return out

