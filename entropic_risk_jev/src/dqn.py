"""Model-free value learning on observations: DQN, entropic DQN, QR-DQN+CVaR.

All learners interact with the environment only through ``reset`` / ``step`` and
observe ``obs_features`` (partial observability: the observation does not reveal
the hidden page severity). They therefore learn from sampled experience, not the
known model.

Modes
-----
* ``neutral``  -- standard DQN. Learns Q(s,a)=E[return]; policy = argmax. This is
  the "without entropic risk" learned baseline.
* ``entropic`` -- exponential-utility DQN. Learns U(s,a)=E[exp(-beta*return)] via
  the multiplicative Bellman  U(s,a)=E[exp(-beta r) * U(s',a'*)], a'*=argmin U.
  Policy = argmin U  <=>  maximise the entropic value -(1/beta) log U.
* ``qr-cvar`` -- quantile-regression DQN (QR-DQN). Learns the return distribution
  as N quantiles; both bootstrap and policy use the CVaR_alpha of the quantiles,
  a model-free tail-risk learner.

A backbone maps an observation feature vector to per-action heads. Two are
supported: a plain MLP, and a fixed matrix of precomputed yev0-4b features
(``obs_matrix``) so the same algorithms can run on yev's representation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------- nets
class QNet(nn.Module):
    def __init__(self, in_dim, n_actions, hidden=128, n_quantiles=1, positive=False):
        super().__init__()
        self.n_actions = n_actions
        self.n_quantiles = n_quantiles
        self.positive = positive
        self.body = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head = nn.Linear(hidden, n_actions * n_quantiles)

    def forward(self, x):
        h = self.head(self.body(x))
        out = h.view(-1, self.n_actions, self.n_quantiles)
        if self.positive:                      # U(s,a) = exp(raw), clamped
            out = torch.exp(out.clamp(-40, 40))
        return out                              # (B, A, Q)


class ReplayBuffer:
    """Preallocated numpy ring buffer (fast sampling)."""

    def __init__(self, cap: int, feat_dim: int):
        self.cap = cap
        self.s = np.zeros((cap, feat_dim), dtype=np.float32)
        self.ns = np.zeros((cap, feat_dim), dtype=np.float32)
        self.a = np.zeros(cap, dtype=np.int64)
        self.r = np.zeros(cap, dtype=np.float32)
        self.d = np.zeros(cap, dtype=np.float32)
        self.n = 0
        self.ptr = 0

    def add(self, s, a, r, ns, d):
        i = self.ptr
        self.s[i] = s; self.a[i] = a; self.r[i] = r; self.ns[i] = ns; self.d[i] = d
        self.ptr = (self.ptr + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def __len__(self):
        return self.n

    def sample(self, n, device):
        idx = np.random.randint(0, self.n, size=n)
        t = lambda arr, dt: torch.as_tensor(arr[idx], dtype=dt, device=device)
        return (t(self.s, torch.float32), t(self.a, torch.long), t(self.r, torch.float32),
                t(self.ns, torch.float32), t(self.d, torch.float32))


# --------------------------------------------------------------------- utils
def _cvar_of_quantiles(q: torch.Tensor, alpha: float) -> torch.Tensor:
    """CVaR_alpha from sorted quantile estimates q (..., Nq). Higher=better.

    With quantiles at evenly spaced tau, CVaR_alpha ~= mean of the lowest
    ceil(alpha*Nq) quantiles.
    """
    nq = q.shape[-1]
    k = max(1, int(np.ceil(alpha * nq)))
    low, _ = torch.sort(q, dim=-1)
    return low[..., :k].mean(dim=-1)


class ObsMatrix:
    """Maps a state id -> fixed feature vector (MLP features or yev features)."""

    def __init__(self, matrix: np.ndarray):
        self.M = np.asarray(matrix, dtype=np.float32)
        self.dim = self.M.shape[1]

    def __call__(self, state: int) -> np.ndarray:
        return self.M[state]


def mlp_obs_matrix(env) -> ObsMatrix:
    return ObsMatrix(np.stack([env.obs_features(s) for s in range(env.n_states)]))


# --------------------------------------------------------------------- train
def train_value_dqn(
    env,
    obs: ObsMatrix,
    mode: str = "neutral",
    beta: float = 0.1,
    alpha: float = 0.1,
    n_quantiles: int = 21,
    episodes: int = 4000,
    gamma: float = 1.0,
    lr: float = 1e-3,
    batch_size: int = 128,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    target_sync: int = 200,
    device: str = "cpu",
    seed: int = 0,
    warmup: int = 500,
    reward_scale: float = 0.1,
    updates_per_step: int = 1,
):
    """Train a model-free value learner; return a greedy policy over all states.

    Rewards are internally scaled by ``reward_scale`` (helps NN conditioning); the
    optimal policy is unchanged (monotone). For the entropic mode beta is divided
    by the same factor so the objective exp(-beta * raw_reward) is preserved.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    nq = n_quantiles if mode == "qr-cvar" else 1
    positive = mode == "entropic"
    beta_eff = beta / reward_scale        # keeps exp(-beta * raw_reward)
    net = QNet(obs.dim, env.n_actions, n_quantiles=nq, positive=positive).to(device)
    tgt = QNet(obs.dim, env.n_actions, n_quantiles=nq, positive=positive).to(device)
    tgt.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    buf = ReplayBuffer(100_000, obs.dim)

    def action_scores(q_net, feats):
        """Per-action scalar score used for greedy selection (higher=better)."""
        out = q_net(feats)                       # (B, A, Q)
        if mode == "neutral":
            return out[..., 0]                   # Q
        if mode == "entropic":
            return -out[..., 0]                  # -U (argmax(-U)=argmin U)
        return _cvar_of_quantiles(out, alpha)    # CVaR of quantiles

    steps = 0
    for ep in range(episodes):
        s = env.reset()
        eps = eps_end + (eps_start - eps_end) * max(0, 1 - ep / (0.7 * episodes))
        for _ in range(env.n_layers + 2):
            feat = obs(s)
            if rng.random() < eps:
                a = int(rng.integers(env.n_actions))
            else:
                with torch.no_grad():
                    sc = action_scores(net, torch.tensor(feat[None], device=device))
                a = int(sc.argmax(dim=1).item())
            ns, r, done = env.step(s, a, rng)
            buf.add(feat, a, r * reward_scale, obs(ns), float(done))
            s = ns
            steps += 1
            if len(buf) >= warmup:
                for _ in range(updates_per_step):
                    _update(net, tgt, opt, buf, mode, beta_eff, alpha, nq, gamma,
                            batch_size, device)
                if steps % target_sync == 0:
                    tgt.load_state_dict(net.state_dict())
            if done:
                break

    # Greedy policy over all states.
    policy = np.zeros(env.n_states, dtype=int)
    feats_all = torch.tensor(obs.M, device=device)
    with torch.no_grad():
        sc = action_scores(net, feats_all)       # (S, A)
    policy = sc.argmax(dim=1).cpu().numpy().astype(int)
    return policy, net


def _update(net, tgt, opt, buf, mode, beta, alpha, nq, gamma, batch_size, device):
    s, a, r, ns, d = buf.sample(batch_size, device)
    B = s.shape[0]

    if mode == "neutral":
        q = net(s)[:, :, 0].gather(1, a[:, None]).squeeze(1)
        with torch.no_grad():
            q_next = tgt(ns)[:, :, 0].max(dim=1).values
            target = r + gamma * (1 - d) * q_next
        loss = F.smooth_l1_loss(q, target)

    elif mode == "entropic":
        u = net(s)[:, :, 0].gather(1, a[:, None]).squeeze(1)       # U(s,a)>0
        with torch.no_grad():
            u_next = tgt(ns)[:, :, 0].min(dim=1).values            # min over a'
            # U = exp(-beta r) * (1 if done else U'); gamma folded as 1 (episodic)
            target = torch.exp((-beta * r).clamp(-40, 40)) * (d + (1 - d) * u_next)
        # MSE in U-space: its fixed point is the conditional mean E[exp(-beta r)U'],
        # i.e. the true exponential-utility value (a log-space loss would converge
        # to the geometric mean instead). Normalise by target scale for stability.
        scale = target.detach().abs().mean().clamp_min(1e-6)
        loss = F.mse_loss(u / scale, target / scale)

    else:  # qr-cvar
        tau = (torch.arange(nq, device=device, dtype=torch.float32) + 0.5) / nq
        q = net(s).gather(1, a[:, None, None].expand(B, 1, nq)).squeeze(1)  # (B,Nq)
        with torch.no_grad():
            out_next = tgt(ns)                                      # (B,A,Nq)
            cvar_next = _cvar_of_quantiles(out_next, alpha)         # (B,A)
            a_next = cvar_next.argmax(dim=1)                        # risk-greedy
            q_next = out_next.gather(1, a_next[:, None, None].expand(B, 1, nq)).squeeze(1)
            target = r[:, None] + gamma * (1 - d)[:, None] * q_next  # (B,Nq)
        # quantile Huber loss (q: predicted quantiles, target: target quantiles)
        q_exp = q[:, :, None].expand(B, nq, nq)                    # (B,Nq_pred,Nq_tgt)
        t_exp = target[:, None, :].expand(B, nq, nq)
        diff = t_exp - q_exp
        huber = F.smooth_l1_loss(q_exp, t_exp, reduction="none")
        rho = (tau[None, :, None] - (diff.detach() < 0).float()).abs() * huber
        loss = rho.sum(dim=2).mean()                               # sum over targets

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
    opt.step()
