"""Value-network backbones for risk-sensitive fitted value iteration.

A backbone maps a batch of environment states to action-values ``Q[B, A]``.
Two interchangeable backbones are provided:

* ``MLPValueNet``  -- a tiny MLP over the env's numeric state features. Runs on
  CPU/GPU with no downloads; used to validate the learning pipeline against the
  exact value-iteration ground truth.
* ``YevValueNet``  -- (see ``yev_backbone.py``) wraps ``choyiny/yev0-4b`` as a
  frozen/LoRA feature extractor with a small value head, consuming the env's
  Jev-style text prompts.

Both share the same forward contract so the trainer in ``train_fittedq.py`` is
backbone-agnostic.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .env import RiskyCheckout


class ValueBackbone(nn.Module):
    """Interface: forward(states: list[int]) -> Q tensor [B, n_actions]."""

    n_actions: int

    def forward(self, states):  # pragma: no cover - interface
        raise NotImplementedError


class MLPValueNet(ValueBackbone):
    """Small MLP over numeric stage features."""

    def __init__(self, env: RiskyCheckout, hidden: int = 128):
        super().__init__()
        self.env = env
        self.n_actions = env.n_actions
        in_dim = env.stage_features(env.start_state).shape[0]
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, env.n_actions),
        )
        # Precompute feature matrix for all states.
        feats = np.stack([env.stage_features(s) for s in range(env.n_states)])
        self.register_buffer("state_feats", torch.tensor(feats, dtype=torch.float32))

    def forward(self, states):
        idx = torch.as_tensor(states, dtype=torch.long, device=self.state_feats.device)
        x = self.state_feats[idx]
        return self.net(x)

    @property
    def device(self):
        return self.state_feats.device


class FeatureMatrixValueNet(ValueBackbone):
    """Trainable head over a *precomputed, frozen* feature matrix [n_states, D].

    This is how the yev0-4b backbone is used efficiently: the 4B base is frozen,
    so each state's hidden-state feature is constant. We extract features once
    (one forward per state), cache them here, and fitted value iteration then
    trains only the small head -- fast, even with many states.
    """

    def __init__(self, feature_matrix, n_actions: int, hidden: int = 256):
        super().__init__()
        feats = torch.as_tensor(np.asarray(feature_matrix), dtype=torch.float32)
        self.register_buffer("state_feats", feats)
        self.n_actions = n_actions
        in_dim = feats.shape[1]
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, states):
        idx = torch.as_tensor(states, dtype=torch.long, device=self.state_feats.device)
        return self.net(self.state_feats[idx])

    @property
    def device(self):
        return self.state_feats.device
