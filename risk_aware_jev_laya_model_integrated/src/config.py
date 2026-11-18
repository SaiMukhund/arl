from dataclasses import dataclass

@dataclass
class Config:
    horizon: int = 4
    cost_scale: float = 100.0
    alpha: float = 0.05
    theta: float = 2.0
    lambda_cost: float = 0.25
    beta_cvar: float = 0.45
    gamma_entropic: float = 0.35
    safety_epsilon: float = 0.025
    risk_hidden: int = 128
    risk_epochs: int = 18
    risk_batch_size: int = 256
    dqn_batch_size: int = 128
    dqn_replay_size: int = 50000
    dqn_warmup: int = 500
    dqn_target_update: int = 250
    dqn_lr: float = 1e-3
    dqn_gamma: float = 0.97
    dqn_epsilon_decay_steps: int = 12000
    seed: int = 7
