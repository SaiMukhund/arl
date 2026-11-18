from dataclasses import dataclass

ACTIONS = ("direct_transfer", "verify_recipient", "request_confirmation")

@dataclass
class Config:
    seed: int = 7
    episodes: int = 30000
    rollouts_per_action: int = 1
    cvar_alpha: float = 0.05
    lambda_cost: float = 0.35
    beta_cvar: float = 0.45
    cost_scale: float = 1000.0
    catastrophic_threshold: float = 100.0
    max_catastrophic_probability: float = 0.01
    test_size: float = 0.25
    validation_size: float = 0.15
    use_laya: bool = True
