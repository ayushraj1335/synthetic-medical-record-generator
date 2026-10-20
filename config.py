from dataclasses import dataclass

@dataclass
class DiffusionConfig:
    steps: int = 100
    hidden_dim: int = 256
    epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 1e-3
    seed: int = 42

@dataclass
class PrivacyConfig:
    enabled: bool = False
    epsilon: float = 8.0
    delta: float = 1e-5
    max_grad_norm: float = 1.0
    noise_multiplier: float = 1.2
