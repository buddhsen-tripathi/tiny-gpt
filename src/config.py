"""Experiment settings fixed before inspecting any generated stories."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class Config:
    seed: int = 2026
    data_dir: Path = Path("data")
    output_dir: Path = Path("results")
    dataset_name: str = "roneneldan/TinyStories"
    train_stories: int = 100_000
    validation_stories: int = 2_000
    tokenizer_vocab_size: int = 8_192
    tokenizer_min_frequency: int = 2
    validation_targets: int = 32_768
    n_layers: int = 6
    d_model: int = 384
    n_heads: int = 6
    context_length: int = 256
    mlp_ratio: int = 4
    dropout: float = 0.0
    init_std: float = 0.02
    target_tokens: int = 25_000_000
    warmup_tokens: int = 1_000_000
    eval_every_tokens: int = 2_500_000
    learning_rate: float = 6e-4
    min_learning_rate: float = 6e-5
    weight_decay: float = 0.1
    adam_betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    effective_batch_size: int = 32
    microbatch_size: int = 8
    cuda_precision: str = "bfloat16"
    mps_precision: str = "float32"
    cpu_precision: str = "float32"
    overfit_batch_size: int = 8
    overfit_contexts: tuple[int, int] = (64, 256)
    overfit_learning_rate: float = 1e-3
    overfit_max_steps: int = 1_000
    overfit_loss_threshold: float = 0.05
    overfit_accuracy_threshold: float = 0.99
    temperature: float = 0.8
    top_k: int = 40
    max_new_tokens: int = 200
    early_stop: bool = False
    prompts: tuple[str, str, str] = (
        "Once upon a time, there was a little girl named Lily.",
        "Tom and his dog walked to the park.",
        "One day, a boy found a shiny red ball.",
    )

    def __post_init__(self) -> None:
        for name in (
            "train_stories", "validation_stories", "tokenizer_vocab_size",
            "tokenizer_min_frequency", "validation_targets",
            "n_layers", "d_model", "n_heads", "context_length", "mlp_ratio",
            "target_tokens", "warmup_tokens", "eval_every_tokens",
            "effective_batch_size", "microbatch_size", "overfit_batch_size",
            "overfit_max_steps", "top_k", "max_new_tokens",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.effective_batch_size % self.microbatch_size:
            raise ValueError("microbatch_size must divide effective_batch_size")
        if self.warmup_tokens >= self.target_tokens:
            raise ValueError("warmup_tokens must be below target_tokens")
        if not 0 < self.min_learning_rate <= self.learning_rate:
            raise ValueError("Require 0 < min_learning_rate <= learning_rate")
        if not 0 <= self.dropout < 1 or self.weight_decay < 0:
            raise ValueError("Invalid dropout or weight decay")
        if self.temperature <= 0 or self.grad_clip <= 0 or self.init_std <= 0:
            raise ValueError("Temperature, gradient clip and init std must be positive")
        if not all(0 <= beta < 1 for beta in self.adam_betas):
            raise ValueError("Adam betas must be in [0, 1)")

    @property
    def accumulation_steps(self) -> int:
        return self.effective_batch_size // self.microbatch_size

    def to_dict(self) -> dict:
        values = asdict(self)
        values["data_dir"] = str(self.data_dir)
        values["output_dir"] = str(self.output_dir)
        return values

    @classmethod
    def from_dict(cls, values: dict) -> "Config":
        values = dict(values)
        for name in ("data_dir", "output_dir"):
            values[name] = Path(values[name])
        for name in ("adam_betas", "overfit_contexts", "prompts"):
            values[name] = tuple(values[name])
        return cls(**values)


def load_config(path: Path | None = None, **overrides) -> Config:
    values = Config().to_dict()
    if path is not None:
        supplied = json.loads(path.read_text())
        unknown = supplied.keys() - values.keys()
        if unknown:
            raise ValueError(f"Unknown config fields: {', '.join(sorted(unknown))}")
        values.update(supplied)
    values.update({name: value for name, value in overrides.items() if value is not None})
    return Config.from_dict(values)
