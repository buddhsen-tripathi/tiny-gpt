"""Atomic checkpoint and JSON writes; checkpoints are for evaluation, not resume."""

import json
from pathlib import Path

import torch

from config import Config
from model import TinyGPT


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_checkpoint(path: Path, model: TinyGPT, tokens_seen: int, tokenizer_hash: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({
        "model": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        "config": model.config.to_dict(), "vocab_size": model.vocab_size,
        "tokens_seen": tokens_seen, "tokenizer_sha256": tokenizer_hash,
    }, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path, device: torch.device, tokenizer_hash: str) -> tuple[TinyGPT, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["tokenizer_sha256"] != tokenizer_hash:
        raise ValueError("Checkpoint and tokenizer hashes do not match")
    model = TinyGPT(checkpoint["vocab_size"], Config.from_dict(checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint
