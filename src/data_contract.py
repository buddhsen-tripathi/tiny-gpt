"""Validate the course loader without replacing its tokenizer or splits.

Adapters return CourseData from a factory named with --adapter module:function.
The only shifting operation is prepare_batch; already-shifted pairs stay shifted.
"""

import argparse
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import random
import sys
from typing import Iterable, Protocol

import torch
from torch import Tensor

from config import Config

IGNORE_INDEX = -100


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, token_ids: list[int]) -> str: ...


@dataclass
class TokenWindow:
    """Unshifted integer tokens [B, T+1]; mask is True for real tokens."""

    tokens: Tensor
    valid_mask: Tensor | None = None


@dataclass
class NextTokenBatch:
    """Already-shifted [B, T] pairs; explicit masks distinguish EOS from padding."""

    inputs: Tensor
    targets: Tensor
    input_mask: Tensor | None = None
    target_mask: Tensor | None = None


@dataclass
class PreparedBatch:
    inputs: Tensor
    targets: Tensor
    input_mask: Tensor
    target_mask: Tensor

    @property
    def target_tokens(self) -> int:
        return int(self.target_mask.sum().item())


@dataclass
class CourseData:
    tokenizer: Tokenizer
    vocab_size: int  # Full vocabulary size, including any added special tokens.
    train_loader: Iterable[TokenWindow | NextTokenBatch]
    validation_loader: Iterable[TokenWindow | NextTokenBatch]
    source: str  # Identify the course files/version, not just "TinyStories".
    pad_id: int | None = None
    bos_id: int | None = None
    eos_id: int | None = None

    def __post_init__(self) -> None:
        if self.vocab_size <= 0 or not self.source.strip():
            raise ValueError("Positive vocab_size and a course source description are required")
        for token_id in (self.pad_id, self.bos_id, self.eos_id):
            if token_id is not None and not 0 <= token_id < self.vocab_size:
                raise ValueError("Special token ids must be inside the vocabulary")


def _check_tokens(tokens: Tensor, name: str) -> None:
    if not isinstance(tokens, Tensor) or tokens.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 tensor [B, T]")
    if tokens.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must use int32 or int64 token ids")
    if not all(tokens.shape):
        raise ValueError(f"{name} cannot be empty")


def _mask(tokens: Tensor, explicit: Tensor | None, data: CourseData) -> Tensor:
    if explicit is not None:
        if explicit.shape != tokens.shape or explicit.device != tokens.device:
            raise ValueError("Mask shape and device must match tokens")
        if explicit.dtype != torch.bool:
            raise ValueError("Masks must have boolean dtype")
        mask = explicit
    elif data.pad_id is None:
        mask = torch.ones_like(tokens, dtype=torch.bool)
    else:
        if data.pad_id in (data.bos_id, data.eos_id):
            raise ValueError("Padding shares a special token id; supply explicit masks")
        mask = tokens != data.pad_id
    if (mask[:, 1:] & ~mask[:, :-1]).any():
        raise ValueError("Use right padding with a contiguous valid prefix")
    return mask


def prepare_batch(
    batch: TokenWindow | NextTokenBatch, data: CourseData, config: Config
) -> PreparedBatch:
    if isinstance(batch, TokenWindow):
        _check_tokens(batch.tokens, "tokens")
        if batch.tokens.shape[1] < 2:
            raise ValueError("A token window needs at least two tokens")
        valid = _mask(batch.tokens, batch.valid_mask, data)
        inputs, targets = batch.tokens[:, :-1], batch.tokens[:, 1:]
        input_mask, target_mask = valid[:, :-1], valid[:, 1:]
    elif isinstance(batch, NextTokenBatch):
        inputs, targets = batch.inputs, batch.targets
        _check_tokens(inputs, "inputs")
        _check_tokens(targets, "targets")
        if inputs.shape != targets.shape or inputs.device != targets.device:
            raise ValueError("Inputs and targets must have matching shapes and devices")
        input_mask = _mask(inputs, batch.input_mask, data)
        target_mask = _mask(targets, batch.target_mask, data)
    else:
        raise TypeError("Wrap the course batch in TokenWindow or NextTokenBatch")
    if inputs.shape[1] > config.context_length:
        raise ValueError(f"Context exceeds {config.context_length} targets")
    if not target_mask.any(dim=1).all():
        raise ValueError("Each row must contain at least one valid target")
    if (target_mask & ~input_mask).any():
        raise ValueError("A valid target cannot follow a padded input")
    if not torch.equal(input_mask[:, 1:], target_mask[:, :-1]):
        raise ValueError("Input and target masks do not satisfy the next-token shift")
    overlap = input_mask[:, 1:]
    if not torch.equal(inputs[:, 1:][overlap], targets[:, :-1][overlap]):
        raise ValueError("Next-token shift failed: x[:, 1:] must equal y[:, :-1]")
    for tokens, mask in ((inputs, input_mask), (targets, target_mask)):
        ids = tokens[mask]
        if (ids < 0).any() or (ids >= data.vocab_size).any():
            raise ValueError("Valid token ids must be inside the tokenizer vocabulary")
    # Keep actual EOS targets even if EOS is also the padding id.
    # Normalize masked inputs to a safe embedding index; never mutate the loader.
    return PreparedBatch(
        inputs.long().masked_fill(~input_mask, 0),
        targets.long().masked_fill(~target_mask, IGNORE_INDEX),
        input_mask,
        target_mask,
    )


def inspect_batch(batch, data: CourseData, config: Config) -> dict:
    prepared = prepare_batch(batch, data, config)
    row = prepared.inputs[0][prepared.input_mask[0]].tolist()
    source_tensor = batch.tokens if isinstance(batch, TokenWindow) else batch.inputs
    valid_ids = torch.cat((
        prepared.inputs[prepared.input_mask],
        prepared.targets[prepared.target_mask],
    ))
    return {
        "source_shape": list(source_tensor.shape),
        "source_dtype": str(source_tensor.dtype),
        "input_shape": list(prepared.inputs.shape),
        "target_shape": list(prepared.targets.shape),
        "device": str(prepared.inputs.device),
        "dtype": str(prepared.inputs.dtype),
        "min_token_id": int(valid_ids.min().item()),
        "max_token_id": int(valid_ids.max().item()),
        "target_tokens": prepared.target_tokens,
        "ignored_targets": int((~prepared.target_mask).sum().item()),
        "decoded_input_row": data.tokenizer.decode(row),
        "shift_check": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, help="Course factory as module:function")
    parser.add_argument("--data-dir", type=Path, default=Config().data_dir)
    parser.add_argument("--output-dir", type=Path, default=Config().output_dir)
    args = parser.parse_args()
    config = Config(data_dir=args.data_dir, output_dir=args.output_dir)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    try:
        module_name, function_name = args.adapter.split(":", 1)
        factory = getattr(importlib.import_module(module_name), function_name)
        data = factory(config)
        if not isinstance(data, CourseData):
            raise TypeError("The adapter must return CourseData")
        report = {
            "course_source": data.source,
            "adapter": args.adapter,
            "vocab_size": data.vocab_size,
            "pad_id": data.pad_id,
            "bos_id": data.bos_id,
            "eos_id": data.eos_id,
            "config": config.to_dict(),
            "command": [sys.executable, *sys.argv],
            "torch_version": torch.__version__,
        }
        for split, loader in (("train", data.train_loader), ("validation", data.validation_loader)):
            try:
                batch = next(iter(loader))
            except StopIteration as exc:
                raise ValueError(f"The {split} loader is empty") from exc
            report[split] = inspect_batch(batch, data, config)
        report["numeric_checks"] = "passed"
        report["readable_text_review"] = "pending human inspection of decoded rows"
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        parser.exit(1, f"Data contract failed: {exc}\n")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    output = config.output_dir / "data_contract.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    # Adapters import these dataclasses by module name. Run the canonical module
    # so direct script execution doesn't compare against __main__ class copies.
    from data_contract import main as run_cli

    run_cli()
