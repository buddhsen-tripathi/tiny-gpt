"""Loss consumes already-shifted targets from data_contract.prepare_batch."""

import torch
from torch import Tensor
from torch.nn import functional as F

from data_contract import IGNORE_INDEX


def next_token_loss(logits: Tensor, targets: Tensor, reduction: str = "mean") -> Tensor:
    if logits.shape[:-1] != targets.shape:
        raise ValueError("Logit and target positions must match")
    if reduction not in ("mean", "sum"):
        raise ValueError("Use mean or sum reduction")
    if not (targets != IGNORE_INDEX).any():
        raise ValueError("Loss requires at least one valid target")
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
        ignore_index=IGNORE_INDEX, reduction=reduction,
    )


@torch.no_grad()
def token_accuracy(logits: Tensor, targets: Tensor) -> float:
    valid = targets != IGNORE_INDEX
    return float((logits.argmax(-1)[valid] == targets[valid]).float().mean().item())
