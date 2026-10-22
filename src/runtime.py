"""Small device and reproducibility helpers shared by experiment commands."""

from contextlib import nullcontext
import random

import numpy as np
import torch

from data_contract import PreparedBatch


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(name: str = "auto") -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    return device


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


class GPUMemoryTracker:
    """Track allocator peaks, with explicit sampling fallback for older MPS builds."""

    def __init__(self, device: torch.device):
        self.device = device
        self.sampled_peak = 0
        self.allocator = None
        self.measurement = "not applicable"
        sync(device)
        if device.type == "cuda":
            self.allocator = torch.cuda
            self.measurement = "CUDA allocator peak"
        elif device.type == "mps":
            accelerator = getattr(torch, "accelerator", None)
            if accelerator is not None and hasattr(accelerator, "memory_stats"):
                try:
                    stats = accelerator.memory_stats(device)
                    if "allocated_bytes.all.peak" in stats:
                        accelerator.reset_peak_memory_stats(device)
                        self.allocator = accelerator
                except (RuntimeError, NotImplementedError):
                    # Older versions expose generic APIs without MPS support.
                    pass
            self.measurement = "MPS allocator peak" if self.allocator else "sampled MPS allocated bytes (lower bound)"
        if device.type == "cuda":
            self.allocator.reset_peak_memory_stats(device)
        self.sample()

    def sample(self) -> None:
        if self.device.type == "mps" and self.allocator is None:
            self.sampled_peak = max(self.sampled_peak, torch.mps.current_allocated_memory())

    def peak_bytes(self) -> int | None:
        if self.allocator is not None:
            return self.allocator.max_memory_allocated(self.device)
        return self.sampled_peak if self.device.type == "mps" else None


def autocast(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def to_device(batch: PreparedBatch, device: torch.device) -> PreparedBatch:
    return PreparedBatch(*(tensor.to(device) for tensor in (
        batch.inputs, batch.targets, batch.input_mask, batch.target_mask,
    )))
