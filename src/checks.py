"""Persist real-data and model checks as reproducible pipeline artifacts."""

import json
import math
from pathlib import Path

import torch

from checkpoint import write_json
from config import Config
from course_data import load_course_data
from data_contract import inspect_batch, prepare_batch
from loss import next_token_loss
from model import TinyGPT
from runtime import seed_all, to_device


def check_data(config: Config, destination: Path) -> dict:
    data = load_course_data(config)
    manifest = json.loads((config.data_dir / "manifest.json").read_text())
    expected = {"dataset": config.dataset_name, "seed": config.seed,
                "train_stories": config.train_stories, "validation_stories": config.validation_stories,
                "requested_vocab_size": config.tokenizer_vocab_size}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Prepared data setting differs from config: {key}")
    report = {"passed": True, "source": data.source, "vocab_size": data.vocab_size,
              "pad_id": data.pad_id, "eos_id": data.eos_id, "bos_id": data.bos_id,
              "data_sha256": manifest["sha256"], "config": config.to_dict()}
    for split, loader in (("train", data.train_loader), ("validation", data.validation_loader)):
        try:
            report[split] = inspect_batch(next(iter(loader)), data, config)
        except StopIteration as exc:
            raise ValueError(f"Empty {split} split") from exc
    report["text_review"] = "Inspect decoded_input_row in each split for readable text"
    write_json(destination, report)
    return report


@torch.no_grad()
def check_model(config: Config, device: torch.device, destination: Path) -> dict:
    seed_all(config.seed)
    data = load_course_data(config)
    batch = to_device(prepare_batch(next(iter(data.train_loader)), data, config), device)
    model = TinyGPT(data.vocab_size, config).to(device).eval()
    logits = model(batch.inputs, batch.input_mask)
    initial_loss = next_token_loss(logits, batch.targets).item()
    count = sum(p.numel() for p in model.parameters())
    d, r = config.d_model, config.mlp_ratio
    expected_count = data.vocab_size * d + config.context_length * d + 2 * d + config.n_layers * (
        (4 + 2 * r) * d * d + (9 + r) * d
    )
    probe = batch.inputs[:2, :min(16, batch.inputs.shape[1])]
    cut = probe.shape[1] // 2
    changed = probe.clone()
    changed[:, cut:] = (changed[:, cut:] + 1) % data.vocab_size
    causal = torch.allclose(model(probe)[:, :cut], model(changed)[:, :cut], atol=1e-6, rtol=1e-5)
    checks = {
        "logits_shape": tuple(logits.shape) == (*batch.inputs.shape, data.vocab_size),
        "finite_logits": bool(torch.isfinite(logits).all()),
        "tied_head": model.lm_head.weight is model.token_embedding.weight,
        "unique_embedding_parameter": sum(p is model.token_embedding.weight for p in model.parameters()) == 1,
        "parameter_count": count == expected_count,
        "causal_independence": causal,
        "initial_loss_near_uniform": abs(initial_loss - math.log(data.vocab_size)) < 0.3,
    }
    report = {"passed": all(checks.values()), "checks": checks, "parameters": count,
              "logits_shape": list(logits.shape), "initial_loss": initial_loss,
              "ln_vocab": math.log(data.vocab_size), "device": str(device), "dtype": str(logits.dtype)}
    write_json(destination, report)
    if not report["passed"]:
        raise ValueError(f"Model checks failed: {checks}")
    return report
