"""Verify saved artifacts and reproduce the best checkpoint's validation loss."""

from dataclasses import replace
import json
import math
from pathlib import Path

import torch

from checkpoint import load_checkpoint, write_json
from config import Config
from course_data import load_course_data
from prepare_data import sha256_file
from train import evaluate, verify_overfit_gate


def verify_run(run_dir: Path, device: torch.device, destination: Path,
               data_dir: Path | None = None) -> dict:
    results = json.loads((run_dir / "results.json").read_text())
    if results["status"] != "complete":
        raise ValueError("Training is incomplete; reporting cannot mark it complete")
    stored = Config.from_dict(results["config"])
    config = replace(stored, data_dir=data_dir or stored.data_dir, output_dir=run_dir)
    tokenizer_hash = sha256_file(run_dir / "metadata/tokenizer.json")
    if tokenizer_hash != results["tokenizer_sha256"]:
        raise ValueError("Run tokenizer hash differs from training results")
    manifest = json.loads((run_dir / "metadata/data_manifest.json").read_text())
    data_manifest = json.loads((config.data_dir / "manifest.json").read_text())
    if manifest["sha256"] != data_manifest["sha256"]:
        raise ValueError("Evaluation data differs from the training run")
    verify_overfit_gate(run_dir / "checks/overfit.json", config, manifest)
    data = load_course_data(config)
    if not math.isclose(math.exp(results["best_validation_loss"]), results["best_perplexity"], rel_tol=1e-10):
        raise ValueError("Reported perplexity differs from exp(validation loss)")
    history = results["validation_history"]
    best = min(history, key=lambda row: row["loss"])
    if best["tokens_seen"] != results["best_tokens_seen"] or best["loss"] != results["best_validation_loss"]:
        raise ValueError("Best checkpoint selection differs from validation history")
    train_history = results["training_history"]
    if (len(train_history) != results["optimizer_steps"] or not train_history or
            train_history[-1]["tokens_seen"] != results["tokens_seen"] or
            any(a["tokens_seen"] >= b["tokens_seen"] for a, b in zip(train_history, train_history[1:]))):
        raise ValueError("Training history does not match final token/step counters")
    if results["tokens_seen"] < config.target_tokens or results["tokens_seen"] - config.target_tokens >= config.effective_batch_size * config.context_length:
        raise ValueError("Processed token count is outside the configured budget tolerance")
    expected_tokens = {"initial": 0, "final": results["tokens_seen"], "best": results["best_tokens_seen"]}
    report = {"passed": True, "device": str(device), "tokenizer_sha256": tokenizer_hash,
              "source_results_sha256": sha256_file(run_dir / "results.json"), "checkpoints": {}}
    for stage in ("initial", "halfway", "final", "best"):
        model, metadata = load_checkpoint(run_dir / f"checkpoints/{stage}.pt", device, tokenizer_hash)
        if metadata["tokens_seen"] != results["checkpoints"][stage]:
            raise ValueError(f"Checkpoint token mismatch: {stage}")
        if stage in expected_tokens and metadata["tokens_seen"] != expected_tokens[stage]:
            raise ValueError(f"Checkpoint stage mismatch: {stage}")
        if stage == "halfway" and not config.target_tokens / 2 <= metadata["tokens_seen"] < config.target_tokens / 2 + config.effective_batch_size * config.context_length:
            raise ValueError("Halfway checkpoint is outside one update of the midpoint")
        if Config.from_dict(metadata["config"]).to_dict() != stored.to_dict():
            raise ValueError(f"Checkpoint config mismatch: {stage}")
        if model.vocab_size != data.vocab_size or not all(torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError(f"Invalid checkpoint parameters: {stage}")
        report["checkpoints"][stage] = {"tokens_seen": metadata["tokens_seen"],
                                           "sha256": sha256_file(run_dir / f"checkpoints/{stage}.pt")}
        if stage == "best":
            loss, count = evaluate(model, data, config, device)
            # Reloading on another backend may introduce small floating-point differences.
            if not math.isclose(loss, results["best_validation_loss"], rel_tol=1e-5, abs_tol=1e-5):
                raise ValueError("Reloaded best validation loss differs from recorded loss")
            if count != results["validation_target_count"]:
                raise ValueError("Evaluation target count changed")
            report.update(reloaded_validation_loss=loss, validation_targets=count)
        del model
    write_json(destination, report)
    return report
