"""Memorize a frozen real batch at contexts 64 and 256 before budgeted training."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import torch

from config import Config
from course_data import load_course_data
from data_contract import NextTokenBatch, prepare_batch
from loss import next_token_loss, token_accuracy
from model import TinyGPT
from runtime import seed_all, select_device, sync, to_device


def run_overfit(config: Config, device: torch.device) -> dict:
    config = replace(config, microbatch_size=config.overfit_batch_size, dropout=0.0)
    data = load_course_data(config)
    source = next(iter(data.train_loader))
    # Both contexts use prefixes of the same frozen sequences.
    frozen_inputs, frozen_targets = source.inputs.clone(), source.targets.clone()
    manifest = json.loads((config.data_dir / "manifest.json").read_text())
    report = {
        "passed": False, "device": str(device), "dtype": "torch.float32",
        "config": config.to_dict(), "data_sha256": manifest["sha256"], "contexts": [],
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    output = config.output_dir / "checks/overfit.json"
    for context in config.overfit_contexts:
        seed_all(config.seed)
        model = TinyGPT(data.vocab_size, config).to(device)
        batch = to_device(prepare_batch(NextTokenBatch(
            frozen_inputs[:, :context], frozen_targets[:, :context],
        ), data, config), device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.overfit_learning_rate,
                                      betas=config.adam_betas, weight_decay=0.0)
        with torch.no_grad():
            initial_loss = next_token_loss(model(batch.inputs, batch.input_mask), batch.targets).item()
        sync(device)
        started = time.perf_counter()
        history = []
        passed = False
        for step in range(1, config.overfit_max_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            loss = next_token_loss(model(batch.inputs, batch.input_mask), batch.targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite overfit loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            if step % 25 == 0 or step == config.overfit_max_steps:
                with torch.no_grad():
                    logits = model(batch.inputs, batch.input_mask)
                    measured_loss = next_token_loss(logits, batch.targets).item()
                    accuracy = token_accuracy(logits, batch.targets)
                history.append({"step": step, "loss": measured_loss, "accuracy": accuracy})
                print(f"context={context} step={step} loss={measured_loss:.5f} accuracy={accuracy:.4%}", flush=True)
                passed = measured_loss < config.overfit_loss_threshold and accuracy > config.overfit_accuracy_threshold
                if passed:
                    break
        sync(device)
        report["contexts"].append({
            "context": context, "initial_loss": initial_loss, "passed": passed,
            "seconds": time.perf_counter() - started, "history": history, **history[-1],
        })
        report["passed"] = len(report["contexts"]) == len(config.overfit_contexts) and all(
            entry["passed"] for entry in report["contexts"]
        )
        output.write_text(json.dumps(report, indent=2) + "\n")
        del model, optimizer, loss, logits
        if device.type == "mps":
            torch.mps.empty_cache()
        if not passed:
            break
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Config().data_dir)
    parser.add_argument("--output-dir", type=Path, default=Config().output_dir)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = parser.parse_args()
    report = run_overfit(Config(data_dir=args.data_dir, output_dir=args.output_dir), select_device(args.device))
    print(f"Overfit gate: {'PASSED' if report['passed'] else 'FAILED'}", flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
