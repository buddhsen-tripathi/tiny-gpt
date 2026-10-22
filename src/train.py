"""Token-budgeted TinyGPT training after a successful real-data overfit gate."""

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import shutil
import time

import torch

from checkpoint import save_checkpoint, write_json
from config import Config
from course_data import load_course_data
from data_contract import prepare_batch
from loss import next_token_loss
from model import TinyGPT
from runtime import GPUMemoryTracker, autocast, seed_all, select_device, sync, to_device


def learning_rate_at(tokens: int, config: Config) -> float:
    if tokens <= config.warmup_tokens:
        return config.learning_rate * tokens / config.warmup_tokens
    progress = min(1.0, (tokens - config.warmup_tokens) / (config.target_tokens - config.warmup_tokens))
    return config.min_learning_rate + (config.learning_rate - config.min_learning_rate) * (
        1 + math.cos(math.pi * progress)
    ) / 2


def build_optimizer(model: TinyGPT, config: Config) -> torch.optim.AdamW:
    embeddings = {id(model.token_embedding.weight), id(model.position_embedding.weight)}
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim >= 2 and id(parameter) not in embeddings else no_decay).append(parameter)
    return torch.optim.AdamW([
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=config.learning_rate, betas=config.adam_betas)


@torch.no_grad()
def evaluate(model, data, config, device) -> tuple[float, int]:
    previous_mode = model.training
    model.eval()
    total_nll, count = 0.0, 0
    try:
        for raw in data.validation_loader:
            batch = to_device(prepare_batch(raw, data, config), device)
            with autocast(device):
                nll = next_token_loss(model(batch.inputs, batch.input_mask), batch.targets, "sum")
            total_nll += nll.item()
            count += batch.target_tokens
    finally:
        model.train(previous_mode)
    if count == 0 or not math.isfinite(total_nll):
        raise ValueError("Validation must contain valid targets and finite loss")
    return total_nll / count, count


def verify_overfit_gate(path: Path, config: Config, manifest: dict) -> None:
    if not path.exists():
        raise ValueError(f"Overfit gate missing: {path}. Run uv run python src/overfit.py first.")
    gate = json.loads(path.read_text())
    if not gate["passed"] or gate["data_sha256"] != manifest["sha256"]:
        raise ValueError("Overfit gate failed or belongs to different prepared data")
    for field in ("n_layers", "d_model", "n_heads", "context_length", "mlp_ratio"):
        if gate["config"][field] != getattr(config, field):
            raise ValueError(f"Overfit model differs: {field}")
    by_context = {entry["context"]: entry for entry in gate["contexts"]}
    for context in config.overfit_contexts:
        entry = by_context.get(context)
        if entry is None or not (
            entry["loss"] < config.overfit_loss_threshold and entry["accuracy"] > config.overfit_accuracy_threshold
        ):
            raise ValueError(f"Overfit gate did not pass at context {context}")


def run_training(config: Config, device: torch.device, gate_path: Path) -> dict:
    wall_start = time.perf_counter()
    manifest = json.loads((config.data_dir / "manifest.json").read_text())
    verify_overfit_gate(gate_path, config, manifest)
    if (config.output_dir / "results.json").exists() or any(config.output_dir.glob("checkpoints/*.pt")):
        raise ValueError("Run directory already contains an experiment; choose a fresh --output-dir")
    data = load_course_data(config)
    (config.output_dir / "checks").mkdir(parents=True, exist_ok=True)
    (config.output_dir / "metadata").mkdir(parents=True, exist_ok=True)
    if gate_path.resolve() != (config.output_dir / "checks/overfit.json").resolve():
        shutil.copyfile(gate_path, config.output_dir / "checks/overfit.json")
    shutil.copyfile(config.data_dir / "tokenizer.json", config.output_dir / "metadata/tokenizer.json")
    write_json(config.output_dir / "metadata/data_manifest.json", manifest)
    seed_all(config.seed)
    model = TinyGPT(data.vocab_size, config).to(device)
    memory = GPUMemoryTracker(device)
    optimizer = build_optimizer(model, config)
    use_fp16 = device.type == "cuda" and not torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    tokenizer_hash = manifest["sha256"]["tokenizer.json"]
    save_checkpoint(config.output_dir / "checkpoints/initial.pt", model, 0, tokenizer_hash)
    initial_loss, validation_count = evaluate(model, data, config, device)
    save_checkpoint(config.output_dir / "checkpoints/best.pt", model, 0, tokenizer_hash)
    precision = "float16" if use_fp16 else "bfloat16" if device.type == "cuda" else "float32"
    results = {
        "status": "running", "config": config.to_dict(), "device": str(device), "precision": precision,
        "torch_version": torch.__version__, "parameters": sum(p.numel() for p in model.parameters()),
        "tokenizer_sha256": tokenizer_hash, "validation_target_count": validation_count,
        "best_validation_loss": initial_loss, "best_perplexity": math.exp(initial_loss),
        "best_tokens_seen": 0, "tokens_seen": 0, "optimizer_steps": 0, "amp_skipped_updates": 0,
        "training_seconds": 0.0, "wall_seconds": 0.0, "tokens_per_second": 0.0,
        "peak_gpu_memory_bytes": None,
        "memory_measurement": memory.measurement,
        "memory_scope": "Live model allocation baseline through initial validation, training, periodic validation, and final checkpoint; excludes later report generation. Allocator memory, not total Metal-driver or system memory.",
        "training_history": [], "validation_history": [{"tokens_seen": 0, "loss": initial_loss}],
        "checkpoints": {"initial": 0, "best": 0},
    }
    write_json(config.output_dir / "results.json", results)
    iterator = iter(data.train_loader)
    next_evaluation = config.eval_every_tokens
    halfway_saved = False
    while results["tokens_seen"] < config.target_tokens:
        sync(device)
        tick = time.perf_counter()
        batches = [prepare_batch(next(iterator), data, config) for _ in range(config.accumulation_steps)]
        step_tokens = sum(batch.target_tokens for batch in batches)
        # Schedule advances once per optimizer step, using its ending token count.
        lr = learning_rate_at(results["tokens_seen"] + step_tokens, config)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        step_nll = 0.0
        model.train()
        for batch in batches:
            batch = to_device(batch, device)
            with autocast(device):
                nll = next_token_loss(model(batch.inputs, batch.input_mask), batch.targets, "sum")
                loss = nll / step_tokens
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss; inspect precision and learning rate")
            scaler.scale(loss).backward()
            step_nll += nll.detach().item()
            results["tokens_seen"] += batch.target_tokens
            memory.sample()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, error_if_nonfinite=not use_fp16)
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        results["amp_skipped_updates"] += int(scaler.get_scale() < old_scale)
        results["optimizer_steps"] += 1
        memory.sample()
        sync(device)
        results["training_seconds"] += time.perf_counter() - tick
        tokens = results["tokens_seen"]
        results["training_history"].append({"tokens_seen": tokens, "loss": step_nll / step_tokens, "lr": lr})
        if results["optimizer_steps"] % 50 == 0:
            results["tokens_per_second"] = tokens / results["training_seconds"]
            results["wall_seconds"] = time.perf_counter() - wall_start
            results["peak_gpu_memory_bytes"] = memory.peak_bytes()
            write_json(config.output_dir / "results.json", results)
            remaining = max(0, config.target_tokens - tokens) / results["tokens_per_second"]
            print(f"progress tokens={tokens:,}/{config.target_tokens:,} loss={step_nll / step_tokens:.4f} "
                  f"tokens/sec={results['tokens_per_second']:.0f} eta_minutes={remaining / 60:.1f}", flush=True)
        if not halfway_saved and tokens >= config.target_tokens / 2:
            save_checkpoint(config.output_dir / "checkpoints/halfway.pt", model, tokens, tokenizer_hash)
            results["checkpoints"]["halfway"] = tokens
            halfway_saved = True
        if tokens >= next_evaluation or tokens >= config.target_tokens:
            val_loss, _ = evaluate(model, data, config, device)
            results["validation_history"].append({"tokens_seen": tokens, "loss": val_loss})
            if val_loss < results["best_validation_loss"]:
                results.update(best_validation_loss=val_loss, best_perplexity=math.exp(val_loss), best_tokens_seen=tokens)
                save_checkpoint(config.output_dir / "checkpoints/best.pt", model, tokens, tokenizer_hash)
                results["checkpoints"]["best"] = tokens
            next_evaluation = (tokens // config.eval_every_tokens + 1) * config.eval_every_tokens
            results["tokens_per_second"] = tokens / results["training_seconds"]
            results["wall_seconds"] = time.perf_counter() - wall_start
            memory.sample()
            results["peak_gpu_memory_bytes"] = memory.peak_bytes()
            write_json(config.output_dir / "results.json", results)
            print(f"tokens={tokens:,} train={step_nll / step_tokens:.4f} val={val_loss:.4f} lr={lr:.3g}", flush=True)
    save_checkpoint(config.output_dir / "checkpoints/final.pt", model, results["tokens_seen"], tokenizer_hash)
    sync(device)
    memory.sample()
    results["peak_gpu_memory_bytes"] = memory.peak_bytes()
    results["checkpoints"]["final"] = results["tokens_seen"]
    results["status"] = "complete"
    results["wall_seconds"] = time.perf_counter() - wall_start
    write_json(config.output_dir / "results.json", results)
    return results


def main() -> None:
    defaults = Config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=defaults.data_dir)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--gate", type=Path, help="Defaults to OUTPUT_DIR/checks/overfit.json")
    parser.add_argument("--token-budget", type=int, default=defaults.target_tokens)
    parser.add_argument("--warmup-tokens", type=int, default=defaults.warmup_tokens)
    parser.add_argument("--eval-every-tokens", type=int, default=defaults.eval_every_tokens)
    parser.add_argument("--microbatch-size", type=int, default=defaults.microbatch_size)
    args = parser.parse_args()
    config = replace(defaults, data_dir=args.data_dir, output_dir=args.output_dir,
                     target_tokens=args.token_budget, warmup_tokens=args.warmup_tokens,
                     eval_every_tokens=args.eval_every_tokens, microbatch_size=args.microbatch_size)
    run_training(config, select_device(args.device), args.gate or config.output_dir / "checks/overfit.json")


if __name__ == "__main__":
    main()
