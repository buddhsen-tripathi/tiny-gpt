"""Generate the nine preselected continuations from initial/halfway/final weights."""

import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from checkpoint import load_checkpoint, write_json
from config import Config
from prepare_data import sha256_file
from runtime import select_device


@torch.no_grad()
def generate_continuation(model, tokenizer, prompt: str, config: Config) -> dict:
    if config.early_stop:
        raise ValueError("The comparison protocol requires no early stopping")
    if not 0 < config.top_k <= model.vocab_size:
        raise ValueError("top_k must fit the tokenizer vocabulary")
    prefix = tokenizer.encode(prompt).ids
    if not prefix:
        raise ValueError("Prompt must encode to at least one token")
    device = next(model.parameters()).device
    ids = torch.tensor([prefix], dtype=torch.long, device=device)
    # MPS does not need a device-specific RNG; every prompt gets the same CPU seed.
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    was_training = model.training
    model.eval()
    try:
        for _ in range(config.max_new_tokens):
            logits = model(ids[:, -model.config.context_length:])[:, -1].float() / config.temperature
            values, indices = torch.topk(logits, config.top_k, dim=-1)
            choice = torch.multinomial(values.softmax(-1).cpu(), 1, generator=generator).to(device)
            next_id = indices.gather(-1, choice)
            ids = torch.cat((ids, next_id), dim=1)
    finally:
        model.train(was_training)
    new_ids = ids[0, len(prefix):].tolist()
    return {
        "prompt": prompt,
        "continuation": tokenizer.decode(new_ids, skip_special_tokens=True),
        "full_text": tokenizer.decode(ids[0].tolist(), skip_special_tokens=True),
        "new_token_ids": new_ids,
        "new_token_count": len(new_ids),
    }


def generate_comparison(run_dir: Path, device: torch.device, output_dir: Path | None = None) -> dict:
    output_dir = output_dir or run_dir / "report"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = json.loads((run_dir / "results.json").read_text())
    if results["status"] != "complete":
        raise ValueError("Complete training before generating the three-stage comparison")
    config = Config.from_dict(results["config"])
    tokenizer_hash = sha256_file(run_dir / "metadata/tokenizer.json")
    if tokenizer_hash != results["tokenizer_sha256"]:
        raise ValueError("Run tokenizer hash does not match results")
    tokenizer = Tokenizer.from_file(str(run_dir / "metadata/tokenizer.json"))
    output = {
        "sampling": {name: getattr(config, name) for name in (
            "seed", "temperature", "top_k", "max_new_tokens", "early_stop",
        )},
        "sampling_rng": "CPU generator reset before each prompt",
        "device": str(device), "precision": "float32", "tokenizer_sha256": tokenizer_hash,
        "stages": {},
    }
    text = ["# TinyGPT story comparison", "",
            f"Temperature {config.temperature}; top-k {config.top_k}; {config.max_new_tokens} new tokens; "
            f"no early stopping; seed {config.seed} reset for each prompt.",
            "Special tokens are hidden in decoded text but retained in samples.json token ids.", ""]
    for stage in ("initial", "halfway", "final"):
        model, metadata = load_checkpoint(run_dir / f"checkpoints/{stage}.pt", device, tokenizer_hash)
        if model.vocab_size != tokenizer.get_vocab_size():
            raise ValueError("Checkpoint vocabulary does not match tokenizer")
        samples = [generate_continuation(model, tokenizer, prompt, config) for prompt in config.prompts]
        output["stages"][stage] = {"tokens_seen": metadata["tokens_seen"], "samples": samples}
        text.extend([f"## {stage.title()} — {metadata['tokens_seen']:,} training tokens", ""])
        for i, sample in enumerate(samples, 1):
            text.extend([f"### Prompt {i}", "", sample["prompt"], "", sample["continuation"], ""])
        print(f"Generated {stage}: 3 prompts × {config.max_new_tokens} new tokens", flush=True)
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
    write_json(output_dir / "samples.json", output)
    (output_dir / "samples.md").write_text("\n".join(text))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Config().output_dir)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = parser.parse_args()
    generate_comparison(args.run_dir, select_device(args.device))


if __name__ == "__main__":
    main()
