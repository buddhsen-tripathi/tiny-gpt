"""Adapt the supplied TinyStories_Reference.ipynb data cells (4 and 6).

Only the real TinyStories route is supported. No model weights are downloaded.
"""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from config import Config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def story_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()


def unique_texts(rows, count: int, excluded=None) -> list[str]:
    seen = set() if excluded is None else set(excluded)
    result = []
    for row in rows:
        text = row["text"].strip()
        digest = story_hash(text)
        if text and digest not in seen:
            result.append(text)
            seen.add(digest)
        if len(result) == count:
            return result
    raise ValueError(f"Expected {count} distinct stories, found {len(result)}")


def encode_stream(tokenizer: Tokenizer, texts: list[str], eos_id: int) -> np.ndarray:
    parts = []
    for start in range(0, len(texts), 512):
        for encoding in tokenizer.encode_batch(texts[start:start + 512]):
            parts.append(np.asarray(encoding.ids + [eos_id], dtype=np.uint16))
    return np.concatenate(parts)


def prepare(config: Config, revision: str | None = None) -> dict:
    from datasets import load_dataset
    from huggingface_hub import HfApi

    destination = config.data_dir
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"{destination} is not empty; choose a new data directory")
    # Pin a single resolved upstream revision for both streams and record it.
    resolved_revision = HfApi().dataset_info(config.dataset_name, revision=revision).sha
    print(f"Streaming {config.dataset_name} at {resolved_revision}", flush=True)
    train = unique_texts(load_dataset(
        config.dataset_name, revision=resolved_revision, split="train", streaming=True,
    ), config.train_stories)
    validation = unique_texts(load_dataset(
        config.dataset_name, revision=resolved_revision, split="validation", streaming=True,
    ), config.validation_stories, {story_hash(text) for text in train})
    random.Random(config.seed).shuffle(train)
    destination.mkdir(parents=True, exist_ok=True)
    for split, texts in (("train", train), ("validation", validation)):
        with (destination / f"{split}.jsonl").open("w") as stream:
            for text in texts:
                stream.write(json.dumps({"text": text}) + "\n")
    print("Training byte-level BPE on the training split only", flush=True)
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(train, trainer=trainers.BpeTrainer(
        vocab_size=config.tokenizer_vocab_size,
        min_frequency=config.tokenizer_min_frequency,
        special_tokens=["<unk>", "<eos>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    ))
    tokenizer.save(str(destination / "tokenizer.json"))
    eos_id = tokenizer.token_to_id("<eos>")
    if tokenizer.get_vocab_size() >= 65536 or eos_id is None:
        raise ValueError("Invalid vocabulary for the reference uint16 format")
    counts = {}
    for split, texts in (("train", train), ("val", validation)):
        tokens = encode_stream(tokenizer, texts, eos_id)
        np.save(destination / f"{split}_tokens.npy", tokens)
        counts[split] = len(tokens)
    files = ["train.jsonl", "validation.jsonl", "tokenizer.json", "train_tokens.npy", "val_tokens.npy"]
    manifest = {
        "source": "TinyStories_Reference.ipynb, real-data full profile, cells 4/6/8/14",
        "dataset": config.dataset_name,
        "dataset_revision": resolved_revision,
        "seed": config.seed,
        "train_stories": len(train),
        "validation_stories": len(validation),
        "requested_vocab_size": config.tokenizer_vocab_size,
        "vocab_size": tokenizer.get_vocab_size(),
        "eos_id": eos_id,
        "token_counts": counts,
        "sha256": {name: sha256_file(destination / name) for name in files},
        "versions": {name: version(name) for name in ("numpy", "tokenizers", "datasets")},
    }
    # Written last so interrupted preparation never appears complete.
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Config().data_dir)
    parser.add_argument("--revision", help="Optional Hugging Face dataset revision")
    args = parser.parse_args()
    print(json.dumps(prepare(Config(data_dir=args.data_dir), args.revision), indent=2))


if __name__ == "__main__":
    main()
