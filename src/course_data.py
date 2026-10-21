"""Reference notebook's random training windows and fixed validation prefix."""

import json

import numpy as np
import torch
from tokenizers import Tokenizer

from config import Config
from data_contract import CourseData, NextTokenBatch
from prepare_data import sha256_file


class CourseTokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)


def get_batch(tokens, batch_size, context, rng) -> NextTokenBatch:
    if len(tokens) <= context:
        raise ValueError("Token stream is shorter than a complete context window")
    starts = rng.integers(0, len(tokens) - context, size=batch_size)
    z = np.stack([tokens[s:s + context + 1] for s in starts]).astype(np.int64)
    return NextTokenBatch(torch.from_numpy(z[:, :-1].copy()), torch.from_numpy(z[:, 1:].copy()))


class TrainingWindows:
    def __init__(self, tokens, config: Config):
        self.tokens, self.config = tokens, config

    def __iter__(self):
        rng = np.random.default_rng(self.config.seed)
        while True:
            yield get_batch(self.tokens, self.config.microbatch_size, self.config.context_length, rng)


class ValidationWindows:
    def __init__(self, tokens, config: Config):
        self.tokens, self.config = tokens, config
        self.target_count = min(
            config.validation_targets,
            ((len(tokens) - 1) // config.context_length) * config.context_length,
        )
        if self.target_count <= 0 or self.target_count % config.context_length:
            raise ValueError("Fixed validation targets must contain complete context windows")

    def __iter__(self):
        context = self.config.context_length
        batch_targets = self.config.microbatch_size * context
        for start in range(0, self.target_count, batch_targets):
            stops = range(start, min(start + batch_targets, self.target_count), context)
            z = np.stack([self.tokens[s:s + context + 1] for s in stops]).astype(np.int64)
            yield NextTokenBatch(torch.from_numpy(z[:, :-1].copy()), torch.from_numpy(z[:, 1:].copy()))


def load_course_data(config: Config) -> CourseData:
    manifest_path = config.data_dir / "manifest.json"
    if not manifest_path.exists():
        raise ValueError("Prepared data missing. Run: uv run python src/prepare_data.py")
    manifest = json.loads(manifest_path.read_text())
    for name in ("tokenizer.json", "train_tokens.npy", "val_tokens.npy", "train.jsonl", "validation.jsonl"):
        if sha256_file(config.data_dir / name) != manifest["sha256"][name]:
            raise ValueError(f"Prepared data hash mismatch: {name}")
    tokenizer = Tokenizer.from_file(str(config.data_dir / "tokenizer.json"))
    if tokenizer.get_vocab_size() != manifest["vocab_size"]:
        raise ValueError("Tokenizer vocabulary differs from manifest")
    train = np.load(config.data_dir / "train_tokens.npy", mmap_mode="r", allow_pickle=False)
    validation = np.load(config.data_dir / "val_tokens.npy", mmap_mode="r", allow_pickle=False)
    for tokens in (train, validation):
        if tokens.ndim != 1 or tokens.dtype != np.uint16 or len(tokens) <= config.context_length:
            raise ValueError("Expected a nonempty reference uint16 token stream")
    return CourseData(
        tokenizer=CourseTokenizer(tokenizer),
        vocab_size=tokenizer.get_vocab_size(),
        train_loader=TrainingWindows(train, config),
        validation_loader=ValidationWindows(validation, config),
        source=f"{manifest['source']}; revision={manifest['dataset_revision']}",
        eos_id=tokenizer.token_to_id("<eos>"),
    )
