"""Torch datasets over chunks.

One dataset per encoder family: the CNN consumes word ids from our own
vocabulary, the transformers consume whatever their tokenizer produces. Both
yield the same six-element multi-hot target, so the training loop does not
care which it is given.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .chunking import Chunk
from .labels import LABEL_NAMES, LABEL_TO_INDEX, NUM_LABELS
from .vocab import Vocabulary


def label_matrix(chunks: list[Chunk]) -> np.ndarray:
    """(n_chunks, 6) multi-hot targets, for loss weighting and metrics."""
    matrix = np.zeros((len(chunks), NUM_LABELS), dtype=np.float32)
    for row, chunk in enumerate(chunks):
        for label in chunk.labels:
            matrix[row, LABEL_TO_INDEX[label]] = 1.0
    return matrix


def _target(chunk: Chunk) -> torch.Tensor:
    target = torch.zeros(NUM_LABELS, dtype=torch.float32)
    for label in chunk.labels:
        target[LABEL_TO_INDEX[label]] = 1.0
    return target


class WordChunkDataset(Dataset):
    """Chunks as padded word-id sequences, for the CNN."""

    def __init__(self, chunks: list[Chunk], vocab: Vocabulary, max_tokens: int = 400) -> None:
        self.chunks = chunks
        self.vocab = vocab
        self.max_tokens = max_tokens

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        chunk = self.chunks[index]
        return {
            "input_ids": torch.tensor(self.vocab.encode(chunk.text, self.max_tokens)),
            "labels": _target(chunk),
        }


class TokenizedChunkDataset(Dataset):
    """Chunks tokenized by a Hugging Face tokenizer, for the transformers.

    Tokenization happens once, in batches, at construction. Doing it per item
    meant re-tokenizing every chunk on every epoch: identical work, repeated,
    on the CPU while the GPU waited. For 13,829 chunks over three epochs that
    is 41,000 wasted tokenizations.
    """

    def __init__(self, chunks: list[Chunk], tokenizer, max_length: int, batch: int = 512) -> None:
        self.chunks = chunks
        self.max_length = max_length

        encoded_batches: list[dict[str, torch.Tensor]] = []
        for start in range(0, len(chunks), batch):
            window = [chunk.text for chunk in chunks[start : start + batch]]
            encoded_batches.append(
                tokenizer(
                    window,
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                    return_tensors="pt",
                )
            )
        keys = encoded_batches[0].keys() if encoded_batches else ()
        self.encoded = {
            key: torch.cat([b[key] for b in encoded_batches], dim=0) for key in keys
        }
        self.targets = torch.stack([_target(chunk) for chunk in chunks]) if chunks else torch.empty(0)

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: value[index] for key, value in self.encoded.items()}
        item["labels"] = self.targets[index]
        return item


__all__ = ["WordChunkDataset", "TokenizedChunkDataset", "label_matrix", "LABEL_NAMES"]
