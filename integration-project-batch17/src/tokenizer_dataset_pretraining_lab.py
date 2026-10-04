"""Batch 17 integration: corpus -> byte BPE -> packed tokens -> tiny LLM."""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


def _load_module(relative_path: str, name: str):
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_documents() -> list[dict[str, object]]:
    pipeline = _load_module(
        "51-llm-dataset-pipeline/src/data_pipeline.py",
        "batch17_pipeline_builder",
    )

    templates = {
        "train": [
            "Transformers use causal attention to predict the next token.",
            "RMSNorm scales hidden states by root mean square magnitude.",
            "Byte pair encoding merges frequent adjacent byte tokens.",
            "Gradient accumulation increases effective tokens per optimizer update.",
            "A clean training corpus needs provenance and deduplication.",
            "Grouped query attention reduces key value cache memory.",
            "Warmup followed by cosine decay is a common learning rate schedule.",
            "Document level splitting prevents adjacent-window leakage.",
        ],
        "validation": [
            "Validation loss estimates generalization on held out documents.",
            "Perplexity is the exponential of mean token cross entropy.",
            "A frozen tokenizer must match the model vocabulary.",
        ],
        "test": [
            "Evaluation documents must remain outside optimizer updates.",
            "Dataset manifests record versions counts and checksums.",
        ],
    }

    documents: list[dict[str, object]] = []

    for desired_split, texts in templates.items():
        candidate = 0

        for text_index, text in enumerate(texts):
            while True:
                document_id = f"{desired_split}-{text_index}-{candidate}"
                split = pipeline.deterministic_split(
                    document_id,
                    train_fraction=0.70,
                    validation_fraction=0.15,
                )
                candidate += 1
                if split == desired_split:
                    break

            documents.append(
                {
                    "document_id": document_id,
                    "text": text,
                    "source": "synthetic-course",
                    "expected_split": desired_split,
                }
            )

    # Exact normalized duplicate: should be removed before splitting/tokenizing.
    documents.append(
        {
            "document_id": "duplicate-copy",
            "text": (
                "Transformers use causal attention to predict the next token.\r\n"
            ),
            "source": "synthetic-course",
            "expected_split": "duplicate",
        }
    )

    return documents


def build_dataset(
    *,
    num_merges: int = 48,
    sequence_length: int = 16,
) -> dict[str, Any]:
    byte_bpe = _load_module(
        "49-tokenizer-from-scratch/src/byte_bpe.py",
        "batch17_byte_bpe",
    )
    pipeline = _load_module(
        "51-llm-dataset-pipeline/src/data_pipeline.py",
        "batch17_data_pipeline",
    )

    documents = make_documents()
    deduped, removed_ids = pipeline.exact_deduplicate(documents)

    splits: dict[str, list[dict[str, object]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for document in deduped:
        split = pipeline.deterministic_split(
            str(document["document_id"]),
            train_fraction=0.70,
            validation_fraction=0.15,
        )
        splits[split].append(document)

    train_texts = [
        pipeline.normalize_text(str(document["text"]))
        for document in splits["train"]
    ]

    merges = byte_bpe.train_bpe(
        train_texts,
        num_merges=num_merges,
    )
    tokenizer = byte_bpe.ByteBPETokenizer(merges)

    tokenized: dict[str, list[list[int]]] = {}

    for split_name, split_documents in splits.items():
        tokenized[split_name] = [
            tokenizer.encode(
                pipeline.normalize_text(str(document["text"]))
            )
            for document in split_documents
        ]

    packed = {
        split_name: pipeline.pack_token_documents(
            docs,
            sequence_length=sequence_length + 1,
            separator_id=10,  # newline byte, valid base token
            drop_remainder=True,
        )
        for split_name, docs in tokenized.items()
    }

    train_hashes = {
        pipeline.canonical_document_hash(str(document["text"]))
        for document in splits["train"]
    }
    validation_hashes = {
        pipeline.canonical_document_hash(str(document["text"]))
        for document in splits["validation"]
    }
    test_hashes = {
        pipeline.canonical_document_hash(str(document["text"]))
        for document in splits["test"]
    }

    manifest = {
        "raw_documents": len(documents),
        "deduplicated_documents": len(deduped),
        "removed_duplicate_ids": removed_ids,
        "split_counts": {
            key: len(value)
            for key, value in splits.items()
        },
        "vocab_size": tokenizer.vocab_size,
        "num_merges": len(merges),
        "sequence_length": sequence_length,
        "packed_sequence_counts": {
            key: len(value)
            for key, value in packed.items()
        },
        "train_validation_hash_overlap": len(
            train_hashes & validation_hashes
        ),
        "train_test_hash_overlap": len(
            train_hashes & test_hashes
        ),
    }

    return {
        "manifest": manifest,
        "tokenizer": tokenizer,
        "splits": splits,
        "packed": packed,
    }


def run_pipeline() -> dict[str, Any]:
    built = build_dataset()

    tokenizer = built["tokenizer"]
    probe = "ภาษาไทย and English tokenizer"
    encoded = tokenizer.encode(probe)
    decoded = tokenizer.decode(encoded)

    return {
        "mode": "pipeline",
        **built["manifest"],
        "roundtrip_ok": decoded == probe,
        "probe_token_count": len(encoded),
        "probe_bytes_per_token": tokenizer.bytes_per_token(probe),
    }


def run_pretraining(
    *,
    steps: int = 3,
    seed: int = 42,
) -> dict[str, Any]:
    import torch
    from torch import nn
    from torch.nn import functional as F

    if steps <= 0:
        raise ValueError("steps must be positive")

    pretrain_utils = _load_module(
        "50-llm-pretraining-from-scratch/src/pretraining_utils.py",
        "batch17_pretraining_utils",
    )
    built = build_dataset()
    packed_train = built["packed"]["train"]

    if len(packed_train) == 0:
        raise RuntimeError("training corpus produced no packed sequences")

    sequence_length = built["manifest"]["sequence_length"]
    vocab_size = built["manifest"]["vocab_size"]

    inputs_np = packed_train[:, :sequence_length]
    targets_np = packed_train[:, 1 : sequence_length + 1]

    torch.manual_seed(seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model_dim = 32
    query_heads = 4
    kv_heads = 2
    head_dim = model_dim // query_heads
    hidden_dim = 64

    class Attention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q = nn.Linear(model_dim, query_heads * head_dim, bias=False)
            self.k = nn.Linear(model_dim, kv_heads * head_dim, bias=False)
            self.v = nn.Linear(model_dim, kv_heads * head_dim, bias=False)
            self.out = nn.Linear(model_dim, model_dim, bias=False)

        def _rope(self, tensor: torch.Tensor) -> torch.Tensor:
            position = torch.arange(
                tensor.shape[2],
                dtype=tensor.dtype,
                device=tensor.device,
            )
            inv = 10000.0 ** (
                -torch.arange(
                    0,
                    head_dim,
                    2,
                    dtype=tensor.dtype,
                    device=tensor.device,
                )
                / head_dim
            )
            angle = position[:, None] * inv[None, :]
            cos = angle.cos()[None, None]
            sin = angle.sin()[None, None]

            even = tensor[..., 0::2]
            odd = tensor[..., 1::2]
            result = torch.empty_like(tensor)
            result[..., 0::2] = even * cos - odd * sin
            result[..., 1::2] = even * sin + odd * cos
            return result

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch, time, _ = x.shape
            q = self.q(x).view(
                batch, time, query_heads, head_dim
            ).transpose(1, 2)
            k = self.k(x).view(
                batch, time, kv_heads, head_dim
            ).transpose(1, 2)
            v = self.v(x).view(
                batch, time, kv_heads, head_dim
            ).transpose(1, 2)

            q = self._rope(q)
            k = self._rope(k)

            repeats = query_heads // kv_heads
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)

            attended = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=0.0,
                is_causal=True,
            )
            attended = attended.transpose(1, 2).contiguous().view(
                batch,
                time,
                model_dim,
            )
            return self.out(attended)

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm1 = nn.RMSNorm(model_dim)
            self.attention = Attention()
            self.norm2 = nn.RMSNorm(model_dim)
            self.gate = nn.Linear(model_dim, hidden_dim, bias=False)
            self.up = nn.Linear(model_dim, hidden_dim, bias=False)
            self.down = nn.Linear(hidden_dim, model_dim, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.attention(self.norm1(x))
            normalized = self.norm2(x)
            mlp = self.down(
                F.silu(self.gate(normalized))
                * self.up(normalized)
            )
            return x + mlp

    class TinyLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, model_dim)
            self.block = Block()
            self.norm = nn.RMSNorm(model_dim)

        def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
            hidden = self.block(self.embedding(token_ids))
            hidden = self.norm(hidden)
            return F.linear(hidden, self.embedding.weight)

    model = TinyLM().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-3,
        weight_decay=0.01,
    )

    x = torch.from_numpy(inputs_np).to(device)
    y = torch.from_numpy(targets_np).to(device)

    losses: list[float] = []
    learning_rates: list[float] = []
    tokens_seen = 0

    for step in range(steps):
        lr = pretrain_utils.warmup_cosine_lr(
            step,
            warmup_steps=1,
            total_steps=max(steps, 2),
            max_lr=3e-3,
            min_lr=3e-4,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        model.train()
        optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            y.reshape(-1),
        )
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )
        if not torch.isfinite(grad_norm):
            raise RuntimeError("non-finite gradient norm")

        optimizer.step()

        losses.append(float(loss.detach().cpu()))
        learning_rates.append(float(lr))
        tokens_seen += int(y.numel())

    # Serialization smoke using recommended state-dict style.
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored_state = torch.load(
        buffer,
        map_location="cpu",
        weights_only=True,
    )

    return {
        "mode": "pretraining",
        "device": str(device),
        "steps": steps,
        "vocab_size": int(vocab_size),
        "sequence_length": int(sequence_length),
        "training_sequences": int(len(inputs_np)),
        "parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "loss_first": float(losses[0]),
        "loss_last": float(losses[-1]),
        "perplexity_last": float(np.exp(min(losses[-1], 20.0))),
        "tokens_seen": int(tokens_seen),
        "learning_rates": learning_rates,
        "checkpoint_tensor_keys": int(len(restored_state)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Batch 17 tokenizer/data/pretraining lab."
    )
    parser.add_argument(
        "--mode",
        choices=["pipeline", "pretraining"],
        required=True,
    )
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/batch17"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.mode == "pipeline":
        report = run_pipeline()
    else:
        report = run_pretraining(
            steps=args.steps,
            seed=args.seed,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.mode}.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"mode: {report['mode']}")
    print(f"report: {output}")


if __name__ == "__main__":
    main()
