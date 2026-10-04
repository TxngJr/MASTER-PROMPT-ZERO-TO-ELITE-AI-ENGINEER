"""Batch 16 integration: RAG, safe agents and a tiny modern LLM."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
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


def _hash_embedding(
    text: str,
    *,
    dimension: int = 96,
) -> np.ndarray:
    if dimension <= 0:
        raise ValueError("dimension must be positive")

    vector = np.zeros(dimension, dtype=float)

    for token in text.lower().split():
        digest = hashlib.blake2b(
            token.encode("utf-8"),
            digest_size=16,
        ).digest()
        index = int.from_bytes(digest[:8], "little") % dimension
        sign = 1.0 if digest[8] % 2 == 0 else -1.0
        vector[index] += sign

    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return vector


def _sparse_overlap_score(query: str, document: str) -> float:
    q = set(query.lower().split())
    d = set(document.lower().split())

    if not q:
        return 0.0
    return float(len(q & d) / len(q))


def run_rag() -> dict[str, Any]:
    rag = _load_module(
        "46-rag/src/rag_numpy.py",
        "batch16_rag_numpy",
    )

    documents = [
        {
            "source_id": "doc-rmsnorm",
            "text": (
                "RMSNorm normalizes hidden states using root mean square "
                "magnitude and is commonly used before attention and MLP "
                "sublayers in modern decoder language models."
            ),
        },
        {
            "source_id": "doc-rope",
            "text": (
                "RoPE rotary position embeddings rotate query and key "
                "features with position dependent angles."
            ),
        },
        {
            "source_id": "doc-rag",
            "text": (
                "RAG retrieves external evidence before generation and "
                "should preserve source provenance for grounded citations."
            ),
        },
        {
            "source_id": "doc-gqa",
            "text": (
                "Grouped query attention uses fewer key value heads than "
                "query heads to reduce KV cache memory."
            ),
        },
    ]

    query = "where is RMSNorm used in modern decoder language models"

    vectors = np.stack(
        [_hash_embedding(doc["text"]) for doc in documents]
    )
    query_vector = _hash_embedding(query)

    dense_ids, _ = rag.cosine_top_k(
        query_vector,
        vectors,
        k=len(documents),
    )
    dense_ranking = [
        documents[int(index)]["source_id"]
        for index in dense_ids
    ]

    sparse_order = sorted(
        range(len(documents)),
        key=lambda index: (
            -_sparse_overlap_score(query, documents[index]["text"]),
            documents[index]["source_id"],
        ),
    )
    sparse_ranking = [
        documents[index]["source_id"]
        for index in sparse_order
    ]

    fused = rag.reciprocal_rank_fusion(
        [dense_ranking, sparse_ranking],
        rank_constant=10,
    )
    fused_ids = [item_id for item_id, _ in fused]

    by_id = {
        document["source_id"]: document
        for document in documents
    }
    ordered_chunks = [by_id[item_id] for item_id in fused_ids]
    packed = rag.pack_context(
        ordered_chunks,
        max_words=45,
    )

    return {
        "mode": "rag",
        "query": query,
        "dense_ranking": dense_ranking,
        "sparse_ranking": sparse_ranking,
        "fused_ranking": fused_ids,
        "packed_source_ids": [
            chunk["source_id"] for chunk in packed
        ],
        "top_source_is_relevant": fused_ids[0] == "doc-rmsnorm",
    }


def run_agent() -> dict[str, Any]:
    agent = _load_module(
        "47-ai-agents-tool-calling/src/agent_core.py",
        "batch16_agent_core",
    )

    registry = agent.ToolRegistry()

    knowledge = {
        "rmsnorm": "RMSNorm uses root mean square magnitude.",
        "rope": "RoPE rotates query/key feature pairs by position.",
    }

    def lookup(topic: str) -> dict[str, str]:
        return {
            "topic": topic,
            "text": knowledge.get(topic.lower(), "not found"),
        }

    registry.register(
        agent.ToolSpec(
            name="lookup",
            required={"topic": str},
            read_only=True,
        ),
        lookup,
    )

    registry.register(
        agent.ToolSpec(
            name="save_note",
            required={"text": str},
            read_only=False,
        ),
        lambda text: {"saved": text},
    )

    def policy(state):
        if not state.observations:
            return {
                "type": "tool",
                "name": "lookup",
                "arguments": {"topic": "rmsnorm"},
            }

        return {
            "type": "finish",
            "answer": state.observations[-1].data["text"],
        }

    state = agent.bounded_agent_loop(
        goal="Explain RMSNorm",
        registry=registry,
        policy=policy,
        max_steps=4,
        allow_write=False,
    )

    blocked = registry.execute(
        "save_note",
        {"text": "side effect"},
        allow_write=False,
    )

    return {
        "mode": "agent",
        "done": state.done,
        "steps": state.steps,
        "answer": state.answer,
        "tool_calls": len(state.observations),
        "write_blocked_by_default": (
            not blocked.ok
            and blocked.error == "write_permission_required"
        ),
    }


def run_llm(
    *,
    steps: int = 2,
    seed: int = 42,
) -> dict[str, Any]:
    import torch
    from torch import nn
    from torch.nn import functional as F

    if steps <= 0:
        raise ValueError("steps must be positive")

    torch.manual_seed(seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    vocab_size = 32
    model_dim = 32
    query_heads = 4
    kv_heads = 2
    head_dim = model_dim // query_heads
    sequence_length = 8
    intermediate_dim = 64

    def apply_rope(
        x: torch.Tensor,
    ) -> torch.Tensor:
        # x: B,H,T,D
        positions = torch.arange(
            x.shape[2],
            dtype=x.dtype,
            device=x.device,
        )
        inverse_frequency = 10000.0 ** (
            -torch.arange(
                0,
                head_dim,
                2,
                dtype=x.dtype,
                device=x.device,
            )
            / head_dim
        )
        angles = positions[:, None] * inverse_frequency[None, :]
        cos = torch.cos(angles)[None, None, :, :]
        sin = torch.sin(angles)[None, None, :, :]

        even = x[..., 0::2]
        odd = x[..., 1::2]

        out = torch.empty_like(x)
        out[..., 0::2] = even * cos - odd * sin
        out[..., 1::2] = even * sin + odd * cos
        return out

    class GQAAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q = nn.Linear(
                model_dim,
                query_heads * head_dim,
                bias=False,
            )
            self.k = nn.Linear(
                model_dim,
                kv_heads * head_dim,
                bias=False,
            )
            self.v = nn.Linear(
                model_dim,
                kv_heads * head_dim,
                bias=False,
            )
            self.out = nn.Linear(model_dim, model_dim, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch, time, _ = x.shape

            q = self.q(x).view(
                batch,
                time,
                query_heads,
                head_dim,
            ).transpose(1, 2)
            k = self.k(x).view(
                batch,
                time,
                kv_heads,
                head_dim,
            ).transpose(1, 2)
            v = self.v(x).view(
                batch,
                time,
                kv_heads,
                head_dim,
            ).transpose(1, 2)

            q = apply_rope(q)
            k = apply_rope(k)

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

    class SwiGLU(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate = nn.Linear(
                model_dim,
                intermediate_dim,
                bias=False,
            )
            self.up = nn.Linear(
                model_dim,
                intermediate_dim,
                bias=False,
            )
            self.down = nn.Linear(
                intermediate_dim,
                model_dim,
                bias=False,
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.down(
                F.silu(self.gate(x)) * self.up(x)
            )

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm1 = nn.RMSNorm(model_dim)
            self.attention = GQAAttention()
            self.norm2 = nn.RMSNorm(model_dim)
            self.mlp = SwiGLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.attention(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x

    class TinyModernLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.token = nn.Embedding(vocab_size, model_dim)
            self.blocks = nn.ModuleList(
                [Block() for _ in range(2)]
            )
            self.norm = nn.RMSNorm(model_dim)

        def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
            x = self.token(token_ids)

            for block in self.blocks:
                x = block(x)

            x = self.norm(x)

            # Weight-tied LM head.
            return F.linear(x, self.token.weight)

    model = TinyModernLM().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-3,
    )

    offsets = torch.arange(16)[:, None]
    positions = torch.arange(sequence_length + 1)[None, :]
    sequences = (
        offsets + positions
    ) % vocab_size

    inputs = sequences[:, :-1].to(device)
    targets = sequences[:, 1:].to(device)

    losses = []

    for _ in range(steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        logits = model(inputs)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1),
        )
        loss.backward()
        optimizer.step()

        losses.append(float(loss.detach().cpu()))

    model.eval()

    with torch.inference_mode():
        first = torch.tensor(
            [[1, 2, 3, 4, 5, 6, 7, 8]],
            device=device,
        )
        second = first.clone()
        second[:, 4:] = torch.tensor(
            [[20, 21, 22, 23]],
            device=device,
        )

        logits_a = model(first)
        logits_b = model(second)

        prefix_difference = float(
            torch.max(
                torch.abs(
                    logits_a[:, :4]
                    - logits_b[:, :4]
                )
            ).cpu()
        )

    kv_bytes = (
        2
        * 2  # layers
        * 1  # batch
        * kv_heads
        * 2048
        * head_dim
        * 2  # fp16/bf16-style bytes
    )

    return {
        "mode": "llm",
        "device": str(device),
        "steps": steps,
        "parameter_count": int(
            sum(p.numel() for p in model.parameters())
        ),
        "loss_first": float(losses[0]),
        "loss_last": float(losses[-1]),
        "query_heads": query_heads,
        "kv_heads": kv_heads,
        "causal_prefix_max_diff": prefix_difference,
        "example_kv_cache_bytes_2048": int(kv_bytes),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Batch 16 RAG/agent/LLM lab."
    )
    parser.add_argument(
        "--mode",
        choices=["rag", "agent", "llm"],
        required=True,
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/batch16"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.mode == "rag":
        report = run_rag()
    elif args.mode == "agent":
        report = run_agent()
    else:
        report = run_llm(
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
