"""Batch 12 PyTorch integration: tiny BERT, GPT and T5-style models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def run_bert(
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_size = 32
    mask_id = 1
    seq_len = 10
    batch_size = 16

    class TinyBert(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            d_model = 24
            self.token = nn.Embedding(vocab_size, d_model)
            self.position = nn.Embedding(seq_len, d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=4,
                dim_feedforward=48,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=1)
            self.norm = nn.LayerNorm(d_model)
            self.lm_head = nn.Linear(d_model, vocab_size)

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            positions = torch.arange(
                tokens.shape[1],
                device=tokens.device,
            )
            x = (
                self.token(tokens)
                + self.position(positions)[None, :, :]
            )
            x = self.encoder(x)
            return self.lm_head(self.norm(x))

    model = TinyBert().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    generator = torch.Generator().manual_seed(seed)
    clean = torch.randint(
        4,
        vocab_size,
        (batch_size, seq_len),
        generator=generator,
    )

    labels = torch.full_like(clean, -100)
    selected = torch.rand(
        (batch_size, seq_len),
        generator=generator,
    ) < 0.30
    labels[selected] = clean[selected]

    corrupted = clean.clone()
    corrupted[selected] = mask_id

    corrupted = corrupted.to(device)
    labels = labels.to(device)

    losses = []

    for _ in range(steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(corrupted)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            labels.reshape(-1),
            ignore_index=-100,
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    return {
        "mode": "bert",
        "device": str(device),
        "steps": steps,
        "parameter_count": int(
            sum(p.numel() for p in model.parameters())
        ),
        "selected_positions": int((labels != -100).sum().cpu()),
        "loss_history": losses,
    }


def run_gpt(
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_size = 24
    context = 12
    batch_size = 16

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            d_model = 24
            self.norm1 = nn.LayerNorm(d_model)
            self.attn = nn.MultiheadAttention(
                d_model,
                4,
                dropout=0.0,
                batch_first=True,
            )
            self.norm2 = nn.LayerNorm(d_model)
            self.ff = nn.Sequential(
                nn.Linear(d_model, 48),
                nn.GELU(),
                nn.Linear(48, d_model),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            time = x.shape[1]
            mask = torch.triu(
                torch.ones(
                    time,
                    time,
                    dtype=torch.bool,
                    device=x.device,
                ),
                diagonal=1,
            )
            normed = self.norm1(x)
            attended, _ = self.attn(
                normed,
                normed,
                normed,
                attn_mask=mask,
                need_weights=False,
            )
            x = x + attended
            return x + self.ff(self.norm2(x))

    class TinyGPT(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            d_model = 24
            self.token = nn.Embedding(vocab_size, d_model)
            self.position = nn.Embedding(context, d_model)
            self.block = Block()
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            positions = torch.arange(
                tokens.shape[1],
                device=tokens.device,
            )
            x = (
                self.token(tokens)
                + self.position(positions)[None, :, :]
            )
            x = self.block(x)
            return self.head(self.norm(x))

    model = TinyGPT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    generator = torch.Generator().manual_seed(seed)
    sequences = torch.randint(
        0,
        vocab_size,
        (batch_size, context + 1),
        generator=generator,
    )
    x = sequences[:, :-1].to(device)
    y = sequences[:, 1:].to(device)

    losses = []

    for _ in range(steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            y.reshape(-1),
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    model.eval()
    generated = [2, 3, 4]

    with torch.inference_mode():
        for _ in range(5):
            prefix = generated[-context:]
            tokens = torch.tensor(
                [prefix],
                dtype=torch.long,
                device=device,
            )
            logits = model(tokens)
            next_id = int(
                torch.argmax(logits[0, -1]).cpu()
            )
            generated.append(next_id)

    return {
        "mode": "gpt",
        "device": str(device),
        "steps": steps,
        "parameter_count": int(
            sum(p.numel() for p in model.parameters())
        ),
        "loss_history": losses,
        "generated_ids": generated,
    }


def run_t5(
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_size = 40
    seq_len = 7
    batch_size = 16
    bos_id = 1

    class TinySeq2Seq(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            d_model = 24
            self.src_token = nn.Embedding(vocab_size, d_model)
            self.tgt_token = nn.Embedding(vocab_size, d_model)
            self.src_position = nn.Embedding(seq_len, d_model)
            self.tgt_position = nn.Embedding(seq_len, d_model)
            self.transformer = nn.Transformer(
                d_model=d_model,
                nhead=4,
                num_encoder_layers=1,
                num_decoder_layers=1,
                dim_feedforward=48,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.head = nn.Linear(d_model, vocab_size)

        def forward(
            self,
            source: torch.Tensor,
            decoder_input: torch.Tensor,
        ) -> torch.Tensor:
            src_pos = torch.arange(
                source.shape[1],
                device=source.device,
            )
            tgt_pos = torch.arange(
                decoder_input.shape[1],
                device=source.device,
            )

            src = (
                self.src_token(source)
                + self.src_position(src_pos)[None, :, :]
            )
            tgt = (
                self.tgt_token(decoder_input)
                + self.tgt_position(tgt_pos)[None, :, :]
            )

            time = decoder_input.shape[1]
            causal = torch.triu(
                torch.ones(
                    time,
                    time,
                    dtype=torch.bool,
                    device=source.device,
                ),
                diagonal=1,
            )

            hidden = self.transformer(
                src,
                tgt,
                tgt_mask=causal,
            )
            return self.head(hidden)

    model = TinySeq2Seq().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    generator = torch.Generator().manual_seed(seed)
    source = torch.randint(
        4,
        vocab_size,
        (batch_size, seq_len),
        generator=generator,
    )
    target = torch.flip(source, dims=[1])

    decoder_input = torch.cat(
        [
            torch.full(
                (batch_size, 1),
                bos_id,
                dtype=torch.long,
            ),
            target[:, :-1],
        ],
        dim=1,
    )

    source = source.to(device)
    target = target.to(device)
    decoder_input = decoder_input.to(device)

    losses = []

    for _ in range(steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(source, decoder_input)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            target.reshape(-1),
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    model.eval()
    with torch.inference_mode():
        logits = model(source[:1], decoder_input[:1])
        prediction = torch.argmax(
            logits,
            dim=-1,
        )[0].cpu().tolist()

    return {
        "mode": "t5",
        "device": str(device),
        "steps": steps,
        "parameter_count": int(
            sum(p.numel() for p in model.parameters())
        ),
        "loss_history": losses,
        "prediction_ids": prediction,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Batch 12 BERT/GPT/T5 lab."
    )
    parser.add_argument(
        "--mode",
        choices=["bert", "gpt", "t5"],
        required=True,
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/batch12"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    runners = {
        "bert": run_bert,
        "gpt": run_gpt,
        "t5": run_t5,
    }
    report = runners[args.mode](
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
    print(f"device: {report['device']}")
    print(f"report: {output}")


if __name__ == "__main__":
    main()
