"""Batch 10 PyTorch lab: a tiny GAN and decoder-only Transformer LM."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np


def make_ring_mixture(
    n_samples: int = 1000,
    *,
    modes: int = 8,
    radius: float = 2.0,
    noise: float = 0.08,
    seed: int = 42,
) -> np.ndarray:
    if n_samples <= 0 or modes <= 1:
        raise ValueError("invalid sample/mode count")

    rng = np.random.default_rng(seed)
    mode_ids = rng.integers(0, modes, size=n_samples)
    angles = 2.0 * np.pi * mode_ids / modes
    centers = np.column_stack(
        [radius * np.cos(angles), radius * np.sin(angles)]
    )
    return (
        centers
        + rng.normal(0.0, noise, size=(n_samples, 2))
    ).astype(np.float32)


def run_gan(
    *,
    steps: int = 100,
    batch_size: int = 128,
    seed: int = 42,
) -> dict[str, Any]:
    import torch
    from torch import nn

    if steps <= 0 or batch_size <= 0:
        raise ValueError("steps and batch_size must be positive")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real_np = make_ring_mixture(3000, seed=seed)
    real = torch.from_numpy(real_np).to(device)

    generator = nn.Sequential(
        nn.Linear(2, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
    ).to(device)

    discriminator = nn.Sequential(
        nn.Linear(2, 32),
        nn.LeakyReLU(0.2),
        nn.Linear(32, 32),
        nn.LeakyReLU(0.2),
        nn.Linear(32, 1),
    ).to(device)

    opt_g = torch.optim.Adam(
        generator.parameters(),
        lr=1e-3,
        betas=(0.5, 0.999),
    )
    opt_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=1e-3,
        betas=(0.5, 0.999),
    )
    criterion = nn.BCEWithLogitsLoss()

    fixed_z = torch.randn(256, 2, device=device)
    history: list[dict[str, float]] = []

    for step in range(steps):
        idx = torch.randint(0, len(real), (batch_size,), device=device)
        real_batch = real[idx]

        z = torch.randn(batch_size, 2, device=device)
        fake_for_d = generator(z).detach()

        opt_d.zero_grad(set_to_none=True)
        real_logits = discriminator(real_batch)
        fake_logits = discriminator(fake_for_d)
        d_loss = 0.5 * (
            criterion(real_logits, torch.ones_like(real_logits))
            + criterion(fake_logits, torch.zeros_like(fake_logits))
        )
        d_loss.backward()
        opt_d.step()

        z = torch.randn(batch_size, 2, device=device)
        opt_g.zero_grad(set_to_none=True)
        fake = generator(z)
        fake_logits_for_g = discriminator(fake)
        g_loss = criterion(
            fake_logits_for_g,
            torch.ones_like(fake_logits_for_g),
        )
        g_loss.backward()
        opt_g.step()

        if step == 0 or (step + 1) % max(1, steps // 5) == 0:
            history.append(
                {
                    "step": float(step + 1),
                    "d_loss": float(d_loss.detach().cpu()),
                    "g_loss": float(g_loss.detach().cpu()),
                }
            )

    generator.eval()
    with torch.inference_mode():
        generated = generator(fixed_z).cpu().numpy()

    covariance = np.cov(generated, rowvar=False)

    return {
        "mode": "gan",
        "device": str(device),
        "steps": steps,
        "history": history,
        "generated_mean": generated.mean(axis=0).tolist(),
        "generated_covariance": covariance.tolist(),
        "real_mean": real_np.mean(axis=0).tolist(),
        "real_covariance": np.cov(real_np, rowvar=False).tolist(),
        "generator_parameters": int(
            sum(p.numel() for p in generator.parameters())
        ),
        "discriminator_parameters": int(
            sum(p.numel() for p in discriminator.parameters())
        ),
    }


def _build_corpus() -> str:
    return (
        "attention lets tokens communicate across distance. "
        "transformers combine attention residual connections normalization "
        "and feed forward networks. "
        "causal language models predict the next token from previous tokens. "
    ) * 40


def run_transformer(
    *,
    steps: int = 100,
    batch_size: int = 32,
    context_length: int = 24,
    seed: int = 42,
) -> dict[str, Any]:
    import torch
    from torch import nn
    from torch.nn import functional as F

    if min(steps, batch_size, context_length) <= 0:
        raise ValueError("training arguments must be positive")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    corpus = _build_corpus()
    chars = sorted(set(corpus))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    encoded = torch.tensor([stoi[ch] for ch in corpus], dtype=torch.long)

    split = int(0.9 * len(encoded))
    train_data = encoded[:split]
    validation_data = encoded[split:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class Block(nn.Module):
        def __init__(self, d_model: int, heads: int, ff_dim: int) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(d_model)
            self.attn = nn.MultiheadAttention(
                d_model,
                heads,
                batch_first=True,
            )
            self.norm2 = nn.LayerNorm(d_model)
            self.ff = nn.Sequential(
                nn.Linear(d_model, ff_dim),
                nn.GELU(),
                nn.Linear(ff_dim, d_model),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            time = x.shape[1]
            causal_mask = torch.triu(
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
                attn_mask=causal_mask,
                need_weights=False,
            )
            x = x + attended
            return x + self.ff(self.norm2(x))

    class TinyTransformerLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            vocab = len(chars)
            d_model = 48
            self.context_length = context_length
            self.token_embedding = nn.Embedding(vocab, d_model)
            self.position_embedding = nn.Embedding(context_length, d_model)
            self.blocks = nn.Sequential(
                Block(d_model, 4, 96),
                Block(d_model, 4, 96),
            )
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab, bias=False)

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            _, time = tokens.shape
            if time > self.context_length:
                raise ValueError("sequence exceeds context length")
            positions = torch.arange(time, device=tokens.device)
            x = (
                self.token_embedding(tokens)
                + self.position_embedding(positions)[None, :, :]
            )
            x = self.blocks(x)
            return self.head(self.norm(x))

    def get_batch(
        data: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_start = len(data) - context_length - 1
        if max_start <= 0:
            raise ValueError("data shorter than context length")
        starts = torch.randint(0, max_start, (batch_size,))
        x = torch.stack(
            [data[i : i + context_length] for i in starts]
        )
        y = torch.stack(
            [data[i + 1 : i + context_length + 1] for i in starts]
        )
        return x.to(device), y.to(device)

    model = TinyTransformerLM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    best_val = float("inf")
    best_state = None
    history: list[dict[str, float]] = []

    for step in range(steps):
        model.train()
        xb, yb = get_batch(train_data)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            yb.reshape(-1),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 0 or (step + 1) % max(1, steps // 5) == 0:
            model.eval()
            with torch.inference_mode():
                vx, vy = get_batch(validation_data)
                v_logits = model(vx)
                val_loss = F.cross_entropy(
                    v_logits.reshape(-1, v_logits.shape[-1]),
                    vy.reshape(-1),
                )
            val_value = float(val_loss.cpu())
            history.append(
                {
                    "step": float(step + 1),
                    "train_loss": float(loss.detach().cpu()),
                    "validation_loss": val_value,
                }
            )
            if val_value < best_val:
                best_val = val_value
                best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("no Transformer checkpoint produced")
    model.load_state_dict(best_state)
    model.eval()

    prompt = "attention "
    generated = [stoi[ch] for ch in prompt if ch in stoi]

    with torch.inference_mode():
        for _ in range(40):
            context = generated[-context_length:]
            x = torch.tensor([context], dtype=torch.long, device=device)
            logits = model(x)
            next_id = int(torch.argmax(logits[0, -1]).cpu())
            generated.append(next_id)

    text = "".join(itos[idx] for idx in generated)

    return {
        "mode": "transformer",
        "device": str(device),
        "steps": steps,
        "context_length": context_length,
        "vocabulary_size": len(chars),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "best_validation_loss": float(best_val),
        "history": history,
        "generated_text": text,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Batch 10 integration lab.")
    parser.add_argument("--mode", choices=["gan", "transformer"], required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/batch10"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.mode == "gan":
        report = run_gan(steps=args.steps, seed=args.seed)
    else:
        report = run_transformer(steps=args.steps, seed=args.seed)

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
