"""Batch 14 integration: diffusion, multimodal alignment and speech CTC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def make_four_mode_points(
    n_samples: int = 1024,
    *,
    seed: int = 42,
) -> np.ndarray:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    rng = np.random.default_rng(seed)
    centers = np.array(
        [
            [-2.0, -2.0],
            [-2.0, 2.0],
            [2.0, -2.0],
            [2.0, 2.0],
        ],
        dtype=np.float32,
    )
    ids = rng.integers(0, len(centers), size=n_samples)

    return (
        centers[ids]
        + rng.normal(0.0, 0.25, size=(n_samples, 2))
    ).astype(np.float32)


def run_diffusion(
    *,
    steps: int = 100,
    seed: int = 42,
) -> dict[str, Any]:
    import torch
    from torch import nn
    from torch.nn import functional as F

    if steps <= 0:
        raise ValueError("steps must be positive")

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = torch.from_numpy(
        make_four_mode_points(1200, seed=seed)
    ).to(device)

    diffusion_steps = 30
    betas = torch.linspace(
        1e-4,
        2e-2,
        diffusion_steps,
        device=device,
    )
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    class NoisePredictor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(3, 64),
                nn.SiLU(),
                nn.Linear(64, 64),
                nn.SiLU(),
                nn.Linear(64, 2),
            )

        def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
        ) -> torch.Tensor:
            t_feature = (
                t.float()
                / max(1, diffusion_steps - 1)
            )[:, None]
            return self.net(
                torch.cat([x, t_feature], dim=1)
            )

    model = NoisePredictor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)

    losses = []

    for _ in range(steps):
        indices = torch.randint(
            0,
            len(data),
            (128,),
            device=device,
        )
        x0 = data[indices]

        t = torch.randint(
            0,
            diffusion_steps,
            (len(x0),),
            device=device,
        )
        epsilon = torch.randn_like(x0)
        alpha_bar = alpha_bars[t][:, None]

        xt = (
            torch.sqrt(alpha_bar) * x0
            + torch.sqrt(1.0 - alpha_bar) * epsilon
        )

        predicted = model(xt, t)
        loss = F.mse_loss(predicted, epsilon)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.detach().cpu()))

    model.eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 1)

    with torch.inference_mode():
        x = torch.randn(
            256,
            2,
            generator=generator,
            device=device,
        )

        for t_index in range(diffusion_steps - 1, -1, -1):
            t = torch.full(
                (len(x),),
                t_index,
                dtype=torch.long,
                device=device,
            )
            eps_hat = model(x, t)

            alpha = alphas[t_index]
            alpha_bar = alpha_bars[t_index]
            beta = betas[t_index]

            mean = (
                x
                - beta
                / torch.sqrt(1.0 - alpha_bar)
                * eps_hat
            ) / torch.sqrt(alpha)

            if t_index > 0:
                previous_bar = alpha_bars[t_index - 1]
                posterior_variance = (
                    beta
                    * (1.0 - previous_bar)
                    / (1.0 - alpha_bar)
                )
                noise = torch.randn(
                    x.shape,
                    generator=generator,
                    device=device,
                )
                x = mean + torch.sqrt(
                    posterior_variance.clamp_min(1e-12)
                ) * noise
            else:
                x = mean

        generated = x.cpu().numpy()

    return {
        "mode": "diffusion",
        "device": str(device),
        "steps": steps,
        "diffusion_steps": diffusion_steps,
        "parameter_count": int(
            sum(p.numel() for p in model.parameters())
        ),
        "loss_first": float(losses[0]),
        "loss_last": float(losses[-1]),
        "generated_mean": generated.mean(axis=0).tolist(),
        "generated_covariance": np.cov(
            generated,
            rowvar=False,
        ).tolist(),
    }


def make_paired_modalities(
    n_samples: int = 512,
    *,
    latent_dim: int = 6,
    observed_dim: int = 12,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    latent = rng.normal(
        size=(n_samples, latent_dim),
    )
    image_projection = rng.normal(
        size=(latent_dim, observed_dim),
    )
    text_projection = rng.normal(
        size=(latent_dim, observed_dim),
    )

    image = (
        latent @ image_projection
        + rng.normal(
            0.0,
            0.12,
            size=(n_samples, observed_dim),
        )
    )
    text = (
        latent @ text_projection
        + rng.normal(
            0.0,
            0.12,
            size=(n_samples, observed_dim),
        )
    )

    return image.astype(np.float32), text.astype(np.float32)


def run_multimodal(
    *,
    steps: int = 100,
    seed: int = 42,
) -> dict[str, Any]:
    import torch
    from torch import nn
    from torch.nn import functional as F

    if steps <= 0:
        raise ValueError("steps must be positive")

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_np, text_np = make_paired_modalities(seed=seed)

    split = 400
    image_train = torch.from_numpy(image_np[:split]).to(device)
    text_train = torch.from_numpy(text_np[:split]).to(device)
    image_test = torch.from_numpy(image_np[split:]).to(device)
    text_test = torch.from_numpy(text_np[split:]).to(device)

    class Encoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(12, 32),
                nn.GELU(),
                nn.Linear(32, 16),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return F.normalize(self.net(x), dim=-1)

    image_encoder = Encoder().to(device)
    text_encoder = Encoder().to(device)

    optimizer = torch.optim.AdamW(
        list(image_encoder.parameters())
        + list(text_encoder.parameters()),
        lr=3e-3,
    )

    batch_size = 64
    losses = []

    for _ in range(steps):
        ids = torch.randint(
            0,
            split,
            (batch_size,),
            device=device,
        )

        image_z = image_encoder(image_train[ids])
        text_z = text_encoder(text_train[ids])

        logits = image_z @ text_z.T / 0.07
        targets = torch.arange(batch_size, device=device)

        loss = 0.5 * (
            F.cross_entropy(logits, targets)
            + F.cross_entropy(logits.T, targets)
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.detach().cpu()))

    image_encoder.eval()
    text_encoder.eval()

    with torch.inference_mode():
        image_z = image_encoder(image_test)
        text_z = text_encoder(text_test)
        similarity = image_z @ text_z.T

        image_to_text = (
            torch.argmax(similarity, dim=1)
            == torch.arange(
                len(image_test),
                device=device,
            )
        ).float().mean()

        text_to_image = (
            torch.argmax(similarity.T, dim=1)
            == torch.arange(
                len(text_test),
                device=device,
            )
        ).float().mean()

    return {
        "mode": "multimodal",
        "device": str(device),
        "steps": steps,
        "parameter_count": int(
            sum(p.numel() for p in image_encoder.parameters())
            + sum(p.numel() for p in text_encoder.parameters())
        ),
        "loss_first": float(losses[0]),
        "loss_last": float(losses[-1]),
        "image_to_text_top1": float(image_to_text.cpu()),
        "text_to_image_top1": float(text_to_image.cpu()),
    }


def synthesize_token_tones(
    token_ids: list[int],
    *,
    sample_rate: int = 8000,
    token_seconds: float = 0.08,
) -> np.ndarray:
    if sample_rate <= 0 or token_seconds <= 0:
        raise ValueError("invalid audio configuration")
    if not token_ids:
        raise ValueError("token_ids must be non-empty")

    frequencies = {
        1: 440.0,
        2: 660.0,
        3: 880.0,
    }

    sample_count = int(sample_rate * token_seconds)
    time = np.arange(sample_count) / sample_rate
    chunks = []

    for token_id in token_ids:
        if token_id not in frequencies:
            raise ValueError("unsupported synthetic token id")

        tone = np.sin(
            2.0
            * np.pi
            * frequencies[token_id]
            * time
        )
        envelope = np.hanning(sample_count)
        chunks.append((tone * envelope).astype(np.float32))

    return np.concatenate(chunks)


def run_speech_ctc(
    *,
    steps: int = 20,
    seed: int = 42,
) -> dict[str, Any]:
    import torch
    from torch import nn
    from torch.nn import functional as F

    if steps <= 0:
        raise ValueError("steps must be positive")

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_rate = 8000
    n_fft = 128
    hop = 64
    window = torch.hann_window(n_fft, device=device)

    sequences = []
    waveforms = []

    for _ in range(48):
        tokens = [
            int(rng.integers(1, 4)),
            int(rng.integers(1, 4)),
        ]
        sequences.append(tokens)
        waveforms.append(
            synthesize_token_tones(
                tokens,
                sample_rate=sample_rate,
            )
        )

    waveform = torch.from_numpy(
        np.stack(waveforms)
    ).to(device)

    spectrum = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        return_complex=True,
    )

    features = torch.log1p(
        spectrum.abs().pow(2)
    ).transpose(1, 2)

    class AcousticModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input = nn.Linear(
                n_fft // 2 + 1,
                24,
            )
            self.gru = nn.GRU(
                24,
                24,
                batch_first=True,
                bidirectional=True,
            )
            self.head = nn.Linear(48, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = F.gelu(self.input(x))
            x, _ = self.gru(x)
            return self.head(x)

    model = AcousticModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    criterion = nn.CTCLoss(
        blank=0,
        zero_infinity=True,
    )

    targets = torch.tensor(
        sequences,
        dtype=torch.long,
        device=device,
    )
    target_lengths = torch.full(
        (len(sequences),),
        2,
        dtype=torch.long,
        device=device,
    )
    input_lengths = torch.full(
        (len(sequences),),
        features.shape[1],
        dtype=torch.long,
        device=device,
    )

    losses = []

    for _ in range(steps):
        model.train()
        logits = model(features)

        log_probs = F.log_softmax(
            logits,
            dim=-1,
        ).transpose(0, 1)

        loss = criterion(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.detach().cpu()))

    model.eval()

    with torch.inference_mode():
        logits = model(features[:1])
        path = torch.argmax(
            logits[0],
            dim=-1,
        ).cpu().tolist()

    collapsed = []
    previous = None

    for token in path:
        if token != previous and token != 0:
            collapsed.append(int(token))
        previous = token

    return {
        "mode": "speech",
        "device": str(device),
        "steps": steps,
        "feature_shape": list(features.shape),
        "parameter_count": int(
            sum(p.numel() for p in model.parameters())
        ),
        "loss_first": float(losses[0]),
        "loss_last": float(losses[-1]),
        "example_target": sequences[0],
        "example_greedy_decode": collapsed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Batch 14 diffusion/multimodal/speech lab."
    )
    parser.add_argument(
        "--mode",
        choices=["diffusion", "multimodal", "speech"],
        required=True,
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/batch14"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.mode == "diffusion":
        report = run_diffusion(
            steps=args.steps,
            seed=args.seed,
        )
    elif args.mode == "multimodal":
        report = run_multimodal(
            steps=args.steps,
            seed=args.seed,
        )
    else:
        report = run_speech_ctc(
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
