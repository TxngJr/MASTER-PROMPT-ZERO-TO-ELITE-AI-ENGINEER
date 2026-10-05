"""Final local capstone smoke: corpus -> BPE -> random-init LM -> SFT -> DPO."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import io
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


def _corpus() -> tuple[list[str], list[str]]:
    train = [
        "Transformers predict the next token using causal attention.",
        "A tokenizer converts UTF-8 text into integer token identifiers.",
        "Gradient descent updates parameters to reduce an objective.",
        "Validation data measures generalization outside optimizer updates.",
        "RMSNorm and residual connections stabilize Transformer blocks.",
        "Byte pair encoding merges frequent adjacent byte token pairs.",
        "Checkpoint metadata must include configuration and tokenizer identity.",
        "Quantization trades numerical precision for smaller representations.",
    ]
    validation = [
        "Held out documents should not update model parameters.",
        "Perplexity is derived from average token cross entropy.",
    ]
    return train, validation


def build_local_dataset(*, num_merges: int = 32, sequence_length: int = 12) -> dict[str, Any]:
    bpe = _load_module(
        "49-tokenizer-from-scratch/src/byte_bpe.py",
        "batch27_byte_bpe",
    )
    utils = _load_module(
        "80-build-llm-from-scratch/src/capstone_utils.py",
        "batch27_capstone_utils",
    )

    train_docs, validation_docs = _corpus()
    merges = bpe.train_bpe(train_docs, num_merges=num_merges)
    tokenizer = bpe.ByteBPETokenizer(merges)

    train_tokens = tokenizer.encode("\n".join(train_docs))
    validation_tokens = tokenizer.encode("\n".join(validation_docs))

    x_train, y_train = utils.causal_windows(
        train_tokens, sequence_length=sequence_length, stride=sequence_length
    )
    x_val, y_val = utils.causal_windows(
        validation_tokens, sequence_length=sequence_length, stride=sequence_length
    )

    if len(x_train) == 0 or len(x_val) == 0:
        raise RuntimeError("corpus too small for requested sequence length")

    return {
        "tokenizer": tokenizer,
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "sequence_length": sequence_length,
    }


def _build_model(vocab_size: int, *, dim: int = 32, heads: int = 4):
    import torch
    from torch import nn
    from torch.nn import functional as F

    if dim % heads != 0:
        raise ValueError("dim must be divisible by heads")

    class CausalSelfAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.heads = heads
            self.head_dim = dim // heads
            self.qkv = nn.Linear(dim, 3 * dim, bias=False)
            self.out = nn.Linear(dim, dim, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch, time, _ = x.shape
            qkv = self.qkv(x).view(batch, time, 3, heads, self.head_dim)
            q, k, v = qkv.unbind(dim=2)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            attended = F.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=True
            )
            merged = attended.transpose(1, 2).contiguous().view(batch, time, dim)
            return self.out(merged)

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm1 = nn.RMSNorm(dim)
            self.attn = CausalSelfAttention()
            self.norm2 = nn.RMSNorm(dim)
            self.gate = nn.Linear(dim, 2 * dim, bias=False)
            self.up = nn.Linear(dim, 2 * dim, bias=False)
            self.down = nn.Linear(2 * dim, dim, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.attn(self.norm1(x))
            normalized = self.norm2(x)
            return x + self.down(F.silu(self.gate(normalized)) * self.up(normalized))

    class TinyDecoderLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, dim)
            self.block = Block()
            self.norm = nn.RMSNorm(dim)

        def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
            hidden = self.norm(self.block(self.embedding(token_ids)))
            return F.linear(hidden, self.embedding.weight)

    return TinyDecoderLM()


def _completion_logp(model, tokenizer, prompt: str, completion: str, device):
    import torch
    from torch.nn import functional as F

    prompt_ids = tokenizer.encode(prompt)
    completion_ids = tokenizer.encode(completion)
    full = prompt_ids + completion_ids
    if len(full) < 2:
        raise ValueError("sequence too short")

    x = torch.tensor([full[:-1]], dtype=torch.long, device=device)
    targets = torch.tensor(full[1:], dtype=torch.long, device=device)
    logits = model(x)[0]
    log_probs = F.log_softmax(logits, dim=-1)

    start = max(len(prompt_ids) - 1, 0)
    selected = log_probs[start:, :].gather(1, targets[start:].unsqueeze(1))
    return selected.sum()


def run_capstone_smoke(
    *,
    pretrain_steps: int = 2,
    sft_steps: int = 1,
    preference_steps: int = 1,
    seed: int = 7,
) -> dict[str, Any]:
    import torch
    from torch.nn import functional as F

    if min(pretrain_steps, sft_steps, preference_steps) < 0:
        raise ValueError("step counts must be non-negative")

    utils = _load_module(
        "80-build-llm-from-scratch/src/capstone_utils.py",
        "batch27_capstone_utils_runtime",
    )
    built = build_local_dataset()
    tokenizer = built["tokenizer"]

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = _build_model(tokenizer.vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)

    x_train = torch.from_numpy(built["x_train"]).to(device)
    y_train = torch.from_numpy(built["y_train"]).to(device)
    x_val = torch.from_numpy(built["x_val"]).to(device)
    y_val = torch.from_numpy(built["y_val"]).to(device)

    losses: list[float] = []
    for _ in range(pretrain_steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_train)
        loss = F.cross_entropy(
            logits.reshape(-1, tokenizer.vocab_size),
            y_train.reshape(-1),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    model.eval()
    with torch.no_grad():
        val_logits = model(x_val)
        val_loss = F.cross_entropy(
            val_logits.reshape(-1, tokenizer.vocab_size),
            y_val.reshape(-1),
        )

    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = _build_model(tokenizer.vocab_size)
    restored.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True)
    )

    sft_text = (
        "User: Define validation data.\n"
        "Assistant: Validation data estimates generalization without updating parameters."
    )
    sft_tokens = tokenizer.encode(sft_text)
    sft_length = min(12, max(2, len(sft_tokens) - 1))
    sx_np, sy_np = utils.causal_windows(
        sft_tokens, sequence_length=sft_length, stride=sft_length
    )
    if len(sx_np) > 0:
        sx = torch.from_numpy(sx_np).to(device)
        sy = torch.from_numpy(sy_np).to(device)
        for _ in range(sft_steps):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(sx)
            loss = F.cross_entropy(
                logits.reshape(-1, tokenizer.vocab_size),
                sy.reshape(-1),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    reference = deepcopy(model).to(device).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)

    preference_optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    preference_losses: list[float] = []
    prompt = "User: What should validation data do?\nAssistant:"
    chosen = " estimate generalization without training updates."
    rejected = " be used to optimize every parameter."

    for _ in range(preference_steps):
        model.train()
        preference_optimizer.zero_grad(set_to_none=True)
        pc = _completion_logp(model, tokenizer, prompt, chosen, device)
        pr = _completion_logp(model, tokenizer, prompt, rejected, device)
        with torch.no_grad():
            rc = _completion_logp(reference, tokenizer, prompt, chosen, device)
            rr = _completion_logp(reference, tokenizer, prompt, rejected, device)
        margin = (pc - pr) - (rc - rr)
        dpo_loss = -F.logsigmoid(0.1 * margin)
        dpo_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        preference_optimizer.step()
        preference_losses.append(float(dpo_loss.detach().cpu()))

    first_parameter = next(model.parameters()).detach().cpu().numpy()
    q, scale = utils.symmetric_int8_quantize(first_parameter)
    dequantized = utils.symmetric_int8_dequantize(q, scale)
    quantization_mae = float(np.mean(np.abs(first_parameter - dequantized)))

    parameter_count = sum(p.numel() for p in model.parameters())
    weight_bytes_fp32 = utils.model_weight_bytes(parameter_count, bytes_per_parameter=4)
    adam_estimate = utils.adam_training_state_bytes(parameter_count)
    kv_estimate = utils.kv_cache_bytes(
        layers=1,
        batch_size=1,
        sequence_length=built["sequence_length"],
        kv_heads=4,
        head_dim=8,
        bytes_per_element=4,
    )

    model.eval()
    generated = tokenizer.encode("AI ")
    with torch.no_grad():
        for _ in range(8):
            context = generated[-built["sequence_length"] :]
            x = torch.tensor([context], dtype=torch.long, device=device)
            next_id = int(model(x)[0, -1].argmax().item())
            generated.append(next_id)

    return {
        "device": str(device),
        "vocab_size": tokenizer.vocab_size,
        "parameter_count": int(parameter_count),
        "weight_bytes_fp32": int(weight_bytes_fp32),
        "adam_state_estimate_bytes": int(adam_estimate),
        "kv_cache_estimate_bytes": int(kv_estimate),
        "pretrain_steps": pretrain_steps,
        "pretrain_loss_first": losses[0] if losses else None,
        "pretrain_loss_last": losses[-1] if losses else None,
        "validation_loss": float(val_loss.detach().cpu()),
        "checkpoint_tensor_keys": len(restored.state_dict()),
        "sft_steps": sft_steps,
        "preference_steps": preference_steps,
        "preference_loss_last": preference_losses[-1] if preference_losses else None,
        "quantization_mae": quantization_mae,
        "generated_text": tokenizer.decode(generated, errors="replace"),
    }


if __name__ == "__main__":
    for key, value in run_capstone_smoke().items():
        print(f"{key}: {value}")
