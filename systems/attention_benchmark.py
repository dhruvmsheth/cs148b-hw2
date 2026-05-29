from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class AttentionBenchmarkConfig:
    head_dims: tuple[int, ...] = (16, 32, 64, 128)
    sequence_lengths: tuple[int, ...] = (64, 128, 256, 512, 1024)
    batch_size: int = 8
    forward_passes: int = 100
    backward_passes: int = 100
    compile_attention: bool = False


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark attention implementations.")
    parser.add_argument("--compile-attention", action="store_true")
    return parser


def iter_benchmark_shapes(config: AttentionBenchmarkConfig) -> Iterable[tuple[int, int]]:
    for head_dim in config.head_dims:
        for sequence_length in config.sequence_lengths:
            yield head_dim, sequence_length


def make_qkv(batch_size: int, sequence_length: int, head_dim: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    """Create random Q, K, and V tensors for the attention benchmark."""
    shape = (batch_size, sequence_length, head_dim)
    q = torch.randn(*shape, device=device, requires_grad=True)
    k = torch.randn(*shape, device=device, requires_grad=True)
    v = torch.randn(*shape, device=device, requires_grad=True)
    return q, k, v


def benchmark_attention_once(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> dict[str, float]:
    """Time the forward and backward pass for a single attention configuration."""
    import math
    device = q.device

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Forward pass timing
    sync()
    t0 = time.perf_counter()
    scale = 1.0 / math.sqrt(q.shape[-1])
    attn_weights = torch.softmax(torch.bmm(q, k.transpose(-2, -1)) * scale, dim=-1)
    out = torch.bmm(attn_weights, v)
    sync()
    fwd_ms = (time.perf_counter() - t0) * 1000

    # Backward pass timing
    sync()
    t0 = time.perf_counter()
    out.sum().backward()
    sync()
    bwd_ms = (time.perf_counter() - t0) * 1000

    return {"forward_ms": fwd_ms, "backward_ms": bwd_ms}


def benchmark_attention_grid(config: AttentionBenchmarkConfig) -> list[dict[str, float | int | str]]:
    """Run the attention benchmark over the Section 2.7 Cartesian product of scales."""
    import math
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def attention_fn(q, k, v):
        scale = 1.0 / math.sqrt(q.shape[-1])
        weights = torch.softmax(torch.bmm(q, k.transpose(-2, -1)) * scale, dim=-1)
        return torch.bmm(weights, v)

    if config.compile_attention:
        attention_fn = torch.compile(attention_fn)

    results = []
    for head_dim, seq_len in iter_benchmark_shapes(config):
        q, k, v = make_qkv(config.batch_size, seq_len, head_dim, device)

        def sync():
            if device.type == "cuda":
                torch.cuda.synchronize()

        # Warmup
        for _ in range(5):
            out = attention_fn(q, k, v)
            out.sum().backward()
            if q.grad is not None:
                q.grad.zero_()
            if k.grad is not None:
                k.grad.zero_()
            if v.grad is not None:
                v.grad.zero_()

        # Forward timing
        sync()
        t0 = time.perf_counter()
        for _ in range(config.forward_passes):
            with torch.no_grad():
                _ = attention_fn(q, k, v)
            sync()
        fwd_total = (time.perf_counter() - t0) * 1000
        fwd_avg_ms = fwd_total / config.forward_passes

        # Backward timing
        sync()
        t0 = time.perf_counter()
        for _ in range(config.backward_passes):
            out = attention_fn(q, k, v)
            out.sum().backward()
            if q.grad is not None:
                q.grad.zero_()
            sync()
        bwd_total = (time.perf_counter() - t0) * 1000
        bwd_avg_ms = bwd_total / config.backward_passes

        row = {
            "head_dim": head_dim,
            "seq_len": seq_len,
            "forward_ms": round(fwd_avg_ms, 4),
            "backward_ms": round(bwd_avg_ms, 4),
        }
        results.append(row)
        print(f"head_dim={head_dim:4d} seq_len={seq_len:5d}: fwd={fwd_avg_ms:.4f}ms bwd={bwd_avg_ms:.4f}ms")

    return results


def main() -> None:
    args = build_argparser().parse_args()
    config = AttentionBenchmarkConfig(compile_attention=args.compile_attention)
    benchmark_attention_grid(config)


if __name__ == "__main__":
    main()
