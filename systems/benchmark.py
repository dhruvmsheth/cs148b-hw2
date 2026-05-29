from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch


@dataclass(frozen=True)
class ModelSpec:
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


MODEL_SPECS: dict[str, ModelSpec] = {
    "small": ModelSpec(d_model=512, d_ff=2048, num_layers=8, num_heads=8),
    "medium": ModelSpec(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "large": ModelSpec(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
}


@dataclass(frozen=True)
class BenchmarkConfig:
    model_size: str
    context_length: int = 128
    batch_size: int = 4
    vocab_size: int = 10_000
    warmup_steps: int = 5
    measure_steps: int = 10
    mode: Literal["forward", "forward-backward", "train-step"] = "forward"
    use_bf16: bool = False
    use_memory_profiler: bool = False
    compile_model: bool = False
    output_dir: Path = Path("artifacts")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark and profile the Basics transformer.")
    parser.add_argument("--model-size", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--mode", choices=["forward", "forward-backward", "train-step"], default="forward")
    parser.add_argument("--use-bf16", action="store_true")
    parser.add_argument("--use-memory-profiler", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    return parser


def build_model(config: BenchmarkConfig) -> torch.nn.Module:
    """Instantiate the staff Basics transformer for the requested model size."""
    from basics.model import BasicsTransformerLM

    spec = MODEL_SPECS[config.model_size]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BasicsTransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=spec.d_model,
        num_layers=spec.num_layers,
        num_heads=spec.num_heads,
        d_ff=spec.d_ff,
        rope_theta=10000.0,
    ).to(device)
    return model


def make_random_batch(config: BenchmarkConfig, device: torch.device) -> torch.Tensor:
    """Construct a random token batch for benchmarking and profiling."""
    return torch.randint(0, config.vocab_size, (config.batch_size, config.context_length), device=device)


def run_single_step(
    model: torch.nn.Module,
    batch: torch.Tensor,
    mode: Literal["forward", "forward-backward", "train-step"],
    autocast_context,
) -> None:
    """Execute one benchmark step and synchronize CUDA before returning."""
    if mode == "forward":
        with autocast_context:
            with torch.no_grad():
                _ = model(batch)
    elif mode == "forward-backward":
        with autocast_context:
            logits = model(batch)
            loss = logits.sum()
        loss.backward()
    elif mode == "train-step":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        optimizer.zero_grad()
        with autocast_context:
            logits = model(batch)
            loss = logits.sum()
        loss.backward()
        optimizer.step()

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_model(config: BenchmarkConfig) -> dict[str, float]:
    """Run warmup steps followed by timed measurement steps."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config)

    if config.compile_model:
        model = torch.compile(model)

    batch = make_random_batch(config, device)
    autocast_ctx = make_autocast_context(config.use_bf16)

    maybe_start_memory_history(config.use_memory_profiler)

    for _ in range(config.warmup_steps):
        run_single_step(model, batch, config.mode, autocast_ctx)

    start = time.perf_counter()
    for _ in range(config.measure_steps):
        run_single_step(model, batch, config.mode, autocast_ctx)
    end = time.perf_counter()

    maybe_dump_memory_snapshot(
        config.use_memory_profiler,
        config.output_dir / f"memory_{config.model_size}_{config.mode}.pickle",
    )

    elapsed = end - start
    avg_ms = (elapsed / config.measure_steps) * 1000
    print(f"[{config.model_size}] mode={config.mode} bf16={config.use_bf16} compile={config.compile_model}: {avg_ms:.2f} ms/step")
    return {"avg_ms_per_step": avg_ms, "total_s": elapsed}


def annotated_scaled_dot_product_attention(*args, **kwargs):
    """Optional NVTX-annotated attention path for Nsight Systems profiling."""
    import torch.cuda.nvtx as nvtx
    from basics.model import scaled_dot_product_attention

    nvtx.range_push("scaled_dot_product_attention")
    result = scaled_dot_product_attention(*args, **kwargs)
    nvtx.range_pop()
    return result


def maybe_start_memory_history(enabled: bool) -> None:
    if enabled:
        torch.cuda.memory._record_memory_history(max_entries=100_000)


def maybe_dump_memory_snapshot(enabled: bool, output_path: Path) -> None:
    if enabled:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.cuda.memory._dump_snapshot(str(output_path))
        torch.cuda.memory._record_memory_history(enabled=None)


def make_autocast_context(use_bf16: bool):
    if use_bf16:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def main() -> None:
    args = build_argparser().parse_args()
    config = BenchmarkConfig(
        model_size=args.model_size,
        context_length=args.context_length,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        mode=args.mode,
        use_bf16=args.use_bf16,
        use_memory_profiler=args.use_memory_profiler,
        compile_model=args.compile_model,
        output_dir=args.output_dir,
    )
    benchmark_model(config)


if __name__ == "__main__":
    main()
