"""Bounded full-training benchmark; no checkpoints, uploads, or cloud launches."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .common import read_json, validate_config, write_json
from .data import Blocks
from .engine import (BatchTransfer, amp, build_model, loss_sum, make_loss,
                     make_optimizer, seed_all, update)


def legacy_update(model, optimizer, batches, lr, device, grad_clip):
    """Original engine at commit 2147268, retained only for an honest A/B."""
    count = sum(int((y != -100).sum()) for _, y in batches)
    if count == 0:
        raise ValueError("Batch contains no supervised targets")
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:
        group["lr"] = lr
    total_loss = 0.0
    for x, y in batches:
        x = torch.from_numpy(x).to(device)
        y = torch.from_numpy(y).to(device)
        with amp(device):
            loss = loss_sum(model, x, y)
        (loss / count).backward()
        total_loss += float(loss.detach())
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
    optimizer.step()
    return total_loss / count, float(norm), count


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run(args):
    cfg = read_json(args.config)
    t = cfg["train"]
    t["micro_batch"] = args.micro_batch
    validate_config(cfg)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("A CUDA GPU with BF16 support is required")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Expose exactly one GPU for comparable results")
        torch.set_float32_matmul_precision("high")
    else:
        torch.set_num_threads(2)
    seed_all(t["seed"])
    model = build_model(cfg["model"], device)
    opt = make_optimizer(model, t, device)
    compiled = not args.no_compile and device.type == "cuda"
    transfer = BatchTransfer(device)
    if args.engine == "baseline":
        run_model = torch.compile(model, mode=args.compile_mode) if compiled else model
        step = lambda batches: legacy_update(run_model, opt, batches, t["lr"], device, t["grad_clip"])
    else:
        fn = make_loss(model, compile=compiled, mode=args.compile_mode)
        step = lambda batches: update(model, opt, batches, t["lr"], device, t["grad_clip"],
                                      loss_fn=fn, transfer=transfer)
    accum = t["global_batch"] // t["micro_batch"]
    reader = Blocks(args.data, "train", t["seq_len"], t["seed"]) if args.data else None
    if reader and reader.meta["kind"] != "pretrain":
        raise ValueError("This benchmark compares pretraining; use pretraining data")
    if reader is None:
        rng = np.random.default_rng(t["seed"])
        rows = rng.integers(0, cfg["model"]["vocab_size"],
                            (t["global_batch"], t["seq_len"] + 1), dtype=np.int64)
        fixed = [(rows[i:i + t["micro_batch"], :-1].copy(),
                  rows[i:i + t["micro_batch"], 1:].copy())
                 for i in range(0, t["global_batch"], t["micro_batch"])]

    def batches():
        return [reader.next_numpy(t["micro_batch"]) for _ in range(accum)] if reader else fixed

    start = time.perf_counter()
    for _ in range(args.warmup):
        step(batches())
    synchronize(device)
    warmup_seconds = time.perf_counter() - start
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    supervised = 0
    for _ in range(args.steps):
        loss, norm, count = step(batches())
        supervised += count
    synchronize(device)
    elapsed = time.perf_counter() - start
    result = {
        "engine": args.engine, "device": str(device), "torch": torch.__version__,
        "cuda": torch.version.cuda, "compile": compiled, "compile_mode": args.compile_mode,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "params": sum(p.numel() for p in model.parameters()),
        "micro_batch": t["micro_batch"], "global_batch": t["global_batch"],
        "seq_len": t["seq_len"], "accumulation_steps": accum,
        "data": str(Path(args.data).resolve()) if reader else "synthetic (no disk loading)",
        "warmup_steps": args.warmup, "warmup_seconds": warmup_seconds,
        "measured_steps": args.steps, "measured_seconds": elapsed,
        "tokens_per_second": args.steps * t["global_batch"] * t["seq_len"] / elapsed,
        "supervised_tokens_per_second": supervised / elapsed,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None,
        "loss": loss, "grad_norm": norm,
        "excludes": "model initialization, warmup/compilation, evaluation, checkpointing, uploads",
    }
    if args.profile_dir:
        folder = Path(args.profile_dir)
        folder.mkdir(parents=True, exist_ok=True)
        trace = folder / f"{args.engine}-mb{args.micro_batch}.json"
        if trace.exists():
            raise ValueError(f"Choose a NEW profile directory: {trace} already exists")
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        # Profiling is separate from the throughput window because tracing adds
        # considerable overhead. Includes host loading/transfer and full updates.
        with torch.profiler.profile(activities=activities, record_shapes=True,
                                    profile_memory=True) as prof:
            for _ in range(3):
                step(batches())
                prof.step()
            synchronize(device)
        prof.export_chrome_trace(str(trace))
        result["profile_trace"] = str(trace)
    if args.out:
        write_json(args.out, result)
    print(json.dumps(result), flush=True)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/123m.json")
    p.add_argument("--data", help="Optional real pretraining shards; includes batch reading in timing")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--engine", choices=["baseline", "optimized"], default="optimized")
    p.add_argument("--micro-batch", type=int, default=4)
    p.add_argument("--micro-batches", help="Sweep comma-separated divisors of global batch in fresh processes")
    p.add_argument("--compile-mode", choices=["default", "max-autotune-no-cudagraphs"], default="default")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--out", help="NEW JSON path for results")
    p.add_argument("--profile-dir", help="Optional NEW directory for separate three-update profiler traces")
    args = p.parse_args(argv)
    if args.warmup < 1 or args.steps < 1:
        p.error("warmup and steps must be positive")
    if args.out and Path(args.out).exists():
        p.error("Choose a NEW output path; existing results are not overwritten")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if not args.micro_batches:
        return run(args)
    global_batch = read_json(args.config)["train"]["global_batch"]
    candidates = [int(s) for s in args.micro_batches.split(",")]
    if any(n <= 0 or global_batch % n for n in candidates):
        p.error("Each microbatch must be a positive divisor of global batch")
    results = []
    for micro in candidates:
        command = [sys.executable, "-m", "mini.benchmark", "--config", args.config,
                   "--device", args.device, "--engine", args.engine,
                   "--micro-batch", str(micro), "--compile-mode", args.compile_mode,
                   "--warmup", str(args.warmup), "--steps", str(args.steps)]
        if args.data:
            command += ["--data", args.data]
        if args.no_compile:
            command += ["--no-compile"]
        if args.profile_dir:
            command += ["--profile-dir", args.profile_dir]
        # Fresh process isolates allocator/compile state and recovers after OOM.
        child = subprocess.run(command, capture_output=True, text=True)
        if child.returncode:
            result = {"micro_batch": micro, "failed": True, "error": child.stderr[-4000:]}
        else:
            try:
                result = json.loads(child.stdout.strip().splitlines()[-1])
            except (ValueError, IndexError):
                raise RuntimeError(f"Benchmark child did not return JSON: {child.stdout}")
        results.append(result)
        print(json.dumps(result), flush=True)
    successful = [r for r in results if not r.get("failed")]
    summary = {"results": results, "best": max(successful, key=lambda r: r["tokens_per_second"]) if successful else None}
    if args.out:
        write_json(args.out, summary)
    if not successful:
        raise RuntimeError("All benchmark candidates failed; see diagnostics")
    print(json.dumps({"best": summary["best"]}), flush=True)
    return summary


if __name__ == "__main__":
    main()
