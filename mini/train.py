"""Single-GPU pretrain / assistant-only SFT with portable optimizer-boundary saves."""
import argparse
import concurrent.futures
import json
import os
import signal
import time
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from .common import read_json, sha256, validate_config, write_json, wsd
from .data import Blocks
from .engine import (build_model, evaluate, load_checkpoint, make_optimizer,
                     restore_rng, resume_signature, save_checkpoint, seed_all, update)
from .hub import upload


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--decay-data", help="Optional separately prepared L3 corpus")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--resume", help="Complete checkpoint directory")
    group.add_argument("--init", help="Pretrained checkpoint; model ONLY, new SFT optimizer")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--max-seconds", type=int, default=1800, help="Per-invocation training timer, not full-run schedule")
    p.add_argument("--max-steps", type=int, default=0, help="Per invocation; 0 means timer/token limit only")
    p.add_argument("--hub-repo")
    p.add_argument("--hub-prefix", default="pretrain")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--micro-batch", type=int)
    p.add_argument("--overfit-batch", action="store_true", help="Debug only; incompatible with resume, HF uploads")
    args = p.parse_args(argv)
    if args.max_seconds <= 0 or args.max_steps < 0:
        p.error("max-seconds must be positive and max-steps nonnegative")
    if args.overfit_batch and (args.resume or args.hub_repo):
        p.error("Overfit diagnostic cannot resume or upload")
    cfg = read_json(args.config)
    t = cfg["train"]
    if args.micro_batch is not None:
        t["micro_batch"] = args.micro_batch
    if args.no_compile:
        t["compile"] = False
    validate_config(cfg)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA GPU with BF16 support required; use --device cpu for smoke tests")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Expose one GPU with CUDA_VISIBLE_DEVICES; this trainer is single-GPU")
        torch.set_float32_matmul_precision("high")
    else:
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "latest.json").exists() and not args.resume:
        raise ValueError("Output already has a run. Use --resume or a NEW --out.")
    token_path = Path(args.tokenizer) / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(token_path))
    if tokenizer.get_vocab_size() != cfg["model"]["vocab_size"]:
        raise ValueError("Tokenizer vocab and model vocab differ")
    token_hash = sha256(token_path)
    readers = {"train": Blocks(args.data, "train", t["seq_len"], t["seed"])}
    val = Blocks(args.data, "val", t["seq_len"], 0)
    if args.decay_data:
        readers["decay"] = Blocks(args.decay_data, "train", t["seq_len"], t["seed"] + 1)
        if readers["decay"].meta["kind"] != "pretrain" or readers["train"].meta["kind"] != "pretrain":
            raise ValueError("decay-data is only supported for pretraining")
    for reader in [*readers.values(), val]:
        if reader.meta["tokenizer_sha256"] != token_hash:
            raise ValueError("Prepared data uses a different tokenizer")
    if readers["train"].meta["kind"] == "sft" and not (args.init or args.resume):
        raise ValueError("SFT needs --init from pretraining or --resume from SFT")
    seed_all(t["seed"])
    model = build_model(cfg["model"], device)
    optimizer = make_optimizer(model, t, device)
    progress = {"step": 0, "tokens": 0, "supervised_tokens": 0}
    mix_rng = np.random.default_rng(t["seed"] + 101)
    if args.resume or args.init:
        state = load_checkpoint(args.resume or args.init)
        if state["config"]["model"] != cfg["model"] or state["tokenizer_sha256"] != token_hash:
            raise ValueError("Checkpoint architecture/tokenizer mismatch")
        model.load_state_dict(state["model"])
        if args.resume:
            if (out / "latest.json").exists():
                latest = out / read_json(out / "latest.json")["directory"]
                if latest.resolve() != Path(args.resume).resolve():
                    raise ValueError("To resume an older/different checkpoint, choose a NEW --out directory")
            if resume_signature(state["config"]) != resume_signature(cfg):
                raise ValueError("Training recipe changed. Resume requires fixed schedule/global batch.")
            if state["readers"].keys() != readers.keys():
                raise ValueError("Resume needs the same data sources")
            optimizer.load_state_dict(state["optimizer"])
            for k, reader in readers.items():
                reader.load_state_dict(state["readers"][k])
            progress = state["progress"]
            mix_rng.bit_generator.state = state["mix_rng"]
            if state["versions"]["litgpt"] != "0.5.9":
                raise ValueError("Use the same pinned LitGPT version that wrote this checkpoint")
        # Restore RNG AFTER optional torch.compile creation below.
    run_model = torch.compile(model) if t["compile"] and device.type == "cuda" else model
    if args.resume:
        restore_rng(state["rng"])
    if args.resume or args.init:
        del state
    print(json.dumps({"params": sum(p.numel() for p in model.parameters()),
                      "device": str(device), "starting": progress,
                      "available_train_blocks": readers["train"].n}), flush=True)
    wb = None
    if os.environ.get("MINI_WANDB") == "1":
        try:
            import wandb
            id_path = out / "wandb.json"
            run_id = read_json(id_path)["id"] if id_path.exists() else wandb.util.generate_id()
            wb = wandb.init(project=os.environ.get("WANDB_PROJECT", "mini-123m"),
                            id=run_id, resume="allow", config=cfg)
            write_json(id_path, {"id": wb.id})
        except Exception as exc:
            print(f"W&B disabled: {exc}", flush=True)
            wb = None
    stop = False

    def request_stop(*_):
        nonlocal stop
        stop = True
        print("Stop requested; finishing current optimizer update before checkpoint.", flush=True)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = None
    start = time.monotonic()
    last_save = last_hub = window_time = start
    window_tokens = 0
    starting_step = progress["step"]
    batch_tokens = t["global_batch"] * t["seq_len"]
    accum = t["global_batch"] // t["micro_batch"]
    fixed = None
    if args.overfit_batch:
        fixed = [readers["train"].next_numpy(t["micro_batch"]) for _ in range(accum)]

    def checkpoint():
        if args.overfit_batch:
            return None
        folder = save_checkpoint(out, model, optimizer, cfg, token_path, readers, progress.copy(), mix_rng)
        # Modal hook is optional. No Modal import on other providers.
        if os.environ.get("MINI_MODAL_VOLUME"):
            import modal
            modal.Volume.from_name(os.environ["MINI_MODAL_VOLUME"]).commit()
        print(f"CHECKPOINT {folder}", flush=True)
        return folder

    try:
        while progress["tokens"] + batch_tokens <= t["total_tokens"]:
            if stop or time.monotonic() - start >= args.max_seconds:
                break
            if args.max_steps and progress["step"] - starting_step >= args.max_steps:
                break
            batches = []
            for i in range(accum):
                use_decay = ("decay" in readers and
                    progress["tokens"] >= t["total_tokens"] * (1 - t["decay_fraction"]) and
                    mix_rng.random() < t["decay_mix"])
                source = readers["decay" if use_decay else "train"]
                batches.append(fixed[i] if fixed is not None else source.next_numpy(t["micro_batch"]))
            lr = wsd(progress["tokens"] + batch_tokens, t)
            loss, grad, supervised = update(run_model, optimizer, batches, lr, device, t["grad_clip"])
            progress["tokens"] += batch_tokens
            progress["supervised_tokens"] += supervised
            progress["step"] += 1
            window_tokens += batch_tokens
            if progress["step"] % t["log_steps"] == 0:
                now = time.monotonic()
                metrics = {**progress, "loss": loss, "lr": lr, "grad_norm": grad,
                           "tokens_per_second": window_tokens / max(1e-6, now - window_time),
                           "train_epoch": readers["train"].cursor / readers["train"].n}
                print(json.dumps(metrics), flush=True)
                if wb:
                    wb.log({k: v for k, v in metrics.items() if k != "step"},
                           step=progress["step"])
                with open(out / "metrics.jsonl", "a") as f:
                    f.write(json.dumps(metrics) + "\n")
                window_tokens, window_time = 0, now
            if progress["step"] % t["eval_steps"] == 0:
                val_loss = evaluate(run_model, val, t["micro_batch"], t["eval_batches"], device)
                print(json.dumps({"step": progress["step"], "val_loss": val_loss}), flush=True)
                if wb:
                    wb.log({"val_loss": val_loss}, step=progress["step"])
            now = time.monotonic()
            due_hub = bool(args.hub_repo and now - last_hub >= t["hub_seconds"])
            if now - last_save >= t["save_seconds"] or due_hub:
                folder = checkpoint()
                last_save = now
                if due_hub and (future is None or future.done()):
                    if future:
                        future.result()  # surface failed uploads; don't claim success
                    future = pool.submit(upload, folder, args.hub_repo, args.hub_prefix)
                    last_hub = now
        folder = checkpoint()
        if future:
            future.result()
        # Final synchronous upload confirms durability before reporting success.
        if args.hub_repo and folder:
            upload(folder, args.hub_repo, args.hub_prefix)
        print(json.dumps({"finished": progress, "checkpoint": str(folder)}), flush=True)
        return folder
    finally:
        pool.shutdown(wait=True)
        if wb:
            wb.finish()


if __name__ == "__main__":
    main()
