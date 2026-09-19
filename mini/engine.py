"""Model, loss, checkpoint helpers; no provider-specific training state."""
import copy
import importlib.metadata
import math
import os
import random
import shutil
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from litgpt import Config
from litgpt.model import GPT

from .common import read_json, sha256, write_json


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(config, device):
    model = GPT(Config(**config))
    # Standard initialization, not MiniCPM muP. Residual output projections scaled.
    for module in model.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                torch.nn.init.zeros_(module.bias)
    for name, param in model.named_parameters():
        if name.endswith("proj.weight"):
            torch.nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config["n_layer"]))
    model.lm_head.weight = model.transformer.wte.weight
    return model.to(device)


def make_optimizer(model, cfg, device):
    decay, no_decay = [], []
    for param in model.parameters():  # deduplicates the tied embedding/head
        (decay if param.ndim >= 2 else no_decay).append(param)
    return torch.optim.AdamW([
        {"params": decay, "weight_decay": cfg["weight_decay"]},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=cfg["lr"], betas=tuple(cfg["betas"]), eps=1e-8, fused=device.type == "cuda")


def amp(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def loss_sum(model, x, y):
    logits = model(x)
    return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                           y.reshape(-1), ignore_index=-100, reduction="sum")


def update(model, optimizer, batches, lr, device, grad_clip):
    # Normalize across ALL supervised tokens of the update, including SFT padding.
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


@torch.no_grad()
def evaluate(model, reader, micro_batch, batches, device):
    state = reader.state_dict()
    reader.cursor = 0  # identical validation subset at every evaluation
    was_training = model.training
    model.eval()
    loss, count = 0.0, 0
    try:
        for _ in range(batches):
            x, y = reader.next_numpy(micro_batch)
            count += int((y != -100).sum())
            with amp(device):
                loss += float(loss_sum(model, torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)))
    finally:
        reader.load_state_dict(state)
        model.train(was_training)
    return loss / max(1, count)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count changed; this project supports single-GPU resume")
        torch.cuda.set_rng_state_all(state["cuda"])


def versions():
    result = {"python": sys.version}
    for package in ["torch", "litgpt", "numpy", "tokenizers", "datasets", "huggingface-hub"]:
        result[package] = importlib.metadata.version(package)
    return result


def resume_signature(config):
    c = copy.deepcopy(config)
    for key in ("micro_batch", "compile", "eval_steps", "eval_batches", "save_seconds", "hub_seconds", "log_steps"):
        c["train"].pop(key, None)
    return c


def save_checkpoint(out, model, optimizer, cfg, tokenizer_path, readers, progress, mix_rng):
    out = Path(out)
    name = f"step-{progress['step']:08d}"
    final = out / name
    if final.exists():
        # Only used when final save repeats a periodic save at the same step.
        if not (final / "COMPLETE.json").exists():
            raise ValueError(f"Incomplete existing checkpoint: {final}")
        return final
    temp = out / (name + ".partial")
    temp.mkdir(parents=True, exist_ok=False)
    state = {
        "format": 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "config": cfg, "progress": progress, "rng": rng_state(),
        "mix_rng": mix_rng.bit_generator.state,
        "readers": {k: v.state_dict() for k, v in readers.items()},
        "tokenizer_sha256": sha256(tokenizer_path), "versions": versions(),
    }
    with open(temp / "training.pt", "wb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    shutil.copy2(tokenizer_path, temp / "tokenizer.json")
    write_json(temp / "config.json", cfg)
    write_json(temp / "versions.json", state["versions"])
    write_json(temp / "COMPLETE.json", {"sha256": sha256(temp / "training.pt"), **progress})
    os.rename(temp, final)
    write_json(out / "latest.json", {"directory": name})
    return final


def load_checkpoint(path):
    path = Path(path)
    info = read_json(path / "COMPLETE.json")
    if sha256(path / "training.pt") != info["sha256"]:
        raise ValueError("Checkpoint checksum mismatch")
    # Contains Python/NumPy RNG state. ONLY load your own trusted checkpoints.
    state = torch.load(path / "training.pt", map_location="cpu", weights_only=False)
    if state["format"] != 1 or sha256(path / "tokenizer.json") != state["tokenizer_sha256"]:
        raise ValueError("Checkpoint format/tokenizer mismatch")
    return state
