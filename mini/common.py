import hashlib
import json
import os
from pathlib import Path

SPECIAL = ["<|pad|>", "<|eos|>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def holdout(text):
    # Exact duplicate documents always go to the same split, before packing.
    # Only a bounded prefix is normalized: full-document str.split() is O(len)
    # Python work that dominates packing at corpus scale.
    digest = hashlib.sha256(" ".join(text[:4096].split()).encode()).digest()
    return int.from_bytes(digest[:8], "big") % 100 == 0


def wsd(tokens, cfg):
    total = cfg["total_tokens"]
    warm = max(1, int(total * cfg["warmup_fraction"]))
    decay_start = int(total * (1 - cfg["decay_fraction"]))
    if tokens < warm:
        return cfg["lr"] * tokens / warm
    if tokens < decay_start:
        return cfg["lr"]
    frac = min(1.0, max(0.0, (tokens - decay_start) / max(1, total - decay_start)))
    return cfg["lr"] + frac * (cfg["min_lr"] - cfg["lr"])


def validate_config(cfg):
    m, t = cfg["model"], cfg["train"]
    if t["micro_batch"] <= 0 or t["global_batch"] <= 0 or t["global_batch"] % t["micro_batch"]:
        raise ValueError("global_batch must be a positive multiple of micro_batch")
    if t["seq_len"] > m["block_size"] or t["seq_len"] < 2:
        raise ValueError("Invalid sequence length")
    if not 0 < t["warmup_fraction"] < 1 - t["decay_fraction"] < 1:
        raise ValueError("WSD needs warmup, stable and decay phases")
    if t["total_tokens"] < t["global_batch"] * t["seq_len"]:
        raise ValueError("Token budget is smaller than one optimizer update")
    if not 0 <= t["decay_mix"] <= 1:
        raise ValueError("Invalid decay_mix")
