"""Immutable uint16 shards, deterministic block order, O(1)-size reader state."""
import bisect
import math
from pathlib import Path

import numpy as np

from .common import read_json, sha256


class Blocks:
    def __init__(self, root, split, seq_len, seed=42, verify=True):
        self.root = Path(root)
        self.meta = read_json(self.root / "manifest.json")
        self.fingerprint = sha256(self.root / "manifest.json")
        self.seq_len = seq_len
        self.seed = seed
        self.cursor = 0
        self.arrays, self.masks, sizes = [], [], []
        for entry in self.meta["splits"][split]:
            p = self.root / entry["file"]
            if verify and sha256(p) != entry["sha256"]:
                raise ValueError(f"Corrupt shard: {p}")
            a = np.memmap(p, mode="r", dtype="<u2")
            if self.meta["kind"] == "sft":
                if self.meta["seq_len"] != seq_len:
                    raise ValueError("SFT packing sequence length differs from model training length")
                a = a.reshape(-1, seq_len + 1)
                mp = self.root / entry["mask"]
                if verify and sha256(mp) != entry["mask_sha256"]:
                    raise ValueError(f"Corrupt mask: {mp}")
                mask = np.memmap(mp, mode="r", dtype="u1").reshape(a.shape)
                self.masks.append(mask)
                n = len(a)
            else:
                self.masks.append(None)
                n = max(0, (len(a) - 1) // seq_len)
            self.arrays.append(a)
            sizes.append(n)
        self.ends = np.cumsum(sizes).tolist()
        self.n = sum(sizes)
        if self.n == 0:
            raise ValueError(f"No usable {split} blocks in {root}. Prepare more data.")

    def permuted(self, index):
        # Affine bijection, not a uniform random permutation; changes each epoch.
        epoch, offset = divmod(index, self.n)
        rng = np.random.default_rng(self.seed + epoch)
        a = int(rng.integers(1, max(2, self.n)))
        while math.gcd(a, self.n) != 1:
            a = (a + 1) % self.n or 1
        b = int(rng.integers(self.n))
        return (a * offset + b) % self.n

    def next_numpy(self, batch):
        xs, ys = [], []
        for _ in range(batch):
            idx = self.permuted(self.cursor)
            self.cursor += 1
            shard = bisect.bisect_right(self.ends, idx)
            local = idx - (self.ends[shard - 1] if shard else 0)
            if self.meta["kind"] == "sft":
                row = np.asarray(self.arrays[shard][local], dtype=np.int64)
                target = row[1:].copy()
                target[self.masks[shard][local, 1:] == 0] = -100
            else:
                start = local * self.seq_len
                row = np.asarray(self.arrays[shard][start:start + self.seq_len + 1], dtype=np.int64)
                target = row[1:].copy()
            xs.append(row[:-1])
            ys.append(target)
        return np.stack(xs), np.stack(ys)

    def state_dict(self):
        return {"cursor": self.cursor, "seed": self.seed, "fingerprint": self.fingerprint}

    def load_state_dict(self, state):
        if state["fingerprint"] != self.fingerprint or state["seed"] != self.seed:
            raise ValueError("Data manifest or shuffle seed changed: cannot safely resume")
        self.cursor = state["cursor"]
