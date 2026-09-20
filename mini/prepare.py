"""CPU-only tokenizer training, bounded streaming pretraining/SFT preparation."""
import argparse
import itertools
import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

from .common import SPECIAL, holdout, read_json, sha256, write_json


def records(args):
    if args.local:
        with open(args.local, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
        return
    from datasets import load_dataset
    from huggingface_hub import HfApi
    # Resolve moving main to one immutable dataset revision and record it.
    args.revision = HfApi().dataset_info(args.dataset, revision=args.revision).sha
    ds = load_dataset(args.dataset, args.subset, split=args.split,
                      revision=args.revision, streaming=True)
    yield from ds.shuffle(seed=42, buffer_size=10000)


def texts(args):
    for row in itertools.islice(records(args), args.max_docs):
        text = row.get(args.field)
        if not isinstance(text, str):
            raise ValueError(f"Missing text field {args.field!r}; available: {list(row)}")
        if len(text.strip()) >= 100:
            yield text


def encode_plain(tok, text):
    # Keep scraped text from injecting reserved conversation control tokens.
    for special in SPECIAL:
        text = text.replace(special, special.replace("<|", "< |"))
    return tok.encode(text, add_special_tokens=False).ids


def tokenizer(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=args.vocab_size, min_frequency=2,
        special_tokens=SPECIAL, initial_alphabet=pre_tokenizers.ByteLevel.alphabet())

    def sample():
        chars = 0
        for text in texts(args):
            if holdout(text):
                continue
            yield text
            chars += len(text)
            if chars >= args.max_chars:
                break

    tok.train_from_iterator(sample(), trainer)
    if tok.get_vocab_size() != args.vocab_size:
        raise ValueError("Not enough varied text for requested vocab. Increase tokenizer sample.")
    tok.save(str(out / "tokenizer.json"))
    write_json(out / "provenance.json", vars(args))
    print(f"Frozen tokenizer: {out}; vocab={tok.get_vocab_size()}")


class Writer:
    def __init__(self, out, split, limit=8_000_000, prefix=""):
        self.out, self.split, self.limit, self.prefix = out, split, limit, prefix
        self.parts, self.masks, self.count, self.entries = [], [], 0, []

    def add(self, ids, mask=None):
        self.parts.append(np.asarray(ids, dtype="<u2"))
        if mask is not None:
            self.masks.append(np.asarray(mask, dtype="u1"))
        self.count += len(ids)
        if self.count >= self.limit:
            self.flush()

    def flush(self):
        if not self.parts:
            return
        p = self.out / f"{self.prefix}{self.split}-{len(self.entries):05d}.bin"
        np.concatenate(self.parts).tofile(p)
        entry = {"file": p.name, "tokens": self.count, "sha256": sha256(p)}
        if self.masks:
            mp = p.with_suffix(".mask")
            np.concatenate(self.masks).tofile(mp)
            entry.update(mask=mp.name, mask_sha256=sha256(mp))
        self.entries.append(entry)
        self.parts, self.masks, self.count = [], [], 0


def conversation(tok, messages, seq_len):
    if not messages or messages[-1].get("role") != "assistant":
        return None
    ids, mask = [], []
    expected = "user"
    for i, message in enumerate(messages):
        role, content = message.get("role"), message.get("content")
        if role == "system" and i == 0:
            pass
        elif role == expected:
            expected = "assistant" if role == "user" else "user"
        else:
            return None
        if not isinstance(content, str) or not content.strip():
            return None
        header = [tok.token_to_id(f"<|{role}|>")]
        body = encode_plain(tok, content) + [tok.token_to_id("<|end|>")]
        ids += header + body
        mask += [0] + [int(role == "assistant")] * len(body)
    # Reject, rather than cut a response in half.
    if len(ids) > seq_len + 1 or not any(mask[1:]):
        return None
    padding = seq_len + 1 - len(ids)
    return ids + [tok.token_to_id("<|pad|>")] * padding, mask + [0] * padding


def pack(args):
    """Assistant-only SFT packing (small corpora; streaming is fine).

    With --tag this writes a shard and a ``manifest.<tag>.json`` part (same
    scheme as ``fast_pack``) so several sources can be packed separately and
    combined with ``merge``. Filenames carry the tag prefix, so parts never
    collide.
    """
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=args.tag is not None)
    tok_path = Path(args.tokenizer) / "tokenizer.json"
    tok = Tokenizer.from_file(str(tok_path))
    if tok.get_vocab_size() > 65536:
        raise ValueError("uint16 format only supports vocab <= 65536")
    prefix = f"{args.tag}-" if args.tag else ""
    writers = {s: Writer(out, s, prefix=prefix) for s in ("train", "val")}
    counts = {"train": 0, "val": 0, "skipped": 0}
    seen = set()
    for row in itertools.islice(records(args), args.max_docs):
        messages = row.get(args.field)
        if not isinstance(messages, list):
            raise ValueError(f"Expected messages list in {args.field!r}; got {list(row)}")
        # Split/group by first user prompt, not assistant text.
        prompt = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
        key = " ".join(prompt.split())
        if key in seen:
            counts["skipped"] += 1
            continue
        seen.add(key)
        item = conversation(tok, messages, args.seq_len)
        if item is None:
            counts["skipped"] += 1
            continue
        split = "val" if holdout(prompt) else "train"
        writers[split].add(*item)
        counts[split] += 1
        if counts["train"] >= args.max_examples and counts["val"] > 0:
            break
    for writer in writers.values():
        writer.flush()
    if not all(w.entries for w in writers.values()):
        raise ValueError("Missing train/val data. Increase source sample and retry with a NEW output path.")
    write_json(out / (f"manifest.{args.tag}.json" if args.tag else "manifest.json"), {
        "version": 1, "kind": "sft",
        "seq_len": args.seq_len, "vocab_size": tok.get_vocab_size(),
        "tokenizer_sha256": sha256(tok_path), "counts": counts,
        "source": vars(args), "splits": {k: v.entries for k, v in writers.items()}
    })
    print(json.dumps({**counts, "tag": args.tag}), flush=True)


def resolve_revision(args):
    if not args.revision:
        from huggingface_hub import HfApi
        args.revision = HfApi(token=os.environ.get("HF_TOKEN")).dataset_info(args.dataset).sha
    return args.revision


def resolve_split_files(args):
    """Repo-relative parquet filenames for the requested (dataset, split).

    The datasets-server index reports synthetic paths, so resolve from the repo
    tree: pick the one source directory whose name ends in ``_<split>`` (e.g.
    ``data/ultrafineweb_en``), or use an explicit ``--source-dir``.
    """
    from huggingface_hub import HfApi
    info = HfApi(token=os.environ.get("HF_TOKEN")).dataset_info(args.dataset, revision=args.revision)
    args.revision = info.sha
    files = [s.rfilename for s in info.siblings if s.rfilename.endswith(".parquet")]
    dirs = sorted({f.rsplit("/", 1)[0] for f in files})
    if args.source_dir:
        matches = [d for d in dirs if d == args.source_dir or d.endswith("/" + args.source_dir)]
    else:
        matches = [d for d in dirs if d.rsplit("/", 1)[-1].endswith("_" + args.split)]
        if args.subset:
            matches = [d for d in matches if args.subset in d]
    if len(matches) != 1:
        raise ValueError(f"Cannot resolve split {args.split!r}; candidate dirs: {matches or dirs}")
    return sorted(f for f in files if f.rsplit("/", 1)[0] == matches[0])


def download_file(repo, filename, revision, local_dir):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset",
                           revision=revision, local_dir=local_dir)


def pack_file(path, tok, field, eos_id, writers, counts, args):
    """Tokenize one parquet shard in batches; return True when the token budget is met."""
    import pyarrow.parquet as pq
    table = pq.read_table(path, columns=[field])
    column = table.column(field)
    for start in range(0, table.num_rows, args.read_batch):
        raw = column.slice(start, args.read_batch).to_pylist()
        originals, prepared = [], []
        for text in raw:
            if not isinstance(text, str) or len(text.strip()) < 100:
                continue
            clean = text
            if "<|" in clean:
                for special in SPECIAL:
                    clean = clean.replace(special, special.replace("<|", "< |"))
            originals.append(text)
            prepared.append(clean)
        if not prepared:
            continue
        for original, ids in zip(originals, tok.encode_batch(prepared)):
            split = "val" if holdout(original) else "train"
            if split == "val" and counts["val"] >= args.val_tokens:
                continue
            ids = ids.ids
            ids.append(eos_id)
            writers[split].add(ids)
            counts[split] += len(ids)
        if counts["train"] >= args.max_tokens and counts["val"] >= args.seq_len + 1:
            return True
    return False


def fast_pack(args):
    """Parallel pretraining pack: threaded shard download + batched Rust tokenization."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=args.shard_count > 1)
    tok_path = Path(args.tokenizer) / "tokenizer.json"
    tok = Tokenizer.from_file(str(tok_path))
    if tok.get_vocab_size() > 65536:
        raise ValueError("uint16 format only supports vocab <= 65536")
    resolve_revision(args)
    files = resolve_split_files(args)
    if args.shard_count > 1:
        cut = len(files)
        start = cut * args.shard_index // args.shard_count
        end = cut * (args.shard_index + 1) // args.shard_count
        files = files[start:end]
    elif args.file_start or args.file_end:
        files = files[args.file_start:(args.file_end or None)]
    if not files:
        raise ValueError("No source files selected")
    prefix = f"{args.tag}-" if args.tag else ""
    writers = {s: Writer(out, s, limit=args.shard_tokens, prefix=prefix) for s in ("train", "val")}
    counts = {"train": 0, "val": 0, "skipped": 0}
    eos_id = tok.token_to_id("<|eos|>")
    cache = args.cache_dir or "/tmp/mini-llm-data"
    os.makedirs(cache, exist_ok=True)
    started = time.monotonic()
    done = False
    pool = ThreadPoolExecutor(max_workers=args.download_workers)
    pending, stream = {}, iter(files)
    for _ in range(args.download_workers):
        try:
            name = next(stream)
        except StopIteration:
            break
        pending[pool.submit(download_file, args.dataset, name, args.revision, cache)] = name
    try:
        while pending and not done:
            finished, _ = wait(list(pending), return_when=FIRST_COMPLETED)
            for fut in finished:
                pending.pop(fut)
                path = fut.result()
                done = pack_file(path, tok, args.field, eos_id, writers, counts, args)
                try:
                    os.remove(path)
                except OSError:
                    pass
                if done:
                    break
                try:
                    name = next(stream)
                except StopIteration:
                    continue
                pending[pool.submit(download_file, args.dataset, name, args.revision, cache)] = name
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    for writer in writers.values():
        writer.flush()
    if not all(w.entries for w in writers.values()):
        raise ValueError("Missing train/val data. Increase source sample and retry with a NEW output path.")
    write_json(out / (f"manifest.{args.tag}.json" if args.tag else "manifest.json"), {
        "version": 1, "kind": "pretrain",
        "seq_len": args.seq_len, "vocab_size": tok.get_vocab_size(),
        "tokenizer_sha256": sha256(tok_path), "counts": counts,
        "source": vars(args), "splits": {k: v.entries for k, v in writers.items()}
    })
    print(json.dumps({**counts, "files": len(files),
                      "seconds": round(time.monotonic() - started, 1)}), flush=True)


def merge(args):
    """Combine parallel pack shards (manifest.<tag>.json parts) into manifest.json."""
    out = Path(args.out)
    parts = sorted(out.glob("manifest.*.json"))
    if not parts:
        raise ValueError(f"No manifest.<tag>.json parts found in {out}")
    base = read_json(parts[0])
    splits = {k: [] for k in base["splits"]}
    counts = {"train": 0, "val": 0, "skipped": 0}
    seen = set()
    for part in parts:
        meta = read_json(part)
        if meta["kind"] != base["kind"] or meta["tokenizer_sha256"] != base["tokenizer_sha256"]:
            raise ValueError(f"Shard {part.name} is not compatible with the others")
        for split, entries in meta["splits"].items():
            for entry in entries:
                if entry["file"] in seen:
                    raise ValueError(f"Duplicate shard filename across parts: {entry['file']}")
                seen.add(entry["file"])
                splits[split].append(entry)
        for key in counts:
            counts[key] += meta["counts"].get(key, 0)
    write_json(out / "manifest.json", {
        "version": 1, "kind": base["kind"],
        "seq_len": base["seq_len"], "vocab_size": base["vocab_size"],
        "tokenizer_sha256": base["tokenizer_sha256"], "counts": counts,
        "source": {"merged": [p.name for p in parts]},
        "splits": splits,
    })
    for part in parts:
        part.unlink()
    print(json.dumps(counts), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["tokenizer", "pack", "sft", "merge"])
    p.add_argument("--dataset", default="openbmb/Ultra-FineWeb")
    p.add_argument("--subset", default=None)
    p.add_argument("--split", default="en")
    p.add_argument("--field", default="content")
    p.add_argument("--source-dir", help="Explicit repo directory of parquet shards")
    p.add_argument("--revision")
    p.add_argument("--local", help="Local JSONL instead of HF")
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer")
    p.add_argument("--vocab-size", type=int, default=32768)
    p.add_argument("--max-chars", type=int, default=100_000_000)
    p.add_argument("--max-docs", type=int, default=5_000_000)
    p.add_argument("--max-tokens", type=int, default=100_000_000)
    p.add_argument("--val-tokens", type=int, default=2_000_000)
    p.add_argument("--max-examples", type=int, default=20000)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--download-workers", type=int, default=8,
                   help="Concurrent parquet downloads (prefetch depth)")
    p.add_argument("--read-batch", type=int, default=2048,
                   help="Rows per tokenizer batch (Rust encode_batch parallelism)")
    p.add_argument("--shard-tokens", type=int, default=64_000_000,
                   help="Tokens per output .bin shard (~128MB at uint16)")
    p.add_argument("--file-start", type=int, default=0, help="First source file index")
    p.add_argument("--file-end", type=int, default=0, help="Exclusive end index; 0 means all")
    p.add_argument("--shard-index", type=int, default=0, help="Index of this parallel shard")
    p.add_argument("--shard-count", type=int, default=1, help="Number of parallel shards")
    p.add_argument("--tag", help="Shard tag; prefixes output files and names the manifest part")
    p.add_argument("--cache-dir", help="Local scratch for downloads (not the Volume)")
    args = p.parse_args()
    if args.command in {"pack", "sft"} and not args.tokenizer:
        p.error("--tokenizer is required for packing")
    if args.command == "tokenizer":
        tokenizer(args)
    elif args.command == "sft":
        pack(args)
    elif args.command == "merge":
        merge(args)
    else:
        fast_pack(args)


if __name__ == "__main__":
    main()
