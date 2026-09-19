"""Thin provider wrapper. No cloud work occurs until `modal run` is invoked."""
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
app = modal.App("ayan-mini-llm")
volume = modal.Volume.from_name("ayan-mini-llm", create_if_missing=True)
secrets = []
if os.environ.get("MINI_HF_AUTH") == "1":
    secrets.append(modal.Secret.from_name("huggingface"))
if os.environ.get("MINI_WANDB") == "1":
    secrets.append(modal.Secret.from_name("wandb"))
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install_from_requirements(str(ROOT / "requirements.txt"))
         .env({"PYTHONPATH": "/project", "HF_HOME": "/work/hf-cache",
               "TOKENIZERS_PARALLELISM": "true", "OMP_NUM_THREADS": "4",
               "MINI_MODAL_VOLUME": "ayan-mini-llm",
               "MINI_HF_AUTH": os.environ.get("MINI_HF_AUTH", "0"),
               "MINI_WANDB": os.environ.get("MINI_WANDB", "0")})
         .add_local_dir(str(ROOT / "mini"), "/project/mini")
         .add_local_dir(str(ROOT / "configs"), "/project/configs"))


def execute(task, args, reload=True):
    if reload:
        volume.reload()
    try:
        subprocess.run([sys.executable, "-m", f"mini.{task}", *shlex.split(args)],
                       check=True, cwd="/work")
    finally:
        volume.commit()


@app.function(image=image, volumes={"/work": volume}, cpu=4, memory=8192,
              timeout=86400, retries=0, max_containers=1, secrets=secrets)
def cpu_job(task: str, args: str):
    execute(task, args)


@app.function(image=image, volumes={"/work": volume}, cpu=32, memory=32768,
              timeout=86400, retries=0, max_containers=1, secrets=secrets)
def prepare_job(task: str, args: str):
    # Tokenization is Rust/rayon-bound: give the batched encoder every core.
    os.environ["RAYON_NUM_THREADS"] = "32"
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    execute(task, args)


@app.function(image=image, volumes={"/work": volume}, gpu="H100", cpu=4,
              memory=16384, timeout=86400, retries=0, max_containers=1, secrets=secrets)
def gpu_job(task: str, args: str):
    execute(task, args)


@app.function(image=image, volumes={"/work": volume}, cpu=32, memory=32768,
              timeout=86400, retries=0, max_containers=8, secrets=secrets)
def pack_shard(base_args: str, index: int, count: int, tag: str):
    # No reload: shards write disjoint files, and the mount already has the tokenizer.
    os.environ["RAYON_NUM_THREADS"] = "32"
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    args = f"{base_args} --shard-index {index} --shard-count {count} --tag {tag}"
    execute("prepare", args, reload=False)


@app.function(image=image, cpu=4, memory=8192, timeout=86400, retries=0, secrets=secrets)
def pack_all(base_args: str, count: int, out: str):
    # Server-side fan-out so the whole pack survives a detached client.
    handles = [pack_shard.spawn(base_args, i, count, f"s{i:02d}") for i in range(count)]
    for handle in handles:
        handle.get()
    prepare_job.remote("prepare", f"merge --out {out}")


@app.local_entrypoint()
def main(task: str = "smoke", args: str = "", spawn: bool = False):
    if task not in {"smoke", "prepare", "train", "hub", "chat", "assets"}:
        raise ValueError("task must be smoke, prepare, train, hub, chat, or assets")
    if task == "prepare":
        shards = re.search(r"--shards\s+(\d+)", args)
        if shards:
            count = int(shards.group(1))
            base = re.sub(r"--shards\s+\d+", "", args).strip()
            out = re.search(r"--out\s+(\S+)", base)
            tokens = re.search(r"--max-tokens\s+(\d+)", base)
            if not out:
                raise ValueError("--out is required for a parallel pack")
            if count > 1 and tokens:
                per = max(1, int(tokens.group(1)) // count)
                base = re.sub(r"--max-tokens\s+\d+", f"--max-tokens {per}", base)
            if spawn:
                call = pack_all.spawn(base, count, out.group(1))
                print(f"Spawned {count}-shard pack {call.object_id} -> {out.group(1)}", flush=True)
            else:
                pack_all.remote(base, count, out.group(1))
            return
    function = gpu_job if task == "train" else (prepare_job if task == "prepare" else cpu_job)
    if spawn:
        call = function.spawn(task, args)
        print(f"Spawned {task} call {call.object_id} (app stays up while detached).",
              flush=True)
    else:
        function.remote(task, args)
