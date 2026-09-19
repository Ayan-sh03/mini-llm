"""Tiny actual forward/backward/optimizer check before a paid training pilot."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .common import read_json
from .engine import build_model, make_optimizer, update


def check_runtime(device):
    if device.type != "cuda":
        return {"device": "cpu", "torch": torch.__version__}
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Check driver and install the CUDA wheel, not CPU torch.")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose one GPU with CUDA_VISIBLE_DEVICES=0 before launching.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This trainer requires a BF16-capable GPU.")
    capability = torch.cuda.get_device_capability()
    cuda = tuple(int(x) for x in (torch.version.cuda or "0.0").split(".")[:2])
    if capability[0] >= 10 and cuda < (12, 8):
        raise RuntimeError("Blackwell requires this project's cu128 build. Run scripts/setup_hyperbolic.sh.")
    return {"device": torch.cuda.get_device_name(), "compute_capability": capability,
            "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "vram_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--compile", action="store_true", help="Also exercise torch.compile kernels")
    args = p.parse_args()
    device = torch.device(args.device)
    print(json.dumps(check_runtime(device)), flush=True)
    torch.set_num_threads(2)
    cfg = read_json(Path(__file__).resolve().parents[1] / "configs/tiny.json")
    model = build_model(cfg["model"], device)
    optimizer = make_optimizer(model, cfg["train"], device)
    run_model = torch.compile(model) if args.compile else model
    x = np.random.default_rng(42).integers(0, cfg["model"]["vocab_size"], size=(2, 65), dtype=np.int64)
    loss, norm, count = update(run_model, optimizer, [(x[:, :-1], x[:, 1:])],
                               0.001, device, 1.0)
    if not np.isfinite(loss):
        raise RuntimeError("Nonfinite preflight loss")
    print(json.dumps({"passed": True, "loss": loss, "grad_norm": norm,
                      "supervised_tokens": count, "compiled": args.compile}))


if __name__ == "__main__":
    main()
