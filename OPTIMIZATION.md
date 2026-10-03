# Training performance pass

Based on commit `2147268`. This patch keeps the 123M architecture, BF16
autocast, FP32 weights and loss arithmetic, global batch, masked-target
normalization, AdamW, clipping, WSD, data order, and checkpoint format.
Floating-point rounding can differ under GPU compilation or different microbatches.

## Changes

- Compile model **and cross-entropy together**, instead of returning full vocabulary
  logits to eager FP32 cross-entropy. This allows compiler fusion; actual allocation
  and speed reductions must be checked on the GPU.
- Accumulate detached loss on device. Remove the CPU scalar conversion after every
  microbatch. Keep metrics and the finite gradient check at optimizer boundaries.
- Reuse a pinned host buffer and copy the whole update to CUDA once, then slice it
  into the same microbatches. Buffer reuse is safe because each update synchronizes
  before returning. This is not asynchronous next-update prefetching.
- Cache the reader's affine permutation coefficients once per epoch, preserving
  exactly the original permutation and resume state.
- Add a bounded full-training benchmark with the original engine retained for A/B,
  fresh-process batch sweeps, explicit CUDA synchronization, peak allocated memory,
  startup time, real-data mode, and optional profiler traces.
- Add `MINI_GPU=B200` selection and its PyTorch 2.7.1 CUDA 12.8 wheel. Keep the
  pinned LitGPT/model versions. Default remains H100.
- Fix an existing preparation bug: pretraining `pack --local` ignored the JSONL
  input and queried/downloaded the remote dataset. It now packs local data offline,
  allowing the end-to-end smoke workflow to validate its intended toy corpus.

Fused AdamW, set-to-none gradient clearing, BF16 autocast, and PyTorch SDPA
attention were already present. They are not new optimizations in this patch.

## Measure before claiming a speedup

Run baseline and optimized on the **same physical GPU, config, and microbatch**.
Synthetic data measures the engine plus host staging; it excludes disk loading.
Add `--data train-data` to both commands to include real reader work. For a complete
run, independently include compilation, validation, checkpointing, and uploads.
The baseline engine retains the original model-only compile and update loop; both
benchmark engines use the improved reader, so real-data A/B isolates engine gains.

GPU commands below consume Modal credits **when invoked**. No cloud run is launched
by installation, import, CPU tests, or this patch.

```bash
# Compare at the old default. Use separate NEW output filenames.
MINI_GPU=H200 modal run modal_app.py --task benchmark --args \
  '--config /project/configs/123m.json --engine baseline --micro-batch 4 --warmup 10 --steps 100 --out h200-baseline-mb4.json'
MINI_GPU=H200 modal run modal_app.py --task benchmark --args \
  '--config /project/configs/123m.json --engine optimized --micro-batch 4 --warmup 10 --steps 100 --out h200-optimized-mb4.json'

# Sweep on B200 with the original global batch of 128 intact.
MINI_GPU=B200 modal run modal_app.py --task benchmark --args \
  '--config /project/configs/123m.json --micro-batches 8,16,32,64,128 --warmup 10 --steps 100 --out b200-sweep.json'

# Try autotuning on the winning size (32 is an example, not the assumed winner).
MINI_GPU=B200 modal run modal_app.py --task benchmark --args \
  '--config /project/configs/123m.json --micro-batch 32 --compile-mode max-autotune-no-cudagraphs --warmup 10 --steps 100 --out b200-mb32-autotune.json --profile-dir b200-mb32-traces'

# Use the measured winning size on your real data. This is a fresh run.
MINI_GPU=B200 modal run modal_app.py --task train --args \
  '--config /project/configs/123m.json --data train-data --tokenizer tokenizer --out pretrain-b200-new --micro-batch 32 --compile-mode max-autotune-no-cudagraphs --max-seconds 1200'
```

Repeat the winning benchmark to check variance. Increasing warmup may be necessary
if compilation, autotuning, or cache warming continues into the measured window.
Candidates that OOM report errors; later candidates run in fresh processes.
Compare optimizer **updates**, including forward, backward, clipping and AdamW;
do not count accumulation twice. All target positions count once in pretraining.
Optional profiler traces load in Perfetto or the Chrome tracing viewer.

The original config defaults to microbatch 4: 32 forward/backward passes per update.
Microbatch 32 reduces that to four; 64 reduces it to two. Bigger is not automatically
faster, and the 32K vocabulary creates large logits and gradients. Use peak allocated
memory and measured speed to choose, rather than automatically using 128.

## B300 correction

Do **not** use `B200+` with this repo's original image. Modal can assign B300,
whose current requirements include CUDA 13.1+. PyTorch 2.7.1/cu128 does not satisfy
that requirement. This patch uses B200 explicitly. B300/B200+ need a separately
validated framework/image upgrade, including the newer GPU's compiled kernels.

## Next experiments, after GPU profiling

1. If vocabulary projection/loss dominates: evaluate fused linear cross-entropy
   (e.g. Liger or Cut Cross Entropy). It can avoid storing the full token-by-vocabulary
   matrix. Require numerical/gradient checks for this tied embedding, sum reduction,
   masked SFT labels, BF16 and checkpoint behavior before making it the default.
2. If attention dominates: verify the actual SDPA kernel selected on Blackwell, then
   evaluate a Blackwell-native attention implementation. SDPA already exists; merely
   adding a library is not evidence of a gain.
3. If gaps between kernels dominate: evaluate CUDA graphs for the complete update,
   including gradient accumulation and optimizer state. Use static buffers and
   validate capture/replay correctness and LR updates; this patch deliberately
   provides autotuning without CUDA graphs.
4. If matmuls dominate: evaluate an FP8 training ablation with scaling and quality
   checks. FP8 is a new numerical recipe, not a free, guaranteed 2x multiplier.
5. If real-data speed lags synthetic speed: profile local staging and prefetching.
   Preserve checkpoint cursor semantics; naive lookahead can skip data on resume.

No B200/H200 performance result is claimed from CPU checks. The user's 600k tokens/s
is a reported measurement; the current repo does not establish the exact settings
that produced it. Hardware ratios and speculative kernel gains are not additive
benchmark results.

## Validation in this workspace

- 17 pytest checks passed, including original-versus-optimized CPU update parity
  with uneven masked targets, complete forward/backward capture with AOT eager,
  permutation parity across epochs, fresh-process benchmark sweeps, and local-data
  packing without remote revision queries.
- End-to-end toy workflow passed: tokenizer, local packing, pretraining, full
  checkpoint, fresh-process resume, assistant-only SFT, and generation. Uninterrupted
  versus resumed CPU model weights and optimizer state were bitwise identical.
- Profiler trace export and the B200 Modal wrapper import passed locally.
- No CUDA runtime was available for validation. CUDA staging, Inductor GPU kernels,
  the built Modal image, and actual H200/B200 throughput still require a GPU run.
