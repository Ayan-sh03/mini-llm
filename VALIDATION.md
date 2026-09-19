# Validation — 2026-09-19

## Hyperbolic update

- Clean isolated CPU environment installed successfully; `pip check` reports no broken requirements.
- `python -m pytest -q`: **12 passed** (original 9 plus 3 asset-transfer tests).
- Mock private HF upload/download preserves manifest hashes and reader resume order;
  unrelated local files are excluded, completed bundles cannot be overwritten, public
  upload destinations and corrupted shards are rejected. No live HF calls in tests.
- `python -m mini.preflight --device cpu`: passed actual forward/backward/AdamW update.
- `bash -n scripts/setup_hyperbolic.sh`: passed; GPU setup itself was not executed.
- Modal CLI help/import and Python compilation: passed.
- `mini/train.py`, `mini/engine.py`, and the 123M config are byte-identical to the
  previous package; the checkpoint format and model architecture did not change.
- CUDA 12.8 installation, Blackwell/H100 kernels, GPU compilation, rental billing and
  live transfers remain untested here. No rental was created and no credits spent.

The original CPU validation below remains applicable to the unchanged trainer.

## Executed locally

- `python -m pytest -q`: **9 passed**.
- `python -m mini.smoke`: **passed**, using generated toy text and no remote data.
- Actual 122,708,736-parameter model instantiated on CPU, forward and backward at
  sequence length 16: logits `[1, 16, 32768]`, all trainable gradients present/finite.
- Uninterrupted six-update training versus three updates + checkpoint + fresh process
  + three updates: **bitwise equal model weights and optimizer tensors on CPU**.
- Data reader order, cursor restoration and shard-corruption rejection tested.
- Tied embedding/head identity and causal attention tested.
- Assistant/user/padding loss masks, complete-response handling and Unicode round-trip tested.
- Unequal assistant-target counts across microbatches: accumulated gradients produce
  equivalent optimizer updates within FP32 tolerance.
- Tiny repeated-batch loss reduced by more than 50% in 25 updates.
- WSD boundary behavior and invalid batch configuration tested.
- Checkpoint checksum and recipe-change rejection tested.
- Full toy pipeline: tokenizer -> pretraining packing -> training -> save -> resume
  -> SFT packing -> SFT initialization -> generation.
- Modal app module imports without authentication or launching cloud compute.
- Python source compilation passed.

Environment: Linux, Python 3.12, PyTorch 2.7.1+cpu, LitGPT 0.5.9,
NumPy 1.26.4; core dependency pins are in requirements.txt.

## Not executed / still required before spending the main budget

- Modal remote image build or paid CPU/GPU function invocation.
- H100 BF16, CUDA fused AdamW, torch.compile throughput and memory checks.
- Authenticated HF upload/download round-trip or gated dataset access.
- Real Ultra-FineWeb/UltraChat ingestion through their live streaming services.
  Published schemas were checked; only local JSONL ingestion was executed here.
- Cross-provider GPU migration and GPU numerical reproducibility.
- Long-run stability, model benchmark scores or conversational quality.
- Dollar-budget enforcement: the timer is a training-loop limit, not a billing cap.

CPU toy throughput is deliberately NOT used to estimate H100 throughput or the
number of tokens your $80 can buy. Perform the 20-minute 123M pilot in README.md.
