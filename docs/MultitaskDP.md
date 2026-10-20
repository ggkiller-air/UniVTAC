# Full-sensor multi-task diffusion policy

This independent Stage-0 implementation lives in `policy/multitask_dp/`. It
does not replace ACT, ACT HyperNet, ViTAL, or any `third_party` dependency. It
trains **one model jointly on all eight locally available tasks**, from scratch.

## Model and data protocol

The AdaViP Stage-0 topology and main widths are retained: independent randomly
initialized ResNet18/GroupNorm image encoders, two-frame temporal projection to
256 dimensions per sensor, learned sensor identities and missing tokens, masked
concatenation plus presence flags, a 256-dimensional fusion MLP, and the same
ConditionalUnet1D (512/1024/2048, kernel 5, eight groups, timestep embedding 128,
scale/bias conditioning). The U-Net adaptation is numerically checked against
the pinned reference; provenance and MIT license accompany its source.

Necessary dataset-specific adaptations are explicit:

- All eight raw tasks have **head + wrist**, not three real camera views. Use
  both, plus left/right `rgb_marker` tactile, eight-joint proprioception and a
  frozen normalized 512-dimensional CLIP task instruction. No fictitious view.
- Joint state/action use the first eight of the nine recorded joints, matching
  the author ACT deployment convention (seven arm joints and one finger).
  At raw time `t`, images/proprioception use `t`, target action uses `joint[t+1]`.
  This matches `HDF5Handler.batch_gather_hdf5`, not an identity-action target.
- Native JPEG images are decoded once using OpenCV **BGR**, matching the author
  preprocessing, bilinearly resized to 256x256 and stored as uint8 CHW. The GPU
  maps them exactly once to [-1,1]. Deployment must use the same convention.
- Horizon 16, observation history 2, execute 8. A window is anchored at real
  timestep `t`, starts at `t-1`, and executes predictions starting at index 1.
  Clipped history/actions never cross episode boundaries. Replicated temporal
  padding has zero action-loss mask; each sample divides by its own valid count.
- The local release has 50 episodes per task: **400 total**, 75,543 valid
  state/next-action pairs. Use all of them for training; no validation split,
  validation passes, or best-validation checkpoints. Normalization uses only
  training rows and is checkpointed. This is not a validation-based result.
- Full input probability 0.8; remaining 0.2 distributed over eight configurable
  removals: head, wrist, left tactile, right tactile, state, language, both
  vision, both tactile. Thus **nine total conditions**, not the ten conditions
  of the three-camera ManiFeel dataset. Dropout occurs before sensor fusion.

## Optimization and throughput

`configs/multitask_dp.yaml` fixes batch **64**. Every epoch draws 75,543 windows
with replacement, balanced by task, giving 1,181 optimizer updates per epoch
(including the partial last batch), **708,600 updates over 600 epochs**. No
gradient accumulation or large-batch update-budget reduction is used. AdamW
LR 8e-5, betas (0.95,0.999), epsilon 1e-8, weight decay 1e-6; 500-update warmup,
then cosine decay across that fixed budget. EMA uses the reference warmup
(power 0.75, maximum decay 0.9999), once per optimizer update.

Throughput comes from one-time CPU decoding, 55.33 GiB compact uint8 image
storage, immutable RAM preload shared through Linux fork, eight workers with
prefetch one, pinned memory/asynchronous transfers, fused AdamW, TF32 and bf16
autocast. Model/optimizer/EMA weights remain float32. Bf16 is an explicit
numerical/runtime choice, not an assertion of bitwise equivalence to fp32
training. No Isaac simulator or GPU evaluation is started by these scripts.

The first 100 **actual updates** form a bounded preflight: five windows of 20,
discard the first warmup window, require median >=150 samples/s and data-wait
fraction <=0.4. Passed preflight continues the same model/optimizer in place;
failure saves latest and exits instead of burning a long allocation. Measured
throughput/remaining-time estimates and GPU occupancy go to persistent logs.
At 150/200/250 samples/s, pure 600-epoch training is approximately 84/63/50 hours,
before checkpoint/startup overhead; these are estimates, not measured results.

## Checkpoints and resume

- Rolling `checkpoints/latest.ckpt`: first 100 updates, every 1,000 updates,
  each epoch boundary, and graceful stop.
- Retained snapshots: `epoch0050.ckpt`, `epoch0100.ckpt`, ..., `epoch0600.ckpt`.
  No top-k deletion. Atomic latest replacement ensures later latest writes do
  not mutate the retained snapshot's hard link.
- Each payload includes raw and EMA weights, limits normalizer, Adam state,
  resolved config/protocol/cache identity, epoch/batch cursor, global step,
  EMA step, LR budget, running epoch aggregates, and Python/NumPy/CPU/CUDA RNG.
- Resume replays the seeded epoch sampling plan from the **completed** batch
  cursor, independent of worker prefetch. Incompatible cache, normalizer,
  batch/architecture/schedule settings are rejected. CPU diagnostics verify
  bitwise raw/EMA/Adam equality against uninterrupted optimization. GPU
  nondeterministic kernels are not claimed to be bitwise reproducible.

## Runtime and submission

The initial deployment reuses the existing offline training runtime at
`/public/home/wangzihao/.local/miniforge3/envs/manifeel/bin/python` without
modifying the UniVTAC ACT environment or importing the AdaViP project. Set
`UNIVTAC_DP_PYTHON` to override. Required dependencies: torch/torchvision,
diffusers 0.11.1 with a compatible huggingface-hub, transformers, h5py, numpy,
opencv-python, einops, PyYAML. The CLIP model/tokenizer must already be cached
under `HF_HOME`; preparation uses local-files-only, never a silent fallback.
The verified versions are recorded in `policy/multitask_dp/requirements.txt`;
the deployed torch/torchvision builds are CUDA 12.1 builds.

Create a persistent, immutable snapshot containing `policy/multitask_dp`,
`configs/multitask_dp.yaml`, and the two sbatch scripts. Set
`UNIVTAC_DP_SOURCE`, `UNIVTAC_DP_RUN`, `UNIVTAC_DP_CACHE`, and optionally
`UNIVTAC_DP_DATA_ROOT`. Save a `source.sha256` inventory in the run directory
and copy the selected config to `launch_config.yaml`. Then:

```bash
cache_job=$(sbatch --parsable --account=yangyc --export=ALL \
    "$UNIVTAC_DP_SOURCE/scripts/multitask_dp_cache.sbatch")
sbatch --account=yangyc --dependency="afterok:$cache_job" --kill-on-invalid-dep=yes \
    --export=ALL "$UNIVTAC_DP_SOURCE/scripts/multitask_dp_train.sbatch"
```

Cache preparation requests CPU only, 16 CPUs/128 GiB/4h. Training requests
one H200, 24 CPUs/192 GiB, the GPU partition maximum **100 hours**, without
shortening the allocation for backfill. Before training it reports physical GPU
UUID/model/memory/utilization/processes, verifies logical cuda:0 and at least
60 GiB free VRAM, CPU affinity, shared-memory capacity and source hashes. If
the 100-hour scheduler boundary is actually reached, USR1 triggers a complete
checkpoint and a dependent continuation as a fallback; it is not a planned
six-hour training cycle. Failed throughput/runtime checks do not auto-resubmit.

CPU regression diagnostics, without needing a GPU:

```bash
python scripts/check_multitask_dp.py --output /data/wangzihao/experiments/univtac/diagnostics/unique_run
```

Optionally add `--reference-source <pinned_diffusion_policy_checkout>` for an
exact U-Net state-dict/forward comparison. Use a new output directory each time.
Actual job IDs and observed results belong in `docs/logs/`, not this protocol.
