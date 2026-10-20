# Full-sensor multi-task DP: implementation and queue submission

- User requested an independent copy of the AdaViP multi-task architecture in
  UniVTAC, submitted for training, emphasizing throughput without materially
  increasing batch size/reducing optimization updates. No changes to ACT,
  ACT HyperNet, ViTAL or third_party; no commit/push or simulator evaluation.
- Audited the actual local raw HDF5 stores: eight tasks, 50 episodes each,
  400 episodes total, 75,543 valid observation/next-joint-action pairs. All
  have head/wrist images and bilateral rgb_marker tactile. Follow the author
  `HDF5Handler.batch_gather_hdf5` alignment: state=joint[:-1,:8],
  action=joint[1:,:8], images=rgb[:-1]. OpenCV BGR is intentionally retained.
- Independent `policy/multitask_dp/` contains the four ResNet18/GroupNorm
  encoder branches, 256D sensor tokens, missing-token/presence fusion and
  reference ConditionalUnet1D (512/1024/2048, kernel5, groups8, timestep128).
  Exactly 301,447,048 parameters in the production configuration. Necessary
  UniVTAC adaptations: two real camera inputs, eight-joint actions; no phantom
  third camera or ManiFeel seven-dimensional Cartesian action assumption.
  Full input 80%, configurable 20% split over eight real-sensor removals,
  nine conditions including full. Frozen CLIP instruction embeddings512D.
- Configuration: batch64, 600epochs, task-balanced replacement sampling,
  1181updates/epoch and 708600totalupdates, AdamW8e-5/warmup500/cosine,
  EMA once/update, all local episodes train-only, no validation. uint8 cache
  55.33GiB is decoded once on CPUs, loaded read-only into fork-shared RAM;
  workers8/prefetch1/pinned transfer, GPU scaling, fused AdamW/TF32/bf16.
  Numerical precision is explicitly recorded, not claimed equal to fp32.
- Checkpoints: rollinglatest at step100, every1000steps, each epoch and
  graceful interruption; retain every50epoch snapshot through600. Complete
  raw/EMA/Adam/normalizer/config/cache identity/schedule/cursor/RNG payloads.
  Atomic replace plus hard-linked periodic snapshots avoids double writes
  while retaining immutable snapshots. No top-k cleanup.
- Verified CPU regressions in
  `/data/wangzihao/experiments/univtac/slurm/dp_cpu_diagnostics_20261019_093301.log`:
  actual JPEG decode/channel order, next-action alignment, causal edge padding,
  temporal action masks, normalization round-trip/constant dimensions, mask
  probabilities, all-sensor gradients, all-condition paired loss equality,
  action inference shapes, reference U-Net strict state-dict and bitwise
  forward equality, interrupted-vs-continuous raw/EMA/Adam bitwise equality
  with both0and2workers, sampler replay, incompatible-resume rejection,
  snapshot retention and failed-throughput-gate save/exit. Python compilation
  and both sbatch shell syntax checks passed. No GPU result claimed yet.
- Offline cached CLIP CPU generation succeeded with transformers4.42.3 and
  local openai/clip-vit-base-patch32. Embedding SHA256:
  `c005ab6cf50ecc170e702db776f027e8ccc2d59903b8ca3baf7c380726735abd`.
  Reused existing manifeel runtime without modifying either environment;
  actual versions recorded in `policy/multitask_dp/requirements.txt`.
- Persistent run:
  `/data/wangzihao/experiments/univtac/full_sensor_dp_batch64_ep600_seed42_20261019_093221`.
  Separate immutable cache source and final training source snapshots, source
  SHA256 inventory, launch config, provenance and CPU diagnostics saved there.
- Submitted CPU cache job **374748**, accountyangyc/partitioncpu,
  16CPU/128GiB/4h. Running onserver43 at09:34+08:00, decoding280/400 reported;
  ready manifest is published atomically only after all workers and CLIP pass.
- Submitted GPU job **374750** at09:34:28+08:00, accountyangyc/partitiongpu,
  1H200/24CPU/192GiB/**100h**, no time-min backfill reduction; initiallyPENDING
  onafterok:374748, kill-on-invalid-dependency enabled. GPU/process/CPU/source
  checks precede training. First100realupdates are a bounded throughput gate:
  discard first20, require median>=150samples/s and mediandatawait<=0.4;
  success continues in place, failure saves/exits. At150/200/250samples/s,
  pure training estimates84/63/50h before overhead, not measured promises.
  Only if the actual100h limit is reached does signal checkpointing submit a
  fallback continuation; no routine short-allocation cycling.
- Existing AdaViP job373435 was RUNNING onserver39 throughout this work;
  CPU interactive366142 untouched. No cancellation or resource takeover.
- Cache job374748 subsequently COMPLETED0:0 in2m23s (controller accounting),
  decode workers93.6s and totalbuilder114.9s. Published cache identity
  `bd4b64755306f5520602f2172edf72bd829ab9a0f5da49fd6687fac28e48084d`.
  Mmap validation loaded all array headers/statistics and compared one interior
  sample from every task directly against raw HDF5: all four image arrays,
  current joint state and next-step actions matched exactly. Full cache uses
  CLIP SHA256 `ca8572f376faf392088b38e0abfd7decf214a638c73f168a50e37489cb22930d`;
  this is the authoritative training embedding identity, distinct from the
  preliminary two-thread diagnostic hash above. CPU reduction thread counts
  can change CLIP low bits; the actual cache identity is persisted and strictly
  checked on resume. Validation saved in the run's`cache_validation.json`.
  GPU dependency cleared; at09:36+08:00 job374750 was stillPENDING(Priority),
  so no allocatedGPU throughput or convergence result is asserted.
