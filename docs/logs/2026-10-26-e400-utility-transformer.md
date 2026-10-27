# E400 K64 utility queue and Transformer port

- User supplied the completed evaluation decision: **UniVTAC E400 is the
  selected teacher**. This supersedes the uncertainty in the earlier local
  completion log; no local rollout scores were invented or rerun.
- Actual `epoch0400.ckpt` strictly restored EMA at epoch400/cursor0/step472400;
  protocol and all normalizer buffers verified. SHA256:
  `8143fb18ad42f335bdcddfea1e41285a1722901435a7fcf0f41eeebe07e5b07d`.
- Added independent stratified K64 generation, paired full+8 removals, FP32
  with TF32 disabled, complete T2/horizon16 anchors only, single image encoding
  per window, deterministic saved draws, atomic checksum shards and resumable
  finalization. Enumerated69,543 windows across eight tasks/400 episodes.
  No changes to Stage0 training, ACT, ViTAL or third_party.
- Label CPU preflight passed deterministic draws/weights, source boundaries,
  vectorized-versus-original paired-loss comparison, removed sensor isolation,
  valid resume and corruption/contract rejection plus actual teacher restore.
  Report: `/data/wangzihao/experiments/univtac/diagnostics/utility_e400_20261026_v1/report.json`.
- Submitted at **2026-10-26 22:25:36 +08:00**: gpu **391730**, 100h;
  gpux **391731**, 48h; accountyangyc, 1H200/16CPU/160GiB each. First allocation
  locks this run and cancels its sibling. Both were PENDING/Priority after
  submission. Scheduler signal stops safely at a committed shard and queues
  continuation; no student-training dependency or automatic submission.
- Immutable label source/output root:
  `/data/wangzihao/experiments/univtac/utility_e400_ema_k64_20261026_2225`.
  Label CPU report and 18-file SHA256 source inventory are retained there.
  Queueing happened before Transformer implementation, per user sequence.
- Login/current CPU node exposes no CUDA devices. Scheduler allocation checks
  retained existing jobs384201/server09 and387293,387295/server13 untouched.
  Read-only NVML sample in387293 reported H200143771MiB,17797MiB used,71% GPU
  utilization. Each new label job logs its exact assigned physical GPU, UUID,
  memory/utilization and processes before work.
- User relayed administrator approval for memory occupancy >=15%. The label
  worker allocates and logs a persistent **16% CUDA memory floor** immediately
  after CUDA initialization, before checkpoint hashing/host-cache loading;
  monitor period300s. No actual new-job occupancy result is claimed while queued.
- Ported `AdaViP-main` T2 Transformer: six256D tokens, eight signed utilities,
  four independent identity FiLMs, unchanged state/language, detached utility
  paths, utility query/direct four-head attention and fresh eight-joint DP.
  Prepared finalized-label loader, A/B/C training gates, main's optimization
  schedule/EMA, retained checkpoint milestones and exact stateful resume.
- CPU gradient test caught copied teacher `requires_grad=False` flags; fixed
  the builder to enable the student encoder before phase gates take control.
  Full new Transformer suite then passed, including exact interrupted-versus-
  continuous raw/EMA/Adam/scheduler equality in A/B/C and with0/2workers,
  corruption rejection and actual full-size E400 inheritance/fresh DP check.
  Added an epoch-boundary restart regression and fixed retry to preserve an
  already retained matching snapshot rather than fail on its existence.
  Final full new suite passed including boundary restart/immutable snapshots
  and encoder unfreeze LR gates. Report:
  `/data/wangzihao/experiments/univtac/diagnostics/stage2_transformer_20261026_v3/report.json`;
  companion `.log` is in the parent diagnostics directory.
- Existing Stage0 CPU diagnostics passed with exact pinned reference U-Net
  comparison; deploy interface tests **6/6 passed**. Diagnostic root:
  `/data/wangzihao/experiments/univtac/diagnostics/port_stage0_regression_20261026_v1`.
  Python compilation, CLI help, shell syntax, Slurm test-only submissions for
  both partitions and whitespace checks passed.
- Existing unrelated uncommitted deployment changes were preserved. No Git
  commit/push, simulator evaluation or student GPU training was started. Wait
  for the user's label-completion notification before the next training step.
- Final scheduler check at22:33+08:00:391730 and391731 remain
  PENDING/Priority. Neither has allocated a GPU or generated labels yet.
