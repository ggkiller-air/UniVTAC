# E400 labels completed; Transformer training launch

- User notified that the utility task had finished. Accounting confirms gpu
  job391730 COMPLETED0:0 onserver35, started2026-10-26 23:51:06+08:00,
  ended2026-10-27 02:35:48+08:00, elapsed2h44m42s. gpux391731 was automatically
  canceled before allocation, as intended.
- Final archive contains69,543 complete windows in1,087 committed shards.
  Rechecked every shard checksum/source index/weighted reduction and matched
  each shard's utility against the final array; all four final file hashes,
  all-window source join, RMS and task balance passed. Actual E400 EMA encoder
  and normalizer initialized the student exactly; its action DP remains fresh.
- Label contract:
  `ab0deea03ee899f5ba344c1c8cd8768da4415a5a61d862a7d7fc5dd986ab2504`.
  Verification report:
  `/data/wangzihao/experiments/univtac/utility_e400_ema_k64_20261026_2225/post_completion_verification.json`.
- Independent K32 half-estimate Pearson correlations range0.8284–0.9050 across
  the eight dimensions. Raw utilities remain signed. Largest pooled mean is
  state0.00142754; language mean0.00003056. These are teacher denoising-loss
  differences, not robot success contributions or causal effects.
- Completed label process used H200 UUID
  `GPU-545f902a-e0d7-a6f2-17e9-eefa25513949` (143771MiB). Explicit16% floor
  allocation24,017,602,048bytes; later torch allocated25,265,344,512bytes.
  NVML logged25741MiB used and99% utilization during labeling.
- Added combined in-process100-update A/B/C preflight and formal reset, safe
  batch-shell stop-file handling, runtime throughput/memory metrics and
  independent milestone copies outside the active training directory. Formal
  protocol remains batch32,600epochs,T2,2174updates/epoch,1,304,400totalupdates,
  original phase gates/LR groups/signed RMS/.05utility weight/fixed EMA.
- Extended CPU regressions passed all previous checks plus exact
  model/EMA/Adam/scheduler/RNG reset, subsequent formal-update equality with a
  fresh run, completed-epoch restart and independent archive retention.
  Report `/data/wangzihao/experiments/univtac/diagnostics/stage2_launch_20261027_v1/report.json`;
  full log is the sibling `stage2_launch_20261027_v1.log`.
- Shell syntax, Python compilation, whitespace and Slurm test-only checks for
  both partitions passed. The combined GPU preflight will run on the actual
  newly assigned H200; CPU tests are not claimed as its result.
- Submitted Transformer **gpu392550** (100h) and **gpux392551** (48h) at
  2026-10-27 09:08:52+08:00. Each requests1H200/32CPU/192GiB, accountyangyc.
  At09:09+08:00 both are PENDING/Priority; neither new GPU allocation nor formal
  update is claimed. First worker locks this run and cancels its sibling.
- Frozen25-file source, launch config, passing CPU report and verified-label
  report are at
  `/data/wangzihao/experiments/univtac/stage2_transformer_e400_batch32_ep600_20261027_0908`.
  Runtime outputs go to `train/`, GPU monitoring every300s to
  `gpu_monitor.log`, and independent immutable milestone copies to
  `/data/wangzihao/experiments/univtac/archives/stage2_transformer_e400_batch32_ep600_20261027_0908`.
  The explicit16% CUDA floor is established before RAM preload. Scheduler
  continuation retains the passed GPU preflight, optimizer and exact cursor.
- No Git commit/push, simulator rollout, MLP control training or interference
  with the existing ManiFeel allocations was performed.
