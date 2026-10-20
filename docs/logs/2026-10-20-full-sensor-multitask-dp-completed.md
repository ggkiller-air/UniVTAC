# Full-sensor multi-task DP: job 374750 completed

## Completion

- Slurm job `374750` finished with state `COMPLETED` and exit code `0:0`.
- Start: `2026-10-19 09:39:54 +08:00`; end: `2026-10-20 21:30:32 +08:00`.
  Allocated elapsed time: **35h 50m 38s**.
- Completed all **600 epochs / 708,600 optimizer updates**; EMA step is also
  `708600`. The final `completed` event was written at `21:30:29 +08:00`.
- Final epoch mean training loss: `0.00010465682030451683`. This is training
  diffusion loss, not validation performance or simulation success rate.
  No simulation evaluation was run.
- Persistent run directory:
  `/data/wangzihao/experiments/univtac/full_sensor_dp_batch64_ep600_seed42_20261019_093221`.
- Scheduler logs:
  `/data/wangzihao/experiments/univtac/slurm/univtac_fullsensor_dp_batch64_ep600_374750.out`
  and the corresponding `.err` file, which is empty.

## Retained checkpoints

The run's `checkpoints/` directory contains:

```text
epoch0050.ckpt
epoch0100.ckpt
epoch0150.ckpt
epoch0200.ckpt
epoch0250.ckpt
epoch0300.ckpt
epoch0350.ckpt
epoch0400.ckpt
epoch0450.ckpt
epoch0500.ckpt
epoch0550.ckpt
epoch0600.ckpt
latest.ckpt
```

There are **12 retained epoch snapshots plus rolling latest**. `latest.ckpt`
and `epoch0600.ckpt` are hard links to the same final checkpoint, not two
independent copies. Each is approximately **4.49 GiB**; `du -sh checkpoints`
reports **54G** of actual storage. Intermediate rolling latest saves were
replaced, not retained as per-step snapshots. No validation split or best-model
selection was configured, so there is no `best.ckpt`.

Each checkpoint contains raw policy weights (`policy`), EMA weights (`ema`),
optimizer state, config/protocol/normalizer identity, epoch/batch/update cursors,
schedule and RNG state. Both raw and EMA weights are in the same checkpoint;
there are no separate EMA checkpoint files.

## Repository publication checks

- The source inventory in the run's `source.sha256` matched all 14 current
  implementation/config/documentation/script files before publication.
- CPU regression diagnostics passed again on `2026-10-20` using the existing
  `manifeel` Python environment with CUDA hidden. This rerun did not supply
  the optional external reference checkout; the original reference comparison
  is recorded in the implementation log dated `2026-10-19`.
- Python compilation and both Slurm scripts' Bash syntax checks passed.
- Publication includes source, config, launch/diagnostic scripts and Markdown
  records only. Checkpoints, datasets, runtime logs and `.env` remain local.
