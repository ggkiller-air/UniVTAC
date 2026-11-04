# Six-task L1 E300 EMA → UViP S0

User selected E300 on2026-11-04. The Teacher is the final six-task L1 run's
`epoch0300.ckpt`, EMA/global_step314400, protocol
`995fc76fe32dee66cd220062eacb80c432e6e88ea90fcebe45c84e7b0a295442`.
This is distinct from the historical eight-task E400/L0 experiment.

The versioned launch template is `configs/l1_e300_uvip.yaml`. Copy it into a
new run directory as `launch_config.yaml`, recording that run's immutable
`source_id` and external `archive_dir`; the active run's exact config remains
in its directory below. Changing a template does not change a running job.

The new `policy.multitask_dp.l1_uvip` entrypoint strictly restores this Teacher
and its archived normalizer. It checks the six-task cache identity, preserves
the native BGR/joint-next protocol, and selects62540complete history2/horizon16
windows from the300selected episodes/67040anchors. The other two tasks are
excluded. Task-balanced replacement sampling continues in Student training.

Labels use native8D-action DDPM loss, nine full/removal conditions, shared
per-window Gaussian/timestep streams and the existing five-stratum K64
weights. They retain raw signed differences, variance, independent K32 halves,
raw losses/noise/timesteps and exact source indices. Shards commit atomically;
continuation validates existing shards rather than recomputing them. Actual
GPU cached scoring is compared against native paired forwards across all six
tasks before formal generation. No original label directory is reused.

The Student preserves the exact five-sensor late-language L1 context,
encoder and normalizer. A separate language token projection supplies S0
attention. Four identity-initialized FiLM heads modulate V1/V2/T1/T2; P/L stay
unmodulated. Utility is8D, scaled with the completed training-label RMS without
centering or clipping. Utility/context and action/Utility gradients are
isolated. The native-shaped action U-Net is freshly initialized.

Training follows the existing UniVTAC S0 protocol: batch32,600epochs,
A=E1/B=E2–6/C=E7–600; C unfreezes inherited encoders. Utility/action LR1e-4,
encoder8e-5,500-update warmup/unfreeze ramp, Utility weight0.05, fixedEMA0.9999.
The existing trainer retains Adam/scheduler/RNG/data cursor/phase metadata.
Latest saves every1000updates, each epoch boundary and graceful stop; fixed
snapshots E1/6/10/20/50/100/200/300/400/500/600 are retained and copied to the
external archive. `load_student(path,weights='ema',device=...)` restores the
L1 Student and native joint-action inference without its original dataset.

Verification: ten dedicated checks pass in both workspace and isolated source,
including native K64 loss equality, microbatch independence, exact Teacher
context, identity FiLM, fresh actor, A/B/C freeze, gradient isolation, native
sampling, six-task filtering and exact raw/EMA/Adam/scheduler continuation
across A/B/C. Actual E300 EMA and12normalizer buffers restore strictly; real
six-task dataset identity matches. Historical utility and Stage2 regression
scripts also pass. GPU scoring checks and100-update A/B/C preflight execute
after allocation, before their respective formal stages.

Job426602 was submitted15:29:53+0800 with normal Account=yangyc,
QOS=user_wangzihao: one H200/32CPU/192GiB, gpu,gpux common48hour maximum,
per-run lock and same-job soft-stop/requeue. Labels then training run
sequentially. An independent16%VRAM holder starts and verifies its reservation
before model loading; it is outside model, optimizer and RNG state. No new
standalone CPU polling job was created. Initial state PENDING.

Run/source/config/logs:
`/data/wangzihao/experiments/univtac/l1_e300_ema_k64_uvip_s0_b32_ep600_seed42_20261104_152658/`.
CPU verification receipts:
`/data/wangzihao/outputs/univtac_l1_e300_uvip_checks_20261104/` and
`/data/wangzihao/outputs/univtac_e300_regressions_20261104/`.
The canonical root progress file remains a table with update time only.
