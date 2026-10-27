# AdaViP T=2 Transformer on UniVTAC

For the uploaded job 392551 checkpoints and commands to run them with the
existing simulator on another server, see
[Stage 2 evaluation](./MultitaskDPEvaluation.md#stage-2-transformer-checkpoints-job-392551).

The selected teacher is **UniVTAC E400 EMA**, selected by the user's completed
evaluation. The earlier local completion log did not contain that evaluation;
its absence is not evidence that E600 was selected. The E400 file was strictly
loaded on CPU and identified as epoch 400, cursor 0, global step 472400:

```text
/data/wangzihao/experiments/univtac/full_sensor_dp_batch64_ep600_seed42_20261019_093221/checkpoints/epoch0400.ckpt
SHA256 8143fb18ad42f335bdcddfea1e41285a1722901435a7fcf0f41eeebe07e5b07d
```

## Port strategy

Follow `AdaViP-main/adavip/manifeel/stage2_policy.py` and
`scripts/manifeel/train_stage2_t2.py`, the current implemented method. The older
research audit's C1/future-prediction proposal is not the implementation being
ported. The MLP fusion control is not included in this port.

| Contract | ManiFeel implementation | UniVTAC port |
|---|---|---|
| Sensor tokens | 3 vision + 2 tactile + state + language | head/wrist + bilateral tactile + state + language |
| Tokens / width / history | 7 / 256 / 2 | 6 / 256 / 2 |
| Utility dimensions | 9 | 8 |
| Full + removal conditions | 10 | 9 |
| Independent FiLM branches | 5 image branches | 4 image branches |
| Action | 7D | 8D absolute joint positions |
| Horizon / execution | 16 / 8 | 16 / 8 |

Utility order is `head, wrist, tactile_left, tactile_right, state, language,
vision, tactile`. A name means that input is removed. `vision` removes both
cameras; `tactile` removes both tactile sensors. Preserve the original uint8
OpenCV channel convention, CLIP task embeddings, observation/action alignment
and teacher normalizer. Do not create a fictitious third camera.

## E400 labels

`policy/multitask_dp/generate_utility.py` computes raw signed utility
`L(masked)-L(full)` with a frozen eval-mode E400 EMA teacher. Each complete
history/horizon window gets 64 deterministic noise/timestep draws. Strata are
`[0,12), [12,25), [25,50), [50,75), [75,100)`, with four independent
`[4,3,3,3,3]` blocks and expectation weights `[.12,.13,.25,.25,.25]`.
This retains the main method's uniform-timestep expectation despite allocating
more samples to low timesteps. Preserve negative values; do not clamp, softmax,
or average the strata with their sample-count frequencies.

For each draw, all nine conditions share exactly the same noisy target, Gaussian
noise, timestep and valid-action mask. Images are encoded once per window.
Masks and draws are vectorized through the original U-Net, with FP32 losses and
TF32 disabled. Only windows with both a complete T=2 history and a complete
16-step action horizon are labeled; no episode boundaries are crossed. This
enumerates **69,543 windows** across all eight tasks and 400 episodes.

Every committed 64-window shard retains source identity, timesteps, Gaussian
draws, 64x9 raw losses, weighted losses, signed utility, analytic utility
variance and two independent K32 estimates. Publication is atomic with SHA256
markers. A restart verifies existing shards and recomputes only uncommitted
work. Final `complete.json` appears only after every shard has been verified;
it hashes `window_index.npy`, `utility.npy`, `utility_variance.npy` and
`weighted_losses.npy` and records pooled per-dimension RMS.

Current run:

```text
/data/wangzihao/experiments/univtac/utility_e400_ema_k64_20261026_2225/
  source/           immutable label-generation source snapshot
  source.sha256     source inventory
  cpu_preflight.json
  queue/            scheduler wave and both job IDs
  gpu_monitor.log   actual GPU UUID, occupancy and process snapshots once running
  labels/           contract, shards, progress and final arrays once generated
```

`scripts/submit_utility_e400.sh` submits both partitions at their current
maximum walltime. Held submissions are released only after both IDs are saved.
A run-specific `flock` permits only one worker; it cancels the sibling request.
The first allocation is used for labeling directly. Scheduler USR1/TERM requests
a safe shard-boundary stop; a clean scheduler stop submits another dual queue
wave with an `afterok` dependency. Failures do not silently loop or submit a
student training job.

Per the user-relayed administrator requirement, the worker holds a separate
explicit CUDA allocation for a **16% VRAM floor**, starting before checkpoint
hashing and RAM loading and ending with the worker. It is logged separately
from useful tensors and does not alter the teacher or loss. The allocation is
on the assigned CUDA device only; it does not touch other jobs. This implements
the requested memory policy, not a guarantee about future scheduler decisions.

## Prepared Transformer implementation

`policy/multitask_dp/stage2.py` inherits only the E400 EMA encoder and normalizer;
the action DP is newly initialized. The actual forward path is:

```text
6 encoder tokens -> original full-sensor fusion -> z_t
stopgrad(z_t) -> Utility Predictor -> 8D RMS-normalized signed u_hat
[projection(z_t), projection(stopgrad(u_hat))] -> 4 independent FiLMs
z_t + query_projection(stopgrad(u_hat)) -> 4-head cross-attention over 6 tokens
LayerNorm -> FF -> LayerNorm -> conditioning -> fresh 8D-action DP
```

State/language tokens bypass FiLM. Each FiLM starts as an exact identity. The
query MLP's final layer starts at zero. The Transformer conditioning is direct,
as in main; it does not add an external residual to the final conditioning or
claim initial action equivalence with the teacher's original MLP fusion.

The utility head trains with signed labels divided by pooled training RMS
(floor 1e-6), SmoothL1 beta=1. Its input is detached; action conditioning uses
detached predicted utility. Action gradients cannot update the utility head;
utility gradients cannot update the encoder.

The trainer and `configs/stage2_transformer_t2.yaml` retain batch32,
600 epochs, task-balanced replacement sampling, fixed EMA .9999, gradient
clipping 1, and the original three parameter groups (encoder LR 8e-5,
utility/action LR 1e-4), 500-update warmup, cosine to 0.1 and encoder-unfreeze
ramp. Phase A (epoch1) trains utility only; B (epochs2–6) freezes the encoder
while training action modules and utility; C (epochs7–600) unfreezes the encoder.
Loss is utility alone in A, otherwise action + .05 utility. All student inputs
are full-sensor. The generic action-loss and inference APIs use the Transformer
path; accidentally using the inherited teacher-mask labeling API is rejected.

The loader requires finalized hash-checked labels and exact source/window
alignment. The trainer preserves raw/EMA weights, RMS/normalizer, optimizer,
scheduler, sampling cursor and RNG. It keeps latest and every configured epoch
milestone without pruning. Formal launch configs also set an independent
external archive directory: each retained milestone is copied there atomically.
The label pipeline itself never submits a student job. After the user's
completion notification, `submit_stage2_transformer.sh` queues both H200
partitions with the same lock/sibling-cancellation mechanism. The first job
does 100 real A/B/C preflight updates in the already RAM-loaded process,
requires finite losses/gradients, at least20samples/s, data wait<=75% and peak
allocation<=90% VRAM, then resets raw/EMA weights, Adam, scheduler and RNG
before formal epoch1. A CPU regression verifies that the formal updates after
reset exactly equal a fresh run. Scheduler early-warning stops after an
optimizer update and queues a dual-partition continuation; resumed formal runs
keep their passed preflight and optimizer cursor rather than restart training.

## CPU verification

Using the existing `manifeel` Python, with CUDA hidden:

```bash
python scripts/check_utility_e400.py --output /persistent/new_diagnostic_dir \
  --teacher /absolute/path/to/epoch0400.ckpt
python scripts/check_stage2_transformer.py --output /persistent/another_new_dir \
  --teacher /absolute/path/to/epoch0400.ckpt
python scripts/check_multitask_dp.py --output /persistent/stage0_diagnostic_dir \
  --reference-source /path/to/diffusion_policy
python scripts/check_multitask_dp_deploy.py
```

The new checks cover complete episode windows, deterministic K64 draws,
weighted signed reductions, vectorized-versus-original paired losses,
mask/noise alignment, corruption rejection, source/label joining, pooled RMS,
balanced sampling, FiLM identity and independence, gradient isolation, phase
gates, Transformer action/inference paths, model restoration and step-exact
raw/EMA/Adam/scheduler resume with 0/2 workers. Full-size E400 encoder/normalizer
inheritance is also checked. CPU verification is not a GPU throughput test or
a simulation success result. Observed job IDs and outcomes are in `docs/logs/`.
