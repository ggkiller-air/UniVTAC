# Joint-condition HyperNet for ACT

## Model and author protocol

The integration follows `AdaViP/configs/manifeel/hyper_resnet_fusion_tasks.yaml`:
`use_fusion: true`, `use_fusion_hypernet: false`. The ManiFeel training budget is not
transferred. The ACT launcher reads the existing author `train_config.yml` or
`train_config_all.yml` and retains batch size 64, 4,000 optimization steps, action
chunk 50, AdamW learning rates 1e-5, weight decay 1e-4, KL weight 10, hidden dimension
512, feedforward dimension 3,200, four encoder layers, seven decoder layers, eight
heads, and dropout 0.1. Checkpoints are saved every 1,000 steps.

The author split remains episode-wise 80/20; the original split seed behavior is
unchanged. The training wrapper defaults to seed 0, as the author's `one.sh` does.
Use 50 demonstrations for the author's default demonstration budget. No policy
training hyperparameters in the original YAML files are modified.

ImageNet ResNet18 vision initialization and the published pretrained tactile encoder
are retained. Training checks the tactile checkpoint exists, matches the published
SHA256, and loads its state dict strictly. Missing weights cannot silently turn the
run into random tactile initialization.

Published checkpoint:

```text
byml2024/UniVTAC/checkpoints/encoder.pth
revision: 5219489db6fcc5c2a5e7a77f158bdaefaa363674
sha256: 28097c91a1a1d051f65266fc7fdd84420dc184da9dcfc2006aa40af144cf15bf
```

The local copy is `$UNIVTAC_DATA_ROOT/checkpoints/encoder.pth`. To obtain it together
with the author's task checkpoints on another machine:

```bash
source scripts/env.sh
bash data/download.sh --checkpoint lift_can
```

## Cameras and joint representation

`policy/task_settings.json` is the shared source for task camera selection, used by
ACT configuration preparation, preprocessing, validation, and deployment:

| Task setting | Raw cameras | ACT names |
| --- | --- | --- |
| `head` | head | `cam_high` |
| `wrist` | wrist | `cam_high` |
| `all` | head and wrist | `cam_high`, `cam_wrist` |

Currently `lift_can` and `insert_tube` select `all`; the other defined tasks select
`head`. The ManiFeel wrist-only setting is not imposed on UniVTAC. Both tactile
streams remain `tac_left` and `tac_right`.

The author ACT configuration and deployment use eight-dimensional states/actions:
seven arm joints plus the first finger position. Recorded nine-joint Franka data
includes the second finger; the ACT-specific preprocessor selects `joint[:, :8]`,
matching the existing deployment selection and the environment's seven-arm-plus-one-
gripper action interface. Other policy preprocessors are not changed to eight dimensions.

New ACT data goes to `$UNIVTAC_DATA_ROOT/processed/ACT/author`, with an
`author_act_v1` protocol marker. Previously generated installation smoke data with
`cam_head` and nine-dimensional actions is left intact and is not accepted for these runs.

## Insertion point

```text
author visual/tactile ResNets
    -> original ACT vision/tactile projections
    -> joint-condition HyperNet + gated vision-led summary fusion
    -> original ACT token concatenation and positional embeddings
    -> original ACT Transformer, CVAE latent and action head
```

Each projected visual/tactile stream retains its full `[batch, channels, tokens]`
representation. Mean pooling is used only to build a joint observation condition:
LayerNorm of each stream summary, concatenated with the normalized eight-dimensional
joint state. A shared MLP with width 256 and SiLU generates independent gamma/beta
for every camera and tactile stream. Channel-wise residual modulation is broadcast
over each stream's tokens:

```text
adapted = feature + alpha * (gamma * feature + beta)
gamma, beta = tanh(HyperNet(condition))
alpha = 0.1 * sigmoid(learnable_scale), initially 0.01
```

The selected profile's four-head vision-led fusion is also retained: modulated visual
summaries query the modulated summaries of all visual and tactile streams. The resulting
residual is broadcast onto visual tokens through a zero-initialized `tanh` gate.
This adapts the original global-vector fusion to ACT without replacing spatial tokens
with pooled vectors or changing the existing ACT Transformer. There is no second,
post-fusion HyperNet in this profile.

The HyperNet final layer and fusion gate are zero-initialized. With identical author
weights, the initial action predictions exactly match unmodified ACT. Condition inputs
contain only observations, not expert actions, future observations, or the action-derived
training CVAE latent. Original action loss, KL loss, ResNet weight sharing, image
normalization, positional encodings, and temporal aggregation remain unchanged.
Adapter parameters join the main AdamW group at the author's main learning rate.

## All published tasks

Eight tasks have published demonstrations: `lift_can`, `insert_hole`, `insert_HDMI`,
`lift_bottle`, `pull_out_key`, `put_bottle_in_shelf`, `insert_tube`, and `grasp_classify`.
The additional camera settings for `insert_card` and `insert_lean` do not imply that
their demonstrations are available in the published dataset.

```bash
source scripts/env.sh
python scripts/train_act_suite.py launch
python scripts/train_act_suite.py status /path/to/suite
```

The suite downloads and SHA256-verifies only the first 50 demonstrations per task,
pinning each file to its published immutable revision. Four CPU preparation jobs
and up to four single-GPU training jobs run concurrently. Each training array task
depends only on its matching preparation task. Task-local registries prevent concurrent
preprocessors from overwriting each other's entries. No simulator evaluation runs.

Each task retains its original ACT hyperparameters and gets an independent run under
`runs/<task>`. Completed runs, including every periodic checkpoint, are copied to
`checkpoints/<task>` together with normalization statistics, configuration, source,
logs, data provenance, and checkpoint SHA256 hashes in `tasks/<task>/complete.json`.
The suite enables a 10% GPU-memory floor after the first optimizer step. If the real
training allocator already exceeds that floor, no extra buffer is allocated; otherwise
a persistent non-model byte buffer fills only the shortfall. Batch size, precision,
architecture, optimizer, losses, random number state, and checkpoint tensors do not
change. Measured reservation and any buffer size are recorded in `gpu_memory.json`.

## Launch

Prepare a config without requesting a GPU or starting training:

```bash
source scripts/env.sh
python scripts/prepare_act_config.py lift_can --output "$UNIVTAC_RUN_ROOT/lift_can-hypernet.yml"
```

Download the complete selected task, then train inside a GPU allocation:

```bash
bash data/download.sh --task lift_can --version 45
bash scripts/train_act_hypernet.sh lift_can 50 0
```

The launcher checks GPU allocation, resolves author settings for that task, verifies
pretrained weights, reuses only matching processed data, and trains without launching
simulation. A fourth argument can specify an explicit author-format training YAML;
the task camera/state protocol and published tactile initialization still apply.
The ordinary `scripts/train_act.sh` now also uses the author input/pretraining protocol.

Each run lives in `$UNIVTAC_RUN_ROOT/ACT-hypernet/<task>/<timestamp>-seed...` and retains
the effective config, source archive, dependency lock, data normalization, training logs,
all periodic checkpoints, and best/last weights. Each save uses a unique optimization
step filename; no periodic checkpoint pruning is enabled.

## Bounded checks and deployment

```bash
python scripts/check_act_hypernet.py --task lift_can --device cpu
python scripts/check_act_hypernet.py --task insert_hole --device cuda
python scripts/check_act_hypernet.py --task lift_can --device cuda \
    --dataset "$UNIVTAC_PROCESSED_ROOT/sim-lift_can/demo-2"
```

These checks verify single/dual camera protocols, identical baseline initialization,
actual tactile pretrained tensors, adapter/fusion gradients, parameter registration,
batch independence, observation-only conditioning, and checkpoint roundtrips. The model
checks reduce the Transformer layers/chunk solely for bounded validation. A full ACT
batch-64/chunk-50 training preflight is separate from a formal experiment.

On the compatible simulation machine, copy the run directory and set:

```bash
export ACT_RUN_DIR=/path/to/copied/run
bash eval_policy.sh lift_can demo ACT/deploy 0
```

The ACT deployment adapter reads `resolved_config.yml` from that run and strictly
loads the full policy, including HyperNet parameters. It accepts both `*_tactile`
and `*_gsmini` tactile observation names. The training-time tactile checkpoint is
not needed after loading the full trained policy. Isaac Sim/Isaac Lab/TacEx/UIPC
and task assets are still required for rollout; rollout success rates have not
been verified on this training cluster.
