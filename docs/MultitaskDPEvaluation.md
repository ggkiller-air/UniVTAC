# Evaluate multi-task DP using the existing UniVTAC evaluator

UniVTAC already supplies the simulation loop, task resets, success checks and
result logging in `scripts/eval_policy.py`. The DP-specific `Policy` wrapper
converts observations and executes actions; it does not replace that framework.
Use the existing rendering-capable evaluation server, not the H200 training node.

## Stage 2 Transformer checkpoints (job 392551)

Pull the latest `main` from `https://github.com/ggkiller-air/UniVTAC` on the
evaluation server. The adapter detects `univtac-stage2-transformer-t2-v1`
checkpoints and builds `Stage2Policy`, restoring its Transformer, utility head,
utility RMS and normalizer strictly. Stage 0 checkpoints remain supported.

ModelScope directory `20261027_Stage2Transformer_job392551/checkpoints` contains
`epoch0010.ckpt`, `epoch0020.ckpt` and `latest.ckpt`. This uploaded `latest` is a
fixed snapshot at epoch 43, cursor 692, global step 92,000; it does not follow
the still-running training job. Each file is 4,840,107,598 bytes.

The same repository's `20261020_MultitaskDP/language.npy` and
`language_manifest.json` supply the original eight task embeddings. Reuse those
two files if already downloaded. No teacher checkpoint, utility labels or
training dataset is needed for inference. The adapter verifies that the
embeddings match the Stage 2 teacher's training-cache identity.

Download a selected checkpoint and these two small companion files directly
over HTTPS, without a proxy or git transfer:

```bash
export UNIVTAC_DP_BUNDLE=/absolute/path/to/univtac_stage2_eval
mkdir -p "$UNIVTAC_DP_BUNDLE/checkpoints"
checkpoint=latest.ckpt  # or epoch0010.ckpt / epoch0020.ckpt
base=https://www.modelscope.cn/models/ggkiller/UniViTac_ckpts/resolve/master
for name in language.npy language_manifest.json; do
  curl --noproxy '*' --fail --location --retry 3 \
    "$base/20261020_MultitaskDP/$name" -o "$UNIVTAC_DP_BUNDLE/$name"
done
curl --noproxy '*' --fail --location --retry 3 \
  "$base/20261027_Stage2Transformer_job392551/checkpoints/$checkpoint" \
  -o "$UNIVTAC_DP_BUNDLE/checkpoints/$checkpoint"
export UNIVTAC_DP_CHECKPOINT="$UNIVTAC_DP_BUNDLE/checkpoints/$checkpoint"
export UNIVTAC_DP_CONDITION=full
export UNIVTAC_DP_WEIGHTS=ema
CUDA_VISIBLE_DEVICES=0 python scripts/eval_policy.py \
  lift_bottle clean multitask_dp/deploy --headless --total_num 100 --device cuda:0
```

Run the final command from the UniVTAC checkout in its working Isaac Lab / TacEx
environment. Keep the simulator, scene assets and compatible torch installation
already used for UniVTAC evaluation. Stage 2 supports `full` sensors only;
Stage 0 removal conditions are rejected. `UNIVTAC_DP_WEIGHTS=policy` explicitly
selects raw student weights (stored as `model` in Stage 2 checkpoints); `ema`
is the default. Repeat with each checkpoint and task as needed.

SHA-256 values for the uploaded snapshots:

```text
3d9196708178cba688e509b3a4104bc5206324208dc2d7253740d046a4a32ea8  epoch0010.ckpt
27df7b8a7e5c08fddd3be9642eed7e8ca257893ec815609afb97255f2dce05fe  epoch0020.ckpt
3e84534c5aeaa2ea1421c84626dde5b56abb56eff4b1402ab00f080ae166bb95  latest.ckpt
```

## Stage 0 deployment overlay for older checkouts

A checkout updated to the latest GitHub `main` already includes the adapter.
The older deployment archive below is only needed for a checkout without that
adapter; extracting it over the latest code would replace Stage 2 support.

The ModelScope repository `ggkiller/UniViTac_ckpts`, directory
`20261020_MultitaskDP`, provides six checkpoints and the companion files.
Download `deployment_adapter.tar.gz` and `SHA256SUMS` as well if you previously
downloaded the checkpoint bundle before the adapter was added.

In an existing, working UniVTAC checkout on the evaluation server:

```bash
tar -xzf /path/to/20261020_MultitaskDP/deployment_adapter.tar.gz -C /path/to/UniVTAC
cd /path/to/UniVTAC
export UNIVTAC_DP_BUNDLE=/absolute/path/to/20261020_MultitaskDP
export UNIVTAC_DP_EPOCH=600
bash eval_policy.sh lift_bottle clean multitask_dp/deploy 0
```

Replace `lift_bottle` with the task of interest. Use `demo` instead of `clean`
only if a visible simulator window is intended. The existing `clean.yml` task
configuration requests both camera RGB and tactile `rgb_marker` observations.
The existing simulator/scene assets are reused; do not reinstall them just to
switch policies. This overlay includes the complete small `policy/multitask_dp`
package, so no Git operation is required to install it.

To explicitly choose headless mode and the episode budget, use the underlying
evaluation entrypoint:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_policy.py \
    lift_bottle clean multitask_dp/deploy --headless --total_num 100 --device cuda:0
```

Select `UNIVTAC_DP_EPOCH=100`, `200`, `300`, `400`, `500` or `600` for the
requested checkpoints. Repeat the command for each checkpoint and task. Outputs
use the existing `eval_result/multitask_dp/<task>/deploy/<timestamp>/` layout;
the evaluator logs the loaded checkpoint path at policy initialization.

## Model dependencies

Keep the existing Isaac/torch installation on the evaluation server. If policy
imports fail, add compatible diffusion-policy dependencies to that environment
without blindly replacing torch, torchvision, NumPy or the simulator runtime.
The training versions are recorded in `requirements.txt`; in particular the
policy uses `diffusers==0.11.1`, paired during training with
`huggingface-hub==0.25.0`. No CLIP model download is needed for the eight tasks:
the adapter reads `language.npy` and validates its identity against the checkpoint.

## Policy behavior and overrides

- Defaults: EMA weights, full sensors, FP32 inference, original 100 DDPM steps.
- `UNIVTAC_DP_WEIGHTS=policy` selects the non-EMA weights for an explicit comparison.
- `UNIVTAC_DP_CHECKPOINT=/absolute/path/to/file.ckpt` overrides the epoch selection.
- `UNIVTAC_DP_DEVICE=cuda:0` selects the policy device after CUDA visibility mapping.
- `UNIVTAC_DP_CONDITION` selects a trained removal condition: `full`, `head`,
  `wrist`, `tactile_left`, `tactile_right`, `state`, `language`, `vision`, `tactile`.
  A sensor name means that sensor is removed, not kept alone.
- Two-frame causal history is filled by duplicating the first observation after
  every reset. Eight queued qpos actions execute one per evaluator call; history
  is updated on every call and is not stale between replans.
- The cached task-specific CLIP vector is used for each task, matching training;
  arbitrary simulator instruction paraphrases are not re-encoded.
- Raw simulator image channel order is preserved, with OpenCV bilinear resize.
  The repository's data writer passes these arrays directly to `cv2.imencode`
  without a color swap, and training uses `cv2.imdecode` without a color swap.
  Do not add an extra RGB/BGR reversal solely because the cache convention is
  described as OpenCV BGR. Inputs remain uint8 and normalize inside the model.

## Validation scope

CPU interface tests can be run without Isaac or CUDA:

```bash
CUDA_VISIBLE_DEVICES='' python scripts/check_multitask_dp_deploy.py
```

These tests check adapter input/output, loading choices, history/queue/reset and
identity rejection with a small mocked predictor. They are not simulation
rollouts or a measurement of success rate. Actual simulation evaluation is left
to the rendering-capable server.
