# Offline policy training on the cluster

## Scope

Offline imitation learning from the released demonstrations does not require Isaac Sim,
Isaac Lab, TacEx, cuRobo, a CUDA compiler, or administrator privileges. The policy-only
environment installs dependencies for ACT, Ablation, and ViTAL. ACT is the validated
starting point; UniT and SmolVLA have separate requirements and are not installed by this path.

The cluster partitions inspected during setup advertise H200 GPUs. These can be used for
PyTorch policy training, but do not treat them as supported Isaac Sim rendering hardware.
NVIDIA's [Isaac Sim requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)
exclude GPUs without RT Cores. Simulation data collection and benchmark success-rate
evaluation require a separate compatible RTX machine and the full simulator installation.

## Install and activate

```bash
cd /public/home/wangzihao/projects/UniVTAC
bash scripts/install.sh --policy-only
source scripts/env.sh
```

The configured environment is `UniVTAC` at `/data/wangzihao/conda_envs/UniVTAC`.
The ignored local `.env` configures download caches, raw/processed datasets, and run outputs
under `/data/wangzihao`. Existing environments are not modified. Re-running the installer
reuses installed packages. The installer supports `POLICY_FIND_LINKS` for a local wheelhouse
and `POLICY_OFFLINE=1` when that wheelhouse contains all required packages.

Validate imports, a forward/backward pass, an optimizer update, checkpoint save/load,
and action inference without requesting a GPU:

```bash
python scripts/check_policy_env.py --device cpu
```

This is a reduced-model dependency smoke check, not a benchmark result.

## Download and preprocess data

The `main` branch uses Isaac Sim 4.5 demonstrations. For a complete task download:

```bash
source scripts/env.sh
bash data/download.sh --task lift_can --version 45
cd policy/ACT
python process_data.py lift_can demo 50
```

The download command retrieves the entire selected task, not just 50 episodes. Check its
size before requesting many tasks. Preprocessing selects the first requested episodes.
Published data at `$UNIVTAC_DATA_ROOT/isaac45/<task>/hdf5` and legacy data at
`$UNIVTAC_DATA_ROOT/<task>/demo` are supported. Processed ACT episodes and the task registry
are placed under `$UNIVTAC_DATA_ROOT/processed/ACT`, outside the repository. Joint/action
data for the author ACT protocol is now placed under its `author` subdirectory.
Recorded nine-joint Franka data is converted to the author's eight-dimensional
state/action representation, matching deployment (seven arm joints plus one finger).
Task camera choices follow `policy/task_settings.json`; `lift_can` uses head and wrist.

Only two `lift_can` episodes were downloaded during setup to test the real data pipeline.
They are installation samples, not a sufficient dataset for a formal training experiment.

## Train ACT

Refresh GPU availability before requesting resources:

```bash
sinfo -N -O NodeList,Partition,StateCompact,Gres,GresUsed
scontrol show partition gpux
```

Use a GPU partition permitted by your account, then request an allocation, for example:

```bash
salloc -p gpux --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=12:00:00
srun --pty bash
```

Inspect `scontrol show job "$SLURM_JOB_ID" -dd` and the allocated GPU before launching.
Do not train on GPUs exposed by a CPU-only allocation. With one allocated GPU, use logical
`cuda:0`; do not substitute a physical `SLURM_JOB_GPUS` index into the training config.

Inside the GPU allocation:

```bash
cd /public/home/wangzihao/projects/UniVTAC
source scripts/env.sh
python scripts/check_policy_env.py --device cuda
bash scripts/train_act.sh lift_can 50 0
```

The wrapper checks CUDA access, preprocesses missing data, and starts training only.
Unlike `policy/ACT/one.sh`, it does not automatically start simulation evaluation.
The wrapper resolves `train_config.yml` or `train_config_all.yml` for the task and
requires the published tactile pretrained encoder at `$UNIVTAC_TACTILE_CHECKPOINT`.
ImageNet-pretrained visual weights and original ACT training hyperparameters are retained.
The old `train_config_cluster.yml` is an installation smoke config, not the author
training protocol. The trainer saves `resolved_config.yml` with effective settings.
For joint-condition HyperNet training, use `bash scripts/train_act_hypernet.sh lift_can 50 0`;
see [ACT HyperNet](./ACTHyperNet.md) for implementation, input protocol, and deployment.

Each run gets a unique directory at `$UNIVTAC_RUN_ROOT/ACT-baseline/<task>/...` or
`$UNIVTAC_RUN_ROOT/ACT-hypernet/<task>/...`, with configuration,
dependency versions, Git revision, normalization statistics, logs, best/last checkpoints,
and every periodic checkpoint. Periodic checkpoint filenames include the optimization
step so multiple saves in one epoch cannot overwrite each other. Keep all periodic
checkpoints and archive the completed run before a long experiment finishes.

ViTAL additionally needs its CLIP encoder pretraining or appropriate pretrained encoder
checkpoints. Installing its Python dependencies alone does not supply those weights.
