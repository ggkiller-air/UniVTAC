import argparse
import copy
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml


def check_adapter():
    from detr.models.hypernet import JointConditionHyperNet

    for visual_count in (1, 2):
        adapter = JointConditionHyperNet(16, 8, visual_count, 2)
        features = [torch.randn(2, 16, 9) for _ in range(visual_count)]
        features += [torch.randn(2, 16, 1), torch.randn(2, 16, 1)]
        state = torch.randn(2, 8)
        for original, adapted in zip(features, adapter(features, state)):
            torch.testing.assert_close(adapted, original, rtol=0, atol=0)
        torch.testing.assert_close(adapter.alpha, torch.tensor(0.01))
        assert adapter.alpha.item() < 0.1
        with torch.no_grad():
            adapter.hypernet[-1].weight.normal_(std=0.01)
            adapter.fusion_gate.fill_(0.2)
        result = adapter(features, state)
        per_sample = adapter([feature[:1] for feature in features], state[:1])
        for full, single in zip(result, per_sample):
            torch.testing.assert_close(full[:1], single, rtol=1e-5, atol=1e-6)
        different_state = adapter(features, state + 1)
        assert any(not torch.equal(original, changed) for original, changed in zip(result, different_state))
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4)
        sum(feature.square().mean() for feature in result).backward()
        for parameter in adapter.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert adapter.hypernet[0].weight.grad.abs().sum() > 0
        assert adapter.cross_attention.in_proj_weight.grad.abs().sum() > 0
        optimizer.step()
        assert 0 < adapter.alpha.item() < 0.1
    print("Single/dual-camera identity, conditional modulation, fusion, gradients and batch independence passed")


def main():
    parser = argparse.ArgumentParser(description="Check author ACT protocol and HyperResNet integration without rollout.")
    parser.add_argument("--task", default="lift_can")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "policy" / "ACT"))
    from act_policy import ACTPolicy
    from detr.models.backbone import build_tactile_backbone
    from imitate_episodes import forward_pass
    from process_data import ACTDataPreprocessor
    from protocol import ACT_ROOT, author_training_config, task_observation_spec, validate_processed_task
    from utils import TacArenaDataset, get_norm_stats

    if args.device == "cuda":
        if os.environ.get("SLURM_JOB_ID") and not (os.environ.get("SLURM_JOB_GPUS") or os.environ.get("SLURM_STEP_GPUS")):
            raise RuntimeError("The current Slurm allocation has no GPU")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
    torch.set_num_threads(min(4, len(os.sched_getaffinity(0))))
    torch.manual_seed(0)
    check_adapter()
    task_settings = json.loads((ACT_ROOT.parent / "task_settings.json").read_text())
    for task_name, settings in task_settings.items():
        spec = task_observation_spec(task_name)
        expected_cameras = ['cam_high', 'cam_wrist'] if settings['camera_type'] == 'all' else ['cam_high']
        assert spec['camera_names'] == expected_cameras
    joints = np.arange(27, dtype=np.float32).reshape(3, 9)
    processor = ACTDataPreprocessor.__new__(ACTDataPreprocessor)
    np.testing.assert_array_equal(processor.joint_transform(joints), joints[:, :8])
    checkpoint = Path(os.environ.get("UNIVTAC_TACTILE_CHECKPOINT", repo_root / 'data/checkpoints/encoder.pth'))
    config = author_training_config(args.task, checkpoint)
    author_path = ACT_ROOT / ('train_config_all.yml' if len(config['camera_names']) == 2 else 'train_config.yml')
    author_config = yaml.safe_load(author_path.read_text())
    for key, value in author_config.items():
        if key != 'tactile_ckpt':
            assert config[key] == value, (key, config[key], value)
    assert config['use_fusion'] and not config['use_fusion_hypernet']
    config.update(device=args.device, enc_layers=1, dec_layers=1, dim_feedforward=512,
                  chunk_size=2, batch_size=2, task_name=f'sim-{args.task}-demo-2',
                  seed=0, num_epochs=1, ckpt_dir='smoke')
    if args.dataset:
        metadata = json.loads((args.dataset / 'metadata.json').read_text())
        validate_processed_task(metadata, args.task, metadata['num_episodes'])
        norm_stats, _ = get_norm_stats(str(args.dataset), metadata['num_episodes'])
        dataset = TacArenaDataset([0, 1], str(args.dataset), config['camera_names'], config['tactile_names'], norm_stats, config['chunk_size'])
        batch = torch.utils.data.default_collate([dataset[0], dataset[1]])
    else:
        batch = (
            torch.rand(2, len(config['camera_names']), 3, 64, 64),
            torch.rand(2, len(config['tactile_names']), 3, 64, 64),
            torch.rand(2, 8), torch.rand(2, 2, 8), torch.zeros(2, 2, dtype=torch.bool),
        )
    baseline_config = copy.deepcopy(config)
    baseline_config['use_hypernet'] = False
    torch.manual_seed(0)
    baseline = ACTPolicy(baseline_config)
    torch.manual_seed(0)
    policy = ACTPolicy(config)
    common_weights = baseline.state_dict()
    for name, weight in common_weights.items():
        torch.testing.assert_close(policy.state_dict()[name], weight, rtol=0, atol=0)
    pretrained = torch.load(checkpoint, map_location='cpu', weights_only=True)
    tactile_backbone = policy.model.backbones[1].backbone
    for name, weight in tactile_backbone.state_dict().items():
        torch.testing.assert_close(weight.cpu(), pretrained['backbone.' + name], rtol=0, atol=0)
    baseline.eval()
    policy.eval()
    camera, tactile, state, _, _ = batch
    device = next(policy.parameters()).device
    with torch.inference_mode():
        expected = baseline(state.to(device), camera.to(device), tactile.to(device))
        actual = policy(state.to(device), camera.to(device), tactile.to(device))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    del baseline, common_weights, pretrained
    adapter = policy.model.perception_adapter
    optimizer = policy.configure_optimizers()
    parameters = [parameter for group in optimizer.param_groups for parameter in group['params']]
    assert len({id(parameter) for parameter in parameters}) == len(parameters)
    for parameter in adapter.parameters():
        assert sum(parameter is candidate for candidate in parameters) == 1
    policy.train()
    contexts = []
    handle = adapter.hypernet.register_forward_pre_hook(lambda module, inputs: contexts.append(inputs[0].detach().clone()))
    losses = []
    for _ in range(2):
        optimizer.zero_grad()
        loss = forward_pass(batch, policy)['loss']
        assert torch.isfinite(loss)
        loss.backward()
        for parameter in adapter.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        optimizer.step()
        losses.append(loss.item())
    assert adapter.hypernet[0].weight.grad.abs().sum() > 0
    assert adapter.cross_attention.in_proj_weight.grad.abs().sum() > 0
    handle.remove()
    policy.eval()
    contexts.clear()
    handle = adapter.hypernet.register_forward_pre_hook(lambda module, inputs: contexts.append(inputs[0].detach().clone()))
    with torch.inference_mode():
        forward_pass(batch, policy)
        changed_batch = (*batch[:3], batch[3] + 1, batch[4])
        forward_pass(changed_batch, policy)
    torch.testing.assert_close(contexts[0], contexts[1], rtol=0, atol=0)
    handle.remove()
    with tempfile.TemporaryDirectory(prefix='act-hypernet-') as checkpoint_dir:
        saved_path = Path(checkpoint_dir) / 'policy.ckpt'
        torch.save(policy.state_dict(), saved_path)
        policy.load_state_dict(torch.load(saved_path, map_location=device, weights_only=True))
    try:
        build_tactile_backbone(argparse.Namespace(require_tactile_checkpoint=True, tactile_ckpt=None))
    except ValueError:
        pass
    else:
        raise AssertionError("Missing author checkpoint was silently accepted")
    result = {
        'task': args.task, 'device': str(device), 'torch': torch.__version__,
        'camera_names': config['camera_names'], 'state_dim': config['state_dim'],
        'tactile_checkpoint_sha256': config['tactile_checkpoint_sha256'],
        'dataset': str(args.dataset) if args.dataset else 'synthetic',
        'losses': losses, 'initial_output_exactly_matches_baseline': True,
        'author_pretrained_weights_verified': True, 'observation_only_condition_verified': True,
        'prediction_shape': list(actual.shape),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    print('act_hypernet_check_passed')


if __name__ == '__main__':
    main()
