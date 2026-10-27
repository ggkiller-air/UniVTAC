"""Bridge the existing UniVTAC evaluation loop to the multi-task diffusion policy."""

from __future__ import annotations

from collections import deque
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from .._base_policy import BasePolicy
from .schema import CONDITION_NAMES, IMAGE_KEYS, SENSOR_KEYS, TASK_TEXTS, fingerprint


class Policy(BasePolicy):
    """Maintain causal observations and execute one queued qpos action per eval call."""

    def __init__(self, args: dict) -> None:
        super().__init__(args)
        from .model import FullSensorPolicy

        bundle_value = os.environ.get('UNIVTAC_DP_BUNDLE') or args.get('bundle_dir')
        if not bundle_value:
            raise ValueError('Set UNIVTAC_DP_BUNDLE to the directory containing language.npy and language_manifest.json')
        bundle = Path(bundle_value).expanduser().resolve()
        epoch = int(os.environ.get('UNIVTAC_DP_EPOCH', args.get('epoch', 600)))
        checkpoint_value = os.environ.get('UNIVTAC_DP_CHECKPOINT') or args.get('checkpoint')
        checkpoint = Path(checkpoint_value).expanduser() if checkpoint_value else bundle / 'checkpoints' / f'epoch{epoch:04d}.ckpt'
        if not checkpoint.is_absolute():
            checkpoint = bundle / checkpoint
        self.task_name = args['task_name']
        if self.task_name not in TASK_TEXTS:
            raise ValueError(f'No trained task embedding for {self.task_name!r}')
        self.device = torch.device(os.environ.get('UNIVTAC_DP_DEVICE') or args.get('device', 'cuda:0'))
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; simulation evaluation requires a rendering-capable GPU')
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
        self.checkpoint_kind = payload.get('kind', 'stage0')
        stage2 = self.checkpoint_kind == 'univtac-stage2-transformer-t2-v1'
        if stage2:
            contract = payload['contract']
            if fingerprint(contract) != payload['contract_id'] or contract['kind'] != self.checkpoint_kind:
                raise ValueError('Stage 2 checkpoint contract identity is incompatible')
            config = payload['policy_config']
            if config != contract['teacher']['config']:
                raise ValueError('Stage 2 policy config differs from its teacher contract')
            cache_id = contract['teacher']['cache_id']
        elif self.checkpoint_kind == 'stage0':
            protocol = payload['protocol']
            if fingerprint(protocol) != payload['protocol_id'] or tuple(protocol['conditions']) != CONDITION_NAMES:
                raise ValueError('Checkpoint protocol identity or sensor conditions are incompatible')
            config = payload['config']
            if config != protocol['config']:
                raise ValueError('Checkpoint config differs from its recorded protocol')
            cache_id = protocol['cache_id']
        else:
            raise ValueError(f'Unsupported checkpoint kind: {self.checkpoint_kind}')
        language_manifest = json.loads((bundle / 'language_manifest.json').read_text())
        language = np.load(bundle / 'language.npy', allow_pickle=False)
        task_names = language_manifest['tasks']
        if language_manifest['cache_id'] != cache_id:
            raise ValueError('Language features and checkpoint refer to different training caches')
        if (language.shape != (len(task_names), 512) or language.dtype != np.float32
                or not np.isfinite(language).all() or len(set(task_names)) != len(task_names)):
            raise ValueError('Invalid task embedding array or task mapping')
        if hashlib.sha256(language.tobytes()).hexdigest() != language_manifest['language']['sha256']:
            raise ValueError('Language embedding SHA256 mismatch')
        if language_manifest['language']['task_texts'] != TASK_TEXTS or self.task_name not in task_names:
            raise ValueError('Language task instructions differ from the training protocol')
        self.language = torch.from_numpy(language[task_names.index(self.task_name)].copy())
        self.image_size = int(language_manifest['image_size'])
        self.history_length = int(config['n_obs_steps'])
        self.action_steps = int(config['n_action_steps'])
        self.condition = os.environ.get('UNIVTAC_DP_CONDITION', args.get('condition', 'full'))
        if self.condition not in CONDITION_NAMES:
            raise ValueError(f'Unknown sensor condition: {self.condition}')
        if stage2 and self.condition != 'full':
            raise ValueError('Stage 2 inference expects all six sensors; condition must be full')
        self.weights = os.environ.get('UNIVTAC_DP_WEIGHTS', args.get('weights', 'ema'))
        if self.weights not in ('ema', 'policy'):
            raise ValueError('weights must be ema or policy')
        torch.manual_seed(int(args.get('seed', 0)))
        if stage2:
            from .stage2 import Stage2Policy

            self.model = Stage2Policy(config, utility_rms=contract['utility_rms'])
        else:
            self.model = FullSensorPolicy(config)
        weights_key = 'model' if stage2 and self.weights == 'policy' else self.weights
        self.model.load_state_dict(payload[weights_key], strict=True)
        if stage2 and not torch.equal(self.model.utility_rms.cpu(), torch.as_tensor(contract['utility_rms'], dtype=torch.float32)):
            raise ValueError('Stage 2 utility RMS differs from its recorded contract')
        self.model.to(self.device).eval().requires_grad_(False)
        self.presence = self.model.condition_presence[CONDITION_NAMES.index(self.condition)].unsqueeze(0)
        self.active_sensors = dict(zip(SENSOR_KEYS, self.presence[0].cpu().tolist()))
        self.reset()
        print(f'Multitask DP loaded {checkpoint} kind={self.checkpoint_kind} weights={self.weights} task={self.task_name} '
              f'condition={self.condition} device={self.device}', flush=True)

    @staticmethod
    def _array(value: object) -> np.ndarray:
        return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)

    def _image(self, observation: dict, sensor: str) -> torch.Tensor:
        if not self.active_sensors[sensor]:
            return torch.zeros((3, self.image_size, self.image_size), dtype=torch.uint8)
        if sensor in ('head', 'wrist'):
            image = observation.get('observation', {}).get(sensor, {}).get('rgb')
        else:
            side = sensor[len('tactile_'):]
            tactile = observation.get('tactile', {})
            stream = next((tactile[key] for key in (f'{side}_gsmini', f'{side}_tactile') if key in tactile), {})
            image = stream.get('rgb_marker')
        if image is None:
            raise ValueError(f'Missing required sensor {sensor}; enable both cameras and rgb_marker tactile observations')
        image = self._array(image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f'{sensor} must be a uint8 HWC three-channel image')
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        return torch.from_numpy(np.ascontiguousarray(np.moveaxis(image, -1, 0)))

    def encode_obs(self, observation: dict) -> dict[str, torch.Tensor]:
        frame = {sensor: self._image(observation, sensor) for sensor in IMAGE_KEYS}
        if self.active_sensors['state']:
            joints = self._array(observation.get('embodiment', {}).get('joint'))
            if joints.ndim != 1 or joints.size < 8 or not np.isfinite(joints[:8]).all():
                raise ValueError('Expected at least eight finite joint values in embodiment/joint')
            frame['state'] = torch.from_numpy(np.array(joints[:8], dtype=np.float32, copy=True))
        else:
            frame['state'] = torch.zeros(8)
        frame['language'] = self.language
        if not self.history:
            self.history.extend([frame] * self.history_length)
        else:
            self.history.append(frame)
        return {key: torch.stack([item[key] for item in self.history]).unsqueeze(0).to(self.device)
                for key in SENSOR_KEYS}

    @torch.inference_mode()
    def eval(self, task, observation: dict) -> None:
        observations = self.encode_obs(observation)
        if not self.actions:
            actions = self.model.predict_action(observations, presence=self.presence)['action']
            if actions.shape != (1, self.action_steps, 8) or not torch.isfinite(actions).all():
                raise ValueError('Policy must return finite [1,n_action_steps,8] joint positions')
            self.actions.extend(actions[0].detach().clone().unbind(0))
        action = self.actions.popleft().to(device=task.device, dtype=torch.float32)
        task.take_action(action, action_type='qpos')

    def reset(self) -> None:
        self.history = deque(maxlen=self.history_length)
        self.actions = deque()
