#!/usr/bin/env python3
"""CPU-only interface checks; does not launch Isaac or run a simulation rollout."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from policy.multitask_dp import Policy
from policy.multitask_dp.schema import CONDITION_NAMES, IMAGE_KEYS, TASK_TEXTS, fingerprint, presence_table


class StubModel(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        self.marker = nn.Parameter(torch.zeros(1))
        self.register_buffer('condition_presence', torch.from_numpy(presence_table()))
        self.calls = []

    def predict_action(self, observations: dict, presence: torch.Tensor) -> dict:
        self.calls.append(({key: value.clone() for key, value in observations.items()}, presence.clone()))
        actions = torch.arange(self.config['n_action_steps'], device=self.marker.device).reshape(1, -1, 1)
        return {'action': (actions + self.marker).expand(1, -1, 8)}


class StubTask:
    device = torch.device('cpu')

    def __init__(self) -> None:
        self.actions = []

    def take_action(self, action: torch.Tensor, action_type: str):
        assert action_type == 'qpos' and action.shape == (8,) and action.dtype == torch.float32
        self.actions.append(action.clone())
        return True, False


class StubStage2Model(StubModel):
    def __init__(self, config: dict, utility_rms) -> None:
        super().__init__(config)
        self.register_buffer('utility_rms', torch.as_tensor(utility_rms, dtype=torch.float32))


def observation(value: int = 24) -> dict:
    image = np.full((5, 7, 3), [value, 80, 160], dtype=np.uint8)
    return {'observation': {key: {'rgb': torch.from_numpy(image.copy())} for key in ('head', 'wrist')},
            'tactile': {'left_gsmini': {'rgb_marker': image.copy()}, 'right_tactile': {'rgb_marker': image.copy()}},
            'embodiment': {'joint': torch.arange(9, dtype=torch.float32) + value}}


class DeploymentChecks(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix='univtac_dp_deploy_check_')
        self.addCleanup(self.temporary.cleanup)
        self.bundle = Path(self.temporary.name)
        (self.bundle / 'checkpoints').mkdir()
        self.config = {'n_obs_steps': 2, 'n_action_steps': 8}
        self.protocol = {'config': self.config, 'cache_id': 'fixture', 'conditions': CONDITION_NAMES}
        raw = StubModel(self.config).state_dict()
        raw['marker'] = torch.tensor([1.0])
        averaged = copy.deepcopy(raw)
        averaged['marker'] = torch.tensor([3.0])
        self.payload = {'config': self.config, 'protocol': self.protocol,
                        'protocol_id': fingerprint(self.protocol), 'policy': raw, 'ema': averaged}
        self.checkpoint = self.bundle / 'checkpoints/epoch0600.ckpt'
        torch.save(self.payload, self.checkpoint)
        language = np.arange(8 * 512, dtype=np.float32).reshape(8, 512)
        np.save(self.bundle / 'language.npy', language)
        self.manifest = {'cache_id': 'fixture', 'image_size': 8, 'tasks': list(TASK_TEXTS),
                         'language': {'sha256': hashlib.sha256(language.tobytes()).hexdigest(), 'task_texts': TASK_TEXTS}}
        self.manifest_path = self.bundle / 'language_manifest.json'
        self.manifest_path.write_text(json.dumps(self.manifest))
        environment = {key: value for key, value in os.environ.items() if not key.startswith('UNIVTAC_DP_')}
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, environment, clear=True).start()
        patch('policy.multitask_dp.model.FullSensorPolicy', StubModel).start()
        self.args = {'task_name': 'lift_bottle', 'bundle_dir': str(self.bundle), 'device': 'cpu'}

    def test_loading_and_observations(self) -> None:
        policy = Policy(self.args)
        self.assertEqual(policy.model.marker.item(), 3)
        inputs = policy.encode_obs(observation())
        for key in IMAGE_KEYS:
            self.assertEqual(inputs[key].shape, (1, 2, 3, 8, 8))
            self.assertEqual(inputs[key].dtype, torch.uint8)
            self.assertTrue(torch.equal(inputs[key][0, 0], inputs[key][0, 1]))
            self.assertEqual(inputs[key][0, 0, :, 0, 0].tolist(), [24, 80, 160])
        self.assertEqual(inputs['state'][0, 0].tolist(), list(range(24, 32)))
        self.assertEqual(inputs['language'].shape, (1, 2, 512))
        row = self.manifest['tasks'].index('lift_bottle')
        np.testing.assert_array_equal(inputs['language'][0, 0].numpy(), np.load(self.bundle / 'language.npy')[row])
        inputs = policy.encode_obs(observation(30))
        self.assertEqual(inputs['head'][0, :, 0, 0, 0].tolist(), [24, 30])
        image = observation()['observation']['head']['rgb'].numpy()
        encoded = cv2.imencode('.jpg', image)[1]
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        self.assertLess(np.abs(decoded.astype(int) - image.astype(int)).max(), 4)

    def test_queue_history_and_reset(self) -> None:
        policy = Policy(self.args)
        task = StubTask()
        for step in range(9):
            policy.eval(task, observation(24 + step))
        self.assertEqual(len(task.actions), 9)
        self.assertEqual([action[0].item() for action in task.actions], [3, 4, 5, 6, 7, 8, 9, 10, 3])
        self.assertEqual(len(policy.model.calls), 2)
        latest = policy.model.calls[-1][0]
        self.assertEqual(latest['head'][0, :, 0, 0, 0].tolist(), [31, 32])
        policy.reset()
        self.assertEqual(len(policy.history), 0)
        self.assertEqual(len(policy.actions), 0)
        policy.eval(task, observation(50))
        self.assertEqual(task.actions[-1][0].item(), 3)
        self.assertEqual(policy.model.calls[-1][0]['head'][0, :, 0, 0, 0].tolist(), [50, 50])

    def test_sensor_removal_and_required_inputs(self) -> None:
        policy = Policy({**self.args, 'condition': 'vision'})
        inputs = policy.encode_obs({'tactile': observation()['tactile'], 'embodiment': observation()['embodiment']})
        self.assertEqual(inputs['head'].count_nonzero().item(), 0)
        self.assertEqual(policy.presence[0].tolist(), [False, False, True, True, True, True])
        full = Policy(self.args)
        with self.assertRaisesRegex(ValueError, 'Missing required sensor head'):
            full.encode_obs({})
        bad = observation()
        bad['observation']['head']['rgb'] = torch.zeros(3, 8, 8)
        with self.assertRaisesRegex(ValueError, 'uint8 HWC'):
            full.encode_obs(bad)

    def test_environment_epoch_and_raw_weights(self) -> None:
        torch.save(self.payload, self.bundle / 'checkpoints/epoch0100.ckpt')
        with patch.dict(os.environ, {'UNIVTAC_DP_EPOCH': '100', 'UNIVTAC_DP_WEIGHTS': 'policy'}):
            policy = Policy(self.args)
        self.assertEqual(policy.model.marker.item(), 1)

    def test_incompatible_bundle_rejected(self) -> None:
        self.manifest['cache_id'] = 'wrong-cache'
        self.manifest_path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, 'different training caches'):
            Policy(self.args)
        self.manifest['cache_id'] = 'fixture'
        self.manifest['language']['sha256'] = '0' * 64
        self.manifest_path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            Policy(self.args)

    def test_bad_predictions_rejected(self) -> None:
        policy = Policy(self.args)
        with patch.object(policy.model, 'predict_action', return_value={'action': torch.zeros(1, 7, 8)}):
            with self.assertRaisesRegex(ValueError, 'finite'):
                policy.eval(StubTask(), observation())

    def stage2_checkpoint(self) -> dict:
        contract = {'kind': 'univtac-stage2-transformer-t2-v1',
                    'teacher': {'config': self.config, 'cache_id': 'fixture'},
                    'utility_rms': [0.1] * 8}
        raw = StubStage2Model(self.config, contract['utility_rms']).state_dict()
        raw['marker'] = torch.tensor([2.0])
        averaged = copy.deepcopy(raw)
        averaged['marker'] = torch.tensor([4.0])
        payload = {'kind': contract['kind'], 'contract': contract, 'contract_id': fingerprint(contract),
                   'policy_config': copy.deepcopy(self.config), 'model': raw, 'ema': averaged}
        torch.save(payload, self.checkpoint)
        patch('policy.multitask_dp.stage2.Stage2Policy', StubStage2Model).start()
        return payload

    def test_stage2_ema_raw_and_action_interface(self) -> None:
        self.stage2_checkpoint()
        policy = Policy(self.args)
        self.assertIsInstance(policy.model, StubStage2Model)
        self.assertEqual(policy.model.marker.item(), 4)
        task = StubTask()
        policy.eval(task, observation())
        self.assertEqual(task.actions[0][0].item(), 4)
        self.assertTrue(policy.model.calls[0][1].all())
        with patch.dict(os.environ, {'UNIVTAC_DP_WEIGHTS': 'policy'}):
            self.assertEqual(Policy(self.args).model.marker.item(), 2)

    def test_stage2_contract_config_and_rms_rejection(self) -> None:
        original = self.stage2_checkpoint()
        for mutation, message in [('contract', 'contract identity'), ('config', 'policy config'),
                                  ('rms', 'utility RMS')]:
            with self.subTest(mutation=mutation):
                payload = copy.deepcopy(original)
                if mutation == 'contract':
                    payload['contract']['teacher']['cache_id'] = 'tampered'
                elif mutation == 'config':
                    payload['policy_config']['n_obs_steps'] = 3
                else:
                    payload['ema']['utility_rms'].fill_(1)
                torch.save(payload, self.checkpoint)
                with self.assertRaisesRegex(ValueError, message):
                    Policy(self.args)

    def test_stage2_sensor_removal_rejected(self) -> None:
        self.stage2_checkpoint()
        with self.assertRaisesRegex(ValueError, 'all six sensors'):
            Policy({**self.args, 'condition': 'vision'})

    def test_unknown_checkpoint_kind_rejected(self) -> None:
        self.stage2_checkpoint()
        payload = {'kind': 'unknown-architecture'}
        torch.save(payload, self.checkpoint)
        with self.assertRaisesRegex(ValueError, 'Unsupported checkpoint kind'):
            Policy(self.args)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
