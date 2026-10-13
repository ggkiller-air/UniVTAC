import sys
import json
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from .._base_policy import BasePolicy

import os
import cv2
import yaml
import numpy as np
import torch
from .act_policy import ACT
# from act_policy import ACT
from torchvision import transforms
from .protocol import ACT_PROTOCOL, ACT_STATE_DIM, task_observation_spec

class Policy(BasePolicy):
# class Policy:
    def __init__(self, args):
        """Initialize ACT policy for TacArena deployment"""
        # Construct checkpoint directory path
        self.train_config_name = os.environ.get('TRAIN_CONFIG', 'train_config')
        self.ep_num = os.environ.get('EP_NUM', '50')
        ckpt_dir = Path(__file__).parent / "act_ckpt" / f"act-{args['task_name']}" / f"{args['task_config']}-{self.ep_num}" / self.train_config_name
 
        self.task_name = args['task_name']
        run_dir = args.get('ckpt_dir') or os.environ.get('ACT_RUN_DIR')
        with open(Path(__file__).parent.parent / 'task_settings.json', 'r') as f:
            task_settings = json.load(f)
        assert self.task_name in task_settings, f"Task '{self.task_name}' not found in task_settings.json"
        self.camera_type = task_settings[self.task_name].get('camera_type', 'head')
        print(f"Using camera type '{self.camera_type}' for task '{self.task_name}'")

        config_path = Path(run_dir) / 'resolved_config.yml' if run_dir else Path(__file__).parent / f'{self.train_config_name}.yml'
        if run_dir:
            ckpt_dir = Path(run_dir)
            for required_file in ('policy_last.ckpt', 'dataset_stats.pkl', 'resolved_config.yml'):
                if not (ckpt_dir / required_file).is_file():
                    raise FileNotFoundError(ckpt_dir / required_file)
        with open(config_path, 'r') as f:
            train_config = yaml.load(f, Loader=yaml.FullLoader)
        self.camera_names = train_config['camera_names']
        self.state_dim = train_config['state_dim']
        if train_config.get('act_protocol') == ACT_PROTOCOL:
            spec = task_observation_spec(self.task_name)
            if self.camera_names != spec['camera_names'] or self.state_dim != ACT_STATE_DIM:
                raise ValueError("Deployment inputs do not match the author ACT task protocol")
        if run_dir:
            train_config['tactile_ckpt'] = None
            train_config['require_tactile_checkpoint'] = False
        if args.get('device'):
            train_config['device'] = args['device']
        
        train_config.update({
            'task_name': f"sim-{args['task_name']}-{args['task_config']}-{self.ep_num}",
            'task_config': args['task_config'],
            'ckpt_dir': str(ckpt_dir),
            "seed": args.get('seed', 0),
            "num_epochs": 1
        })
        
        # Initialize ACT model (RoboTwin_Config=None for TacArena)
        self.model = ACT(train_config)
        print(f"ACT policy loaded from {ckpt_dir}")

    def encode_obs(self, observation):
        """
        Encode TacArena observation to ACT input format
        
        Input (TacArena):
            observation = {
                "observation": {"head": {"rgb": torch.Tensor([H, W, 3])}},  # HWC, 0-255
                "joint_action": torch.Tensor([9])  # [arm(7), gripper(1), extra(1)]
            }
            camera: 480x270
            tactile: 320x240
        
        Output (ACT):
            obs = {
                "qpos": torch.Tensor([8])  # [arm(7), gripper(1)]
                "cam_high": torch.Tensor([3, 256, 256]),  # CHW, 0-1
                "tac_left": torch.Tensor([3, 256, 256]),  # CHW, 0-1
                "tac_right": torch.Tensor([3, 256, 256]),  # CHW, 0-1
            }
        """
        # Debug: observation structure validated
        # observation['embodiment']['joint'] contains joint state
        def camera_transform(img: torch.Tensor):
            img = transforms.Resize((256, 256))(img.permute(2, 0, 1))  # HWC -> CHW
            img = img / 255.0  # Normalize to [0, 1]
            img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
            return img
        
        def tactile_transform(img: torch.Tensor):
            img = transforms.Resize((256, 256))(img.permute(2, 0, 1)) # HWC -> CHW
            img = img / 255.0  # Normalize to [0, 1]
            return img

        if self.camera_type == 'all':
            cam_high = camera_transform(observation["observation"]["head"]["rgb"])
            cam_wrist = camera_transform(observation["observation"]["wrist"]["rgb"])
        else:
            cam_high = camera_transform(observation["observation"][self.camera_type]["rgb"])

        tactile_streams = observation['tactile']
        left_key = 'left_tactile' if 'left_tactile' in tactile_streams else 'left_gsmini'
        right_key = 'right_tactile' if 'right_tactile' in tactile_streams else 'right_gsmini'
        left_tac = tactile_transform(tactile_streams[left_key]['rgb_marker'])
        right_tac = tactile_transform(tactile_streams[right_key]['rgb_marker'])
        
        # Extract joint positions (8D: 7 arm + 1 gripper)
        qpos = observation["embodiment"]["joint"][:self.state_dim]

        ret = {
            "cam_high": cam_high,
            "tac_left": left_tac,
            "tac_right": right_tac,
            "qpos": qpos.cpu().numpy()
        }
        if self.camera_type == 'all':
            ret["cam_wrist"] = cam_wrist
        if 'cam_head' in self.camera_names:
            ret['cam_head'] = ret['cam_high']
        return ret

    def eval(self, task, observation):
        """
        Evaluate ACT policy on TacArena task
        
        Args:
            task: TacArena BaseTask instance
            observation: Current observation from environment
        """
        
        # Get action from ACT model (returns (1, 8) numpy array)
        obs = self.encode_obs(observation)
        if self.model.t % 10 == 0:
            self.save(task.get_frame_shot(observation), task.take_action_cnt)
        action = self.model.get_action(obs).reshape(-1)
        action = torch.from_numpy(action).to(task.device).float()
        exec_succ, eval_succ = task.take_action(action, action_type='qpos')

    def reset(self):
        """Reset ACT model state (temporal aggregation and timestep counter)"""
        if hasattr(self.model, 'reset'):
            self.model.reset()

    def save(self, img, t):
        from PIL import Image
        from PIL import ImageDraw, ImageFont
        
        obs = Image.fromarray(img.cpu().numpy())

        draw = ImageDraw.Draw(obs)
        font = ImageFont.load_default()

        draw.text((obs.width-100, obs.height-60), f'{t:03d}', fill=(255, 0, 0), font=font)
        obs.save(f'ACT_{self.task_name}_{self.train_config_name}.png')
