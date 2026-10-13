import os
import cv2
import sys
import h5py
import json
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

PROJECT_ROOT_PATH = (Path(__file__).parent / '..').resolve()
DATA_ROOT_PATH = Path(os.environ.get('UNIVTAC_DATA_ROOT', PROJECT_ROOT_PATH / 'data')).expanduser()
sys.path.append(str(PROJECT_ROOT_PATH.absolute()))

from envs.utils.data import HDF5Handler

class BaseDataPreprocessor:
    def __init__(self, task_name:str, collect_config_name:str):
        self.task_name = task_name
        self.collect_config_name = collect_config_name
        self.raw_root_path = DATA_ROOT_PATH / task_name / collect_config_name
        if not self.raw_root_path.exists() and collect_config_name == 'demo':
            self.raw_root_path = DATA_ROOT_PATH / 'isaac45' / task_name / 'hdf5'
        if not self.raw_root_path.exists():
            raise FileNotFoundError(f"Raw demonstration directory not found: {self.raw_root_path}")
        self.raw_hdf5_path = sorted(
            self.raw_root_path.rglob('*.hdf5'), key=lambda x: int(x.stem))

        self._data = None
        self.visual_cameras = []
        self.tactile_cameras = []
        self.save_root_path = None
        self.selected_raw_hdf5_paths = []
        self.camera_key_map = {
            'visual/head': {
                'raw_key': 'observation/head/rgb',
                'transform': self.visual_transform, 'save_key': 'cam_head'
            },
            'visual/wrist': {
                'raw_key': 'observation/wrist/rgb',
                'transform': self.visual_transform, 'save_key': 'cam_wrist'
            },
            'tactile/left': {
                # 'raw_key': 'tactile/left_tactile/rgb_marker',
                'transform': self.tactile_transform, 'save_key': 'tac_left'
            },
            'tactile/right': {
                # 'raw_key': 'tactile/right_tactile/rgb_marker',
                'transform': self.tactile_transform, 'save_key': 'tac_right'
            },
        }
    
    def load_data(
        self, visual_cameras, tactile_cameras,
        downsample_factor=1, episode_num=50, random_select=False
    ):
        assert episode_num <= len(self.raw_hdf5_path), \
            f"Requested {episode_num} episodes, but only found {len(self.raw_hdf5_path)}"

        if random_select:
            self.selected_raw_hdf5_paths = np.random.choice(self.raw_hdf5_path, episode_num, replace=False)
        else:
            self.selected_raw_hdf5_paths = self.raw_hdf5_path[:episode_num]
        
        self.visual_cameras = visual_cameras
        self.tactile_cameras = tactile_cameras
        
        data_paths = [('embodiment/joint', self.joint_transform)]
        for cam in self.visual_cameras:
            cam_cfg = self.camera_key_map[f'visual/{cam}']
            data_paths.append((
                cam_cfg['raw_key'], cam_cfg.get('transform')
            ))
        
        # test and add tactile data (for compatibility with old datasets)
        with h5py.File(str(self.selected_raw_hdf5_paths[0]), 'r') as f:
            try:
                f['tactile/left_tactile/rgb_marker']
                for cam in self.tactile_cameras:
                    cam_cfg = self.camera_key_map[f'tactile/{cam}']
                    cam_cfg['raw_key'] = f'tactile/{cam}_tactile/rgb_marker'
                    data_paths.append((
                        cam_cfg['raw_key'], cam_cfg.get('transform')
                    ))
            except:
                for cam in self.tactile_cameras:
                    cam_cfg = self.camera_key_map[f'tactile/{cam}']
                    cam_cfg['raw_key'] = f'tactile/{cam}_gsmini/rgb_marker'
                    data_paths.append((
                        cam_cfg['raw_key'], cam_cfg.get('transform')
                    ))
        
        self._data = HDF5Handler().batch_gather_hdf5(
            hdf5_paths=self.selected_raw_hdf5_paths,
            data_paths=data_paths,
            downsample_factor=downsample_factor,
        )
        return self._data

    def visual_transform(self, images:np.ndarray) -> np.ndarray:
        '''
            input : ndarray of (N, H, W, 3) uint8 images
            output: ndarray of (N, H, W, 3) images after transformation
        '''
        resized = np.stack([
            cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
            for img in images
        ], axis=0, dtype=np.uint8)
        return resized

    def tactile_transform(self, images:np.ndarray) -> np.ndarray:
        '''
            input : ndarray of (N, H, W, 3) uint8 tactile images
            output: ndarray of (N, H, W, 3) tactile images after transformation
        '''
        resized = np.stack([
            cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
            for img in images
        ], axis=0, dtype=np.uint8)
        return resized

    def joint_transform(self, joints:np.ndarray) -> np.ndarray:
        '''
            input : ndarray of (N, D) joint states or actions
            output: ndarray of (N, D) joint states or actions after transformation
        '''
        return joints
    
    def export_to_hdf5(self, save_root_path):
        assert self._data is not None, "Data not loaded. Please call load_data() before export_to_hdf5()."

        self.save_root_path = Path(save_root_path)
        self.save_root_path.mkdir(parents=True, exist_ok=True)
        
        start_idx = 0
        for i in tqdm(range(len(self.selected_raw_hdf5_paths)), desc='Writing episodes'):
            end_idx = self._data['episode_ends'][i]
            hdf5_path = str(self.save_root_path / f"episode_{i}.hdf5")
            with h5py.File(hdf5_path, "w") as f:
                f.create_dataset("action", data=np.array(
                    self._data['embodiment/joint_action'][start_idx:end_idx], dtype=np.float32))
                obs = f.create_group("observations")
                obs.create_dataset("qpos", data=np.array(
                    self._data['embodiment/joint_state'][start_idx:end_idx], dtype=np.float32))
                image = obs.create_group("images")
                for cam in self.visual_cameras:
                    cam_cfg = self.camera_key_map[f'visual/{cam}']
                    image.create_dataset(cam_cfg['save_key'], data=np.array(
                        self._data[cam_cfg['raw_key']][start_idx:end_idx], dtype=np.uint8
                    ), dtype=np.uint8)
                for cam in self.tactile_cameras:
                    cam_cfg = self.camera_key_map[f'tactile/{cam}']
                    image.create_dataset(cam_cfg['save_key'], data=np.array(
                        self._data[cam_cfg['raw_key']][start_idx:end_idx], dtype=np.uint8
                    ), dtype=np.uint8)
            start_idx = end_idx
    
    def get_metadata(self):
        assert self._data is not None, "Data not loaded. Please call load_data() before get_metadata()."
        max_episode_len = np.diff([0, *self._data['episode_ends']]).max()
        metadata = {
            'num_episodes': len(self.selected_raw_hdf5_paths),
            'state_dim': int(self._data['embodiment/joint_state'].shape[-1]),
            'episode_len': int(max_episode_len),
            'dataset_dir': str(self.save_root_path),
            'camera_names': [f'cam_{cam}' for cam in self.visual_cameras] \
                + [f'tac_{cam}' for cam in self.tactile_cameras],
            'episode_map': {i: str(path) for i, path in enumerate(self.selected_raw_hdf5_paths)},
        }
        if self.save_root_path is not None:
            with open(self.save_root_path / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=4)
        return metadata
    
    def run(
        self, save_root_path, visual_cameras=['head'], tactile_cameras=['left', 'right'],
        downsample_factor=1, episode_num=50, random_select=False
    ):
        self.load_data(
            visual_cameras=visual_cameras,
            tactile_cameras=tactile_cameras,
            downsample_factor=downsample_factor,
            episode_num=episode_num,
            random_select=random_select,
        )
        self.export_to_hdf5(save_root_path)
        metadata = self.get_metadata()
        return metadata

def main(task_name, task_config, expert_data_num):
    output_path = f"./data/sim-{task_name}/{task_config}-{expert_data_num}"
    
    task_settings_path = Path(__file__).parent / './task_settings.json'
    if task_settings_path.exists():
        with open(task_settings_path, 'r') as f:
            task_settings = json.load(f)
    else:
        task_settings = {}
    
    camera_type = task_settings.get(task_name, {}).get('camera_type', 'head')
    if camera_type == 'all':
        visual_cameras = ['head', 'wrist']
    else:
        visual_cameras = [camera_type]
    tactile_cameras = ['left', 'right']
    downsample_factor = task_settings.get(task_name, {}).get('downsample_factor', 1)
 
    processor = BaseDataPreprocessor(task_name, task_config)
    metadata = processor.run(
        save_root_path=output_path,
        visual_cameras=visual_cameras,
        tactile_cameras=tactile_cameras,
        downsample_factor=downsample_factor,
        episode_num=expert_data_num,
        random_select=False,
    )

    SIM_TASK_CONFIGS_PATH = "./SIM_TASK_CONFIGS.json"
    try:
        with open(SIM_TASK_CONFIGS_PATH, "r") as f:
            SIM_TASK_CONFIGS = json.load(f)
    except Exception:
        SIM_TASK_CONFIGS = {}

    SIM_TASK_CONFIGS[f"sim-{task_name}-{task_config}-{expert_data_num}"] = {
        "dataset_dir": metadata['dataset_dir'],
        "num_episodes": metadata['num_episodes'],
        "episode_len": metadata['episode_len'],
        "camera_names": metadata['camera_names'],
    }

    with open(SIM_TASK_CONFIGS_PATH, "w") as f:
        json.dump(SIM_TASK_CONFIGS, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process TacArena episodes for ACT training.")
    parser.add_argument(
        "task_name",
        type=str,
        help="The name of the task (e.g., insert_hole)",
    )
    parser.add_argument("task_config", type=str, help="Task config (e.g., demo)")
    parser.add_argument("expert_data_num", type=int, help="Number of episodes to process")

    args = parser.parse_args()

    task_name = args.task_name
    task_config = args.task_config
    expert_data_num = args.expert_data_num

    main(task_name, task_config, expert_data_num)
