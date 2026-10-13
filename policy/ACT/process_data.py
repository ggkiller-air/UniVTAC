import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from _base_data_preprocessor import *
from protocol import ACT_PROTOCOL, ACT_STATE_DIM, task_observation_spec


class ACTDataPreprocessor(BaseDataPreprocessor):
    def joint_transform(self, joints):
        if joints.ndim != 2 or joints.shape[-1] not in (ACT_STATE_DIM, ACT_STATE_DIM + 1):
            raise ValueError(f"Expected eight or nine recorded Franka joints, got {joints.shape}")
        return joints[:, :ACT_STATE_DIM]

def main(task_name, task_config, expert_data_num):
    output_root = Path(os.environ.get("UNIVTAC_PROCESSED_ROOT", "./data"))
    output_path = str(output_root / f"sim-{task_name}" / f"{task_config}-{expert_data_num}")
    
    spec = task_observation_spec(task_name)
    visual_cameras = spec['visual_cameras']
    tactile_cameras = ['left', 'right']
    downsample_factor = spec['downsample_factor']
 
    processor = ACTDataPreprocessor(task_name, task_config)
    for camera, save_key in zip(visual_cameras, spec['camera_names']):
        processor.camera_key_map[f'visual/{camera}']['save_key'] = save_key
    metadata = processor.run(
        save_root_path=output_path,
        visual_cameras=visual_cameras,
        tactile_cameras=tactile_cameras,
        downsample_factor=downsample_factor,
        episode_num=expert_data_num,
        random_select=False,
    )
    metadata['camera_names'] = spec['camera_names'] + spec['tactile_names']
    metadata['act_protocol'] = ACT_PROTOCOL
    with open(Path(output_path) / 'metadata.json', 'w') as metadata_file:
        json.dump(metadata, metadata_file, indent=4)

    SIM_TASK_CONFIGS_PATH = os.environ.get("UNIVTAC_TASK_CONFIGS", "./SIM_TASK_CONFIGS.json")
    Path(SIM_TASK_CONFIGS_PATH).parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(SIM_TASK_CONFIGS_PATH, "r") as f:
            SIM_TASK_CONFIGS = json.load(f)
    except Exception:
        SIM_TASK_CONFIGS = {}

    SIM_TASK_CONFIGS[f"sim-{task_name}-{task_config}-{expert_data_num}"] = {
        "dataset_dir": metadata['dataset_dir'],
        "num_episodes": metadata['num_episodes'],
        "state_dim": metadata['state_dim'],
        "episode_len": metadata['episode_len'],
        "camera_names": metadata['camera_names'],
        "act_protocol": ACT_PROTOCOL,
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
