import hashlib
import json
from pathlib import Path

import yaml


ACT_PROTOCOL = "author_act_v1"
ACT_STATE_DIM = 8
AUTHOR_TACTILE_SHA256 = "28097c91a1a1d051f65266fc7fdd84420dc184da9dcfc2006aa40af144cf15bf"
ACT_ROOT = Path(__file__).resolve().parent


def task_observation_spec(task_name):
    with (ACT_ROOT.parent / "task_settings.json").open() as settings_file:
        task_settings = json.load(settings_file)
    if task_name not in task_settings:
        raise ValueError(f"No ACT camera protocol is defined for task {task_name!r}")
    settings = task_settings[task_name]
    camera_type = settings["camera_type"]
    if camera_type == "all":
        visual_cameras = ["head", "wrist"]
        camera_names = ["cam_high", "cam_wrist"]
    elif camera_type in ("head", "wrist"):
        visual_cameras = [camera_type]
        camera_names = ["cam_high"]
    else:
        raise ValueError(f"Unsupported ACT camera type: {camera_type!r}")
    return {
        "visual_cameras": visual_cameras,
        "camera_names": camera_names,
        "tactile_names": ["tac_left", "tac_right"],
        "downsample_factor": settings.get("downsample_factor", settings.get("downsample", 1)),
    }


def author_training_config(task_name, tactile_checkpoint, variant="hypernet", config_path=None):
    if variant not in ("baseline", "hypernet"):
        raise ValueError(f"Unknown ACT variant: {variant}")
    spec = task_observation_spec(task_name)
    if config_path is None:
        config_path = ACT_ROOT / ("train_config_all.yml" if len(spec["camera_names"]) == 2 else "train_config.yml")
    with Path(config_path).open() as config_file:
        config = yaml.safe_load(config_file)
    if config["state_dim"] != ACT_STATE_DIM:
        raise ValueError("Author ACT protocol requires eight-dimensional states and actions")
    config.update(
        camera_names=spec["camera_names"], tactile_names=spec["tactile_names"],
        act_protocol=ACT_PROTOCOL, require_tactile_checkpoint=True,
        use_hypernet=variant == "hypernet",
    )
    if variant == "hypernet":
        with (ACT_ROOT / "hypernet.yml").open() as hypernet_file:
            config.update(yaml.safe_load(hypernet_file))
    checkpoint = Path(tactile_checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Author tactile encoder checkpoint not found: {checkpoint}")
    digest = hashlib.sha256()
    with checkpoint.open("rb") as checkpoint_file:
        for block in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != AUTHOR_TACTILE_SHA256:
        raise ValueError("Tactile checkpoint SHA256 does not match the published author encoder")
    config["tactile_ckpt"] = str(checkpoint)
    config["tactile_checkpoint_sha256"] = digest.hexdigest()
    return config


def validate_processed_task(task, task_name, episode_count):
    spec = task_observation_spec(task_name)
    if task.get("act_protocol") != ACT_PROTOCOL or task.get("state_dim") != ACT_STATE_DIM:
        raise ValueError("Processed data does not use the author ACT eight-dimensional protocol; preprocess again")
    if task.get("camera_names") != spec["camera_names"] + spec["tactile_names"]:
        raise ValueError("Processed camera streams do not match the task's author ACT camera protocol")
    if task.get("num_episodes") != episode_count:
        raise ValueError("Processed episode count does not match the requested training data")
    for episode_id in range(episode_count):
        path = Path(task["dataset_dir"]) / f"episode_{episode_id}.hdf5"
        if not path.is_file():
            raise FileNotFoundError(path)
