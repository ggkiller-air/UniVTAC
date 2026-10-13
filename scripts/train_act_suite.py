import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "policy" / "ACT"))

from protocol import author_training_config, validate_processed_task


TASKS = (
    "lift_can", "insert_hole", "insert_HDMI", "lift_bottle",
    "pull_out_key", "put_bottle_in_shelf", "insert_tube", "grasp_classify",
)
DATASET = "byml2024/UniVTAC"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_manifest(task, episodes):
    from modelscope.hub.api import HubApi

    api = HubApi()
    prefix = f"/isaac45/{task}"
    entries = api.get_dataset_files(DATASET, root_path=f"{prefix}/hdf5", recursive=False, page_size=200)
    episodes_available = sorted(
        [entry for entry in entries if entry["Path"].endswith(".hdf5")],
        key=lambda entry: int(Path(entry["Path"]).stem),
    )
    if len(episodes_available) < episodes:
        raise ValueError(f"{task}: found only {len(episodes_available)} published episodes")
    metadata = api.get_dataset_files(DATASET, root_path=prefix, recursive=False, page_size=200)
    selected = episodes_available[:episodes] + [entry for entry in metadata if entry["Name"] == "metadata.json"]
    if len(selected) != episodes + 1:
        raise ValueError(f"{task}: missing metadata")
    for entry in selected:
        if not entry.get("Revision") or not entry.get("Sha256"):
            raise ValueError(f"Missing immutable revision or SHA256: {entry['Path']}")
    return {"available_episodes": len(episodes_available), "files": selected}


def submit(arguments):
    return subprocess.check_output(["sbatch", "--parsable", *arguments], text=True).strip().split(";")[0]


def launch(args):
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(task_manifest, task, args.episodes): task for task in TASKS}
        manifests = {futures[future]: future.result() for future in as_completed(futures)}
    for task in TASKS:
        author_training_config(task, os.environ["UNIVTAC_TACTILE_CHECKPOINT"])
    suite = Path(os.environ["UNIVTAC_RUN_ROOT"]) / "ACT-hypernet-all" / datetime.now().strftime("%Y%m%d-%H%M%S")
    suite.mkdir(parents=True, exist_ok=False)
    (suite / "logs").mkdir()
    config = {
        "dataset": DATASET, "isaac_version": "4.5", "episodes": args.episodes,
        "seed": args.seed, "tasks": list(TASKS), "download_workers": args.download_workers,
        "prepare_parallel": args.prepare_parallel, "train_parallel": args.train_parallel,
        "gpu_memory_floor_fraction": 0.10,
        "data_root": os.environ["UNIVTAC_DATA_ROOT"],
        "processed_root": os.environ["UNIVTAC_PROCESSED_ROOT"],
        "manifests": manifests, "jobs": {}, "created_at": datetime.now().isoformat(),
    }
    write_json(suite / "suite.json", config)
    total = sum(entry["Size"] for manifest in manifests.values() for entry in manifest["files"])
    print(f"Suite: {suite}\nSelected data: {total / 1024**3:.2f} GiB", flush=True)
    common = ["--account", args.account, "--nodes=1", "--ntasks=1", "--time=12:00:00", "--open-mode=append"]
    worker = [str(ROOT / "scripts" / "act_suite_job.sh"), str(ROOT), str(suite)]
    prepare = submit([
        *common, "--partition=cpu", "--cpus-per-task=4", "--mem=64G",
        f"--array=0-{len(TASKS) - 1}%{args.prepare_parallel}", "--job-name=univtac-data",
        f"--output={suite}/logs/prepare-%A_%a.log", *worker, "prepare",
    ])
    config["jobs"]["prepare"] = prepare
    write_json(suite / "suite.json", config)
    train = submit([
        *common, "--partition=gpux", "--cpus-per-task=8", "--mem=64G", "--gres=gpu:1",
        f"--array=0-{len(TASKS) - 1}%{args.train_parallel}", "--job-name=univtac-hyper",
        f"--dependency=aftercorr:{prepare}", "--kill-on-invalid-dep=yes",
        f"--output={suite}/logs/train-%A_%a.log", *worker, "train",
    ])
    config["jobs"]["train"] = train
    write_json(suite / "suite.json", config)
    collector = submit([
        *common, "--partition=cpu", "--cpus-per-task=1", "--mem=4G", "--job-name=univtac-collect",
        f"--dependency=afterany:{train}", f"--output={suite}/logs/collect-%j.log", *worker, "collect",
    ])
    config["jobs"]["collect"] = collector
    write_json(suite / "suite.json", config)
    print(f"Preparation array: {prepare}; training array: {train}", flush=True)
    print(f"Status: python scripts/train_act_suite.py status {suite}", flush=True)


def download_file(entry, config):
    from modelscope.hub.file_download import dataset_file_download

    destination = Path(config["data_root"]) / entry["Path"]
    def valid():
        return destination.is_file() and destination.stat().st_size == entry["Size"] and sha256(destination) == entry["Sha256"]

    if valid():
        return entry["Path"]
    if destination.exists():
        destination.rename(destination.with_name(destination.name + f".invalid-{time.time_ns()}"))
    for attempt in range(4):
        try:
            dataset_file_download(
                config["dataset"], entry["Path"], revision=entry["Revision"],
                local_dir=config["data_root"],
            )
            if not valid():
                raise ValueError(f"Downloaded file failed SHA256 verification: {entry['Path']}")
            return entry["Path"]
        except Exception:
            if attempt == 3:
                raise
            if destination.exists():
                destination.rename(destination.with_name(destination.name + f".invalid-{time.time_ns()}"))
            time.sleep(5 * (attempt + 1))


def prepare(suite, config, task, environment):
    entries = config["manifests"][task]["files"]
    with ThreadPoolExecutor(max_workers=config["download_workers"]) as executor:
        futures = [executor.submit(download_file, entry, config) for entry in entries]
        for completed, future in enumerate(as_completed(futures), 1):
            print(f"Verified {completed}/{len(entries)}: {future.result()}", flush=True)
    subprocess.run(
        [sys.executable, "process_data.py", task, "demo", str(config["episodes"])],
        cwd=ROOT / "policy" / "ACT", env=environment, check=True,
    )
    registry = json.loads(Path(environment["UNIVTAC_TASK_CONFIGS"]).read_text())
    validate_processed_task(registry[f"sim-{task}-demo-{config['episodes']}"], task, config["episodes"])
    write_json(suite / "tasks" / task / "data-ready.json", {
        "task": task, "episodes": config["episodes"], "verified_files": entries,
        "completed_at": datetime.now().isoformat(),
    })


def train(suite, config, task, environment):
    if not os.environ.get("SLURM_JOB_GPUS"):
        raise RuntimeError("Training requires an allocated Slurm GPU")
    subprocess.run(["scontrol", "show", "job", os.environ["SLURM_JOB_ID"], "-dd"], check=True)
    subprocess.run([
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu", "--format=csv",
    ], check=True)
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one CUDA-visible allocated GPU")
    properties = torch.cuda.get_device_properties(0)
    gpu_uuid = str(properties.uuid)
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader",
    ], text=True).strip()
    other_processes = [line for line in processes.splitlines()
                       if line.split(",")[0].strip() == gpu_uuid
                       and line.split(",")[1].strip() != str(os.getpid())]
    if other_processes:
        raise RuntimeError(f"Allocated GPU already has compute processes: {other_processes}")
    write_json(suite / "tasks" / task / "gpu.json", {
        "physical_indices": os.environ["SLURM_JOB_GPUS"], "uuid": gpu_uuid,
        "name": properties.name, "total_memory": properties.total_memory,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "logical_device": "cuda:0",
    })
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}; training device=cuda:0", flush=True)
    run = suite / "runs" / task
    environment["ACT_RUN_DIR"] = str(run)
    subprocess.run([
        "bash", str(ROOT / "scripts" / "train_act_hypernet.sh"),
        task, str(config["episodes"]), str(config["seed"]),
    ], cwd=ROOT, env=environment, check=True)
    export = suite / "checkpoints" / task
    export.mkdir(parents=True, exist_ok=False)
    for name in ("policy_best.ckpt", "policy_last.ckpt", "dataset_stats.pkl", "resolved_config.yml"):
        if not (run / name).is_file():
            raise FileNotFoundError(run / name)
    expected = [run / f"policy_step_{step}_seed_{config['seed']}.ckpt" for step in range(1000, 4001, 1000)]
    if not all(checkpoint.is_file() for checkpoint in expected):
        raise RuntimeError(f"Missing periodic checkpoints in {run}")
    for artifact in run.iterdir():
        if artifact.is_file():
            shutil.copy2(artifact, export / artifact.name)
    shutil.copy2(suite / "tasks" / task / "data-ready.json", export / "data-provenance.json")
    write_json(suite / "tasks" / task / "complete.json", {
        "task": task, "run_dir": str(run), "export_dir": str(export),
        "checkpoints": {checkpoint.name: sha256(checkpoint) for checkpoint in export.glob("*.ckpt")},
        "completed_at": datetime.now().isoformat(),
    })
    print(f"COMPLETED {task}: {export}", flush=True)


def execute(args):
    suite = Path(args.suite).resolve()
    config = json.loads((suite / "suite.json").read_text())
    task = config["tasks"][args.index]
    task_dir = suite / "tasks" / task
    environment = dict(os.environ)
    environment["UNIVTAC_TASK_CONFIGS"] = str(task_dir / "SIM_TASK_CONFIGS.json")
    environment["UNIVTAC_PROCESSED_ROOT"] = config["processed_root"]
    environment["ACT_GPU_MEMORY_FLOOR"] = str(config.get("gpu_memory_floor_fraction", 0.10))
    status = {"stage": args.stage, "state": "running", "job_id": os.environ.get("SLURM_JOB_ID"),
              "started_at": datetime.now().isoformat()}
    write_json(task_dir / f"{args.stage}-status.json", status)
    try:
        {"prepare": prepare, "train": train}[args.stage](suite, config, task, environment)
    except BaseException as error:
        status.update(state="failed", error=str(error), finished_at=datetime.now().isoformat())
        write_json(task_dir / f"{args.stage}-status.json", status)
        raise
    status.update(state="complete", finished_at=datetime.now().isoformat())
    write_json(task_dir / f"{args.stage}-status.json", status)


def status(args):
    suite = Path(args.suite).resolve()
    config = json.loads((suite / "suite.json").read_text())
    print(f"Suite: {suite}")
    for task in config["tasks"]:
        states = []
        for stage in ("prepare", "train"):
            path = suite / "tasks" / task / f"{stage}-status.json"
            state = json.loads(path.read_text())["state"] if path.exists() else "queued"
            states.append(f"{stage}={state}")
        raw = Path(config["data_root"]) / "isaac45" / task / "hdf5"
        print(f"{task:24s} {' '.join(states)} raw_files={len(list(raw.glob('*.hdf5')))}")
    if config["jobs"]:
        subprocess.run(["squeue", "-j", ",".join(config["jobs"].values()), "-o", "%.20i %.9P %.20j %.9T %.10M %R"])


def collect(args):
    suite = Path(args.suite).resolve()
    config = json.loads((suite / "suite.json").read_text())
    completed = {}
    for task in config["tasks"]:
        report = suite / "tasks" / task / "complete.json"
        if not report.is_file():
            raise RuntimeError(f"Cannot deliver an incomplete suite: {task} has no completed checkpoint export")
        completed[task] = json.loads(report.read_text())
    write_json(suite / "checkpoints" / "manifest.json", {
        "dataset": config["dataset"], "isaac_version": config["isaac_version"],
        "episodes_per_task": config["episodes"], "seed": config["seed"],
        "steps_per_task": 4000, "tasks": completed, "completed_at": datetime.now().isoformat(),
    })
    lines = ["# UniVTAC ACT HyperNet checkpoints", "",
             f"All eight tasks completed 4000 optimization steps with {config['episodes']} demonstrations and seed {config['seed']}.",
             "Author ACT inputs, camera choices, pretrained encoders, and training settings are retained.",
             "No simulator evaluation was run. Validation-selected weights are not rollout-selected weights.",
             "", "Each task directory contains `policy_best.ckpt`, `policy_last.ckpt`, every periodic",
             "checkpoint, `dataset_stats.pkl`, `resolved_config.yml`, source, and logs.",
             "See `manifest.json` for checkpoint SHA256 hashes and data provenance.", "", "## Tasks", ""]
    lines.extend(f"- `{task}/policy_best.ckpt`" for task in config["tasks"])
    (suite / "checkpoints" / "README.md").write_text("\n".join(lines) + "\n")
    print(f"ALL EIGHT TASKS COMPLETE: {suite / 'checkpoints'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Prepare and train all eight published UniVTAC ACT tasks.")
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("launch")
    start.add_argument("--episodes", type=int, choices=[50], default=50)
    start.add_argument("--seed", type=int, default=0)
    start.add_argument("--account", default="yangyc")
    start.add_argument("--prepare-parallel", type=int, choices=range(1, 9), default=4)
    start.add_argument("--train-parallel", type=int, choices=range(1, 9), default=4)
    start.add_argument("--download-workers", type=int, choices=range(1, 9), default=4)
    worker = commands.add_parser("execute")
    worker.add_argument("suite")
    worker.add_argument("stage", choices=["prepare", "train"])
    worker.add_argument("--index", type=int, choices=range(8), required=True)
    report = commands.add_parser("status")
    report.add_argument("suite")
    delivery = commands.add_parser("collect")
    delivery.add_argument("suite")
    args = parser.parse_args()
    {"launch": launch, "execute": execute, "status": status, "collect": collect}[args.command](args)


if __name__ == "__main__":
    main()
