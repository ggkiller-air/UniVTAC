"""Decode every local demonstration once into a versioned, reusable uint8 cache."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import cv2
import h5py
import numpy as np

from .schema import CACHE_VERSION, IMAGE_KEYS, TASK_TEXTS, fingerprint


def inspect_sources(root: Path) -> list[dict]:
    """Require all actual sensors and record source metadata and episode boundaries."""
    episodes = []
    cursor = 0
    for task_index, task in enumerate(TASK_TEXTS):
        paths = sorted((root / task / "hdf5").glob("*.hdf5"), key=lambda path: int(path.stem))
        if not paths:
            raise FileNotFoundError(f"No demonstrations for {task}")
        for path in paths:
            with h5py.File(path, "r") as store:
                count, dimension = store["embodiment/joint"].shape
                if dimension not in (8, 9) or count < 2:
                    raise ValueError(f"Invalid recorded joint shape in {path}")
                keys = {"head": "observation/head/rgb", "wrist": "observation/wrist/rgb"}
                for side in ("left", "right"):
                    candidates = [f"tactile/{side}_{kind}/rgb_marker" for kind in ("gsmini", "tactile")]
                    keys[f"tactile_{side}"] = next((key for key in candidates if key in store), "")
                for key in keys.values():
                    if not key or key not in store or len(store[key]) != count:
                        raise ValueError(f"Missing/misaligned full sensor {key!r} in {path}")
            stat = path.stat()
            episodes.append({
                "task": task, "task_index": task_index, "path": str(path.resolve()),
                "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "keys": keys,
                "start": cursor, "end": cursor + count - 1,
            })
            cursor += count - 1
    return episodes


def decode_episode(arguments: tuple[dict, str, int]) -> int:
    """Write disjoint memory-mapped ranges; preserve the author's BGR convention."""
    episode, directory, image_size = arguments
    cv2.setNumThreads(1)
    start, end = episode["start"], episode["end"]
    length = end - start
    with h5py.File(episode["path"], "r") as store:
        joints = store["embodiment/joint"][:, :8].astype(np.float32)
        if not np.isfinite(joints).all():
            raise ValueError(f"Nonfinite joint data: {episode['path']}")
        for name, values in (("state", joints[:-1]), ("action", joints[1:])):
            output = np.load(Path(directory) / f"{name}.npy", mmap_mode="r+")
            output[start:end] = values
            output.flush()
            del output
        for sensor in IMAGE_KEYS:
            output = np.load(Path(directory) / f"{sensor}.npy", mmap_mode="r+")
            encoded_frames = store[episode["keys"][sensor]][:length]
            for frame_index, encoded in enumerate(encoded_frames):
                image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"Corrupt {sensor} frame {frame_index}: {episode['path']}")
                image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
                output[start + frame_index] = np.moveaxis(image, -1, 0)
            output.flush()
            del output
    return length


def make_language_embeddings(output: Path) -> dict:
    """Encode versioned task instructions with the locally cached, frozen CLIP text model."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    model_name = "openai/clip-vit-base-patch32"
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModel.from_pretrained(model_name, local_files_only=True).eval()
    inputs = tokenizer(list(TASK_TEXTS.values()), padding=True, return_tensors="pt")
    with torch.inference_mode():
        embeddings = torch.nn.functional.normalize(model.get_text_features(**inputs), dim=-1)
    values = embeddings.cpu().numpy().astype(np.float32)
    if values.shape != (len(TASK_TEXTS), 512) or not np.isfinite(values).all():
        raise ValueError("Invalid CLIP task embeddings")
    np.save(output / "language.npy", values)
    return {"model": model_name, "task_texts": TASK_TEXTS, "sha256": hashlib.sha256(values.tobytes()).hexdigest()}


def build_cache(root: Path, output: Path, workers: int = 16, image_size: int = 256) -> dict:
    """Build under an exclusive lock and publish only after all workers finish."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output.with_suffix(".lock"), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        episodes = inspect_sources(root)
        protocol = {"version": CACHE_VERSION, "episodes": episodes, "image_size": image_size, "tasks": list(TASK_TEXTS)}
        source_id = fingerprint(protocol)
        if (output / "manifest.json").exists():
            manifest = json.loads((output / "manifest.json").read_text())
            if manifest["source_id"] != source_id:
                raise ValueError("Existing cache does not match the source dataset")
            print(f"cache_reused={output} source_id={source_id}", flush=True)
            return manifest
        staging = output.with_name(f"{output.name}.building.{os.getpid()}")
        staging.mkdir()
        started = time.monotonic()
        total = episodes[-1]["end"]
        print(f"episodes={len(episodes)} steps={total} image_GiB={total * 4 * 3 * image_size ** 2 / 2 ** 30:.2f}", flush=True)
        for key in (*IMAGE_KEYS, "state", "action"):
            shape = (total, 3, image_size, image_size) if key in IMAGE_KEYS else (total, 8)
            array = np.lib.format.open_memmap(staging / f"{key}.npy", mode="w+", shape=shape,
                                             dtype=np.uint8 if key in IMAGE_KEYS else np.float32)
            del array
        arguments = [(episode, str(staging), image_size) for episode in episodes]
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(decode_episode, argument) for argument in arguments]
            for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
                future.result()
                if completed % 20 == 0 or completed == len(episodes):
                    print(f"decoded={completed}/{len(episodes)} seconds={time.monotonic() - started:.1f}", flush=True)
        language = make_language_embeddings(staging)
        manifest = {**protocol, "source_id": source_id, "language": language, "total_steps": total,
                    "action_alignment": "state=joint[:-1,:8],action=joint[1:,:8],images=rgb[:-1]",
                    "image_convention": "OpenCV BGR, uint8 CHW, bilinear resize; GPU maps once to [-1,1]"}
        manifest["cache_id"] = fingerprint(manifest)
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if output.exists():
            raise FileExistsError(output)
        staging.rename(output)
        print(f"cache_complete={output} cache_id={manifest['cache_id']} seconds={time.monotonic() - started:.1f}", flush=True)
        return manifest


def main() -> None:
    """Build the persistent offline cache without requesting a GPU."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    arguments = parser.parse_args()
    build_cache(arguments.root, arguments.output, arguments.workers)


if __name__ == "__main__":
    main()
