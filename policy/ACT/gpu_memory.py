import json
import math
import os
from pathlib import Path

import torch


def reserve_memory_floor(device, report_dir):
    fraction = float(os.environ.get("ACT_GPU_MEMORY_FLOOR", "0"))
    if fraction == 0:
        return None
    if not 0 < fraction < 1:
        raise ValueError("ACT_GPU_MEMORY_FLOOR must be between zero and one")
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("A GPU memory floor requires a CUDA device")
    torch.cuda.synchronize(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    initial_reserved = torch.cuda.memory_reserved(device)
    target_bytes = math.ceil(total_bytes * fraction)
    reserve_bytes = max(0, target_bytes - initial_reserved)
    if reserve_bytes and reserve_bytes + 1024**3 > free_bytes:
        raise RuntimeError("Not enough free GPU memory to safely reserve the requested floor")
    buffer = torch.empty(reserve_bytes, dtype=torch.uint8, device=device) if reserve_bytes else None
    if buffer is not None:
        buffer.zero_()
    torch.cuda.synchronize(device)
    reserved_bytes = torch.cuda.memory_reserved(device)
    report = {
        "minimum_fraction": fraction, "total_bytes": total_bytes,
        "initial_reserved_bytes": initial_reserved, "buffer_bytes": reserve_bytes,
        "reserved_bytes": reserved_bytes, "reserved_fraction": reserved_bytes / total_bytes,
        "device": str(device), "checked_after_first_optimizer_step": True,
    }
    (Path(report_dir) / "gpu_memory.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"GPU memory floor: {report}", flush=True)
    return buffer
