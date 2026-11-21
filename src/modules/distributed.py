"""Multi-GPU training helpers (DistributedDataParallel, one process per GPU).

The server starts one worker process (train_worker.py = rank 0). For more than one selected GPU the worker spawns
the other ranks itself and passes RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT through the environment.
run_training() calls build_context(options) to find out which device this process uses and to join the process group.

options (TrainConfig.options):
    gpus: list[int] | None      CUDA device indices to train on (None -> [0]); ignored without CUDA
    ddp_cpu_processes: int      testing only: that many CPU processes with the gloo backend
"""
import datetime
import os
import socket
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist


@dataclass
class DistContext:
    device: torch.device
    rank: int = 0
    world_size: int = 1
    backend: Optional[str] = None
    initialized: bool = False

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    def all_reduce(self, tensor: torch.Tensor, op=None) -> torch.Tensor:
        if self.distributed:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM if op is None else op)
        return tensor

    def shutdown(self):
        if self.initialized and dist.is_available() and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass
        self.initialized = False


def cuda_device_list() -> list[dict]:
    """[{index, name, memory_gb}] of the visible CUDA devices (empty without CUDA)."""
    if not torch.cuda.is_available():
        return []
    devices = []
    for index in range(torch.cuda.device_count()):
        try:
            props = torch.cuda.get_device_properties(index)
            devices.append({"index": index, "name": props.name, "memory_gb": round(props.total_memory / 1024 ** 3, 1)})
        except Exception:
            devices.append({"index": index, "name": f"cuda:{index}", "memory_gb": None})
    return devices


def normalize_gpus(gpus, device_count: int) -> list[int]:
    """Sorted unique valid device indices; None / empty -> [0] (if there is a device at all)."""
    if device_count <= 0:
        return []
    picked = []
    for value in (gpus if isinstance(gpus, (list, tuple)) else []):
        try:
            index = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= index < device_count and index not in picked:
            picked.append(index)
    return sorted(picked) if picked else [0]


def plan_devices(options) -> list[tuple[str, Optional[int]]]:
    """Devices of all ranks, e.g. [('cuda', 0), ('cuda', 1)]. One entry means plain single-process training."""
    options = dict(options or {})
    cpu_processes = int(options.get("ddp_cpu_processes") or 0)
    if cpu_processes > 1:
        return [("cpu", None)] * cpu_processes
    if torch.cuda.is_available():
        return [("cuda", index) for index in normalize_gpus(options.get("gpus"), torch.cuda.device_count())]
    if torch.backends.mps.is_available():
        return [("mps", None)]
    return [("cpu", None)]


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _device_from_spec(spec) -> torch.device:
    kind, index = spec
    return torch.device(f"cuda:{index}") if kind == "cuda" else torch.device(kind)


def build_context(options, timeout_minutes: int = 30) -> DistContext:
    """Device of this process and, for world_size > 1, membership in the process group (env:// rendezvous)."""
    plan = plan_devices(options)
    world_size = int(os.environ.get("WORLD_SIZE", "1") or 1)
    rank = int(os.environ.get("RANK", "0") or 0)
    if world_size <= 1 or len(plan) < world_size or rank >= world_size:
        # Not started by a launcher (CLI, tests): plain single-process training on the first planned device.
        device = _device_from_spec(plan[0])
        if device.type == "cuda":
            torch.cuda.set_device(device)
        return DistContext(device=device)

    spec = plan[rank]
    device = _device_from_spec(spec)
    backend = "nccl" if device.type == "cuda" else "gloo"
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(
        backend=backend, init_method="env://", rank=rank, world_size=world_size,
        timeout=datetime.timedelta(minutes=int(timeout_minutes)),
    )
    return DistContext(device=device, rank=rank, world_size=world_size, backend=backend, initialized=True)
