"""Multi-GPU training helpers (DistributedDataParallel, one process per GPU).

The server starts one worker process (train_worker.py = rank 0). For more than one selected GPU the worker spawns
the other ranks itself and passes RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT through the environment.
run_training() calls build_context(options) to find out which device this process uses and to join the process group.

options (TrainConfig.options):
    gpus: list[int] | None      CUDA device indices to train on (None -> the training GPUs of device_roles()); ignored
                                without CUDA. The server always sets it from device_roles(), it is not user-selectable.
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
    gpu_indices: tuple = ()  # CUDA device indices of all ranks (for the power / utilisation sampling on rank 0)

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


def device_roles(device_count: int) -> dict:
    """Automatic split of the CUDA devices between training and inference.

    - 2 or more GPUs: the last GPU is reserved for inference, all others train (2 GPUs: training on GPU 0, inference on
      GPU 1; more GPUs: DistributedDataParallel over all but the last one).
    - 1 GPU (or MPS / CPU): training and inference share the device.
    Returns {train_gpus, inference_gpu, shared}; the GPU values are empty / None without CUDA.
    """
    count = max(0, int(device_count))
    if count <= 0:
        return {"train_gpus": [], "inference_gpu": None, "shared": True}
    if count == 1:
        return {"train_gpus": [0], "inference_gpu": 0, "shared": True}
    return {"train_gpus": list(range(count - 1)), "inference_gpu": count - 1, "shared": False}


def inference_device() -> torch.device:
    """Device of the inference runtime in the server (the last GPU if there are several, see device_roles())."""
    if torch.cuda.is_available():
        index = device_roles(torch.cuda.device_count())["inference_gpu"]
        return torch.device(f"cuda:{0 if index is None else index}")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
        count = torch.cuda.device_count()
        gpus = options.get("gpus")
        if gpus is None:
            gpus = device_roles(count)["train_gpus"]
        return [("cuda", index) for index in normalize_gpus(gpus, count)]
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
        return DistContext(device=device, gpu_indices=tuple(i for k, i in plan[:1] if k == "cuda"))

    spec = plan[rank]
    device = _device_from_spec(spec)
    backend = "nccl" if device.type == "cuda" else "gloo"
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(
        backend=backend, init_method="env://", rank=rank, world_size=world_size,
        timeout=datetime.timedelta(minutes=int(timeout_minutes)),
    )
    return DistContext(device=device, rank=rank, world_size=world_size, backend=backend, initialized=True,
                       gpu_indices=tuple(i for k, i in plan[:world_size] if k == "cuda"))
