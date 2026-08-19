from __future__ import annotations

import platform
import subprocess
import time
from contextlib import contextmanager
from typing import Any, Iterable

import torch


def cuda_synchronize(device: torch.device | str | None = None) -> None:
    if not torch.cuda.is_available():
        return
    resolved = torch.device(device) if device is not None else torch.device("cuda")
    if resolved.type == "cuda":
        torch.cuda.synchronize(resolved)


def reset_peak_vram(device: torch.device | str | None = None) -> None:
    if not torch.cuda.is_available():
        return
    resolved = torch.device(device) if device is not None else torch.device("cuda")
    if resolved.type == "cuda":
        cuda_synchronize(resolved)
        torch.cuda.reset_peak_memory_stats(resolved)


def peak_vram(device: torch.device | str | None = None) -> dict[str, int | None]:
    if not torch.cuda.is_available():
        return {"peak_vram_allocated_bytes": None, "peak_vram_reserved_bytes": None}
    resolved = torch.device(device) if device is not None else torch.device("cuda")
    if resolved.type != "cuda":
        return {"peak_vram_allocated_bytes": None, "peak_vram_reserved_bytes": None}
    cuda_synchronize(resolved)
    return {
        "peak_vram_allocated_bytes": int(torch.cuda.max_memory_allocated(resolved)),
        "peak_vram_reserved_bytes": int(torch.cuda.max_memory_reserved(resolved)),
    }


@contextmanager
def timed(device: torch.device | str | None = None):
    cuda_synchronize(device)
    started = time.perf_counter()
    elapsed = [0.0]
    try:
        yield elapsed
    finally:
        cuda_synchronize(device)
        elapsed[0] = float(time.perf_counter() - started)


def module_statistics(modules: torch.nn.Module | Iterable[torch.nn.Module]) -> dict[str, int]:
    if isinstance(modules, torch.nn.Module):
        modules = [modules]
    unique_parameters: dict[int, torch.Tensor] = {}
    unique_state: dict[int, torch.Tensor] = {}
    for module in modules:
        for parameter in module.parameters():
            unique_parameters[id(parameter)] = parameter
        for tensor in module.state_dict().values():
            if isinstance(tensor, torch.Tensor):
                unique_state[id(tensor)] = tensor
    parameters = list(unique_parameters.values())
    return {
        "total_params": int(sum(p.numel() for p in parameters)),
        "trainable_params": int(sum(p.numel() for p in parameters if p.requires_grad)),
        "persistent_state_bytes": int(sum(v.numel() * v.element_size() for v in unique_state.values())),
    }


def runtime_metadata(device: torch.device | str | None = None) -> dict[str, Any]:
    resolved = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    gpu_name = None
    gpu_total_memory_bytes = None
    driver_version = None
    if resolved.type == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(resolved)
        gpu_total_memory_bytes = int(torch.cuda.get_device_properties(resolved).total_memory)
        try:
            driver_version = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True,
                timeout=5,
            ).splitlines()[0].strip()
        except (OSError, subprocess.SubprocessError, IndexError):
            pass
    total_ram_bytes = None
    try:
        import psutil
        total_ram_bytes = int(psutil.virtual_memory().total)
    except ImportError:
        pass
    return {
        "schema_version": 1,
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor() or None,
        "total_ram_bytes": total_ram_bytes,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(resolved),
        "gpu": gpu_name,
        "gpu_total_memory_bytes": gpu_total_memory_bytes,
        "gpu_driver": driver_version,
    }
