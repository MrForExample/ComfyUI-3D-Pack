import gc
import os
import re
import time
from collections import defaultdict
from contextlib import contextmanager

import psutil
import torch
from packaging import version

from .config import config_to_primitive
from .core import debug, find, info, warn
from .typing import *


def parse_version(ver: str):
    return version.parse(ver)


def get_rank():
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


def get_device():
    return torch.device(f"cuda:{get_rank()}")


def load_module_weights(
    path, module_name=None, ignore_modules=None, mapping=None, map_location=None
) -> Tuple[dict, int, int]:
    if module_name is not None and ignore_modules is not None:
        raise ValueError("module_name and ignore_modules cannot be both set")
    if map_location is None:
        map_location = get_device()

    ckpt = torch.load(path, map_location=map_location)
    state_dict = ckpt["state_dict"]

    if mapping is not None:
        state_dict_to_load = {}
        for k, v in state_dict.items():
            if any([k.startswith(m["to"]) for m in mapping]):
                pass
            else:
                state_dict_to_load[k] = v
        for k, v in state_dict.items():
            for m in mapping:
                if k.startswith(m["from"]):
                    k_dest = k.replace(m["from"], m["to"])
                    info(f"Mapping {k} => {k_dest}")
                    state_dict_to_load[k_dest] = v.clone()
        state_dict = state_dict_to_load

    state_dict_to_load = state_dict

    if ignore_modules is not None:
        state_dict_to_load = {}
        for k, v in state_dict.items():
            ignore = any(
                [k.startswith(ignore_module + ".") for ignore_module in ignore_modules]
            )
            if ignore:
                continue
            state_dict_to_load[k] = v

    if module_name is not None:
        state_dict_to_load = {}
        for k, v in state_dict.items():
            m = re.match(rf"^{module_name}\.(.*)$", k)
            if m is None:
                continue
            state_dict_to_load[m.group(1)] = v

    return state_dict_to_load, ckpt["epoch"], ckpt["global_step"]


def C(value: Any, epoch: int, global_step: int) -> float:
    if isinstance(value, int) or isinstance(value, float):
        pass
    else:
        value = config_to_primitive(value)
        if not isinstance(value, list):
            raise TypeError("Scalar specification only supports list, got", type(value))
        if len(value) == 3:
            value = [0] + value
        assert len(value) == 4
        start_step, start_value, end_value, end_step = value
        if isinstance(end_step, int):
            current_step = global_step
            value = start_value + (end_value - start_value) * max(
                min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0
            )
        elif isinstance(end_step, float):
            current_step = epoch
            value = start_value + (end_value - start_value) * max(
                min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0
            )
    return value


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    try:
        import tinycudann as tcnn

        tcnn.free_temporary_memory()
    except:
        pass


def finish_with_cleanup(func: Callable):
    def wrapper(*args, **kwargs):
        out = func(*args, **kwargs)
        cleanup()
        return out

    return wrapper


def _distributed_available():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def barrier():
    if not _distributed_available():
        return
    else:
        torch.distributed.barrier()


def broadcast(tensor, src=0):
    if not _distributed_available():
        return tensor
    else:
        torch.distributed.broadcast(tensor, src=src)
        return tensor


def enable_gradient(model, enabled: bool = True) -> None:
    for param in model.parameters():
        param.requires_grad_(enabled)


class TimeRecorder:
    _instance = None

    def __init__(self):
        self.items = {}
        self.accumulations = defaultdict(list)
        self.time_scale = 1000.0  # ms
        self.time_unit = "ms"
        self.enabled = False

    def __new__(cls):
        # singleton
        if cls._instance is None:
            cls._instance = super(TimeRecorder, cls).__new__(cls)
        return cls._instance

    def enable(self, enabled: bool) -> None:
        self.enabled = enabled

    def start(self, name: str) -> None:
        if not self.enabled:
            return
        torch.cuda.synchronize()
        self.items[name] = time.time()

    def end(self, name: str, accumulate: bool = False) -> float:
        if not self.enabled or name not in self.items:
            return
        torch.cuda.synchronize()
        start_time = self.items.pop(name)
        delta = time.time() - start_time
        if accumulate:
            self.accumulations[name].append(delta)
        t = delta * self.time_scale
        info(f"{name}: {t:.2f}{self.time_unit}")

    def get_accumulation(self, name: str, average: bool = False) -> float:
        if not self.enabled or name not in self.accumulations:
            return
        acc = self.accumulations.pop(name)
        total = sum(acc)
        if average:
            t = total / len(acc) * self.time_scale
        else:
            t = total * self.time_scale
        info(f"{name} for {len(acc)} times: {t:.2f}{self.time_unit}")


### global time recorder
time_recorder = TimeRecorder()


@contextmanager
def time_recorder_enabled():
    enabled = time_recorder.enabled
    time_recorder.enable(enabled=True)
    try:
        yield
    finally:
        time_recorder.enable(enabled=enabled)


def show_vram_usage(name):
    available, total = torch.cuda.mem_get_info()
    used = total - available
    print(
        f"{name}: {used / 1024**2:.1f}MB, {psutil.Process(os.getpid()).memory_info().rss / 1024**2:.1f}MB"
    )
