"""Probe installed runtimes, never infer training support from the GPU name."""
import os
import platform
import subprocess
from dataclasses import asdict, dataclass


@dataclass
class Device:
    id: str
    name: str
    backend: str
    training: bool
    memory_bytes: int = 0
    shared_memory: bool = False
    error: str = ""


def devices(verify=True):
    result = [Device("cpu", platform.processor() or platform.machine(), "cpu", True, shared_memory=True)]
    try:
        import torch
    except ImportError:
        result[0].training = False
        result[0].error = "Install PyTorch in COMPUTE_PYTHON to train"
        torch = None
    if torch:
        candidates = []
        if torch.cuda.is_available():
            backend = "rocm" if torch.version.hip else "cuda"
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                candidates.append(Device(f"cuda:{i}", p.name, backend, True, p.total_memory))
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            for i in range(torch.xpu.device_count()):
                p = torch.xpu.get_device_properties(i)
                candidates.append(Device(f"xpu:{i}", p.name, "xpu", True, getattr(p, "total_memory", 0)))
        if torch.backends.mps.is_available():
            candidates.append(Device("mps", "Apple Metal GPU", "mps", True, shared_memory=True))
        if platform.system() == "Windows":
            try:
                import torch_directml
                for i in range(torch_directml.device_count()):
                    candidates.append(Device(str(torch_directml.device(i)), torch_directml.device_name(i), "directml", True, shared_memory=True))
            except ImportError:
                pass
            except Exception as error:
                result.append(Device("directml", "DirectML", "directml", False, error=str(error)))
        for item in candidates:
            if verify:
                try:
                    x = torch.ones((4, 4), device=item.id, requires_grad=True)
                    (x @ x).square().mean().backward()
                    if not bool(torch.isfinite(x.grad).all().cpu()):
                        raise RuntimeError("non-finite backward result")
                except Exception as error:
                    item.training = False
                    item.error = str(error)
            result.append(item)
    try:
        import openvino as ov
        core = ov.Core()
        for name in core.available_devices:
            if name.startswith(("GPU", "NPU")):
                result.append(Device(f"openvino:{name}", core.get_property(name, "FULL_DEVICE_NAME"),
                                     "openvino", False, shared_memory=True,
                                     error="Inference only; no PyTorch autograd"))
    except ImportError:
        pass
    except Exception as error:
        result.append(Device("openvino", "OpenVINO", "openvino", False, error=str(error)))
    return [asdict(item) for item in result]


def select(device="auto", training=True):
    inventory = devices()
    if device == "auto":
        for backend in ("cuda", "rocm", "xpu", "mps", "directml", "cpu"):
            for item in inventory:
                if item["backend"] == backend and item["training"]:
                    return item["id"]
    else:
        for item in inventory:
            if item["id"] == device and (item["training"] or (not training and item["backend"] == "openvino")):
                return device
    raise RuntimeError(f"Training device {device!r} unavailable: {inventory}")


def initialize(device):
    if device.startswith("privateuseone"):
        import torch_directml
        index = int(device.split(":")[1]) if ":" in device else 0
        return torch_directml.device(index)
    return device
