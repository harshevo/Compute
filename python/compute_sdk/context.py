import json
import os
import time
import uuid
from pathlib import Path

from .storage import atomic_json, digest, read_json


def encode_state(value, tensors, path="state"):
    import torch
    if isinstance(value, torch.Tensor):
        key = str(len(tensors))
        tensors[key] = value.detach().cpu().contiguous().clone()
        return {"type": "tensor", "key": key}
    if isinstance(value, dict):
        return {"type": "dict", "items": [[encode_state(k, tensors), encode_state(v, tensors)] for k, v in value.items()]}
    if isinstance(value, (list, tuple)):
        return {"type": "tuple" if isinstance(value, tuple) else "list", "items": [encode_state(v, tensors) for v in value]}
    if value is None or isinstance(value, (str, int, float, bool)):
        return {"type": "scalar", "value": value}
    raise TypeError(f"Unsupported checkpoint type {type(value)}")


def decode_state(value, tensors):
    kind = value["type"]
    if kind == "tensor":
        return tensors[value["key"]]
    if kind == "dict":
        return {decode_state(k, tensors): decode_state(v, tensors) for k, v in value["items"]}
    if kind in ("tuple", "list"):
        items = [decode_state(v, tensors) for v in value["items"]]
        return tuple(items) if kind == "tuple" else items
    if kind == "scalar":
        return value["value"]
    raise ValueError("invalid checkpoint metadata")


class Context:
    def __init__(self, results=None):
        self.results = Path(results or os.environ.get("COMPUTE_RESULTS", "./results")).resolve()
        self.results.mkdir(parents=True, exist_ok=True)
        self.device = os.environ.get("COMPUTE_DEVICE")
        if self.device is None:
            from .hardware import select
            self.device = select()
        from .hardware import initialize
        self.device = initialize(self.device)

    def log(self, **metrics):
        record = {"time": time.time(), **metrics}
        print(json.dumps(record, allow_nan=False), flush=True)
        with (self.results / "metrics.jsonl").open("a") as file:
            file.write(json.dumps(record, allow_nan=False) + "\n")

    def checkpoint(self, model, optimizer, step):
        import torch
        from safetensors.torch import save_file
        generation = f"checkpoint-{int(step)}-{uuid.uuid4().hex}"
        directory = self.results / generation
        directory.mkdir()
        tensors = {}
        metadata = encode_state({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                 "step": int(step), "rng": torch.get_rng_state()}, tensors)
        save_file(tensors, str(directory / "state.safetensors"))
        atomic_json(directory / "metadata.json", metadata)
        atomic_json(self.results / "latest.json", {"generation": generation,
                     "weights_hash": digest((directory / "state.safetensors").read_bytes()),
                     "metadata_hash": digest((directory / "metadata.json").read_bytes())})
        return directory

    def restore(self, model, optimizer):
        import torch
        from safetensors.torch import load_file
        if not (self.results / "latest.json").exists():
            return 0
        pointer = read_json(self.results / "latest.json")
        from .storage import safe_path
        directory = safe_path(self.results, pointer["generation"])
        weights = directory / "state.safetensors"
        metadata = directory / "metadata.json"
        if digest(weights.read_bytes()) != pointer["weights_hash"] or digest(metadata.read_bytes()) != pointer["metadata_hash"]:
            raise ValueError("checkpoint integrity check failed")
        state = decode_state(read_json(metadata), load_file(str(weights)))
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng"])
        return state["step"]


def init():
    return Context()
