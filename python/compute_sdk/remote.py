"""Explicit remote devices. No CUDA driver emulation or transparent torch.device."""
import base64
import tempfile
import time
import uuid
from pathlib import Path

from .storage import CHUNK, digest


class RemoteDevice:
    def __init__(self, cluster, node, device="auto", directory=None, inference=False):
        self.cluster, self.node = cluster, node
        self.id = digest(uuid.uuid4().bytes)
        files = cluster.bundle(node, directory) if directory else {}
        cluster.rpc(node, "submit", {"id": self.id, "kind": "tensor", "device": device,
                                    "files": files, "timeout": 86400, "inference": inference})
        self.job = cluster.job(node, self.id)
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            state = self.job.status()
            if state["state"] == "running":
                self.backend_device = state["device"]
                return
            if state["state"] in {"failed", "interrupted", "cancelled"}:
                raise RuntimeError(str(state))
            time.sleep(.1)
        self.close()
        raise TimeoutError("remote device startup")

    def call(self, op, **payload):
        return self.cluster.rpc(self.node, "tensor", {"id": self.id, "op": op, **payload})

    def send(self, tensors):
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tensors.safetensors"
            save_file({key: tensor.detach().cpu().contiguous().clone() for key, tensor in tensors.items()}, str(path))
            return self.cluster.upload(self.node, path)

    def download_blob(self, value, path):
        import blake3
        path = Path(path)
        offset, h = 0, blake3.blake3()
        with path.open("wb") as file:
            while True:
                response = self.cluster.rpc(self.node, "blob", {"hash": value, "offset": offset})
                data = base64.b64decode(response["data"])
                file.write(data)
                h.update(data)
                offset += len(data)
                if offset == response["size"]:
                    break
                if not data:
                    raise RuntimeError("truncated blob")
        if h.hexdigest() != value:
            path.unlink()
            raise ValueError("blob hash mismatch")
        return path

    def receive(self, value):
        from safetensors.torch import load_file
        with tempfile.TemporaryDirectory() as directory:
            path = self.download_blob(value, Path(directory) / "tensors.safetensors")
            return load_file(str(path))

    def module(self, script, index=0):
        from .split import RemoteModule
        return RemoteModule(self, script, index)

    def tensor(self, tensor):
        reply = self.call("put", hash=self.send({"tensor": tensor}))
        return RemoteTensor(self, **reply)

    def linear(self, in_features, out_features, bias=True, device=None):
        import torch
        remote = self
        class Apply(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, w, b):
                tensors = {"x": x, "w": w}
                if b is not None:
                    tensors["b"] = b
                reply = remote.call("linear_forward", hash=remote.send(tensors))
                ctx.graph = reply["graph"]
                ctx.devices = [x.device, w.device, b.device if b is not None else None]
                return remote.receive(reply["hash"])["y"].to(x.device)

            @staticmethod
            def backward(ctx, gradient):
                reply = remote.call("linear_backward", graph=ctx.graph, hash=remote.send({"gradient": gradient}))
                values = remote.receive(reply["hash"])
                return tuple(values[key].to(device) if device is not None else None
                             for key, device in zip(("x", "w", "b"), ctx.devices))
        class Linear(torch.nn.Linear):
            def forward(self, x):
                return Apply.apply(x, self.weight, self.bias)
        return Linear(in_features, out_features, bias=bias, device=device)

    def close(self):
        self.job.cancel()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class RemoteTensor:
    def __init__(self, device, handle, shape, dtype):
        self.device, self.handle, self.shape, self.dtype = device, handle, tuple(shape), dtype

    def cpu(self):
        return self.device.receive(self.device.call("get", handle=self.handle)["hash"])["tensor"]

    def operation(self, op, other=None):
        if other is not None and other.device is not self.device:
            raise ValueError("tensors must belong to the same remote session")
        payload = {"a": self.handle}
        if other is not None:
            payload["b"] = other.handle
        return RemoteTensor(self.device, **self.device.call(op, **payload))

    def __matmul__(self, other):
        return self.operation("matmul", other)

    def __add__(self, other):
        return self.operation("add", other)

    def relu(self):
        return self.operation("relu")

    def release(self):
        return self.device.call("delete", handles=[self.handle])
