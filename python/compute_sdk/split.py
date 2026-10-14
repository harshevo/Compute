"""Explicit split stages with remote parameters, autograd and snapshot migration."""
import threading
from pathlib import Path

import torch
from .storage import atomic_json, digest, read_json, safe_path


class RemoteModule(torch.nn.Module):
    def __init__(self, remote, script, index=0):
        super().__init__()
        self.remote, self.script, self.index = remote, Path(script).name, index
        self.lock = threading.RLock()
        self.outstanding = 0
        self._trigger = torch.nn.Parameter(torch.zeros(()))
        self.remote.call("module_setup", module=self.script, index=index)
        self.timings = []

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, "remote"):
            self.remote.call("module_mode", training=mode)
        return self

    def forward(self, x):
        import time
        stage = self
        if not torch.is_grad_enabled():
            reply = self.remote.call("module_inference", hash=self.remote.send({"x": x}))
            return self.remote.receive(reply["hash"])["y"].to(x.device)
        class Apply(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, trigger):
                with stage.lock:
                    start = time.monotonic()
                    reply = stage.remote.call("module_forward", hash=stage.remote.send({"x": x}))
                    stage.outstanding += 1
                    ctx.graph, ctx.remote, ctx.device = reply["graph"], stage.remote, x.device
                    result = stage.remote.receive(reply["hash"])["y"].to(x.device)
                    stage.timings.append(time.monotonic() - start)
                    return result

            @staticmethod
            def backward(ctx, gradient):
                with stage.lock:
                    reply = ctx.remote.call("module_backward", graph=ctx.graph,
                                            hash=ctx.remote.send({"gradient": gradient}))
                    result = ctx.remote.receive(reply["hash"])["x"].to(ctx.device)
                    stage.outstanding -= 1
                    return result, None
        return Apply.apply(x, self._trigger)

    def zero_grad(self, set_to_none=True):
        self.remote.call("module_zero_grad")
        super().zero_grad(set_to_none=set_to_none)

    def step(self, max_norm=None):
        return self.remote.call("module_step", max_norm=max_norm)

    def export(self, directory):
        with self.lock:
            if self.outstanding:
                raise RuntimeError("finish all backward passes before snapshot")
            manifest = self.remote.call("module_export")
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            for key, filename in (("hash", "state.safetensors"), ("metadata", "metadata.json")):
                self.remote.download_blob(manifest[key], directory / filename)
            atomic_json(directory / "manifest.json", manifest)
            return manifest

    def restore(self, directory):
        with self.lock:
            if self.outstanding:
                raise RuntimeError("finish all backward passes before restore")
            directory = Path(directory)
            manifest = read_json(directory / "manifest.json")
            for key, filename in (("hash", "state.safetensors"), ("metadata", "metadata.json")):
                path = directory / filename
                if digest(path.read_bytes()) != manifest[key]:
                    raise ValueError("stage checkpoint hash mismatch")
            weights = self.remote.cluster.upload(self.remote.node, directory / "state.safetensors")
            metadata = self.remote.cluster.upload(self.remote.node, directory / "metadata.json")
            self.remote.call("module_import", hash=weights, metadata=metadata)

    def move_to(self, destination):
        import tempfile
        with self.lock:
            if self.outstanding:
                raise RuntimeError("migration requires a step boundary")
            with tempfile.TemporaryDirectory() as tmp:
                self.export(tmp)
                destination.call("module_setup", module=self.script, index=self.index)
                manifest = read_json(Path(tmp) / "manifest.json")
                weights = destination.cluster.upload(destination.node, Path(tmp) / "state.safetensors")
                metadata = destination.cluster.upload(destination.node, Path(tmp) / "metadata.json")
                destination.call("module_import", hash=weights, metadata=metadata)
                old = self.remote
                self.remote = destination
                return old


class SplitTrainer:
    def __init__(self, model, local_optimizer=None, results="split-results"):
        self.model, self.local_optimizer = model, local_optimizer
        self.stages = [module for module in model.modules() if isinstance(module, RemoteModule)]
        self.results = Path(results).resolve()
        self.results.mkdir(parents=True, exist_ok=True)

    def zero_grad(self):
        if self.local_optimizer:
            self.local_optimizer.zero_grad()
        for stage in self.stages:
            stage.zero_grad()

    def step(self, max_norm=None):
        for stage in self.stages:
            stage.step(max_norm)
        if self.local_optimizer:
            self.local_optimizer.step()

    def checkpoint(self, step):
        import uuid
        from .context import Context
        generation = f"snapshot-{int(step)}-{uuid.uuid4().hex}"
        directory = self.results / generation
        directory.mkdir()
        for index, stage in enumerate(self.stages):
            stage.export(directory / f"stage-{index}")
        if self.local_optimizer:
            Context(directory / "local").checkpoint(self.model, self.local_optimizer, step)
        atomic_json(directory / "snapshot.json", {"step": step, "stages": len(self.stages)})
        atomic_json(self.results / "latest.json", {"generation": generation})

    def restore(self):
        from .context import Context
        pointer = self.results / "latest.json"
        if not pointer.exists():
            return 0
        directory = safe_path(self.results, read_json(pointer)["generation"])
        manifest = read_json(directory / "snapshot.json")
        if manifest["stages"] != len(self.stages):
            raise ValueError("snapshot stage count mismatch")
        for index, stage in enumerate(self.stages):
            stage.restore(directory / f"stage-{index}")
        if self.local_optimizer:
            Context(directory / "local").restore(self.model, self.local_optimizer)
        return manifest["step"]
