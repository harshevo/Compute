import importlib.util
import sys
import time
import uuid

from .storage import atomic_json, blob_path, digest, read_json, safe_path


def publish(root, tensors):
    from safetensors.torch import save
    data = save({key: value.detach().cpu().contiguous().clone() for key, value in tensors.items()})
    value = digest(data)
    path = blob_path(root, value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_name(".tmp-" + uuid.uuid4().hex)
        temporary.write_bytes(data)
        temporary.replace(path)
    return value


def load(root, value, device):
    from safetensors.torch import load_file
    return {key: tensor.to(device) for key, tensor in load_file(str(blob_path(root, value))).items()}


class Session:
    def __init__(self, root, work, device):
        from .hardware import initialize
        self.root, self.work, self.device = root, work, initialize(device)
        self.tensors, self.graphs = {}, {}
        self.model = self.optimizer = self.module = None
        self.trainable_only = False

    def handle(self, request):
        op = request["op"]
        try:
            import torch
            inference_context = torch.inference_mode
        except ImportError:
            if op not in {"inference_setup", "inference_start"}:
                raise
            from contextlib import nullcontext
            inference_context = nullcontext
        if op == "put":
            values = load(self.root, request["hash"], self.device)
            tensor = values["tensor"]
            key = uuid.uuid4().hex
            self.tensors[key] = tensor
            return {"handle": key, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        if op == "get":
            return {"hash": publish(self.root, {"tensor": self.tensors[request["handle"]]})}
        if op == "delete":
            for handle in request["handles"]:
                self.tensors.pop(handle, None)
                self.graphs.pop(handle, None)
            return {"ok": True}
        if op in {"matmul", "add", "relu"}:
            a = self.tensors[request["a"]]
            if op == "relu":
                result = a.relu()
            else:
                b = self.tensors[request["b"]]
                result = a @ b if op == "matmul" else a + b
            handle = uuid.uuid4().hex
            self.tensors[handle] = result
            return {"handle": handle, "shape": list(result.shape), "dtype": str(result.dtype)}
        if op == "linear_forward":
            if len(self.graphs) >= 128:
                raise RuntimeError("too many outstanding autograd graphs")
            values = load(self.root, request["hash"], self.device)
            x, w = (values[name].requires_grad_() for name in ("x", "w"))
            bias = values.get("b")
            if bias is not None:
                bias.requires_grad_()
            with torch.enable_grad():
                y = torch.nn.functional.linear(x, w, bias)
            token = uuid.uuid4().hex
            self.graphs[token] = (x, w, bias, y)
            return {"graph": token, "hash": publish(self.root, {"y": y})}
        if op == "linear_backward":
            x, w, bias, y = self.graphs.pop(request["graph"])
            gradient = load(self.root, request["hash"], self.device)["gradient"]
            inputs = (x, w, bias) if bias is not None else (x, w)
            values = torch.autograd.grad(y, inputs, gradient)
            return {"hash": publish(self.root, dict(zip(("x", "w", "b"), values)))}
        if op == "module_mode":
            self.stage.train(request["training"])
            return {"ok": True}
        if op == "module_setup":
            if hasattr(self, "stage"):
                raise ValueError("one split stage per remote session; create another session")
            path = safe_path(self.work, request["module"])
            sys.path.insert(0, str(self.work))
            spec = importlib.util.spec_from_file_location("compute_stage", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.stage = module.build_stage(request.get("index", 0)).to(self.device)
            parameters = list(self.stage.parameters())
            self.stage_optimizer = module.make_optimizer(parameters) if parameters else None
            self.stage_graphs = {}
            return {"parameters": sum(p.numel() for p in parameters)}
        if op == "module_zero_grad":
            if self.stage_optimizer is not None:
                self.stage_optimizer.zero_grad(set_to_none=True)
            return {"ok": True}
        if op == "module_forward":
            if len(self.stage_graphs) >= 128:
                raise RuntimeError("too many outstanding stage graphs")
            x = load(self.root, request["hash"], self.device)["x"].requires_grad_(True)
            with torch.enable_grad():
                y = self.stage(x)
            if not isinstance(y, torch.Tensor):
                raise ValueError("split stage must return one tensor")
            token = uuid.uuid4().hex
            self.stage_graphs[token] = (x, y)
            return {"graph": token, "hash": publish(self.root, {"y": y})}
        if op == "module_backward":
            x, y = self.stage_graphs.pop(request["graph"])
            gradient = load(self.root, request["hash"], self.device)["gradient"]
            torch.autograd.backward(y, gradient)
            return {"hash": publish(self.root, {"x": x.grad})}
        if op == "module_step":
            if self.stage_graphs:
                raise RuntimeError("finish backward passes before updating remote parameters")
            if self.stage_optimizer is not None:
                params = list(self.stage.parameters())
                if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in params):
                    raise ValueError("non-finite remote gradient")
                if request.get("max_norm"):
                    torch.nn.utils.clip_grad_norm_(params, request["max_norm"])
                self.stage_optimizer.step()
            return {"ok": True}
        if op == "module_export":
            if self.stage_graphs:
                raise RuntimeError("snapshot requires a step boundary")
            from .context import encode_state
            tensors = {}
            metadata = encode_state({"model": self.stage.state_dict(),
                                     "optimizer": self.stage_optimizer.state_dict() if self.stage_optimizer else None}, tensors)
            import json
            encoded = json.dumps(metadata, allow_nan=False).encode()
            meta_hash = digest(encoded)
            path = blob_path(self.root, meta_hash)
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                temporary = path.with_name(".tmp-" + uuid.uuid4().hex)
                temporary.write_bytes(encoded)
                temporary.replace(path)
            return {"hash": publish(self.root, tensors), "metadata": meta_hash}
        if op == "module_import":
            if self.stage_graphs:
                raise RuntimeError("restore requires a step boundary")
            import json
            from .context import decode_state
            state = decode_state(json.loads(blob_path(self.root, request["metadata"]).read_text()),
                                 load(self.root, request["hash"], "cpu"))
            expected = self.stage.state_dict()
            if set(expected) != set(state["model"]):
                raise ValueError("stage checkpoint keys mismatch")
            for key, value in state["model"].items():
                if value.shape != expected[key].shape or value.dtype != expected[key].dtype:
                    raise ValueError("stage checkpoint shape/dtype mismatch")
                if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                    raise ValueError("stage checkpoint is non-finite")
            self.stage.load_state_dict(state["model"])
            if self.stage_optimizer:
                self.stage_optimizer.load_state_dict(state["optimizer"])
            return {"ok": True}
        if op == "module_inference":
            x = load(self.root, request["hash"], self.device)["x"]
            with torch.inference_mode():
                y = self.stage(x)
            return {"hash": publish(self.root, {"y": y})}
        if op == "inference_setup":
            path = safe_path(self.work, request["module"])
            sys.path.insert(0, str(self.work))
            spec = importlib.util.spec_from_file_location("compute_inference", path)
            self.inference_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.inference_module)
            self.inference_model = self.inference_module.build_model(self.device)
            import threading
            self.inference_lock = threading.Lock()
            return {"ready": True}
        if op == "inference_start":
            if not self.inference_lock.acquire(blocking=False):
                raise RuntimeError("inference replica busy")
            import threading
            token = uuid.uuid4().hex
            path = self.work.parent / "inference" / (token + ".jsonl")
            path.parent.mkdir(exist_ok=True)
            def generate():
                import json
                try:
                    with path.open("w") as output:
                        with inference_context():
                            stream = self.inference_module.generate(self.inference_model, request["messages"], request.get("parameters", {}))
                            if isinstance(stream, str):
                                stream = [stream]
                            for text in stream:
                                output.write(json.dumps({"token": str(text)}) + "\n")
                                output.flush()
                        output.write(json.dumps({"done": True}) + "\n")
                        output.flush()
                except Exception as error:
                    with path.open("a") as output:
                        output.write(json.dumps({"error": str(error), "done": True}) + "\n")
                finally:
                    self.inference_lock.release()
            threading.Thread(target=generate, daemon=True).start()
            return {"request": token}
        if op == "train_setup":
            path = safe_path(self.work, request["module"])
            sys.path.insert(0, str(self.work))
            spec = importlib.util.spec_from_file_location("compute_training", path)
            self.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.module)
            torch.manual_seed(request.get("seed", 0))
            self.model = self.module.build_model().to(self.device)
            self.trainable_only = request.get("strategy") == "lora_avg"
            if self.trainable_only:
                frozen = 0
                for name, param in self.model.named_parameters():
                    if "lora_A" in name:
                        param.requires_grad_(False)
                        frozen += 1
                if not frozen:
                    raise ValueError("lora_avg requires lora_A parameters to freeze")
                for name, param in self.model.named_parameters():
                    if "lora_B" not in name:
                        param.requires_grad_(False)
                if not any(p.requires_grad for p in self.model.parameters()):
                    raise ValueError("no trainable lora_B parameters")
            if hasattr(self.module, "configure"):
                self.module.configure(request.get("rank", 0), request.get("world_size", 1), str(self.work / "data"))
            elif request.get("world_size", 1) > 1:
                raise ValueError("pooled module must define configure(rank, world_size, data_path) for sharding")
            self.optimizer = self.module.make_optimizer([p for p in self.model.parameters() if p.requires_grad])
            return self.state()
        if op == "train_round":
            before = time.monotonic()
            metrics = self.module.train_step(self.model, self.optimizer, request["batch_size"], request["steps"])
            return {**self.state(), "metrics": metrics, "seconds": time.monotonic() - before,
                    "samples": request["batch_size"] * request["steps"]}
        if op == "state":
            return self.state()
        if op == "set_state":
            values = load(self.root, request["hash"], self.device)
            expected = self.weights()
            if set(expected) != set(values):
                raise ValueError("model state keys mismatch")
            for name, value in values.items():
                if value.shape != expected[name].shape or value.dtype != expected[name].dtype:
                    raise ValueError("model state shape/dtype mismatch")
                if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                    raise ValueError("non-finite model update")
            with torch.no_grad():
                for name, value in values.items():
                    expected[name].copy_(value)
            return {"ok": True}
        raise ValueError(f"unsupported tensor operation {op}")

    def weights(self):
        if self.model is None:
            raise ValueError("training model not initialized")
        if self.trainable_only:
            return {name: p for name, p in self.model.named_parameters() if p.requires_grad}
        return self.model.state_dict()

    def state(self):
        return {"hash": publish(self.root, self.weights())}


def serve(directory, device, interrupted, timeout):
    session = Session(directory.parent.parent, directory / "work", device)
    inbox, outbox = directory / "inbox", directory / "outbox"
    inbox.mkdir(exist_ok=True)
    outbox.mkdir(exist_ok=True)
    deadline = time.monotonic() + timeout
    while not interrupted.is_set() and time.monotonic() < deadline:
        for path in sorted(inbox.glob("*.json")):
            try:
                result = session.handle(read_json(path))
            except Exception as error:
                result = {"error": f"{type(error).__name__}: {error}"}
            atomic_json(outbox / path.name, result)
            path.unlink()
        time.sleep(.005)

if __name__ == "__main__":
    import threading
    from pathlib import Path
    serve(Path(sys.argv[1]), sys.argv[2], threading.Event(), 86400)
