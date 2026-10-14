import base64
import json
import threading
import time
import uuid
from pathlib import Path


class RemoteModel:
    def __init__(self, script, node, cluster=None, device="auto", name=None):
        from .client import Cluster
        self.script = Path(script).resolve()
        self.cluster, self.node, self.device = cluster or Cluster(), node, device
        self.name = name or self.script.stem
        self.lock = threading.Lock()
        self.session = None
        self.restart()

    def restart(self):
        if self.session is not None:
            try:
                self.session.close()
            except Exception:
                pass
        self.session = self.cluster.remote(self.node, self.device, self.script.parent, inference=True)
        try:
            self.session.call("inference_setup", module=self.script.name)
        except BaseException:
            self.session.close()
            raise

    def generate(self, prompt=None, stream=False, messages=None, timeout=300, **parameters):
        messages = messages or [{"role": "user", "content": prompt or ""}]
        iterator = self._stream(messages, parameters, timeout)
        return iterator if stream else "".join(iterator)

    def _stream(self, messages, parameters, timeout):
        with self.lock:
            if self.session.job.status()["state"] != "running":
                self.restart()
            response = self.session.call("inference_start", messages=messages, parameters=parameters)
            offset, pending = 0, b""
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                reply = self.cluster.rpc(self.node, "inference_read", {"id": self.session.id,
                           "request": response["request"], "offset": offset})
                data = base64.b64decode(reply["data"])
                offset += len(data)
                pending += data
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    record = json.loads(line)
                    if "error" in record:
                        raise RuntimeError(record["error"])
                    if record.get("done"):
                        return
                    yield record["token"]
                if not data:
                    if self.session.job.status()["state"] != "running":
                        raise RuntimeError("inference worker stopped")
                    time.sleep(.05)
            raise TimeoutError("inference request deadline exceeded")

    def close(self):
        if self.session is not None:
            self.session.close()


class OpenVINOModel:
    """Inference on supported Intel integrated GPUs, including compatible Iris Xe."""
    def __init__(self, path, device="GPU"):
        import openvino as ov
        self.model = ov.Core().compile_model(str(path), device)

    def __call__(self, inputs):
        return self.model(inputs)


def create_gateway(models):
    from contextlib import asynccontextmanager
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field
    from starlette.concurrency import run_in_threadpool
    from typing import Literal

    @asynccontextmanager
    async def lifespan(app):
        yield

    app = FastAPI(title="Compute inference gateway", lifespan=lifespan)
    guard = threading.Lock()
    active = {id(model): 0 for model in models}

    class Message(BaseModel):
        role: Literal["system", "user", "assistant", "tool"]
        content: str = Field(max_length=100000)

    class ChatRequest(BaseModel):
        model: str
        messages: list[Message] = Field(min_length=1, max_length=100)
        stream: bool = False
        max_tokens: int = Field(default=256, ge=1, le=8192)
        temperature: float = Field(default=1, ge=0, le=2)

    def reserve(name):
        matching = [model for model in models if model.name == name]
        if not matching:
            raise HTTPException(404, "model not configured")
        with guard:
            ordered = sorted(matching, key=lambda model: active[id(model)])
            chosen = ordered[0]
            if active[id(chosen)] >= 8:
                raise HTTPException(429, "inference queue is full")
            active[id(chosen)] += 1
        return chosen, ordered

    def release(model):
        with guard:
            active[id(model)] -= 1

    @app.get("/v1/models")
    def list_models():
        return {"object": "list", "data": [{"id": name, "object": "model", "owned_by": "compute"}
                 for name in sorted({model.name for model in models})]}

    @app.post("/v1/chat/completions")
    async def chat(body: ChatRequest):
        chosen, replicas = reserve(body.model)
        messages = [item.model_dump() for item in body.messages]
        parameters = {"max_tokens": body.max_tokens, "temperature": body.temperature}
        request_id = "chatcmpl-" + uuid.uuid4().hex
        if body.stream:
            def events():
                emitted = False
                try:
                    last_error = None
                    for replica in replicas:
                        try:
                            for token in replica.generate(messages=messages, stream=True, **parameters):
                                emitted = True
                                record = {"id": request_id, "object": "chat.completion.chunk", "model": body.model,
                                          "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}]}
                                yield "data: " + json.dumps(record) + "\n\n"
                            yield "data: " + json.dumps({"id": request_id, "object": "chat.completion.chunk", "model": body.model,
                                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}) + "\n\n"
                            yield "data: [DONE]\n\n"
                            return
                        except Exception as error:
                            last_error = str(error)
                            if emitted:
                                break
                    yield "data: " + json.dumps({"error": {"message": last_error, "type": "worker_error"}}) + "\n\n"
                finally:
                    release(chosen)
            return StreamingResponse(events(), media_type="text/event-stream")
        try:
            last_error = None
            for replica in replicas:
                try:
                    text = await run_in_threadpool(replica.generate, messages=messages, **parameters)
                    return {"id": request_id, "object": "chat.completion", "model": body.model,
                            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}
                except Exception as error:
                    last_error = str(error)
            raise HTTPException(503, last_error)
        finally:
            release(chosen)
    return app
