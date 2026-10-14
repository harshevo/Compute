import base64
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .storage import CHUNK, digest, safe_path
from .worker import TERMINAL


class Cluster:
    def __init__(self, binary=None, home=None):
        self.binary = binary or os.environ.get("COMPUTE_BINARY") or shutil.which("compute")
        if not self.binary:
            raise RuntimeError("Set COMPUTE_BINARY to the built compute executable")
        self.env = os.environ.copy()
        if home:
            self.env["COMPUTE_HOME"] = str(home)

    def rpc(self, node, operation, payload=None):
        result = subprocess.run([self.binary, "rpc", "--node", node, operation], input=json.dumps(payload or {}),
                                text=True, capture_output=True, env=self.env, timeout=165)
        if result.returncode:
            raise RuntimeError(result.stderr.strip())
        value = json.loads(result.stdout)
        if isinstance(value, dict) and "error" in value:
            raise RuntimeError(value["error"])
        return value

    def nodes(self):
        result = subprocess.run([self.binary, "nodes", "--json"], text=True, capture_output=True,
                                env=self.env, timeout=20, check=True)
        return json.loads(result.stdout)

    def devices(self, include_local=True):
        result = []
        if include_local:
            from .hardware import devices
            result.extend({**item, "node": "local", "remote": False} for item in devices())
        for node in self.nodes():
            if node["liveness"] == "Online":
                inventory = self.rpc(node["name"], "inventory")
                result.extend({**item, "node": node["name"], "remote": True,
                               "paused": inventory["paused"], "execution_enabled": inventory["execution_enabled"]}
                              for item in inventory["devices"])
        return result

    def upload(self, node, path):
        import blake3
        path = Path(path)
        h = blake3.blake3()
        with path.open("rb") as file:
            while chunk := file.read(CHUNK):
                h.update(chunk)
        value, size, offset = h.hexdigest(), path.stat().st_size, 0
        with path.open("rb") as file:
            while True:
                file.seek(offset)
                chunk = file.read(CHUNK)
                reply = self.rpc(node, "upload", {"hash": value, "size": size, "offset": offset,
                                                 "data": base64.b64encode(chunk).decode()})
                next_offset = reply["offset"]
                if reply["complete"]:
                    return value
                if next_offset == offset:
                    raise RuntimeError("upload made no progress")
                offset = next_offset

    def bundle(self, node, directory):
        directory = Path(directory).resolve()
        excluded = {".git", ".venv", "__pycache__", "target", "results", "node_modules"}
        files = {}
        for folder, dirs, names in os.walk(directory, followlinks=False):
            dirs[:] = [name for name in dirs if name not in excluded and not name.startswith(".")
                       and not (Path(folder) / name).is_symlink()]
            for name in sorted(names):
                path = Path(folder) / name
                if path.is_symlink() or name.startswith(".") or path.suffix in {".pem", ".key", ".pyc"}:
                    continue
                files[path.relative_to(directory).as_posix()] = self.upload(node, path)
        return files

    def choose(self, device="auto", min_memory=0, exclude=()):
        candidates = [d for d in self.devices(False) if d["training"] and not d["paused"]
                      and d["execution_enabled"] and d["node"] not in exclude
                      and (device == "auto" or d["id"] == device)
                      and (not min_memory or d["memory_bytes"] >= min_memory)]
        if not candidates:
            raise RuntimeError("No available device satisfies the request")
        candidates.sort(key=lambda d: (d["backend"] != "cpu", d["memory_bytes"]), reverse=True)
        return candidates[0]["node"]

    def submit(self, script, node="auto", data=None, args=(), device="auto", timeout=86400, min_memory=0):
        script = Path(script).resolve()
        if node == "auto":
            node = self.choose(device, min_memory)
        files = self.bundle(node, script.parent)
        if data:
            source = Path(data).resolve()
            files.update({"data/" + name: value for name, value in self.bundle(node, source).items()})
        import uuid
        job = digest(uuid.uuid4().bytes)
        self.rpc(node, "submit", {"id": job, "files": files, "script": script.name,
                                 "args": list(args), "device": device, "timeout": timeout})
        return Job(self, node, job)

    def job(self, node, job_id):
        return Job(self, node, job_id)

    def remote(self, node, device="auto", directory=None, inference=False):
        from .remote import RemoteDevice
        return RemoteDevice(self, node, device, directory, inference)

    def sweep(self, script, trials, nodes=None, device="auto", data=None):
        nodes = nodes or [n["name"] for n in self.nodes() if n["liveness"] == "Online" and not n["status"]["paused"]]
        if not nodes:
            raise RuntimeError("No workers for sweep")
        # One dispatcher per node bounds concurrency; every trial has a durable worker job ID.
        groups = [list(enumerate(trials))[i::len(nodes)] for i in range(len(nodes))]
        def execute(node, group):
            results = []
            for index, arguments in group:
                job = self.submit(script, node, data, arguments, device)
                job.wait()
                results.append((index, job))
            return results
        with ThreadPoolExecutor(max_workers=len(nodes)) as executor:
            futures = [executor.submit(execute, node, group) for node, group in zip(nodes, groups)]
            jobs = [pair for future in futures for pair in future.result()]
        return [job for _, job in sorted(jobs)]


class Job:
    def __init__(self, cluster, node, job_id):
        self.cluster, self.node, self.id = cluster, node, job_id

    def status(self):
        return self.cluster.rpc(self.node, "status", {"id": self.id})

    def cancel(self):
        return self.cluster.rpc(self.node, "cancel", {"id": self.id})

    def resume(self):
        return self.cluster.rpc(self.node, "resume", {"id": self.id})

    def wait(self, timeout=None, poll=1):
        deadline = time.monotonic() + timeout if timeout else float("inf")
        while time.monotonic() < deadline:
            value = self.status()
            if value["state"] in TERMINAL:
                if value["state"] != "completed":
                    raise RuntimeError(f"Job {self.id}: {value}")
                return value
            time.sleep(poll)
        raise TimeoutError(self.id)

    def stream_logs(self, follow=True):
        offset = 0
        while True:
            reply = self.cluster.rpc(self.node, "logs", {"id": self.id, "offset": offset})
            data = base64.b64decode(reply["data"])
            offset += len(data)
            if data:
                print(data.decode(errors="replace"), end="", flush=True)
            if not follow:
                return
            if self.status()["state"] in TERMINAL and offset >= reply["size"]:
                # Fetch once more because completion may occur just after the preceding log read.
                reply = self.cluster.rpc(self.node, "logs", {"id": self.id, "offset": offset})
                print(base64.b64decode(reply["data"]).decode(errors="replace"), end="", flush=True)
                return
            time.sleep(.5)

    def download(self, destination):
        destination = Path(destination).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        files = self.cluster.rpc(self.node, "artifacts", {"id": self.id})
        for name, metadata in files.items():
            target = safe_path(destination, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".partial")
            offset = 0
            with temporary.open("wb") as file:
                while offset < metadata["size"]:
                    reply = self.cluster.rpc(self.node, "download", {"id": self.id, "path": name, "offset": offset})
                    data = base64.b64decode(reply["data"])
                    if not data:
                        raise RuntimeError("artifact truncated during transfer")
                    file.write(data)
                    offset += len(data)
            import blake3
            h = blake3.blake3()
            with temporary.open("rb") as file:
                while chunk := file.read(CHUNK):
                    h.update(chunk)
            if offset != metadata["size"] or h.hexdigest() != metadata["hash"]:
                temporary.unlink()
                raise RuntimeError("artifact changed or failed integrity check; retry after checkpoint publication")
            temporary.replace(target)
        return destination
