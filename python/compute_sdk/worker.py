"""Authenticated Rust-agent backend. Each invocation handles one bounded request."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

from .storage import (CHUNK, FileLock, atomic_json, blob_path, digest, identifier,
                      read_chunk, read_json, safe_path, write_chunk)

TERMINAL = {"completed", "failed", "cancelled", "interrupted"}


def job_dir(root, job):
    return root / "jobs" / identifier(job)


def policies(root):
    path = root / "policy.json"
    return read_json(path) if path.exists() else {"max_jobs": 1, "max_disk_bytes": 50 * 1024**3}


def require_execution(root):
    if not (root / "execution-enabled").exists():
        raise PermissionError("Owner must enable jobs with compute agent run --allow-execution")
    from .policy import allowed
    okay, reason = allowed(root)
    if not okay:
        raise PermissionError(reason)
    if (root / "paused").exists():
        raise PermissionError("Owner paused this worker")


def launch(root, directory):
    command = [sys.executable, "-m", "compute_sdk.runner", "--root", str(root), str(directory)]
    kwargs = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen(command, **kwargs)


def submit(root, request):
    require_execution(root)
    import math
    timeout = request.get("timeout", 86400)
    if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 1 <= timeout <= 30 * 86400:
        raise ValueError("timeout must be between one second and thirty days")
    threads = request.get("cpu_threads", 1)
    if not isinstance(threads, int) or not 1 <= threads <= (os.cpu_count() or 1):
        raise ValueError("invalid cpu_threads")
    job = identifier(request["id"])
    directory = job_dir(root, job)
    with FileLock(root / "dispatch.lock"):
        if directory.exists():
            if read_json(directory / "request.json") != request:
                raise ValueError("job ID already belongs to a different request")
            return status(root, {"id": job})
        active = []
        for state in (root / "jobs").glob("*/status.json"):
            current = status(root, {"id": state.parent.name})
            if current["state"] not in TERMINAL:
                active.append(current)
        if len(active) >= policies(root).get("max_jobs", 1):
            raise RuntimeError("worker busy: capacity exhausted")
        files = request.get("files", {})
        if len(files) > 10000:
            raise ValueError("too many files")
        total = sum(blob_path(root, value).stat().st_size for value in files.values())
        if shutil.disk_usage(root).free < total + 64 * 1024**2:
            raise RuntimeError("not enough free disk")
        staging = root / "jobs" / (".staging-" + job)
        if staging.exists():
            shutil.rmtree(staging)
        (staging / "work").mkdir(parents=True)
        try:
            for name, value in files.items():
                target = safe_path(staging / "work", name)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(blob_path(root, value), target)
            if request.get("kind", "script") == "script":
                script = safe_path(staging / "work", request["script"])
                if not script.is_file():
                    raise ValueError("script not in bundle")
            elif request["kind"] != "tensor":
                raise ValueError("unsupported worker kind")
            args = request.get("args", [])
            if not isinstance(args, list) or not all(isinstance(x, str) for x in args):
                raise ValueError("args must be strings")
            atomic_json(staging / "request.json", request)
            status = {"id": job, "state": "queued", "exit_code": None, "created": time.time(),
                      "device": request.get("device", "auto")}
            atomic_json(staging / "status.json", status)
            staging.rename(directory)
            launch(root, directory)
            return status
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise


def status(root, request):
    directory = job_dir(root, request["id"])
    value = read_json(directory / "status.json")
    if value["state"] not in TERMINAL:
        heartbeat = directory / "alive"
        last = heartbeat.stat().st_mtime if heartbeat.exists() else value["created"]
        if time.time() - last > 30:
            value = {**value, "state": "interrupted", "error": "runner heartbeat expired; explicit resume required"}
    return value


def child_process(value):
    import psutil
    pid = value.get("child_pid")
    if not pid:
        return None
    try:
        process = psutil.Process(pid)
        if abs(process.create_time() - value.get("child_created", -1)) < .01 and process.status() != psutil.STATUS_ZOMBIE:
            return process
    except psutil.NoSuchProcess:
        pass
    return None


def dispatch(root, operation, request):
    root.mkdir(parents=True, exist_ok=True)
    if operation == "inventory":
        from .hardware import devices
        inventory = devices()
        atomic_json(root / "inventory.json", inventory)
        return {"os": sys.platform, "architecture": __import__("platform").machine(),
                "cpu_threads": os.cpu_count(), "devices": inventory, "policy": policies(root),
                "execution_enabled": (root / "execution-enabled").exists(),
                "paused": (root / "paused").exists()}
    if operation == "upload":
        require_execution(root)
        with FileLock(root / "upload.lock"):
            used = sum(p.stat().st_size for p in (root / "blobs").glob("*") if p.is_file())
            if used + min(CHUNK, request["size"] - request["offset"]) > policies(root).get("max_disk_bytes", 50 * 1024**3):
                raise RuntimeError("cache quota exceeded")
            if shutil.disk_usage(root).free < CHUNK + 64 * 1024**2:
                raise RuntimeError("insufficient disk space")
            from .policy import transfer
            transfer(root, len(request["data"]) * 3 // 4)
            return write_chunk(root, request)
    if operation == "blob":
        reply = read_chunk(blob_path(root, request["hash"]), request.get("offset", 0))
        from .policy import transfer
        transfer(root, len(reply["data"]) * 3 // 4)
        return reply
    if operation == "submit":
        return submit(root, request)
    if operation == "status":
        return status(root, request)
    if operation == "jobs":
        return [status(root, {"id": p.parent.name}) for p in (root / "jobs").glob("*/status.json") if not p.parent.name.startswith(".")]
    if operation == "logs":
        path = job_dir(root, request["id"]) / "output.log"
        return read_chunk(path, request.get("offset", 0)) if path.exists() else {"data": "", "size": 0}
    if operation == "cancel":
        directory = job_dir(root, request["id"])
        current = status(root, request)
        if current["state"] not in {"completed", "failed", "cancelled"}:
            (directory / "cancel").touch()
            if current["state"] == "interrupted":
                process = child_process(current)
                if process is not None:
                    import psutil
                    descendants = process.children(recursive=True)
                    for target in [*reversed(descendants), process]:
                        try:
                            target.terminate()
                        except psutil.NoSuchProcess:
                            pass
                    _, alive = psutil.wait_procs([*descendants, process], timeout=5)
                    for target in alive:
                        target.kill()
                atomic_json(directory / "status.json", {**current, "state": "cancelled"})
        return status(root, request)
    if operation == "resume":
        require_execution(root)
        directory = job_dir(root, request["id"])
        with FileLock(root / "dispatch.lock"):
            value = status(root, request)
            if value["state"] not in {"interrupted", "failed", "cancelled"}:
                raise ValueError("only stopped jobs can be resumed")
            # An expired heartbeat alone is insufficient proof that the old process stopped.
            if child_process(value) is not None:
                raise RuntimeError("previous job process is still alive; cancel it before resume")
            active = [p for p in (root / "jobs").glob("*/status.json")
                      if p.parent != directory and status(root, {"id": p.parent.name})["state"] not in TERMINAL]
            if len(active) >= policies(root).get("max_jobs", 1):
                raise RuntimeError("worker busy: capacity exhausted")
            with FileLock(directory / "runner.lock", blocking=False):
                (directory / "cancel").unlink(missing_ok=True)
                atomic_json(directory / "status.json", {**value, "state": "queued", "created": time.time()})
                launch(root, directory)
        return {"id": request["id"], "state": "queued"}
    if operation == "artifacts":
        directory = job_dir(root, request["id"]) / "work" / "results"
        files = {}
        if directory.exists():
            for path in directory.rglob("*"):
                if path.is_file() and not path.is_symlink():
                    with path.open("rb") as f:
                        import blake3
                        h = blake3.blake3()
                        while chunk := f.read(CHUNK):
                            h.update(chunk)
                    files[str(path.relative_to(directory))] = {"hash": h.hexdigest(), "size": path.stat().st_size}
        return files
    if operation == "download":
        directory = job_dir(root, request["id"]) / "work" / "results"
        reply = read_chunk(safe_path(directory, request["path"]), request.get("offset", 0))
        from .policy import transfer
        transfer(root, len(reply["data"]) * 3 // 4)
        return reply
    if operation == "inference_read":
        token = request["request"]
        if not isinstance(token, str) or len(token) != 32 or any(c not in "0123456789abcdef" for c in token):
            raise ValueError("invalid inference request")
        path = job_dir(root, request["id"]) / "inference" / (token + ".jsonl")
        return read_chunk(path, request.get("offset", 0)) if path.exists() else {"data": "", "size": 0}
    if operation == "tensor":
        require_execution(root)
        directory = job_dir(root, request["id"])
        if status(root, request)["state"] != "running":
            raise RuntimeError("remote device session is not running")
        token = uuid.uuid4().hex
        inbox, outbox = directory / "inbox", directory / "outbox"
        inbox.mkdir(exist_ok=True)
        outbox.mkdir(exist_ok=True)
        atomic_json(inbox / f"{token}.json", request)
        output = outbox / f"{token}.json"
        deadline = time.monotonic() + 90
        while not output.exists():
            if time.monotonic() > deadline:
                raise TimeoutError("remote tensor operation timed out")
            time.sleep(.01)
        response = read_json(output)
        output.unlink()
        return response
    raise ValueError(f"unsupported operation: {operation}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("operation")
    args = parser.parse_args()
    try:
        request = json.loads(sys.stdin.read(900001))
        result = dispatch(args.root.resolve(), args.operation, request)
    except Exception as error:
        result = {"error": f"{type(error).__name__}: {error}"}
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
