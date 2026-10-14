import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from .storage import FileLock, atomic_json, read_json, safe_path


def stop(child):
    if child.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"], capture_output=True)
    else:
        try:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    child.wait(timeout=10)


def run(root, directory):
    with FileLock(directory / "runner.lock"):
        request = read_json(directory / "request.json")
        value = read_json(directory / "status.json")
        work = directory / "work"
        child = None
        ended = threading.Event()
        interrupted = threading.Event()
        def heartbeat():
            while not ended.wait(1):
                (directory / "alive").touch()
                from .policy import allowed
                okay, _ = allowed(root)
                if not okay or (root / "paused").exists() or (directory / "cancel").exists():
                    interrupted.set()
        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        (directory / "alive").touch()
        try:
            from .hardware import select
            device = select(request.get("device", "auto"), training=not request.get("inference", False))
            value = {**value, "state": "running", "device": device, "pid": os.getpid(), "started": time.time()}
            atomic_json(directory / "status.json", value)
            env = {key: val for key, val in os.environ.items() if key in {
                "PATH", "SYSTEMROOT", "WINDIR", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH",
                "LANG", "TMP", "TEMP", "CUDA_VISIBLE_DEVICES", "ONEAPI_DEVICE_SELECTOR"}}
            env.update({"COMPUTE_JOB_ID": request["id"], "COMPUTE_DEVICE": device,
                        "COMPUTE_RESULTS": str(work / "results"), "PYTHONUNBUFFERED": "1",
                        "HOME": str(work), "USERPROFILE": str(work),
                        "OMP_NUM_THREADS": str(request.get("cpu_threads", max(1, (os.cpu_count() or 1) // 2)))})
            (work / "results").mkdir(exist_ok=True)
            if request.get("kind", "script") == "tensor":
                command = [sys.executable, "-u", "-m", "compute_sdk.tensor_server", str(directory), device]
            else:
                command = [sys.executable, "-u", str(safe_path(work, request["script"])), *request.get("args", [])]
            kwargs = {"start_new_session": True} if os.name != "nt" else {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
            with (directory / "output.log").open("ab") as log:
                child = subprocess.Popen(command, cwd=work, env=env, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, **kwargs)
                import psutil
                value.update(child_pid=child.pid, child_created=psutil.Process(child.pid).create_time())
                atomic_json(directory / "status.json", value)
                deadline = time.monotonic() + request.get("timeout", 86400)
                while child.poll() is None:
                    if interrupted.is_set() or time.monotonic() >= deadline:
                        interrupted.set()
                        stop(child)
                        break
                    time.sleep(.2)
                code = child.wait()
            state = "cancelled" if interrupted.is_set() else "completed" if code == 0 else "failed"
            atomic_json(directory / "status.json", {**value, "state": state, "exit_code": code, "finished": time.time()})
        except BaseException as error:
            if child is not None:
                stop(child)
            with (directory / "output.log").open("a") as log:
                log.write(f"\n{type(error).__name__}: {error}\n")
            atomic_json(directory / "status.json", {**value, "state": "failed", "error": str(error), "finished": time.time()})
        finally:
            ended.set()
            thread.join(timeout=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    run(args.root, args.directory)


if __name__ == "__main__":
    main()
