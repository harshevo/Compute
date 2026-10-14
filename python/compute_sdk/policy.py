import datetime
import subprocess
import time

from .storage import FileLock, atomic_json, read_json


def load(root):
    path = root / "policy.json"
    return read_json(path) if path.exists() else {}


def allowed(root):
    policy = load(root)
    hours = policy.get("allowed_hours")
    if hours is not None and datetime.datetime.now().hour not in hours:
        return False, "outside owner allowed hours"
    if policy.get("pause_when_processes"):
        try:
            import psutil
        except ImportError:
            return False, "install psutil to enforce pause_when_processes"
        blocked = {name.lower() for name in policy["pause_when_processes"]}
        for process in psutil.process_iter(["name"]):
            if (process.info["name"] or "").lower() in blocked:
                return False, "owner priority process is running"
    cap = policy.get("max_temperature_c")
    if cap is not None:
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                                    capture_output=True, text=True, timeout=3, check=True)
            if any(int(line.strip()) >= cap for line in result.stdout.splitlines()):
                return False, "GPU temperature cap reached"
        except Exception:
            return False, "temperature telemetry unavailable"
    return True, ""


def transfer(root, count):
    policy = load(root)
    if count <= 0:
        return
    path = root / "transfer.json"
    with FileLock(root / "transfer.lock"):
        month = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m")
        usage = read_json(path) if path.exists() else {}
        if usage.get("month") != month:
            usage = {"month": month, "bytes": 0}
        limit = policy.get("monthly_transfer_bytes")
        if limit is not None and usage["bytes"] + count > limit:
            raise RuntimeError("owner monthly transfer budget exhausted")
        rate = policy.get("transfer_bytes_per_second")
        now = time.time()
        delay = 0
        if rate is not None:
            if rate < 4096:
                raise ValueError("transfer_bytes_per_second must be at least 4096")
            start = max(now, usage.get("next_transfer", now))
            delay = start - now
            usage["next_transfer"] = start + count / rate
        usage["bytes"] += count
        atomic_json(path, usage)
    if delay > 100:
        raise RuntimeError("transfer rate queue exhausted; retry later")
    time.sleep(delay)
