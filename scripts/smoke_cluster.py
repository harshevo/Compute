"""Real QUIC acceptance test, isolated state, no permanent services."""
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
import compute_sdk as cs

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "target" / "debug" / "compute"


def wait_until(fn, timeout=45):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            value = fn()
            if value:
                return value
        except Exception:
            pass
        time.sleep(.2)
    raise TimeoutError("condition did not become ready")


def main():
    children, logs = [], []
    with tempfile.TemporaryDirectory(prefix="compute-smoke-") as tmp:
        base = Path(tmp)
        env = {**os.environ, "COMPUTE_HOME": str(base / "owner"), "COMPUTE_PYTHON": sys.executable,
               "COMPUTE_BINARY": str(BINARY)}
        def start(args, environment):
            log = (base / f"process-{len(children)}.log").open("w")
            logs.append(log)
            process = subprocess.Popen([str(BINARY), *args], env=environment, stdout=log, stderr=log)
            children.append(process)
            return process
        try:
            owner = start(["serve"], env)
            wait_until(lambda: (base / "owner" / "coordinator" / "admin.sock").exists())
            for name in ("worker-a", "worker-b"):
                invite = subprocess.run([str(BINARY), "invite", name], env=env, capture_output=True,
                                        text=True, check=True).stdout
                ticket = re.search(r"agent join (\S+)", invite)[1]
                agent_env = {**env, "COMPUTE_HOME": str(base / name)}
                subprocess.run([str(BINARY), "agent", "join", ticket], env=agent_env, capture_output=True, check=True)
                start(["agent", "run", "--allow-execution"], agent_env)
            cluster = cs.connect(str(BINARY), base / "owner")
            wait_until(lambda: len([n for n in cluster.nodes() if n["liveness"] == "Online"]) == 2)
            print("PASS: two authenticated QUIC workers", flush=True)
            inventory = cluster.rpc("worker-a", "inventory")
            assert any(d["id"] == "cpu" and d["training"] for d in inventory["devices"])
            print("INVENTORY:", json.dumps(inventory), flush=True)
            job = cluster.submit(ROOT / "examples" / "train.py", "worker-a", args=["--steps", "10"], device="cpu")
            job.wait(timeout=60)
            result = job.download(base / "results")
            assert (result / "latest.json").exists()
            print("PASS: training, logs, checkpoint and BLAKE3-verified result transfer", flush=True)
            with cluster.remote("worker-a", "cpu") as remote:
                a, b = torch.randn(3, 4), torch.randn(4, 2)
                ra, rb = remote.tensor(a), remote.tensor(b)
                rc = ra @ rb
                torch.testing.assert_close(rc.cpu(), a @ b)
                for tensor in (ra, rb, rc):
                    tensor.release()
                local = torch.nn.Linear(4, 2)
                layer = remote.linear(4, 2)
                layer.load_state_dict(local.state_dict())
                x = torch.randn(3, 4, requires_grad=True)
                clone = x.detach().clone().requires_grad_()
                y = layer(x)
                expected = local(clone)
                torch.testing.assert_close(y, expected)
                y.square().sum().backward()
                expected.square().sum().backward()
                torch.testing.assert_close(x.grad, clone.grad)
                torch.testing.assert_close(layer.weight.grad, local.weight.grad)
            wait_until(lambda: remote.job.status()["state"] == "cancelled")
            print("PASS: remote tensor math and mixed autograd gradients over QUIC", flush=True)
            pool = cs.PooledTrainer(ROOT / "examples" / "pooled.py", ["worker-a", "worker-b"],
                                    cluster, device="cpu", batch_size=4, sync_every=2, results=base / "pooled")
            pool.fit(rounds=2)
            assert len(pool.history) == 2
            assert (base / "pooled" / "latest.json").exists()
            wait_until(lambda: all(s["state"] in {"completed", "cancelled"} for s in cluster.rpc("worker-a", "jobs")))
            print("PASS: two-worker local SGD, adaptive allocation, durable round checkpoint", flush=True)
            model = cs.RemoteModel(ROOT / "examples" / "inference.py", "worker-a", cluster, "cpu")
            try:
                tokens = list(model.generate("test", stream=True))
                assert len(tokens) == 3 and "GPU inference result:" in "".join(tokens)
                print("PASS: live inference tokens over QUIC", flush=True)
            finally:
                model.close()
            wait_until(lambda: all(s["state"] in {"completed", "cancelled"} for s in cluster.rpc("worker-a", "jobs")))
            if any(d["id"] == "mps" and d["training"] for d in inventory["devices"]):
                job = cluster.submit(ROOT / "examples" / "train.py", "worker-a", args=["--steps", "5"], device="mps")
                assert job.wait(timeout=60)["device"] == "mps"
                print("PASS: real Apple MPS training through remote worker", flush=True)
            print("ALL ACCEPTANCE CHECKS PASSED", flush=True)
        finally:
            # Detached jobs are intentionally independent of Rust; cancel them before cleaning state.
            try:
                for name in ("worker-a", "worker-b"):
                    for state in cluster.rpc(name, "jobs"):
                        if state["state"] not in {"completed", "cancelled", "failed", "interrupted"}:
                            cluster.job(name, state["id"]).cancel()
                time.sleep(2)
            except Exception:
                pass
            for process in reversed(children):
                if process.poll() is None:
                    process.terminate()
            for process in children:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            for log in logs:
                log.close()
            if sys.exc_info()[0]:
                for path in base.glob("process-*.log"):
                    print(path.name, path.read_text()[-4000:], file=sys.stderr)


if __name__ == "__main__":
    main()
