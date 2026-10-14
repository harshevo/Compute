"""Experimental local SGD and exact frozen-A LoRA aggregation over remote sessions."""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .storage import atomic_json


class PooledTrainer:
    def __init__(self, script, nodes, cluster=None, strategy="local_sgd", device="auto",
                 batch_size=8, sync_every=10, results="pool-results", max_update_norm=1000):
        from .client import Cluster
        if strategy not in {"local_sgd", "lora_avg"}:
            raise ValueError("Use explicit remote linear layers for split training")
        if not nodes or len(set(nodes)) != len(nodes):
            raise ValueError("nodes must be a nonempty unique list")
        if batch_size < 1 or sync_every < 1:
            raise ValueError("batch_size and sync_every must be positive")
        self.script, self.nodes = Path(script).resolve(), list(nodes)
        self.cluster, self.strategy = cluster or Cluster(), strategy
        self.device, self.batch_size, self.sync_every = device, batch_size, sync_every
        self.results, self.max_update_norm = Path(results).resolve(), max_update_norm
        self.results.mkdir(parents=True, exist_ok=True)
        self.history = []

    def fit(self, rounds=10, adaptive=True):
        import torch
        from safetensors.torch import load_file, save_file
        sessions = {}
        batch_sizes = {node: self.batch_size for node in self.nodes}
        try:
            for rank, node in enumerate(self.nodes):
                session = self.cluster.remote(node, self.device, self.script.parent)
                sessions[node] = session
                reply = session.call("train_setup", module=self.script.name, strategy=self.strategy,
                                     rank=rank, world_size=len(self.nodes), seed=0)
                state = session.receive(reply["hash"])
                if rank == 0:
                    global_state = state
                elif state.keys() != global_state.keys() or any(not torch.equal(state[k], global_state[k]) for k in state):
                    raise ValueError("initial model states differ; initialize identical models on all nodes")
            pointer = self.results / "latest.json"
            first_round = 0
            if pointer.exists():
                from .storage import digest, read_json, safe_path
                saved = read_json(pointer)
                path = safe_path(self.results, saved["file"])
                if digest(path.read_bytes()) != saved["hash"]:
                    raise ValueError("pooled checkpoint hash mismatch")
                global_state = load_file(str(path))
                first_round = saved["round"] + 1
            for session in sessions.values():
                session.call("set_state", hash=session.send(global_state))
            for round_id in range(first_round, first_round + rounds):
                started = time.monotonic()
                updates, failed = [], []
                def train(node, session):
                    reply = session.call("train_round", batch_size=batch_sizes[node], steps=self.sync_every)
                    return reply, session.receive(reply["hash"])
                with ThreadPoolExecutor(max_workers=len(sessions)) as executor:
                    futures = {node: executor.submit(train, node, session) for node, session in sessions.items()}
                    for node, future in futures.items():
                        try:
                            reply, state = future.result()
                            if state.keys() != global_state.keys():
                                raise ValueError("update keys mismatch")
                            norm = 0.0
                            for key, value in state.items():
                                base = global_state[key]
                                if value.shape != base.shape or value.dtype != base.dtype:
                                    raise ValueError("update shape/dtype mismatch")
                                if value.is_floating_point():
                                    if not torch.isfinite(value).all():
                                        raise ValueError("non-finite update")
                                    norm += float((value.double() - base.double()).square().sum())
                            if norm**.5 > self.max_update_norm:
                                raise ValueError("update norm exceeds limit")
                            updates.append((node, reply, state))
                        except Exception as error:
                            failed.append({"node": node, "error": str(error)})
                if not updates:
                    raise RuntimeError(f"No valid updates in round {round_id}: {failed}")
                total = sum(reply["samples"] for _, reply, _ in updates)
                new_state = {}
                for key, base in global_state.items():
                    if base.is_floating_point():
                        delta = sum((state[key].double() - base.double()) * (reply["samples"] / total)
                                    for _, reply, state in updates)
                        new_state[key] = (base.double() + delta).to(base.dtype)
                    else:
                        # Integer counters cannot be averaged. Require consistent values.
                        values = [state[key] for _, _, state in updates]
                        if any(not torch.equal(values[0], value) for value in values[1:]):
                            raise ValueError(f"integer buffer {key} differs between workers")
                        new_state[key] = values[0]
                global_state = new_state
                temporary = self.results / f".round-{round_id}.safetensors"
                published = self.results / f"round-{round_id}.safetensors"
                save_file(global_state, str(temporary))
                temporary.replace(published)
                from .storage import digest
                atomic_json(pointer, {"round": round_id, "file": published.name,
                                      "hash": digest(published.read_bytes()), "optimizer": "worker-local; resets on new session"})
                for failure in failed:
                    session = sessions.pop(failure["node"])
                    try:
                        session.close()
                    except Exception:
                        pass
                for node, session in list(sessions.items()):
                    try:
                        session.call("set_state", hash=session.send(global_state))
                    except Exception:
                        sessions.pop(node)
                        try:
                            session.close()
                        except Exception:
                            pass
                record = {"round": round_id, "seconds": time.monotonic() - started, "failures": failed,
                          "workers": [{"node": node, "samples": reply["samples"], "seconds": reply["seconds"],
                                       "metrics": reply["metrics"]} for node, reply, _ in updates]}
                if adaptive:
                    speeds = {node: reply["samples"] / max(reply["seconds"], .001) for node, reply, _ in updates}
                    mean_speed = sum(speeds.values()) / len(speeds)
                    for node, speed in speeds.items():
                        target = max(1, min(self.batch_size * 4, round(self.batch_size * speed / mean_speed)))
                        batch_sizes[node] = max(1, round(.7 * batch_sizes[node] + .3 * target))
                record["next_batch_sizes"] = dict(batch_sizes)
                self.history.append(record)
                with (self.results / "rounds.jsonl").open("a") as log:
                    log.write(json.dumps(record, allow_nan=False) + "\n")
                if not sessions:
                    raise RuntimeError("All workers lost after checkpoint publication")
            return global_state
        finally:
            for session in sessions.values():
                try:
                    session.close()
                except Exception:
                    pass
