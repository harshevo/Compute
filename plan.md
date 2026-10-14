# Plan: A custom distributed GPU stack (Rust + PyTorch)

## Goal

Build my own system to use friends' and family's GPUs over the internet (≥20 ms latency) in two modes:

- **Mode 1 – Standalone:** run a training job or inference server entirely on one remote GPU.
- **Mode 2 – Pooled:** use several GPUs (mine + theirs) at once for the same workload.

The infrastructure is written in **Rust**; the models stay in **PyTorch (Python)**.

## Constraint that shapes the design

Latency of 20 ms is fine. The limit is **home upload bandwidth** (often 10–50 Mbps). Building the stack myself does not remove it, so the design is built around it:

| Workload | Mode 1 | Mode 2 |
|---|---|---|
| Training one model | Full speed of that GPU | Only with small synced state (LoRA adapters) or infrequent syncing; per-step gradient sync of a full model is slower than one GPU |
| Many runs (sweeps, seeds) | n/a | Near-linear speedup |
| Inference, many requests | Works well | Throughput adds up across replicas |
| Inference, model too big for one GPU | n/a | Works by splitting layers, but each token pays the network hop |

The custom stack can beat off-the-shelf tools on the things I control: tensor compression on the wire, sync frequency, scheduling around flaky home PCs, and one small binary to install.

## Phase 0 – Measure before writing code (one evening, go/no-go gate)

Use throwaway tools (Tailscale, SSH, `iperf3`) on one friend's machine to:

1. Record GPU, VRAM, OS, free disk, upload/download speed, any ISP data cap, and available hours for every machine.
2. Run one real training job and one inference server on the remote GPU by hand.

Decisions this gate makes:

- If the remote GPU is not meaningfully faster than mine, the project has no Mode 1 payoff; stop or find better hardware.
- If uploads are under about 10 Mbps, M7 and M8 are dropped from scope; Mode 2 means sweeps and inference replicas only.
- If a location has a data cap, that node gets a monthly transfer budget.

## Key decisions

| Decision | Choice | Why |
|---|---|---|
| Systems language | **Rust** | Memory-safe networking code, good async (tokio), single static binary per OS, first-class Python bindings (PyO3) |
| Where PyTorch runs | **Normal Python processes**, supervised by the Rust agent | Keeps the whole PyTorch ecosystem. Rust never touches autograd |
| Rust ↔ Python boundary | **PyO3 extension module** (built with maturin) | The Python worker calls into Rust for networking and tensor transport, with zero-copy access to tensor memory |
| Transport | **QUIC** via `iroh` (built on `quinn`), hidden behind the `net` crate | Multiplexed streams, encryption, NAT hole punching and relay fallback. `iroh` is pre-1.0, so the version is pinned and nothing outside `net` imports it |
| Identity and trust | Ed25519 node keys; workers accept jobs only from my key | Friends are running my code, so the allow-list is the security boundary |
| Wire format | `postcard` or protobuf (`prost`) for control; raw tensor frames for data | Tensors must not go through a general serializer |
| Durable state | SQLite (WAL mode) on the coordinator; a small job journal on each agent | Both sides survive a restart and can reconcile |
| Hashing and transfer | BLAKE3 content hashes, chunked and resumable, write-to-temp then atomic rename | No partial or corrupt file is ever used |
| Weights on disk | **safetensors only**; never `torch.load` a pickle that came from another machine | A pickle from a compromised worker would run code on my machine |
| GPU inventory | `nvml-wrapper` (NVIDIA); report-only fallback for others | VRAM-aware scheduling |

## Architecture

```
 My machine                                Friend / family PC
┌─────────────────────────┐              ┌──────────────────────────────┐
│ cli  ──► coordinator    │◄── QUIC ───► │ agent (Rust, OS service)     │
│          - node registry│   (iroh)     │  - GPU inventory, heartbeat  │
│          - scheduler    │              │  - owner pause / auto-pause  │
│          - job queue    │              │  - job journal, log buffer   │
│          - SQLite state │              │  - spawns / kills workers    │
│ agent (my own GPU)      │              │        │                     │
└─────────────────────────┘              │  Python worker (PyTorch)     │
                                         │   └─ sdk (PyO3) ─► QUIC      │
                                         └──────────────────────────────┘
```

- The coordinator handles control only. Tensor data flows directly between the peers that need it.
- Agents are the source of truth for what is running. A job keeps running when the coordinator is unreachable (my laptop asleep), and the two reconcile on reconnect.
- The parameter server (M7) is a role, not a fixed place. It runs on the node with the best upload link, because broadcasting to N nodes costs N uploads from wherever it lives.

## How I use it from my machine

There is no SSH or remote desktop involved. My machine runs the coordinator; every worker's agent keeps an encrypted QUIC connection to it. I drive everything through three interfaces that all talk to the local coordinator:

| Interface | For | Delivered in |
|---|---|---|
| `compute` CLI | Submitting and managing jobs from the terminal | M0–M4 |
| Python SDK (`compute_sdk`) | Use inside training scripts and from notebooks | M2–M7 |
| HTTP API (gateway) | OpenAI-compatible inference endpoint on `localhost`, usable by any existing client | M3 |

### Adding a machine (once per friend)

```bash
compute invite --name friend-a
```

This prints a one-time join ticket. The friend runs the installer and pastes the ticket; their agent registers its key with my coordinator and appears in `compute nodes`.

### Getting data and models onto the nodes

```bash
compute data push ./my_dataset
```

The dataset is hashed, uploaded once to each node that needs it, and cached. Jobs refer to it by name; nothing is streamed during training.

### Mode 1: train on one remote GPU

My training script stays ordinary PyTorch, with three SDK calls so that it can be resumed:

```python
import compute_sdk as cs

ctx = cs.init()                         # no-op defaults when run locally
model, opt = build()
start = ctx.restore(model, opt)         # loads the last checkpoint if one exists

for step in range(start, total_steps):
    loss = train_step(model, opt, next(batches))
    ctx.log(step=step, loss=loss.item())
    if step % 500 == 0:
        ctx.checkpoint(model, opt, step)  # safetensors, synced back to my machine
```

Run it remotely:

```bash
compute run --node friend-a --data my_dataset train.py -- --epochs 10
```

```bash
compute logs -f <job-id>
```

```bash
compute pull <job-id> ./results
```

`--node auto` picks the best free GPU with enough VRAM. The job continues if I close my laptop. The same script runs unchanged on my own machine with `python train.py`.

The same thing from a notebook:

```python
cluster = cs.connect()
job = cluster.submit("train.py", node="friend-a", data="my_dataset")
job.stream_logs()
job.wait()
job.download("./results")
```

For interactive work, `compute forward --node friend-a 8888` tunnels a port (Jupyter, TensorBoard) from the worker to my `localhost` over the same connection.

### Mode 2: many runs across the pool

```bash
compute sweep sweep.toml
```

`sweep.toml` lists the script and the parameter grid. Each trial goes to a free GPU; results and checkpoints are collected on my machine.

### Mode 2: one model trained on pooled GPUs

The script uses a pooled trainer in place of its own loop. Every node runs the same script; the SDK assigns data shards, pushes and pulls updates, and applies the dynamic division from M8.

```python
import compute_sdk as cs

ctx = cs.init()
model = build_lora_model()

trainer = cs.PooledTrainer(
    model,
    make_optimizer=lambda params: torch.optim.AdamW(params, lr=2e-4),
    dataset="my_dataset",
    loss_fn=loss_fn,
    strategy="lora_avg",     # or "local_sgd", "split"
    sync_every="auto",       # chosen from measured link speed
)
trainer.fit(steps=20_000)
```

```bash
compute run --pool all --data my_dataset train_pooled.py
```

`--pool` accepts `all`, a list of nodes, or a minimum VRAM. My own GPU joins the pool like any other node.

Strategies and what they sync:

| Strategy | What crosses the network | Use when |
|---|---|---|
| `lora_avg` | Adapter deltas (a few MB) every K steps | Fine-tuning; the realistic default |
| `local_sgd` | Compressed full-model deltas every K steps | Small models, good upload links |
| `split` | Activations and gradients at layer cuts, every step | The model fits on no single GPU |

### Inference on remote or pooled GPUs

```bash
compute serve --node friend-a ./my_model
```

```bash
compute serve --pool all --replicas 3 ./my_model
```

Either command makes the model available at `http://localhost:8080/v1` on my machine, so existing OpenAI-style client code works by changing the base URL. From Python, `cs.RemoteModel("my_model").generate(prompt, stream=True)` does the same without HTTP.

## Repository layout

```
Compute/
  crates/
    proto/        message types, protocol version, tensor frame format
    net/          iroh/QUIC connection, streams, auth, bandwidth probe
    sim/          in-process simulated network (latency, bandwidth, loss, partitions)
    agent/        worker daemon
    coordinator/  registry, scheduler, queue
    paramserver/  aggregation role for pooled training
    cli/          `compute` command
    gateway/      OpenAI-compatible HTTP front end (axum)
  python/
    compute_sdk/  PyO3 bindings + PyTorch helpers
  examples/
```

## Tensor frame format

Header: dtype, shape, codec, payload length, checksum. Payload: raw bytes.

Codecs, chosen per stream: `raw`, `fp16` / `int8` (quantized), `topk` (sparse), with optional `zstd` on top.

## Fault tolerance

Rules that apply everywhere:

- **Every job has an ID and a lease with a generation number.** If a node is declared lost and its job is re-queued, anything the old node sends later carries a stale generation and is rejected. This prevents duplicate or zombie results.
- **Every call has a timeout, every queue is bounded, every retry has backoff with jitter.**
- **Failure detection is generous.** A node that misses heartbeats is "unreachable" first and "lost" only when its lease expires, so a short Wi-Fi drop does not kill a job.
- **Jobs are resumable by contract.** The SDK gives each job a checkpoint directory and a resume hook; a job that does not checkpoint is restarted from zero.

| Failure | Detected by | Response |
|---|---|---|
| Worker process crashes | Agent sees the exit code | Retry from last checkpoint, up to N times with backoff. Out-of-memory is reported, not blindly retried; in pooled jobs the batch size is reduced |
| Agent crashes or PC reboots | Heartbeat stops | OS service restarts the agent; it reads its journal and resumes or reports each job |
| Owner pauses, PC sleeps, or a game starts | Agent (manual switch, or another process using the GPU) | Worker gets a grace period to checkpoint, then is stopped; the job is re-queued elsewhere |
| Network drop between worker and coordinator | Missed heartbeats | Job keeps running; logs and results buffer on the worker's disk; reconnect with backoff |
| Coordinator crashes or my laptop sleeps | Agents lose the connection | Standalone jobs continue. Coordinator restarts from SQLite and reconciles with agents. Pooled rounds pause and resume |
| Corrupt or partial transfer | Hash mismatch | Re-fetch the bad chunks only |
| Disk full on a worker | Preflight check and per-node quota | Evict least-recently-used cache entries; refuse the job if still short |
| Parameter server node lost | Round timeout | Global state is checkpointed off-node every round; the role moves to another node and continues from the last round |
| Bad update (NaN or exploding norm) | Validation before aggregation | That node's contribution is dropped for the round; repeated offences bench the node |
| Straggler in a pooled round | Round deadline | Late contributions are dropped or applied with a staleness discount |
| Node holding a model block dies (split training) | Stream error | All blocks roll back to the last **consistent global snapshot** (every block saved at the same step, stored off-node), the model is re-partitioned over the remaining nodes, and training resumes |
| Node dies during split inference | Stream error | Re-partition and replay the prompt; the caller sees a delay, not an error |
| Protocol version mismatch | Handshake | Connection refused with a clear message; the agent self-updates |

What this does not cover: a worker that deliberately returns wrong results. The system assumes friends and family are honest and only guards against accidents.

## Security

- **Authorisation:** agents accept jobs only when signed by my key. Keys can be revoked and rotated from the coordinator.
- **Isolation on the friend's PC:** workers run as a dedicated low-privilege user (or in a container where available) with access only to the agent's own directory. This is required from M1, before any second machine joins.
- **Protecting my machine:** everything coming back is treated as untrusted. Weights are safetensors, artifacts are hash-checked, and nothing received is executed.
- **Secrets:** tokens (for example for model downloads) are never placed in job bundles; downloads happen on my machine or through short-lived scoped tokens.
- **Updates:** agent self-update accepts only binaries signed by my release key.
- **Confidentiality limit:** anyone with admin rights on a worker can read my code, data and weights. Do not send data there that the owner should not see.

## Good-neighbour controls on each worker

- Runs as an OS service that starts on boot; visible status and a one-click pause.
- Allowed hours, GPU power or temperature cap, upload rate limit, and monthly transfer budget, all set by the owner.
- Clean uninstall that removes caches.

## Testing

- **`sim` crate:** the whole system runs in one process over a simulated network with configurable latency, bandwidth, loss and partitions. Every row in the fault table has a test here.
- **Local cluster:** several agents on localhost with artificial delay, for end-to-end runs with real PyTorch on CPU.
- **Fault injection on real machines:** kill the agent, pull the network, sleep the PC and fill the disk during each milestone's acceptance test.
- A milestone is done only when its fault tests pass, not when the happy path works.

## Milestones

### M0 – Two nodes talking
- Cargo workspace; `proto` (with version handshake), `net` and `sim` crates.
- Two machines connect, authenticate by key, exchange ping.
- Probe that measures RTT and throughput between any two nodes, and reports whether the path is direct or relayed.

**Done when:** `compute probe <node>` works across the internet, and the same code passes under simulated loss and partition.

### M1 – Agent and registry
- Agent installs as a service on Linux and Windows, runs workers as a restricted user, and self-updates.
- GPU, VRAM, utilisation and disk reporting; heartbeats; leases.
- Manual pause and automatic pause when another process uses the GPU.
- Coordinator state in SQLite; restart and reconcile.
- Metrics per node and per link are recorded from here on (needed by M8).

**Done when:** `compute nodes` shows live stats, and killing or restarting either side recovers without manual steps.

### M2 – Mode 1 training
- `compute run --node <n> train.py`: content-hashed code bundle, Python env from a lockfile (`uv`, with the right PyTorch build per platform), run.
- Jobs are detached: they continue if my machine disconnects.
- Live and buffered log streaming, cancel, exit status.
- Checkpoint sync back (safetensors, resumable); dataset cache keyed by hash.

**Done when:** a real training run finishes on a friend's GPU after a network drop, an agent restart and a coordinator restart during the run.

### M3 – Mode 1 inference
- Long-lived model server worker; `RemoteModel(...).generate(...)` with token streaming.
- `gateway` exposes an OpenAI-compatible HTTP endpoint.

**Done when:** streaming chat works against a friend's GPU, and a server crash is restarted automatically.

### M4 – Mode 2 task farm
- Queue with VRAM-aware placement, retries, re-queue on lost lease.
- Sweep API (one trial per GPU); sharded batch inference with exactly-once output per shard.

**Done when:** a 20-trial sweep finishes correctly after one node is unplugged mid-run, with no duplicated or missing results.

### M5 – Mode 2 inference routing
- Replicas across nodes; least-loaded routing, health checks, failover, request retry.

**Done when:** throughput with N nodes is close to the sum of the nodes, and killing a replica mid-load drops no requests.

### M6 – Split inference
- Layers partitioned across nodes; activations travel peer to peer as `fp16` frames.
- KV cache stays on each node so only the new token's activations cross the network.

**Done when:** a model too large for any single GPU generates text, with measured tokens/s, and survives the loss of one node.

### M7 – Pooled training (experimental)
- Parameter-server role with per-round checkpoints and update validation.
- LoRA adapters first: each node trains on its shard and pushes adapter deltas every K steps.
  - Averaging the two LoRA matrices separately is not the same as averaging their product; start with one matrix frozen, which makes averaging exact.
  - Data is shuffled globally before sharding so each node sees a similar distribution.
  - Optimizer state stays local to each node.
- Then a full small model with `topk` + `int8` compression and local SGD.
- Bounded-staleness updates so a slow node does not stall the round.

**Done when:** a benchmark compares pooled training with the best single GPU, in wall-clock time to the same eval loss. If it is not faster, Mode 1 plus M4 remains the training path.

### M8 – Dynamic runtime partitioning (experimental)

The system measures every node and link every few seconds and re-divides the training work while the job runs.

**Measured (per node, every 2–5 s, smoothed):** forward and backward time per layer block, free VRAM, GPU utilisation, per-link RTT and throughput from real traffic, samples/s. Only durations are compared across nodes, never wall-clock timestamps.

**Data division (do first).** Every node holds the whole model or adapter and the work per node is resized:
- batch size and local steps per round proportional to measured samples/s
- sync interval raised on slow links, lowered on fast ones
- codec chosen per link from measured bandwidth
- a slowed node gets less work within seconds; a vanished node is dropped from the round

**Layer division (split training).** The model is cut into consecutive blocks on different nodes; activations go forward and gradients come back across each cut.
- Cut points minimise estimated step time: compute time per block on each node + activation bytes at each cut ÷ link throughput, subject to VRAM.
- Micro-batches are pipelined so that all nodes work at once.

**Repartitioning rules**
- Moving layers ships their weights and optimizer state (often hundreds of MB), so a move happens only when the predicted saving over the next N minutes exceeds the transfer cost, with hysteresis.
- A move is a two-phase change at a step boundary: copy the block to the new node, verify its hash, switch routing, then delete the old copy. If anything fails before the switch, the old layout continues.
- Changes to data division need no migration and can apply every round.

**Limits to expect**
- Layer division pools VRAM, so it trains models that fit on no single GPU. It is rarely faster than one GPU that can hold the model (batch 8 × 512 tokens × 768 hidden in `fp16` is about 6 MB each way per step).
- Data division is where the speedup comes from, and it has the same bandwidth limit as M7.

**Done when:** a training job rebalances by itself after one node is slowed and one link is throttled, recovers after a node is killed mid-migration, and the logs show each decision and its measured gain.

## Order and effort

Phase 0 first. M0–M3 give a usable Mode 1 and the foundation. M4–M5 are the reliable Mode 2 wins. M6–M8 are research and run only if Phase 0 allows.

Rough solo effort, now including the fault-tolerance and security work: M0–M1 about 3–4 weeks, M2–M3 about 3–4 weeks, M4–M5 about 3 weeks, M6–M8 open-ended.

## Remaining risks

- **NAT traversal:** some routers force relayed connections, which are slow. The M0 probe shows this.
- **Windows:** service install, restricted users and process control differ from Linux; test on real Windows hardware in M1.
- **Mixed hardware:** an Apple Silicon Mac and NVIDIA cards cannot sensibly share a synchronized training job; the Mac is coordinator and inference client.
- **Research milestones may not pay off:** M7 and M8 can work correctly and still lose to one good GPU. The benchmarks decide.
- **Scope:** this is months of work. If the aim is only faster training, renting a cloud GPU by the hour is cheaper than the time spent; this plan makes sense when owning and learning the stack is part of the goal.

## Open questions

1. GPU, VRAM and OS of each machine, including mine.
2. Upload speed and data cap at each location.
3. Which models and sizes I train and serve.
4. Is Rust acceptable, or do I prefer C++ or Go?
