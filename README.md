# Compute

Pool GPUs across machines you trust. Design and roadmap: [plan.md](plan.md).

## Status

Implemented: milestone M0 and the core of M1 — encrypted peer-to-peer connections, one-time
join tickets, live node registry with leases, owner pause switch, and a latency/bandwidth probe.

Not yet implemented: running jobs (M2 onward), the Python SDK, installing the agent as an OS
service, self-update, and running workers as a restricted user.

## Layout

| Crate | Responsibility |
|---|---|
| `crates/proto` | Message types, node ids, join tickets, protocol version |
| `crates/net` | Transport traits, message framing, handshake, probe, QUIC transport (`quic.rs` is the only file that uses iroh) |
| `crates/sim` | In-process network with latency, bandwidth, partitions and connection loss, for tests |
| `crates/coordinator` | Membership store (SQLite), liveness registry, agent sessions, local admin socket |
| `crates/agent` | Session loop with reconnect and backoff, GPU and disk status, pause switch |
| `crates/cli` | The `compute` binary |

The coordinator and agent depend only on the `Endpoint` and `Connection` traits, so the same
code runs over real QUIC and over the simulated network.

## Use

On my machine:

```bash
cargo run -p compute-cli -- serve
```

```bash
cargo run -p compute-cli -- invite friend-a
```

On the friend's machine, with the ticket that `invite` printed:

```bash
compute agent join <ticket>
```

```bash
compute agent run
```

Back on my machine:

```bash
cargo run -p compute-cli -- nodes
```

```bash
cargo run -p compute-cli -- probe friend-a
```

The friend can stop lending at any time with `compute agent pause` and `compute agent resume`.

State lives in `~/.compute` (override with `COMPUTE_HOME`).

Peers find each other by public key through iroh's public discovery and relay servers, so node
ids and network addresses are published there.

## Test

```bash
cargo test --workspace
```

The failure scenarios in `crates/coordinator/tests/cluster.rs` run on the simulated network with
a paused clock, so minutes of partitions and restarts take milliseconds and are deterministic.
`crates/net/tests/quic_loopback.rs` exercises the real QUIC transport on this machine.
