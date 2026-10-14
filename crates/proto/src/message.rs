//! Messages exchanged between an agent and the coordinator.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::{Generation, TicketSecret};

/// First message on a new connection, sent by the agent.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Hello {
    pub protocol: u32,
    pub ticket: Option<TicketSecret>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Welcome {
    Accepted { session: Generation },
    Rejected(Rejection),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Rejection {
    ProtocolMismatch { supported: u32 },
    Unauthorized,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Heartbeat {
    pub session: Generation,
    pub status: NodeStatus,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum HeartbeatReply {
    Ack,
    StaleSession,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeStatus {
    pub gpus: Vec<GpuInfo>,
    pub disk_free_bytes: u64,
    pub paused: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuInfo {
    pub name: String,
    pub vram_total_mib: u64,
    pub vram_free_mib: u64,
    pub utilization_percent: u8,
}

/// First message on every stream the coordinator opens towards an agent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StreamKind {
    Probe,
    Worker,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProbeRequest {
    Ping,
    /// The requester sends this many raw bytes next; the responder reads them all.
    Sink {
        bytes: u64,
    },
    /// The responder sends this many raw bytes.
    Source {
        bytes: u64,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProbeReply {
    Pong,
    Sunk(Transfer),
}

/// What a receiver observed of a bulk transfer, timed from its first arrival so that
/// signalling delay is excluded.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Transfer {
    pub bytes_after_first_arrival: u64,
    pub elapsed: Duration,
}

impl Transfer {
    /// Zero when the transfer was too short to time.
    pub fn bits_per_second(&self) -> u64 {
        if self.elapsed.is_zero() {
            return 0;
        }
        (self.bytes_after_first_arrival as f64 * 8.0 / self.elapsed.as_secs_f64()) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn throughput_is_bits_over_time() {
        let transfer = Transfer {
            bytes_after_first_arrival: 1_000_000,
            elapsed: Duration::from_secs(1),
        };
        assert_eq!(transfer.bits_per_second(), 8_000_000);
    }

    #[test]
    fn an_untimed_transfer_reports_zero_instead_of_infinity() {
        assert_eq!(Transfer::default().bits_per_second(), 0);
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkerRequest {
    pub operation: String,
    pub payload: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkerReply {
    pub payload: String,
}
