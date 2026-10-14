//! Messages between the `compute` CLI and the local coordinator.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::NodeId;
use crate::message::NodeStatus;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdminRequest {
    ListNodes,
    Invite {
        name: String,
    },
    Probe {
        node: String,
    },
    Worker {
        node: String,
        operation: String,
        payload: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdminReply {
    Nodes(Vec<NodeSummary>),
    Invite { ticket: String },
    Probe(ProbeReport),
    Failed(String),
    Json(String),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeSummary {
    pub id: NodeId,
    pub name: String,
    pub liveness: Liveness,
    pub status: NodeStatus,
    pub last_seen: Duration,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Liveness {
    Online,
    /// Heartbeats are overdue but the node's lease has not expired.
    Unreachable,
    Lost,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProbeReport {
    pub rtt: Duration,
    pub upload_bps: u64,
    pub download_bps: u64,
    pub path: PathKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PathKind {
    Direct,
    Relayed,
    Unknown,
}
