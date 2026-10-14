#![allow(dead_code)]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use compute_agent::{Agent, AgentConfig, BackoffPolicy, StatusSource};
use compute_coordinator::{Coordinator, CoordinatorConfig, LivenessPolicy, Members};
use compute_net::Endpoint;
use compute_net::probe::ProbeConfig;
use compute_proto::admin::{Liveness, NodeSummary};
use compute_proto::message::{GpuInfo, NodeStatus, Rejection};
use compute_proto::{NodeId, TicketSecret};
use compute_sim::{LinkProfile, SimNetwork};
use tempfile::TempDir;
use tokio::task::JoinHandle;
use tokio::time::sleep;

pub const HOME_LINK: LinkProfile = LinkProfile {
    latency: Duration::from_millis(20),
    bandwidth_bps: 20_000_000,
};

pub const UNREACHABLE_AFTER: Duration = Duration::from_secs(15);
pub const LOST_AFTER: Duration = Duration::from_secs(60);

pub fn coordinator_config() -> CoordinatorConfig {
    CoordinatorConfig {
        liveness: LivenessPolicy {
            unreachable_after: UNREACHABLE_AFTER,
            lost_after: LOST_AFTER,
        },
        probe: ProbeConfig {
            pings: 3,
            payload_bytes: 256 << 10,
        },
        ..CoordinatorConfig::default()
    }
}

pub fn agent_config() -> AgentConfig {
    AgentConfig {
        heartbeat_interval: Duration::from_secs(5),
        reply_timeout: Duration::from_secs(10),
        connect_timeout: Duration::from_secs(10),
        backoff: BackoffPolicy {
            initial: Duration::from_secs(1),
            max: Duration::from_secs(8),
        },
    }
}

/// A coordinator and its agents on a simulated network.
pub struct Cluster {
    pub network: SimNetwork,
    pub coordinator: Arc<Coordinator>,
    pub coordinator_id: NodeId,
    state_dir: TempDir,
    serving: JoinHandle<()>,
}

impl Cluster {
    pub fn start() -> Self {
        let network = SimNetwork::new(HOME_LINK);
        let state_dir = tempfile::tempdir().unwrap();
        let coordinator_id = network.endpoint().id();
        let (coordinator, serving) = launch(&network, coordinator_id, &state_dir);
        Self {
            network,
            coordinator,
            coordinator_id,
            state_dir,
            serving,
        }
    }

    pub fn stop_coordinator(&mut self) {
        self.serving.abort();
    }

    pub fn restart_coordinator(&mut self) {
        self.stop_coordinator();
        let (coordinator, serving) = launch(&self.network, self.coordinator_id, &self.state_dir);
        self.coordinator = coordinator;
        self.serving = serving;
    }

    pub fn invite(&self, name: &str) -> TicketSecret {
        self.coordinator.invite(name).unwrap().secret
    }

    /// Starts an agent on a new machine.
    pub fn spawn_agent(&self, ticket: Option<TicketSecret>) -> AgentHandle {
        let id = self.network.endpoint().id();
        self.spawn_agent_as(id, ticket)
    }

    /// Starts an agent under an existing identity, as after a reboot.
    pub fn spawn_agent_as(&self, id: NodeId, ticket: Option<TicketSecret>) -> AgentHandle {
        let status = Arc::new(FakeStatus::default());
        let agent = Agent {
            endpoint: Arc::new(self.network.endpoint_with_id(id)),
            coordinator: self.coordinator_id,
            ticket,
            status: status.clone(),
            config: agent_config(),
        };
        AgentHandle {
            id,
            status,
            running: tokio::spawn(async move { agent.run().await }),
        }
    }

    pub async fn join(&self, name: &str) -> AgentHandle {
        let agent = self.spawn_agent(Some(self.invite(name)));
        self.wait_for(name, Liveness::Online).await;
        agent
    }

    pub fn node(&self, name: &str) -> Option<NodeSummary> {
        self.coordinator
            .nodes()
            .into_iter()
            .find(|node| node.name == name)
    }

    pub fn liveness(&self, name: &str) -> Option<Liveness> {
        self.node(name).map(|node| node.liveness)
    }

    pub async fn wait_for(&self, name: &str, expected: Liveness) {
        let reached = || self.liveness(name) == Some(expected);
        assert!(
            eventually(Duration::from_secs(180), reached).await,
            "{name} never became {expected:?}; it is {:?}",
            self.liveness(name)
        );
    }

    pub fn partition(&self, agent: &AgentHandle) {
        self.network.partition(self.coordinator_id, agent.id);
    }

    pub fn heal(&self, agent: &AgentHandle) {
        self.network.heal(self.coordinator_id, agent.id);
    }

    pub fn sever(&self, agent: &AgentHandle) {
        self.network.sever(self.coordinator_id, agent.id);
    }
}

impl Drop for Cluster {
    fn drop(&mut self) {
        self.serving.abort();
    }
}

fn launch(
    network: &SimNetwork,
    id: NodeId,
    state_dir: &TempDir,
) -> (Arc<Coordinator>, JoinHandle<()>) {
    let members = Members::open(&state_dir.path().join("state.db")).unwrap();
    let endpoint = Arc::new(network.endpoint_with_id(id));
    let coordinator = Coordinator::new(endpoint, members, coordinator_config());
    let serving = tokio::spawn({
        let coordinator = coordinator.clone();
        async move {
            let _ = coordinator.run().await;
        }
    });
    (coordinator, serving)
}

pub struct AgentHandle {
    pub id: NodeId,
    pub status: Arc<FakeStatus>,
    running: JoinHandle<Rejection>,
}

impl AgentHandle {
    pub fn crash(&self) {
        self.running.abort();
    }

    /// Resolves if the coordinator turns this agent away for good.
    pub async fn rejection(mut self) -> Rejection {
        (&mut self.running).await.unwrap()
    }
}

impl Drop for AgentHandle {
    fn drop(&mut self) {
        self.running.abort();
    }
}

#[derive(Default)]
pub struct FakeStatus {
    paused: AtomicBool,
}

impl FakeStatus {
    pub fn set_paused(&self, paused: bool) {
        self.paused.store(paused, Ordering::Relaxed);
    }

    pub fn gpu() -> GpuInfo {
        GpuInfo {
            name: "Simulated RTX".into(),
            vram_total_mib: 24_000,
            vram_free_mib: 20_000,
            utilization_percent: 10,
        }
    }
}

#[async_trait]
impl StatusSource for FakeStatus {
    async fn status(&self) -> NodeStatus {
        NodeStatus {
            gpus: vec![Self::gpu()],
            disk_free_bytes: 500 << 30,
            paused: self.paused.load(Ordering::Relaxed),
        }
    }
}

pub async fn eventually(limit: Duration, condition: impl Fn() -> bool) -> bool {
    let step = Duration::from_millis(100);
    let mut waited = Duration::ZERO;
    while waited < limit {
        if condition() {
            return true;
        }
        sleep(step).await;
        waited += step;
    }
    condition()
}
