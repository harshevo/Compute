use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use compute_net::handshake::receive_greeting;
use compute_net::probe::{self, ProbeConfig};
use compute_net::{BiStream, Connection, Endpoint, NetError, within};
use compute_proto::admin::{NodeSummary, ProbeReport};
use compute_proto::message::{Heartbeat, HeartbeatReply, Rejection};
use compute_proto::{Generation, JoinTicket, NodeId};
use tokio::task::JoinSet;
use tokio::time::{Instant, MissedTickBehavior, interval};
use tracing::{debug, info};

use crate::{CoordinatorError, LivenessPolicy, Members, Registry};

#[derive(Clone, Copy, Debug)]
pub struct CoordinatorConfig {
    pub liveness: LivenessPolicy,
    pub sweep_interval: Duration,
    pub handshake_timeout: Duration,
    pub probe: ProbeConfig,
    pub probe_timeout: Duration,
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        Self {
            liveness: LivenessPolicy::default(),
            sweep_interval: Duration::from_secs(1),
            handshake_timeout: Duration::from_secs(10),
            probe: ProbeConfig::default(),
            probe_timeout: Duration::from_secs(60),
        }
    }
}

struct Session {
    generation: Generation,
    connection: Arc<dyn Connection>,
}

pub struct Coordinator {
    endpoint: Arc<dyn Endpoint>,
    members: Members,
    config: CoordinatorConfig,
    registry: Mutex<Registry>,
    sessions: Mutex<HashMap<NodeId, Session>>,
}

impl Coordinator {
    pub fn new(
        endpoint: Arc<dyn Endpoint>,
        members: Members,
        config: CoordinatorConfig,
    ) -> Arc<Self> {
        Arc::new(Self {
            endpoint,
            members,
            config,
            registry: Mutex::new(Registry::new(config.liveness)),
            sessions: Mutex::default(),
        })
    }

    /// Serves agents until the endpoint fails. Dropping the future stops all serving.
    pub async fn run(self: Arc<Self>) -> Result<(), CoordinatorError> {
        let mut tasks = JoinSet::new();
        tasks.spawn(self.clone().sweep_forever());
        loop {
            let connection = self.endpoint.accept().await?;
            while tasks.try_join_next().is_some() {}
            tasks.spawn(self.clone().serve(connection));
        }
    }

    pub fn invite(&self, name: &str) -> Result<JoinTicket, CoordinatorError> {
        Ok(JoinTicket {
            coordinator: self.endpoint.id(),
            secret: self.members.invite(name)?,
        })
    }

    pub fn nodes(&self) -> Vec<NodeSummary> {
        self.registry.lock().unwrap().summaries(Instant::now())
    }

    pub async fn probe(&self, node: &str) -> Result<ProbeReport, CoordinatorError> {
        let id = self
            .registry
            .lock()
            .unwrap()
            .resolve(node)
            .ok_or_else(|| CoordinatorError::UnknownNode(node.to_owned()))?;
        let connection = self
            .sessions
            .lock()
            .unwrap()
            .get(&id)
            .map(|session| session.connection.clone())
            .ok_or_else(|| CoordinatorError::NotConnected(node.to_owned()))?;
        let measurement = probe::measure(&*connection, self.config.probe);
        Ok(within(self.config.probe_timeout, measurement).await?)
    }

    pub async fn worker(
        &self,
        node: &str,
        operation: String,
        payload: String,
    ) -> Result<String, CoordinatorError> {
        let id = self
            .registry
            .lock()
            .unwrap()
            .resolve(node)
            .ok_or_else(|| CoordinatorError::UnknownNode(node.to_owned()))?;
        let connection = self
            .sessions
            .lock()
            .unwrap()
            .get(&id)
            .map(|session| session.connection.clone())
            .ok_or_else(|| CoordinatorError::NotConnected(node.to_owned()))?;
        let exchange = async {
            let mut stream = compute_net::stream::open_stream(
                &*connection,
                compute_proto::message::StreamKind::Worker,
            )
            .await?;
            stream
                .send(&compute_proto::message::WorkerRequest { operation, payload })
                .await?;
            let reply: compute_proto::message::WorkerReply = stream.receive().await?;
            Ok(reply.payload)
        };
        Ok(within(Duration::from_secs(150), exchange).await?)
    }

    async fn sweep_forever(self: Arc<Self>) {
        let mut ticks = interval(self.config.sweep_interval);
        ticks.set_missed_tick_behavior(MissedTickBehavior::Delay);
        loop {
            ticks.tick().await;
            let changes = self.registry.lock().unwrap().sweep(Instant::now());
            for (node, liveness) in changes {
                info!(?node, ?liveness, "node liveness changed");
            }
        }
    }

    async fn serve(self: Arc<Self>, connection: Arc<dyn Connection>) {
        let peer = connection.peer();
        if let Err(error) = self.serve_agent(connection).await {
            debug!(?peer, %error, "agent session ended");
        }
    }

    async fn serve_agent(&self, connection: Arc<dyn Connection>) -> Result<(), CoordinatorError> {
        let peer = connection.peer();
        let greeting = within(
            self.config.handshake_timeout,
            receive_greeting(&*connection),
        )
        .await?;
        let Some(name) = self.members.admit(peer, greeting.ticket)? else {
            return Ok(greeting.reject(Rejection::Unauthorized).await?);
        };

        let generation = self.open_session(peer, name, connection);
        let outcome = match greeting.accept(generation).await {
            Ok(control) => self.receive_heartbeats(peer, generation, control).await,
            Err(error) => Err(error),
        };
        self.close_session(peer, generation);
        Ok(outcome?)
    }

    fn open_session(
        &self,
        peer: NodeId,
        name: String,
        connection: Arc<dyn Connection>,
    ) -> Generation {
        let generation = self
            .registry
            .lock()
            .unwrap()
            .register(peer, name, Instant::now());
        let session = Session {
            generation,
            connection,
        };
        self.sessions.lock().unwrap().insert(peer, session);
        info!(?peer, ?generation, "agent connected");
        generation
    }

    fn close_session(&self, peer: NodeId, generation: Generation) {
        let mut sessions = self.sessions.lock().unwrap();
        if sessions
            .get(&peer)
            .is_some_and(|session| session.generation == generation)
        {
            sessions.remove(&peer);
        }
    }

    async fn receive_heartbeats(
        &self,
        peer: NodeId,
        generation: Generation,
        mut control: BiStream,
    ) -> Result<(), NetError> {
        loop {
            let heartbeat: Heartbeat = control.receive().await?;
            let accepted = heartbeat.session == generation
                && self
                    .registry
                    .lock()
                    .unwrap()
                    .heartbeat(peer, generation, heartbeat.status, Instant::now())
                    .is_ok();
            if !accepted {
                control.send(&HeartbeatReply::StaleSession).await?;
                return Ok(());
            }
            control.send(&HeartbeatReply::Ack).await?;
        }
    }
}
