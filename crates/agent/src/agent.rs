use std::sync::Arc;
use std::time::Duration;

use compute_net::handshake::greet;
use compute_net::stream::accept_stream;
use compute_net::{BiStream, Connection, Endpoint, NetError, Result, probe, within};
use compute_proto::message::{Heartbeat, HeartbeatReply, Rejection, StreamKind};
use compute_proto::{Generation, NodeId, TicketSecret};
use tokio::time::sleep;
use tracing::{debug, info, warn};

use crate::{Backoff, BackoffPolicy, StatusSource};

#[derive(Clone, Copy, Debug)]
pub struct AgentConfig {
    pub heartbeat_interval: Duration,
    pub reply_timeout: Duration,
    pub connect_timeout: Duration,
    pub backoff: BackoffPolicy,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            heartbeat_interval: Duration::from_secs(5),
            reply_timeout: Duration::from_secs(10),
            connect_timeout: Duration::from_secs(20),
            backoff: BackoffPolicy::default(),
        }
    }
}

pub struct Agent {
    pub endpoint: Arc<dyn Endpoint>,
    pub coordinator: NodeId,
    pub ticket: Option<TicketSecret>,
    pub status: Arc<dyn StatusSource>,
    pub config: AgentConfig,
}

impl Agent {
    /// Keeps a session with the coordinator alive, reconnecting after any failure.
    /// Returns only when the coordinator refuses this node, which retrying cannot fix.
    pub async fn run(&self) -> Rejection {
        let mut backoff = Backoff::new(self.config.backoff);
        loop {
            match self.session(&mut backoff).await {
                Err(NetError::Rejected(rejection)) => return rejection,
                Err(error) => debug!(%error, "session with coordinator ended"),
                Ok(()) => debug!("coordinator ended the session"),
            }
            sleep(backoff.next_delay()).await;
        }
    }

    async fn session(&self, backoff: &mut Backoff) -> Result<()> {
        let dialing = self.endpoint.connect(self.coordinator);
        let connection = within(self.config.connect_timeout, dialing).await?;
        let greeting = greet(&*connection, self.ticket);
        let (session, control) = within(self.config.reply_timeout, greeting).await?;
        info!(?session, "connected to coordinator");
        backoff.reset();

        tokio::select! {
            ended = self.send_heartbeats(session, control) => ended,
            ended = serve_streams(&*connection, self.status.clone()) => ended,
        }
    }

    async fn send_heartbeats(&self, session: Generation, mut control: BiStream) -> Result<()> {
        loop {
            let heartbeat = Heartbeat {
                session,
                status: self.status.status().await,
            };
            control.send(&heartbeat).await?;
            match within(self.config.reply_timeout, control.receive()).await? {
                HeartbeatReply::Ack => sleep(self.config.heartbeat_interval).await,
                HeartbeatReply::StaleSession => return Ok(()),
            }
        }
    }
}

async fn serve_streams(connection: &dyn Connection, status: Arc<dyn StatusSource>) -> Result<()> {
    let limit = Arc::new(tokio::sync::Semaphore::new(8));
    loop {
        let permit = limit
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| NetError::Closed)?;
        match accept_stream(connection).await? {
            (StreamKind::Worker, mut stream) => {
                let status = status.clone();
                tokio::spawn(async move {
                    let _permit = permit;
                    let result: Result<()> = async {
                        let request = within(Duration::from_secs(30), stream.receive()).await?;
                        let payload = status.worker(request).await;
                        stream
                            .send(&compute_proto::message::WorkerReply { payload })
                            .await
                    }
                    .await;
                    if let Err(error) = result {
                        warn!(%error, "worker request failed");
                    }
                });
            }
            (StreamKind::Probe, stream) => {
                tokio::spawn(async move {
                    let _permit = permit;
                    if let Err(error) =
                        within(Duration::from_secs(120), probe::respond(stream)).await
                    {
                        warn!(%error, "probe failed");
                    }
                });
            }
        }
    }
}
