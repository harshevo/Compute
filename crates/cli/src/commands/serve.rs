use std::sync::Arc;

use anyhow::{Context, Result};
use compute_coordinator::{Coordinator, CoordinatorConfig, Members, admin};
use compute_net::Endpoint;
use compute_net::quic::QuicEndpoint;

use crate::home::Home;
use crate::identity;

pub async fn run(home: &Home) -> Result<()> {
    let dir = home.coordinator()?;
    let lock_path = dir.state().with_extension("lock");
    let _lock = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(lock_path)?;
    _lock
        .try_lock()
        .context("another coordinator already owns COMPUTE_HOME")?;
    let secret = identity::load_or_create(&dir.identity())?;
    let endpoint = Arc::new(QuicEndpoint::bind(&secret).await?);
    let members = Members::open(&dir.state()).context("cannot open the membership store")?;
    let listener = admin::bind(&dir.admin_socket()).context("cannot open the admin socket")?;
    let coordinator = Coordinator::new(endpoint.clone(), members, CoordinatorConfig::default());

    println!("coordinator {} is running", endpoint.id());
    let outcome = tokio::select! {
        ended = coordinator.clone().run() => ended.map_err(Into::into),
        ended = admin::serve(coordinator, listener) => ended.map_err(Into::into),
        interrupted = tokio::signal::ctrl_c() => interrupted.map_err(Into::into),
    };
    endpoint.close().await;
    outcome
}
