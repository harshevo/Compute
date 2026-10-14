//! The real transport: QUIC between public keys, with hole punching and relay fallback.
//! This is the only module that knows about iroh.

use std::net::SocketAddr;
use std::sync::Arc;

use async_trait::async_trait;
use compute_proto::NodeId;
use compute_proto::admin::PathKind;
use iroh::address_lookup::MemoryLookup;
use iroh::endpoint::presets;
use iroh::{EndpointAddr, PublicKey, RelayMode, SecretKey};

use crate::{BiStream, Connection, Endpoint, NetError, Result};

const ALPN: &[u8] = b"compute/1";

pub type Secret = [u8; 32];

pub fn generate_secret() -> Secret {
    SecretKey::generate().to_bytes()
}

pub fn node_id(secret: &Secret) -> NodeId {
    to_node_id(SecretKey::from_bytes(secret).public())
}

pub struct QuicEndpoint {
    inner: iroh::Endpoint,
    known_peers: MemoryLookup,
}

impl QuicEndpoint {
    /// Reachable from anywhere: peers find each other by key through public discovery
    /// and fall back to relays when a direct path cannot be established.
    pub async fn bind(secret: &Secret) -> Result<Self> {
        Self::build(iroh::Endpoint::builder(presets::N0), secret).await
    }

    /// No discovery and no relays: only peers added with [`Self::add_peer`] can be dialled.
    pub async fn bind_isolated(secret: &Secret) -> Result<Self> {
        let builder = iroh::Endpoint::builder(presets::Minimal).relay_mode(RelayMode::Disabled);
        Self::build(builder, secret).await
    }

    async fn build(builder: iroh::endpoint::Builder, secret: &Secret) -> Result<Self> {
        let known_peers = MemoryLookup::new();
        let inner = builder
            .secret_key(SecretKey::from_bytes(secret))
            .alpns(vec![ALPN.to_vec()])
            .address_lookup(known_peers.clone())
            .bind()
            .await
            .map_err(|error| NetError::Connect(error.to_string()))?;
        Ok(Self { inner, known_peers })
    }

    pub fn add_peer(&self, peer: NodeId, address: SocketAddr) -> Result<()> {
        let address = EndpointAddr::new(to_public_key(peer)?).with_ip_addr(address);
        self.known_peers.add_endpoint_info(address);
        Ok(())
    }

    pub fn local_addresses(&self) -> Vec<SocketAddr> {
        self.inner.bound_sockets()
    }

    pub async fn close(&self) {
        self.inner.close().await;
    }
}

#[async_trait]
impl Endpoint for QuicEndpoint {
    fn id(&self) -> NodeId {
        to_node_id(self.inner.id())
    }

    async fn connect(&self, peer: NodeId) -> Result<Arc<dyn Connection>> {
        let connection = self
            .inner
            .connect(to_public_key(peer)?, ALPN)
            .await
            .map_err(|error| NetError::Connect(error.to_string()))?;
        Ok(Arc::new(QuicConnection(connection)))
    }

    async fn accept(&self) -> Result<Arc<dyn Connection>> {
        loop {
            let incoming = self.inner.accept().await;
            let incoming = incoming.ok_or_else(|| NetError::Lost("endpoint closed".into()))?;
            match incoming.await {
                Ok(connection) => return Ok(Arc::new(QuicConnection(connection))),
                Err(error) => tracing::debug!(%error, "incoming connection failed"),
            }
        }
    }
}

struct QuicConnection(iroh::endpoint::Connection);

#[async_trait]
impl Connection for QuicConnection {
    fn peer(&self) -> NodeId {
        to_node_id(self.0.remote_id())
    }

    fn path(&self) -> PathKind {
        let paths = self.0.paths();
        match paths.iter().find(|path| path.is_selected()) {
            Some(path) if path.is_relay() => PathKind::Relayed,
            Some(path) if path.is_ip() => PathKind::Direct,
            _ => PathKind::Unknown,
        }
    }

    async fn open(&self) -> Result<BiStream> {
        let (writer, reader) = self.0.open_bi().await.map_err(lost)?;
        Ok(BiStream::new(writer, reader))
    }

    async fn accept(&self) -> Result<BiStream> {
        let (writer, reader) = self.0.accept_bi().await.map_err(lost)?;
        Ok(BiStream::new(writer, reader))
    }
}

fn lost(error: impl std::fmt::Display) -> NetError {
    NetError::Lost(error.to_string())
}

fn to_node_id(key: PublicKey) -> NodeId {
    NodeId::from_bytes(*key.as_bytes())
}

fn to_public_key(id: NodeId) -> Result<PublicKey> {
    PublicKey::from_bytes(id.as_bytes())
        .map_err(|_| NetError::Connect(format!("{id} is not a valid public key")))
}
