use std::sync::Arc;

use async_trait::async_trait;
use compute_proto::NodeId;
use compute_proto::admin::PathKind;

use crate::{BiStream, Result};

/// An authenticated, encrypted link to one peer, carrying any number of streams.
#[async_trait]
pub trait Connection: Send + Sync {
    fn peer(&self) -> NodeId;

    fn path(&self) -> PathKind;

    async fn open(&self) -> Result<BiStream>;

    async fn accept(&self) -> Result<BiStream>;
}

#[async_trait]
pub trait Endpoint: Send + Sync {
    fn id(&self) -> NodeId;

    async fn connect(&self, peer: NodeId) -> Result<Arc<dyn Connection>>;

    async fn accept(&self) -> Result<Arc<dyn Connection>>;
}
