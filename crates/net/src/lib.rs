pub mod error;
pub mod framing;
pub mod handshake;
pub mod probe;
pub mod quic;
pub mod stream;
pub mod transport;

pub use error::NetError;
pub use stream::BiStream;
pub use transport::{Connection, Endpoint};

pub type Result<T> = std::result::Result<T, NetError>;

/// Bounds how long a network operation may take.
pub async fn within<T>(
    limit: std::time::Duration,
    operation: impl Future<Output = Result<T>>,
) -> Result<T> {
    tokio::time::timeout(limit, operation)
        .await
        .unwrap_or(Err(NetError::TimedOut))
}
