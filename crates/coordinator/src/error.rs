use compute_net::NetError;

#[derive(Debug, thiserror::Error)]
pub enum CoordinatorError {
    #[error(transparent)]
    Net(#[from] NetError),
    #[error("membership store: {0}")]
    Store(#[from] rusqlite::Error),
    #[error("no node matches '{0}'")]
    UnknownNode(String),
    #[error("node '{0}' is not connected")]
    NotConnected(String),
}
