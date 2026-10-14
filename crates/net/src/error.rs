use compute_proto::message::Rejection;

#[derive(Debug, thiserror::Error)]
pub enum NetError {
    #[error("could not connect: {0}")]
    Connect(String),
    #[error("connection lost: {0}")]
    Lost(String),
    #[error("timed out")]
    TimedOut,
    #[error("stream closed by peer")]
    Closed,
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("malformed message: {0}")]
    Decode(#[from] postcard::Error),
    #[error("message of {0} bytes exceeds the frame limit")]
    TooLarge(usize),
    #[error("rejected by peer: {0:?}")]
    Rejected(Rejection),
    #[error("protocol violation: {0}")]
    Protocol(&'static str),
}
