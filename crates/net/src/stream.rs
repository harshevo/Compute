use compute_proto::message::StreamKind;
use serde::Serialize;
use serde::de::DeserializeOwned;
use tokio::io::{AsyncRead, AsyncWrite};

use crate::{Connection, Result, framing};

pub type BoxWriter = Box<dyn AsyncWrite + Send + Unpin>;
pub type BoxReader = Box<dyn AsyncRead + Send + Unpin>;

pub struct BiStream {
    pub writer: BoxWriter,
    pub reader: BoxReader,
}

impl BiStream {
    pub fn new(
        writer: impl AsyncWrite + Send + Unpin + 'static,
        reader: impl AsyncRead + Send + Unpin + 'static,
    ) -> Self {
        Self {
            writer: Box::new(writer),
            reader: Box::new(reader),
        }
    }

    pub async fn send<T: Serialize>(&mut self, message: &T) -> Result<()> {
        framing::write_message(&mut self.writer, message).await
    }

    pub async fn receive<T: DeserializeOwned>(&mut self) -> Result<T> {
        framing::read_message(&mut self.reader).await
    }
}

pub async fn open_stream(connection: &dyn Connection, kind: StreamKind) -> Result<BiStream> {
    let mut stream = connection.open().await?;
    stream.send(&kind).await?;
    Ok(stream)
}

pub async fn accept_stream(connection: &dyn Connection) -> Result<(StreamKind, BiStream)> {
    let mut stream = connection.accept().await?;
    let kind = stream.receive().await?;
    Ok((kind, stream))
}
