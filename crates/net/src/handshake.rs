//! The first exchange on a connection. Its stream stays open as the agent's control stream.

use compute_proto::message::{Hello, Rejection, Welcome};
use compute_proto::{Generation, PROTOCOL_VERSION, TicketSecret};

use crate::{BiStream, Connection, NetError, Result};

pub async fn greet(
    connection: &dyn Connection,
    ticket: Option<TicketSecret>,
) -> Result<(Generation, BiStream)> {
    let mut stream = connection.open().await?;
    let hello = Hello {
        protocol: PROTOCOL_VERSION,
        ticket,
    };
    stream.send(&hello).await?;
    match stream.receive().await? {
        Welcome::Accepted { session } => Ok((session, stream)),
        Welcome::Rejected(rejection) => Err(NetError::Rejected(rejection)),
    }
}

/// A peer that speaks our protocol version and is waiting to be admitted or turned away.
pub struct Greeting {
    pub ticket: Option<TicketSecret>,
    stream: BiStream,
}

pub async fn receive_greeting(connection: &dyn Connection) -> Result<Greeting> {
    let mut stream = connection.accept().await?;
    let hello: Hello = stream.receive().await?;
    if hello.protocol != PROTOCOL_VERSION {
        let rejection = Rejection::ProtocolMismatch {
            supported: PROTOCOL_VERSION,
        };
        stream.send(&Welcome::Rejected(rejection)).await?;
        return Err(NetError::Rejected(rejection));
    }
    Ok(Greeting {
        ticket: hello.ticket,
        stream,
    })
}

impl Greeting {
    pub async fn accept(mut self, session: Generation) -> Result<BiStream> {
        self.stream.send(&Welcome::Accepted { session }).await?;
        Ok(self.stream)
    }

    pub async fn reject(mut self, rejection: Rejection) -> Result<()> {
        self.stream.send(&Welcome::Rejected(rejection)).await
    }
}
