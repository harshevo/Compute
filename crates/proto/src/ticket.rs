use std::fmt;
use std::str::FromStr;

use data_encoding::BASE32_NOPAD;
use serde::{Deserialize, Serialize};

use crate::NodeId;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TicketSecret([u8; 16]);

impl TicketSecret {
    pub const fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }
}

/// What a new machine needs to enrol: who to dial and a one-time secret to present.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JoinTicket {
    pub coordinator: NodeId,
    pub secret: TicketSecret,
}

impl fmt::Display for JoinTicket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let bytes = postcard::to_stdvec(self).map_err(|_| fmt::Error)?;
        f.write_str(&BASE32_NOPAD.encode(&bytes).to_lowercase())
    }
}

#[derive(Debug, PartialEq, Eq, thiserror::Error)]
#[error("not a valid join ticket")]
pub struct InvalidTicket;

impl FromStr for JoinTicket {
    type Err = InvalidTicket;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let bytes = BASE32_NOPAD
            .decode(s.trim().to_uppercase().as_bytes())
            .map_err(|_| InvalidTicket)?;
        postcard::from_bytes(&bytes).map_err(|_| InvalidTicket)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ticket_round_trips_through_text() {
        let ticket = JoinTicket {
            coordinator: NodeId::from_bytes([7; 32]),
            secret: TicketSecret::from_bytes([9; 16]),
        };
        assert_eq!(format!("  {ticket}\n").parse(), Ok(ticket));
    }

    #[test]
    fn garbage_is_not_a_ticket() {
        assert_eq!("hello".parse::<JoinTicket>(), Err(InvalidTicket));
        assert_eq!("".parse::<JoinTicket>(), Err(InvalidTicket));
    }
}
