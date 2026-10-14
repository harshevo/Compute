use std::fmt;
use std::str::FromStr;

use data_encoding::HEXLOWER;
use serde::{Deserialize, Serialize};

/// A node's Ed25519 public key.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct NodeId([u8; 32]);

impl NodeId {
    const SHORT_LEN: usize = 10;

    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn short(&self) -> String {
        self.to_string()[..Self::SHORT_LEN].to_owned()
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&HEXLOWER.encode(&self.0))
    }
}

impl fmt::Debug for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NodeId({})", self.short())
    }
}

#[derive(Debug, PartialEq, Eq, thiserror::Error)]
#[error("a node id is 64 hex characters")]
pub struct InvalidNodeId;

impl FromStr for NodeId {
    type Err = InvalidNodeId;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let bytes = HEXLOWER.decode(s.as_bytes()).map_err(|_| InvalidNodeId)?;
        bytes.try_into().map(Self).map_err(|_| InvalidNodeId)
    }
}

/// Fencing token: a message carrying an older generation than the current one is rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Generation(u64);

impl Generation {
    pub const FIRST: Self = Self(1);

    pub fn next(self) -> Self {
        Self(self.0 + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_id_round_trips_through_text() {
        let id = NodeId::from_bytes([0xab; 32]);
        assert_eq!(id.to_string().parse(), Ok(id));
    }

    #[test]
    fn node_id_rejects_wrong_length_and_non_hex() {
        assert_eq!("abcd".parse::<NodeId>(), Err(InvalidNodeId));
        assert_eq!("z".repeat(64).parse::<NodeId>(), Err(InvalidNodeId));
    }

    #[test]
    fn generations_increase() {
        assert!(Generation::FIRST.next() > Generation::FIRST);
    }
}
