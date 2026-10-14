//! An in-process network with configurable latency, bandwidth, partitions and connection loss.
//!
//! All delays use tokio time, so tests that pause the clock run instantly and deterministically.

mod link;
mod network;
mod pipe;

pub use link::LinkProfile;
pub use network::{SimEndpoint, SimNetwork};
