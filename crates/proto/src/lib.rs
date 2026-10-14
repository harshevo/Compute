pub mod admin;
pub mod id;
pub mod message;
pub mod ticket;

pub use id::{Generation, NodeId};
pub use ticket::{JoinTicket, TicketSecret};

pub const PROTOCOL_VERSION: u32 = 2;
