use std::sync::Mutex;
use std::time::Duration;

use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinkProfile {
    /// One-way delay.
    pub latency: Duration,
    pub bandwidth_bps: u64,
}

impl LinkProfile {
    pub const LAN: Self = Self {
        latency: Duration::from_micros(200),
        bandwidth_bps: 1_000_000_000,
    };

    pub fn transmit_time(&self, bytes: usize) -> Duration {
        Duration::from_secs_f64(bytes as f64 * 8.0 / self.bandwidth_bps as f64)
    }
}

/// The path between one pair of endpoints, shared by every connection between them.
pub(crate) struct Link {
    profile: Mutex<LinkProfile>,
    up: watch::Sender<bool>,
    connections: Mutex<Vec<CancellationToken>>,
}

impl Link {
    pub fn new(profile: LinkProfile) -> Self {
        Self {
            profile: Mutex::new(profile),
            up: watch::Sender::new(true),
            connections: Mutex::new(Vec::new()),
        }
    }

    pub fn profile(&self) -> LinkProfile {
        *self.profile.lock().unwrap()
    }

    pub fn set_profile(&self, profile: LinkProfile) {
        *self.profile.lock().unwrap() = profile;
    }

    pub fn set_up(&self, up: bool) {
        self.up.send_replace(up);
    }

    /// Resolves once traffic can flow; data sent during a partition is held, not dropped.
    pub async fn passable(&self) {
        let mut up = self.up.subscribe();
        let _ = up.wait_for(|up| *up).await;
    }

    pub fn track_connection(&self) -> CancellationToken {
        let severed = CancellationToken::new();
        self.connections.lock().unwrap().push(severed.clone());
        severed
    }

    pub fn sever_connections(&self) {
        for severed in self.connections.lock().unwrap().drain(..) {
            severed.cancel();
        }
    }
}
