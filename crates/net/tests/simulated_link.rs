use std::sync::Arc;
use std::time::Duration;

use compute_net::probe::{self, ProbeConfig};
use compute_net::stream::accept_stream;
use compute_net::{Connection, Endpoint, NetError};
use compute_proto::message::StreamKind;
use compute_sim::{LinkProfile, SimEndpoint, SimNetwork};
use tokio::time::{Instant, timeout};

const HOME_LINK: LinkProfile = LinkProfile {
    latency: Duration::from_millis(20),
    bandwidth_bps: 20_000_000,
};

struct Pair {
    network: SimNetwork,
    dialer: SimEndpoint,
    listener: SimEndpoint,
}

impl Pair {
    fn new(profile: LinkProfile) -> Self {
        let network = SimNetwork::new(profile);
        Self {
            dialer: network.endpoint(),
            listener: network.endpoint(),
            network,
        }
    }

    async fn connect(&self) -> (Arc<dyn Connection>, Arc<dyn Connection>) {
        let outbound = self.dialer.connect(self.listener.id()).await.unwrap();
        let inbound = self.listener.accept().await.unwrap();
        (outbound, inbound)
    }

    fn partition(&self) {
        self.network.partition(self.dialer.id(), self.listener.id());
    }

    fn heal(&self) {
        self.network.heal(self.dialer.id(), self.listener.id());
    }

    fn sever(&self) {
        self.network.sever(self.dialer.id(), self.listener.id());
    }
}

fn spawn_probe_responder(connection: Arc<dyn Connection>) {
    tokio::spawn(async move {
        while let Ok((StreamKind::Probe, stream)) = accept_stream(&*connection).await {
            tokio::spawn(probe::respond(stream));
        }
    });
}

fn assert_within(actual: u64, expected: u64, tolerance_percent: u64) {
    let allowed = expected * tolerance_percent / 100;
    assert!(
        actual.abs_diff(expected) <= allowed,
        "{actual} is not within {tolerance_percent}% of {expected}"
    );
}

#[tokio::test(start_paused = true)]
async fn connections_identify_both_peers() {
    let pair = Pair::new(HOME_LINK);
    let (outbound, inbound) = pair.connect().await;

    assert_eq!(outbound.peer(), pair.listener.id());
    assert_eq!(inbound.peer(), pair.dialer.id());
}

#[tokio::test(start_paused = true)]
async fn connecting_to_an_absent_peer_fails() {
    let pair = Pair::new(HOME_LINK);
    let absent = pair.listener.id();
    drop(pair.listener);

    let result = pair.dialer.connect(absent).await;

    assert!(matches!(result, Err(NetError::Connect(_))));
}

#[tokio::test(start_paused = true)]
async fn a_message_arrives_after_the_link_latency() {
    let pair = Pair::new(HOME_LINK);
    let (outbound, inbound) = pair.connect().await;
    let mut sending = outbound.open().await.unwrap();
    let mut receiving = inbound.accept().await.unwrap();

    let sent_at = Instant::now();
    sending.send(&7u8).await.unwrap();
    let received: u8 = receiving.receive().await.unwrap();

    assert_eq!(received, 7);
    assert_within(sent_at.elapsed().as_millis() as u64, 20, 10);
}

#[tokio::test(start_paused = true)]
async fn probe_reports_the_configured_latency_and_bandwidth() {
    let pair = Pair::new(HOME_LINK);
    let (outbound, inbound) = pair.connect().await;
    spawn_probe_responder(inbound);

    let report = probe::measure(&*outbound, ProbeConfig::default())
        .await
        .unwrap();

    assert_within(report.rtt.as_millis() as u64, 40, 10);
    assert_within(report.upload_bps, 20_000_000, 5);
    assert_within(report.download_bps, 20_000_000, 5);
}

#[tokio::test(start_paused = true)]
async fn probe_sees_a_link_that_got_slower() {
    let pair = Pair::new(HOME_LINK);
    let (outbound, inbound) = pair.connect().await;
    spawn_probe_responder(inbound);
    let throttled = LinkProfile {
        latency: Duration::from_millis(80),
        bandwidth_bps: 2_000_000,
    };
    pair.network
        .set_link(pair.dialer.id(), pair.listener.id(), throttled);

    let report = probe::measure(&*outbound, ProbeConfig::default())
        .await
        .unwrap();

    assert_within(report.rtt.as_millis() as u64, 160, 10);
    assert_within(report.upload_bps, 2_000_000, 5);
}

#[tokio::test(start_paused = true)]
async fn a_partition_holds_traffic_until_healed() {
    let pair = Pair::new(HOME_LINK);
    let (outbound, inbound) = pair.connect().await;
    let mut sending = outbound.open().await.unwrap();
    let mut receiving = inbound.accept().await.unwrap();

    pair.partition();
    sending.send(&1u8).await.unwrap();
    let during = timeout(Duration::from_secs(30), receiving.receive::<u8>()).await;
    assert!(during.is_err(), "nothing should arrive during a partition");

    pair.heal();
    let after = timeout(Duration::from_secs(1), receiving.receive::<u8>()).await;
    assert_eq!(after.unwrap().unwrap(), 1);
}

#[tokio::test(start_paused = true)]
async fn connecting_stalls_during_a_partition() {
    let pair = Pair::new(HOME_LINK);
    pair.partition();

    let attempt = timeout(
        Duration::from_secs(10),
        pair.dialer.connect(pair.listener.id()),
    )
    .await;

    assert!(attempt.is_err());
}

#[tokio::test(start_paused = true)]
async fn severing_breaks_open_streams_and_the_connection() {
    let pair = Pair::new(HOME_LINK);
    let (outbound, inbound) = pair.connect().await;
    let _sending = outbound.open().await.unwrap();
    let mut receiving = inbound.accept().await.unwrap();

    pair.sever();

    assert!(receiving.receive::<u8>().await.is_err());
    assert!(matches!(outbound.open().await, Err(NetError::Lost(_))));
    assert!(matches!(inbound.accept().await, Err(NetError::Lost(_))));
}

#[tokio::test(start_paused = true)]
async fn a_new_connection_works_after_severing() {
    let pair = Pair::new(HOME_LINK);
    let _old = pair.connect().await;
    pair.sever();

    let (outbound, inbound) = pair.connect().await;
    let mut sending = outbound.open().await.unwrap();
    let mut receiving = inbound.accept().await.unwrap();
    sending.send(&5u8).await.unwrap();

    assert_eq!(receiving.receive::<u8>().await.unwrap(), 5);
}

#[tokio::test(start_paused = true)]
async fn a_restarted_node_keeps_listening_after_its_old_endpoint_is_dropped() {
    let pair = Pair::new(HOME_LINK);
    let id = pair.listener.id();
    let restarted = pair.network.endpoint_with_id(id);
    drop(pair.listener);

    let outbound = pair.dialer.connect(id).await.unwrap();
    let inbound = restarted.accept().await.unwrap();

    assert_eq!(outbound.peer(), id);
    assert_eq!(inbound.peer(), pair.dialer.id());
}

#[tokio::test(start_paused = true)]
async fn a_probe_payload_above_the_limit_is_refused() {
    let pair = Pair::new(HOME_LINK);
    let (outbound, inbound) = pair.connect().await;
    spawn_probe_responder(inbound);
    let oversized = ProbeConfig {
        pings: 1,
        payload_bytes: probe::MAX_PAYLOAD_BYTES + 1,
    };

    let result = timeout(
        Duration::from_secs(5),
        probe::measure(&*outbound, oversized),
    )
    .await;

    assert!(!matches!(result, Ok(Ok(_))));
}
