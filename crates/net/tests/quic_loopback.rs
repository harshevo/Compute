//! The real QUIC transport, exercised between two endpoints on this machine.

use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use compute_net::probe::{self, ProbeConfig};
use compute_net::quic::{self, QuicEndpoint};
use compute_net::stream::accept_stream;
use compute_net::{Connection, Endpoint, within};
use compute_proto::admin::PathKind;

const LIMIT: Duration = Duration::from_secs(20);

async fn isolated_endpoint() -> QuicEndpoint {
    QuicEndpoint::bind_isolated(&quic::generate_secret())
        .await
        .unwrap()
}

fn loopback_address(endpoint: &QuicEndpoint) -> SocketAddr {
    let port = endpoint
        .local_addresses()
        .iter()
        .find(|address| address.is_ipv4())
        .expect("an IPv4 socket")
        .port();
    (Ipv4Addr::LOCALHOST, port).into()
}

async fn connected() -> (
    QuicEndpoint,
    QuicEndpoint,
    Arc<dyn Connection>,
    Arc<dyn Connection>,
) {
    let dialer = isolated_endpoint().await;
    let listener = isolated_endpoint().await;
    dialer
        .add_peer(listener.id(), loopback_address(&listener))
        .unwrap();
    let (outbound, inbound) = tokio::join!(
        within(LIMIT, dialer.connect(listener.id())),
        within(LIMIT, listener.accept()),
    );
    (dialer, listener, outbound.unwrap(), inbound.unwrap())
}

#[test]
fn a_secret_always_yields_the_same_node_id() {
    let secret = quic::generate_secret();

    assert_eq!(quic::node_id(&secret), quic::node_id(&secret));
    assert_ne!(
        quic::node_id(&secret),
        quic::node_id(&quic::generate_secret())
    );
}

#[tokio::test]
async fn an_endpoint_is_identified_by_its_secret() {
    let secret = quic::generate_secret();

    let endpoint = QuicEndpoint::bind_isolated(&secret).await.unwrap();

    assert_eq!(endpoint.id(), quic::node_id(&secret));
    endpoint.close().await;
}

#[tokio::test]
async fn peers_authenticate_each_other_and_exchange_messages() {
    let (dialer, listener, outbound, inbound) = connected().await;
    assert_eq!(outbound.peer(), listener.id());
    assert_eq!(inbound.peer(), dialer.id());

    let mut sending = outbound.open().await.unwrap();
    sending.send(&"over real QUIC").await.unwrap();
    let mut receiving = within(LIMIT, inbound.accept()).await.unwrap();
    let received: String = within(LIMIT, receiving.receive()).await.unwrap();
    assert_eq!(received, "over real QUIC");

    receiving.send(&42u32).await.unwrap();
    let reply: u32 = within(LIMIT, sending.receive()).await.unwrap();
    assert_eq!(reply, 42);

    dialer.close().await;
    listener.close().await;
}

#[tokio::test]
async fn the_probe_runs_over_a_real_connection() {
    let (dialer, listener, outbound, inbound) = connected().await;
    tokio::spawn(async move {
        while let Ok((_, stream)) = accept_stream(&*inbound).await {
            tokio::spawn(probe::respond(stream));
        }
    });

    let report = within(LIMIT, probe::measure(&*outbound, ProbeConfig::default()))
        .await
        .unwrap();

    assert!(report.rtt < Duration::from_millis(100), "{:?}", report.rtt);
    assert!(report.upload_bps > 1_000_000, "{}", report.upload_bps);
    assert!(report.download_bps > 1_000_000, "{}", report.download_bps);
    assert_eq!(report.path, PathKind::Direct);

    dialer.close().await;
    listener.close().await;
}

#[tokio::test]
async fn dialling_an_unknown_peer_fails() {
    let dialer = isolated_endpoint().await;
    let stranger = quic::node_id(&quic::generate_secret());

    let result = within(LIMIT, dialer.connect(stranger)).await;

    assert!(result.is_err());
    dialer.close().await;
}
