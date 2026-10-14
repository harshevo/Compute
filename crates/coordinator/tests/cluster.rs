//! End-to-end behaviour of a coordinator and its agents over a simulated home network.

mod support;

use std::time::Duration;

use compute_coordinator::CoordinatorError;
use compute_net::Endpoint;
use compute_net::handshake::greet;
use compute_proto::admin::Liveness;
use compute_proto::message::{Heartbeat, HeartbeatReply, Hello, NodeStatus, Rejection, Welcome};
use compute_proto::{PROTOCOL_VERSION, TicketSecret};
use support::{Cluster, FakeStatus, LOST_AFTER, UNREACHABLE_AFTER, eventually};
use tokio::time::sleep;

fn assert_within(actual: u64, expected: u64, tolerance_percent: u64) {
    let allowed = expected * tolerance_percent / 100;
    assert!(
        actual.abs_diff(expected) <= allowed,
        "{actual} is not within {tolerance_percent}% of {expected}"
    );
}

#[tokio::test(start_paused = true)]
async fn an_invited_agent_comes_online_and_reports_its_hardware() {
    let cluster = Cluster::start();

    let agent = cluster.join("friend-a").await;

    let reported = || cluster.node("friend-a").unwrap().status.gpus == [FakeStatus::gpu()];
    assert!(eventually(Duration::from_secs(10), reported).await);
    assert_eq!(cluster.node("friend-a").unwrap().id, agent.id);
}

#[tokio::test(start_paused = true)]
async fn an_agent_without_an_invitation_is_refused() {
    let cluster = Cluster::start();

    let stranger = cluster.spawn_agent(None);

    assert_eq!(stranger.rejection().await, Rejection::Unauthorized);
    assert!(cluster.coordinator.nodes().is_empty());
}

#[tokio::test(start_paused = true)]
async fn a_forged_ticket_is_refused() {
    let cluster = Cluster::start();
    cluster.invite("friend-a");

    let forger = cluster.spawn_agent(Some(TicketSecret::from_bytes([0; 16])));

    assert_eq!(forger.rejection().await, Rejection::Unauthorized);
}

#[tokio::test(start_paused = true)]
async fn a_ticket_cannot_enrol_a_second_machine() {
    let cluster = Cluster::start();
    let ticket = cluster.invite("friend-a");
    let _first = cluster.spawn_agent(Some(ticket));
    cluster.wait_for("friend-a", Liveness::Online).await;

    let second = cluster.spawn_agent(Some(ticket));

    assert_eq!(second.rejection().await, Rejection::Unauthorized);
}

#[tokio::test(start_paused = true)]
async fn a_peer_speaking_another_protocol_version_is_refused() {
    let cluster = Cluster::start();
    let outdated = cluster.network.endpoint();
    let connection = outdated.connect(cluster.coordinator_id).await.unwrap();
    let mut stream = connection.open().await.unwrap();

    let hello = Hello {
        protocol: PROTOCOL_VERSION + 1,
        ticket: Some(cluster.invite("friend-a")),
    };
    stream.send(&hello).await.unwrap();

    let expected = Welcome::Rejected(Rejection::ProtocolMismatch {
        supported: PROTOCOL_VERSION,
    });
    assert_eq!(stream.receive::<Welcome>().await.unwrap(), expected);
    assert!(cluster.coordinator.nodes().is_empty());
}

#[tokio::test(start_paused = true)]
async fn the_owners_pause_switch_is_visible_to_the_coordinator() {
    let cluster = Cluster::start();
    let agent = cluster.join("friend-a").await;

    agent.status.set_paused(true);

    let paused = || cluster.node("friend-a").unwrap().status.paused;
    assert!(eventually(Duration::from_secs(10), paused).await);
}

#[tokio::test(start_paused = true)]
async fn a_short_network_blip_does_not_change_liveness() {
    let cluster = Cluster::start();
    let agent = cluster.join("friend-a").await;

    cluster.partition(&agent);
    sleep(Duration::from_secs(4)).await;
    cluster.heal(&agent);

    for _ in 0..30 {
        sleep(Duration::from_secs(1)).await;
        assert_eq!(cluster.liveness("friend-a"), Some(Liveness::Online));
    }
}

#[tokio::test(start_paused = true)]
async fn a_partition_makes_a_node_unreachable_then_lost() {
    let cluster = Cluster::start();
    let agent = cluster.join("friend-a").await;

    cluster.partition(&agent);

    sleep(UNREACHABLE_AFTER + Duration::from_secs(6)).await;
    assert_eq!(cluster.liveness("friend-a"), Some(Liveness::Unreachable));
    sleep(LOST_AFTER).await;
    assert_eq!(cluster.liveness("friend-a"), Some(Liveness::Lost));
}

#[tokio::test(start_paused = true)]
async fn an_unreachable_node_recovers_when_the_network_heals() {
    let cluster = Cluster::start();
    let agent = cluster.join("friend-a").await;
    cluster.partition(&agent);
    cluster.wait_for("friend-a", Liveness::Unreachable).await;

    cluster.heal(&agent);

    cluster.wait_for("friend-a", Liveness::Online).await;
}

#[tokio::test(start_paused = true)]
async fn a_lost_node_rejoins_by_itself_when_the_network_heals() {
    let cluster = Cluster::start();
    let agent = cluster.join("friend-a").await;
    cluster.partition(&agent);
    cluster.wait_for("friend-a", Liveness::Lost).await;

    cluster.heal(&agent);

    cluster.wait_for("friend-a", Liveness::Online).await;
    sleep(LOST_AFTER * 2).await;
    assert_eq!(cluster.liveness("friend-a"), Some(Liveness::Online));
}

#[tokio::test(start_paused = true)]
async fn an_agent_reconnects_after_its_connection_is_cut() {
    let cluster = Cluster::start();
    let agent = cluster.join("friend-a").await;

    cluster.sever(&agent);

    sleep(LOST_AFTER * 2).await;
    assert_eq!(cluster.liveness("friend-a"), Some(Liveness::Online));
    assert!(cluster.coordinator.probe("friend-a").await.is_ok());
}

#[tokio::test(start_paused = true)]
async fn a_crashed_agent_is_lost_and_returns_after_a_restart_without_a_ticket() {
    let cluster = Cluster::start();
    let agent = cluster.join("friend-a").await;

    agent.crash();
    cluster.wait_for("friend-a", Liveness::Lost).await;
    let _restarted = cluster.spawn_agent_as(agent.id, None);

    cluster.wait_for("friend-a", Liveness::Online).await;
}

#[tokio::test(start_paused = true)]
async fn agents_rejoin_a_restarted_coordinator_without_new_tickets() {
    let mut cluster = Cluster::start();
    let _a = cluster.join("friend-a").await;
    let _b = cluster.join("friend-b").await;

    cluster.restart_coordinator();
    assert!(cluster.coordinator.nodes().is_empty());

    cluster.wait_for("friend-a", Liveness::Online).await;
    cluster.wait_for("friend-b", Liveness::Online).await;
}

#[tokio::test(start_paused = true)]
async fn agents_keep_retrying_while_the_coordinator_is_down() {
    let mut cluster = Cluster::start();
    let _agent = cluster.join("friend-a").await;

    cluster.stop_coordinator();
    sleep(Duration::from_secs(600)).await;
    cluster.restart_coordinator();

    cluster.wait_for("friend-a", Liveness::Online).await;
}

#[tokio::test(start_paused = true)]
async fn a_superseded_session_cannot_send_heartbeats() {
    let cluster = Cluster::start();
    let node = cluster.network.endpoint();
    let ticket = cluster.invite("friend-a");

    let old_connection = node.connect(cluster.coordinator_id).await.unwrap();
    let (old_session, mut old_control) = greet(&*old_connection, Some(ticket)).await.unwrap();
    let new_connection = node.connect(cluster.coordinator_id).await.unwrap();
    let (new_session, mut new_control) = greet(&*new_connection, None).await.unwrap();

    let beat = |session| Heartbeat {
        session,
        status: NodeStatus::default(),
    };
    old_control.send(&beat(old_session)).await.unwrap();
    new_control.send(&beat(new_session)).await.unwrap();

    assert_eq!(
        old_control.receive::<HeartbeatReply>().await.unwrap(),
        HeartbeatReply::StaleSession
    );
    assert_eq!(
        new_control.receive::<HeartbeatReply>().await.unwrap(),
        HeartbeatReply::Ack
    );
}

#[tokio::test(start_paused = true)]
async fn a_heartbeat_claiming_another_session_is_refused() {
    let cluster = Cluster::start();
    let node = cluster.network.endpoint();
    let connection = node.connect(cluster.coordinator_id).await.unwrap();
    let ticket = Some(cluster.invite("friend-a"));
    let (session, mut control) = greet(&*connection, ticket).await.unwrap();

    let forged = Heartbeat {
        session: session.next(),
        status: NodeStatus::default(),
    };
    control.send(&forged).await.unwrap();

    assert_eq!(
        control.receive::<HeartbeatReply>().await.unwrap(),
        HeartbeatReply::StaleSession
    );
}

#[tokio::test(start_paused = true)]
async fn probing_a_node_measures_its_link() {
    let cluster = Cluster::start();
    let _agent = cluster.join("friend-a").await;

    let report = cluster.coordinator.probe("friend-a").await.unwrap();

    assert_within(report.rtt.as_millis() as u64, 40, 10);
    assert_within(report.upload_bps, 20_000_000, 10);
    assert_within(report.download_bps, 20_000_000, 10);
}

#[tokio::test(start_paused = true)]
async fn probing_fails_cleanly_for_unknown_and_disconnected_nodes() {
    let cluster = Cluster::start();
    let agent = cluster.join("friend-a").await;

    let unknown = cluster.coordinator.probe("nobody").await;
    assert!(matches!(unknown, Err(CoordinatorError::UnknownNode(_))));

    agent.crash();
    cluster.wait_for("friend-a", Liveness::Lost).await;
    let disconnected = cluster.coordinator.probe("friend-a").await;
    assert!(matches!(
        disconnected,
        Err(CoordinatorError::NotConnected(_))
    ));
}

#[tokio::test(start_paused = true)]
async fn a_probe_across_a_partition_fails_instead_of_hanging() {
    let cluster = Cluster::start();
    let agent = cluster.join("friend-a").await;
    cluster.partition(&agent);

    let result = cluster.coordinator.probe("friend-a").await;

    assert!(matches!(result, Err(CoordinatorError::Net(_))));
}

#[tokio::test(start_paused = true)]
async fn one_nodes_failure_does_not_affect_another() {
    let cluster = Cluster::start();
    let failing = cluster.join("friend-a").await;
    let _healthy = cluster.join("friend-b").await;

    cluster.partition(&failing);
    cluster.wait_for("friend-a", Liveness::Lost).await;

    assert_eq!(cluster.liveness("friend-b"), Some(Liveness::Online));
    assert!(cluster.coordinator.probe("friend-b").await.is_ok());
}
