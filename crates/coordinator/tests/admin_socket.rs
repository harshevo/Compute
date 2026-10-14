mod support;

use compute_coordinator::admin;
use compute_net::NetError;
use compute_proto::JoinTicket;
use compute_proto::admin::{AdminReply, AdminRequest};
use support::Cluster;

#[tokio::test]
async fn the_cli_socket_answers_requests() {
    let cluster = Cluster::start();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("admin.sock");
    let listener = admin::bind(&path).unwrap();
    tokio::spawn(admin::serve(cluster.coordinator.clone(), listener));

    let nodes = admin::request(&path, &AdminRequest::ListNodes).await;
    assert_eq!(nodes.unwrap(), AdminReply::Nodes(vec![]));

    let invite = AdminRequest::Invite {
        name: "friend-a".into(),
    };
    let AdminReply::Invite { ticket } = admin::request(&path, &invite).await.unwrap() else {
        panic!("expected an invitation");
    };
    let ticket: JoinTicket = ticket.parse().unwrap();
    assert_eq!(ticket.coordinator, cluster.coordinator_id);

    let probe = AdminRequest::Probe {
        node: "nobody".into(),
    };
    let failed = admin::request(&path, &probe).await.unwrap();
    assert_eq!(
        failed,
        AdminReply::Failed("no node matches 'nobody'".into())
    );
}

#[tokio::test]
async fn binding_replaces_a_socket_left_by_a_previous_run() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("admin.sock");
    drop(admin::bind(&path).unwrap());

    assert!(admin::bind(&path).is_ok());
}

#[tokio::test]
async fn a_missing_coordinator_is_reported_clearly() {
    let dir = tempfile::tempdir().unwrap();

    let result = admin::request(&dir.path().join("admin.sock"), &AdminRequest::ListNodes).await;

    assert!(matches!(result, Err(NetError::Connect(_))));
}
