use anyhow::{Result, bail};
use compute_coordinator::admin;
use compute_proto::admin::{AdminReply, AdminRequest};

use crate::home::Home;
use crate::render;

pub async fn invite(home: &Home, name: String) -> Result<()> {
    match ask(home, AdminRequest::Invite { name }).await? {
        AdminReply::Invite { ticket } => {
            println!(
                "On the new machine, run:\n\n  compute agent join {ticket}\n  compute agent run"
            );
            Ok(())
        }
        other => unexpected(other),
    }
}

pub async fn nodes(home: &Home, json: bool) -> Result<()> {
    match ask(home, AdminRequest::ListNodes).await? {
        AdminReply::Nodes(nodes) => {
            if json {
                println!("{}", serde_json::to_string(&nodes)?);
            } else {
                println!("{}", render::nodes(&nodes));
            }
            Ok(())
        }
        other => unexpected(other),
    }
}

pub async fn probe(home: &Home, node: String) -> Result<()> {
    match ask(home, AdminRequest::Probe { node }).await? {
        AdminReply::Probe(report) => {
            println!("{}", render::probe(&report));
            Ok(())
        }
        other => unexpected(other),
    }
}

async fn ask(home: &Home, request: AdminRequest) -> Result<AdminReply> {
    let socket = home.coordinator()?.admin_socket();
    Ok(admin::request(&socket, &request).await?)
}

fn unexpected(reply: AdminReply) -> Result<()> {
    match reply {
        AdminReply::Failed(reason) => bail!(reason),
        _ => bail!("the coordinator sent an unexpected reply"),
    }
}

pub async fn rpc(home: &Home, node: String, operation: String) -> Result<()> {
    use tokio::io::AsyncReadExt;
    let mut payload = String::new();
    tokio::io::stdin()
        .take(900_001)
        .read_to_string(&mut payload)
        .await?;
    if payload.len() > 900_000 {
        bail!("request exceeds 900 KB; use chunked transfer");
    }
    serde_json::from_str::<serde_json::Value>(&payload)?;
    match ask(
        home,
        AdminRequest::Worker {
            node,
            operation,
            payload,
        },
    )
    .await?
    {
        AdminReply::Json(reply) => {
            println!("{reply}");
            Ok(())
        }
        other => unexpected(other),
    }
}
