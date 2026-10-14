use std::sync::Arc;

use anyhow::{Context, Result, bail};
use compute_agent::StatusSource;
use compute_agent::{Agent, AgentConfig, HostStatus, PauseSwitch};
use compute_net::quic::QuicEndpoint;
use compute_proto::JoinTicket;
use compute_proto::message::Rejection;

use crate::home::{AgentDir, Home};
use crate::identity;

pub fn join(home: &Home, ticket: &str) -> Result<()> {
    let parsed: JoinTicket = ticket.parse()?;
    let dir = home.agent()?;
    std::fs::write(dir.ticket(), parsed.to_string()).context("cannot store the ticket")?;
    println!("Ticket stored. Start lending this machine with `compute agent run`.");
    Ok(())
}

pub async fn run(home: &Home) -> Result<()> {
    let dir = home.agent()?;
    let ticket = stored_ticket(&dir)?;
    let secret = identity::load_or_create(&dir.identity())?;
    let endpoint = Arc::new(QuicEndpoint::bind(&secret).await?);
    let status = Arc::new(HostStatus::new(dir.path().to_owned(), pause_switch(&dir)));
    let inventory = status
        .worker(compute_proto::message::WorkerRequest {
            operation: "inventory".into(),
            payload: "{}".into(),
        })
        .await;
    if inventory.contains("\"error\"") {
        eprintln!("Hardware inventory unavailable: {inventory}");
    }
    let agent = Agent {
        endpoint: endpoint.clone(),
        coordinator: ticket.coordinator,
        ticket: Some(ticket.secret),
        status,
        config: AgentConfig::default(),
    };

    println!("agent is running; pause any time with `compute agent pause`");
    let outcome = tokio::select! {
        rejection = agent.run() => Err(explain(rejection)),
        interrupted = tokio::signal::ctrl_c() => interrupted.map_err(Into::into),
    };
    endpoint.close().await;
    outcome
}

pub fn pause(home: &Home) -> Result<()> {
    pause_switch(&home.agent()?).pause()?;
    println!("Paused. This machine will not accept work until `compute agent resume`.");
    Ok(())
}

pub fn resume(home: &Home) -> Result<()> {
    pause_switch(&home.agent()?).resume()?;
    println!("Resumed.");
    Ok(())
}

fn pause_switch(dir: &AgentDir) -> PauseSwitch {
    PauseSwitch::new(dir.pause_flag())
}

fn stored_ticket(dir: &AgentDir) -> Result<JoinTicket> {
    let Ok(text) = std::fs::read_to_string(dir.ticket()) else {
        bail!("this machine has not joined a pool; run `compute agent join <ticket>` first");
    };
    Ok(text.parse()?)
}

fn explain(rejection: Rejection) -> anyhow::Error {
    match rejection {
        Rejection::Unauthorized => anyhow::anyhow!(
            "the coordinator does not recognise this machine; ask the owner for a new ticket"
        ),
        Rejection::ProtocolMismatch { supported } => anyhow::anyhow!(
            "this agent is incompatible with the coordinator (it speaks protocol {supported}); \
             install the matching version"
        ),
    }
}

#[cfg(test)]
mod tests {
    use compute_proto::{NodeId, TicketSecret};

    use super::*;

    fn ticket() -> JoinTicket {
        JoinTicket {
            coordinator: NodeId::from_bytes([3; 32]),
            secret: TicketSecret::from_bytes([4; 16]),
        }
    }

    #[test]
    fn a_joined_ticket_is_available_to_later_runs() {
        let root = tempfile::tempdir().unwrap();
        let home = Home::at(root.path());

        join(&home, &ticket().to_string()).unwrap();

        assert_eq!(stored_ticket(&home.agent().unwrap()).unwrap(), ticket());
    }

    #[test]
    fn an_invalid_ticket_is_not_stored() {
        let root = tempfile::tempdir().unwrap();
        let home = Home::at(root.path());

        assert!(join(&home, "nonsense").is_err());
        assert!(stored_ticket(&home.agent().unwrap()).is_err());
    }

    #[test]
    fn pause_and_resume_flip_the_switch_the_agent_reads() {
        let root = tempfile::tempdir().unwrap();
        let home = Home::at(root.path());
        let switch = pause_switch(&home.agent().unwrap());

        pause(&home).unwrap();
        assert!(switch.is_paused());

        resume(&home).unwrap();
        assert!(!switch.is_paused());
    }
}
