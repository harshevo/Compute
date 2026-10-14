mod commands;
mod home;
mod identity;
mod render;

use clap::{Parser, Subcommand};

use crate::home::Home;

#[derive(Parser)]
#[command(
    name = "compute",
    version,
    about = "Pool GPUs across machines you trust"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run the coordinator on this machine
    Serve,
    /// Create a one-time ticket that lets a new machine join
    Invite { name: String },
    /// List the machines in the pool
    Nodes {
        #[arg(long)]
        json: bool,
    },
    /// Low-level SDK request; JSON payload is read from stdin
    Rpc {
        #[arg(long)]
        node: String,
        operation: String,
    },
    /// Python SDK commands
    #[command(external_subcommand)]
    Python(Vec<String>),
    /// Measure latency and bandwidth to a machine
    Probe { node: String },
    /// Lend this machine's GPU to a pool
    #[command(subcommand)]
    Agent(AgentCommand),
}

#[derive(Subcommand)]
enum AgentCommand {
    /// Store the ticket received from the pool's owner
    Join { ticket: String },
    /// Connect to the pool and stay connected
    Run {
        #[arg(long)]
        allow_execution: bool,
    },
    /// Stop accepting work
    Pause,
    /// Accept work again
    Resume,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .init();
    let home = Home::locate()?;
    match Cli::parse().command {
        Command::Serve => commands::serve::run(&home).await,
        Command::Invite { name } => commands::admin::invite(&home, name).await,
        Command::Nodes { json } => commands::admin::nodes(&home, json).await,
        Command::Rpc { node, operation } => commands::admin::rpc(&home, node, operation).await,
        Command::Python(args) => {
            let python = std::env::var("COMPUTE_PYTHON").unwrap_or_else(|_| {
                if cfg!(windows) {
                    "python".into()
                } else {
                    "python3".into()
                }
            });
            let exit = tokio::process::Command::new(python)
                .args(["-m", "compute_sdk.cli"])
                .args(args)
                .status()
                .await?;
            if !exit.success() {
                std::process::exit(exit.code().unwrap_or(1));
            }
            Ok(())
        }
        Command::Probe { node } => commands::admin::probe(&home, node).await,
        Command::Agent(AgentCommand::Join { ticket }) => commands::agent::join(&home, &ticket),
        Command::Agent(AgentCommand::Run { allow_execution }) => {
            if allow_execution {
                std::fs::write(
                    home.agent()?.path().join("execution-enabled"),
                    b"owner enabled",
                )?;
            }
            commands::agent::run(&home).await
        }
        Command::Agent(AgentCommand::Pause) => commands::agent::pause(&home),
        Command::Agent(AgentCommand::Resume) => commands::agent::resume(&home),
    }
}
