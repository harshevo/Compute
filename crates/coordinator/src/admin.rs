//! Local control socket used by the `compute` CLI.

use std::io;
use std::path::Path;
use std::sync::Arc;

use compute_net::{BiStream, NetError};
use compute_proto::admin::{AdminReply, AdminRequest};
#[cfg(windows)]
use tokio::net::{TcpListener, TcpStream};
#[cfg(unix)]
use tokio::net::{UnixListener, UnixStream};

use crate::{Coordinator, CoordinatorError};

#[cfg(unix)]
pub fn bind(path: &Path) -> io::Result<UnixListener> {
    match std::fs::remove_file(path) {
        Err(error) if error.kind() != io::ErrorKind::NotFound => return Err(error),
        _ => {}
    }
    let listener = UnixListener::bind(path)?;
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    Ok(listener)
}

#[cfg(unix)]
pub async fn serve(coordinator: Arc<Coordinator>, listener: UnixListener) -> io::Result<()> {
    loop {
        let (socket, _) = listener.accept().await?;
        tokio::spawn(answer(coordinator.clone(), framed(socket)));
    }
}

#[cfg(unix)]
pub async fn request(path: &Path, request: &AdminRequest) -> Result<AdminReply, NetError> {
    let socket = UnixStream::connect(path)
        .await
        .map_err(|error| NetError::Connect(format!("coordinator is not running ({error})")))?;
    let mut stream = framed(socket);
    stream.send(request).await?;
    stream.receive().await
}

#[cfg(unix)]
fn framed(socket: UnixStream) -> BiStream {
    let (reader, writer) = socket.into_split();
    BiStream::new(writer, reader)
}

async fn answer(coordinator: Arc<Coordinator>, mut stream: BiStream) {
    while let Ok(request) = stream.receive().await {
        let reply = handle(&coordinator, request)
            .await
            .unwrap_or_else(|error| AdminReply::Failed(error.to_string()));
        if stream.send(&reply).await.is_err() {
            return;
        }
    }
}

async fn handle(
    coordinator: &Coordinator,
    request: AdminRequest,
) -> Result<AdminReply, CoordinatorError> {
    Ok(match request {
        AdminRequest::Worker {
            node,
            operation,
            payload,
        } => AdminReply::Json(coordinator.worker(&node, operation, payload).await?),
        AdminRequest::ListNodes => AdminReply::Nodes(coordinator.nodes()),
        AdminRequest::Invite { name } => AdminReply::Invite {
            ticket: coordinator.invite(&name)?.to_string(),
        },
        AdminRequest::Probe { node } => AdminReply::Probe(coordinator.probe(&node).await?),
    })
}

#[cfg(windows)]
pub struct LocalListener {
    listener: TcpListener,
    token: String,
}

#[cfg(windows)]
pub fn bind(path: &Path) -> io::Result<LocalListener> {
    use rand::Rng;
    let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
    listener.set_nonblocking(true)?;
    let token = format!("{:032x}", rand::rng().random::<u128>());
    std::fs::write(path, format!("{}\n{token}", listener.local_addr()?))?;
    Ok(LocalListener {
        listener: TcpListener::from_std(listener)?,
        token,
    })
}

#[cfg(windows)]
pub async fn serve(coordinator: Arc<Coordinator>, local: LocalListener) -> io::Result<()> {
    loop {
        let (socket, _) = local.listener.accept().await?;
        let token = local.token.clone();
        let coordinator = coordinator.clone();
        tokio::spawn(async move {
            let (reader, writer) = socket.into_split();
            let mut stream = BiStream::new(writer, reader);
            let received: Result<String, _> =
                compute_net::within(std::time::Duration::from_secs(5), stream.receive()).await;
            if received.is_ok_and(|value| value == token) {
                answer(coordinator, stream).await;
            }
        });
    }
}

#[cfg(windows)]
pub async fn request(path: &Path, request: &AdminRequest) -> Result<AdminReply, NetError> {
    let text = std::fs::read_to_string(path).map_err(|e| NetError::Connect(e.to_string()))?;
    let (address, token) = text
        .split_once('\n')
        .ok_or(NetError::Protocol("invalid admin address"))?;
    let address: std::net::SocketAddr = address
        .parse()
        .map_err(|_| NetError::Protocol("invalid admin address"))?;
    if !address.ip().is_loopback() {
        return Err(NetError::Protocol("admin address must be loopback"));
    }
    let socket = TcpStream::connect(address)
        .await
        .map_err(|e| NetError::Connect(e.to_string()))?;
    let (reader, writer) = socket.into_split();
    let mut stream = BiStream::new(writer, reader);
    stream.send(&token).await?;
    stream.send(request).await?;
    stream.receive().await
}
