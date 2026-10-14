use std::collections::HashMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use compute_net::{BiStream, Connection, Endpoint, NetError, Result};
use compute_proto::NodeId;
use compute_proto::admin::PathKind;
use tokio::sync::mpsc;
use tokio::time::sleep;
use tokio_util::sync::CancellationToken;

use crate::link::{Link, LinkProfile};
use crate::pipe;

type LinkKey = (NodeId, NodeId);
type Listener = mpsc::UnboundedSender<Arc<SimConnection>>;
type Inbox<T> = tokio::sync::Mutex<mpsc::UnboundedReceiver<T>>;

#[derive(Clone)]
pub struct SimNetwork {
    state: Arc<State>,
}

struct State {
    default_profile: LinkProfile,
    endpoints_created: AtomicU8,
    listeners: Mutex<HashMap<NodeId, Listener>>,
    links: Mutex<HashMap<LinkKey, Arc<Link>>>,
}

impl SimNetwork {
    pub fn new(default_profile: LinkProfile) -> Self {
        Self {
            state: Arc::new(State {
                default_profile,
                endpoints_created: AtomicU8::new(0),
                listeners: Mutex::default(),
                links: Mutex::default(),
            }),
        }
    }

    /// A new node with a fresh identity.
    pub fn endpoint(&self) -> SimEndpoint {
        let serial = self.state.endpoints_created.fetch_add(1, Ordering::Relaxed) + 1;
        self.endpoint_with_id(NodeId::from_bytes([serial; 32]))
    }

    /// Brings a node up under a known identity, as after a restart.
    pub fn endpoint_with_id(&self, id: NodeId) -> SimEndpoint {
        let (listener, incoming) = mpsc::unbounded_channel();
        let registered = listener.clone();
        self.state.listeners.lock().unwrap().insert(id, registered);
        SimEndpoint {
            id,
            state: self.state.clone(),
            listener,
            incoming: tokio::sync::Mutex::new(incoming),
        }
    }

    pub fn set_link(&self, a: NodeId, b: NodeId, profile: LinkProfile) {
        self.state.link(a, b).set_profile(profile);
    }

    /// Stops traffic between two nodes without closing their connections.
    pub fn partition(&self, a: NodeId, b: NodeId) {
        self.state.link(a, b).set_up(false);
    }

    pub fn heal(&self, a: NodeId, b: NodeId) {
        self.state.link(a, b).set_up(true);
    }

    /// Breaks every open connection between two nodes; new connections can still be made.
    pub fn sever(&self, a: NodeId, b: NodeId) {
        self.state.link(a, b).sever_connections();
    }
}

impl State {
    fn link(&self, a: NodeId, b: NodeId) -> Arc<Link> {
        let key = if a <= b { (a, b) } else { (b, a) };
        self.links
            .lock()
            .unwrap()
            .entry(key)
            .or_insert_with(|| Arc::new(Link::new(self.default_profile)))
            .clone()
    }
}

pub struct SimEndpoint {
    id: NodeId,
    state: Arc<State>,
    listener: Listener,
    incoming: Inbox<Arc<SimConnection>>,
}

impl Drop for SimEndpoint {
    /// Stops listening, unless a restarted node has already taken over this identity.
    fn drop(&mut self) {
        let mut listeners = self.state.listeners.lock().unwrap();
        if listeners
            .get(&self.id)
            .is_some_and(|current| current.same_channel(&self.listener))
        {
            listeners.remove(&self.id);
        }
    }
}

#[async_trait]
impl Endpoint for SimEndpoint {
    fn id(&self) -> NodeId {
        self.id
    }

    async fn connect(&self, peer: NodeId) -> Result<Arc<dyn Connection>> {
        let link = self.state.link(self.id, peer);
        link.passable().await;
        sleep(link.profile().latency * 2).await;
        link.passable().await;

        let severed = link.track_connection();
        let (ours, theirs) = SimConnection::pair(self.id, peer, link, severed);
        let listener = self.state.listeners.lock().unwrap().get(&peer).cloned();
        listener
            .and_then(|listener| listener.send(Arc::new(theirs)).ok())
            .ok_or_else(|| NetError::Connect(format!("{peer:?} is not listening")))?;
        Ok(Arc::new(ours))
    }

    async fn accept(&self) -> Result<Arc<dyn Connection>> {
        let connection = self.incoming.lock().await.recv().await;
        Ok(connection.ok_or_else(|| NetError::Lost("network shut down".into()))?)
    }
}

struct SimConnection {
    peer: NodeId,
    link: Arc<Link>,
    severed: CancellationToken,
    to_peer: mpsc::UnboundedSender<BiStream>,
    from_peer: Inbox<BiStream>,
}

impl SimConnection {
    fn pair(a: NodeId, b: NodeId, link: Arc<Link>, severed: CancellationToken) -> (Self, Self) {
        let (to_a, from_b) = mpsc::unbounded_channel();
        let (to_b, from_a) = mpsc::unbounded_channel();
        let side = |peer, to_peer, from_peer| Self {
            peer,
            link: link.clone(),
            severed: severed.clone(),
            to_peer,
            from_peer: tokio::sync::Mutex::new(from_peer),
        };
        (side(b, to_b, from_b), side(a, to_a, from_a))
    }

    fn lost() -> NetError {
        NetError::Lost("simulated connection severed".into())
    }
}

#[async_trait]
impl Connection for SimConnection {
    fn peer(&self) -> NodeId {
        self.peer
    }

    fn path(&self) -> PathKind {
        PathKind::Direct
    }

    async fn open(&self) -> Result<BiStream> {
        if self.severed.is_cancelled() {
            return Err(Self::lost());
        }
        let (outbound_writer, outbound_reader) =
            pipe::shaped(self.link.clone(), self.severed.clone());
        let (inbound_writer, inbound_reader) =
            pipe::shaped(self.link.clone(), self.severed.clone());
        self.to_peer
            .send(BiStream::new(inbound_writer, outbound_reader))
            .map_err(|_| Self::lost())?;
        Ok(BiStream::new(outbound_writer, inbound_reader))
    }

    async fn accept(&self) -> Result<BiStream> {
        let mut from_peer = self.from_peer.lock().await;
        let stream = self.severed.run_until_cancelled(from_peer.recv()).await;
        stream.flatten().ok_or_else(Self::lost)
    }
}
