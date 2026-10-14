//! Tracks which nodes are alive. Pure state: callers supply the current time.

use std::collections::HashMap;
use std::time::Duration;

use compute_proto::admin::{Liveness, NodeSummary};
use compute_proto::message::NodeStatus;
use compute_proto::{Generation, NodeId};
use tokio::time::Instant;

#[derive(Clone, Copy, Debug)]
pub struct LivenessPolicy {
    pub unreachable_after: Duration,
    /// When a node's lease expires. It must register again to come back.
    pub lost_after: Duration,
}

impl Default for LivenessPolicy {
    fn default() -> Self {
        Self {
            unreachable_after: Duration::from_secs(15),
            lost_after: Duration::from_secs(60),
        }
    }
}

#[derive(Debug, PartialEq, Eq, thiserror::Error)]
#[error("heartbeat does not belong to the node's current session")]
pub struct StaleSession;

struct NodeRecord {
    name: String,
    session: Generation,
    status: NodeStatus,
    last_seen: Instant,
    liveness: Liveness,
}

pub struct Registry {
    policy: LivenessPolicy,
    nodes: HashMap<NodeId, NodeRecord>,
    next_session: Generation,
}

impl Registry {
    pub fn new(policy: LivenessPolicy) -> Self {
        Self {
            policy,
            nodes: HashMap::new(),
            next_session: Generation::FIRST,
        }
    }

    /// Starts a new session for the node, invalidating any earlier one.
    pub fn register(&mut self, id: NodeId, name: String, now: Instant) -> Generation {
        let session = self.next_session;
        self.next_session = session.next();
        let record = NodeRecord {
            name,
            session,
            status: NodeStatus::default(),
            last_seen: now,
            liveness: Liveness::Online,
        };
        self.nodes.insert(id, record);
        session
    }

    pub fn heartbeat(
        &mut self,
        id: NodeId,
        session: Generation,
        status: NodeStatus,
        now: Instant,
    ) -> Result<(), StaleSession> {
        let record = self
            .nodes
            .get_mut(&id)
            .filter(|record| record.session == session && record.liveness != Liveness::Lost)
            .ok_or(StaleSession)?;
        record.status = status;
        record.last_seen = now;
        record.liveness = Liveness::Online;
        Ok(())
    }

    /// Re-evaluates every node and returns those whose liveness changed.
    pub fn sweep(&mut self, now: Instant) -> Vec<(NodeId, Liveness)> {
        let policy = self.policy;
        let mut changes = Vec::new();
        for (id, record) in &mut self.nodes {
            let liveness = match record.liveness {
                Liveness::Lost => Liveness::Lost,
                _ => policy.judge(now.saturating_duration_since(record.last_seen)),
            };
            if liveness != record.liveness {
                record.liveness = liveness;
                changes.push((*id, liveness));
            }
        }
        changes
    }

    pub fn summaries(&self, now: Instant) -> Vec<NodeSummary> {
        let mut summaries: Vec<_> = self
            .nodes
            .iter()
            .map(|(id, record)| NodeSummary {
                id: *id,
                name: record.name.clone(),
                liveness: record.liveness,
                status: record.status.clone(),
                last_seen: now.saturating_duration_since(record.last_seen),
            })
            .collect();
        summaries.sort_by(|a, b| a.name.cmp(&b.name));
        summaries
    }

    /// Finds a node by exact name, or by an id prefix that matches exactly one node.
    pub fn resolve(&self, query: &str) -> Option<NodeId> {
        let by_name = self.nodes.iter().find(|(_, record)| record.name == query);
        if let Some((id, _)) = by_name {
            return Some(*id);
        }
        let mut by_prefix = self
            .nodes
            .keys()
            .filter(|id| !query.is_empty() && id.to_string().starts_with(query));
        match (by_prefix.next(), by_prefix.next()) {
            (Some(id), None) => Some(*id),
            _ => None,
        }
    }
}

impl LivenessPolicy {
    fn judge(&self, silence: Duration) -> Liveness {
        if silence >= self.lost_after {
            Liveness::Lost
        } else if silence >= self.unreachable_after {
            Liveness::Unreachable
        } else {
            Liveness::Online
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const POLICY: LivenessPolicy = LivenessPolicy {
        unreachable_after: Duration::from_secs(10),
        lost_after: Duration::from_secs(30),
    };

    fn node(byte: u8) -> NodeId {
        NodeId::from_bytes([byte; 32])
    }

    fn secs(seconds: u64) -> Duration {
        Duration::from_secs(seconds)
    }

    fn liveness_of(registry: &Registry, id: NodeId, now: Instant) -> Liveness {
        let summaries = registry.summaries(now);
        summaries.iter().find(|s| s.id == id).unwrap().liveness
    }

    struct Fixture {
        registry: Registry,
        start: Instant,
        id: NodeId,
        session: Generation,
    }

    fn registered() -> Fixture {
        let mut registry = Registry::new(POLICY);
        let start = Instant::now();
        let id = node(1);
        let session = registry.register(id, "alpha".into(), start);
        Fixture {
            registry,
            start,
            id,
            session,
        }
    }

    #[test]
    fn a_registered_node_is_online() {
        let f = registered();
        assert_eq!(liveness_of(&f.registry, f.id, f.start), Liveness::Online);
    }

    #[test]
    fn silence_makes_a_node_unreachable_then_lost() {
        let mut f = registered();

        assert_eq!(f.registry.sweep(f.start + secs(9)), vec![]);
        assert_eq!(
            f.registry.sweep(f.start + secs(10)),
            vec![(f.id, Liveness::Unreachable)]
        );
        assert_eq!(
            f.registry.sweep(f.start + secs(30)),
            vec![(f.id, Liveness::Lost)]
        );
    }

    #[test]
    fn a_heartbeat_brings_an_unreachable_node_back() {
        let mut f = registered();
        f.registry.sweep(f.start + secs(12));
        let now = f.start + secs(13);

        let result = f
            .registry
            .heartbeat(f.id, f.session, NodeStatus::default(), now);

        assert_eq!(result, Ok(()));
        assert_eq!(liveness_of(&f.registry, f.id, now), Liveness::Online);
    }

    #[test]
    fn a_heartbeat_updates_the_reported_status() {
        let mut f = registered();
        let status = NodeStatus {
            paused: true,
            ..NodeStatus::default()
        };

        f.registry
            .heartbeat(f.id, f.session, status.clone(), f.start)
            .unwrap();

        assert_eq!(f.registry.summaries(f.start)[0].status, status);
    }

    #[test]
    fn a_lost_node_cannot_resume_its_old_session() {
        let mut f = registered();
        let now = f.start + secs(31);
        f.registry.sweep(now);

        let result = f
            .registry
            .heartbeat(f.id, f.session, NodeStatus::default(), now);

        assert_eq!(result, Err(StaleSession));
        assert_eq!(liveness_of(&f.registry, f.id, now), Liveness::Lost);
    }

    #[test]
    fn a_lost_node_stays_lost_until_it_registers_again() {
        let mut f = registered();
        f.registry.sweep(f.start + secs(31));
        assert_eq!(f.registry.sweep(f.start + secs(32)), vec![]);

        let now = f.start + secs(40);
        f.registry.register(f.id, "alpha".into(), now);

        assert_eq!(liveness_of(&f.registry, f.id, now), Liveness::Online);
    }

    #[test]
    fn registering_again_invalidates_the_previous_session() {
        let mut f = registered();
        let newer = f.registry.register(f.id, "alpha".into(), f.start);

        let old = f
            .registry
            .heartbeat(f.id, f.session, NodeStatus::default(), f.start);
        let new = f
            .registry
            .heartbeat(f.id, newer, NodeStatus::default(), f.start);

        assert!(newer > f.session);
        assert_eq!(old, Err(StaleSession));
        assert_eq!(new, Ok(()));
    }

    #[test]
    fn heartbeats_from_unregistered_nodes_are_rejected() {
        let mut f = registered();

        let result = f
            .registry
            .heartbeat(node(9), f.session, NodeStatus::default(), f.start);

        assert_eq!(result, Err(StaleSession));
    }

    #[test]
    fn nodes_resolve_by_name_or_unique_id_prefix() {
        let mut f = registered();
        f.registry.register(node(2), "beta".into(), f.start);

        assert_eq!(f.registry.resolve("beta"), Some(node(2)));
        assert_eq!(f.registry.resolve("0101"), Some(node(1)));
        assert_eq!(f.registry.resolve("0"), None, "ambiguous prefix");
        assert_eq!(f.registry.resolve(""), None);
        assert_eq!(f.registry.resolve("gamma"), None);
    }

    #[test]
    fn summaries_are_sorted_by_name_and_report_silence() {
        let mut f = registered();
        f.registry.register(node(2), "aardvark".into(), f.start);

        let summaries = f.registry.summaries(f.start + secs(4));

        let names: Vec<_> = summaries.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["aardvark", "alpha"]);
        assert_eq!(summaries[0].last_seen, secs(4));
    }
}
