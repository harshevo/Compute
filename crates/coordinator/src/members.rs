//! Durable record of which nodes may join, and of unused invitations.

use std::path::Path;
use std::sync::Mutex;

use compute_proto::{NodeId, TicketSecret};
use rusqlite::{Connection, OptionalExtension, params};

const SCHEMA: &str = "
    CREATE TABLE IF NOT EXISTS members (
        node_id BLOB PRIMARY KEY,
        name    TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS invitations (
        secret BLOB PRIMARY KEY,
        name   TEXT NOT NULL
    );
";

pub struct Members {
    db: Mutex<Connection>,
}

impl Members {
    pub fn open(path: &Path) -> rusqlite::Result<Self> {
        let db = Connection::open(path)?;
        db.pragma_update(None, "journal_mode", "WAL")?;
        Self::with_connection(db)
    }

    pub fn in_memory() -> rusqlite::Result<Self> {
        Self::with_connection(Connection::open_in_memory()?)
    }

    fn with_connection(db: Connection) -> rusqlite::Result<Self> {
        db.execute_batch(SCHEMA)?;
        Ok(Self { db: Mutex::new(db) })
    }

    pub fn invite(&self, name: &str) -> rusqlite::Result<TicketSecret> {
        let secret = TicketSecret::from_bytes(rand::random());
        self.db.lock().unwrap().execute(
            "INSERT INTO invitations (secret, name) VALUES (?1, ?2)",
            params![secret.as_bytes(), name],
        )?;
        Ok(secret)
    }

    /// Returns the node's name if it is a member, enrolling it first when it presents an
    /// unused invitation. An invitation works once.
    pub fn admit(
        &self,
        node: NodeId,
        ticket: Option<TicketSecret>,
    ) -> rusqlite::Result<Option<String>> {
        let mut db = self.db.lock().unwrap();
        let tx = db.transaction()?;
        let mut name = tx
            .query_row(
                "SELECT name FROM members WHERE node_id = ?1",
                params![node.as_bytes()],
                |row| row.get(0),
            )
            .optional()?;
        if name.is_none()
            && let Some(secret) = ticket
        {
            name = tx
                .query_row(
                    "DELETE FROM invitations WHERE secret = ?1 RETURNING name",
                    params![secret.as_bytes()],
                    |row| row.get(0),
                )
                .optional()?;
            if let Some(name) = &name {
                tx.execute(
                    "INSERT INTO members (node_id, name) VALUES (?1, ?2)",
                    params![node.as_bytes(), name],
                )?;
            }
        }
        tx.commit()?;
        Ok(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(byte: u8) -> NodeId {
        NodeId::from_bytes([byte; 32])
    }

    #[test]
    fn strangers_are_not_admitted() {
        let members = Members::in_memory().unwrap();
        let forged = TicketSecret::from_bytes([1; 16]);

        assert_eq!(members.admit(node(1), None).unwrap(), None);
        assert_eq!(members.admit(node(1), Some(forged)).unwrap(), None);
    }

    #[test]
    fn an_invitation_enrols_its_bearer_under_the_invited_name() {
        let members = Members::in_memory().unwrap();
        let secret = members.invite("friend-a").unwrap();

        let name = members.admit(node(1), Some(secret)).unwrap();

        assert_eq!(name.as_deref(), Some("friend-a"));
    }

    #[test]
    fn an_invitation_works_only_once() {
        let members = Members::in_memory().unwrap();
        let secret = members.invite("friend-a").unwrap();
        members.admit(node(1), Some(secret)).unwrap();

        assert_eq!(members.admit(node(2), Some(secret)).unwrap(), None);
    }

    #[test]
    fn a_member_is_admitted_without_a_ticket_afterwards() {
        let members = Members::in_memory().unwrap();
        let secret = members.invite("friend-a").unwrap();
        members.admit(node(1), Some(secret)).unwrap();

        assert_eq!(
            members.admit(node(1), None).unwrap().as_deref(),
            Some("friend-a")
        );
        assert_eq!(
            members.admit(node(1), Some(secret)).unwrap().as_deref(),
            Some("friend-a"),
            "a member re-sending its spent ticket is still a member"
        );
    }

    #[test]
    fn a_member_does_not_consume_a_fresh_invitation() {
        let members = Members::in_memory().unwrap();
        let first = members.invite("friend-a").unwrap();
        let second = members.invite("friend-b").unwrap();
        members.admit(node(1), Some(first)).unwrap();

        members.admit(node(1), Some(second)).unwrap();

        assert_eq!(
            members.admit(node(2), Some(second)).unwrap().as_deref(),
            Some("friend-b")
        );
    }

    #[test]
    fn membership_survives_a_restart() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("state.db");
        let secret = Members::open(&path).unwrap().invite("friend-a").unwrap();
        Members::open(&path)
            .unwrap()
            .admit(node(1), Some(secret))
            .unwrap();

        let reopened = Members::open(&path).unwrap();

        assert_eq!(
            reopened.admit(node(1), None).unwrap().as_deref(),
            Some("friend-a")
        );
    }
}
