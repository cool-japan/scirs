//! Transport infrastructure for distributed consensus communication
//!
//! Provides in-memory message passing between simulated cluster nodes.
//! The `Transport` trait abstracts over network transports so that consensus
//! algorithms can be unit-tested without real networking.

use crate::error::{MetricsError, Result};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// All messages that consensus nodes exchange.
///
/// `ConsensusMessage` covers both Raft and PBFT messages so a single
/// transport can serve all consensus implementations.
#[derive(Debug, Clone)]
pub enum ConsensusMessage {
    // ── Raft messages ──────────────────────────────────────────────────────────
    /// Candidate requests votes from peers.
    RequestVote {
        term: u64,
        candidate_id: String,
        last_log_index: u64,
        last_log_term: u64,
    },
    /// Peer responds to a vote request.
    RequestVoteResponse { term: u64, granted: bool },
    /// Leader sends log entries (or heartbeat when `entries` is empty).
    AppendEntries {
        term: u64,
        leader_id: String,
        prev_log_index: u64,
        prev_log_term: u64,
        /// Raw serialised log entries.
        entries: Vec<Vec<u8>>,
        leader_commit: u64,
    },
    /// Follower responds to AppendEntries.
    AppendEntriesResponse {
        term: u64,
        success: bool,
        match_index: u64,
    },

    // ── PBFT messages ─────────────────────────────────────────────────────────
    /// Primary broadcasts a pre-prepare to all replicas.
    PbftPrePrepare {
        view: u64,
        sequence: u64,
        digest: String,
        data: Vec<u8>,
        node_id: String,
    },
    /// Replica broadcasts prepare after receiving pre-prepare.
    PbftPrepare {
        view: u64,
        sequence: u64,
        digest: String,
        node_id: String,
    },
    /// Replica broadcasts commit after collecting 2f+1 prepares.
    PbftCommit {
        view: u64,
        sequence: u64,
        digest: String,
        node_id: String,
    },
    /// Reply to client after commit.
    PbftReply {
        view: u64,
        sequence: u64,
        result: Vec<u8>,
        node_id: String,
    },
}

/// Abstraction over node-to-node message passing.
pub trait Transport: Send + Sync + std::fmt::Debug {
    /// Send a message to a specific peer.
    fn send(&self, peer_id: &str, msg: ConsensusMessage) -> Result<()>;

    /// Non-blocking receive: returns the next `(sender_id, message)` or `None`.
    fn try_recv(&self) -> Option<(String, ConsensusMessage)>;

    /// Broadcast a message to every peer (not to self).
    fn broadcast(&self, msg: ConsensusMessage) -> Result<()>;

    /// Return the IDs of all known peers (excluding self).
    fn peer_ids(&self) -> Vec<String>;

    /// Return our own node ID.
    fn node_id(&self) -> &str;
}

// ─────────────────────────────────────────────────────────────────────────────
// InMemoryTransport
// ─────────────────────────────────────────────────────────────────────────────

type Sender = crossbeam_channel::Sender<(String, ConsensusMessage)>;
type Receiver = crossbeam_channel::Receiver<(String, ConsensusMessage)>;

/// In-memory transport backed by `crossbeam_channel` unbounded queues.
///
/// Create an entire network at once with [`InMemoryTransport::create_network`].
/// Each entry is `(node_id, transport)`.
pub struct InMemoryTransport {
    node_id: String,
    /// Senders to every node's inbox (keyed by node_id).
    senders: Arc<Mutex<HashMap<String, Sender>>>,
    /// Our own inbox receiver.
    receiver: Receiver,
}

impl std::fmt::Debug for InMemoryTransport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InMemoryTransport")
            .field("node_id", &self.node_id)
            .finish()
    }
}

impl InMemoryTransport {
    /// Build a fully-connected network of N nodes.
    ///
    /// Returns one `InMemoryTransport` per `node_id`, ordered to match the
    /// input slice.
    ///
    /// # Example
    /// ```ignore
    /// let pairs = InMemoryTransport::create_network(&["n0", "n1", "n2"]);
    /// let (id0, t0) = pairs.remove(0);
    /// ```
    pub fn create_network(node_ids: &[&str]) -> Vec<(String, Self)> {
        let mut senders: HashMap<String, Sender> = HashMap::new();
        let mut receivers: Vec<(String, Receiver)> = Vec::with_capacity(node_ids.len());

        // Create one channel per node.
        for &id in node_ids {
            let (tx, rx) = crossbeam_channel::unbounded();
            senders.insert(id.to_string(), tx);
            receivers.push((id.to_string(), rx));
        }

        let shared_senders = Arc::new(Mutex::new(senders));

        receivers
            .into_iter()
            .map(|(id, rx)| {
                let transport = InMemoryTransport {
                    node_id: id.clone(),
                    senders: Arc::clone(&shared_senders),
                    receiver: rx,
                };
                (id, transport)
            })
            .collect()
    }
}

impl Transport for InMemoryTransport {
    fn send(&self, peer_id: &str, msg: ConsensusMessage) -> Result<()> {
        let senders = self
            .senders
            .lock()
            .map_err(|e| MetricsError::ComputationError(format!("transport lock poisoned: {e}")))?;
        if let Some(tx) = senders.get(peer_id) {
            tx.send((self.node_id.clone(), msg)).map_err(|e| {
                MetricsError::ComputationError(format!("send to {peer_id} failed: {e}"))
            })?;
        }
        Ok(())
    }

    fn try_recv(&self) -> Option<(String, ConsensusMessage)> {
        self.receiver.try_recv().ok()
    }

    fn broadcast(&self, msg: ConsensusMessage) -> Result<()> {
        let senders = self
            .senders
            .lock()
            .map_err(|e| MetricsError::ComputationError(format!("transport lock poisoned: {e}")))?;
        for (peer_id, tx) in senders.iter() {
            if peer_id != &self.node_id {
                // Best-effort: ignore individual send errors during broadcast.
                let _ = tx.send((self.node_id.clone(), msg.clone()));
            }
        }
        Ok(())
    }

    fn peer_ids(&self) -> Vec<String> {
        let senders = self.senders.lock().unwrap_or_else(|e| e.into_inner());
        senders
            .keys()
            .filter(|k| *k != &self.node_id)
            .cloned()
            .collect()
    }

    fn node_id(&self) -> &str {
        &self.node_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_network_sizes() {
        let pairs = InMemoryTransport::create_network(&["n0", "n1", "n2"]);
        assert_eq!(pairs.len(), 3);
    }

    #[test]
    fn test_send_and_recv() {
        let mut pairs = InMemoryTransport::create_network(&["n0", "n1"]);
        let (_, t0) = pairs.remove(0);
        let (_, t1) = pairs.remove(0);

        t0.send(
            "n1",
            ConsensusMessage::RequestVote {
                term: 1,
                candidate_id: "n0".to_string(),
                last_log_index: 0,
                last_log_term: 0,
            },
        )
        .expect("send failed");

        let (from, msg) = t1.try_recv().expect("should have received a message");
        assert_eq!(from, "n0");
        assert!(matches!(msg, ConsensusMessage::RequestVote { term: 1, .. }));
    }

    #[test]
    fn test_broadcast_reaches_all_peers() {
        let pairs = InMemoryTransport::create_network(&["n0", "n1", "n2"]);
        // n0 broadcasts
        pairs[0]
            .1
            .broadcast(ConsensusMessage::RequestVoteResponse {
                term: 1,
                granted: true,
            })
            .expect("broadcast failed");

        // n1 and n2 should each receive exactly one message
        assert!(pairs[1].1.try_recv().is_some());
        assert!(pairs[2].1.try_recv().is_some());
        // n0 itself should not receive its own broadcast
        assert!(pairs[0].1.try_recv().is_none());
    }

    #[test]
    fn test_peer_ids_excludes_self() {
        let pairs = InMemoryTransport::create_network(&["n0", "n1", "n2"]);
        let peers = pairs[0].1.peer_ids();
        assert!(!peers.contains(&"n0".to_string()));
        assert_eq!(peers.len(), 2);
    }
}
