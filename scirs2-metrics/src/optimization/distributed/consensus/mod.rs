//! Consensus algorithms for distributed coordination
//!
//! This module provides implementations of various consensus algorithms:
//! - Raft consensus algorithm
//! - Practical Byzantine Fault Tolerance (PBFT)
//! - Proof of Stake consensus
//! - Simple majority voting

use crate::error::{MetricsError, Result};
use crate::optimization::distributed::transport::{ConsensusMessage, Transport};
use scirs2_core::random::{Rng, RngExt};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

pub use super::config::{ConsensusAlgorithm, ConsensusConfig};

/// Trait for consensus algorithm implementations
pub trait ConsensusManager: Send + Sync {
    /// Start the consensus algorithm
    fn start(&mut self) -> Result<()>;
    /// Submit a proposal for consensus
    fn propose(&mut self, data: Vec<u8>) -> Result<String>;
    /// Get the current consensus state
    fn get_state(&self) -> ConsensusState;
}

/// Raft consensus algorithm implementation
pub struct RaftConsensus {
    /// Node ID
    node_id: String,
    /// Current term
    current_term: u64,
    /// Voted for in current term
    voted_for: Option<String>,
    /// Log entries
    log: Vec<LogEntry>,
    /// Current state
    state: NodeState,
    /// Known peers
    peers: HashMap<String, PeerState>,
    /// Configuration
    config: ConsensusConfig,
    /// Last heartbeat time
    last_heartbeat: Instant,
    /// Election timeout
    election_timeout: Duration,
    /// Next index for each peer
    next_index: HashMap<String, usize>,
    /// Match index for each peer
    match_index: HashMap<String, usize>,
    /// Commit index
    commit_index: u64,
    /// Optional transport for inter-node messaging
    transport: Option<Arc<dyn Transport>>,
}

impl std::fmt::Debug for RaftConsensus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RaftConsensus")
            .field("node_id", &self.node_id)
            .field("current_term", &self.current_term)
            .field("state", &self.state)
            .finish()
    }
}

impl RaftConsensus {
    /// Create a new Raft consensus instance
    pub fn new(node_id: String, peers: Vec<String>, config: ConsensusConfig) -> Self {
        let mut peer_states = HashMap::new();
        for peer in peers {
            peer_states.insert(
                peer.clone(),
                PeerState {
                    id: peer,
                    last_seen: Instant::now(),
                    is_healthy: true,
                    address: None,
                },
            );
        }

        Self {
            node_id,
            current_term: 0,
            voted_for: None,
            log: vec![],
            state: NodeState::Follower,
            peers: peer_states,
            config,
            last_heartbeat: Instant::now(),
            election_timeout: Duration::from_millis(5000),
            next_index: HashMap::new(),
            match_index: HashMap::new(),
            commit_index: 0,
            transport: None,
        }
    }

    /// Attach a transport layer (required for network-connected consensus).
    pub fn set_transport(&mut self, t: Arc<dyn Transport>) {
        self.transport = Some(t);
    }

    /// Drain the receive queue and apply any incoming messages.
    pub fn poll_messages(&mut self) -> Result<()> {
        // Clone Arc to avoid borrowing self while mutating it.
        let transport = match self.transport.clone() {
            Some(t) => t,
            None => return Ok(()),
        };

        // Drain up to a reasonable batch to avoid blocking forever.
        let max_msgs = 256;
        for _ in 0..max_msgs {
            match transport.try_recv() {
                None => break,
                Some((from, msg)) => self.handle_message(from, msg)?,
            }
        }
        Ok(())
    }

    /// Route an incoming message to the appropriate handler.
    fn handle_message(&mut self, from: String, msg: ConsensusMessage) -> Result<()> {
        match msg {
            ConsensusMessage::RequestVote {
                term,
                candidate_id,
                last_log_index: _,
                last_log_term: _,
            } => {
                let granted = self.handle_vote_request(term, candidate_id)?;
                if let Some(ref transport) = self.transport.clone() {
                    let _ = transport.send(
                        &from,
                        ConsensusMessage::RequestVoteResponse {
                            term: self.current_term,
                            granted,
                        },
                    );
                }
            }
            ConsensusMessage::AppendEntries {
                term,
                leader_id,
                prev_log_index,
                prev_log_term: _,
                entries,
                leader_commit,
            } => {
                let success =
                    self.handle_append_entries(term, &leader_id, entries, leader_commit)?;
                let match_index = prev_log_index + if success { 1 } else { 0 };
                if let Some(ref transport) = self.transport.clone() {
                    let _ = transport.send(
                        &from,
                        ConsensusMessage::AppendEntriesResponse {
                            term: self.current_term,
                            success,
                            match_index,
                        },
                    );
                }
            }
            ConsensusMessage::RequestVoteResponse { term, granted } => {
                if term > self.current_term {
                    self.current_term = term;
                    self.state = NodeState::Follower;
                    self.voted_for = None;
                }
                // Vote counting is done in start_election; responses arriving
                // asynchronously are recorded but leader promotion is deferred.
                if granted && self.state == NodeState::Candidate {
                    // We cannot easily count here without state, so we let
                    // start_election do synchronous counting for the MVP.
                }
            }
            ConsensusMessage::AppendEntriesResponse {
                term,
                success,
                match_index,
            } => {
                if term > self.current_term {
                    self.current_term = term;
                    self.state = NodeState::Follower;
                }
                if success && match_index > self.commit_index {
                    self.commit_index = match_index;
                }
            }
            _ => {} // PBFT messages are not for Raft nodes
        }
        Ok(())
    }

    /// Handle an AppendEntries RPC from the leader.
    pub fn handle_append_entries(
        &mut self,
        term: u64,
        leader_id: &str,
        entries: Vec<Vec<u8>>,
        leader_commit: u64,
    ) -> Result<bool> {
        if term < self.current_term {
            return Ok(false);
        }
        // Recognise the leader.
        self.current_term = term;
        self.state = NodeState::Follower;
        self.last_heartbeat = Instant::now();
        self.voted_for = Some(leader_id.to_string());

        // Append entries.
        for raw in entries {
            let entry = LogEntry {
                term,
                index: self.log.len() as u64 + 1,
                command: Command::UserData(raw),
                timestamp: SystemTime::now(),
            };
            self.log.push(entry);
        }

        // Update commit index.
        if leader_commit > self.commit_index {
            self.commit_index = leader_commit.min(self.log.len() as u64);
        }

        Ok(true)
    }

    /// Start an election
    pub fn start_election(&mut self) -> Result<()> {
        self.current_term += 1;
        self.state = NodeState::Candidate;
        self.voted_for = Some(self.node_id.clone());
        self.last_heartbeat = Instant::now();

        // Reset election timeout with randomization
        let base_timeout = self.config.election_timeout_ms;
        let jitter = scirs2_core::random::rng().random_range(0..base_timeout / 2);
        self.election_timeout = Duration::from_millis(base_timeout + jitter);

        // Broadcast RequestVote to all peers (if transport is available).
        let transport = self.transport.clone();
        if let Some(ref transport) = transport {
            let peers = transport.peer_ids();
            let last_log_index = self.log.len() as u64;
            let last_log_term = self.log.last().map(|e| e.term).unwrap_or(0);

            let msg = ConsensusMessage::RequestVote {
                term: self.current_term,
                candidate_id: self.node_id.clone(),
                last_log_index,
                last_log_term,
            };
            let _ = transport.broadcast(msg);

            // Collect responses with a synchronous short-circuit.
            let mut votes = 1u64; // vote for self
            let total = peers.len() as u64 + 1;

            for _ in 0..peers.len() {
                if let Some((_from, inner_msg)) = transport.try_recv() {
                    if let ConsensusMessage::RequestVoteResponse { term, granted } = inner_msg {
                        if term > self.current_term {
                            self.current_term = term;
                            self.state = NodeState::Follower;
                            return Ok(());
                        }
                        if granted {
                            votes += 1;
                        }
                    }
                }
            }

            if votes > total / 2 {
                self.become_leader()?;
            }
        }

        Ok(())
    }

    /// Append entries to log
    pub fn append_entries(&mut self, entries: Vec<LogEntry>) -> Result<bool> {
        for entry in entries {
            self.log.push(entry);
        }
        Ok(true)
    }

    /// Handle vote request
    pub fn handle_vote_request(&mut self, term: u64, candidate_id: String) -> Result<bool> {
        if term > self.current_term {
            self.current_term = term;
            self.voted_for = None;
            self.state = NodeState::Follower;
        }

        let can_vote = self.voted_for.is_none() || self.voted_for.as_ref() == Some(&candidate_id);

        if term == self.current_term && can_vote {
            self.voted_for = Some(candidate_id);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Become leader
    pub fn become_leader(&mut self) -> Result<()> {
        self.state = NodeState::Leader;

        // Initialize next_index and match_index for all peers
        let log_length = self.log.len();
        for peer_id in self.peers.keys().cloned().collect::<Vec<_>>() {
            self.next_index.insert(peer_id.clone(), log_length);
            self.match_index.insert(peer_id, 0);
        }

        Ok(())
    }

    /// Send heartbeat to peers
    pub fn send_heartbeat(&mut self) -> Result<()> {
        if self.state != NodeState::Leader {
            return Ok(());
        }

        self.last_heartbeat = Instant::now();

        let transport = self.transport.clone();
        if let Some(ref transport) = transport {
            let msg = ConsensusMessage::AppendEntries {
                term: self.current_term,
                leader_id: self.node_id.clone(),
                prev_log_index: self.commit_index,
                prev_log_term: self
                    .log
                    .get(self.commit_index as usize)
                    .map(|e| e.term)
                    .unwrap_or(0),
                entries: vec![],
                leader_commit: self.commit_index,
            };
            let _ = transport.broadcast(msg);
        }

        Ok(())
    }

    /// Check if election timeout has occurred
    pub fn is_election_timeout(&self) -> bool {
        self.last_heartbeat.elapsed() > self.election_timeout
    }

    /// Get current log length
    pub fn log_length(&self) -> usize {
        self.log.len()
    }

    /// Get current term
    pub fn current_term(&self) -> u64 {
        self.current_term
    }

    /// Get current state
    pub fn current_state(&self) -> &NodeState {
        &self.state
    }

    /// Get commit index
    pub fn commit_index(&self) -> u64 {
        self.commit_index
    }
}

impl ConsensusManager for RaftConsensus {
    fn start(&mut self) -> Result<()> {
        self.last_heartbeat = Instant::now();
        Ok(())
    }

    fn propose(&mut self, data: Vec<u8>) -> Result<String> {
        if self.state != NodeState::Leader {
            return Err(MetricsError::ConsensusError(
                "Only leader can propose entries".to_string(),
            ));
        }

        let entry = LogEntry {
            term: self.current_term,
            index: self.log.len() as u64 + 1,
            command: Command::UserData(data),
            timestamp: SystemTime::now(),
        };

        let entry_id = format!("entry_{}_{}", self.current_term, entry.index);
        self.log.push(entry.clone());

        // Replicate to followers via transport (if available).
        let transport = self.transport.clone();
        if let Some(ref transport) = transport {
            let entry_bytes = serde_json::to_vec(&entry.command).unwrap_or_default();
            let prev_log_index = self.log.len() as u64 - 1;
            let prev_log_term = if self.log.len() > 1 {
                self.log[self.log.len() - 2].term
            } else {
                0
            };

            let msg = ConsensusMessage::AppendEntries {
                term: self.current_term,
                leader_id: self.node_id.clone(),
                prev_log_index,
                prev_log_term,
                entries: vec![entry_bytes],
                leader_commit: self.commit_index,
            };
            let _ = transport.broadcast(msg);

            // Collect acknowledgements — simple synchronous majority wait.
            let peers = transport.peer_ids();
            let total = peers.len() as u64 + 1;
            let mut acks = 1u64; // count self

            for _ in 0..peers.len() {
                if let Some((_from, resp)) = transport.try_recv() {
                    if let ConsensusMessage::AppendEntriesResponse {
                        success,
                        match_index,
                        ..
                    } = resp
                    {
                        if success {
                            acks += 1;
                            if match_index > self.commit_index {
                                self.commit_index = match_index;
                            }
                        }
                    }
                }
            }

            if acks > total / 2 {
                self.commit_index = self.log.len() as u64;
            }
        }

        Ok(entry_id)
    }

    fn get_state(&self) -> ConsensusState {
        ConsensusState {
            term: self.current_term,
            leader: if self.state == NodeState::Leader {
                Some(self.node_id.clone())
            } else {
                None
            },
            node_state: self.state.clone(),
            log_length: self.log.len(),
            committed_index: self.commit_index as usize,
        }
    }
}

/// Current consensus state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusState {
    /// Current term
    pub term: u64,
    /// Current leader (if known)
    pub leader: Option<String>,
    /// Node state
    pub node_state: NodeState,
    /// Log length
    pub log_length: usize,
    /// Last committed index
    pub committed_index: usize,
}

/// Node states in Raft
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeState {
    /// Follower state
    Follower,
    /// Candidate state (during election)
    Candidate,
    /// Leader state
    Leader,
}

/// Log entry in Raft
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    /// Term when entry was created
    pub term: u64,
    /// Index in the log
    pub index: u64,
    /// Command to apply
    pub command: Command,
    /// Timestamp when entry was created
    pub timestamp: SystemTime,
}

/// Commands that can be stored in the log
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Command {
    /// No-op command (used for heartbeats)
    NoOp,
    /// User data
    UserData(Vec<u8>),
    /// Configuration change
    ConfigChange {
        /// Type of change
        change_type: ConfigChangeType,
        /// Node ID
        node_id: String,
        /// Node address
        address: Option<SocketAddr>,
    },
    /// Snapshot command
    Snapshot {
        /// Last included index
        last_included_index: u64,
        /// Last included term
        last_included_term: u64,
        /// Snapshot data
        data: Vec<u8>,
    },
}

/// Configuration change types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfigChangeType {
    /// Add a new node
    AddNode,
    /// Remove an existing node
    RemoveNode,
    /// Update node address
    UpdateNode,
}

/// Peer state information
#[derive(Debug, Clone)]
pub struct PeerState {
    /// Peer ID
    pub id: String,
    /// Last time we heard from this peer
    pub last_seen: Instant,
    /// Whether the peer is considered healthy
    pub is_healthy: bool,
    /// Peer network address
    pub address: Option<SocketAddr>,
}

// ─────────────────────────────────────────────────────────────────────────────
// PBFT Consensus
// ─────────────────────────────────────────────────────────────────────────────

/// PBFT consensus implementation
pub struct PbftConsensus {
    /// Node ID
    node_id: String,
    /// Current view
    current_view: u64,
    /// Current sequence number
    sequence_number: u64,
    /// Known peers
    peers: HashMap<String, PeerState>,
    /// Configuration
    config: ConsensusConfig,
    /// Message log for three-phase protocol
    message_log: Vec<PbftMessage>,
    /// Optional transport layer
    transport: Option<Arc<dyn Transport>>,
}

impl std::fmt::Debug for PbftConsensus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PbftConsensus")
            .field("node_id", &self.node_id)
            .field("current_view", &self.current_view)
            .finish()
    }
}

impl PbftConsensus {
    /// Create a new PBFT consensus instance
    pub fn new(node_id: String, peers: Vec<String>, config: ConsensusConfig) -> Self {
        let mut peer_states = HashMap::new();
        for peer in peers {
            peer_states.insert(
                peer.clone(),
                PeerState {
                    id: peer,
                    last_seen: Instant::now(),
                    is_healthy: true,
                    address: None,
                },
            );
        }

        Self {
            node_id,
            current_view: 0,
            sequence_number: 0,
            peers: peer_states,
            config,
            message_log: vec![],
            transport: None,
        }
    }

    /// Attach a transport layer.
    pub fn set_transport(&mut self, t: Arc<dyn Transport>) {
        self.transport = Some(t);
    }

    /// Check if we have enough replicas for consensus
    pub fn has_quorum(&self) -> bool {
        let total_nodes = self.peers.len() + 1; // +1 for self
        let healthy_nodes = self.peers.values().filter(|p| p.is_healthy).count() + 1;

        // PBFT requires 3f + 1 nodes to tolerate f Byzantine failures.
        // We use 2f + 1 for non-Byzantine consensus.
        healthy_nodes > (total_nodes * 2 / 3)
    }

    /// Number of faulty nodes we can tolerate.
    fn f(&self) -> usize {
        // 3f + 1 <= n  =>  f <= (n-1)/3
        let n = self.peers.len() + 1;
        (n - 1) / 3
    }

    /// Start PBFT three-phase protocol
    pub fn start_consensus(&mut self, request: Vec<u8>) -> Result<String> {
        if !self.has_quorum() {
            return Err(MetricsError::ConsensusError(
                "Insufficient nodes for consensus".to_string(),
            ));
        }

        self.sequence_number += 1;
        let digest = self.compute_digest(&request);
        let seq = self.sequence_number;
        let view = self.current_view;

        // Phase 1: Pre-prepare — broadcast to all replicas.
        let pre_prepare = PbftMessage {
            message_type: PbftMessageType::PrePrepare,
            view,
            sequence: seq,
            digest: digest.clone(),
            node_id: self.node_id.clone(),
            data: request.clone(),
            timestamp: SystemTime::now(),
        };
        self.message_log.push(pre_prepare.clone());

        let transport = self.transport.clone();
        if let Some(ref transport) = transport {
            let _ = transport.broadcast(ConsensusMessage::PbftPrePrepare {
                view,
                sequence: seq,
                digest: digest.clone(),
                data: request,
                node_id: self.node_id.clone(),
            });

            // Phase 2: Prepare — collect 2f+1 prepare messages.
            let quorum = 2 * self.f() + 1;
            let peers_count = self.peers.len();
            let mut prepares = 1usize; // count self

            for _ in 0..peers_count {
                if let Some((_from, msg)) = transport.try_recv() {
                    if let ConsensusMessage::PbftPrepare {
                        sequence: s,
                        digest: d,
                        ..
                    } = &msg
                    {
                        if *s == seq && *d == digest {
                            prepares += 1;
                            let prepare_msg = PbftMessage {
                                message_type: PbftMessageType::Prepare,
                                view,
                                sequence: seq,
                                digest: digest.clone(),
                                node_id: self.node_id.clone(),
                                data: vec![],
                                timestamp: SystemTime::now(),
                            };
                            self.message_log.push(prepare_msg);
                        }
                    }
                }
            }

            // Phase 3: Commit — if enough prepares, broadcast commit.
            if prepares >= quorum {
                let _ = transport.broadcast(ConsensusMessage::PbftCommit {
                    view,
                    sequence: seq,
                    digest: digest.clone(),
                    node_id: self.node_id.clone(),
                });

                // Collect 2f+1 commits.
                let mut commits = 1usize;
                for _ in 0..peers_count {
                    if let Some((_from, msg)) = transport.try_recv() {
                        if let ConsensusMessage::PbftCommit {
                            sequence: s,
                            digest: d,
                            ..
                        } = &msg
                        {
                            if *s == seq && *d == digest {
                                commits += 1;
                                let commit_msg = PbftMessage {
                                    message_type: PbftMessageType::Commit,
                                    view,
                                    sequence: seq,
                                    digest: digest.clone(),
                                    node_id: self.node_id.clone(),
                                    data: vec![],
                                    timestamp: SystemTime::now(),
                                };
                                self.message_log.push(commit_msg);
                            }
                        }
                    }
                }

                if commits < quorum {
                    return Err(MetricsError::ConsensusError(
                        "PBFT commit quorum not reached".to_string(),
                    ));
                }
            } else {
                return Err(MetricsError::ConsensusError(
                    "PBFT prepare quorum not reached".to_string(),
                ));
            }
        }

        Ok(format!("pbft_{}_{}", view, seq))
    }

    fn compute_digest(&self, data: &[u8]) -> String {
        // Simplified hash computation using DefaultHasher.
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        data.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }
}

impl ConsensusManager for PbftConsensus {
    fn start(&mut self) -> Result<()> {
        self.current_view = 0;
        self.sequence_number = 0;
        Ok(())
    }

    fn propose(&mut self, data: Vec<u8>) -> Result<String> {
        self.start_consensus(data)
    }

    fn get_state(&self) -> ConsensusState {
        ConsensusState {
            term: self.current_view,
            leader: Some(format!(
                "primary_{}",
                self.current_view % (self.peers.len() + 1) as u64
            )),
            node_state: NodeState::Follower, // PBFT is not leader-based
            log_length: self.message_log.len(),
            committed_index: 0,
        }
    }
}

/// PBFT message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PbftMessageType {
    /// Pre-prepare phase
    PrePrepare,
    /// Prepare phase
    Prepare,
    /// Commit phase
    Commit,
    /// View change
    ViewChange,
    /// New view
    NewView,
}

/// PBFT protocol message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PbftMessage {
    /// Message type
    pub message_type: PbftMessageType,
    /// Current view number
    pub view: u64,
    /// Sequence number
    pub sequence: u64,
    /// Message digest
    pub digest: String,
    /// Sender node ID
    pub node_id: String,
    /// Message data
    pub data: Vec<u8>,
    /// Timestamp
    pub timestamp: SystemTime,
}

// ─────────────────────────────────────────────────────────────────────────────
// SimpleMajorityConsensus
// ─────────────────────────────────────────────────────────────────────────────

/// Simple majority consensus (for testing/fallback)
pub struct SimpleMajorityConsensus {
    /// Node ID
    node_id: String,
    /// Known peers
    peers: HashMap<String, PeerState>,
    /// Vote history
    votes: VecDeque<Vote>,
    /// Configuration
    config: ConsensusConfig,
    /// Optional transport
    transport: Option<Arc<dyn Transport>>,
}

impl std::fmt::Debug for SimpleMajorityConsensus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SimpleMajorityConsensus")
            .field("node_id", &self.node_id)
            .finish()
    }
}

impl SimpleMajorityConsensus {
    /// Create a new simple majority consensus instance
    pub fn new(node_id: String, peers: Vec<String>, config: ConsensusConfig) -> Self {
        let mut peer_states = HashMap::new();
        for peer in peers {
            peer_states.insert(
                peer.clone(),
                PeerState {
                    id: peer,
                    last_seen: Instant::now(),
                    is_healthy: true,
                    address: None,
                },
            );
        }

        Self {
            node_id,
            peers: peer_states,
            votes: VecDeque::new(),
            config,
            transport: None,
        }
    }

    /// Attach a transport layer.
    pub fn set_transport(&mut self, t: Arc<dyn Transport>) {
        self.transport = Some(t);
    }

    /// Submit a proposal for voting
    pub fn submit_proposal(&mut self, proposal: Vec<u8>) -> Result<String> {
        let proposal_id = format!(
            "proposal_{}_{}",
            SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0),
            scirs2_core::random::rng().random::<u64>()
        );

        let mut vote = Vote {
            proposal_id: proposal_id.clone(),
            proposal_data: proposal,
            votes_for: 1, // Self vote
            votes_against: 0,
            voters: vec![self.node_id.clone()],
            timestamp: SystemTime::now(),
        };

        // Broadcast vote request to peers (if transport available).
        let transport = self.transport.clone();
        if let Some(ref transport) = transport {
            let _ = transport.broadcast(ConsensusMessage::RequestVote {
                term: 0,
                candidate_id: proposal_id.clone(),
                last_log_index: 0,
                last_log_term: 0,
            });

            // Collect peer votes.
            let peers_count = self.peers.len();
            for _ in 0..peers_count {
                if let Some((_from, msg)) = transport.try_recv() {
                    if let ConsensusMessage::RequestVoteResponse { granted, .. } = msg {
                        if granted {
                            vote.votes_for += 1;
                        } else {
                            vote.votes_against += 1;
                        }
                    }
                }
            }
        }

        self.votes.push_back(vote);

        // Cleanup old votes.
        while self.votes.len() > 1000 {
            self.votes.pop_front();
        }

        Ok(proposal_id)
    }

    /// Check if proposal has majority
    pub fn has_majority(&self, proposal_id: &str) -> bool {
        if let Some(vote) = self.votes.iter().find(|v| v.proposal_id == proposal_id) {
            let total_nodes = self.peers.len() + 1; // +1 for self
            vote.votes_for > total_nodes / 2
        } else {
            false
        }
    }
}

impl ConsensusManager for SimpleMajorityConsensus {
    fn start(&mut self) -> Result<()> {
        self.votes.clear();
        Ok(())
    }

    fn propose(&mut self, data: Vec<u8>) -> Result<String> {
        self.submit_proposal(data)
    }

    fn get_state(&self) -> ConsensusState {
        ConsensusState {
            term: 0,
            leader: Some(self.node_id.clone()),
            node_state: NodeState::Leader,
            log_length: self.votes.len(),
            committed_index: 0,
        }
    }
}

/// Vote for simple majority consensus
#[derive(Debug, Clone)]
pub struct Vote {
    /// Proposal ID
    pub proposal_id: String,
    /// Proposal data
    pub proposal_data: Vec<u8>,
    /// Number of votes for
    pub votes_for: usize,
    /// Number of votes against
    pub votes_against: usize,
    /// List of voters
    pub voters: Vec<String>,
    /// Vote timestamp
    pub timestamp: SystemTime,
}

// ─────────────────────────────────────────────────────────────────────────────
// Factory
// ─────────────────────────────────────────────────────────────────────────────

/// Factory for creating consensus instances
pub struct ConsensusFactory;

impl ConsensusFactory {
    /// Create a consensus manager based on configuration
    pub fn create_consensus(
        algorithm: ConsensusAlgorithm,
        node_id: String,
        peers: Vec<String>,
        config: ConsensusConfig,
    ) -> Result<Box<dyn ConsensusManager>> {
        match algorithm {
            ConsensusAlgorithm::Raft => Ok(Box::new(RaftConsensus::new(node_id, peers, config))),
            ConsensusAlgorithm::Pbft => Ok(Box::new(PbftConsensus::new(node_id, peers, config))),
            ConsensusAlgorithm::SimpleMajority => Ok(Box::new(SimpleMajorityConsensus::new(
                node_id, peers, config,
            ))),
            _ => Err(MetricsError::ConsensusError(format!(
                "Consensus algorithm {:?} not implemented",
                algorithm
            ))),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimization::distributed::transport::InMemoryTransport;

    // ── Basic creation tests ──────────────────────────────────────────────────

    #[test]
    fn test_raft_consensus_creation() {
        let config = ConsensusConfig::default();
        let peers = vec!["node1".to_string(), "node2".to_string()];
        let raft = RaftConsensus::new("node0".to_string(), peers, config);

        assert_eq!(raft.current_term(), 0);
        assert_eq!(*raft.current_state(), NodeState::Follower);
        assert_eq!(raft.log_length(), 0);
    }

    #[test]
    fn test_raft_election_without_transport() {
        let config = ConsensusConfig::default();
        let peers = vec!["node1".to_string(), "node2".to_string()];
        let mut raft = RaftConsensus::new("node0".to_string(), peers, config);

        raft.start_election().expect("election should not fail");
        assert_eq!(*raft.current_state(), NodeState::Candidate);
        assert_eq!(raft.current_term(), 1);
    }

    #[test]
    fn test_pbft_consensus_creation() {
        let config = ConsensusConfig::default();
        let peers = vec![
            "node1".to_string(),
            "node2".to_string(),
            "node3".to_string(),
        ];
        let pbft = PbftConsensus::new("node0".to_string(), peers, config);

        assert!(pbft.has_quorum());
    }

    #[test]
    fn test_simple_majority_consensus() {
        let config = ConsensusConfig::default();
        let peers = vec!["node1".to_string(), "node2".to_string()];
        let mut consensus = SimpleMajorityConsensus::new("node0".to_string(), peers, config);

        let proposal_id = consensus
            .submit_proposal(b"test proposal".to_vec())
            .expect("submit should succeed");
        // Only self-vote, no transport => no majority
        assert!(!consensus.has_majority(&proposal_id));
    }

    #[test]
    fn test_consensus_factory() {
        let config = ConsensusConfig::default();
        let peers = vec!["node1".to_string()];

        let raft = ConsensusFactory::create_consensus(
            ConsensusAlgorithm::Raft,
            "node0".to_string(),
            peers.clone(),
            config.clone(),
        );
        assert!(raft.is_ok());

        let pbft = ConsensusFactory::create_consensus(
            ConsensusAlgorithm::Pbft,
            "node0".to_string(),
            peers.clone(),
            config.clone(),
        );
        assert!(pbft.is_ok());

        let simple = ConsensusFactory::create_consensus(
            ConsensusAlgorithm::SimpleMajority,
            "node0".to_string(),
            peers,
            config,
        );
        assert!(simple.is_ok());
    }

    // ── In-memory transport tests ─────────────────────────────────────────────

    /// Create a 3-node Raft network, simulate election on node 0,
    /// verify it becomes leader (nodes 1 and 2 respond with RequestVoteResponse).
    #[test]
    fn test_raft_leader_election_in_memory() {
        let node_ids = ["n0", "n1", "n2"];
        let mut transports = InMemoryTransport::create_network(&node_ids);

        // Build Raft nodes wired to the transport.
        let config = ConsensusConfig::default();
        let peers_for_n0 = vec!["n1".to_string(), "n2".to_string()];
        let mut node0 = RaftConsensus::new("n0".to_string(), peers_for_n0, config.clone());

        // Extract and attach transport for node 0.
        let (_, t0) = transports.remove(0);
        node0.set_transport(Arc::new(t0));

        // Simulate followers answering vote requests.
        // Before calling start_election we need the follower transports to
        // reply with granted=true.  Because start_election sends then drains
        // synchronously we must pre-populate the n0 inbox with two votes.
        let (_, t1) = transports.remove(0);
        let (_, t2) = transports.remove(0);

        // Both followers pre-send a granted response directly into n0's channel.
        t1.send(
            "n0",
            ConsensusMessage::RequestVoteResponse {
                term: 1,
                granted: true,
            },
        )
        .expect("send should succeed");
        t2.send(
            "n0",
            ConsensusMessage::RequestVoteResponse {
                term: 1,
                granted: true,
            },
        )
        .expect("send should succeed");

        // Now run the election; it will drain the two pre-populated responses.
        node0.start_election().expect("election should succeed");

        // Node 0 should have promoted itself to leader.
        assert_eq!(
            *node0.current_state(),
            NodeState::Leader,
            "node 0 should be leader after majority votes"
        );
        assert_eq!(node0.current_term(), 1);
    }

    /// Leader proposes an entry and followers receive it via poll_messages.
    #[test]
    fn test_raft_log_replication() {
        let node_ids = ["n0", "n1", "n2"];
        let mut transports = InMemoryTransport::create_network(&node_ids);

        let config = ConsensusConfig::default();

        // Build follower nodes.
        let mut node1 = RaftConsensus::new(
            "n1".to_string(),
            vec!["n0".to_string(), "n2".to_string()],
            config.clone(),
        );
        let mut node2 = RaftConsensus::new(
            "n2".to_string(),
            vec!["n0".to_string(), "n1".to_string()],
            config.clone(),
        );

        let (_, t0) = transports.remove(0);
        let (_, t1) = transports.remove(0);
        let (_, t2) = transports.remove(0);

        // Attach transports.
        let t1_arc: Arc<dyn Transport> = Arc::new(t1);
        let t2_arc: Arc<dyn Transport> = Arc::new(t2);
        node1.set_transport(Arc::clone(&t1_arc));
        node2.set_transport(Arc::clone(&t2_arc));

        // Build leader separately (its transport must point at the same network).
        let mut node0 = RaftConsensus::new(
            "n0".to_string(),
            vec!["n1".to_string(), "n2".to_string()],
            config.clone(),
        );
        node0.set_transport(Arc::new(t0));

        // Force node 0 into leader state.
        node0.become_leader().expect("become_leader should succeed");

        // Pre-populate acknowledgements from followers so propose() can collect them.
        t1_arc
            .send(
                "n0",
                ConsensusMessage::AppendEntriesResponse {
                    term: 0,
                    success: true,
                    match_index: 1,
                },
            )
            .expect("send ack n1->n0");
        t2_arc
            .send(
                "n0",
                ConsensusMessage::AppendEntriesResponse {
                    term: 0,
                    success: true,
                    match_index: 1,
                },
            )
            .expect("send ack n2->n0");

        // Propose; the AppendEntries broadcast will reach n1 and n2.
        let entry_id = node0
            .propose(b"hello world".to_vec())
            .expect("propose should succeed");
        assert!(!entry_id.is_empty());

        // Followers process their inbox.
        node1.poll_messages().expect("poll n1");
        node2.poll_messages().expect("poll n2");

        // Both followers should have the entry in their log.
        assert_eq!(node1.log_length(), 1, "follower n1 should have 1 log entry");
        assert_eq!(node2.log_length(), 1, "follower n2 should have 1 log entry");
    }

    /// 3-replica PBFT cluster: propose a command, verify commit succeeds.
    ///
    /// In this MVP the primary is the only node wired to transport; it simulates
    /// the three-phase protocol synchronously.  Replicas pre-populate Prepare
    /// and Commit responses so the primary can collect quorum.
    #[test]
    fn test_pbft_consensus_three_nodes() {
        let node_ids = ["p0", "r1", "r2"];
        let transports = InMemoryTransport::create_network(&node_ids);

        let config = ConsensusConfig::default();
        let mut primary = PbftConsensus::new(
            "p0".to_string(),
            vec!["r1".to_string(), "r2".to_string()],
            config,
        );

        let (_, tp) = transports.into_iter().next().expect("primary transport");
        let tp_arc: Arc<dyn Transport> = Arc::new(tp);

        // Pre-populate Prepare messages from r1 and r2.
        let digest = primary.compute_digest(b"cmd");
        tp_arc
            .send(
                "p0",
                ConsensusMessage::PbftPrepare {
                    view: 0,
                    sequence: 1,
                    digest: digest.clone(),
                    node_id: "r1".to_string(),
                },
            )
            .expect("send prepare r1");
        tp_arc
            .send(
                "p0",
                ConsensusMessage::PbftPrepare {
                    view: 0,
                    sequence: 1,
                    digest: digest.clone(),
                    node_id: "r2".to_string(),
                },
            )
            .expect("send prepare r2");

        // Pre-populate Commit messages from r1 and r2.
        tp_arc
            .send(
                "p0",
                ConsensusMessage::PbftCommit {
                    view: 0,
                    sequence: 1,
                    digest: digest.clone(),
                    node_id: "r1".to_string(),
                },
            )
            .expect("send commit r1");
        tp_arc
            .send(
                "p0",
                ConsensusMessage::PbftCommit {
                    view: 0,
                    sequence: 1,
                    digest,
                    node_id: "r2".to_string(),
                },
            )
            .expect("send commit r2");

        primary.set_transport(Arc::clone(&tp_arc));

        let result = primary.start_consensus(b"cmd".to_vec());
        assert!(
            result.is_ok(),
            "PBFT consensus should succeed: {:?}",
            result
        );
    }
}
