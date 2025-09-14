//! Consensus algorithms for distributed coordination
//!
//! This module provides implementations of various consensus algorithms:
//! - Raft consensus algorithm
//! - Practical Byzantine Fault Tolerance (PBFT)
//! - Proof of Stake consensus
//! - Simple majority voting

use crate::error::{MetricsError, Result};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
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
#[derive(Debug)]
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
        }
    }

    /// Start an election
    pub fn start_election(&mut self) -> Result<()> {
        self.current_term += 1;
        self.state = NodeState::Candidate;
        self.voted_for = Some(self.node_id.clone());
        self.last_heartbeat = Instant::now();

        // Reset election timeout with randomization
        let base_timeout = self.config.election_timeout_ms;
        let jitter = rand::rng().gen_range(0..base_timeout / 2);
        self.election_timeout = Duration::from_millis(base_timeout + jitter);

        // TODO: Send vote requests to all peers
        // This would be implemented with actual network communication

        Ok(())
    }

    /// Append entries to log
    pub fn append_entries(&mut self, entries: Vec<LogEntry>) -> Result<bool> {
        // Simplified append entries implementation
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
        for peer_id in self.peers.keys() {
            self.next_index.insert(peer_id.clone(), log_length);
            self.match_index.insert(peer_id.clone(), 0);
        }

        Ok(())
    }

    /// Send heartbeat to peers
    pub fn send_heartbeat(&mut self) -> Result<()> {
        if self.state != NodeState::Leader {
            return Ok(());
        }

        self.last_heartbeat = Instant::now();

        // TODO: Send heartbeat messages to all peers
        // This would be implemented with actual network communication

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
}

impl ConsensusManager for RaftConsensus {
    fn start(&mut self) -> Result<()> {
        // Initialize Raft consensus
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
            index: self.log.len() as u64,
            command: Command::UserData(data),
            timestamp: SystemTime::now(),
        };

        let entry_id = format!("entry_{}_{}", self.current_term, entry.index);
        self.log.push(entry);

        // TODO: Replicate to followers

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
            committed_index: 0, // Simplified
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

/// PBFT consensus implementation (simplified)
#[derive(Debug)]
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
        }
    }

    /// Check if we have enough replicas for consensus
    pub fn has_quorum(&self) -> bool {
        let total_nodes = self.peers.len() + 1; // +1 for self
        let healthy_nodes = self.peers.values().filter(|p| p.is_healthy).count() + 1;

        // PBFT requires 3f + 1 nodes to tolerate f Byzantine failures
        // For simplicity, we use 2f + 1 for non-Byzantine consensus
        healthy_nodes >= (total_nodes * 2 / 3) + 1
    }

    /// Start PBFT three-phase protocol
    pub fn start_consensus(&mut self, request: Vec<u8>) -> Result<String> {
        if !self.has_quorum() {
            return Err(MetricsError::ConsensusError(
                "Insufficient nodes for consensus".to_string(),
            ));
        }

        self.sequence_number += 1;
        let message = PbftMessage {
            message_type: PbftMessageType::PrePrepare,
            view: self.current_view,
            sequence: self.sequence_number,
            digest: self.compute_digest(&request),
            node_id: self.node_id.clone(),
            data: request,
            timestamp: SystemTime::now(),
        };

        self.message_log.push(message.clone());

        // TODO: Send pre-prepare to all replicas

        Ok(format!(
            "pbft_{}_{}",
            self.current_view, self.sequence_number
        ))
    }

    fn compute_digest(&self, data: &[u8]) -> String {
        // Simplified hash computation
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        data.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }
}

impl ConsensusManager for PbftConsensus {
    fn start(&mut self) -> Result<()> {
        // Initialize PBFT
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
            node_state: NodeState::Follower, // Simplified
            log_length: self.message_log.len(),
            committed_index: 0, // Simplified
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

/// Simple majority consensus (for testing/fallback)
#[derive(Debug)]
pub struct SimpleMajorityConsensus {
    /// Node ID
    node_id: String,
    /// Known peers
    peers: HashMap<String, PeerState>,
    /// Vote history
    votes: VecDeque<Vote>,
    /// Configuration
    config: ConsensusConfig,
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
        }
    }

    /// Submit a proposal for voting
    pub fn submit_proposal(&mut self, proposal: Vec<u8>) -> Result<String> {
        let proposal_id = format!(
            "proposal_{}_{}",
            SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis(),
            rand::rng().gen::<u64>()
        );

        let vote = Vote {
            proposal_id: proposal_id.clone(),
            proposal_data: proposal,
            votes_for: 1, // Self vote
            votes_against: 0,
            voters: vec![self.node_id.clone()],
            timestamp: SystemTime::now(),
        };

        self.votes.push_back(vote);

        // Cleanup old votes
        while self.votes.len() > 1000 {
            self.votes.pop_front();
        }

        // TODO: Send vote request to peers

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
        // Initialize simple majority consensus
        self.votes.clear();
        Ok(())
    }

    fn propose(&mut self, data: Vec<u8>) -> Result<String> {
        self.submit_proposal(data)
    }

    fn get_state(&self) -> ConsensusState {
        ConsensusState {
            term: 0,                            // No term concept in simple majority
            leader: Some(self.node_id.clone()), // Everyone can propose
            node_state: NodeState::Leader,      // Simplified
            log_length: self.votes.len(),
            committed_index: 0, // Simplified
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_raft_consensus_creation() {
        let config = ConsensusConfig::default();
        let peers = vec!["node1".to_string(), "node2".to_string()];
        let mut raft = RaftConsensus::new("node0".to_string(), peers, config);

        assert_eq!(raft.current_term(), 0);
        assert_eq!(*raft.current_state(), NodeState::Follower);
        assert_eq!(raft.log_length(), 0);
    }

    #[test]
    fn test_raft_election() {
        let config = ConsensusConfig::default();
        let peers = vec!["node1".to_string(), "node2".to_string()];
        let mut raft = RaftConsensus::new("node0".to_string(), peers, config);

        raft.start_election().unwrap();
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
            .unwrap();
        assert!(!consensus.has_majority(&proposal_id)); // Need more votes
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
}
