//! Consensus algorithms for distributed systems
//!
//! This module provides implementations of various consensus algorithms
//! including Raft, PBFT, Proof of Stake, and simple majority consensus.

pub mod coordinator;
pub mod majority;
pub mod pbft;
pub mod proof_of_stake;
pub mod raft;

use crate::error::{MetricsError, Result};
use scirs2_core::numeric::Float;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant, SystemTime};

// Re-export main components
pub use coordinator::*;
pub use majority::*;
pub use pbft::*;
pub use proof_of_stake::*;
pub use raft::*;

/// Consensus algorithm configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusConfig {
    /// Algorithm to use
    pub algorithm: ConsensusAlgorithm,

    /// Minimum number of nodes for quorum
    pub quorum_size: usize,

    /// Election timeout (milliseconds)
    pub election_timeout_ms: u64,

    /// Heartbeat interval (milliseconds)
    pub heartbeat_interval_ms: u64,

    /// Maximum entries per append
    pub max_entries_per_append: usize,

    /// Log compaction threshold
    pub log_compaction_threshold: usize,

    /// Snapshot creation interval
    pub snapshot_interval: Duration,

    /// Byzantine fault tolerance threshold
    pub byzantine_threshold: usize,

    /// Enable consensus optimization
    pub enable_optimization: bool,

    /// Consensus timeout (milliseconds)
    pub consensus_timeout_ms: u64,
}

/// Consensus algorithms supported
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ConsensusAlgorithm {
    /// Raft consensus algorithm
    Raft,

    /// Practical Byzantine Fault Tolerance
    PBFT,

    /// Proof of Stake consensus
    ProofOfStake,

    /// Simple majority consensus
    SimpleMajority,

    /// Custom consensus algorithm
    Custom(String),
}

/// Consensus system state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusSystemState {
    /// Current consensus leader
    pub current_leader: Option<String>,

    /// Active participants
    pub active_participants: HashSet<String>,

    /// Current term/epoch
    pub current_term: u64,

    /// Last consensus timestamp
    pub last_consensus: SystemTime,

    /// Consensus statistics
    pub consensus_stats: ConsensusStats,

    /// Algorithm-specific state
    pub algorithm_state: AlgorithmSpecificState,
}

/// Algorithm-specific consensus state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlgorithmSpecificState {
    /// Raft-specific state
    Raft(raft::RaftState),

    /// PBFT-specific state
    PBFT(pbft::PbftState),

    /// Proof of Stake specific state
    ProofOfStake(proof_of_stake::PoSState),

    /// Simple majority state
    SimpleMajority(majority::MajorityState),

    /// Custom algorithm state
    Custom(HashMap<String, String>),
}

/// Consensus statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConsensusStats {
    /// Total consensus decisions made
    pub total_decisions: u64,

    /// Average consensus time (milliseconds)
    pub avg_consensus_time_ms: f64,

    /// Failed consensus attempts
    pub failed_attempts: u64,

    /// Leader changes
    pub leader_changes: u64,

    /// Byzantine failures detected
    pub byzantine_failures: u64,

    /// Network partitions handled
    pub network_partitions: u64,

    /// Current consensus health (0.0-1.0)
    pub health_score: f64,
}

/// Consensus proposal for distributed decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusProposal<T> {
    /// Proposal identifier
    pub proposal_id: String,

    /// Proposer node identifier
    pub proposer: String,

    /// Proposal timestamp
    pub timestamp: SystemTime,

    /// Proposal data
    pub data: T,

    /// Proposal priority
    pub priority: u32,

    /// Required quorum size
    pub required_quorum: usize,

    /// Proposal metadata
    pub metadata: HashMap<String, String>,
}

/// Consensus decision result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusDecision<T> {
    /// Decision identifier
    pub decision_id: String,

    /// Original proposal
    pub proposal: ConsensusProposal<T>,

    /// Decision outcome
    pub outcome: DecisionOutcome,

    /// Participating nodes
    pub participants: HashSet<String>,

    /// Decision timestamp
    pub timestamp: SystemTime,

    /// Consensus algorithm used
    pub algorithm: ConsensusAlgorithm,

    /// Decision metadata
    pub metadata: HashMap<String, String>,
}

/// Consensus decision outcomes
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DecisionOutcome {
    /// Consensus reached, proposal accepted
    Accepted,

    /// Consensus reached, proposal rejected
    Rejected,

    /// Consensus failed, timeout occurred
    Timeout,

    /// Consensus failed, insufficient participants
    InsufficientQuorum,

    /// Consensus failed, Byzantine fault detected
    ByzantineFault,

    /// Consensus failed, network partition
    NetworkPartition,
}

/// Node vote in consensus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusVote {
    /// Voting node identifier
    pub voter: String,

    /// Vote decision
    pub vote: VoteDecision,

    /// Vote timestamp
    pub timestamp: SystemTime,

    /// Vote signature/proof
    pub signature: Option<String>,

    /// Vote metadata
    pub metadata: HashMap<String, String>,
}

/// Vote decisions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum VoteDecision {
    /// Vote in favor
    Accept,

    /// Vote against
    Reject,

    /// Abstain from voting
    Abstain,
}

/// Consensus manager trait for different algorithms
#[allow(async_fn_in_trait)]
pub trait ConsensusManager<T>: Send + Sync {
    /// Initialize consensus manager
    async fn initialize(&mut self) -> Result<()>;

    /// Propose a value for consensus
    async fn propose(&mut self, proposal: ConsensusProposal<T>) -> Result<String>;

    /// Vote on a proposal
    async fn vote(&mut self, proposal_id: &str, vote: ConsensusVote) -> Result<()>;

    /// Get consensus decision if available
    async fn get_decision(&self, proposal_id: &str) -> Result<Option<ConsensusDecision<T>>>;

    /// Handle node failure
    async fn handle_node_failure(&mut self, node_id: &str) -> Result<()>;

    /// Handle node recovery
    async fn handle_node_recovery(&mut self, node_id: &str) -> Result<()>;

    /// Get current consensus state
    async fn get_state(&self) -> Result<ConsensusSystemState>;

    /// Get health score
    async fn get_health_score(&self) -> Result<f64>;

    /// Shutdown consensus manager
    async fn shutdown(&mut self) -> Result<()>;
}

/// Consensus event for monitoring and debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusEvent {
    /// Event identifier
    pub event_id: String,

    /// Event type
    pub event_type: ConsensusEventType,

    /// Event timestamp
    pub timestamp: SystemTime,

    /// Affected nodes
    pub nodes: Vec<String>,

    /// Event data
    pub data: HashMap<String, String>,
}

/// Types of consensus events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsensusEventType {
    /// New proposal submitted
    ProposalSubmitted,

    /// Vote cast by node
    VoteCast,

    /// Consensus decision reached
    DecisionReached,

    /// Leader election started
    LeaderElection,

    /// Leader change occurred
    LeaderChange,

    /// Node failure detected
    NodeFailure,

    /// Node recovery detected
    NodeRecovery,

    /// Byzantine behavior detected
    ByzantineBehavior,

    /// Network partition detected
    NetworkPartition,

    /// Network partition healed
    PartitionHealed,
}

/// Consensus performance metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConsensusPerformanceMetrics {
    /// Average proposal processing time
    pub avg_proposal_time_ms: f64,

    /// Average voting time
    pub avg_voting_time_ms: f64,

    /// Consensus success rate
    pub success_rate: f64,

    /// Network message overhead
    pub message_overhead: f64,

    /// Memory usage (bytes)
    pub memory_usage_bytes: usize,

    /// CPU utilization (0.0-1.0)
    pub cpu_utilization: f64,
}

impl Default for ConsensusConfig {
    fn default() -> Self {
        Self {
            algorithm: ConsensusAlgorithm::Raft,
            quorum_size: 3,
            election_timeout_ms: 5000,
            heartbeat_interval_ms: 1000,
            max_entries_per_append: 100,
            log_compaction_threshold: 10000,
            snapshot_interval: Duration::from_secs(300), // 5 minutes
            byzantine_threshold: 1,
            enable_optimization: true,
            consensus_timeout_ms: 30000, // 30 seconds
        }
    }
}

impl<T> ConsensusProposal<T> {
    /// Create new consensus proposal
    pub fn new(proposal_id: String, proposer: String, data: T) -> Self {
        Self {
            proposal_id,
            proposer,
            timestamp: SystemTime::now(),
            data,
            priority: 0,
            required_quorum: 3,
            metadata: HashMap::new(),
        }
    }

    /// Set proposal priority
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Set required quorum size
    pub fn with_quorum(mut self, quorum: usize) -> Self {
        self.required_quorum = quorum;
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

impl ConsensusVote {
    /// Create new consensus vote
    pub fn new(voter: String, vote: VoteDecision) -> Self {
        Self {
            voter,
            vote,
            timestamp: SystemTime::now(),
            signature: None,
            metadata: HashMap::new(),
        }
    }

    /// Add signature to vote
    pub fn with_signature(mut self, signature: String) -> Self {
        self.signature = Some(signature);
        self
    }

    /// Add metadata to vote
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consensus_config_default() {
        let config = ConsensusConfig::default();
        assert_eq!(config.algorithm, ConsensusAlgorithm::Raft);
        assert_eq!(config.quorum_size, 3);
        assert!(config.enable_optimization);
    }

    #[test]
    fn test_consensus_proposal() {
        let proposal = ConsensusProposal::new(
            "test_proposal".to_string(),
            "node1".to_string(),
            "test_data".to_string(),
        );

        assert_eq!(proposal.proposal_id, "test_proposal");
        assert_eq!(proposal.proposer, "node1");
        assert_eq!(proposal.priority, 0);
    }

    #[test]
    fn test_consensus_vote() {
        let vote = ConsensusVote::new("node1".to_string(), VoteDecision::Accept);
        assert_eq!(vote.voter, "node1");
        assert_eq!(vote.vote, VoteDecision::Accept);
        assert!(vote.signature.is_none());
    }

    #[test]
    fn test_consensus_stats() {
        let stats = ConsensusStats::default();
        assert_eq!(stats.total_decisions, 0);
        assert_eq!(stats.health_score, 0.0);
    }
}
