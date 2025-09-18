//! Core Consensus Infrastructure
//!
//! This module provides the foundational consensus protocol infrastructure including
//! the consensus protocol trait, protocol manager, configuration types, and basic
//! consensus operations for TPU pod coordination.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::tpu::tpu_backend::DeviceId;
use crate::error::{Result, OptimError};

/// Consensus protocol configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusConfig {
    /// Consensus protocol type
    pub protocol: ConsensusProtocol,
    /// Consensus parameters
    pub parameters: ConsensusParameters,
    /// Fault tolerance settings
    pub fault_tolerance: ConsensusFaultTolerance,
    /// Performance optimization
    pub optimization: ConsensusOptimization,
}

/// Consensus protocols
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsensusProtocol {
    /// Raft consensus protocol
    Raft,
    /// PBFT (Practical Byzantine Fault Tolerance)
    PBFT,
    /// Two-phase commit
    TwoPhaseCommit,
    /// Three-phase commit
    ThreePhaseCommit,
    /// Paxos consensus protocol
    Paxos,
    /// Custom consensus protocol
    Custom { protocol: String },
}

/// Consensus parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusParameters {
    /// Election timeout
    pub election_timeout: Duration,
    /// Heartbeat interval
    pub heartbeat_interval: Duration,
    /// Commit timeout
    pub commit_timeout: Duration,
    /// Quorum size
    pub quorum_size: usize,
    /// Maximum proposal size
    pub max_proposal_size: usize,
}

/// Consensus fault tolerance settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusFaultTolerance {
    /// Maximum tolerable failures
    pub max_failures: usize,
    /// Failure detection timeout
    pub failure_detection_timeout: Duration,
    /// Recovery strategy
    pub recovery_strategy: ConsensusRecoveryStrategy,
}

/// Consensus recovery strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsensusRecoveryStrategy {
    /// Leader re-election
    LeaderReelection,
    /// State synchronization
    StateSynchronization,
    /// Configuration change
    ConfigurationChange,
    /// Custom recovery strategy
    Custom { strategy: String },
}

/// Consensus optimization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusOptimization {
    /// Enable optimization
    pub enable: bool,
    /// Batching settings
    pub batching: ConsensusBatching,
    /// Pipelining settings
    pub pipelining: ConsensusPipelining,
    /// Compression settings
    pub compression: ConsensusCompression,
}

/// Consensus batching settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusBatching {
    /// Enable batching
    pub enable: bool,
    /// Batch size
    pub batch_size: usize,
    /// Batch timeout
    pub timeout: Duration,
}

/// Consensus pipelining settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusPipelining {
    /// Enable pipelining
    pub enable: bool,
    /// Pipeline depth
    pub depth: usize,
    /// Pipeline timeout
    pub timeout: Duration,
}

/// Consensus compression settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusCompression {
    /// Enable compression
    pub enable: bool,
    /// Compression algorithm
    pub algorithm: CompressionAlgorithm,
    /// Compression threshold
    pub threshold: usize,
}

/// Compression algorithms for consensus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    /// LZ4 compression
    LZ4,
    /// GZIP compression
    GZIP,
    /// Snappy compression
    Snappy,
    /// Custom compression
    Custom { algorithm: String },
}

/// Consensus protocol manager
#[derive(Debug)]
pub struct ConsensusProtocolManager {
    /// Consensus configuration
    pub config: ConsensusConfig,
    /// Protocol implementation
    pub protocol: Box<dyn ConsensusProtocolTrait + Send + Sync>,
    /// Consensus state
    pub state: ConsensusState,
    /// Consensus statistics
    pub statistics: ConsensusStatistics,
}

/// Consensus protocol trait
pub trait ConsensusProtocolTrait: std::fmt::Debug + Send + Sync {
    /// Propose a value for consensus
    fn propose(&mut self, value: Vec<u8>) -> Result<ProposalId>;

    /// Vote on a proposal
    fn vote(&mut self, proposal_id: ProposalId, vote: Vote) -> Result<()>;

    /// Get consensus result
    fn get_result(&self, proposal_id: ProposalId) -> Option<ConsensusResult>;

    /// Get protocol status
    fn status(&self) -> ConsensusProtocolStatus;

    /// Handle leader election
    fn handle_election(&mut self) -> Result<()>;

    /// Synchronize state with peers
    fn sync_state(&mut self, peers: &[DeviceId]) -> Result<()>;

    /// Process timeout events
    fn handle_timeout(&mut self) -> Result<()>;
}

/// Proposal identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProposalId(pub u64);

/// Vote in consensus protocol
#[derive(Debug, Clone)]
pub enum Vote {
    /// Accept the proposal
    Accept,
    /// Reject the proposal
    Reject,
    /// Abstain from voting
    Abstain,
    /// Custom vote type
    Custom { vote_type: String, data: Vec<u8> },
}

/// Consensus result
#[derive(Debug, Clone)]
pub struct ConsensusResult {
    /// Proposal identifier
    pub proposal_id: ProposalId,
    /// Consensus decision
    pub decision: ConsensusDecision,
    /// Vote summary
    pub vote_summary: VoteSummary,
    /// Consensus timestamp
    pub timestamp: Instant,
}

/// Consensus decision
#[derive(Debug, Clone)]
pub enum ConsensusDecision {
    /// Consensus reached - proposal accepted
    Accepted { value: Vec<u8> },
    /// Consensus reached - proposal rejected
    Rejected,
    /// No consensus reached
    NoConsensus,
    /// Consensus timed out
    TimedOut,
}

/// Vote summary for consensus
#[derive(Debug, Clone)]
pub struct VoteSummary {
    /// Accept votes
    pub accept_votes: usize,
    /// Reject votes
    pub reject_votes: usize,
    /// Abstain votes
    pub abstain_votes: usize,
    /// Total votes
    pub total_votes: usize,
    /// Quorum reached
    pub quorum_reached: bool,
}

/// Consensus protocol status
#[derive(Debug, Clone)]
pub struct ConsensusProtocolStatus {
    /// Protocol state
    pub state: ProtocolState,
    /// Active proposals
    pub active_proposals: usize,
    /// Leader information
    pub leader: Option<DeviceId>,
    /// Participant count
    pub participant_count: usize,
}

/// Protocol state for consensus
#[derive(Debug, Clone, PartialEq)]
pub enum ProtocolState {
    /// Protocol is initializing
    Initializing,
    /// Protocol is running normally
    Running,
    /// Protocol is in leader election
    LeaderElection,
    /// Protocol is recovering from failure
    Recovering,
    /// Protocol has failed
    Failed { error: String },
}

/// Consensus state tracking
#[derive(Debug)]
pub struct ConsensusState {
    /// Current term/epoch
    pub current_term: u64,
    /// Voted for in current term
    pub voted_for: Option<DeviceId>,
    /// Log of consensus decisions
    pub decision_log: Vec<ConsensusResult>,
    /// Pending proposals
    pub pending_proposals: HashMap<ProposalId, PendingProposal>,
}

/// Pending proposal information
#[derive(Debug, Clone)]
pub struct PendingProposal {
    /// Proposal identifier
    pub id: ProposalId,
    /// Proposed value
    pub value: Vec<u8>,
    /// Proposer device
    pub proposer: DeviceId,
    /// Proposal timestamp
    pub timestamp: Instant,
    /// Received votes
    pub votes: HashMap<DeviceId, Vote>,
    /// Proposal timeout
    pub timeout: Instant,
}

/// Consensus statistics
#[derive(Debug, Clone)]
pub struct ConsensusStatistics {
    /// Total proposals
    pub total_proposals: usize,
    /// Accepted proposals
    pub accepted_proposals: usize,
    /// Rejected proposals
    pub rejected_proposals: usize,
    /// Timed out proposals
    pub timed_out_proposals: usize,
    /// Average consensus time
    pub avg_consensus_time: Duration,
    /// Consensus throughput
    pub throughput: f64,
}

// Implementation blocks

impl ConsensusProtocolManager {
    /// Create a new consensus protocol manager
    pub fn new(
        config: ConsensusConfig,
        protocol: Box<dyn ConsensusProtocolTrait + Send + Sync>,
    ) -> Result<Self> {
        Ok(Self {
            config,
            protocol,
            state: ConsensusState::new(),
            statistics: ConsensusStatistics::new(),
        })
    }

    /// Start consensus protocol
    pub fn start(&mut self) -> Result<()> {
        // Initialize protocol state
        self.state = ConsensusState::new();

        // Start protocol-specific initialization
        // This would typically involve:
        // - Establishing connections to peers
        // - Initializing protocol state machines
        // - Starting heartbeat timers
        // - Loading persisted state if any

        Ok(())
    }

    /// Stop consensus protocol
    pub fn stop(&mut self) -> Result<()> {
        // Graceful shutdown sequence:
        // - Finish pending proposals
        // - Persist current state
        // - Close peer connections
        // - Clean up resources

        Ok(())
    }

    /// Propose a new value for consensus
    pub fn propose(&mut self, value: Vec<u8>) -> Result<ProposalId> {
        // Validate proposal size
        if value.len() > self.config.parameters.max_proposal_size {
            return Err(OptimError::invalid_argument("Proposal size exceeds maximum"));
        }

        // Delegate to protocol implementation
        let proposal_id = self.protocol.propose(value.clone())?;

        // Create pending proposal
        let pending = PendingProposal {
            id: proposal_id,
            value,
            proposer: DeviceId(0), // TODO: Get actual proposer ID
            timestamp: Instant::now(),
            votes: HashMap::new(),
            timeout: Instant::now() + self.config.parameters.commit_timeout,
        };

        // Store pending proposal
        self.state.pending_proposals.insert(proposal_id, pending);
        self.statistics.total_proposals += 1;

        Ok(proposal_id)
    }

    /// Vote on a proposal
    pub fn vote(&mut self, proposal_id: ProposalId, vote: Vote) -> Result<()> {
        // Validate proposal exists
        if !self.state.pending_proposals.contains_key(&proposal_id) {
            return Err(OptimError::not_found("Proposal not found"));
        }

        // Delegate to protocol implementation
        self.protocol.vote(proposal_id, vote)
    }

    /// Get consensus result for a proposal
    pub fn get_result(&self, proposal_id: ProposalId) -> Option<ConsensusResult> {
        self.protocol.get_result(proposal_id)
    }

    /// Get protocol status
    pub fn get_status(&self) -> ConsensusProtocolStatus {
        self.protocol.status()
    }

    /// Handle proposal timeout
    pub fn handle_timeout(&mut self, proposal_id: ProposalId) -> Result<()> {
        if let Some(mut pending) = self.state.pending_proposals.remove(&proposal_id) {
            if Instant::now() >= pending.timeout {
                // Mark proposal as timed out
                let result = ConsensusResult {
                    proposal_id,
                    decision: ConsensusDecision::TimedOut,
                    vote_summary: VoteSummary {
                        accept_votes: 0,
                        reject_votes: 0,
                        abstain_votes: 0,
                        total_votes: pending.votes.len(),
                        quorum_reached: false,
                    },
                    timestamp: Instant::now(),
                };

                self.state.decision_log.push(result);
                self.statistics.timed_out_proposals += 1;
            } else {
                // Put it back if not actually timed out
                self.state.pending_proposals.insert(proposal_id, pending);
            }
        }

        Ok(())
    }

    /// Process completed proposal
    pub fn complete_proposal(&mut self, proposal_id: ProposalId, result: ConsensusResult) -> Result<()> {
        // Remove from pending proposals
        self.state.pending_proposals.remove(&proposal_id);

        // Add to decision log
        self.state.decision_log.push(result.clone());

        // Update statistics
        match result.decision {
            ConsensusDecision::Accepted { .. } => {
                self.statistics.accepted_proposals += 1;
            },
            ConsensusDecision::Rejected => {
                self.statistics.rejected_proposals += 1;
            },
            ConsensusDecision::TimedOut => {
                self.statistics.timed_out_proposals += 1;
            },
            _ => {},
        }

        // Update average consensus time
        let consensus_time = result.timestamp.duration_since(
            self.state.pending_proposals.get(&proposal_id)
                .map(|p| p.timestamp)
                .unwrap_or(Instant::now())
        );

        self.update_average_consensus_time(consensus_time);

        Ok(())
    }

    /// Update average consensus time
    fn update_average_consensus_time(&mut self, new_time: Duration) {
        let total_completed = self.statistics.accepted_proposals +
                            self.statistics.rejected_proposals +
                            self.statistics.timed_out_proposals;

        if total_completed > 0 {
            let current_avg_millis = self.statistics.avg_consensus_time.as_millis() as f64;
            let new_time_millis = new_time.as_millis() as f64;
            let new_avg_millis = (current_avg_millis * (total_completed - 1) as f64 + new_time_millis) / total_completed as f64;
            self.statistics.avg_consensus_time = Duration::from_millis(new_avg_millis as u64);
        } else {
            self.statistics.avg_consensus_time = new_time;
        }
    }

    /// Update throughput statistics
    pub fn update_throughput(&mut self, window_duration: Duration) {
        let total_completed = self.statistics.accepted_proposals +
                            self.statistics.rejected_proposals +
                            self.statistics.timed_out_proposals;

        if window_duration.as_secs_f64() > 0.0 {
            self.statistics.throughput = total_completed as f64 / window_duration.as_secs_f64();
        } else {
            self.statistics.throughput = 0.0;
        }
    }

    /// Get consensus statistics
    pub fn get_statistics(&self) -> &ConsensusStatistics {
        &self.statistics
    }

    /// Reset statistics
    pub fn reset_statistics(&mut self) {
        self.statistics = ConsensusStatistics::new();
    }

    /// Get pending proposal count
    pub fn pending_proposal_count(&self) -> usize {
        self.state.pending_proposals.len()
    }

    /// Get decision log size
    pub fn decision_log_size(&self) -> usize {
        self.state.decision_log.len()
    }

    /// Clear decision log (keeping recent entries)
    pub fn trim_decision_log(&mut self, keep_recent: usize) {
        if self.state.decision_log.len() > keep_recent {
            let remove_count = self.state.decision_log.len() - keep_recent;
            self.state.decision_log.drain(0..remove_count);
        }
    }

    /// Check if quorum is available
    pub fn has_quorum(&self, available_nodes: usize) -> bool {
        available_nodes >= self.config.parameters.quorum_size
    }

    /// Get configuration
    pub fn get_config(&self) -> &ConsensusConfig {
        &self.config
    }

    /// Update configuration
    pub fn update_config(&mut self, config: ConsensusConfig) -> Result<()> {
        // Validate configuration changes
        if config.parameters.quorum_size == 0 {
            return Err(OptimError::invalid_argument("Quorum size must be greater than 0"));
        }

        if config.parameters.max_proposal_size == 0 {
            return Err(OptimError::invalid_argument("Max proposal size must be greater than 0"));
        }

        self.config = config;
        Ok(())
    }
}

impl ConsensusState {
    /// Create new consensus state
    pub fn new() -> Self {
        Self {
            current_term: 0,
            voted_for: None,
            decision_log: Vec::new(),
            pending_proposals: HashMap::new(),
        }
    }

    /// Increment current term
    pub fn increment_term(&mut self) -> u64 {
        self.current_term += 1;
        self.voted_for = None; // Clear vote when term changes
        self.current_term
    }

    /// Set vote for current term
    pub fn set_vote(&mut self, device_id: DeviceId) -> Result<()> {
        if self.voted_for.is_some() {
            return Err(OptimError::invalid_state("Already voted in current term"));
        }
        self.voted_for = Some(device_id);
        Ok(())
    }

    /// Check if already voted in current term
    pub fn has_voted(&self) -> bool {
        self.voted_for.is_some()
    }

    /// Get vote for current term
    pub fn get_vote(&self) -> Option<DeviceId> {
        self.voted_for
    }

    /// Add pending proposal
    pub fn add_pending_proposal(&mut self, proposal: PendingProposal) {
        self.pending_proposals.insert(proposal.id, proposal);
    }

    /// Remove pending proposal
    pub fn remove_pending_proposal(&mut self, proposal_id: ProposalId) -> Option<PendingProposal> {
        self.pending_proposals.remove(&proposal_id)
    }

    /// Get pending proposal
    pub fn get_pending_proposal(&self, proposal_id: ProposalId) -> Option<&PendingProposal> {
        self.pending_proposals.get(&proposal_id)
    }

    /// Get pending proposal (mutable)
    pub fn get_pending_proposal_mut(&mut self, proposal_id: ProposalId) -> Option<&mut PendingProposal> {
        self.pending_proposals.get_mut(&proposal_id)
    }

    /// Get all pending proposals
    pub fn get_pending_proposals(&self) -> &HashMap<ProposalId, PendingProposal> {
        &self.pending_proposals
    }

    /// Add decision to log
    pub fn add_decision(&mut self, result: ConsensusResult) {
        self.decision_log.push(result);
    }

    /// Get decision log
    pub fn get_decision_log(&self) -> &Vec<ConsensusResult> {
        &self.decision_log
    }
}

impl Default for ConsensusState {
    fn default() -> Self {
        Self::new()
    }
}

impl ConsensusStatistics {
    /// Create new consensus statistics
    pub fn new() -> Self {
        Self {
            total_proposals: 0,
            accepted_proposals: 0,
            rejected_proposals: 0,
            timed_out_proposals: 0,
            avg_consensus_time: Duration::from_millis(0),
            throughput: 0.0,
        }
    }

    /// Get acceptance rate
    pub fn acceptance_rate(&self) -> f64 {
        if self.total_proposals > 0 {
            self.accepted_proposals as f64 / self.total_proposals as f64
        } else {
            0.0
        }
    }

    /// Get rejection rate
    pub fn rejection_rate(&self) -> f64 {
        if self.total_proposals > 0 {
            self.rejected_proposals as f64 / self.total_proposals as f64
        } else {
            0.0
        }
    }

    /// Get timeout rate
    pub fn timeout_rate(&self) -> f64 {
        if self.total_proposals > 0 {
            self.timed_out_proposals as f64 / self.total_proposals as f64
        } else {
            0.0
        }
    }

    /// Get completion rate
    pub fn completion_rate(&self) -> f64 {
        if self.total_proposals > 0 {
            let completed = self.accepted_proposals + self.rejected_proposals;
            completed as f64 / self.total_proposals as f64
        } else {
            0.0
        }
    }
}

impl Default for ConsensusStatistics {
    fn default() -> Self {
        Self::new()
    }
}

// Default implementations for configuration types

impl Default for ConsensusConfig {
    fn default() -> Self {
        Self {
            protocol: ConsensusProtocol::Raft,
            parameters: ConsensusParameters::default(),
            fault_tolerance: ConsensusFaultTolerance::default(),
            optimization: ConsensusOptimization::default(),
        }
    }
}

impl Default for ConsensusParameters {
    fn default() -> Self {
        Self {
            election_timeout: Duration::from_millis(150),
            heartbeat_interval: Duration::from_millis(50),
            commit_timeout: Duration::from_secs(10),
            quorum_size: 3,
            max_proposal_size: 1024 * 1024, // 1 MB
        }
    }
}

impl Default for ConsensusFaultTolerance {
    fn default() -> Self {
        Self {
            max_failures: 1,
            failure_detection_timeout: Duration::from_secs(30),
            recovery_strategy: ConsensusRecoveryStrategy::LeaderReelection,
        }
    }
}

impl Default for ConsensusOptimization {
    fn default() -> Self {
        Self {
            enable: true,
            batching: ConsensusBatching::default(),
            pipelining: ConsensusPipelining::default(),
            compression: ConsensusCompression::default(),
        }
    }
}

impl Default for ConsensusBatching {
    fn default() -> Self {
        Self {
            enable: true,
            batch_size: 100,
            timeout: Duration::from_millis(10),
        }
    }
}

impl Default for ConsensusPipelining {
    fn default() -> Self {
        Self {
            enable: true,
            depth: 10,
            timeout: Duration::from_millis(100),
        }
    }
}

impl Default for ConsensusCompression {
    fn default() -> Self {
        Self {
            enable: true,
            algorithm: CompressionAlgorithm::LZ4,
            threshold: 1024,
        }
    }
}

/// Consensus protocol builder for easy configuration
pub struct ConsensusConfigBuilder {
    config: ConsensusConfig,
}

impl ConsensusConfigBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: ConsensusConfig::default(),
        }
    }

    /// Set consensus protocol
    pub fn with_protocol(mut self, protocol: ConsensusProtocol) -> Self {
        self.config.protocol = protocol;
        self
    }

    /// Set election timeout
    pub fn with_election_timeout(mut self, timeout: Duration) -> Self {
        self.config.parameters.election_timeout = timeout;
        self
    }

    /// Set heartbeat interval
    pub fn with_heartbeat_interval(mut self, interval: Duration) -> Self {
        self.config.parameters.heartbeat_interval = interval;
        self
    }

    /// Set commit timeout
    pub fn with_commit_timeout(mut self, timeout: Duration) -> Self {
        self.config.parameters.commit_timeout = timeout;
        self
    }

    /// Set quorum size
    pub fn with_quorum_size(mut self, size: usize) -> Self {
        self.config.parameters.quorum_size = size;
        self
    }

    /// Set maximum proposal size
    pub fn with_max_proposal_size(mut self, size: usize) -> Self {
        self.config.parameters.max_proposal_size = size;
        self
    }

    /// Enable/disable optimization
    pub fn with_optimization(mut self, enable: bool) -> Self {
        self.config.optimization.enable = enable;
        self
    }

    /// Set batching configuration
    pub fn with_batching(mut self, enable: bool, batch_size: usize, timeout: Duration) -> Self {
        self.config.optimization.batching.enable = enable;
        self.config.optimization.batching.batch_size = batch_size;
        self.config.optimization.batching.timeout = timeout;
        self
    }

    /// Set pipelining configuration
    pub fn with_pipelining(mut self, enable: bool, depth: usize, timeout: Duration) -> Self {
        self.config.optimization.pipelining.enable = enable;
        self.config.optimization.pipelining.depth = depth;
        self.config.optimization.pipelining.timeout = timeout;
        self
    }

    /// Set compression configuration
    pub fn with_compression(mut self, enable: bool, algorithm: CompressionAlgorithm, threshold: usize) -> Self {
        self.config.optimization.compression.enable = enable;
        self.config.optimization.compression.algorithm = algorithm;
        self.config.optimization.compression.threshold = threshold;
        self
    }

    /// Set fault tolerance settings
    pub fn with_fault_tolerance(mut self, max_failures: usize, detection_timeout: Duration, recovery_strategy: ConsensusRecoveryStrategy) -> Self {
        self.config.fault_tolerance.max_failures = max_failures;
        self.config.fault_tolerance.failure_detection_timeout = detection_timeout;
        self.config.fault_tolerance.recovery_strategy = recovery_strategy;
        self
    }

    /// Build the final configuration
    pub fn build(self) -> ConsensusConfig {
        self.config
    }
}

impl Default for ConsensusConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration presets for common consensus scenarios
pub struct ConsensusPresets;

impl ConsensusPresets {
    /// Fast consensus configuration for low-latency scenarios
    pub fn fast() -> ConsensusConfig {
        ConsensusConfigBuilder::new()
            .with_election_timeout(Duration::from_millis(50))
            .with_heartbeat_interval(Duration::from_millis(10))
            .with_commit_timeout(Duration::from_secs(1))
            .with_batching(true, 50, Duration::from_millis(5))
            .with_pipelining(true, 5, Duration::from_millis(50))
            .build()
    }

    /// Reliable consensus configuration with strong consistency
    pub fn reliable() -> ConsensusConfig {
        ConsensusConfigBuilder::new()
            .with_election_timeout(Duration::from_millis(300))
            .with_heartbeat_interval(Duration::from_millis(100))
            .with_commit_timeout(Duration::from_secs(30))
            .with_quorum_size(5)
            .with_batching(true, 100, Duration::from_millis(20))
            .build()
    }

    /// High-throughput configuration for batch processing
    pub fn high_throughput() -> ConsensusConfig {
        ConsensusConfigBuilder::new()
            .with_batching(true, 1000, Duration::from_millis(50))
            .with_pipelining(true, 20, Duration::from_millis(100))
            .with_max_proposal_size(10 * 1024 * 1024) // 10 MB
            .build()
    }

    /// Byzantine fault tolerant configuration
    pub fn byzantine_fault_tolerant() -> ConsensusConfig {
        ConsensusConfigBuilder::new()
            .with_protocol(ConsensusProtocol::PBFT)
            .with_fault_tolerance(2, Duration::from_secs(10), ConsensusRecoveryStrategy::StateSynchronization)
            .build()
    }
}