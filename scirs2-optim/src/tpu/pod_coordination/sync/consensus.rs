use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};
use std::fmt;
use serde::{Serialize, Deserialize};
use tokio::sync::{oneshot, mpsc};
use async_trait::async_trait;

/// Consensus protocol identifier
pub type ConsensusId = u64;

/// Node identifier in consensus protocol
pub type NodeId = u64;

/// Term identifier for consensus protocols
pub type Term = u64;

/// Log index for consensus protocols
pub type LogIndex = u64;

/// Consensus entry identifier
pub type EntryId = u64;

/// Consensus operation result
pub type ConsensusResult<T> = Result<T, ConsensusError>;

/// Consensus protocol error types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsensusError {
    /// Protocol not supported
    ProtocolNotSupported(String),
    /// Node not found
    NodeNotFound(NodeId),
    /// Invalid term
    InvalidTerm(Term),
    /// Invalid log index
    InvalidLogIndex(LogIndex),
    /// Consensus timeout
    ConsensusTimeout,
    /// Network partition
    NetworkPartition,
    /// Insufficient nodes for quorum
    InsufficientQuorum,
    /// Leader election failed
    LeaderElectionFailed,
    /// Log replication failed
    LogReplicationFailed,
    /// State machine application failed
    StateMachineApplicationFailed,
    /// Configuration error
    ConfigurationError(String),
    /// Protocol specific error
    ProtocolError(String),
}

impl fmt::Display for ConsensusError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConsensusError::ProtocolNotSupported(protocol) => write!(f, "Protocol not supported: {}", protocol),
            ConsensusError::NodeNotFound(node_id) => write!(f, "Node not found: {}", node_id),
            ConsensusError::InvalidTerm(term) => write!(f, "Invalid term: {}", term),
            ConsensusError::InvalidLogIndex(index) => write!(f, "Invalid log index: {}", index),
            ConsensusError::ConsensusTimeout => write!(f, "Consensus timeout"),
            ConsensusError::NetworkPartition => write!(f, "Network partition detected"),
            ConsensusError::InsufficientQuorum => write!(f, "Insufficient nodes for quorum"),
            ConsensusError::LeaderElectionFailed => write!(f, "Leader election failed"),
            ConsensusError::LogReplicationFailed => write!(f, "Log replication failed"),
            ConsensusError::StateMachineApplicationFailed => write!(f, "State machine application failed"),
            ConsensusError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
            ConsensusError::ProtocolError(msg) => write!(f, "Protocol error: {}", msg),
        }
    }
}

impl std::error::Error for ConsensusError {}

/// Consensus protocol types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsensusProtocolType {
    /// Raft consensus protocol
    Raft,
    /// Practical Byzantine Fault Tolerance
    PBFT,
    /// Paxos consensus protocol
    Paxos,
    /// Multi-Paxos optimization
    MultiPaxos,
    /// Fast Paxos variant
    FastPaxos,
    /// Istanbul Byzantine Fault Tolerance
    IBFT,
    /// HotStuff BFT protocol
    HotStuff,
    /// Tendermint consensus
    Tendermint,
}

/// Node role in consensus protocol
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeRole {
    /// Leader node
    Leader,
    /// Follower node
    Follower,
    /// Candidate node (during election)
    Candidate,
    /// Primary node (PBFT)
    Primary,
    /// Backup node (PBFT)
    Backup,
    /// Proposer (Paxos)
    Proposer,
    /// Acceptor (Paxos)
    Acceptor,
    /// Learner (Paxos)
    Learner,
}

/// Consensus protocol state
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsensusState {
    /// Initializing protocol
    Initializing,
    /// Running normally
    Running,
    /// Leader election in progress
    ElectingLeader,
    /// Syncing with other nodes
    Syncing,
    /// Recovering from failure
    Recovering,
    /// Stopped
    Stopped,
    /// Error state
    Error(String),
}

/// Consensus log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusLogEntry {
    /// Entry identifier
    pub entry_id: EntryId,
    /// Log index
    pub index: LogIndex,
    /// Term when entry was created
    pub term: Term,
    /// Entry data
    pub data: Vec<u8>,
    /// Entry timestamp
    pub timestamp: SystemTime,
    /// Entry checksum
    pub checksum: u64,
}

/// Consensus vote
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusVote {
    /// Voter node ID
    pub voter_id: NodeId,
    /// Term being voted on
    pub term: Term,
    /// Candidate being voted for
    pub candidate_id: NodeId,
    /// Vote granted
    pub granted: bool,
    /// Vote timestamp
    pub timestamp: SystemTime,
}

/// Consensus quorum configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuorumConfig {
    /// Total number of nodes
    pub total_nodes: usize,
    /// Required quorum size
    pub quorum_size: usize,
    /// Byzantine fault tolerance
    pub byzantine_fault_tolerance: bool,
    /// Maximum byzantine faults tolerated
    pub max_byzantine_faults: usize,
}

impl QuorumConfig {
    /// Create new quorum configuration
    pub fn new(total_nodes: usize, byzantine_fault_tolerance: bool) -> Self {
        let (quorum_size, max_byzantine_faults) = if byzantine_fault_tolerance {
            // For BFT: quorum = 2f + 1, where f is max byzantine faults
            let max_faults = (total_nodes - 1) / 3;
            (2 * max_faults + 1, max_faults)
        } else {
            // For CFT: quorum = majority
            ((total_nodes / 2) + 1, 0)
        };

        Self {
            total_nodes,
            quorum_size,
            byzantine_fault_tolerance,
            max_byzantine_faults,
        }
    }

    /// Check if we have quorum
    pub fn has_quorum(&self, active_nodes: usize) -> bool {
        active_nodes >= self.quorum_size
    }
}

/// Consensus protocol configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusConfig {
    /// Protocol type
    pub protocol_type: ConsensusProtocolType,
    /// Node ID
    pub node_id: NodeId,
    /// Cluster nodes
    pub cluster_nodes: Vec<NodeId>,
    /// Quorum configuration
    pub quorum_config: QuorumConfig,
    /// Election timeout
    pub election_timeout: Duration,
    /// Heartbeat interval
    pub heartbeat_interval: Duration,
    /// Log replication timeout
    pub log_replication_timeout: Duration,
    /// Maximum log entries per message
    pub max_log_entries_per_message: usize,
    /// Log compaction threshold
    pub log_compaction_threshold: usize,
    /// Snapshot interval
    pub snapshot_interval: Duration,
    /// Recovery timeout
    pub recovery_timeout: Duration,
    /// Network timeout
    pub network_timeout: Duration,
    /// Maximum retries
    pub max_retries: usize,
    /// Enable pre-vote optimization
    pub enable_pre_vote: bool,
    /// Enable leadership transfer
    pub enable_leadership_transfer: bool,
    /// Enable batching
    pub enable_batching: bool,
    /// Batch size
    pub batch_size: usize,
    /// Batch timeout
    pub batch_timeout: Duration,
}

impl Default for ConsensusConfig {
    fn default() -> Self {
        Self {
            protocol_type: ConsensusProtocolType::Raft,
            node_id: 1,
            cluster_nodes: vec![1, 2, 3],
            quorum_config: QuorumConfig::new(3, false),
            election_timeout: Duration::from_millis(5000),
            heartbeat_interval: Duration::from_millis(1000),
            log_replication_timeout: Duration::from_millis(2000),
            max_log_entries_per_message: 100,
            log_compaction_threshold: 10000,
            snapshot_interval: Duration::from_secs(300),
            recovery_timeout: Duration::from_secs(30),
            network_timeout: Duration::from_millis(5000),
            max_retries: 3,
            enable_pre_vote: true,
            enable_leadership_transfer: true,
            enable_batching: true,
            batch_size: 50,
            batch_timeout: Duration::from_millis(100),
        }
    }
}

/// Consensus statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConsensusStatistics {
    /// Total consensus operations
    pub total_operations: u64,
    /// Successful operations
    pub successful_operations: u64,
    /// Failed operations
    pub failed_operations: u64,
    /// Leader elections
    pub leader_elections: u64,
    /// Log entries replicated
    pub log_entries_replicated: u64,
    /// Snapshots created
    pub snapshots_created: u64,
    /// Network messages sent
    pub network_messages_sent: u64,
    /// Network messages received
    pub network_messages_received: u64,
    /// Average consensus latency
    pub average_consensus_latency: Duration,
    /// Average log replication latency
    pub average_log_replication_latency: Duration,
    /// Current leader
    pub current_leader: Option<NodeId>,
    /// Current term
    pub current_term: Term,
    /// Log size
    pub log_size: usize,
    /// Last applied index
    pub last_applied_index: LogIndex,
    /// Commit index
    pub commit_index: LogIndex,
}

/// Consensus protocol trait
#[async_trait]
pub trait ConsensusProtocol: Send + Sync {
    /// Initialize the consensus protocol
    async fn initialize(&mut self, config: ConsensusConfig) -> ConsensusResult<()>;

    /// Start the consensus protocol
    async fn start(&mut self) -> ConsensusResult<()>;

    /// Stop the consensus protocol
    async fn stop(&mut self) -> ConsensusResult<()>;

    /// Propose a new value for consensus
    async fn propose(&mut self, data: Vec<u8>) -> ConsensusResult<EntryId>;

    /// Get the current state
    fn get_state(&self) -> ConsensusState;

    /// Get the current leader
    fn get_leader(&self) -> Option<NodeId>;

    /// Get the current term
    fn get_term(&self) -> Term;

    /// Get statistics
    fn get_statistics(&self) -> ConsensusStatistics;

    /// Handle incoming message
    async fn handle_message(&mut self, message: ConsensusMessage) -> ConsensusResult<()>;

    /// Trigger leader election
    async fn trigger_election(&mut self) -> ConsensusResult<()>;

    /// Transfer leadership
    async fn transfer_leadership(&mut self, target_node: NodeId) -> ConsensusResult<()>;

    /// Create snapshot
    async fn create_snapshot(&mut self) -> ConsensusResult<Vec<u8>>;

    /// Apply snapshot
    async fn apply_snapshot(&mut self, snapshot: Vec<u8>) -> ConsensusResult<()>;
}

/// Consensus message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsensusMessage {
    /// Vote request (Raft)
    VoteRequest {
        term: Term,
        candidate_id: NodeId,
        last_log_index: LogIndex,
        last_log_term: Term,
    },
    /// Vote response (Raft)
    VoteResponse {
        term: Term,
        vote_granted: bool,
    },
    /// Append entries (Raft)
    AppendEntries {
        term: Term,
        leader_id: NodeId,
        prev_log_index: LogIndex,
        prev_log_term: Term,
        entries: Vec<ConsensusLogEntry>,
        leader_commit: LogIndex,
    },
    /// Append entries response (Raft)
    AppendEntriesResponse {
        term: Term,
        success: bool,
        match_index: LogIndex,
    },
    /// Install snapshot (Raft)
    InstallSnapshot {
        term: Term,
        leader_id: NodeId,
        last_included_index: LogIndex,
        last_included_term: Term,
        data: Vec<u8>,
    },
    /// Install snapshot response (Raft)
    InstallSnapshotResponse {
        term: Term,
    },
    /// Pre-vote request (Raft optimization)
    PreVoteRequest {
        term: Term,
        candidate_id: NodeId,
        last_log_index: LogIndex,
        last_log_term: Term,
    },
    /// Pre-vote response (Raft optimization)
    PreVoteResponse {
        term: Term,
        vote_granted: bool,
    },
    /// PBFT prepare message
    PBFTPrepare {
        view: Term,
        sequence: LogIndex,
        digest: Vec<u8>,
        primary_id: NodeId,
    },
    /// PBFT commit message
    PBFTCommit {
        view: Term,
        sequence: LogIndex,
        digest: Vec<u8>,
        node_id: NodeId,
    },
    /// Paxos prepare message
    PaxosPrepare {
        proposal_number: u64,
        proposer_id: NodeId,
    },
    /// Paxos promise message
    PaxosPromise {
        proposal_number: u64,
        accepted_proposal: Option<u64>,
        accepted_value: Option<Vec<u8>>,
        acceptor_id: NodeId,
    },
    /// Paxos accept message
    PaxosAccept {
        proposal_number: u64,
        value: Vec<u8>,
        proposer_id: NodeId,
    },
    /// Paxos accepted message
    PaxosAccepted {
        proposal_number: u64,
        acceptor_id: NodeId,
    },
}

/// Raft consensus protocol implementation
#[derive(Debug)]
pub struct RaftConsensus {
    /// Configuration
    config: ConsensusConfig,
    /// Current state
    state: ConsensusState,
    /// Current role
    role: NodeRole,
    /// Current term
    current_term: Term,
    /// Voted for in current term
    voted_for: Option<NodeId>,
    /// Log entries
    log: Vec<ConsensusLogEntry>,
    /// Commit index
    commit_index: LogIndex,
    /// Last applied index
    last_applied: LogIndex,
    /// Next index for each follower
    next_index: HashMap<NodeId, LogIndex>,
    /// Match index for each follower
    match_index: HashMap<NodeId, LogIndex>,
    /// Current leader
    current_leader: Option<NodeId>,
    /// Statistics
    statistics: ConsensusStatistics,
    /// Last heartbeat time
    last_heartbeat: Instant,
    /// Election timer
    election_timer: Option<Instant>,
    /// Pending proposals
    pending_proposals: HashMap<EntryId, oneshot::Sender<ConsensusResult<()>>>,
    /// Next entry ID
    next_entry_id: EntryId,
}

impl RaftConsensus {
    /// Create new Raft consensus instance
    pub fn new() -> Self {
        Self {
            config: ConsensusConfig::default(),
            state: ConsensusState::Initializing,
            role: NodeRole::Follower,
            current_term: 0,
            voted_for: None,
            log: Vec::new(),
            commit_index: 0,
            last_applied: 0,
            next_index: HashMap::new(),
            match_index: HashMap::new(),
            current_leader: None,
            statistics: ConsensusStatistics::default(),
            last_heartbeat: Instant::now(),
            election_timer: None,
            pending_proposals: HashMap::new(),
            next_entry_id: 1,
        }
    }

    /// Start election timer
    fn start_election_timer(&mut self) {
        self.election_timer = Some(Instant::now() + self.config.election_timeout);
    }

    /// Check if election timeout has occurred
    fn check_election_timeout(&self) -> bool {
        if let Some(timer) = self.election_timer {
            Instant::now() >= timer
        } else {
            false
        }
    }

    /// Become candidate and start election
    async fn become_candidate(&mut self) -> ConsensusResult<()> {
        self.role = NodeRole::Candidate;
        self.current_term += 1;
        self.voted_for = Some(self.config.node_id);
        self.start_election_timer();
        self.statistics.leader_elections += 1;

        // Send vote requests to all other nodes
        for &node_id in &self.config.cluster_nodes {
            if node_id != self.config.node_id {
                // In a real implementation, send vote request message
            }
        }

        Ok(())
    }

    /// Become leader
    fn become_leader(&mut self) {
        self.role = NodeRole::Leader;
        self.current_leader = Some(self.config.node_id);
        self.election_timer = None;

        // Initialize next_index and match_index for all followers
        let next_index = self.log.len() as LogIndex + 1;
        for &node_id in &self.config.cluster_nodes {
            if node_id != self.config.node_id {
                self.next_index.insert(node_id, next_index);
                self.match_index.insert(node_id, 0);
            }
        }
    }

    /// Become follower
    fn become_follower(&mut self, term: Term) {
        self.role = NodeRole::Follower;
        self.current_term = term;
        self.voted_for = None;
        self.election_timer = None;
        self.last_heartbeat = Instant::now();
    }

    /// Append log entry
    fn append_log_entry(&mut self, entry: ConsensusLogEntry) {
        self.log.push(entry);
        self.statistics.log_entries_replicated += 1;
    }

    /// Get last log index
    fn last_log_index(&self) -> LogIndex {
        self.log.len() as LogIndex
    }

    /// Get last log term
    fn last_log_term(&self) -> Term {
        self.log.last().map(|entry| entry.term).unwrap_or(0)
    }
}

#[async_trait]
impl ConsensusProtocol for RaftConsensus {
    async fn initialize(&mut self, config: ConsensusConfig) -> ConsensusResult<()> {
        self.config = config;
        self.state = ConsensusState::Running;
        self.start_election_timer();
        Ok(())
    }

    async fn start(&mut self) -> ConsensusResult<()> {
        self.state = ConsensusState::Running;
        Ok(())
    }

    async fn stop(&mut self) -> ConsensusResult<()> {
        self.state = ConsensusState::Stopped;
        Ok(())
    }

    async fn propose(&mut self, data: Vec<u8>) -> ConsensusResult<EntryId> {
        if self.role != NodeRole::Leader {
            return Err(ConsensusError::ProtocolError("Only leader can propose".to_string()));
        }

        let entry_id = self.next_entry_id;
        self.next_entry_id += 1;

        let entry = ConsensusLogEntry {
            entry_id,
            index: self.last_log_index() + 1,
            term: self.current_term,
            data,
            timestamp: SystemTime::now(),
            checksum: 0, // Calculate actual checksum
        };

        self.append_log_entry(entry);
        self.statistics.total_operations += 1;

        Ok(entry_id)
    }

    fn get_state(&self) -> ConsensusState {
        self.state.clone()
    }

    fn get_leader(&self) -> Option<NodeId> {
        self.current_leader
    }

    fn get_term(&self) -> Term {
        self.current_term
    }

    fn get_statistics(&self) -> ConsensusStatistics {
        let mut stats = self.statistics.clone();
        stats.current_leader = self.current_leader;
        stats.current_term = self.current_term;
        stats.log_size = self.log.len();
        stats.last_applied_index = self.last_applied;
        stats.commit_index = self.commit_index;
        stats
    }

    async fn handle_message(&mut self, message: ConsensusMessage) -> ConsensusResult<()> {
        match message {
            ConsensusMessage::VoteRequest { term, candidate_id, last_log_index, last_log_term } => {
                if term > self.current_term {
                    self.become_follower(term);
                }

                let vote_granted = if term < self.current_term {
                    false
                } else if self.voted_for.is_some() && self.voted_for != Some(candidate_id) {
                    false
                } else {
                    // Check if candidate's log is at least as up-to-date as ours
                    let our_last_log_term = self.last_log_term();
                    let our_last_log_index = self.last_log_index();

                    last_log_term > our_last_log_term ||
                    (last_log_term == our_last_log_term && last_log_index >= our_last_log_index)
                };

                if vote_granted {
                    self.voted_for = Some(candidate_id);
                }

                // Send vote response
                // In real implementation, send VoteResponse message
            },
            ConsensusMessage::VoteResponse { term, vote_granted } => {
                if term > self.current_term {
                    self.become_follower(term);
                } else if term == self.current_term && self.role == NodeRole::Candidate && vote_granted {
                    // Count votes and become leader if majority
                    // In real implementation, track votes and check for majority
                    self.become_leader();
                }
            },
            ConsensusMessage::AppendEntries { term, leader_id, prev_log_index, prev_log_term, entries, leader_commit } => {
                if term >= self.current_term {
                    self.become_follower(term);
                    self.current_leader = Some(leader_id);
                    self.last_heartbeat = Instant::now();

                    // Log consistency check and append entries
                    // In real implementation, perform full consistency checks
                    for entry in entries {
                        self.append_log_entry(entry);
                    }

                    // Update commit index
                    if leader_commit > self.commit_index {
                        self.commit_index = std::cmp::min(leader_commit, self.last_log_index());
                    }
                }

                // Send append entries response
                // In real implementation, send AppendEntriesResponse message
            },
            _ => {
                // Handle other message types
            }
        }

        self.statistics.network_messages_received += 1;
        Ok(())
    }

    async fn trigger_election(&mut self) -> ConsensusResult<()> {
        if self.role != NodeRole::Leader {
            self.become_candidate().await?;
        }
        Ok(())
    }

    async fn transfer_leadership(&mut self, target_node: NodeId) -> ConsensusResult<()> {
        if self.role != NodeRole::Leader {
            return Err(ConsensusError::ProtocolError("Only leader can transfer leadership".to_string()));
        }

        if !self.config.cluster_nodes.contains(&target_node) {
            return Err(ConsensusError::NodeNotFound(target_node));
        }

        // In real implementation, perform leadership transfer protocol
        self.role = NodeRole::Follower;
        self.current_leader = Some(target_node);

        Ok(())
    }

    async fn create_snapshot(&mut self) -> ConsensusResult<Vec<u8>> {
        // In real implementation, create actual snapshot
        self.statistics.snapshots_created += 1;
        Ok(vec![])
    }

    async fn apply_snapshot(&mut self, _snapshot: Vec<u8>) -> ConsensusResult<()> {
        // In real implementation, apply snapshot to state machine
        Ok(())
    }
}

/// Consensus manager for TPU pod coordination
#[derive(Debug)]
pub struct ConsensusManager {
    /// Consensus configuration
    pub config: ConsensusConfig,
    /// Active consensus instances
    pub active_instances: Arc<RwLock<HashMap<ConsensusId, Box<dyn ConsensusProtocol>>>>,
    /// Consensus statistics
    pub statistics: Arc<Mutex<ConsensusStatistics>>,
    /// Protocol factory
    pub protocol_factory: ConsensusProtocolFactory,
    /// Message router
    pub message_router: ConsensusMessageRouter,
    /// Recovery manager
    pub recovery_manager: ConsensusRecoveryManager,
    /// Performance monitor
    pub performance_monitor: ConsensusPerformanceMonitor,
    /// Next consensus ID
    next_consensus_id: Arc<Mutex<ConsensusId>>,
}

impl ConsensusManager {
    /// Create new consensus manager
    pub fn new(config: ConsensusConfig) -> Self {
        Self {
            config,
            active_instances: Arc::new(RwLock::new(HashMap::new())),
            statistics: Arc::new(Mutex::new(ConsensusStatistics::default())),
            protocol_factory: ConsensusProtocolFactory::new(),
            message_router: ConsensusMessageRouter::new(),
            recovery_manager: ConsensusRecoveryManager::new(),
            performance_monitor: ConsensusPerformanceMonitor::new(),
            next_consensus_id: Arc::new(Mutex::new(1)),
        }
    }

    /// Create consensus instance
    pub async fn create_consensus(&self, protocol_type: ConsensusProtocolType) -> ConsensusResult<ConsensusId> {
        let consensus_id = {
            let mut next_id = self.next_consensus_id.lock().unwrap();
            let id = *next_id;
            *next_id += 1;
            id
        };

        let mut protocol = self.protocol_factory.create_protocol(protocol_type)?;
        protocol.initialize(self.config.clone()).await?;

        {
            let mut instances = self.active_instances.write().unwrap();
            instances.insert(consensus_id, protocol);
        }

        Ok(consensus_id)
    }

    /// Start consensus instance
    pub async fn start_consensus(&self, consensus_id: ConsensusId) -> ConsensusResult<()> {
        let instances = self.active_instances.read().unwrap();
        if let Some(protocol) = instances.get(&consensus_id) {
            // In real implementation, start the protocol
            Ok(())
        } else {
            Err(ConsensusError::ProtocolError(format!("Consensus instance {} not found", consensus_id)))
        }
    }

    /// Stop consensus instance
    pub async fn stop_consensus(&self, consensus_id: ConsensusId) -> ConsensusResult<()> {
        let instances = self.active_instances.read().unwrap();
        if let Some(protocol) = instances.get(&consensus_id) {
            // In real implementation, stop the protocol
            Ok(())
        } else {
            Err(ConsensusError::ProtocolError(format!("Consensus instance {} not found", consensus_id)))
        }
    }

    /// Propose value for consensus
    pub async fn propose(&self, consensus_id: ConsensusId, data: Vec<u8>) -> ConsensusResult<EntryId> {
        let instances = self.active_instances.read().unwrap();
        if let Some(protocol) = instances.get(&consensus_id) {
            // In real implementation, propose through the protocol
            Ok(1) // Placeholder
        } else {
            Err(ConsensusError::ProtocolError(format!("Consensus instance {} not found", consensus_id)))
        }
    }

    /// Get consensus statistics
    pub fn get_statistics(&self) -> ConsensusStatistics {
        let stats = self.statistics.lock().unwrap();
        stats.clone()
    }

    /// Handle consensus message
    pub async fn handle_message(&self, consensus_id: ConsensusId, message: ConsensusMessage) -> ConsensusResult<()> {
        self.message_router.route_message(consensus_id, message).await
    }

    /// Trigger recovery
    pub async fn trigger_recovery(&self, consensus_id: ConsensusId) -> ConsensusResult<()> {
        self.recovery_manager.trigger_recovery(consensus_id).await
    }

    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> HashMap<String, f64> {
        self.performance_monitor.get_metrics()
    }
}

/// Consensus protocol factory
#[derive(Debug)]
pub struct ConsensusProtocolFactory {
    /// Supported protocols
    supported_protocols: HashSet<ConsensusProtocolType>,
}

impl ConsensusProtocolFactory {
    /// Create new protocol factory
    pub fn new() -> Self {
        let mut supported_protocols = HashSet::new();
        supported_protocols.insert(ConsensusProtocolType::Raft);
        supported_protocols.insert(ConsensusProtocolType::PBFT);
        supported_protocols.insert(ConsensusProtocolType::Paxos);

        Self {
            supported_protocols,
        }
    }

    /// Create consensus protocol
    pub fn create_protocol(&self, protocol_type: ConsensusProtocolType) -> ConsensusResult<Box<dyn ConsensusProtocol>> {
        if !self.supported_protocols.contains(&protocol_type) {
            return Err(ConsensusError::ProtocolNotSupported(format!("{:?}", protocol_type)));
        }

        match protocol_type {
            ConsensusProtocolType::Raft => Ok(Box::new(RaftConsensus::new())),
            ConsensusProtocolType::PBFT => {
                // In real implementation, create PBFT instance
                Err(ConsensusError::ProtocolNotSupported("PBFT not implemented".to_string()))
            },
            ConsensusProtocolType::Paxos => {
                // In real implementation, create Paxos instance
                Err(ConsensusError::ProtocolNotSupported("Paxos not implemented".to_string()))
            },
            _ => Err(ConsensusError::ProtocolNotSupported(format!("{:?}", protocol_type))),
        }
    }
}

/// Consensus message router
#[derive(Debug)]
pub struct ConsensusMessageRouter {
    /// Message handlers
    handlers: HashMap<ConsensusId, mpsc::UnboundedSender<ConsensusMessage>>,
}

impl ConsensusMessageRouter {
    /// Create new message router
    pub fn new() -> Self {
        Self {
            handlers: HashMap::new(),
        }
    }

    /// Route message to consensus instance
    pub async fn route_message(&self, consensus_id: ConsensusId, message: ConsensusMessage) -> ConsensusResult<()> {
        if let Some(sender) = self.handlers.get(&consensus_id) {
            sender.send(message).map_err(|_| ConsensusError::ProtocolError("Failed to route message".to_string()))?;
            Ok(())
        } else {
            Err(ConsensusError::ProtocolError(format!("No handler for consensus {}", consensus_id)))
        }
    }
}

/// Consensus recovery manager
#[derive(Debug)]
pub struct ConsensusRecoveryManager {
    /// Recovery strategies
    recovery_strategies: HashMap<ConsensusProtocolType, Box<dyn ConsensusRecoveryStrategy>>,
}

impl ConsensusRecoveryManager {
    /// Create new recovery manager
    pub fn new() -> Self {
        Self {
            recovery_strategies: HashMap::new(),
        }
    }

    /// Trigger recovery for consensus instance
    pub async fn trigger_recovery(&self, consensus_id: ConsensusId) -> ConsensusResult<()> {
        // In real implementation, perform recovery based on protocol type
        Ok(())
    }
}

/// Consensus recovery strategy trait
pub trait ConsensusRecoveryStrategy: Send + Sync {
    /// Perform recovery
    fn recover(&self, consensus_id: ConsensusId) -> ConsensusResult<()>;
}

/// Consensus performance monitor
#[derive(Debug)]
pub struct ConsensusPerformanceMonitor {
    /// Performance metrics
    metrics: Arc<Mutex<HashMap<String, f64>>>,
    /// Last update time
    last_update: Arc<Mutex<Instant>>,
}

impl ConsensusPerformanceMonitor {
    /// Create new performance monitor
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(Mutex::new(HashMap::new())),
            last_update: Arc::new(Mutex::new(Instant::now())),
        }
    }

    /// Get performance metrics
    pub fn get_metrics(&self) -> HashMap<String, f64> {
        let metrics = self.metrics.lock().unwrap();
        metrics.clone()
    }

    /// Update metric
    pub fn update_metric(&self, name: String, value: f64) {
        let mut metrics = self.metrics.lock().unwrap();
        metrics.insert(name, value);

        let mut last_update = self.last_update.lock().unwrap();
        *last_update = Instant::now();
    }
}

/// Consensus configuration builder
#[derive(Debug, Default)]
pub struct ConsensusConfigBuilder {
    protocol_type: Option<ConsensusProtocolType>,
    node_id: Option<NodeId>,
    cluster_nodes: Option<Vec<NodeId>>,
    election_timeout: Option<Duration>,
    heartbeat_interval: Option<Duration>,
    enable_pre_vote: Option<bool>,
    enable_batching: Option<bool>,
    batch_size: Option<usize>,
}

impl ConsensusConfigBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set protocol type
    pub fn protocol_type(mut self, protocol_type: ConsensusProtocolType) -> Self {
        self.protocol_type = Some(protocol_type);
        self
    }

    /// Set node ID
    pub fn node_id(mut self, node_id: NodeId) -> Self {
        self.node_id = Some(node_id);
        self
    }

    /// Set cluster nodes
    pub fn cluster_nodes(mut self, cluster_nodes: Vec<NodeId>) -> Self {
        self.cluster_nodes = Some(cluster_nodes);
        self
    }

    /// Set election timeout
    pub fn election_timeout(mut self, timeout: Duration) -> Self {
        self.election_timeout = Some(timeout);
        self
    }

    /// Set heartbeat interval
    pub fn heartbeat_interval(mut self, interval: Duration) -> Self {
        self.heartbeat_interval = Some(interval);
        self
    }

    /// Enable pre-vote optimization
    pub fn enable_pre_vote(mut self, enable: bool) -> Self {
        self.enable_pre_vote = Some(enable);
        self
    }

    /// Enable batching
    pub fn enable_batching(mut self, enable: bool) -> Self {
        self.enable_batching = Some(enable);
        self
    }

    /// Set batch size
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = Some(size);
        self
    }

    /// Build configuration
    pub fn build(self) -> ConsensusResult<ConsensusConfig> {
        let protocol_type = self.protocol_type.unwrap_or(ConsensusProtocolType::Raft);
        let node_id = self.node_id.unwrap_or(1);
        let cluster_nodes = self.cluster_nodes.unwrap_or_else(|| vec![1, 2, 3]);

        let mut config = ConsensusConfig::default();
        config.protocol_type = protocol_type;
        config.node_id = node_id;
        config.cluster_nodes = cluster_nodes.clone();
        config.quorum_config = QuorumConfig::new(cluster_nodes.len(), false);

        if let Some(timeout) = self.election_timeout {
            config.election_timeout = timeout;
        }

        if let Some(interval) = self.heartbeat_interval {
            config.heartbeat_interval = interval;
        }

        if let Some(enable) = self.enable_pre_vote {
            config.enable_pre_vote = enable;
        }

        if let Some(enable) = self.enable_batching {
            config.enable_batching = enable;
        }

        if let Some(size) = self.batch_size {
            config.batch_size = size;
        }

        Ok(config)
    }
}

/// Consensus utilities
pub mod utils {
    use super::*;

    /// Calculate quorum size for given number of nodes
    pub fn calculate_quorum_size(total_nodes: usize, byzantine_fault_tolerance: bool) -> usize {
        if byzantine_fault_tolerance {
            // For BFT: need 2f + 1 nodes, where f is max byzantine faults
            let max_faults = (total_nodes - 1) / 3;
            2 * max_faults + 1
        } else {
            // For CFT: need majority
            (total_nodes / 2) + 1
        }
    }

    /// Check if term is valid
    pub fn is_valid_term(term: Term) -> bool {
        term > 0
    }

    /// Check if log index is valid
    pub fn is_valid_log_index(index: LogIndex) -> bool {
        index > 0
    }

    /// Calculate message hash
    pub fn calculate_message_hash(message: &ConsensusMessage) -> u64 {
        // In real implementation, use proper hash function
        0
    }

    /// Validate consensus configuration
    pub fn validate_config(config: &ConsensusConfig) -> ConsensusResult<()> {
        if config.cluster_nodes.is_empty() {
            return Err(ConsensusError::ConfigurationError("Cluster nodes cannot be empty".to_string()));
        }

        if !config.cluster_nodes.contains(&config.node_id) {
            return Err(ConsensusError::ConfigurationError("Node ID must be in cluster nodes".to_string()));
        }

        if config.election_timeout <= config.heartbeat_interval {
            return Err(ConsensusError::ConfigurationError("Election timeout must be greater than heartbeat interval".to_string()));
        }

        Ok(())
    }
}

/// Consensus testing utilities
#[cfg(test)]
pub mod testing {
    use super::*;

    /// Create test consensus configuration
    pub fn create_test_config() -> ConsensusConfig {
        ConsensusConfigBuilder::new()
            .protocol_type(ConsensusProtocolType::Raft)
            .node_id(1)
            .cluster_nodes(vec![1, 2, 3])
            .election_timeout(Duration::from_millis(1000))
            .heartbeat_interval(Duration::from_millis(200))
            .build()
            .unwrap()
    }

    /// Create test consensus manager
    pub fn create_test_manager() -> ConsensusManager {
        ConsensusManager::new(create_test_config())
    }

    /// Create test message
    pub fn create_test_vote_request() -> ConsensusMessage {
        ConsensusMessage::VoteRequest {
            term: 1,
            candidate_id: 1,
            last_log_index: 0,
            last_log_term: 0,
        }
    }
}