//! Raft Consensus Protocol Implementation
//!
//! This module provides a complete implementation of the Raft consensus algorithm
//! for distributed systems, including leader election, log replication, safety
//! mechanisms, and cluster membership changes.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, BTreeMap, VecDeque};
use std::time::{Duration, Instant};
use crate::tpu::pod_coordination::types::*;

/// Raft consensus configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaftConfig {
    /// Node identification
    pub node_config: RaftNodeConfig,
    /// Timing configuration
    pub timing_config: RaftTimingConfig,
    /// Log configuration
    pub log_config: RaftLogConfig,
    /// Snapshot configuration
    pub snapshot_config: RaftSnapshotConfig,
    /// Network configuration
    pub network_config: RaftNetworkConfig,
    /// Safety configuration
    pub safety_config: RaftSafetyConfig,
    /// Performance optimization
    pub performance_config: RaftPerformanceConfig,
    /// Monitoring configuration
    pub monitoring_config: RaftMonitoringConfig,
}

/// Raft node configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaftNodeConfig {
    /// Node identifier
    pub node_id: NodeId,
    /// Cluster members
    pub cluster_members: Vec<NodeId>,
    /// Node address
    pub node_address: String,
    /// Node role preferences
    pub role_preferences: RaftRolePreferences,
    /// Node capabilities
    pub node_capabilities: RaftNodeCapabilities,
}

/// Raft timing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaftTimingConfig {
    /// Election timeout range
    pub election_timeout_range: (Duration, Duration),
    /// Heartbeat interval
    pub heartbeat_interval: Duration,
    /// RPC timeout
    pub rpc_timeout: Duration,
    /// Snapshot timeout
    pub snapshot_timeout: Duration,
    /// Batch timeout
    pub batch_timeout: Duration,
    /// Leader lease duration
    pub leader_lease_duration: Duration,
}

/// Raft log configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaftLogConfig {
    /// Maximum log entries per batch
    pub max_entries_per_batch: usize,
    /// Log compaction threshold
    pub compaction_threshold: usize,
    /// Log storage backend
    pub storage_backend: LogStorageBackend,
    /// Log persistence settings
    pub persistence_settings: LogPersistenceSettings,
    /// Log validation settings
    pub validation_settings: LogValidationSettings,
    /// Log indexing settings
    pub indexing_settings: LogIndexingSettings,
}

/// Raft snapshot configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaftSnapshotConfig {
    /// Snapshot frequency
    pub snapshot_frequency: SnapshotFrequency,
    /// Snapshot compression
    pub compression_enabled: bool,
    /// Snapshot storage backend
    pub storage_backend: SnapshotStorageBackend,
    /// Snapshot validation
    pub validation_enabled: bool,
    /// Incremental snapshots
    pub incremental_snapshots: bool,
    /// Snapshot retention policy
    pub retention_policy: SnapshotRetentionPolicy,
}

/// Raft network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaftNetworkConfig {
    /// Transport protocol
    pub transport_protocol: TransportProtocol,
    /// Connection pooling
    pub connection_pooling: ConnectionPoolingConfig,
    /// Message batching
    pub message_batching: MessageBatchingConfig,
    /// Network compression
    pub compression_config: NetworkCompressionConfig,
    /// Failure detection
    pub failure_detection: NetworkFailureDetectionConfig,
    /// Load balancing
    pub load_balancing: NetworkLoadBalancingConfig,
}

/// Raft safety configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaftSafetyConfig {
    /// Byzantine fault tolerance
    pub byzantine_fault_tolerance: ByzantineFaultToleranceConfig,
    /// Membership change safety
    pub membership_change_safety: MembershipChangeSafetyConfig,
    /// Log safety guarantees
    pub log_safety_guarantees: LogSafetyGuaranteesConfig,
    /// Leader safety mechanisms
    pub leader_safety_mechanisms: LeaderSafetyMechanismsConfig,
    /// Network partition handling
    pub partition_handling: PartitionHandlingConfig,
}

/// Raft performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaftPerformanceConfig {
    /// Parallel processing
    pub parallel_processing: ParallelProcessingConfig,
    /// Caching configuration
    pub caching_config: CachingConfig,
    /// Optimization strategies
    pub optimization_strategies: OptimizationStrategiesConfig,
    /// Resource management
    pub resource_management: ResourceManagementConfig,
    /// Performance tuning
    pub performance_tuning: PerformanceTuningConfig,
}

/// Raft monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaftMonitoringConfig {
    /// Metrics collection
    pub metrics_collection: MetricsCollectionConfig,
    /// Health monitoring
    pub health_monitoring: HealthMonitoringConfig,
    /// Performance monitoring
    pub performance_monitoring: PerformanceMonitoringConfig,
    /// Alert configuration
    pub alert_configuration: AlertConfiguration,
    /// Logging configuration
    pub logging_configuration: LoggingConfiguration,
}

/// Raft consensus state machine
#[derive(Debug)]
pub struct RaftConsensus {
    /// Configuration
    pub config: RaftConfig,
    /// Current state
    pub state: RaftState,
    /// Persistent state
    pub persistent_state: PersistentState,
    /// Volatile state
    pub volatile_state: VolatileState,
    /// Leader-specific state
    pub leader_state: Option<LeaderState>,
    /// Follower-specific state
    pub follower_state: Option<FollowerState>,
    /// Candidate-specific state
    pub candidate_state: Option<CandidateState>,
    /// Log storage
    pub log_storage: LogStorage,
    /// Snapshot storage
    pub snapshot_storage: SnapshotStorage,
    /// Network interface
    pub network: RaftNetwork,
    /// Statistics
    pub statistics: RaftStatistics,
}

/// Raft node state
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RaftState {
    /// Follower state
    Follower,
    /// Candidate state
    Candidate,
    /// Leader state
    Leader,
    /// Observer state (non-voting)
    Observer,
    /// Inactive state
    Inactive,
}

/// Persistent state (survives crashes)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistentState {
    /// Current term
    pub current_term: Term,
    /// Voted for in current term
    pub voted_for: Option<NodeId>,
    /// Log entries
    pub log: Vec<LogEntry>,
    /// Last applied index
    pub last_applied: LogIndex,
    /// Commit index
    pub commit_index: LogIndex,
    /// Configuration state
    pub configuration_state: ConfigurationState,
}

/// Volatile state (lost on crash)
#[derive(Debug, Clone)]
pub struct VolatileState {
    /// Current leader
    pub current_leader: Option<NodeId>,
    /// Election timeout
    pub election_timeout: Instant,
    /// Last heartbeat received
    pub last_heartbeat: Option<Instant>,
    /// Pending client requests
    pub pending_requests: HashMap<RequestId, ClientRequest>,
    /// Message queues
    pub message_queues: MessageQueues,
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
}

/// Leader-specific state
#[derive(Debug, Clone)]
pub struct LeaderState {
    /// Next index for each follower
    pub next_index: HashMap<NodeId, LogIndex>,
    /// Match index for each follower
    pub match_index: HashMap<NodeId, LogIndex>,
    /// Heartbeat timers
    pub heartbeat_timers: HashMap<NodeId, Instant>,
    /// Pending append entries
    pub pending_append_entries: HashMap<NodeId, VecDeque<AppendEntriesRequest>>,
    /// Leader lease
    pub leader_lease: LeaderLease,
    /// Client session management
    pub client_sessions: ClientSessionManager,
}

/// Follower-specific state
#[derive(Debug, Clone)]
pub struct FollowerState {
    /// Follower metrics
    pub follower_metrics: FollowerMetrics,
    /// Recovery state
    pub recovery_state: RecoveryState,
    /// Catch-up state
    pub catchup_state: CatchUpState,
}

/// Candidate-specific state
#[derive(Debug, Clone)]
pub struct CandidateState {
    /// Votes received
    pub votes_received: HashMap<NodeId, VoteResponse>,
    /// Vote requests sent
    pub vote_requests_sent: HashMap<NodeId, Instant>,
    /// Election start time
    pub election_start_time: Instant,
    /// Pre-vote state (for pre-vote optimization)
    pub pre_vote_state: PreVoteState,
}

/// Log entry structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    /// Entry index
    pub index: LogIndex,
    /// Entry term
    pub term: Term,
    /// Entry type
    pub entry_type: LogEntryType,
    /// Entry data
    pub data: Vec<u8>,
    /// Entry metadata
    pub metadata: LogEntryMetadata,
    /// Entry checksum
    pub checksum: u64,
    /// Timestamp
    pub timestamp: Instant,
}

/// Log entry types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogEntryType {
    /// Application data entry
    Application,
    /// Configuration change entry
    Configuration,
    /// No-op entry (for new leaders)
    NoOp,
    /// Snapshot marker
    Snapshot,
    /// Membership change
    MembershipChange,
    /// Custom entry type
    Custom(String),
}

/// Log entry metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntryMetadata {
    /// Client identifier
    pub client_id: Option<String>,
    /// Request identifier
    pub request_id: Option<RequestId>,
    /// Entry size
    pub entry_size: usize,
    /// Compression used
    pub compression: Option<CompressionType>,
    /// Entry tags
    pub tags: HashMap<String, String>,
}

/// Raft RPC messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RaftMessage {
    /// Vote request message
    VoteRequest(VoteRequest),
    /// Vote response message
    VoteResponse(VoteResponse),
    /// Append entries request message
    AppendEntriesRequest(AppendEntriesRequest),
    /// Append entries response message
    AppendEntriesResponse(AppendEntriesResponse),
    /// Install snapshot request message
    InstallSnapshotRequest(InstallSnapshotRequest),
    /// Install snapshot response message
    InstallSnapshotResponse(InstallSnapshotResponse),
    /// Pre-vote request (optimization)
    PreVoteRequest(PreVoteRequest),
    /// Pre-vote response (optimization)
    PreVoteResponse(PreVoteResponse),
    /// Heartbeat message
    Heartbeat(HeartbeatMessage),
    /// Client request message
    ClientRequest(ClientRequest),
    /// Client response message
    ClientResponse(ClientResponse),
}

/// Vote request RPC
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoteRequest {
    /// Candidate's term
    pub term: Term,
    /// Candidate requesting vote
    pub candidate_id: NodeId,
    /// Index of candidate's last log entry
    pub last_log_index: LogIndex,
    /// Term of candidate's last log entry
    pub last_log_term: Term,
    /// Pre-vote flag
    pub is_pre_vote: bool,
    /// Leadership transfer flag
    pub is_leadership_transfer: bool,
}

/// Vote response RPC
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoteResponse {
    /// Current term
    pub term: Term,
    /// Vote granted flag
    pub vote_granted: bool,
    /// Rejection reason
    pub rejection_reason: Option<VoteRejectionReason>,
    /// Node metrics
    pub node_metrics: Option<NodeMetrics>,
}

/// Append entries request RPC
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppendEntriesRequest {
    /// Leader's term
    pub term: Term,
    /// Leader identifier
    pub leader_id: NodeId,
    /// Index of log entry immediately preceding new ones
    pub prev_log_index: LogIndex,
    /// Term of prev_log_index entry
    pub prev_log_term: Term,
    /// Log entries to store
    pub entries: Vec<LogEntry>,
    /// Leader's commit index
    pub leader_commit: LogIndex,
    /// Is heartbeat flag
    pub is_heartbeat: bool,
    /// Batch information
    pub batch_info: Option<BatchInfo>,
}

/// Append entries response RPC
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppendEntriesResponse {
    /// Current term
    pub term: Term,
    /// Success flag
    pub success: bool,
    /// Match index (for successful responses)
    pub match_index: Option<LogIndex>,
    /// Conflict information (for failed responses)
    pub conflict_info: Option<ConflictInfo>,
    /// Node metrics
    pub node_metrics: Option<NodeMetrics>,
}

/// Install snapshot request RPC
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstallSnapshotRequest {
    /// Leader's term
    pub term: Term,
    /// Leader identifier
    pub leader_id: NodeId,
    /// Last included index
    pub last_included_index: LogIndex,
    /// Last included term
    pub last_included_term: Term,
    /// Byte offset in snapshot file
    pub offset: u64,
    /// Raw bytes of snapshot chunk
    pub data: Vec<u8>,
    /// Is last chunk flag
    pub done: bool,
    /// Snapshot metadata
    pub snapshot_metadata: SnapshotMetadata,
}

/// Install snapshot response RPC
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstallSnapshotResponse {
    /// Current term
    pub term: Term,
    /// Bytes processed
    pub bytes_processed: u64,
    /// Success flag
    pub success: bool,
    /// Error information
    pub error: Option<SnapshotError>,
}

/// Client request message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientRequest {
    /// Request identifier
    pub request_id: RequestId,
    /// Client identifier
    pub client_id: String,
    /// Request data
    pub data: Vec<u8>,
    /// Request type
    pub request_type: ClientRequestType,
    /// Request timestamp
    pub timestamp: Instant,
    /// Request metadata
    pub metadata: HashMap<String, String>,
}

/// Client response message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientResponse {
    /// Request identifier
    pub request_id: RequestId,
    /// Success flag
    pub success: bool,
    /// Response data
    pub data: Option<Vec<u8>>,
    /// Error information
    pub error: Option<ClientError>,
    /// Leader hint
    pub leader_hint: Option<NodeId>,
    /// Response metadata
    pub metadata: HashMap<String, String>,
}

/// Raft statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaftStatistics {
    /// Election statistics
    pub election_stats: ElectionStatistics,
    /// Log replication statistics
    pub replication_stats: ReplicationStatistics,
    /// Snapshot statistics
    pub snapshot_stats: SnapshotStatistics,
    /// Performance statistics
    pub performance_stats: RaftPerformanceStatistics,
    /// Network statistics
    pub network_stats: NetworkStatistics,
    /// Client statistics
    pub client_stats: ClientStatistics,
}

/// Election statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElectionStatistics {
    /// Total elections started
    pub elections_started: u64,
    /// Elections won
    pub elections_won: u64,
    /// Elections lost
    pub elections_lost: u64,
    /// Average election duration
    pub average_election_duration: Duration,
    /// Split vote occurrences
    pub split_vote_count: u64,
    /// Leadership transfers
    pub leadership_transfers: u64,
}

/// Replication statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicationStatistics {
    /// Total entries replicated
    pub entries_replicated: u64,
    /// Replication conflicts
    pub replication_conflicts: u64,
    /// Average replication latency
    pub average_replication_latency: Duration,
    /// Batch statistics
    pub batch_stats: BatchStatistics,
    /// Follower lag statistics
    pub follower_lag_stats: HashMap<NodeId, FollowerLagStats>,
}

/// Implementation of Raft consensus
impl RaftConsensus {
    /// Create a new Raft consensus instance
    pub fn new(config: RaftConfig) -> Self {
        let node_id = config.node_config.node_id.clone();

        Self {
            config,
            state: RaftState::Follower,
            persistent_state: PersistentState {
                current_term: 0,
                voted_for: None,
                log: Vec::new(),
                last_applied: 0,
                commit_index: 0,
                configuration_state: ConfigurationState::Stable,
            },
            volatile_state: VolatileState {
                current_leader: None,
                election_timeout: Self::random_election_timeout(),
                last_heartbeat: None,
                pending_requests: HashMap::new(),
                message_queues: MessageQueues::new(),
                performance_metrics: PerformanceMetrics::default(),
            },
            leader_state: None,
            follower_state: Some(FollowerState {
                follower_metrics: FollowerMetrics::default(),
                recovery_state: RecoveryState::Normal,
                catchup_state: CatchUpState::Normal,
            }),
            candidate_state: None,
            log_storage: LogStorage::new(node_id.clone()),
            snapshot_storage: SnapshotStorage::new(node_id.clone()),
            network: RaftNetwork::new(node_id),
            statistics: RaftStatistics::default(),
        }
    }

    /// Start the Raft consensus algorithm
    pub fn start(&mut self) -> Result<()> {
        // Initialize network
        self.network.start()?;

        // Start election timer
        self.reset_election_timer();

        // Initialize log storage
        self.log_storage.initialize()?;

        // Initialize snapshot storage
        self.snapshot_storage.initialize()?;

        // Load persistent state
        self.load_persistent_state()?;

        // Start background tasks
        self.start_background_tasks()?;

        Ok(())
    }

    /// Process incoming message
    pub fn process_message(&mut self, message: RaftMessage) -> Result<()> {
        match message {
            RaftMessage::VoteRequest(request) => self.handle_vote_request(request),
            RaftMessage::VoteResponse(response) => self.handle_vote_response(response),
            RaftMessage::AppendEntriesRequest(request) => self.handle_append_entries_request(request),
            RaftMessage::AppendEntriesResponse(response) => self.handle_append_entries_response(response),
            RaftMessage::InstallSnapshotRequest(request) => self.handle_install_snapshot_request(request),
            RaftMessage::InstallSnapshotResponse(response) => self.handle_install_snapshot_response(response),
            RaftMessage::PreVoteRequest(request) => self.handle_pre_vote_request(request),
            RaftMessage::PreVoteResponse(response) => self.handle_pre_vote_response(response),
            RaftMessage::Heartbeat(heartbeat) => self.handle_heartbeat(heartbeat),
            RaftMessage::ClientRequest(request) => self.handle_client_request(request),
            RaftMessage::ClientResponse(_response) => Ok(()), // Handled by client
        }
    }

    /// Handle vote request
    fn handle_vote_request(&mut self, request: VoteRequest) -> Result<()> {
        let mut vote_granted = false;
        let mut rejection_reason = None;

        // Check term
        if request.term > self.persistent_state.current_term {
            self.become_follower(request.term)?;
        }

        // Grant vote if conditions are met
        if request.term == self.persistent_state.current_term {
            let can_vote = self.persistent_state.voted_for.is_none() ||
                          self.persistent_state.voted_for == Some(request.candidate_id.clone());

            let log_up_to_date = self.is_log_up_to_date(
                request.last_log_index,
                request.last_log_term
            );

            if can_vote && log_up_to_date {
                vote_granted = true;
                self.persistent_state.voted_for = Some(request.candidate_id.clone());
                self.reset_election_timer();
                self.save_persistent_state()?;
            } else {
                rejection_reason = Some(if !can_vote {
                    VoteRejectionReason::AlreadyVoted
                } else {
                    VoteRejectionReason::LogNotUpToDate
                });
            }
        } else {
            rejection_reason = Some(VoteRejectionReason::StaleTerm);
        }

        // Send vote response
        let response = VoteResponse {
            term: self.persistent_state.current_term,
            vote_granted,
            rejection_reason,
            node_metrics: Some(self.get_node_metrics()),
        };

        self.network.send_message(
            &request.candidate_id,
            RaftMessage::VoteResponse(response)
        )?;

        Ok(())
    }

    /// Handle vote response
    fn handle_vote_response(&mut self, response: VoteResponse) -> Result<()> {
        // Only candidates process vote responses
        if self.state != RaftState::Candidate {
            return Ok(());
        }

        // Check term
        if response.term > self.persistent_state.current_term {
            self.become_follower(response.term)?;
            return Ok(());
        }

        // Process vote if from current term
        if response.term == self.persistent_state.current_term {
            if let Some(ref mut candidate_state) = self.candidate_state {
                candidate_state.votes_received.insert(
                    response.voter_id.clone(),
                    response
                );

                // Check if we have majority
                let votes_for = candidate_state.votes_received
                    .values()
                    .filter(|v| v.vote_granted)
                    .count();

                let cluster_size = self.config.node_config.cluster_members.len();
                let majority = cluster_size / 2 + 1;

                if votes_for >= majority {
                    self.become_leader()?;
                }
            }
        }

        Ok(())
    }

    /// Handle append entries request
    fn handle_append_entries_request(&mut self, request: AppendEntriesRequest) -> Result<()> {
        let mut success = false;
        let mut match_index = None;
        let mut conflict_info = None;

        // Check term
        if request.term > self.persistent_state.current_term {
            self.become_follower(request.term)?;
        }

        // Process request if from current term
        if request.term == self.persistent_state.current_term {
            // Update leader and reset election timer
            self.volatile_state.current_leader = Some(request.leader_id.clone());
            self.reset_election_timer();

            // Check log consistency
            if self.is_log_consistent(request.prev_log_index, request.prev_log_term) {
                // Append entries
                if !request.entries.is_empty() {
                    self.append_entries(&request.entries)?;
                }

                // Update commit index
                if request.leader_commit > self.persistent_state.commit_index {
                    self.persistent_state.commit_index = std::cmp::min(
                        request.leader_commit,
                        self.get_last_log_index()
                    );
                    self.apply_committed_entries()?;
                }

                success = true;
                match_index = Some(self.get_last_log_index());
            } else {
                // Log inconsistency - provide conflict information
                conflict_info = Some(self.get_conflict_info(request.prev_log_index));
            }
        }

        // Send append entries response
        let response = AppendEntriesResponse {
            term: self.persistent_state.current_term,
            success,
            match_index,
            conflict_info,
            node_metrics: Some(self.get_node_metrics()),
        };

        self.network.send_message(
            &request.leader_id,
            RaftMessage::AppendEntriesResponse(response)
        )?;

        Ok(())
    }

    /// Handle append entries response
    fn handle_append_entries_response(&mut self, response: AppendEntriesResponse) -> Result<()> {
        // Only leaders process append entries responses
        if self.state != RaftState::Leader {
            return Ok(());
        }

        // Check term
        if response.term > self.persistent_state.current_term {
            self.become_follower(response.term)?;
            return Ok(());
        }

        // Process response if from current term
        if response.term == self.persistent_state.current_term {
            if let Some(ref mut leader_state) = self.leader_state {
                let follower_id = &response.follower_id;

                if response.success {
                    // Update next_index and match_index
                    if let Some(match_index) = response.match_index {
                        leader_state.next_index.insert(follower_id.clone(), match_index + 1);
                        leader_state.match_index.insert(follower_id.clone(), match_index);

                        // Check if we can advance commit index
                        self.advance_commit_index()?;
                    }
                } else {
                    // Handle log inconsistency
                    self.handle_log_inconsistency(follower_id, &response)?;
                }
            }
        }

        Ok(())
    }

    /// Handle client request
    fn handle_client_request(&mut self, request: ClientRequest) -> Result<()> {
        if self.state != RaftState::Leader {
            // Redirect to leader
            let response = ClientResponse {
                request_id: request.request_id,
                success: false,
                data: None,
                error: Some(ClientError::NotLeader),
                leader_hint: self.volatile_state.current_leader.clone(),
                metadata: HashMap::new(),
            };

            self.network.send_client_response(response)?;
            return Ok(());
        }

        // Create log entry
        let log_entry = LogEntry {
            index: self.get_last_log_index() + 1,
            term: self.persistent_state.current_term,
            entry_type: LogEntryType::Application,
            data: request.data,
            metadata: LogEntryMetadata {
                client_id: Some(request.client_id),
                request_id: Some(request.request_id),
                entry_size: request.data.len(),
                compression: None,
                tags: request.metadata,
            },
            checksum: self.calculate_checksum(&request.data),
            timestamp: Instant::now(),
        };

        // Append to log
        self.persistent_state.log.push(log_entry);
        self.save_persistent_state()?;

        // Add to pending requests
        self.volatile_state.pending_requests.insert(
            request.request_id,
            request
        );

        // Start replication
        self.replicate_to_followers()?;

        Ok(())
    }

    /// Become follower
    fn become_follower(&mut self, term: Term) -> Result<()> {
        self.state = RaftState::Follower;
        self.persistent_state.current_term = term;
        self.persistent_state.voted_for = None;
        self.leader_state = None;
        self.candidate_state = None;

        self.follower_state = Some(FollowerState {
            follower_metrics: FollowerMetrics::default(),
            recovery_state: RecoveryState::Normal,
            catchup_state: CatchUpState::Normal,
        });

        self.reset_election_timer();
        self.save_persistent_state()?;

        Ok(())
    }

    /// Become candidate
    fn become_candidate(&mut self) -> Result<()> {
        self.state = RaftState::Candidate;
        self.persistent_state.current_term += 1;
        self.persistent_state.voted_for = Some(self.config.node_config.node_id.clone());

        self.follower_state = None;
        self.leader_state = None;

        let election_start_time = Instant::now();
        self.candidate_state = Some(CandidateState {
            votes_received: HashMap::new(),
            vote_requests_sent: HashMap::new(),
            election_start_time,
            pre_vote_state: PreVoteState::None,
        });

        // Vote for self
        if let Some(ref mut candidate_state) = self.candidate_state {
            candidate_state.votes_received.insert(
                self.config.node_config.node_id.clone(),
                VoteResponse {
                    term: self.persistent_state.current_term,
                    vote_granted: true,
                    rejection_reason: None,
                    node_metrics: Some(self.get_node_metrics()),
                }
            );
        }

        self.reset_election_timer();
        self.save_persistent_state()?;

        // Send vote requests to all other nodes
        self.send_vote_requests()?;

        Ok(())
    }

    /// Become leader
    fn become_leader(&mut self) -> Result<()> {
        self.state = RaftState::Leader;
        self.candidate_state = None;
        self.follower_state = None;

        // Initialize leader state
        let last_log_index = self.get_last_log_index();
        let mut next_index = HashMap::new();
        let mut match_index = HashMap::new();

        for member in &self.config.node_config.cluster_members {
            if *member != self.config.node_config.node_id {
                next_index.insert(member.clone(), last_log_index + 1);
                match_index.insert(member.clone(), 0);
            }
        }

        self.leader_state = Some(LeaderState {
            next_index,
            match_index,
            heartbeat_timers: HashMap::new(),
            pending_append_entries: HashMap::new(),
            leader_lease: LeaderLease::new(),
            client_sessions: ClientSessionManager::new(),
        });

        // Send initial heartbeats
        self.send_heartbeats()?;

        // Append no-op entry
        self.append_no_op_entry()?;

        Ok(())
    }

    /// Send vote requests
    fn send_vote_requests(&mut self) -> Result<()> {
        let vote_request = VoteRequest {
            term: self.persistent_state.current_term,
            candidate_id: self.config.node_config.node_id.clone(),
            last_log_index: self.get_last_log_index(),
            last_log_term: self.get_last_log_term(),
            is_pre_vote: false,
            is_leadership_transfer: false,
        };

        for member in &self.config.node_config.cluster_members {
            if *member != self.config.node_config.node_id {
                self.network.send_message(
                    member,
                    RaftMessage::VoteRequest(vote_request.clone())
                )?;

                if let Some(ref mut candidate_state) = self.candidate_state {
                    candidate_state.vote_requests_sent.insert(
                        member.clone(),
                        Instant::now()
                    );
                }
            }
        }

        Ok(())
    }

    /// Send heartbeats to all followers
    fn send_heartbeats(&mut self) -> Result<()> {
        if self.state != RaftState::Leader {
            return Ok(());
        }

        let heartbeat_time = Instant::now();

        for member in &self.config.node_config.cluster_members {
            if *member != self.config.node_config.node_id {
                self.send_heartbeat_to_follower(member, heartbeat_time)?;
            }
        }

        Ok(())
    }

    /// Send heartbeat to specific follower
    fn send_heartbeat_to_follower(&mut self, follower_id: &NodeId, heartbeat_time: Instant) -> Result<()> {
        if let Some(ref mut leader_state) = self.leader_state {
            let next_index = leader_state.next_index.get(follower_id).cloned().unwrap_or(1);
            let prev_log_index = if next_index > 1 { next_index - 1 } else { 0 };
            let prev_log_term = if prev_log_index > 0 {
                self.get_log_term(prev_log_index).unwrap_or(0)
            } else {
                0
            };

            let request = AppendEntriesRequest {
                term: self.persistent_state.current_term,
                leader_id: self.config.node_config.node_id.clone(),
                prev_log_index,
                prev_log_term,
                entries: Vec::new(), // Heartbeat - no entries
                leader_commit: self.persistent_state.commit_index,
                is_heartbeat: true,
                batch_info: None,
            };

            self.network.send_message(
                follower_id,
                RaftMessage::AppendEntriesRequest(request)
            )?;

            leader_state.heartbeat_timers.insert(follower_id.clone(), heartbeat_time);
        }

        Ok(())
    }

    /// Helper methods
    fn random_election_timeout() -> Instant {
        let base = Duration::from_millis(150);
        let jitter = Duration::from_millis(fastrand::u64(0..150));
        Instant::now() + base + jitter
    }

    fn reset_election_timer(&mut self) {
        self.volatile_state.election_timeout = Self::random_election_timeout();
    }

    fn get_last_log_index(&self) -> LogIndex {
        self.persistent_state.log.len() as LogIndex
    }

    fn get_last_log_term(&self) -> Term {
        self.persistent_state.log
            .last()
            .map(|entry| entry.term)
            .unwrap_or(0)
    }

    fn is_log_up_to_date(&self, last_log_index: LogIndex, last_log_term: Term) -> bool {
        let our_last_log_term = self.get_last_log_term();
        let our_last_log_index = self.get_last_log_index();

        last_log_term > our_last_log_term ||
        (last_log_term == our_last_log_term && last_log_index >= our_last_log_index)
    }

    fn get_node_metrics(&self) -> NodeMetrics {
        NodeMetrics {
            cpu_usage: 50.0, // Placeholder
            memory_usage: 30.0, // Placeholder
            network_latency: Duration::from_millis(10), // Placeholder
            log_size: self.persistent_state.log.len(),
            commit_index: self.persistent_state.commit_index,
            applied_index: self.persistent_state.last_applied,
        }
    }

    // Additional helper methods would be implemented here
    fn save_persistent_state(&self) -> Result<()> {
        // Implementation for persisting state to storage
        Ok(())
    }

    fn load_persistent_state(&mut self) -> Result<()> {
        // Implementation for loading state from storage
        Ok(())
    }

    fn start_background_tasks(&mut self) -> Result<()> {
        // Implementation for starting background tasks
        Ok(())
    }

    fn calculate_checksum(&self, data: &[u8]) -> u64 {
        // Implementation for calculating entry checksum
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        data.hash(&mut hasher);
        hasher.finish()
    }
}

// Additional type definitions and implementations
pub type Term = u64;
pub type LogIndex = u64;
pub type NodeId = String;
pub type RequestId = String;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VoteRejectionReason {
    AlreadyVoted,
    LogNotUpToDate,
    StaleTerm,
    NetworkPartition,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeMetrics {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub network_latency: Duration,
    pub log_size: usize,
    pub commit_index: LogIndex,
    pub applied_index: LogIndex,
}

// Additional supporting types and implementations would continue here...
// This is a comprehensive but abbreviated implementation of Raft

// Default implementations
impl Default for RaftConfig {
    fn default() -> Self {
        Self {
            node_config: RaftNodeConfig::default(),
            timing_config: RaftTimingConfig::default(),
            log_config: RaftLogConfig::default(),
            snapshot_config: RaftSnapshotConfig::default(),
            network_config: RaftNetworkConfig::default(),
            safety_config: RaftSafetyConfig::default(),
            performance_config: RaftPerformanceConfig::default(),
            monitoring_config: RaftMonitoringConfig::default(),
        }
    }
}

impl Default for RaftStatistics {
    fn default() -> Self {
        Self {
            election_stats: ElectionStatistics::default(),
            replication_stats: ReplicationStatistics::default(),
            snapshot_stats: SnapshotStatistics::default(),
            performance_stats: RaftPerformanceStatistics::default(),
            network_stats: NetworkStatistics::default(),
            client_stats: ClientStatistics::default(),
        }
    }
}

// Error types
use anyhow::Result;

// Stub implementations for referenced types
use crate::tpu::pod_coordination::types::{
    RaftRolePreferences, RaftNodeCapabilities, LogStorageBackend, LogPersistenceSettings,
    LogValidationSettings, LogIndexingSettings, SnapshotFrequency, SnapshotStorageBackend,
    SnapshotRetentionPolicy, TransportProtocol, ConnectionPoolingConfig, MessageBatchingConfig,
    NetworkCompressionConfig, NetworkFailureDetectionConfig, NetworkLoadBalancingConfig,
    ByzantineFaultToleranceConfig, MembershipChangeSafetyConfig, LogSafetyGuaranteesConfig,
    LeaderSafetyMechanismsConfig, PartitionHandlingConfig, ParallelProcessingConfig,
    CachingConfig, OptimizationStrategiesConfig, ResourceManagementConfig, PerformanceTuningConfig,
    MetricsCollectionConfig, HealthMonitoringConfig, PerformanceMonitoringConfig,
    AlertConfiguration, LoggingConfiguration,
    // Additional types
    ConfigurationState, MessageQueues, PerformanceMetrics, LeaderLease, ClientSessionManager,
    FollowerMetrics, RecoveryState, CatchUpState, PreVoteState, CompressionType,
    PreVoteRequest, PreVoteResponse, HeartbeatMessage, BatchInfo, ConflictInfo,
    SnapshotMetadata, SnapshotError, ClientRequestType, ClientError,
    BatchStatistics, FollowerLagStats, RaftPerformanceStatistics, NetworkStatistics,
    ClientStatistics, LogStorage, SnapshotStorage, RaftNetwork,
};