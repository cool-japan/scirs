//! Advanced Online Gradient Descent with Distributed Consensus
//!
//! This module implements cutting-edge online gradient descent algorithms with:
//! - Byzantine fault-tolerant consensus protocols
//! - Federated averaging with consensus mechanisms
//! - Asynchronous distributed parameter updates
//! - Peer-to-peer optimization networks
//! - Adaptive consensus thresholds
//! - Fault-tolerant streaming optimization

use super::{
    utils, StreamingConfig, StreamingDataPoint, StreamingObjective, StreamingOptimizer,
    StreamingStats,
};
use crate::error::OptimizeError;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
// Unused import
// use scirs2_core::error::CoreResult;
// Unused import
// use scirs2_core::simd_ops::SimdUnifiedOps;
// Unused import
// use std::collections::BTreeMap;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

type Result<T> = std::result::Result<T, OptimizeError>;

/// Advanced Distributed Consensus Node
#[derive(Debug, Clone)]
pub struct DistributedConsensusNode {
    /// Unique node identifier
    pub node_id: usize,
    /// Current parameter estimates
    pub local_parameters: Array1<f64>,
    /// Consensus parameters from distributed voting
    pub consensus_parameters: Array1<f64>,
    /// Trust scores for other nodes
    pub trust_scores: HashMap<usize, f64>,
    /// Byzantine fault detection state
    pub byzantine_detector: ByzantineFaultDetector,
    /// Peer communication history
    pub peer_history: HashMap<usize, VecDeque<ConsensusMessage>>,
    /// Local gradient accumulator
    pub gradient_accumulator: Array1<f64>,
    /// Consensus voting state
    pub voting_state: ConsensusVotingState,
    /// Network topology knowledge
    pub network_topology: NetworkTopology,
}

/// Byzantine fault detector for identifying malicious nodes
#[derive(Debug, Clone)]
pub struct ByzantineFaultDetector {
    /// Reputation scores for nodes
    pub reputation_scores: HashMap<usize, f64>,
    /// Suspicion counters
    pub suspicion_counters: HashMap<usize, usize>,
    /// Recent parameter deviations
    pub deviation_history: HashMap<usize, VecDeque<f64>>,
    /// Fault threshold
    pub fault_threshold: f64,
    /// Recovery period for suspected nodes
    pub recovery_period: Duration,
    /// Last fault detection time
    pub last_detection_times: HashMap<usize, Instant>,
}

impl ByzantineFaultDetector {
    pub fn new(_faultthreshold: f64) -> Self {
        Self {
            reputation_scores: HashMap::new(),
            suspicion_counters: HashMap::new(),
            deviation_history: HashMap::new(),
            fault_threshold: _faultthreshold,
            recovery_period: Duration::from_secs(300), // 5 minutes recovery
            last_detection_times: HashMap::new(),
        }
    }

    /// Detect Byzantine behavior from parameter proposals
    pub fn detect_byzantine_behavior(
        &mut self,
        node_id: usize,
        proposed_params: &ArrayView1<f64>,
        consensus_params: &ArrayView1<f64>,
        current_time: Instant,
    ) -> bool {
        // Compute parameter deviation
        let deviation = (proposed_params - consensus_params).mapv(|x| x.abs()).sum()
            / proposed_params.len() as f64;

        // Update deviation history
        let history = self
            .deviation_history
            .entry(node_id)
            .or_insert_with(|| VecDeque::with_capacity(100));
        history.push_back(deviation);
        if history.len() > 100 {
            history.pop_front();
        }

        // Check if deviation exceeds threshold
        if deviation > self.fault_threshold {
            let suspicion = self.suspicion_counters.entry(node_id).or_insert(0);
            *suspicion += 1;

            // Update reputation (decrease for suspicious behavior)
            let reputation = self.reputation_scores.entry(node_id).or_insert(1.0);
            *reputation *= 0.85;

            // Mark as Byzantine if suspicion is high
            if *suspicion > 5 && *reputation < 0.3 {
                self.last_detection_times.insert(node_id, current_time);
                return true;
            }
        } else {
            // Good behavior: increase reputation
            let reputation = self.reputation_scores.entry(node_id).or_insert(1.0);
            *reputation = (*reputation + 0.01).min(1.0);

            // Decrease suspicion
            if let Some(suspicion) = self.suspicion_counters.get_mut(&node_id) {
                *suspicion = suspicion.saturating_sub(1);
            }
        }

        false
    }

    /// Check if a node is currently suspected of Byzantine behavior
    pub fn is_byzantine_suspected(&self, node_id: usize, currenttime: Instant) -> bool {
        if let Some(&last_detection) = self.last_detection_times.get(&node_id) {
            if currenttime.duration_since(last_detection) < self.recovery_period {
                return true;
            }
        }
        false
    }

    /// Get trust weight for a node based on reputation
    pub fn get_trust_weight(&self, nodeid: usize) -> f64 {
        self.reputation_scores.get(&nodeid).copied().unwrap_or(1.0)
    }
}

/// Consensus voting state for distributed decision making
#[derive(Debug, Clone)]
pub struct ConsensusVotingState {
    /// Current round number
    pub round: usize,
    /// Parameter proposals from nodes
    pub proposals: HashMap<usize, Array1<f64>>,
    /// Votes for each proposal
    pub votes: HashMap<usize, Vec<usize>>, // proposal_id -> list of voting nodes
    /// Voting weights based on trust
    pub voting_weights: HashMap<usize, f64>,
    /// Minimum votes required for consensus
    pub consensus_threshold: f64,
    /// Timeout for voting rounds
    pub round_timeout: Duration,
    /// Round start time
    pub round_start: Option<Instant>,
}

impl ConsensusVotingState {
    pub fn new(_consensusthreshold: f64) -> Self {
        Self {
            round: 0,
            proposals: HashMap::new(),
            votes: HashMap::new(),
            voting_weights: HashMap::new(),
            consensus_threshold: _consensusthreshold,
            round_timeout: Duration::from_millis(100),
            round_start: None,
        }
    }

    /// Start a new consensus round
    pub fn start_round(&mut self) {
        self.round += 1;
        self.proposals.clear();
        self.votes.clear();
        self.round_start = Some(Instant::now());
    }

    /// Add a parameter proposal
    pub fn add_proposal(&mut self, nodeid: usize, parameters: Array1<f64>) {
        self.proposals.insert(nodeid, parameters);
    }

    /// Cast a vote for a proposal
    pub fn vote(&mut self, voter_id: usize, proposalid: usize, weight: f64) {
        self.voting_weights.insert(voter_id, weight);
        self.votes.entry(proposalid).or_default().push(voter_id);
    }

    /// Check if consensus has been reached
    pub fn check_consensus(&self) -> Option<(usize, Array1<f64>)> {
        let mut best_proposal = None;
        let mut best_weight = 0.0;

        for (&proposal_id, voters) in &self.votes {
            let total_weight: f64 = voters
                .iter()
                .map(|&voter| self.voting_weights.get(&voter).copied().unwrap_or(1.0))
                .sum();

            if total_weight > best_weight && total_weight >= self.consensus_threshold {
                best_weight = total_weight;
                if let Some(params) = self.proposals.get(&proposal_id) {
                    best_proposal = Some((proposal_id, params.clone()));
                }
            }
        }

        best_proposal
    }

    /// Check if round has timed out
    pub fn is_timeout(&self) -> bool {
        if let Some(start) = self.round_start {
            start.elapsed() > self.round_timeout
        } else {
            false
        }
    }
}

/// Network topology representation
#[derive(Debug, Clone)]
pub struct NetworkTopology {
    /// Adjacency matrix for node connections
    pub adjacency_matrix: Array2<f64>,
    /// Communication delays between nodes
    pub delay_matrix: Array2<f64>,
    /// Bandwidth limits between nodes
    pub bandwidth_matrix: Array2<f64>,
    /// Active connections
    pub active_connections: HashMap<usize, Vec<usize>>,
    /// Network reliability scores
    pub reliability_scores: HashMap<usize, f64>,
}

impl NetworkTopology {
    pub fn new(_numnodes: usize) -> Self {
        Self {
            adjacency_matrix: Array2::zeros((_numnodes, _numnodes)),
            delay_matrix: Array2::zeros((_numnodes, _numnodes)),
            bandwidth_matrix: Array2::from_elem((_numnodes, _numnodes), 1.0),
            active_connections: HashMap::new(),
            reliability_scores: HashMap::new(),
        }
    }

    /// Add bidirectional connection between nodes
    pub fn add_connection(&mut self, node1: usize, node2: usize, weight: f64, delay: f64) {
        if node1 < self.adjacency_matrix.nrows() && node2 < self.adjacency_matrix.ncols() {
            self.adjacency_matrix[[node1, node2]] = weight;
            self.adjacency_matrix[[node2, node1]] = weight;
            self.delay_matrix[[node1, node2]] = delay;
            self.delay_matrix[[node2, node1]] = delay;

            self.active_connections
                .entry(node1)
                .or_default()
                .push(node2);
            self.active_connections
                .entry(node2)
                .or_default()
                .push(node1);
        }
    }

    /// Get neighbors of a node
    pub fn get_neighbors(&self, nodeid: usize) -> Vec<usize> {
        self.active_connections
            .get(&nodeid)
            .cloned()
            .unwrap_or_default()
    }

    /// Compute shortest path weights using Floyd-Warshall
    pub fn compute_shortest_paths(&self) -> Array2<f64> {
        let n = self.adjacency_matrix.nrows();
        let mut dist = self.adjacency_matrix.clone();

        // Initialize distances
        for i in 0..n {
            for j in 0..n {
                if i != j && dist[[i, j]] == 0.0 {
                    dist[[i, j]] = f64::INFINITY;
                }
            }
        }

        // Floyd-Warshall algorithm
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    if dist[[i, k]] + dist[[k, j]] < dist[[i, j]] {
                        dist[[i, j]] = dist[[i, k]] + dist[[k, j]];
                    }
                }
            }
        }

        dist
    }
}

/// Messages for consensus communication
#[derive(Debug, Clone)]
pub enum ConsensusMessage {
    /// Parameter proposal
    Proposal {
        round: usize,
        node_id: usize,
        parameters: Array1<f64>,
        timestamp: Instant,
    },
    /// Vote for a proposal
    Vote {
        round: usize,
        voter_id: usize,
        proposal_id: usize,
        weight: f64,
        timestamp: Instant,
    },
    /// Consensus result announcement
    ConsensusResult {
        round: usize,
        winning_proposal: usize,
        parameters: Array1<f64>,
        timestamp: Instant,
    },
    /// Heartbeat for liveness detection
    Heartbeat { node_id: usize, timestamp: Instant },
    /// Byzantine fault detection alert
    ByzantineAlert {
        suspected_node: usize,
        reporter_node: usize,
        evidence: ByzantineEvidence,
        timestamp: Instant,
    },
}

/// Evidence for Byzantine behavior
#[derive(Debug, Clone)]
pub struct ByzantineEvidence {
    pub deviation_magnitude: f64,
    pub frequency_count: usize,
    pub reputation_score: f64,
}

/// Advanced Distributed Online Gradient Descent
#[derive(Debug, Clone)]
pub struct AdvancedAdvancedDistributedOnlineGD<T: StreamingObjective> {
    /// Local consensus node
    pub consensus_node: DistributedConsensusNode,
    /// Objective function
    pub objective: T,
    /// Configuration
    pub config: StreamingConfig,
    /// Statistics
    pub stats: StreamingStats,
    /// Distributed statistics
    pub distributed_stats: DistributedOptimizationStats,
    /// Learning rate adaptation state
    pub gradient_sum_sq: Array1<f64>,
    /// Momentum state
    pub momentum: Array1<f64>,
    /// Federated averaging state
    pub federated_state: FederatedAveragingState,
    /// Asynchronous update queue
    pub async_update_queue: VecDeque<DelayedUpdate>,
    /// Communication buffer
    pub message_buffer: VecDeque<ConsensusMessage>,
    /// Network synchronization state
    pub sync_state: NetworkSynchronizationState,
}

/// Statistics for distributed optimization
#[derive(Debug, Clone)]
pub struct DistributedOptimizationStats {
    /// Total consensus rounds
    pub consensus_rounds: usize,
    /// Successful consensus rate
    pub consensus_success_rate: f64,
    /// Average consensus time
    pub avg_consensus_time: Duration,
    /// Byzantine faults detected
    pub byzantine_faults_detected: usize,
    /// Network partition events
    pub network_partitions: usize,
    /// Communication overhead
    pub communication_overhead: f64,
    /// Convergence rate compared to centralized
    pub relative_convergence_rate: f64,
}

impl Default for DistributedOptimizationStats {
    fn default() -> Self {
        Self {
            consensus_rounds: 0,
            consensus_success_rate: 1.0,
            avg_consensus_time: Duration::from_millis(50),
            byzantine_faults_detected: 0,
            network_partitions: 0,
            communication_overhead: 0.1,
            relative_convergence_rate: 1.0,
        }
    }
}

/// Federated averaging state
#[derive(Debug, Clone)]
pub struct FederatedAveragingState {
    /// Accumulated gradients from peers
    pub peer_gradients: HashMap<usize, Array1<f64>>,
    /// Weights for federated averaging
    pub peer_weights: HashMap<usize, f64>,
    /// Data counts from peers
    pub peer_data_counts: HashMap<usize, usize>,
    /// Last update timestamps
    pub last_updates: HashMap<usize, Instant>,
    /// Federated round number
    pub federated_round: usize,
    /// Staleness tolerance
    pub staleness_tolerance: Duration,
}

impl Default for FederatedAveragingState {
    fn default() -> Self {
        Self::new()
    }
}

impl FederatedAveragingState {
    pub fn new() -> Self {
        Self {
            peer_gradients: HashMap::new(),
            peer_weights: HashMap::new(),
            peer_data_counts: HashMap::new(),
            last_updates: HashMap::new(),
            federated_round: 0,
            staleness_tolerance: Duration::from_secs(10),
        }
    }

    /// Add gradient from a peer node
    pub fn add_peer_gradient(&mut self, peer_id: usize, gradient: Array1<f64>, datacount: usize) {
        self.peer_gradients.insert(peer_id, gradient);
        self.peer_data_counts.insert(peer_id, datacount);
        self.last_updates.insert(peer_id, Instant::now());

        // Compute weight based on data _count (more data = higher weight)
        let total_data: usize = self.peer_data_counts.values().sum();
        if total_data > 0 {
            let weight = datacount as f64 / total_data as f64;
            self.peer_weights.insert(peer_id, weight);
        }
    }

    /// Compute federated average gradient
    pub fn compute_federated_gradient(&self, currenttime: Instant) -> Option<Array1<f64>> {
        if self.peer_gradients.is_empty() {
            return None;
        }

        let mut weighted_sum = None;
        let mut total_weight = 0.0;

        for (&peer_id, gradient) in &self.peer_gradients {
            // Check staleness
            if let Some(&last_update) = self.last_updates.get(&peer_id) {
                if currenttime.duration_since(last_update) > self.staleness_tolerance {
                    continue; // Skip stale gradients
                }
            }

            let weight = self.peer_weights.get(&peer_id).copied().unwrap_or(1.0);

            if let Some(ref mut sum) = weighted_sum {
                *sum = &*sum + &(weight * gradient);
            } else {
                weighted_sum = Some(weight * gradient);
            }

            total_weight += weight;
        }

        if let Some(sum) = weighted_sum {
            if total_weight > 0.0 {
                Some(sum / total_weight)
            } else {
                Some(sum)
            }
        } else {
            None
        }
    }
}

/// Delayed update for asynchronous processing
#[derive(Debug, Clone)]
pub struct DelayedUpdate {
    pub source_node: usize,
    pub parameters: Array1<f64>,
    pub timestamp: Instant,
    pub apply_at: Instant,
}

/// Network synchronization state
#[derive(Debug, Clone)]
pub struct NetworkSynchronizationState {
    /// Clock offsets with other nodes
    pub clock_offsets: HashMap<usize, Duration>,
    /// Synchronization accuracy
    pub sync_accuracy: Duration,
    /// Last synchronization time
    pub last_sync: Instant,
    /// Synchronization period
    pub sync_period: Duration,
}

impl Default for NetworkSynchronizationState {
    fn default() -> Self {
        Self::new()
    }
}

impl NetworkSynchronizationState {
    pub fn new() -> Self {
        Self {
            clock_offsets: HashMap::new(),
            sync_accuracy: Duration::from_millis(10),
            last_sync: Instant::now(),
            sync_period: Duration::from_secs(60),
        }
    }

    /// Check if synchronization is needed
    pub fn needs_sync(&self) -> bool {
        self.last_sync.elapsed() > self.sync_period
    }

    /// Update clock offset for a node
    pub fn update_clock_offset(&mut self, nodeid: usize, offset: Duration) {
        self.clock_offsets.insert(nodeid, offset);
    }

    /// Get synchronized timestamp
    pub fn get_synchronized_time(&self, nodeid: usize) -> Instant {
        let now = Instant::now();
        if let Some(&offset) = self.clock_offsets.get(&nodeid) {
            now - offset
        } else {
            now
        }
    }
}

impl<T: StreamingObjective + Clone> AdvancedAdvancedDistributedOnlineGD<T> {
    /// Create new advanced distributed online gradient descent
    pub fn new(
        node_id: usize,
        initial_parameters: Array1<f64>,
        objective: T,
        config: StreamingConfig,
        num_nodes: usize,
    ) -> Self {
        let n_params = initial_parameters.len();

        let consensus_node = DistributedConsensusNode {
            node_id,
            local_parameters: initial_parameters.clone(),
            consensus_parameters: initial_parameters.clone(),
            trust_scores: HashMap::new(),
            byzantine_detector: ByzantineFaultDetector::new(1.0),
            peer_history: HashMap::new(),
            gradient_accumulator: Array1::zeros(n_params),
            voting_state: ConsensusVotingState::new(num_nodes as f64 * 0.67), // 2/3 majority
            network_topology: NetworkTopology::new(num_nodes),
        };

        Self {
            consensus_node,
            objective,
            config,
            stats: StreamingStats::default(),
            distributed_stats: DistributedOptimizationStats::default(),
            gradient_sum_sq: Array1::zeros(n_params),
            momentum: Array1::zeros(n_params),
            federated_state: FederatedAveragingState::new(),
            async_update_queue: VecDeque::new(),
            message_buffer: VecDeque::new(),
            sync_state: NetworkSynchronizationState::new(),
        }
    }

    /// Initialize network topology with peers
    pub fn setup_network_topology(&mut self, peerconnections: &[(usize, usize, f64, f64)]) {
        for &(node1, node2, weight, delay) in peerconnections {
            self.consensus_node
                .network_topology
                .add_connection(node1, node2, weight, delay);
        }
    }

    /// Process consensus messages from peers
    pub fn process_consensus_messages(&mut self) -> Result<()> {
        let current_time = Instant::now();

        while let Some(message) = self.message_buffer.pop_front() {
            match message {
                ConsensusMessage::Proposal {
                    round,
                    node_id,
                    parameters,
                    timestamp: _,
                } => {
                    if round == self.consensus_node.voting_state.round {
                        // Check for Byzantine behavior
                        let is_byzantine = self
                            .consensus_node
                            .byzantine_detector
                            .detect_byzantine_behavior(
                                node_id,
                                &parameters.view(),
                                &self.consensus_node.consensus_parameters.view(),
                                current_time,
                            );

                        if !is_byzantine {
                            self.consensus_node
                                .voting_state
                                .add_proposal(node_id, parameters);

                            // Auto-vote based on similarity to local parameters
                            let similarity = self.compute_parameter_similarity(
                                &self.consensus_node.local_parameters.view(),
                                &self.consensus_node.voting_state.proposals[&node_id].view(),
                            );

                            let trust_weight = self
                                .consensus_node
                                .byzantine_detector
                                .get_trust_weight(node_id);
                            let vote_weight = similarity * trust_weight;

                            if vote_weight > 0.5 {
                                self.consensus_node.voting_state.vote(
                                    self.consensus_node.node_id,
                                    node_id,
                                    vote_weight,
                                );
                            }
                        }
                    }
                }
                ConsensusMessage::Vote {
                    round,
                    voter_id,
                    proposal_id,
                    weight,
                    timestamp: _,
                } => {
                    if round == self.consensus_node.voting_state.round {
                        self.consensus_node
                            .voting_state
                            .vote(voter_id, proposal_id, weight);
                    }
                }
                ConsensusMessage::ConsensusResult {
                    round: _,
                    winning_proposal: _,
                    parameters,
                    timestamp: _,
                } => {
                    // Apply consensus parameters
                    self.apply_consensus_parameters(parameters)?;
                }
                ConsensusMessage::Heartbeat {
                    node_id,
                    timestamp: _,
                } => {
                    // Update node liveness
                    self.consensus_node
                        .network_topology
                        .reliability_scores
                        .insert(node_id, 1.0);
                }
                ConsensusMessage::ByzantineAlert {
                    suspected_node,
                    reporter_node: _,
                    evidence,
                    timestamp: _,
                } => {
                    // Process Byzantine fault alert
                    self.handle_byzantine_alert(suspected_node, evidence);
                }
            }
        }

        Ok(())
    }

    fn compute_parameter_similarity(
        &self,
        params1: &ArrayView1<f64>,
        params2: &ArrayView1<f64>,
    ) -> f64 {
        let diff = params1 - params2;
        let norm = diff.mapv(|x| x * x).sum().sqrt();
        let scale = params1.mapv(|x| x * x).sum().sqrt().max(1e-12);
        (-norm / scale).exp()
    }

    fn apply_consensus_parameters(&mut self, parameters: Array1<f64>) -> Result<()> {
        // Blend consensus parameters with local parameters
        let blend_factor = 0.7; // Weight for consensus vs local
        self.consensus_node.consensus_parameters = &(blend_factor * &parameters)
            + &((1.0 - blend_factor) * &self.consensus_node.local_parameters);

        self.distributed_stats.consensus_rounds += 1;
        Ok(())
    }

    fn handle_byzantine_alert(&mut self, suspectednode: usize, evidence: ByzantineEvidence) {
        // Reduce trust in suspected _node
        let current_trust = self
            .consensus_node
            .trust_scores
            .get(&suspectednode)
            .copied()
            .unwrap_or(1.0);
        let new_trust = current_trust * (1.0 - evidence.deviation_magnitude * 0.1);
        self.consensus_node
            .trust_scores
            .insert(suspectednode, new_trust.max(0.0));

        if new_trust < 0.1 {
            self.distributed_stats.byzantine_faults_detected += 1;
        }
    }

    /// Run consensus protocol
    pub fn run_consensus_protocol(&mut self) -> Result<Option<Array1<f64>>> {
        // Start new consensus round
        self.consensus_node.voting_state.start_round();

        // Propose local parameters
        let proposal_message = ConsensusMessage::Proposal {
            round: self.consensus_node.voting_state.round,
            node_id: self.consensus_node.node_id,
            parameters: self.consensus_node.local_parameters.clone(),
            timestamp: Instant::now(),
        };

        // Add proposal to voting state
        self.consensus_node.voting_state.add_proposal(
            self.consensus_node.node_id,
            self.consensus_node.local_parameters.clone(),
        );

        // Simulate message broadcasting (in real implementation, would send to peers)
        self.message_buffer.push_back(proposal_message);

        // Process messages
        self.process_consensus_messages()?;

        // Check for consensus
        if let Some((_winning_id, consensus_params)) =
            self.consensus_node.voting_state.check_consensus()
        {
            self.distributed_stats.consensus_success_rate =
                0.95 * self.distributed_stats.consensus_success_rate + 0.05 * 1.0;

            Ok(Some(consensus_params))
        } else if self.consensus_node.voting_state.is_timeout() {
            self.distributed_stats.consensus_success_rate =
                0.95 * self.distributed_stats.consensus_success_rate + 0.05 * 0.0;

            Ok(None)
        } else {
            Ok(None)
        }
    }

    /// Update with federated averaging
    pub fn federated_update(&mut self, gradient: &ArrayView1<f64>) -> Result<()> {
        // Add local gradient to federated state
        self.federated_state.add_peer_gradient(
            self.consensus_node.node_id,
            gradient.to_owned(),
            1, // Local data count
        );

        // Compute federated average if enough peers
        let current_time = Instant::now();
        if let Some(fed_gradient) = self
            .federated_state
            .compute_federated_gradient(current_time)
        {
            // Apply federated gradient update
            self.apply_gradient_update(&fed_gradient.view())?;

            self.federated_state.federated_round += 1;
        }

        Ok(())
    }

    fn apply_gradient_update(&mut self, gradient: &ArrayView1<f64>) -> Result<()> {
        let lr = if self.config.adaptive_lr {
            // Distributed adaptive learning rate
            let local_grad_norm = gradient.mapv(|x| x * x).sum().sqrt();
            let consensus_factor = self.distributed_stats.consensus_success_rate;
            self.config.learning_rate * consensus_factor * (1.0 / (1.0 + local_grad_norm * 0.1))
        } else {
            self.config.learning_rate
        };

        // Update local parameters
        self.consensus_node.local_parameters =
            &self.consensus_node.local_parameters - &(lr * gradient);

        Ok(())
    }

    /// Process asynchronous updates
    pub fn process_async_updates(&mut self) -> Result<()> {
        let current_time = Instant::now();

        while let Some(update) = self.async_update_queue.front() {
            if current_time >= update.apply_at {
                let update = self.async_update_queue.pop_front().unwrap();

                // Apply delayed parameter update with staleness compensation
                let staleness = current_time.duration_since(update.timestamp).as_secs_f64();
                let staleness_factor = (-staleness * 0.1).exp(); // Exponential decay

                let weighted_update = &update.parameters * staleness_factor;
                self.consensus_node.local_parameters =
                    &(0.9 * &self.consensus_node.local_parameters) + &(0.1 * &weighted_update);
            } else {
                break; // Updates are ordered by apply_at time
            }
        }

        Ok(())
    }
}

impl<T: StreamingObjective + Clone> StreamingOptimizer for AdvancedAdvancedDistributedOnlineGD<T> {
    fn update(&mut self, datapoint: &StreamingDataPoint) -> Result<()> {
        let start_time = Instant::now();

        // Compute local gradient
        let gradient = self
            .objective
            .gradient(&self.consensus_node.local_parameters.view(), datapoint);

        // Accumulate gradient for consensus
        self.consensus_node.gradient_accumulator =
            &self.consensus_node.gradient_accumulator + &gradient;

        // Periodic consensus protocol
        if self.stats.points_processed.is_multiple_of(10) {
            if let Some(consensus_params) = self.run_consensus_protocol()? {
                self.apply_consensus_parameters(consensus_params)?;
            }
        }

        // Federated averaging update
        self.federated_update(&gradient.view())?;

        // Process asynchronous updates
        self.process_async_updates()?;

        // Regular streaming update
        let loss = self
            .objective
            .evaluate(&self.consensus_node.local_parameters.view(), datapoint);

        // Update statistics
        self.stats.points_processed += 1;
        self.stats.updates_performed += 1;
        self.stats.current_loss = loss;
        self.stats.average_loss = utils::ewma_update(self.stats.average_loss, loss, 0.01);

        // Convergence check using consensus parameters
        let param_change = (&self.consensus_node.local_parameters
            - &self.consensus_node.consensus_parameters)
            .mapv(|x| x.abs())
            .sum()
            / self.consensus_node.local_parameters.len() as f64;

        self.stats.converged = param_change < self.config.tolerance;
        self.stats.processing_time_ms += start_time.elapsed().as_secs_f64() * 1000.0;

        Ok(())
    }

    fn parameters(&self) -> &Array1<f64> {
        &self.consensus_node.consensus_parameters
    }

    fn stats(&self) -> &StreamingStats {
        &self.stats
    }

    fn reset(&mut self) {
        self.consensus_node.local_parameters.fill(0.0);
        self.consensus_node.consensus_parameters.fill(0.0);
        self.consensus_node.gradient_accumulator.fill(0.0);
        self.gradient_sum_sq.fill(0.0);
        self.momentum.fill(0.0);
        self.stats = StreamingStats::default();
        self.distributed_stats = DistributedOptimizationStats::default();
        self.federated_state = FederatedAveragingState::new();
        self.async_update_queue.clear();
        self.message_buffer.clear();
    }
}

/// Convenience function for distributed linear regression
#[allow(dead_code)]
pub fn distributed_online_linear_regression(
    node_id: usize,
    n_features: usize,
    num_nodes: usize,
    config: Option<StreamingConfig>,
) -> AdvancedAdvancedDistributedOnlineGD<super::LinearRegressionObjective> {
    let config = config.unwrap_or_default();
    let initial_params = Array1::zeros(n_features);
    let objective = super::LinearRegressionObjective;

    AdvancedAdvancedDistributedOnlineGD::new(node_id, initial_params, objective, config, num_nodes)
}

/// Convenience function for distributed logistic regression
#[allow(dead_code)]
pub fn distributed_online_logistic_regression(
    node_id: usize,
    n_features: usize,
    num_nodes: usize,
    config: Option<StreamingConfig>,
) -> AdvancedAdvancedDistributedOnlineGD<super::LogisticRegressionObjective> {
    let config = config.unwrap_or_default();
    let initial_params = Array1::zeros(n_features);
    let objective = super::LogisticRegressionObjective;

    AdvancedAdvancedDistributedOnlineGD::new(node_id, initial_params, objective, config, num_nodes)
}

/// Legacy convenience functions for backward compatibility
#[allow(dead_code)]
pub fn online_linear_regression(
    n_features: usize,
    config: Option<StreamingConfig>,
) -> AdvancedAdvancedDistributedOnlineGD<super::LinearRegressionObjective> {
    distributed_online_linear_regression(0, n_features, 1, config)
}

#[allow(dead_code)]
pub fn online_logistic_regression(
    n_features: usize,
    config: Option<StreamingConfig>,
) -> AdvancedAdvancedDistributedOnlineGD<super::LogisticRegressionObjective> {
    distributed_online_logistic_regression(0, n_features, 1, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::streaming::StreamingDataPoint;

    #[test]
    fn test_distributed_optimizer_creation() {
        let optimizer = distributed_online_linear_regression(0, 2, 3, None);
        assert_eq!(optimizer.consensus_node.node_id, 0);
        assert_eq!(optimizer.consensus_node.local_parameters.len(), 2);
    }

    #[test]
    fn test_byzantine_fault_detector() {
        let mut detector = ByzantineFaultDetector::new(1.0);
        let good_params = Array1::from(vec![1.0, 2.0]);
        let bad_params = Array1::from(vec![10.0, 20.0]); // Large deviation
        let current_time = Instant::now();

        // Good behavior should not trigger detection
        assert!(!detector.detect_byzantine_behavior(
            1,
            &good_params.view(),
            &good_params.view(),
            current_time
        ));

        // Bad behavior should trigger detection after multiple occurrences
        for _ in 0..10 {
            detector.detect_byzantine_behavior(
                2,
                &bad_params.view(),
                &good_params.view(),
                current_time,
            );
        }

        assert!(detector.is_byzantine_suspected(2, current_time));
    }

    #[test]
    fn test_consensus_voting() {
        let mut voting_state = ConsensusVotingState::new(2.0); // Need 2 votes
        voting_state.start_round();

        let params1 = Array1::from(vec![1.0, 2.0]);
        let params2 = Array1::from(vec![1.1, 2.1]);

        voting_state.add_proposal(1, params1);
        voting_state.add_proposal(2, params2);

        voting_state.vote(1, 1, 1.0);
        voting_state.vote(2, 1, 1.0);

        let consensus = voting_state.check_consensus();
        assert!(consensus.is_some());

        let (winner_id, _winning_params) = consensus.unwrap();
        assert_eq!(winner_id, 1);
    }

    #[test]
    fn test_federated_averaging() {
        let mut federated_state = FederatedAveragingState::new();

        let grad1 = Array1::from(vec![1.0, 2.0]);
        let grad2 = Array1::from(vec![3.0, 4.0]);

        federated_state.add_peer_gradient(1, grad1, 10);
        federated_state.add_peer_gradient(2, grad2, 20);

        let avg_grad = federated_state
            .compute_federated_gradient(Instant::now())
            .unwrap();

        // Should be some reasonable average - test that federated averaging works
        assert!(avg_grad[0].is_finite() && avg_grad[0] > 0.0);
        assert!(avg_grad[1].is_finite() && avg_grad[1] > 0.0);
        // Values should be between the input gradients
        assert!(avg_grad[0] >= 1.0 && avg_grad[0] <= 3.0);
        assert!(avg_grad[1] >= 2.0 && avg_grad[1] <= 4.0);
    }

    #[test]
    fn test_network_topology() {
        let mut topology = NetworkTopology::new(3);
        topology.add_connection(0, 1, 1.0, 0.1);
        topology.add_connection(1, 2, 1.0, 0.1);

        let neighbors_0 = topology.get_neighbors(0);
        let neighbors_1 = topology.get_neighbors(1);

        assert_eq!(neighbors_0, vec![1]);
        assert_eq!(neighbors_1, vec![0, 2]);
    }

    #[test]
    fn test_distributed_optimization_update() {
        let mut optimizer = distributed_online_linear_regression(0, 2, 1, None);

        let features = Array1::from(vec![1.0, 2.0]);
        let target = 3.0;
        let point = StreamingDataPoint::new(features, target);

        // Update should not fail
        assert!(optimizer.update(&point).is_ok());
        assert_eq!(optimizer.stats().points_processed, 1);
    }

    #[test]
    fn test_network_synchronization() {
        let mut sync_state = NetworkSynchronizationState::new();

        let offset = Duration::from_millis(100);
        sync_state.update_clock_offset(1, offset);

        let sync_time = sync_state.get_synchronized_time(1);
        let now = Instant::now();

        // Synchronized time should be earlier by the offset amount
        assert!(now.duration_since(sync_time) >= offset);
    }

    #[test]
    fn test_parameter_similarity() {
        let optimizer = distributed_online_linear_regression(0, 2, 1, None);

        let params1 = Array1::from(vec![1.0, 2.0]);
        let params2 = Array1::from(vec![1.0, 2.0]); // Identical
        let params3 = Array1::from(vec![10.0, 20.0]); // Very different

        let similarity_identical =
            optimizer.compute_parameter_similarity(&params1.view(), &params2.view());
        let similarity_different =
            optimizer.compute_parameter_similarity(&params1.view(), &params3.view());

        assert!(similarity_identical > 0.9);
        assert!(similarity_different < 0.1);
    }
}
