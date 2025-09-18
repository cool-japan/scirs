//! Leader Election Management
//!
//! This module provides comprehensive leader election functionality for consensus
//! protocols including various election algorithms, priority management, candidate
//! tracking, and election statistics for TPU pod coordination.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::tpu::tpu_backend::DeviceId;
use crate::error::{Result, OptimError};

/// Leader election manager
#[derive(Debug)]
pub struct LeaderElectionManager {
    /// Election configuration
    pub config: LeaderElectionConfig,
    /// Current leader
    pub current_leader: Option<DeviceId>,
    /// Election state
    pub election_state: ElectionState,
    /// Candidate information
    pub candidates: HashMap<DeviceId, CandidateInfo>,
    /// Election statistics
    pub statistics: ElectionStatistics,
    /// Election start time
    pub election_start_time: Option<Instant>,
    /// Election timeout
    pub election_timeout: Option<Instant>,
    /// Vote tracking
    pub votes: HashMap<DeviceId, DeviceId>, // voter -> candidate
}

/// Leader election configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderElectionConfig {
    /// Election timeout
    pub timeout: Duration,
    /// Heartbeat interval
    pub heartbeat_interval: Duration,
    /// Election algorithm
    pub algorithm: ElectionAlgorithm,
    /// Priority-based settings
    pub priority_settings: PrioritySettings,
    /// Election retry settings
    pub retry_settings: ElectionRetrySettings,
    /// Candidate requirements
    pub candidate_requirements: CandidateRequirements,
}

/// Election algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ElectionAlgorithm {
    /// Bully algorithm
    Bully,
    /// Ring-based election
    Ring,
    /// Raft-style election
    Raft,
    /// Priority-based election
    Priority,
    /// Random election
    Random,
    /// Performance-based election
    Performance,
    /// Custom election algorithm
    Custom { algorithm: String },
}

/// Priority settings for elections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrioritySettings {
    /// Enable priority-based election
    pub enable: bool,
    /// Priority calculation method
    pub calculation: PriorityCalculation,
    /// Dynamic priority adjustment
    pub dynamic_adjustment: bool,
    /// Priority weights
    pub weights: PriorityWeights,
    /// Priority decay settings
    pub decay: PriorityDecay,
}

/// Priority calculation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PriorityCalculation {
    /// Static priority based on device ID
    Static,
    /// Performance-based priority
    Performance,
    /// Load-based priority
    Load,
    /// Availability-based priority
    Availability,
    /// Composite priority calculation
    Composite { factors: Vec<PriorityFactor> },
    /// Hybrid priority calculation
    Hybrid { weights: HashMap<String, f64> },
}

/// Priority factors for composite calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorityFactor {
    /// Factor name
    pub name: String,
    /// Factor type
    pub factor_type: PriorityFactorType,
    /// Weight in calculation
    pub weight: f64,
    /// Normalization method
    pub normalization: NormalizationMethod,
}

/// Priority factor types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PriorityFactorType {
    /// CPU utilization (lower is better)
    CpuUtilization,
    /// Memory usage (lower is better)
    MemoryUsage,
    /// Network latency (lower is better)
    NetworkLatency,
    /// Availability score (higher is better)
    Availability,
    /// Uptime (higher is better)
    Uptime,
    /// Load average (lower is better)
    LoadAverage,
    /// Custom factor
    Custom { metric: String, higher_is_better: bool },
}

/// Normalization methods for priority factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NormalizationMethod {
    /// Min-max normalization
    MinMax { min: f64, max: f64 },
    /// Z-score normalization
    ZScore { mean: f64, std_dev: f64 },
    /// Percentile-based normalization
    Percentile { percentiles: Vec<f64> },
    /// Custom normalization
    Custom { method: String },
}

/// Priority weights for different aspects
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorityWeights {
    /// Performance weight
    pub performance: f64,
    /// Availability weight
    pub availability: f64,
    /// Load weight
    pub load: f64,
    /// Network weight
    pub network: f64,
    /// Stability weight
    pub stability: f64,
}

/// Priority decay settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorityDecay {
    /// Enable priority decay
    pub enable: bool,
    /// Decay rate per time unit
    pub rate: f64,
    /// Decay interval
    pub interval: Duration,
    /// Minimum priority floor
    pub minimum_priority: f64,
    /// Decay function
    pub function: DecayFunction,
}

/// Priority decay functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecayFunction {
    /// Linear decay
    Linear,
    /// Exponential decay
    Exponential { half_life: Duration },
    /// Logarithmic decay
    Logarithmic { scale: f64 },
    /// Custom decay function
    Custom { function: String },
}

/// Election retry settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElectionRetrySettings {
    /// Maximum retry attempts
    pub max_retries: usize,
    /// Retry delay
    pub retry_delay: Duration,
    /// Backoff strategy
    pub backoff_strategy: RetryBackoffStrategy,
    /// Jitter settings
    pub jitter: JitterSettings,
}

/// Retry backoff strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RetryBackoffStrategy {
    /// Fixed delay
    Fixed,
    /// Linear backoff
    Linear { increment: Duration },
    /// Exponential backoff
    Exponential { base: f64, max_delay: Duration },
    /// Custom backoff
    Custom { strategy: String },
}

/// Jitter settings for election timing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JitterSettings {
    /// Enable jitter
    pub enable: bool,
    /// Jitter type
    pub jitter_type: JitterType,
    /// Maximum jitter amount
    pub max_jitter: Duration,
    /// Jitter distribution
    pub distribution: JitterDistribution,
}

/// Jitter types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JitterType {
    /// Additive jitter
    Additive,
    /// Multiplicative jitter
    Multiplicative { factor: f64 },
    /// Full jitter
    Full,
    /// Custom jitter
    Custom { jitter_type: String },
}

/// Jitter distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JitterDistribution {
    /// Uniform distribution
    Uniform,
    /// Normal distribution
    Normal { mean: f64, std_dev: f64 },
    /// Exponential distribution
    Exponential { lambda: f64 },
    /// Custom distribution
    Custom { distribution: String },
}

/// Candidate requirements for elections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateRequirements {
    /// Minimum uptime required
    pub min_uptime: Duration,
    /// Minimum performance score
    pub min_performance: f64,
    /// Maximum resource utilization
    pub max_resource_utilization: f64,
    /// Required capabilities
    pub required_capabilities: Vec<String>,
    /// Health check requirements
    pub health_requirements: HealthRequirements,
}

/// Health requirements for candidates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthRequirements {
    /// Minimum health score
    pub min_health_score: f64,
    /// Health check interval
    pub check_interval: Duration,
    /// Required health metrics
    pub required_metrics: Vec<String>,
    /// Health history requirements
    pub history_requirements: HealthHistoryRequirements,
}

/// Health history requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthHistoryRequirements {
    /// History window
    pub window: Duration,
    /// Minimum uptime percentage
    pub min_uptime_percentage: f64,
    /// Maximum failure rate
    pub max_failure_rate: f64,
    /// Minimum response rate
    pub min_response_rate: f64,
}

/// Election state
#[derive(Debug, Clone, PartialEq)]
pub enum ElectionState {
    /// No election in progress
    Idle,
    /// Election initiated
    Initiated,
    /// Election in progress
    InProgress {
        /// Election round
        round: u32,
        /// Started at
        started_at: Instant,
    },
    /// Election completed
    Completed {
        /// Winner device ID
        winner: DeviceId,
        /// Completed at
        completed_at: Instant,
    },
    /// Election failed
    Failed {
        /// Failure reason
        reason: String,
        /// Failed at
        failed_at: Instant,
    },
    /// Election timed out
    TimedOut {
        /// Timed out at
        timed_out_at: Instant,
    },
}

/// Candidate information
#[derive(Debug, Clone)]
pub struct CandidateInfo {
    /// Candidate device ID
    pub device_id: DeviceId,
    /// Candidate priority
    pub priority: f64,
    /// Performance metrics
    pub performance: PerformanceMetrics,
    /// Vote count
    pub vote_count: usize,
    /// Last heartbeat
    pub last_heartbeat: Instant,
    /// Candidate status
    pub status: CandidateStatus,
    /// Registration time
    pub registration_time: Instant,
    /// Capabilities
    pub capabilities: Vec<String>,
}

/// Candidate status
#[derive(Debug, Clone, PartialEq)]
pub enum CandidateStatus {
    /// Candidate is active
    Active,
    /// Candidate is inactive
    Inactive,
    /// Candidate is disqualified
    Disqualified { reason: String },
    /// Candidate withdrew
    Withdrew,
}

/// Performance metrics for candidates
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// CPU utilization (0.0 to 1.0)
    pub cpu_utilization: f64,
    /// Memory usage (0.0 to 1.0)
    pub memory_usage: f64,
    /// Network latency
    pub network_latency: Duration,
    /// Availability score (0.0 to 1.0)
    pub availability: f64,
    /// Response time
    pub response_time: Duration,
    /// Throughput
    pub throughput: f64,
    /// Error rate (0.0 to 1.0)
    pub error_rate: f64,
}

/// Election statistics
#[derive(Debug, Clone)]
pub struct ElectionStatistics {
    /// Total elections
    pub total_elections: usize,
    /// Successful elections
    pub successful_elections: usize,
    /// Failed elections
    pub failed_elections: usize,
    /// Timed out elections
    pub timed_out_elections: usize,
    /// Average election time
    pub avg_election_time: Duration,
    /// Shortest election time
    pub shortest_election_time: Duration,
    /// Longest election time
    pub longest_election_time: Duration,
    /// Election history
    pub election_history: Vec<ElectionRecord>,
}

/// Election record for history tracking
#[derive(Debug, Clone)]
pub struct ElectionRecord {
    /// Election ID
    pub id: u64,
    /// Start time
    pub start_time: Instant,
    /// End time
    pub end_time: Option<Instant>,
    /// Winner
    pub winner: Option<DeviceId>,
    /// Participants
    pub participants: Vec<DeviceId>,
    /// Election outcome
    pub outcome: ElectionOutcome,
    /// Duration
    pub duration: Option<Duration>,
}

/// Election outcome
#[derive(Debug, Clone, PartialEq)]
pub enum ElectionOutcome {
    /// Election succeeded
    Success { winner: DeviceId },
    /// Election failed
    Failed { reason: String },
    /// Election timed out
    TimedOut,
    /// Election was aborted
    Aborted { reason: String },
}

// Implementation blocks

impl LeaderElectionManager {
    /// Create new leader election manager
    pub fn new(config: LeaderElectionConfig) -> Result<Self> {
        Ok(Self {
            config,
            current_leader: None,
            election_state: ElectionState::Idle,
            candidates: HashMap::new(),
            statistics: ElectionStatistics::new(),
            election_start_time: None,
            election_timeout: None,
            votes: HashMap::new(),
        })
    }

    /// Start leader election
    pub fn start_election(&mut self) -> Result<()> {
        match self.election_state {
            ElectionState::InProgress { .. } => {
                return Err(OptimError::invalid_state("Election already in progress"));
            },
            _ => {},
        }

        let now = Instant::now();
        self.election_state = ElectionState::InProgress {
            round: 1,
            started_at: now,
        };
        self.election_start_time = Some(now);
        self.election_timeout = Some(now + self.config.timeout);
        self.votes.clear();

        // Create election record
        let record = ElectionRecord {
            id: self.statistics.total_elections as u64,
            start_time: now,
            end_time: None,
            winner: None,
            participants: self.candidates.keys().copied().collect(),
            outcome: ElectionOutcome::Failed { reason: "In progress".to_string() },
            duration: None,
        };

        self.statistics.election_history.push(record);
        self.statistics.total_elections += 1;

        Ok(())
    }

    /// Register candidate for election
    pub fn register_candidate(&mut self, device_id: DeviceId, info: CandidateInfo) -> Result<()> {
        // Check candidate requirements
        if !self.meets_requirements(&info)? {
            return Err(OptimError::invalid_argument("Candidate does not meet requirements"));
        }

        // Calculate priority if priority-based election
        let mut candidate_info = info;
        if self.config.priority_settings.enable {
            candidate_info.priority = self.calculate_priority(&candidate_info)?;
        }

        self.candidates.insert(device_id, candidate_info);
        Ok(())
    }

    /// Unregister candidate
    pub fn unregister_candidate(&mut self, device_id: DeviceId) -> Result<()> {
        self.candidates.remove(&device_id);
        self.votes.retain(|_, candidate| *candidate != device_id);
        Ok(())
    }

    /// Cast vote for candidate
    pub fn cast_vote(&mut self, voter: DeviceId, candidate: DeviceId) -> Result<()> {
        if !matches!(self.election_state, ElectionState::InProgress { .. }) {
            return Err(OptimError::invalid_state("No election in progress"));
        }

        if !self.candidates.contains_key(&candidate) {
            return Err(OptimError::invalid_argument("Invalid candidate"));
        }

        // Record vote
        self.votes.insert(voter, candidate);

        // Update candidate vote count
        if let Some(candidate_info) = self.candidates.get_mut(&candidate) {
            candidate_info.vote_count += 1;
        }

        Ok(())
    }

    /// Check election completion
    pub fn check_completion(&mut self) -> Result<Option<DeviceId>> {
        match &self.election_state {
            ElectionState::InProgress { started_at, .. } => {
                // Check timeout
                if let Some(timeout) = self.election_timeout {
                    if Instant::now() >= timeout {
                        self.handle_election_timeout()?;
                        return Ok(None);
                    }
                }

                // Check if we have enough votes to determine winner
                let winner = self.determine_winner()?;
                if let Some(winner_id) = winner {
                    self.complete_election(winner_id)?;
                    return Ok(Some(winner_id));
                }
            },
            _ => {},
        }

        Ok(None)
    }

    /// Handle election timeout
    pub fn handle_election_timeout(&mut self) -> Result<()> {
        let now = Instant::now();
        self.election_state = ElectionState::TimedOut { timed_out_at: now };
        self.statistics.timed_out_elections += 1;

        // Update election record
        if let Some(record) = self.statistics.election_history.last_mut() {
            record.end_time = Some(now);
            record.outcome = ElectionOutcome::TimedOut;
            if let Some(start_time) = self.election_start_time {
                record.duration = Some(now.duration_since(start_time));
            }
        }

        Ok(())
    }

    /// Complete election with winner
    pub fn complete_election(&mut self, winner: DeviceId) -> Result<()> {
        let now = Instant::now();
        self.current_leader = Some(winner);
        self.election_state = ElectionState::Completed {
            winner,
            completed_at: now,
        };
        self.statistics.successful_elections += 1;

        // Update statistics
        if let Some(start_time) = self.election_start_time {
            let duration = now.duration_since(start_time);
            self.update_election_time_statistics(duration);
        }

        // Update election record
        if let Some(record) = self.statistics.election_history.last_mut() {
            record.end_time = Some(now);
            record.winner = Some(winner);
            record.outcome = ElectionOutcome::Success { winner };
            if let Some(start_time) = self.election_start_time {
                record.duration = Some(now.duration_since(start_time));
            }
        }

        Ok(())
    }

    /// Determine election winner based on algorithm
    pub fn determine_winner(&self) -> Result<Option<DeviceId>> {
        match self.config.algorithm {
            ElectionAlgorithm::Bully => self.determine_winner_bully(),
            ElectionAlgorithm::Priority => self.determine_winner_priority(),
            ElectionAlgorithm::Performance => self.determine_winner_performance(),
            ElectionAlgorithm::Random => self.determine_winner_random(),
            ElectionAlgorithm::Raft => self.determine_winner_raft(),
            _ => Ok(None),
        }
    }

    /// Determine winner using bully algorithm
    fn determine_winner_bully(&self) -> Result<Option<DeviceId>> {
        // In bully algorithm, highest device ID wins
        Ok(self.candidates.keys().max().copied())
    }

    /// Determine winner using priority-based algorithm
    fn determine_winner_priority(&self) -> Result<Option<DeviceId>> {
        // Find candidate with highest priority
        let winner = self.candidates.iter()
            .filter(|(_, info)| info.status == CandidateStatus::Active)
            .max_by(|(_, a), (_, b)| a.priority.partial_cmp(&b.priority).unwrap())
            .map(|(id, _)| *id);
        Ok(winner)
    }

    /// Determine winner using performance-based algorithm
    fn determine_winner_performance(&self) -> Result<Option<DeviceId>> {
        // Calculate performance score and find best candidate
        let winner = self.candidates.iter()
            .filter(|(_, info)| info.status == CandidateStatus::Active)
            .max_by(|(_, a), (_, b)| {
                let score_a = self.calculate_performance_score(a);
                let score_b = self.calculate_performance_score(b);
                score_a.partial_cmp(&score_b).unwrap()
            })
            .map(|(id, _)| *id);
        Ok(winner)
    }

    /// Determine winner using random algorithm
    fn determine_winner_random(&self) -> Result<Option<DeviceId>> {
        use std::collections::hash_map::RandomState;
        use std::hash::{BuildHasher, Hasher};

        let active_candidates: Vec<_> = self.candidates.iter()
            .filter(|(_, info)| info.status == CandidateStatus::Active)
            .map(|(id, _)| *id)
            .collect();

        if active_candidates.is_empty() {
            return Ok(None);
        }

        // Use a deterministic random selection based on current time
        let hash_builder = RandomState::new();
        let mut hasher = hash_builder.build_hasher();
        hasher.write_u64(Instant::now().elapsed().as_nanos() as u64);
        let hash = hasher.finish();

        let index = (hash as usize) % active_candidates.len();
        Ok(Some(active_candidates[index]))
    }

    /// Determine winner using Raft-style majority vote
    fn determine_winner_raft(&self) -> Result<Option<DeviceId>> {
        let majority_threshold = (self.candidates.len() / 2) + 1;

        // Count votes for each candidate
        let mut vote_counts: HashMap<DeviceId, usize> = HashMap::new();
        for candidate_id in self.votes.values() {
            *vote_counts.entry(*candidate_id).or_insert(0) += 1;
        }

        // Find candidate with majority votes
        for (candidate_id, count) in vote_counts {
            if count >= majority_threshold {
                return Ok(Some(candidate_id));
            }
        }

        Ok(None)
    }

    /// Calculate performance score for candidate
    fn calculate_performance_score(&self, candidate: &CandidateInfo) -> f64 {
        let metrics = &candidate.performance;

        // Calculate weighted performance score (higher is better)
        let cpu_score = 1.0 - metrics.cpu_utilization; // Lower CPU usage is better
        let memory_score = 1.0 - metrics.memory_usage; // Lower memory usage is better
        let latency_score = 1.0 / (1.0 + metrics.network_latency.as_millis() as f64 / 1000.0);
        let availability_score = metrics.availability;
        let error_score = 1.0 - metrics.error_rate; // Lower error rate is better

        let weights = &self.config.priority_settings.weights;
        cpu_score * weights.performance +
        memory_score * weights.load +
        latency_score * weights.network +
        availability_score * weights.availability +
        error_score * weights.stability
    }

    /// Calculate priority for candidate
    fn calculate_priority(&self, candidate: &CandidateInfo) -> Result<f64> {
        match &self.config.priority_settings.calculation {
            PriorityCalculation::Static => Ok(candidate.device_id.0 as f64),
            PriorityCalculation::Performance => Ok(self.calculate_performance_score(candidate)),
            PriorityCalculation::Load => Ok(1.0 - candidate.performance.cpu_utilization),
            PriorityCalculation::Availability => Ok(candidate.performance.availability),
            _ => Ok(candidate.priority), // Use existing priority
        }
    }

    /// Check if candidate meets requirements
    fn meets_requirements(&self, candidate: &CandidateInfo) -> Result<bool> {
        let requirements = &self.config.candidate_requirements;

        // Check minimum uptime
        let uptime = candidate.last_heartbeat.duration_since(candidate.registration_time);
        if uptime < requirements.min_uptime {
            return Ok(false);
        }

        // Check minimum performance
        let performance_score = self.calculate_performance_score(candidate);
        if performance_score < requirements.min_performance {
            return Ok(false);
        }

        // Check resource utilization
        if candidate.performance.cpu_utilization > requirements.max_resource_utilization ||
           candidate.performance.memory_usage > requirements.max_resource_utilization {
            return Ok(false);
        }

        // Check required capabilities
        for required_capability in &requirements.required_capabilities {
            if !candidate.capabilities.contains(required_capability) {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Update election time statistics
    fn update_election_time_statistics(&mut self, duration: Duration) {
        let total_elections = self.statistics.successful_elections + self.statistics.failed_elections;

        if total_elections == 1 {
            self.statistics.avg_election_time = duration;
            self.statistics.shortest_election_time = duration;
            self.statistics.longest_election_time = duration;
        } else {
            // Update average
            let current_avg_millis = self.statistics.avg_election_time.as_millis() as f64;
            let new_duration_millis = duration.as_millis() as f64;
            let new_avg_millis = (current_avg_millis * (total_elections - 1) as f64 + new_duration_millis) / total_elections as f64;
            self.statistics.avg_election_time = Duration::from_millis(new_avg_millis as u64);

            // Update min/max
            if duration < self.statistics.shortest_election_time {
                self.statistics.shortest_election_time = duration;
            }
            if duration > self.statistics.longest_election_time {
                self.statistics.longest_election_time = duration;
            }
        }
    }

    /// Get current leader
    pub fn get_current_leader(&self) -> Option<DeviceId> {
        self.current_leader
    }

    /// Get election state
    pub fn get_election_state(&self) -> &ElectionState {
        &self.election_state
    }

    /// Get election statistics
    pub fn get_statistics(&self) -> &ElectionStatistics {
        &self.statistics
    }

    /// Reset election state
    pub fn reset_election(&mut self) {
        self.election_state = ElectionState::Idle;
        self.election_start_time = None;
        self.election_timeout = None;
        self.votes.clear();

        // Reset candidate vote counts
        for candidate in self.candidates.values_mut() {
            candidate.vote_count = 0;
        }
    }

    /// Update candidate performance metrics
    pub fn update_candidate_metrics(&mut self, device_id: DeviceId, metrics: PerformanceMetrics) -> Result<()> {
        if let Some(candidate) = self.candidates.get_mut(&device_id) {
            candidate.performance = metrics;
            candidate.last_heartbeat = Instant::now();

            // Recalculate priority if priority-based
            if self.config.priority_settings.enable {
                candidate.priority = self.calculate_priority(candidate)?;
            }
        }
        Ok(())
    }

    /// Remove inactive candidates
    pub fn cleanup_inactive_candidates(&mut self, inactive_threshold: Duration) {
        let now = Instant::now();
        let mut to_remove = Vec::new();

        for (device_id, candidate) in &mut self.candidates {
            if now.duration_since(candidate.last_heartbeat) > inactive_threshold {
                candidate.status = CandidateStatus::Inactive;
                to_remove.push(*device_id);
            }
        }

        for device_id in to_remove {
            self.unregister_candidate(device_id).ok();
        }
    }
}

impl ElectionStatistics {
    /// Create new election statistics
    pub fn new() -> Self {
        Self {
            total_elections: 0,
            successful_elections: 0,
            failed_elections: 0,
            timed_out_elections: 0,
            avg_election_time: Duration::from_millis(0),
            shortest_election_time: Duration::MAX,
            longest_election_time: Duration::from_millis(0),
            election_history: Vec::new(),
        }
    }

    /// Get success rate
    pub fn success_rate(&self) -> f64 {
        if self.total_elections > 0 {
            self.successful_elections as f64 / self.total_elections as f64
        } else {
            0.0
        }
    }

    /// Get failure rate
    pub fn failure_rate(&self) -> f64 {
        if self.total_elections > 0 {
            self.failed_elections as f64 / self.total_elections as f64
        } else {
            0.0
        }
    }

    /// Get timeout rate
    pub fn timeout_rate(&self) -> f64 {
        if self.total_elections > 0 {
            self.timed_out_elections as f64 / self.total_elections as f64
        } else {
            0.0
        }
    }
}

impl Default for ElectionStatistics {
    fn default() -> Self {
        Self::new()
    }
}

// Default implementations

impl Default for LeaderElectionConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_millis(300),
            heartbeat_interval: Duration::from_millis(50),
            algorithm: ElectionAlgorithm::Raft,
            priority_settings: PrioritySettings::default(),
            retry_settings: ElectionRetrySettings::default(),
            candidate_requirements: CandidateRequirements::default(),
        }
    }
}

impl Default for PrioritySettings {
    fn default() -> Self {
        Self {
            enable: false,
            calculation: PriorityCalculation::Static,
            dynamic_adjustment: false,
            weights: PriorityWeights::default(),
            decay: PriorityDecay::default(),
        }
    }
}

impl Default for PriorityWeights {
    fn default() -> Self {
        Self {
            performance: 0.3,
            availability: 0.25,
            load: 0.2,
            network: 0.15,
            stability: 0.1,
        }
    }
}

impl Default for PriorityDecay {
    fn default() -> Self {
        Self {
            enable: false,
            rate: 0.1,
            interval: Duration::from_minutes(1),
            minimum_priority: 0.1,
            function: DecayFunction::Linear,
        }
    }
}

impl Default for ElectionRetrySettings {
    fn default() -> Self {
        Self {
            max_retries: 3,
            retry_delay: Duration::from_millis(100),
            backoff_strategy: RetryBackoffStrategy::Exponential {
                base: 2.0,
                max_delay: Duration::from_secs(30),
            },
            jitter: JitterSettings::default(),
        }
    }
}

impl Default for JitterSettings {
    fn default() -> Self {
        Self {
            enable: true,
            jitter_type: JitterType::Additive,
            max_jitter: Duration::from_millis(50),
            distribution: JitterDistribution::Uniform,
        }
    }
}

impl Default for CandidateRequirements {
    fn default() -> Self {
        Self {
            min_uptime: Duration::from_secs(30),
            min_performance: 0.5,
            max_resource_utilization: 0.8,
            required_capabilities: vec![],
            health_requirements: HealthRequirements::default(),
        }
    }
}

impl Default for HealthRequirements {
    fn default() -> Self {
        Self {
            min_health_score: 0.7,
            check_interval: Duration::from_secs(10),
            required_metrics: vec!["cpu".to_string(), "memory".to_string()],
            history_requirements: HealthHistoryRequirements::default(),
        }
    }
}

impl Default for HealthHistoryRequirements {
    fn default() -> Self {
        Self {
            window: Duration::from_minutes(10),
            min_uptime_percentage: 0.95,
            max_failure_rate: 0.05,
            min_response_rate: 0.9,
        }
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            cpu_utilization: 0.0,
            memory_usage: 0.0,
            network_latency: Duration::from_millis(0),
            availability: 1.0,
            response_time: Duration::from_millis(0),
            throughput: 0.0,
            error_rate: 0.0,
        }
    }
}

/// Leader election builder for easy configuration
pub struct LeaderElectionConfigBuilder {
    config: LeaderElectionConfig,
}

impl LeaderElectionConfigBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: LeaderElectionConfig::default(),
        }
    }

    /// Set election algorithm
    pub fn with_algorithm(mut self, algorithm: ElectionAlgorithm) -> Self {
        self.config.algorithm = algorithm;
        self
    }

    /// Set election timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.config.timeout = timeout;
        self
    }

    /// Set heartbeat interval
    pub fn with_heartbeat_interval(mut self, interval: Duration) -> Self {
        self.config.heartbeat_interval = interval;
        self
    }

    /// Enable priority-based election
    pub fn with_priority_based(mut self, calculation: PriorityCalculation) -> Self {
        self.config.priority_settings.enable = true;
        self.config.priority_settings.calculation = calculation;
        self
    }

    /// Set candidate requirements
    pub fn with_requirements(mut self, requirements: CandidateRequirements) -> Self {
        self.config.candidate_requirements = requirements;
        self
    }

    /// Build the final configuration
    pub fn build(self) -> LeaderElectionConfig {
        self.config
    }
}

impl Default for LeaderElectionConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Election configuration presets
pub struct ElectionPresets;

impl ElectionPresets {
    /// Fast election configuration for low-latency scenarios
    pub fn fast() -> LeaderElectionConfig {
        LeaderElectionConfigBuilder::new()
            .with_algorithm(ElectionAlgorithm::Bully)
            .with_timeout(Duration::from_millis(100))
            .with_heartbeat_interval(Duration::from_millis(20))
            .build()
    }

    /// Reliable election configuration with comprehensive requirements
    pub fn reliable() -> LeaderElectionConfig {
        LeaderElectionConfigBuilder::new()
            .with_algorithm(ElectionAlgorithm::Priority)
            .with_timeout(Duration::from_secs(5))
            .with_priority_based(PriorityCalculation::Performance)
            .build()
    }

    /// Performance-based election configuration
    pub fn performance_based() -> LeaderElectionConfig {
        LeaderElectionConfigBuilder::new()
            .with_algorithm(ElectionAlgorithm::Performance)
            .with_timeout(Duration::from_secs(2))
            .with_priority_based(PriorityCalculation::Composite {
                factors: vec![
                    PriorityFactor {
                        name: "cpu".to_string(),
                        factor_type: PriorityFactorType::CpuUtilization,
                        weight: 0.4,
                        normalization: NormalizationMethod::MinMax { min: 0.0, max: 1.0 },
                    },
                    PriorityFactor {
                        name: "memory".to_string(),
                        factor_type: PriorityFactorType::MemoryUsage,
                        weight: 0.3,
                        normalization: NormalizationMethod::MinMax { min: 0.0, max: 1.0 },
                    },
                    PriorityFactor {
                        name: "availability".to_string(),
                        factor_type: PriorityFactorType::Availability,
                        weight: 0.3,
                        normalization: NormalizationMethod::MinMax { min: 0.0, max: 1.0 },
                    },
                ]
            })
            .build()
    }
}