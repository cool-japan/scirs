//! Advanced load balancing algorithms for distributed clustering
//!
//! This module provides various load balancing strategies to optimize
//! resource utilization and performance across worker nodes.

use ndarray::Array1;
use num_traits::Float;
use rand::Rng;
use std::collections::HashMap;
use std::fmt::Debug;

use crate::error::{ClusteringError, Result};

/// Advanced load balancing coordinator
#[derive(Debug)]
pub struct LoadBalancingCoordinator {
    pub worker_profiles: HashMap<usize, WorkerProfile>,
    pub load_history: Vec<LoadBalanceSnapshot>,
    pub config: LoadBalancingConfig,
    pub current_strategy: LoadBalancingStrategy,
}

/// Worker performance profile
#[derive(Debug, Clone)]
pub struct WorkerProfile {
    pub worker_id: usize,
    pub cpu_cores: usize,
    pub memory_gb: f64,
    pub network_bandwidth_mbps: f64,
    pub historical_throughput: f64,
    pub reliability_score: f64,
    pub processing_efficiency: f64,
    pub communication_latency_ms: f64,
}

/// Load balancing configuration
#[derive(Debug, Clone)]
pub struct LoadBalancingConfig {
    pub enable_dynamic_balancing: bool,
    pub rebalance_threshold: f64,
    pub min_rebalance_interval_ms: u64,
    pub max_migration_size: usize,
    pub consider_network_topology: bool,
    pub fairness_weight: f64,
    pub efficiency_weight: f64,
    pub stability_weight: f64,
}

impl Default for LoadBalancingConfig {
    fn default() -> Self {
        Self {
            enable_dynamic_balancing: true,
            rebalance_threshold: 0.2,
            min_rebalance_interval_ms: 30000,
            max_migration_size: 1000,
            consider_network_topology: false,
            fairness_weight: 0.4,
            efficiency_weight: 0.4,
            stability_weight: 0.2,
        }
    }
}

/// Available load balancing strategies
#[derive(Debug, Clone)]
pub enum LoadBalancingStrategy {
    /// Proportional to worker capacity
    ProportionalCapacity,
    /// Game-theoretic Nash equilibrium
    GameTheoretic {
        convergence_threshold: f64,
        max_iterations: usize,
    },
    /// Reinforcement learning based
    AdaptiveLearning {
        learning_rate: f64,
        exploration_rate: f64,
    },
    /// Multi-objective optimization
    MultiObjective {
        objectives: Vec<OptimizationObjective>,
        weights: Vec<f64>,
    },
    /// Round-robin with capacity awareness
    WeightedRoundRobin,
    /// Least loaded first
    LeastLoaded,
}

/// Optimization objectives for multi-objective balancing
#[derive(Debug, Clone)]
pub enum OptimizationObjective {
    MinimizeTotalTime,
    MaximizeThroughput,
    MinimizeCommunication,
    MaximizeReliability,
    MinimizeEnergyConsumption,
    MaximizeResourceUtilization,
}

/// Snapshot of load balance state
#[derive(Debug, Clone)]
pub struct LoadBalanceSnapshot {
    pub timestamp: u64,
    pub worker_loads: HashMap<usize, f64>,
    pub load_variance: f64,
    pub total_throughput: f64,
    pub rebalance_triggered: bool,
    pub migration_count: usize,
}

/// Load balancing decision
#[derive(Debug, Clone)]
pub struct LoadBalanceDecision {
    pub should_rebalance: bool,
    pub new_assignments: HashMap<usize, usize>,
    pub migrations: Vec<DataMigration>,
    pub expected_improvement: f64,
}

/// Data migration instruction
#[derive(Debug, Clone)]
pub struct DataMigration {
    pub from_worker: usize,
    pub to_worker: usize,
    pub datasize: usize,
    pub priority: MigrationPriority,
}

/// Migration priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MigrationPriority {
    Critical,
    High,
    Normal,
    Low,
}

impl LoadBalancingCoordinator {
    /// Create new load balancing coordinator
    pub fn new(config: LoadBalancingConfig) -> Self {
        Self {
            worker_profiles: HashMap::new(),
            load_history: Vec::new(),
            config,
            current_strategy: LoadBalancingStrategy::ProportionalCapacity,
        }
    }

    /// Register worker with performance profile
    pub fn register_worker(&mut self, profile: WorkerProfile) {
        self.worker_profiles.insert(profile.worker_id, profile);
    }

    /// Update worker profile with recent performance data
    pub fn update_worker_profile(
        &mut self,
        worker_id: usize,
        throughput: f64,
        efficiency: f64,
        latency_ms: f64,
    ) -> Result<()> {
        if let Some(profile) = self.worker_profiles.get_mut(&worker_id) {
            // Exponential moving average for smoothing
            let alpha = 0.3;
            profile.historical_throughput =
                alpha * throughput + (1.0 - alpha) * profile.historical_throughput;
            profile.processing_efficiency =
                alpha * efficiency + (1.0 - alpha) * profile.processing_efficiency;
            profile.communication_latency_ms =
                alpha * latency_ms + (1.0 - alpha) * profile.communication_latency_ms;
        } else {
            return Err(ClusteringError::InvalidInput(format!(
                "Worker {} not registered",
                worker_id
            )));
        }
        Ok(())
    }

    /// Evaluate current load balance and decide if rebalancing is needed
    pub fn evaluate_balance(
        &mut self,
        current_assignments: &HashMap<usize, usize>,
        datasize: usize,
    ) -> Result<LoadBalanceDecision> {
        // Calculate current load distribution
        let current_loads = self.calculate_current_loads(current_assignments, datasize);
        let load_variance = self.calculate_load_variance(&current_loads);

        // Record current state
        let snapshot = LoadBalanceSnapshot {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            worker_loads: current_loads.clone(),
            load_variance,
            total_throughput: self.calculate_total_throughput(&current_loads),
            rebalance_triggered: false,
            migration_count: 0,
        };
        self.load_history.push(snapshot);

        // Keep history manageable
        if self.load_history.len() > 100 {
            self.load_history.remove(0);
        }

        let should_rebalance = self.should_trigger_rebalance(load_variance);

        if should_rebalance {
            let new_assignments = self.compute_optimal_assignments(datasize)?;
            let migrations = self.plan_data_migrations(current_assignments, &new_assignments);
            let expected_improvement =
                self.estimate_improvement(&current_loads, &new_assignments, datasize);

            Ok(LoadBalanceDecision {
                should_rebalance: true,
                new_assignments,
                migrations,
                expected_improvement,
            })
        } else {
            Ok(LoadBalanceDecision {
                should_rebalance: false,
                new_assignments: current_assignments.clone(),
                migrations: Vec::new(),
                expected_improvement: 0.0,
            })
        }
    }

    /// Calculate current load distribution
    fn calculate_current_loads(
        &self,
        assignments: &HashMap<usize, usize>,
        totaldata: usize,
    ) -> HashMap<usize, f64> {
        let mut loads = HashMap::new();

        // Initialize all workers with zero load
        for &worker_id in self.worker_profiles.keys() {
            loads.insert(worker_id, 0.0);
        }

        // Calculate actual loads
        for (&worker_id, &assigned_data) in assignments {
            if totaldata > 0 {
                loads.insert(worker_id, assigned_data as f64 / totaldata as f64);
            }
        }

        loads
    }

    /// Calculate load variance across workers
    fn calculate_load_variance(&self, loads: &HashMap<usize, f64>) -> f64 {
        if loads.is_empty() {
            return 0.0;
        }

        let mean_load = loads.values().sum::<f64>() / loads.len() as f64;
        let variance = loads
            .values()
            .map(|&load| (load - mean_load).powi(2))
            .sum::<f64>()
            / loads.len() as f64;

        variance.sqrt()
    }

    /// Calculate total system throughput
    fn calculate_total_throughput(&self, loads: &HashMap<usize, f64>) -> f64 {
        loads
            .iter()
            .map(|(&worker_id, &load)| {
                if let Some(profile) = self.worker_profiles.get(&worker_id) {
                    load * profile.historical_throughput
                } else {
                    0.0
                }
            })
            .sum()
    }

    /// Determine if rebalancing should be triggered
    fn should_trigger_rebalance(&self, loadvariance: f64) -> bool {
        if !self.config.enable_dynamic_balancing {
            return false;
        }

        // Check if enough time has passed since last rebalance
        if let Some(last_snapshot) = self.load_history.last() {
            let time_since_last = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64
                - last_snapshot.timestamp;

            if time_since_last < self.config.min_rebalance_interval_ms {
                return false;
            }
        }

        loadvariance > self.config.rebalance_threshold
    }

    /// Compute optimal work assignments using selected strategy
    fn compute_optimal_assignments(&mut self, datasize: usize) -> Result<HashMap<usize, usize>> {
        match &self.current_strategy {
            LoadBalancingStrategy::ProportionalCapacity => {
                self.proportional_capacity_balancing(datasize)
            }
            LoadBalancingStrategy::GameTheoretic {
                convergence_threshold,
                max_iterations,
            } => self.game_theoretic_balancing(datasize, *convergence_threshold, *max_iterations),
            LoadBalancingStrategy::AdaptiveLearning {
                learning_rate,
                exploration_rate,
            } => {
                let current_assignments = HashMap::new(); // Would get from state
                self.adaptive_balancing(
                    datasize,
                    &current_assignments,
                    *learning_rate,
                    *exploration_rate,
                )
            }
            LoadBalancingStrategy::MultiObjective {
                objectives,
                weights,
            } => self.multi_objective_balancing(datasize, objectives, weights),
            LoadBalancingStrategy::WeightedRoundRobin => {
                self.weighted_round_robin_balancing(datasize)
            }
            LoadBalancingStrategy::LeastLoaded => self.least_loaded_balancing(datasize),
        }
    }

    /// Proportional capacity-based load balancing
    fn proportional_capacity_balancing(&self, datasize: usize) -> Result<HashMap<usize, usize>> {
        let worker_efficiency: Vec<(usize, f64)> = self
            .worker_profiles
            .iter()
            .map(|(&id, profile)| {
                let capacity_score = profile.cpu_cores as f64 * profile.memory_gb;
                let efficiency_score = profile.processing_efficiency * profile.reliability_score;
                let latency_penalty = 1.0 / (1.0 + profile.communication_latency_ms / 100.0);
                (id, capacity_score * efficiency_score * latency_penalty)
            })
            .collect();

        if worker_efficiency.is_empty() {
            return Ok(HashMap::new());
        }

        let mut new_assignments = HashMap::new();
        let total_efficiency: f64 = worker_efficiency.iter().map(|(_, eff)| eff).sum();
        let mut remaining_data = datasize;

        for (i, (worker_id, efficiency)) in worker_efficiency.iter().enumerate() {
            let assignment = if i == worker_efficiency.len() - 1 {
                remaining_data // Last worker gets remaining data
            } else {
                let proportion = efficiency / total_efficiency;
                let assignment = (datasize as f64 * proportion).round() as usize;
                assignment.min(remaining_data)
            };

            new_assignments.insert(*worker_id, assignment);
            remaining_data = remaining_data.saturating_sub(assignment);
        }

        Ok(new_assignments)
    }

    /// Game theoretic load balancing using Nash equilibrium
    fn game_theoretic_balancing(
        &self,
        datasize: usize,
        convergence_threshold: f64,
        max_iterations: usize,
    ) -> Result<HashMap<usize, usize>> {
        let mut assignments = HashMap::new();
        let worker_ids: Vec<usize> = self.worker_profiles.keys().copied().collect();

        // Initialize with equal distribution
        let base_assignment = datasize / worker_ids.len();
        let remainder = datasize % worker_ids.len();

        for (i, &worker_id) in worker_ids.iter().enumerate() {
            let assignment = base_assignment + if i < remainder { 1 } else { 0 };
            assignments.insert(worker_id, assignment);
        }

        // Iterate to find Nash equilibrium
        for _iteration in 0..max_iterations {
            let mut converged = true;
            let _old_assignments = assignments.clone();

            // Each worker adjusts their load based on others' decisions
            for &worker_id in &worker_ids {
                let optimal_load = self.compute_best_response(worker_id, &assignments, datasize);
                let current_load = assignments[&worker_id];

                if (optimal_load as f64 - current_load as f64).abs() / current_load as f64
                    > convergence_threshold
                {
                    assignments.insert(worker_id, optimal_load);
                    converged = false;
                }
            }

            // Normalize to ensure total equals datasize
            let total_assigned: usize = assignments.values().sum();
            if total_assigned != datasize {
                let adjustment_factor = datasize as f64 / total_assigned as f64;
                for assignment in assignments.values_mut() {
                    *assignment = (*assignment as f64 * adjustment_factor).round() as usize;
                }
            }

            if converged {
                break;
            }
        }

        Ok(assignments)
    }

    /// Compute best response for a worker in game theoretic setting
    fn compute_best_response(
        &self,
        worker_id: usize,
        current_assignments: &HashMap<usize, usize>,
        totaldata: usize,
    ) -> usize {
        let _profile = self.worker_profiles.get(&worker_id).unwrap();

        // Utility function considers throughput, reliability, and coordination cost
        let mut best_assignment = current_assignments[&worker_id];
        let mut best_utility =
            self.compute_worker_utility(worker_id, best_assignment, current_assignments);

        // Try different assignment levels
        let current = current_assignments[&worker_id];
        let others_total: usize = current_assignments
            .iter()
            .filter(|(&id, _)| id != worker_id)
            .map(|(_, &assignment)| assignment)
            .sum();

        let max_possible = totaldata.saturating_sub(others_total);

        for test_assignment in 0..=max_possible.min(current * 2) {
            let utility =
                self.compute_worker_utility(worker_id, test_assignment, current_assignments);
            if utility > best_utility {
                best_utility = utility;
                best_assignment = test_assignment;
            }
        }

        best_assignment
    }

    /// Compute utility for a worker's assignment
    fn compute_worker_utility(
        &self,
        worker_id: usize,
        assignment: usize,
        all_assignments: &HashMap<usize, usize>,
    ) -> f64 {
        let profile = self.worker_profiles.get(&worker_id).unwrap();

        // Throughput component
        let load_factor = assignment as f64 / (profile.memory_gb * 1000.0); // Rough capacity estimate
        let throughput_utility = profile.historical_throughput * (1.0 - load_factor.min(1.0));

        // Reliability component
        let reliability_utility = profile.reliability_score * (1.0 - load_factor * 0.5);

        // Communication overhead (increases with imbalance)
        let avg_assignment: f64 =
            all_assignments.values().map(|&v| v as f64).sum::<f64>() / all_assignments.len() as f64;
        let imbalance = (assignment as f64 - avg_assignment).abs() / avg_assignment;
        let communication_penalty = imbalance * 0.2;

        throughput_utility + reliability_utility - communication_penalty
    }

    /// Adaptive balancing using reinforcement learning principles
    fn adaptive_balancing(
        &mut self,
        datasize: usize,
        current_assignments: &HashMap<usize, usize>,
        learning_rate: f64,
        exploration_rate: f64,
    ) -> Result<HashMap<usize, usize>> {
        let mut new_assignments = current_assignments.clone();

        // ε-greedy exploration strategy
        let mut rng = rand::rng();

        for (&worker_id, &current_assignment) in current_assignments {
            if rng.gen::<f64>() < exploration_rate {
                // Explore: random adjustment
                let max_change = (current_assignment as f64 * 0.2) as usize; // Max 20% change
                let change = rng.random_range(0..=max_change * 2) as i32 - max_change as i32;
                let new_assignment = (current_assignment as i32 + change).max(0) as usize;
                new_assignments.insert(worker_id, new_assignment);
            } else {
                // Exploit: use learned policy
                let optimal_assignment =
                    self.compute_learned_optimal_assignment(worker_id, datasize);

                // Apply learning rate to smooth transitions
                let adjusted_assignment = current_assignment as f64
                    + learning_rate * (optimal_assignment as f64 - current_assignment as f64);
                new_assignments.insert(worker_id, adjusted_assignment.round() as usize);
            }
        }

        // Normalize assignments to match total data size
        let total_assigned: usize = new_assignments.values().sum();
        if total_assigned != datasize && total_assigned > 0 {
            let scale_factor = datasize as f64 / total_assigned as f64;
            for assignment in new_assignments.values_mut() {
                *assignment = (*assignment as f64 * scale_factor).round() as usize;
            }
        }

        Ok(new_assignments)
    }

    /// Multi-objective optimization for load balancing
    fn multi_objective_balancing(
        &self,
        datasize: usize,
        objectives: &[OptimizationObjective],
        weights: &[f64],
    ) -> Result<HashMap<usize, usize>> {
        let worker_ids: Vec<usize> = self.worker_profiles.keys().copied().collect();
        let _n_workers = worker_ids.len();

        // Generate Pareto-optimal solutions using weighted sum approach
        let mut best_assignment = HashMap::new();
        let mut best_score = f64::NEG_INFINITY;

        // Try different assignment combinations
        for _ in 0..1000 {
            // Monte Carlo sampling
            let mut trial_assignment = HashMap::new();
            let mut remaining_data = datasize;

            // Random assignment generation
            let mut rng = rand::rng();
            for (i, &worker_id) in worker_ids.iter().enumerate() {
                let assignment = if i == worker_ids.len() - 1 {
                    remaining_data
                } else {
                    let max_assignment = remaining_data.min(datasize / 2);
                    let assignment = rng.random_range(0..=max_assignment);
                    assignment.min(remaining_data)
                };

                trial_assignment.insert(worker_id, assignment);
                remaining_data = remaining_data.saturating_sub(assignment);
            }

            // Evaluate multi-objective score
            let score = self.evaluate_multi_objective_score(&trial_assignment, objectives, weights);
            if score > best_score {
                best_score = score;
                best_assignment = trial_assignment;
            }
        }

        if best_assignment.is_empty() {
            // Fallback to proportional balancing
            return self.proportional_capacity_balancing(datasize);
        }

        Ok(best_assignment)
    }

    /// Weighted round-robin balancing
    fn weighted_round_robin_balancing(&self, datasize: usize) -> Result<HashMap<usize, usize>> {
        let mut assignments = HashMap::new();
        let worker_weights: Vec<(usize, f64)> = self
            .worker_profiles
            .iter()
            .map(|(&id, profile)| {
                (
                    id,
                    profile.processing_efficiency * profile.reliability_score,
                )
            })
            .collect();

        if worker_weights.is_empty() {
            return Ok(assignments);
        }

        let total_weight: f64 = worker_weights.iter().map(|(_, w)| w).sum();
        let mut remaining_data = datasize;

        for (i, (worker_id, weight)) in worker_weights.iter().enumerate() {
            let assignment = if i == worker_weights.len() - 1 {
                remaining_data
            } else {
                let proportion = weight / total_weight;
                let assignment = (datasize as f64 * proportion).round() as usize;
                assignment.min(remaining_data)
            };

            assignments.insert(*worker_id, assignment);
            remaining_data = remaining_data.saturating_sub(assignment);
        }

        Ok(assignments)
    }

    /// Least loaded balancing strategy
    fn least_loaded_balancing(&self, datasize: usize) -> Result<HashMap<usize, usize>> {
        let mut assignments = HashMap::new();
        let worker_ids: Vec<usize> = self.worker_profiles.keys().copied().collect();

        // Initialize with zero assignments
        for &worker_id in &worker_ids {
            assignments.insert(worker_id, 0);
        }

        // Distribute data one unit at a time to least loaded worker
        for _ in 0..datasize {
            // Find worker with minimum current load (considering capacity)
            let mut min_normalized_load = f64::INFINITY;
            let mut best_worker = worker_ids[0];

            for &worker_id in &worker_ids {
                if let Some(profile) = self.worker_profiles.get(&worker_id) {
                    let current_load = assignments[&worker_id] as f64;
                    let capacity = profile.cpu_cores as f64 * profile.memory_gb;
                    let normalized_load = if capacity > 0.0 {
                        current_load / capacity
                    } else {
                        current_load
                    };

                    if normalized_load < min_normalized_load {
                        min_normalized_load = normalized_load;
                        best_worker = worker_id;
                    }
                }
            }

            *assignments.get_mut(&best_worker).unwrap() += 1;
        }

        Ok(assignments)
    }

    /// Compute learned optimal assignment for adaptive strategy
    fn compute_learned_optimal_assignment(&self, worker_id: usize, totaldata: usize) -> usize {
        // Simplified learning model - in practice would use historical performance data
        if let Some(profile) = self.worker_profiles.get(&worker_id) {
            let capacity_ratio = (profile.cpu_cores as f64 * profile.memory_gb) / 100.0; // Normalize
            let efficiency_factor = profile.processing_efficiency * profile.reliability_score;
            let optimal_ratio = capacity_ratio * efficiency_factor;

            (totaldata as f64 * optimal_ratio / self.worker_profiles.len() as f64).round() as usize
        } else {
            totaldata / self.worker_profiles.len()
        }
    }

    /// Evaluate multi-objective score for an assignment
    fn evaluate_multi_objective_score(
        &self,
        assignment: &HashMap<usize, usize>,
        objectives: &[OptimizationObjective],
        weights: &[f64],
    ) -> f64 {
        let mut total_score = 0.0;

        for (objective, &weight) in objectives.iter().zip(weights.iter()) {
            let objective_score = match objective {
                OptimizationObjective::MinimizeTotalTime => {
                    self.evaluate_total_time_objective(assignment)
                }
                OptimizationObjective::MaximizeThroughput => {
                    self.evaluate_throughput_objective(assignment)
                }
                OptimizationObjective::MinimizeCommunication => {
                    self.evaluate_communication_objective(assignment)
                }
                OptimizationObjective::MaximizeReliability => {
                    self.evaluate_reliability_objective(assignment)
                }
                OptimizationObjective::MinimizeEnergyConsumption => {
                    self.evaluate_energy_objective(assignment)
                }
                OptimizationObjective::MaximizeResourceUtilization => {
                    self.evaluate_utilization_objective(assignment)
                }
            };

            total_score += weight * objective_score;
        }

        total_score
    }

    /// Evaluate total time objective
    fn evaluate_total_time_objective(&self, assignment: &HashMap<usize, usize>) -> f64 {
        let max_time = assignment
            .iter()
            .map(|(&worker_id, &load)| {
                if let Some(profile) = self.worker_profiles.get(&worker_id) {
                    if profile.historical_throughput > 0.0 {
                        load as f64 / profile.historical_throughput
                    } else {
                        f64::INFINITY
                    }
                } else {
                    f64::INFINITY
                }
            })
            .fold(0.0, f64::max);

        1.0 / (1.0 + max_time) // Convert to maximization objective
    }

    /// Evaluate throughput objective
    fn evaluate_throughput_objective(&self, assignment: &HashMap<usize, usize>) -> f64 {
        assignment
            .iter()
            .map(|(&worker_id, &load)| {
                if let Some(profile) = self.worker_profiles.get(&worker_id) {
                    load as f64 * profile.historical_throughput
                } else {
                    0.0
                }
            })
            .sum()
    }

    /// Evaluate communication objective (simplified)
    fn evaluate_communication_objective(&self, assignment: &HashMap<usize, usize>) -> f64 {
        let avg_load = assignment.values().sum::<usize>() as f64 / assignment.len() as f64;
        let variance = assignment
            .values()
            .map(|&load| (load as f64 - avg_load).powi(2))
            .sum::<f64>()
            / assignment.len() as f64;

        1.0 / (1.0 + variance.sqrt()) // Lower variance = better communication
    }

    /// Evaluate reliability objective
    fn evaluate_reliability_objective(&self, assignment: &HashMap<usize, usize>) -> f64 {
        assignment
            .iter()
            .map(|(&worker_id, &load)| {
                if let Some(profile) = self.worker_profiles.get(&worker_id) {
                    load as f64 * profile.reliability_score
                } else {
                    0.0
                }
            })
            .sum()
    }

    /// Evaluate energy objective (simplified)
    fn evaluate_energy_objective(&self, assignment: &HashMap<usize, usize>) -> f64 {
        assignment
            .iter()
            .map(|(&worker_id, &load)| {
                if let Some(profile) = self.worker_profiles.get(&worker_id) {
                    // Simplified energy model: quadratic in load
                    let normalized_load = load as f64 / (profile.memory_gb * 1000.0);
                    normalized_load.powi(2)
                } else {
                    0.0
                }
            })
            .sum()
    }

    /// Evaluate resource utilization objective
    fn evaluate_utilization_objective(&self, assignment: &HashMap<usize, usize>) -> f64 {
        assignment
            .iter()
            .map(|(&worker_id, &load)| {
                if let Some(profile) = self.worker_profiles.get(&worker_id) {
                    let capacity = profile.cpu_cores as f64 * profile.memory_gb;
                    if capacity > 0.0 {
                        (load as f64 / capacity).min(1.0)
                    } else {
                        0.0
                    }
                } else {
                    0.0
                }
            })
            .sum::<f64>()
            / assignment.len() as f64
    }

    /// Plan data migrations between workers
    fn plan_data_migrations(
        &self,
        current: &HashMap<usize, usize>,
        target: &HashMap<usize, usize>,
    ) -> Vec<DataMigration> {
        let mut migrations = Vec::new();
        let mut surplus_workers = Vec::new();
        let mut deficit_workers = Vec::new();

        // Identify workers with surplus or deficit
        for (&worker_id, &current_load) in current {
            let target_load = target.get(&worker_id).copied().unwrap_or(0);

            if current_load > target_load {
                surplus_workers.push((worker_id, current_load - target_load));
            } else if current_load < target_load {
                deficit_workers.push((worker_id, target_load - current_load));
            }
        }

        // Create migration plans
        let mut surplus_idx = 0;
        let mut deficit_idx = 0;

        while surplus_idx < surplus_workers.len() && deficit_idx < deficit_workers.len() {
            let (surplus_worker, mut surplus_amount) = surplus_workers[surplus_idx];
            let (deficit_worker, mut deficit_amount) = deficit_workers[deficit_idx];

            let migration_size = surplus_amount
                .min(deficit_amount)
                .min(self.config.max_migration_size);

            if migration_size > 0 {
                let priority = if migration_size > self.config.max_migration_size / 2 {
                    MigrationPriority::High
                } else {
                    MigrationPriority::Normal
                };

                migrations.push(DataMigration {
                    from_worker: surplus_worker,
                    to_worker: deficit_worker,
                    datasize: migration_size,
                    priority,
                });

                surplus_amount -= migration_size;
                deficit_amount -= migration_size;

                // Update remaining amounts
                surplus_workers[surplus_idx].1 = surplus_amount;
                deficit_workers[deficit_idx].1 = deficit_amount;

                // Move to next worker if current one is balanced
                if surplus_amount == 0 {
                    surplus_idx += 1;
                }
                if deficit_amount == 0 {
                    deficit_idx += 1;
                }
            } else {
                break;
            }
        }

        migrations
    }

    /// Estimate improvement from rebalancing
    fn estimate_improvement(
        &self,
        current_loads: &HashMap<usize, f64>,
        new_assignments: &HashMap<usize, usize>,
        totaldata: usize,
    ) -> f64 {
        let current_variance = self.calculate_load_variance(current_loads);

        let new_loads = self.calculate_current_loads(new_assignments, totaldata);
        let new_variance = self.calculate_load_variance(&new_loads);

        let current_throughput = self.calculate_total_throughput(current_loads);
        let new_throughput = self.calculate_total_throughput(&new_loads);

        // Weighted improvement score
        let variance_improvement = (current_variance - new_variance) / current_variance.max(0.001);
        let throughput_improvement =
            (new_throughput - current_throughput) / current_throughput.max(0.001);

        self.config.efficiency_weight * throughput_improvement
            + self.config.fairness_weight * variance_improvement
    }

    /// Set load balancing strategy
    pub fn set_strategy(&mut self, strategy: LoadBalancingStrategy) {
        self.current_strategy = strategy;
    }

    /// Get load balancing history
    pub fn get_load_history(&self) -> &[LoadBalanceSnapshot] {
        &self.load_history
    }

    /// Get worker profiles
    pub fn get_worker_profiles(&self) -> &HashMap<usize, WorkerProfile> {
        &self.worker_profiles
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_balancing_coordinator_creation() {
        let config = LoadBalancingConfig::default();
        let coordinator = LoadBalancingCoordinator::new(config);

        assert!(coordinator.worker_profiles.is_empty());
        assert!(coordinator.load_history.is_empty());
    }

    #[test]
    fn test_worker_profile_registration() {
        let config = LoadBalancingConfig::default();
        let mut coordinator = LoadBalancingCoordinator::new(config);

        let profile = WorkerProfile {
            worker_id: 1,
            cpu_cores: 4,
            memory_gb: 8.0,
            network_bandwidth_mbps: 1000.0,
            historical_throughput: 100.0,
            reliability_score: 0.95,
            processing_efficiency: 0.8,
            communication_latency_ms: 10.0,
        };

        coordinator.register_worker(profile);
        assert!(coordinator.worker_profiles.contains_key(&1));
    }

    #[test]
    fn test_load_variance_calculation() {
        let config = LoadBalancingConfig::default();
        let coordinator = LoadBalancingCoordinator::new(config);

        let mut loads = HashMap::new();
        loads.insert(1, 0.5);
        loads.insert(2, 0.5);

        let variance = coordinator.calculate_load_variance(&loads);
        assert!((variance - 0.0).abs() < 0.001); // Perfect balance

        loads.insert(2, 0.7);
        loads.insert(1, 0.3);
        let variance = coordinator.calculate_load_variance(&loads);
        assert!(variance > 0.0); // Imbalanced
    }

    #[test]
    fn test_proportional_capacity_balancing() {
        let config = LoadBalancingConfig::default();
        let mut coordinator = LoadBalancingCoordinator::new(config);

        // Add workers with different capacities
        let profile1 = WorkerProfile {
            worker_id: 1,
            cpu_cores: 2,
            memory_gb: 4.0,
            network_bandwidth_mbps: 1000.0,
            historical_throughput: 50.0,
            reliability_score: 0.9,
            processing_efficiency: 0.8,
            communication_latency_ms: 10.0,
        };

        let profile2 = WorkerProfile {
            worker_id: 2,
            cpu_cores: 4,
            memory_gb: 8.0,
            network_bandwidth_mbps: 1000.0,
            historical_throughput: 100.0,
            reliability_score: 0.95,
            processing_efficiency: 0.9,
            communication_latency_ms: 5.0,
        };

        coordinator.register_worker(profile1);
        coordinator.register_worker(profile2);

        let assignments = coordinator.proportional_capacity_balancing(1000).unwrap();
        assert_eq!(assignments.len(), 2);
        assert!(assignments.values().sum::<usize>() == 1000);

        // Worker 2 should get more work due to higher capacity
        assert!(assignments[&2] > assignments[&1]);
    }
}
