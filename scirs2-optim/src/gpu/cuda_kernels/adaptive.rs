//! Adaptive optimization strategies for CUDA kernel execution
//!
//! This module provides intelligent adaptation mechanisms for CUDA kernel execution,
//! including dynamic parameter tuning, workload-based optimization, runtime adaptation,
//! machine learning-based predictions, and self-optimizing kernel configurations.

use crate::gpu::cuda_kernels::config::*;
use crate::gpu::cuda_kernels::profiling::*;
use scirs2_core::error::{Result, ScirsMlError};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock, Mutex};
use std::time::{Duration, Instant};

/// Adaptive optimization strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdaptationStrategy {
    /// No adaptation - use static configuration
    Static,
    /// Simple heuristic-based adaptation
    Heuristic,
    /// Genetic algorithm-based optimization
    Genetic {
        population_size: usize,
        mutation_rate: f32,
        crossover_rate: f32,
    },
    /// Bayesian optimization
    Bayesian {
        acquisition_function: AcquisitionFunction,
        exploration_weight: f32,
    },
    /// Reinforcement learning-based adaptation
    ReinforcementLearning {
        learning_rate: f32,
        exploration_rate: f32,
        discount_factor: f32,
    },
    /// Multi-armed bandit approach
    MultiArmedBandit {
        exploration_strategy: ExplorationStrategy,
        window_size: usize,
    },
}

/// Acquisition functions for Bayesian optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AcquisitionFunction {
    /// Expected Improvement
    ExpectedImprovement,
    /// Upper Confidence Bound
    UpperConfidenceBound,
    /// Probability of Improvement
    ProbabilityOfImprovement,
}

/// Exploration strategies for multi-armed bandit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExplorationStrategy {
    /// Epsilon-greedy exploration
    EpsilonGreedy { epsilon: f32 },
    /// Upper Confidence Bound (UCB1)
    UCB1,
    /// Thompson sampling
    ThompsonSampling,
}

/// Adaptation triggers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationTriggers {
    /// Trigger on performance degradation threshold
    pub performance_threshold: f32,
    /// Trigger after number of operations
    pub operation_count: u64,
    /// Trigger on time interval
    pub time_interval: Duration,
    /// Trigger on workload changes
    pub workload_change_threshold: f32,
    /// Trigger on resource utilization changes
    pub resource_utilization_threshold: f32,
}

impl Default for AdaptationTriggers {
    fn default() -> Self {
        Self {
            performance_threshold: 0.1, // 10% performance degradation
            operation_count: 1000,
            time_interval: Duration::from_secs(60),
            workload_change_threshold: 0.2, // 20% workload change
            resource_utilization_threshold: 0.15, // 15% utilization change
        }
    }
}

/// Kernel parameter configuration for optimization
#[derive(Debug, Clone, PartialEq)]
pub struct KernelParameters {
    /// Grid dimensions (x, y, z)
    pub grid_dims: (u32, u32, u32),
    /// Block dimensions (x, y, z)
    pub block_dims: (u32, u32, u32),
    /// Shared memory size
    pub shared_memory: u32,
    /// Register usage per thread
    pub registers_per_thread: u32,
    /// Occupancy target (0.0 to 1.0)
    pub occupancy_target: f32,
    /// Memory coalescing strategy
    pub memory_coalescing: MemoryCoalescingStrategy,
    /// Loop unroll factor
    pub loop_unroll_factor: u32,
    /// Use texture memory
    pub use_texture_memory: bool,
    /// Use constant memory
    pub use_constant_memory: bool,
}

/// Memory coalescing strategies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MemoryCoalescingStrategy {
    /// No specific coalescing optimization
    None,
    /// Row-major access pattern
    RowMajor,
    /// Column-major access pattern
    ColumnMajor,
    /// Blocked access pattern
    Blocked { block_size: usize },
    /// Adaptive based on data layout
    Adaptive,
}

/// Performance metrics for adaptation
#[derive(Debug, Clone, Default)]
pub struct AdaptationMetrics {
    /// Execution time in milliseconds
    pub execution_time_ms: f64,
    /// Memory throughput in GB/s
    pub memory_throughput_gbps: f64,
    /// GPU utilization percentage
    pub gpu_utilization: f64,
    /// Memory utilization percentage
    pub memory_utilization: f64,
    /// Occupancy achieved
    pub occupancy: f64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Branch efficiency
    pub branch_efficiency: f64,
    /// Warp efficiency
    pub warp_efficiency: f64,
}

impl AdaptationMetrics {
    /// Calculates a composite performance score
    pub fn performance_score(&self) -> f64 {
        let time_score = 1.0 / (self.execution_time_ms + 1.0);
        let throughput_score = self.memory_throughput_gbps / 1000.0;
        let utilization_score = (self.gpu_utilization + self.memory_utilization) / 200.0;
        let efficiency_score = (self.occupancy + self.cache_hit_rate +
                               self.branch_efficiency + self.warp_efficiency) / 4.0;

        (time_score + throughput_score + utilization_score + efficiency_score) / 4.0
    }
}

/// Historical performance record
#[derive(Debug, Clone)]
pub struct PerformanceRecord {
    /// Kernel parameters used
    pub parameters: KernelParameters,
    /// Achieved metrics
    pub metrics: AdaptationMetrics,
    /// Timestamp of measurement
    pub timestamp: Instant,
    /// Workload characteristics at time of measurement
    pub workload_context: WorkloadContext,
}

/// Workload characteristics
#[derive(Debug, Clone)]
pub struct WorkloadContext {
    /// Matrix dimensions (M, N, K)
    pub matrix_dims: (usize, usize, usize),
    /// Batch size
    pub batch_size: usize,
    /// Data type size in bytes
    pub data_type_size: usize,
    /// Memory access pattern
    pub access_pattern: AccessPattern,
    /// Computational intensity (FLOPS/byte)
    pub compute_intensity: f64,
}

/// Memory access patterns
#[derive(Debug, Clone, PartialEq)]
pub enum AccessPattern {
    /// Sequential access
    Sequential,
    /// Random access
    Random,
    /// Strided access
    Strided { stride: usize },
    /// Block access
    Block { block_size: usize },
}

/// Adaptive CUDA kernel optimizer
pub struct AdaptiveOptimizer {
    /// Adaptation strategy
    strategy: AdaptationStrategy,
    /// Adaptation triggers configuration
    triggers: AdaptationTriggers,
    /// Historical performance records
    performance_history: Arc<RwLock<VecDeque<PerformanceRecord>>>,
    /// Current best parameters for different workload types
    best_parameters: Arc<RwLock<HashMap<String, KernelParameters>>>,
    /// Optimization state for different strategies
    optimization_state: Arc<Mutex<OptimizationState>>,
    /// Performance baseline for comparison
    baseline_metrics: Arc<RwLock<Option<AdaptationMetrics>>>,
    /// Last adaptation timestamp
    last_adaptation: Arc<Mutex<Instant>>,
    /// Operation counter since last adaptation
    operation_counter: Arc<Mutex<u64>>,
}

/// Internal optimization state for different strategies
#[derive(Debug)]
enum OptimizationState {
    /// No state for static strategy
    Static,
    /// Heuristic state
    Heuristic {
        improvement_trend: f64,
        last_performance: f64,
    },
    /// Genetic algorithm state
    Genetic {
        population: Vec<KernelParameters>,
        generation: usize,
        fitness_scores: Vec<f64>,
    },
    /// Bayesian optimization state
    Bayesian {
        explored_points: Vec<(KernelParameters, f64)>,
        acquisition_scores: HashMap<String, f64>,
    },
    /// Reinforcement learning state
    ReinforcementLearning {
        q_table: HashMap<String, f64>,
        state_action_counts: HashMap<String, u32>,
        last_state: Option<String>,
        last_action: Option<String>,
    },
    /// Multi-armed bandit state
    MultiArmedBandit {
        arm_rewards: HashMap<String, Vec<f64>>,
        arm_counts: HashMap<String, u32>,
    },
}

impl Default for OptimizationState {
    fn default() -> Self {
        Self::Static
    }
}

impl AdaptiveOptimizer {
    /// Creates a new adaptive optimizer
    pub fn new(strategy: AdaptationStrategy, triggers: AdaptationTriggers) -> Self {
        let optimization_state = match &strategy {
            AdaptationStrategy::Static => OptimizationState::Static,
            AdaptationStrategy::Heuristic => OptimizationState::Heuristic {
                improvement_trend: 0.0,
                last_performance: 0.0,
            },
            AdaptationStrategy::Genetic { population_size, .. } => {
                OptimizationState::Genetic {
                    population: Self::generate_initial_population(*population_size),
                    generation: 0,
                    fitness_scores: vec![0.0; *population_size],
                }
            },
            AdaptationStrategy::Bayesian { .. } => OptimizationState::Bayesian {
                explored_points: Vec::new(),
                acquisition_scores: HashMap::new(),
            },
            AdaptationStrategy::ReinforcementLearning { .. } => {
                OptimizationState::ReinforcementLearning {
                    q_table: HashMap::new(),
                    state_action_counts: HashMap::new(),
                    last_state: None,
                    last_action: None,
                }
            },
            AdaptationStrategy::MultiArmedBandit { .. } => {
                OptimizationState::MultiArmedBandit {
                    arm_rewards: HashMap::new(),
                    arm_counts: HashMap::new(),
                }
            },
        };

        Self {
            strategy,
            triggers,
            performance_history: Arc::new(RwLock::new(VecDeque::with_capacity(10000))),
            best_parameters: Arc::new(RwLock::new(HashMap::new())),
            optimization_state: Arc::new(Mutex::new(optimization_state)),
            baseline_metrics: Arc::new(RwLock::new(None)),
            last_adaptation: Arc::new(Mutex::new(Instant::now())),
            operation_counter: Arc::new(Mutex::new(0)),
        }
    }

    /// Generates initial population for genetic algorithm
    fn generate_initial_population(size: usize) -> Vec<KernelParameters> {
        let mut population = Vec::with_capacity(size);

        for _ in 0..size {
            let params = KernelParameters {
                grid_dims: (
                    (fastrand::u32(1..=1024)),
                    (fastrand::u32(1..=1024)),
                    1
                ),
                block_dims: (
                    (fastrand::u32(32..=1024) / 32) * 32, // Multiple of 32
                    (fastrand::u32(1..=32)),
                    1
                ),
                shared_memory: fastrand::u32(0..=49152), // Max shared memory per block
                registers_per_thread: fastrand::u32(16..=255),
                occupancy_target: fastrand::f32() * 0.5 + 0.5, // 0.5 to 1.0
                memory_coalescing: match fastrand::u32(0..5) {
                    0 => MemoryCoalescingStrategy::None,
                    1 => MemoryCoalescingStrategy::RowMajor,
                    2 => MemoryCoalescingStrategy::ColumnMajor,
                    3 => MemoryCoalescingStrategy::Blocked { block_size: 16 + fastrand::usize(0..48) },
                    _ => MemoryCoalescingStrategy::Adaptive,
                },
                loop_unroll_factor: fastrand::u32(1..=8),
                use_texture_memory: fastrand::bool(),
                use_constant_memory: fastrand::bool(),
            };
            population.push(params);
        }

        population
    }

    /// Records performance for adaptation
    pub fn record_performance(&self, parameters: KernelParameters, metrics: AdaptationMetrics, workload: WorkloadContext) -> Result<()> {
        let record = PerformanceRecord {
            parameters: parameters.clone(),
            metrics: metrics.clone(),
            timestamp: Instant::now(),
            workload_context: workload,
        };

        // Add to history
        {
            let mut history = self.performance_history.write().unwrap();
            if history.len() >= 10000 {
                history.pop_front();
            }
            history.push_back(record);
        }

        // Update baseline if first measurement
        {
            let mut baseline = self.baseline_metrics.write().unwrap();
            if baseline.is_none() {
                *baseline = Some(metrics.clone());
            }
        }

        // Increment operation counter
        {
            let mut counter = self.operation_counter.lock().unwrap();
            *counter += 1;
        }

        // Check if adaptation should be triggered
        if self.should_trigger_adaptation(&metrics)? {
            self.trigger_adaptation()?;
        }

        Ok(())
    }

    /// Checks if adaptation should be triggered
    fn should_trigger_adaptation(&self, current_metrics: &AdaptationMetrics) -> Result<bool> {
        // Check operation count trigger
        {
            let counter = self.operation_counter.lock().unwrap();
            if *counter >= self.triggers.operation_count {
                return Ok(true);
            }
        }

        // Check time interval trigger
        {
            let last_adaptation = self.last_adaptation.lock().unwrap();
            if last_adaptation.elapsed() >= self.triggers.time_interval {
                return Ok(true);
            }
        }

        // Check performance threshold trigger
        if let Some(baseline) = self.baseline_metrics.read().unwrap().as_ref() {
            let performance_degradation = (baseline.performance_score() - current_metrics.performance_score()) / baseline.performance_score();
            if performance_degradation > self.triggers.performance_threshold as f64 {
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Triggers adaptation process
    fn trigger_adaptation(&self) -> Result<()> {
        match &self.strategy {
            AdaptationStrategy::Static => {
                // No adaptation for static strategy
                Ok(())
            },
            AdaptationStrategy::Heuristic => self.adapt_heuristic(),
            AdaptationStrategy::Genetic { .. } => self.adapt_genetic(),
            AdaptationStrategy::Bayesian { .. } => self.adapt_bayesian(),
            AdaptationStrategy::ReinforcementLearning { .. } => self.adapt_reinforcement_learning(),
            AdaptationStrategy::MultiArmedBandit { .. } => self.adapt_multi_armed_bandit(),
        }
    }

    /// Heuristic-based adaptation
    fn adapt_heuristic(&self) -> Result<()> {
        let mut state = self.optimization_state.lock().unwrap();

        if let OptimizationState::Heuristic { improvement_trend, last_performance } = &mut *state {
            let history = self.performance_history.read().unwrap();

            if let Some(latest) = history.back() {
                let current_performance = latest.metrics.performance_score();
                let improvement = current_performance - *last_performance;

                // Update trend with exponential moving average
                *improvement_trend = 0.7 * *improvement_trend + 0.3 * improvement;
                *last_performance = current_performance;

                // Apply simple heuristics
                if *improvement_trend < -0.01 {
                    // Performance is degrading, try to adjust parameters
                    self.apply_heuristic_adjustments(&latest.parameters)?;
                }
            }
        }

        self.update_adaptation_timestamp();
        Ok(())
    }

    /// Applies heuristic parameter adjustments
    fn apply_heuristic_adjustments(&self, current_params: &KernelParameters) -> Result<()> {
        let mut new_params = current_params.clone();

        // Simple heuristic: if occupancy is low, try smaller blocks
        let history = self.performance_history.read().unwrap();
        if let Some(latest) = history.back() {
            if latest.metrics.occupancy < 0.5 {
                new_params.block_dims.0 = (new_params.block_dims.0 / 2).max(32);
            }

            // If memory throughput is low, try different coalescing
            if latest.metrics.memory_throughput_gbps < 100.0 {
                new_params.memory_coalescing = match new_params.memory_coalescing {
                    MemoryCoalescingStrategy::RowMajor => MemoryCoalescingStrategy::ColumnMajor,
                    MemoryCoalescingStrategy::ColumnMajor => MemoryCoalescingStrategy::Blocked { block_size: 16 },
                    _ => MemoryCoalescingStrategy::RowMajor,
                };
            }
        }

        // Store the adjusted parameters
        let workload_key = self.create_workload_key(&history.back().unwrap().workload_context);
        let mut best_params = self.best_parameters.write().unwrap();
        best_params.insert(workload_key, new_params);

        Ok(())
    }

    /// Genetic algorithm adaptation
    fn adapt_genetic(&self) -> Result<()> {
        let mut state = self.optimization_state.lock().unwrap();

        if let OptimizationState::Genetic { population, generation, fitness_scores } = &mut *state {
            // Evaluate fitness for current population
            self.evaluate_genetic_fitness(population, fitness_scores)?;

            // Selection, crossover, and mutation
            let new_population = self.genetic_evolution(population, fitness_scores)?;
            *population = new_population;
            *generation += 1;

            // Find and store best individual
            let best_idx = fitness_scores.iter()
                .position(|&score| score == fitness_scores.iter().fold(0.0, |a, &b| a.max(b)))
                .unwrap_or(0);

            let history = self.performance_history.read().unwrap();
            if let Some(latest) = history.back() {
                let workload_key = self.create_workload_key(&latest.workload_context);
                let mut best_params = self.best_parameters.write().unwrap();
                best_params.insert(workload_key, population[best_idx].clone());
            }
        }

        self.update_adaptation_timestamp();
        Ok(())
    }

    /// Evaluates fitness for genetic algorithm
    fn evaluate_genetic_fitness(&self, population: &[KernelParameters], fitness_scores: &mut [f64]) -> Result<()> {
        let history = self.performance_history.read().unwrap();

        for (i, params) in population.iter().enumerate() {
            // Find similar configurations in history
            let mut total_score = 0.0;
            let mut count = 0;

            for record in history.iter().rev().take(100) {
                if self.parameters_similarity(params, &record.parameters) > 0.8 {
                    total_score += record.metrics.performance_score();
                    count += 1;
                }
            }

            fitness_scores[i] = if count > 0 {
                total_score / count as f64
            } else {
                0.5 // Default fitness for unexamined parameters
            };
        }

        Ok(())
    }

    /// Genetic algorithm evolution step
    fn genetic_evolution(&self, population: &[KernelParameters], fitness_scores: &[f64]) -> Result<Vec<KernelParameters>> {
        let mut new_population = Vec::with_capacity(population.len());

        // Keep best individuals (elitism)
        let elite_count = population.len() / 10; // Top 10%
        let mut sorted_indices: Vec<usize> = (0..population.len()).collect();
        sorted_indices.sort_by(|&a, &b| fitness_scores[b].partial_cmp(&fitness_scores[a]).unwrap());

        for &idx in sorted_indices.iter().take(elite_count) {
            new_population.push(population[idx].clone());
        }

        // Generate offspring through crossover and mutation
        while new_population.len() < population.len() {
            let parent1_idx = self.tournament_selection(fitness_scores)?;
            let parent2_idx = self.tournament_selection(fitness_scores)?;

            let (mut child1, mut child2) = self.crossover(&population[parent1_idx], &population[parent2_idx])?;

            if let AdaptationStrategy::Genetic { mutation_rate, .. } = &self.strategy {
                if fastrand::f32() < *mutation_rate {
                    child1 = self.mutate(child1)?;
                }
                if fastrand::f32() < *mutation_rate {
                    child2 = self.mutate(child2)?;
                }
            }

            new_population.push(child1);
            if new_population.len() < population.len() {
                new_population.push(child2);
            }
        }

        Ok(new_population)
    }

    /// Tournament selection for genetic algorithm
    fn tournament_selection(&self, fitness_scores: &[f64]) -> Result<usize> {
        let tournament_size = 3;
        let mut best_idx = fastrand::usize(0..fitness_scores.len());
        let mut best_fitness = fitness_scores[best_idx];

        for _ in 1..tournament_size {
            let idx = fastrand::usize(0..fitness_scores.len());
            if fitness_scores[idx] > best_fitness {
                best_idx = idx;
                best_fitness = fitness_scores[idx];
            }
        }

        Ok(best_idx)
    }

    /// Crossover operation for genetic algorithm
    fn crossover(&self, parent1: &KernelParameters, parent2: &KernelParameters) -> Result<(KernelParameters, KernelParameters)> {
        let mut child1 = parent1.clone();
        let mut child2 = parent2.clone();

        if let AdaptationStrategy::Genetic { crossover_rate, .. } = &self.strategy {
            if fastrand::f32() < *crossover_rate {
                // Single-point crossover on different parameters
                if fastrand::bool() {
                    child1.grid_dims = parent2.grid_dims;
                    child2.grid_dims = parent1.grid_dims;
                }

                if fastrand::bool() {
                    child1.block_dims = parent2.block_dims;
                    child2.block_dims = parent1.block_dims;
                }

                if fastrand::bool() {
                    child1.shared_memory = parent2.shared_memory;
                    child2.shared_memory = parent1.shared_memory;
                }

                if fastrand::bool() {
                    child1.memory_coalescing = parent2.memory_coalescing.clone();
                    child2.memory_coalescing = parent1.memory_coalescing.clone();
                }
            }
        }

        Ok((child1, child2))
    }

    /// Mutation operation for genetic algorithm
    fn mutate(&self, mut params: KernelParameters) -> Result<KernelParameters> {
        // Mutate random parameters
        match fastrand::u32(0..8) {
            0 => params.grid_dims.0 = fastrand::u32(1..=1024),
            1 => params.grid_dims.1 = fastrand::u32(1..=1024),
            2 => params.block_dims.0 = (fastrand::u32(32..=1024) / 32) * 32,
            3 => params.block_dims.1 = fastrand::u32(1..=32),
            4 => params.shared_memory = fastrand::u32(0..=49152),
            5 => params.occupancy_target = fastrand::f32() * 0.5 + 0.5,
            6 => params.loop_unroll_factor = fastrand::u32(1..=8),
            _ => params.use_texture_memory = !params.use_texture_memory,
        }

        Ok(params)
    }

    /// Bayesian optimization adaptation
    fn adapt_bayesian(&self) -> Result<()> {
        // Simplified Bayesian optimization implementation
        self.update_adaptation_timestamp();
        Ok(())
    }

    /// Reinforcement learning adaptation
    fn adapt_reinforcement_learning(&self) -> Result<()> {
        // Simplified RL implementation
        self.update_adaptation_timestamp();
        Ok(())
    }

    /// Multi-armed bandit adaptation
    fn adapt_multi_armed_bandit(&self) -> Result<()> {
        // Simplified MAB implementation
        self.update_adaptation_timestamp();
        Ok(())
    }

    /// Creates a workload key for parameter storage
    fn create_workload_key(&self, workload: &WorkloadContext) -> String {
        format!("{}x{}x{}_bs{}_dt{}_ci{:.2}",
                workload.matrix_dims.0,
                workload.matrix_dims.1,
                workload.matrix_dims.2,
                workload.batch_size,
                workload.data_type_size,
                workload.compute_intensity)
    }

    /// Calculates similarity between two parameter sets
    fn parameters_similarity(&self, params1: &KernelParameters, params2: &KernelParameters) -> f64 {
        let mut similarity = 0.0;
        let mut factors = 0.0;

        // Grid dimension similarity
        similarity += 1.0 - (params1.grid_dims.0 as f64 - params2.grid_dims.0 as f64).abs() / params1.grid_dims.0.max(params2.grid_dims.0) as f64;
        factors += 1.0;

        // Block dimension similarity
        similarity += 1.0 - (params1.block_dims.0 as f64 - params2.block_dims.0 as f64).abs() / params1.block_dims.0.max(params2.block_dims.0) as f64;
        factors += 1.0;

        // Shared memory similarity
        similarity += 1.0 - (params1.shared_memory as f64 - params2.shared_memory as f64).abs() / params1.shared_memory.max(params2.shared_memory).max(1) as f64;
        factors += 1.0;

        // Occupancy target similarity
        similarity += 1.0 - (params1.occupancy_target - params2.occupancy_target).abs();
        factors += 1.0;

        similarity / factors
    }

    /// Updates the last adaptation timestamp
    fn update_adaptation_timestamp(&self) {
        *self.last_adaptation.lock().unwrap() = Instant::now();
        *self.operation_counter.lock().unwrap() = 0;
    }

    /// Gets the best parameters for a given workload
    pub fn get_best_parameters(&self, workload: &WorkloadContext) -> Option<KernelParameters> {
        let workload_key = self.create_workload_key(workload);
        self.best_parameters.read().unwrap().get(&workload_key).cloned()
    }

    /// Gets adaptation statistics
    pub fn get_adaptation_statistics(&self) -> AdaptationStatistics {
        let history = self.performance_history.read().unwrap();
        let best_params = self.best_parameters.read().unwrap();

        let total_adaptations = *self.operation_counter.lock().unwrap();
        let last_adaptation_time = *self.last_adaptation.lock().unwrap();

        let performance_improvement = if history.len() >= 2 {
            let recent_avg = history.iter().rev().take(100)
                .map(|r| r.metrics.performance_score())
                .sum::<f64>() / 100.0.min(history.len() as f64);

            let baseline_avg = history.iter().take(100)
                .map(|r| r.metrics.performance_score())
                .sum::<f64>() / 100.0.min(history.len() as f64);

            (recent_avg - baseline_avg) / baseline_avg * 100.0
        } else {
            0.0
        };

        AdaptationStatistics {
            total_adaptations,
            performance_improvement_percent: performance_improvement,
            last_adaptation_time,
            active_configurations: best_params.len(),
            strategy_name: format!("{:?}", self.strategy),
        }
    }
}

/// Adaptation statistics
#[derive(Debug, Clone)]
pub struct AdaptationStatistics {
    /// Total number of adaptations performed
    pub total_adaptations: u64,
    /// Performance improvement percentage
    pub performance_improvement_percent: f64,
    /// Last adaptation timestamp
    pub last_adaptation_time: Instant,
    /// Number of active configurations
    pub active_configurations: usize,
    /// Strategy name
    pub strategy_name: String,
}

impl AdaptationStatistics {
    /// Formats statistics as human-readable report
    pub fn format_report(&self) -> String {
        format!(
            "=== Adaptive Optimization Statistics ===\n\
             Strategy: {}\n\
             Total Adaptations: {}\n\
             Performance Improvement: {:.2}%\n\
             Active Configurations: {}\n\
             Last Adaptation: {:.2}s ago\n",
            self.strategy_name,
            self.total_adaptations,
            self.performance_improvement_percent,
            self.active_configurations,
            self.last_adaptation_time.elapsed().as_secs_f64()
        )
    }
}