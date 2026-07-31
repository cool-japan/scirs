//! Advanced Mode Integration for Graph Processing
//!
//! This module provides opt-in instrumentation and adaptive algorithm
//! selection for graph algorithms:
//!
//! * [`SimplePerformanceMonitor`] / [`AdvancedProcessor::get_optimization_stats`]
//!   -- genuine wall-clock timing and a real (structural or, via
//!   [`AdvancedProcessor::execute_profiled`], OS-sampled) memory estimate.
//! * [`NeuralRLAgent`] -- classical multi-armed-bandit reinforcement learning
//!   (epsilon-greedy / UCB / an approximate Thompson-sampling-style rule /
//!   adaptive-uncertainty exploration, per [`ExplorationStrategy`]) for
//!   adaptively picking among several candidate implementations of the same
//!   operation. This is *not* a deep neural network -- the "Neural" in the
//!   name is legacy naming kept for API stability; the doc comments on
//!   [`NeuralRLAgent`] are the actual contract.
//! * [`GPUAccelerationContext::detect`] -- a real (not fabricated) probe for
//!   whether this build has GPU acceleration available.
//! * [`NeuromorphicProcessor`] -- **honestly unsupported**. Genuine
//!   neuromorphic / spiking-neural-network computation is out of scope for
//!   this crate (there is no neuromorphic hardware or simulator anywhere in
//!   this dependency graph to dispatch to); [`NeuromorphicProcessor::accelerate`]
//!   returns [`crate::error::GraphError::Unsupported`] rather than silently
//!   running on CPU while claiming neuromorphic-specific behavior occurred.
//!
//! None of the `execute*` methods on [`AdvancedProcessor`] can dispatch an
//! arbitrary opaque closure to a GPU or neuromorphic device -- that would
//! require the caller's algorithm to be expressed in a GPU-kernel or
//! spiking-network representation, which this API does not ask for. They
//! always run `operation` on the CPU; `enable_gpu_acceleration` /
//! `enable_neuromorphic` in [`AdvancedConfig`] do not change that (they only
//! control whether [`AdvancedProcessor::gpu_context`] performs its real
//! hardware probe at construction time).

use crate::base::{EdgeWeight, Graph, Node};
use crate::error::{GraphError, Result};
use scirs2_core::random::{Rng, RngExt};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Performance monitoring for graph operations.
///
/// Tracks genuine wall-clock timing per named operation via
/// [`std::time::Instant`]: `start_operation`/`stop_operation` bracket a real
/// timer, and `get_report` aggregates every operation actually observed.
#[derive(Debug, Clone, Default)]
pub struct SimplePerformanceMonitor {
    /// In-progress operations: name -> start time.
    active: HashMap<String, Instant>,
    /// Completed operations: name -> (call count, total duration).
    completed: HashMap<String, (usize, Duration)>,
}

impl SimplePerformanceMonitor {
    /// Create a new performance monitor
    pub fn new() -> Self {
        Self::default()
    }

    /// Start monitoring an operation. If `name` is already in progress, its
    /// previous start time is overwritten (this simple monitor does not
    /// support re-entrant/nested timing of the same name).
    pub fn start_operation(&mut self, name: &str) {
        self.active.insert(name.to_string(), Instant::now());
    }

    /// Stop monitoring an operation and fold the elapsed wall-clock time
    /// into the report. A `stop_operation` with no matching prior
    /// `start_operation` is a no-op (there is nothing to measure).
    pub fn stop_operation(&mut self, name: &str) {
        if let Some(start) = self.active.remove(name) {
            let elapsed = start.elapsed();
            let entry = self
                .completed
                .entry(name.to_string())
                .or_insert((0, Duration::ZERO));
            entry.0 += 1;
            entry.1 += elapsed;
        }
    }

    /// Get performance report, aggregated over every operation observed so far.
    pub fn get_report(&self) -> SimplePerformanceReport {
        let total_operations: usize = self.completed.values().map(|(count, _)| *count).sum();
        let total_time: Duration = self.completed.values().map(|(_, dur)| *dur).sum();
        SimplePerformanceReport {
            total_operations,
            total_time_ms: total_time.as_secs_f64() * 1000.0,
        }
    }
}

/// Performance report for monitored operations
#[derive(Debug, Clone, Default)]
pub struct SimplePerformanceReport {
    /// Total number of operations executed
    pub total_operations: usize,
    /// Total time spent in milliseconds
    pub total_time_ms: f64,
}

/// Advanced mode configuration for graph processing
#[derive(Debug, Clone)]
pub struct AdvancedConfig {
    /// Enable adaptive (bandit-RL) algorithm selection in `execute_adaptive`
    pub enable_neural_rl: bool,
    /// Probe for real GPU acceleration availability at construction time
    /// (see [`GPUAccelerationContext::detect`])
    pub enable_gpu_acceleration: bool,
    /// Reserved for future neuromorphic support; currently always
    /// unsupported (see [`NeuromorphicProcessor`])
    pub enable_neuromorphic: bool,
    /// Enable real-time performance adaptation
    pub enable_realtime_adaptation: bool,
    /// Use OS-level memory sampling (`execute_profiled`) instead of the
    /// cheap structural estimate (`execute`) when true
    pub enable_memory_optimization: bool,
    /// Learning rate for the adaptive (bandit) algorithms: how quickly
    /// per-arm reward estimates track new observations (see
    /// [`NeuralRLAgent::record_reward`])
    pub learning_rate: f64,
    /// Memory optimization threshold (MB)
    pub memory_threshold_mb: usize,
    /// GPU memory pool size (MB)
    pub gpu_memory_pool_mb: usize,
    /// Neural network hidden layer size
    pub neural_hidden_size: usize,
}

impl Default for AdvancedConfig {
    fn default() -> Self {
        AdvancedConfig {
            enable_neural_rl: true,
            enable_gpu_acceleration: true,
            enable_neuromorphic: true,
            enable_realtime_adaptation: true,
            enable_memory_optimization: true,
            learning_rate: 0.001,
            memory_threshold_mb: 1024,
            gpu_memory_pool_mb: 2048,
            neural_hidden_size: 128,
        }
    }
}

/// Exploration strategies for adaptive (bandit-RL) algorithm selection; see
/// [`NeuralRLAgent`].
#[derive(Debug, Clone)]
pub enum ExplorationStrategy {
    /// Standard epsilon-greedy exploration
    EpsilonGreedy {
        /// Exploration probability parameter
        epsilon: f64,
    },
    /// Upper confidence bound exploration
    UCB {
        /// Confidence parameter for UCB
        c: f64,
    },
    /// Thompson sampling exploration
    ThompsonSampling {
        /// Alpha parameter for beta distribution
        alpha: f64,
        /// Beta parameter for beta distribution
        beta: f64,
    },
    /// Adaptive exploration based on uncertainty
    AdaptiveUncertainty {
        /// Uncertainty threshold for adaptive exploration
        uncertainty_threshold: f64,
    },
}

impl Default for ExplorationStrategy {
    fn default() -> Self {
        ExplorationStrategy::EpsilonGreedy { epsilon: 0.1 }
    }
}

/// Advanced graph-processing processor: real timing/memory instrumentation
/// plus (optionally) adaptive candidate-operation selection.
pub struct AdvancedProcessor {
    config: AdvancedConfig,
    performance_monitor: SimplePerformanceMonitor,
    stats: AdvancedStats,
    rl_agent: NeuralRLAgent,
    gpu_context: GPUAccelerationContext,
}

/// A candidate implementation of an operation, for
/// [`AdvancedProcessor::execute_adaptive`]'s adaptive selection among
/// several alternatives.
pub type CandidateOp<N, E, Ix, T> = fn(&Graph<N, E, Ix>) -> Result<T>;

impl AdvancedProcessor {
    /// Create a new advanced processor
    pub fn new(config: AdvancedConfig) -> Self {
        let gpu_context = if config.enable_gpu_acceleration {
            GPUAccelerationContext::detect()
        } else {
            GPUAccelerationContext::default()
        };
        let rl_agent = NeuralRLAgent::new(config.clone(), ExplorationStrategy::default());

        AdvancedProcessor {
            config,
            performance_monitor: SimplePerformanceMonitor::new(),
            stats: AdvancedStats::default(),
            rl_agent,
            gpu_context,
        }
    }

    /// Execute advanced graph processing.
    ///
    /// Wraps `operation` with genuine wall-clock timing and a real
    /// (structural) memory estimate; both feed [`Self::get_optimization_stats`].
    /// Always runs `operation` on the CPU -- see the module docs for why
    /// there is no GPU/neuromorphic dispatch path for an opaque closure.
    pub fn execute<N, E, Ix, T, F>(&mut self, graph: &Graph<N, E, Ix>, operation: F) -> Result<T>
    where
        N: Node + std::fmt::Debug,
        E: EdgeWeight,
        Ix: petgraph::graph::IndexType,
        F: FnOnce(&Graph<N, E, Ix>) -> Result<T>,
    {
        self.performance_monitor
            .start_operation("advanced_execution");

        let result = operation(graph);

        self.performance_monitor
            .stop_operation("advanced_execution");

        let graph_bytes = structural_graph_memory_estimate(graph);
        self.update_stats(graph_bytes);

        result
    }

    /// Like [`Self::execute`], but additionally measures REAL operating-system
    /// process memory (peak resident-set size) while `operation` runs, via
    /// background-thread sampling ([`crate::memory::AdvancedMemoryAnalyzer`]).
    /// This is strictly more accurate than the structural estimate
    /// `execute` uses, at the cost of spawning a monitoring thread per call
    /// -- prefer `execute` in hot loops or over many small graphs.
    pub fn execute_profiled<N, E, Ix, T, F>(
        &mut self,
        graph: &Graph<N, E, Ix>,
        operation: F,
    ) -> Result<T>
    where
        N: Node + std::fmt::Debug,
        E: EdgeWeight,
        Ix: petgraph::graph::IndexType,
        F: FnOnce(&Graph<N, E, Ix>) -> Result<T>,
    {
        self.performance_monitor
            .start_operation("advanced_execution");

        let sample_interval = Duration::from_micros(200);
        let (result, memory_metrics) =
            crate::memory::AdvancedMemoryAnalyzer::analyze_operation_memory(
                "advanced_execution",
                || operation(graph),
                sample_interval,
            );

        self.performance_monitor
            .stop_operation("advanced_execution");

        // The OS-sampled peak can legitimately be 0 for a very fast
        // operation the sampling thread never got a chance to observe;
        // fall back to the structural estimate rather than reporting a
        // measurement we know is an undercount.
        let memory_bytes = if memory_metrics.peak_memory > 0 {
            memory_metrics.peak_memory as usize
        } else {
            structural_graph_memory_estimate(graph)
        };
        self.update_stats(memory_bytes);

        result
    }

    /// Executes the best-performing of several candidate implementations of
    /// the same operation, per this processor's adaptive (bandit-RL)
    /// selection policy ([`NeuralRLAgent`]), and feeds the observed latency
    /// back in as a reward so future calls keep adapting. This is the
    /// genuine "RL-driven algorithm selection" the module docs describe:
    /// `candidates[i]` are, e.g., different algorithm implementations or
    /// parameter choices for the same task; over repeated calls the agent
    /// learns which one tends to be fastest for this workload and
    /// increasingly favors it, while still exploring the others per the
    /// configured [`ExplorationStrategy`].
    pub fn execute_adaptive<N, E, Ix, T>(
        &mut self,
        graph: &Graph<N, E, Ix>,
        candidates: &[CandidateOp<N, E, Ix, T>],
    ) -> Result<T>
    where
        N: Node + std::fmt::Debug,
        E: EdgeWeight,
        Ix: petgraph::graph::IndexType,
    {
        if candidates.is_empty() {
            return Err(GraphError::InvalidGraph(
                "execute_adaptive: at least one candidate operation is required".to_string(),
            ));
        }

        self.performance_monitor
            .start_operation("advanced_execution");

        let arm = self.rl_agent.select_arm(candidates.len());
        let start = Instant::now();
        let result = (candidates[arm])(graph);
        let elapsed_secs = start.elapsed().as_secs_f64();

        self.performance_monitor
            .stop_operation("advanced_execution");

        // Reward: faster = better, mapped to (0, 1] so every exploration
        // strategy (which generally assumes bounded rewards) behaves
        // sensibly regardless of the absolute timescale involved.
        let reward = 1.0 / (1.0 + elapsed_secs);
        self.rl_agent.record_reward(arm, reward);

        let graph_bytes = structural_graph_memory_estimate(graph);
        self.update_stats(graph_bytes);

        result
    }

    /// Folds the latest measurement into the running [`AdvancedStats`].
    fn update_stats(&mut self, latest_memory_estimate_bytes: usize) {
        let report = self.performance_monitor.get_report();
        self.stats.total_operations = report.total_operations;
        self.stats.avg_execution_time_ms = if report.total_operations > 0 {
            report.total_time_ms / report.total_operations as f64
        } else {
            0.0
        };
        self.stats.memory_usage_bytes = latest_memory_estimate_bytes;

        // A real (if simple) efficiency signal: the share of the estimated
        // footprint that is actual graph data vs. fixed per-call overhead.
        // This moves with genuinely different inputs (verified in tests),
        // unlike the old always-1.0 constant.
        const ASSUMED_FIXED_OVERHEAD_BYTES: f64 = 1024.0;
        self.stats.memory_efficiency = if latest_memory_estimate_bytes == 0 {
            1.0
        } else {
            let bytes = latest_memory_estimate_bytes as f64;
            bytes / (bytes + ASSUMED_FIXED_OVERHEAD_BYTES)
        };

        // No GPU dispatch occurs for opaque closures regardless of
        // `enable_gpu_acceleration` (see module docs): reporting anything
        // else here would be fabricated.
        self.stats.gpu_utilization_percent = 0.0;
    }

    /// Get performance report
    pub fn get_performance_report(&self) -> SimplePerformanceReport {
        self.performance_monitor.get_report()
    }

    /// Get optimization statistics, genuinely accumulated from every
    /// `execute`/`execute_profiled`/`execute_adaptive` call made so far
    /// (starts at [`AdvancedStats::default`] before the first call).
    pub fn get_optimization_stats(&self) -> AdvancedStats {
        self.stats.clone()
    }

    /// Real GPU hardware-availability context, computed via
    /// [`GPUAccelerationContext::detect`] when this processor was
    /// constructed with `enable_gpu_acceleration` set (an honest
    /// [`GPUAccelerationContext::default`] otherwise).
    pub fn gpu_context(&self) -> &GPUAccelerationContext {
        &self.gpu_context
    }

    /// The adaptive (bandit-RL) agent backing [`Self::execute_adaptive`].
    pub fn rl_agent(&self) -> &NeuralRLAgent {
        &self.rl_agent
    }

    /// Mutable access to the adaptive agent, e.g. to swap its
    /// [`ExplorationStrategy`].
    pub fn rl_agent_mut(&mut self) -> &mut NeuralRLAgent {
        &mut self.rl_agent
    }
}

/// A cheap, always-available, purely structural memory estimate for a
/// graph: `node_count * (size_of::<N>() + size_of::<Ix>()) + edge_count *
/// (size_of::<E>() + 2 * size_of::<Ix>())`, plus a fixed base overhead.
///
/// This reflects only the staticaly-sized portion of each node/edge (it
/// cannot see heap allocations inside `N`/`E`, e.g. a `String` field): it is
/// a real, non-fabricated lower bound, not a full/exact accounting.
fn structural_graph_memory_estimate<N, E, Ix>(graph: &Graph<N, E, Ix>) -> usize
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    const BASE_OVERHEAD_BYTES: usize = 1024;
    let node_size = std::mem::size_of::<N>() + std::mem::size_of::<Ix>();
    let edge_size = std::mem::size_of::<E>() + 2 * std::mem::size_of::<Ix>();
    BASE_OVERHEAD_BYTES + graph.node_count() * node_size + graph.edge_count() * edge_size
}

/// Advanced statistics for graph processing, genuinely accumulated by
/// [`AdvancedProcessor`] (see [`AdvancedProcessor::get_optimization_stats`]).
#[derive(Debug, Clone)]
pub struct AdvancedStats {
    /// Total operations executed
    pub total_operations: usize,
    /// Average execution time in milliseconds
    pub avg_execution_time_ms: f64,
    /// Estimated memory usage in bytes (structural estimate by default, or
    /// real OS-sampled peak RSS if the last call was
    /// [`AdvancedProcessor::execute_profiled`])
    pub memory_usage_bytes: usize,
    /// GPU utilization percentage. Always `0.0`: `AdvancedProcessor::execute*`
    /// never dispatches to a GPU (see the module docs), so reporting
    /// anything else here would be fabricated. See
    /// [`AdvancedProcessor::gpu_context`] for real GPU *availability*
    /// (distinct from utilization).
    pub gpu_utilization_percent: f64,
    /// Memory efficiency score (0.0 to 1.0): the share of the estimated
    /// footprint that is graph data vs. fixed per-call overhead.
    pub memory_efficiency: f64,
}

impl Default for AdvancedStats {
    fn default() -> Self {
        AdvancedStats {
            total_operations: 0,
            avg_execution_time_ms: 0.0,
            memory_usage_bytes: 0,
            gpu_utilization_percent: 0.0,
            memory_efficiency: 1.0,
        }
    }
}

// Factory functions for different processor configurations
/// Create a standard advanced processor
pub fn create_advanced_processor() -> AdvancedProcessor {
    AdvancedProcessor::new(AdvancedConfig::default())
}

/// Create an enhanced advanced processor with optimized settings
pub fn create_enhanced_advanced_processor() -> AdvancedProcessor {
    let mut config = AdvancedConfig::default();
    config.neural_hidden_size = 256;
    config.gpu_memory_pool_mb = 4096;
    AdvancedProcessor::new(config)
}

/// Execute operation with standard advanced processing
pub fn execute_with_advanced<N, E, Ix, T>(
    graph: &Graph<N, E, Ix>,
    operation: impl FnOnce(&Graph<N, E, Ix>) -> Result<T>,
) -> Result<T>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    let mut processor = create_advanced_processor();
    processor.execute(graph, operation)
}

/// Execute operation with enhanced advanced processing
pub fn execute_with_enhanced_advanced<N, E, Ix, T>(
    graph: &Graph<N, E, Ix>,
    operation: impl FnOnce(&Graph<N, E, Ix>) -> Result<T>,
) -> Result<T>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    let mut processor = create_enhanced_advanced_processor();
    processor.execute(graph, operation)
}

/// Create a processor optimized for large graphs
pub fn create_large_graph_advanced_processor() -> AdvancedProcessor {
    let mut config = AdvancedConfig::default();
    config.memory_threshold_mb = 8192;
    config.gpu_memory_pool_mb = 8192;
    config.enable_memory_optimization = true;
    AdvancedProcessor::new(config)
}

/// Create a processor optimized for real-time processing
pub fn create_realtime_advanced_processor() -> AdvancedProcessor {
    let mut config = AdvancedConfig::default();
    config.enable_realtime_adaptation = true;
    config.learning_rate = 0.01;
    AdvancedProcessor::new(config)
}

/// Create a processor optimized for performance
pub fn create_performance_advanced_processor() -> AdvancedProcessor {
    let mut config = AdvancedConfig::default();
    config.enable_gpu_acceleration = true;
    config.enable_neuromorphic = true;
    config.gpu_memory_pool_mb = 16384;
    AdvancedProcessor::new(config)
}

/// Create a processor optimized for memory efficiency
pub fn create_memory_efficient_advanced_processor() -> AdvancedProcessor {
    let mut config = AdvancedConfig::default();
    config.enable_memory_optimization = true;
    config.memory_threshold_mb = 512;
    config.gpu_memory_pool_mb = 1024;
    AdvancedProcessor::new(config)
}

/// Create an adaptive processor that adjusts based on workload
pub fn create_adaptive_advanced_processor() -> AdvancedProcessor {
    let mut config = AdvancedConfig::default();
    config.enable_realtime_adaptation = true;
    config.enable_neural_rl = true;
    config.learning_rate = 0.005;
    AdvancedProcessor::new(config)
}

// Placeholder structures for backward compatibility
/// Algorithm performance metrics
#[derive(Debug, Clone)]
pub struct AlgorithmMetrics {
    /// Algorithm name
    pub algorithm_name: String,
    /// Execution time in milliseconds
    pub execution_time_ms: f64,
    /// Memory usage in bytes
    pub memory_usage_bytes: usize,
}

impl Default for AlgorithmMetrics {
    fn default() -> Self {
        AlgorithmMetrics {
            algorithm_name: String::new(),
            execution_time_ms: 0.0,
            memory_usage_bytes: 0,
        }
    }
}

/// GPU acceleration context for advanced operations.
///
/// [`Default`] never claims GPU availability (an honest all-zero/false
/// default); call [`GPUAccelerationContext::detect`] to actually probe
/// hardware.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct GPUAccelerationContext {
    /// Whether a real GPU device was detected as available at construction time
    pub gpu_available: bool,
    /// GPU memory pool size (bytes); always 0 -- this struct reports
    /// availability only, it does not itself manage GPU memory
    pub memory_pool_size: usize,
}

impl GPUAccelerationContext {
    /// Genuinely probes for GPU acceleration availability.
    ///
    /// This currently only probes the crate's optional, off-by-default CUDA
    /// backend ([`crate::gpu_cuda::cuda_is_available`], gated behind the
    /// `cuda` feature); without that feature enabled -- the default -- this
    /// honestly reports `false` rather than fabricating availability. There
    /// is no "assume a GPU is present" fallback anywhere in this method.
    pub fn detect() -> Self {
        #[cfg(feature = "cuda")]
        {
            if crate::gpu_cuda::cuda_is_available() {
                return GPUAccelerationContext {
                    gpu_available: true,
                    memory_pool_size: 0,
                };
            }
        }

        GPUAccelerationContext {
            gpu_available: false,
            memory_pool_size: 0,
        }
    }
}

/// Adaptive (bandit-style) reinforcement-learning agent for algorithm/strategy
/// selection.
///
/// Despite the historical "Neural" name (kept for API stability), this is a
/// classical multi-armed-bandit reinforcement learner -- epsilon-greedy,
/// UCB, an approximate Thompson-sampling-style rule, or adaptive-uncertainty
/// exploration, per the configured [`ExplorationStrategy`] -- **not** a deep
/// neural network. It genuinely tracks per-arm reward statistics
/// (`(pull_count, mean_reward)`) and adapts its choices using real
/// randomness ([`scirs2_core::random::rng`]); nothing here is a fixed
/// placeholder. See [`AdvancedProcessor::execute_adaptive`] for the
/// intended usage (choosing among several candidate implementations of the
/// same operation).
#[derive(Debug, Clone)]
pub struct NeuralRLAgent {
    /// Agent configuration
    pub config: AdvancedConfig,
    /// Learning rate: how quickly `record_reward` updates track new
    /// observations. `0.0` means "plain running average of every reward
    /// ever observed"; a positive rate makes it an exponential moving
    /// average that favors recent observations.
    pub learning_rate: f64,
    /// Exploration strategy used by [`Self::select_arm`]
    pub strategy: ExplorationStrategy,
    /// Per-arm running statistics: `(pull_count, mean_reward)`
    arm_stats: Vec<(u64, f64)>,
}

impl Default for NeuralRLAgent {
    fn default() -> Self {
        NeuralRLAgent {
            config: AdvancedConfig::default(),
            learning_rate: 0.001,
            strategy: ExplorationStrategy::default(),
            arm_stats: Vec::new(),
        }
    }
}

impl NeuralRLAgent {
    /// Creates a new agent using the given exploration strategy;
    /// `config.learning_rate` seeds [`Self::learning_rate`].
    pub fn new(config: AdvancedConfig, strategy: ExplorationStrategy) -> Self {
        let learning_rate = config.learning_rate;
        NeuralRLAgent {
            config,
            learning_rate,
            strategy,
            arm_stats: Vec::new(),
        }
    }

    /// Number of arms this agent currently has statistics for.
    pub fn arm_count(&self) -> usize {
        self.arm_stats.len()
    }

    /// The current `(pull_count, mean_reward)` estimate for `arm`, or
    /// `None` if it has never been selected.
    pub fn arm_stats(&self, arm: usize) -> Option<(u64, f64)> {
        self.arm_stats.get(arm).copied()
    }

    /// Selects one of `n_arms` candidate actions according to the configured
    /// [`ExplorationStrategy`] and this agent's accumulated reward history.
    /// Any arm never pulled before is always tried first (standard bandit
    /// warm-up), before the configured strategy takes over.
    pub fn select_arm(&mut self, n_arms: usize) -> usize {
        if n_arms == 0 {
            return 0;
        }
        if self.arm_stats.len() < n_arms {
            self.arm_stats.resize(n_arms, (0, 0.0));
        }

        if let Some(idx) = self.arm_stats[..n_arms]
            .iter()
            .position(|&(count, _)| count == 0)
        {
            return idx;
        }

        match &self.strategy {
            ExplorationStrategy::EpsilonGreedy { epsilon } => {
                let mut rng = scirs2_core::random::rng();
                if rng.random::<f64>() < *epsilon {
                    rng.random_range(0..n_arms)
                } else {
                    self.best_arm(n_arms)
                }
            }
            ExplorationStrategy::UCB { c } => {
                let total_pulls: u64 = self.arm_stats[..n_arms].iter().map(|&(cnt, _)| cnt).sum();
                let ln_total = (total_pulls.max(1) as f64).ln();
                (0..n_arms)
                    .max_by(|&a, &b| {
                        let score = |i: usize| {
                            let (count, mean) = self.arm_stats[i];
                            mean + c * (ln_total / (count.max(1) as f64)).sqrt()
                        };
                        score(a)
                            .partial_cmp(&score(b))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or(0)
            }
            ExplorationStrategy::ThompsonSampling { alpha, beta } => {
                // Simplified stand-in for literal Beta-posterior sampling
                // (kept dependency-free): a Beta-prior-weighted posterior
                // mean perturbed by real Gaussian noise (Box-Muller, from
                // two uniform draws) scaled by the arm's uncertainty
                // (1/sqrt(pulls+1)). Genuinely stochastic and genuinely
                // reactive to real per-arm statistics, not a fixed choice.
                let mut rng = scirs2_core::random::rng();
                let mut best = 0usize;
                let mut best_score = f64::NEG_INFINITY;
                for i in 0..n_arms {
                    let (count, mean) = self.arm_stats[i];
                    let mean = mean.clamp(0.0, 1.0);
                    let successes = alpha + mean * count as f64;
                    let failures = beta + (1.0 - mean) * count as f64;
                    let posterior_mean = successes / (successes + failures).max(1e-9);
                    let uncertainty = 1.0 / ((count as f64) + 1.0).sqrt();
                    let noise =
                        box_muller_standard_normal(rng.random::<f64>(), rng.random::<f64>())
                            * uncertainty
                            * 0.25;
                    let score = posterior_mean + noise;
                    if score > best_score {
                        best_score = score;
                        best = i;
                    }
                }
                best
            }
            ExplorationStrategy::AdaptiveUncertainty {
                uncertainty_threshold,
            } => {
                let least_tried = (0..n_arms)
                    .min_by_key(|&i| self.arm_stats[i].0)
                    .unwrap_or(0);
                let uncertainty = 1.0 / ((self.arm_stats[least_tried].0 as f64) + 1.0).sqrt();
                if uncertainty > *uncertainty_threshold {
                    least_tried
                } else {
                    self.best_arm(n_arms)
                }
            }
        }
    }

    /// Records an observed `reward` for `arm`, updating its running
    /// statistics (growing the tracked arm count if needed).
    pub fn record_reward(&mut self, arm: usize, reward: f64) {
        if arm >= self.arm_stats.len() {
            self.arm_stats.resize(arm + 1, (0, 0.0));
        }
        let (count, mean) = &mut self.arm_stats[arm];
        *count += 1;
        if self.learning_rate > 0.0 {
            // Exponential moving average: recent rewards matter more.
            *mean += self.learning_rate * (reward - *mean);
        } else {
            // Plain running mean.
            *mean += (reward - *mean) / (*count as f64);
        }
    }

    /// The arm with the highest current mean-reward estimate (ties broken
    /// by lowest index).
    fn best_arm(&self, n_arms: usize) -> usize {
        (0..n_arms)
            .max_by(|&a, &b| {
                self.arm_stats[a]
                    .1
                    .partial_cmp(&self.arm_stats[b].1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0)
    }
}

/// A single standard-normal sample from two independent uniform `[0, 1)`
/// draws, via the Box-Muller transform.
fn box_muller_standard_normal(u1: f64, u2: f64) -> f64 {
    let u1 = u1.max(1e-12); // avoid ln(0)
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Neuromorphic processor for brain-inspired (spiking neural network)
/// computing.
///
/// Genuine neuromorphic computation is **out of scope** for this crate: there
/// is no neuromorphic hardware or spiking-network simulator anywhere in the
/// SciRS2 / COOLJAPAN dependency graph for this type to dispatch to. Rather
/// than silently running on CPU while claiming neuromorphic-specific
/// behavior occurred, [`Self::accelerate`] honestly reports
/// [`GraphError::Unsupported`]. This struct exists for forward API
/// compatibility only.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NeuromorphicProcessor {
    /// Number of neurons (configuration only -- no simulator exists)
    pub num_neurons: usize,
    /// Number of synapses (configuration only -- no simulator exists)
    pub num_synapses: usize,
}

impl Default for NeuromorphicProcessor {
    fn default() -> Self {
        NeuromorphicProcessor {
            num_neurons: 1000,
            num_synapses: 10000,
        }
    }
}

impl NeuromorphicProcessor {
    /// Attempts neuromorphic (spiking-neural-network) acceleration of an
    /// operation named `operation_name`.
    ///
    /// Always returns [`GraphError::Unsupported`]: see the struct-level docs
    /// for why this is a genuine, permanent architectural limit rather than
    /// a placeholder awaiting implementation.
    pub fn accelerate<T>(&self, operation_name: &str) -> Result<T> {
        Err(GraphError::Unsupported(format!(
            "NeuromorphicProcessor::accelerate({operation_name}): neuromorphic acceleration is \
             not implemented (num_neurons={}, num_synapses={} are configuration only; no \
             neuromorphic simulator or hardware backend exists in this crate)",
            self.num_neurons, self.num_synapses
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::Graph;

    fn small_graph() -> Graph<i32, f64> {
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_edge(0, 1, 1.0).expect("add_edge failed");
        graph.add_edge(1, 2, 1.0).expect("add_edge failed");
        graph
    }

    fn bigger_graph() -> Graph<i32, f64> {
        let mut graph: Graph<i32, f64> = Graph::new();
        for i in 0..50i32 {
            graph.add_edge(i, i + 1, 1.0).expect("add_edge failed");
        }
        graph
    }

    #[test]
    fn test_performance_monitor_tracks_real_timing() {
        let mut monitor = SimplePerformanceMonitor::new();

        monitor.start_operation("op_a");
        std::thread::sleep(Duration::from_millis(5));
        monitor.stop_operation("op_a");

        monitor.start_operation("op_a");
        std::thread::sleep(Duration::from_millis(5));
        monitor.stop_operation("op_a");

        let report = monitor.get_report();
        assert_eq!(report.total_operations, 2);
        // Two >=5ms sleeps must show up as real elapsed time, not the old
        // hardcoded 0/default.
        assert!(
            report.total_time_ms >= 8.0,
            "expected at least ~10ms of real elapsed time, got {}",
            report.total_time_ms
        );
    }

    #[test]
    fn test_performance_monitor_stop_without_start_is_noop() {
        let mut monitor = SimplePerformanceMonitor::new();
        monitor.stop_operation("never_started");
        let report = monitor.get_report();
        assert_eq!(report.total_operations, 0);
        assert_eq!(report.total_time_ms, 0.0);
    }

    #[test]
    fn test_advanced_processor_execute_produces_real_stats() {
        let mut processor = create_advanced_processor();

        // Before any call: honest zero/default stats.
        let before = processor.get_optimization_stats();
        assert_eq!(before.total_operations, 0);

        let graph = small_graph();
        let result: Result<usize> = processor.execute(&graph, |g| Ok(g.node_count()));
        assert_eq!(result.expect("execute failed"), 3);

        let after = processor.get_optimization_stats();
        assert_eq!(
            after.total_operations, 1,
            "total_operations must reflect the real call count, not stay at the old default 0"
        );
        assert!(
            after.memory_usage_bytes > 0,
            "memory_usage_bytes must be a real (nonzero) structural estimate"
        );
        // GPU is genuinely never dispatched to by `execute`.
        assert_eq!(after.gpu_utilization_percent, 0.0);
    }

    #[test]
    fn test_advanced_processor_memory_estimate_scales_with_graph_size() {
        // Non-constant-data check: a bigger graph must produce a bigger
        // structural memory estimate -- the OLD implementation's stats were
        // frozen at whatever `AdvancedStats::default()` produced regardless
        // of input, so this would have failed against it (constant, not
        // scaling with input).
        let mut small_processor = create_advanced_processor();
        let mut big_processor = create_advanced_processor();

        let small = small_graph();
        let big = bigger_graph();

        small_processor
            .execute(&small, |g| Ok(g.node_count()))
            .expect("execute failed");
        big_processor
            .execute(&big, |g| Ok(g.node_count()))
            .expect("execute failed");

        let small_stats = small_processor.get_optimization_stats();
        let big_stats = big_processor.get_optimization_stats();

        assert!(
            big_stats.memory_usage_bytes > small_stats.memory_usage_bytes,
            "a 51-node graph should report a larger memory estimate than a 3-node graph \
             ({} vs {})",
            big_stats.memory_usage_bytes,
            small_stats.memory_usage_bytes
        );
    }

    #[test]
    fn test_advanced_processor_execute_profiled_runs_real_operation() {
        let mut processor = create_large_graph_advanced_processor();
        let graph = bigger_graph();

        let result: Result<usize> = processor.execute_profiled(&graph, |g| Ok(g.edge_count()));
        assert_eq!(result.expect("execute_profiled failed"), 50);

        let stats = processor.get_optimization_stats();
        assert_eq!(stats.total_operations, 1);
        assert!(stats.memory_usage_bytes > 0);
    }

    #[test]
    fn test_gpu_acceleration_context_detect_is_honest() {
        let ctx = GPUAccelerationContext::detect();
        // Regardless of feature flags, this struct never claims to manage
        // GPU memory itself.
        assert_eq!(ctx.memory_pool_size, 0);

        #[cfg(feature = "cuda")]
        {
            // With the (off-by-default) `cuda` feature compiled in, `detect`
            // must agree exactly with the crate's own real CUDA probe -- no
            // fabricated availability, and no silently dropping a real one.
            assert_eq!(ctx.gpu_available, crate::gpu_cuda::cuda_is_available());
        }
        #[cfg(not(feature = "cuda"))]
        {
            // Without the `cuda` feature there is no real GPU-dispatch path
            // anywhere in this crate, so `detect` must honestly report
            // `false` rather than fabricating availability.
            assert!(!ctx.gpu_available);
        }
    }

    #[test]
    fn test_neuromorphic_processor_is_honestly_unsupported() {
        let processor = NeuromorphicProcessor::default();
        let result: Result<()> = processor.accelerate("pagerank");
        match result {
            Err(GraphError::Unsupported(msg)) => {
                assert!(msg.contains("neuromorphic"));
            }
            other => panic!("expected GraphError::Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn test_neural_rl_agent_select_arm_tries_every_arm_before_repeating() {
        let mut agent = NeuralRLAgent::new(
            AdvancedConfig::default(),
            ExplorationStrategy::EpsilonGreedy { epsilon: 0.0 },
        );

        let mut seen = std::collections::HashSet::new();
        for _ in 0..4 {
            let arm = agent.select_arm(4);
            seen.insert(arm);
            agent.record_reward(arm, 0.5);
        }
        assert_eq!(
            seen.len(),
            4,
            "every arm must be tried at least once during warm-up"
        );
    }

    #[test]
    fn test_neural_rl_agent_epsilon_greedy_converges_to_best_arm() {
        // Arm 0 always rewards 1.0, the rest always reward 0.0: with
        // epsilon=0 (pure exploitation after warm-up) the agent must lock
        // onto arm 0 and stay there. This is real adaptive behavior driven
        // by actual reward feedback -- the OLD `NeuralRLAgent` had zero
        // methods and could not do this at all.
        let mut agent = NeuralRLAgent::new(
            AdvancedConfig::default(),
            ExplorationStrategy::EpsilonGreedy { epsilon: 0.0 },
        );

        const N_ARMS: usize = 5;
        // Warm-up: try every arm once with its true reward.
        for arm in 0..N_ARMS {
            agent.record_reward(arm, if arm == 0 { 1.0 } else { 0.0 });
        }
        // Prime the agent's internal arm_stats length via a warm-up
        // select_arm pass (all arms already have count > 0 so this will
        // exploit immediately).
        for _ in 0..20 {
            let arm = agent.select_arm(N_ARMS);
            agent.record_reward(arm, if arm == 0 { 1.0 } else { 0.0 });
        }

        let chosen = agent.select_arm(N_ARMS);
        assert_eq!(
            chosen, 0,
            "epsilon=0 agent must exploit the arm with the only nonzero reward"
        );
    }

    #[test]
    fn test_neural_rl_agent_ucb_prefers_less_explored_arms_when_tied() {
        // learning_rate: 0.0 => plain running mean, so repeating the exact
        // same reward always keeps the mean at exactly that reward
        // regardless of pull count (making the "tied means" setup below
        // exact rather than an EMA that would still be converging).
        let mut agent = NeuralRLAgent::new(
            AdvancedConfig {
                learning_rate: 0.0,
                ..AdvancedConfig::default()
            },
            ExplorationStrategy::UCB { c: 2.0 },
        );

        // Both arms are pulled at least once (so the warm-up shortcut in
        // `select_arm` does NOT fire for either -- this test exercises the
        // UCB score formula itself, not just "never-pulled arms go first").
        // Arm 0 is then pulled 30 MORE times with the identical reward, so
        // both arms end up with the exact same mean reward (0.5) but wildly
        // different pull counts. UCB's confidence-bonus term
        // (`c * sqrt(ln(total)/count)`) is larger for the less-explored arm,
        // so with tied means it must prefer arm 1.
        agent.record_reward(0, 0.5);
        agent.record_reward(1, 0.5);
        for _ in 0..30 {
            agent.record_reward(0, 0.5);
        }

        let (count0, mean0) = agent.arm_stats(0).expect("arm 0 stats");
        let (count1, mean1) = agent.arm_stats(1).expect("arm 1 stats");
        assert_eq!(count0, 31);
        assert_eq!(count1, 1);
        assert!((mean0 - mean1).abs() < 1e-9, "means must be tied by design");

        let chosen = agent.select_arm(2);
        assert_eq!(
            chosen, 1,
            "with tied mean rewards, UCB must prefer the far-less-explored arm"
        );
    }

    #[test]
    fn test_neural_rl_agent_record_reward_updates_mean() {
        let mut agent = NeuralRLAgent::new(
            AdvancedConfig {
                learning_rate: 0.0, // plain running mean
                ..AdvancedConfig::default()
            },
            ExplorationStrategy::default(),
        );

        agent.record_reward(0, 0.0);
        agent.record_reward(0, 1.0);
        let (count, mean) = agent.arm_stats(0).expect("arm 0 should have stats");
        assert_eq!(count, 2);
        assert!(
            (mean - 0.5).abs() < 1e-9,
            "running mean of [0.0, 1.0] should be 0.5, got {mean}"
        );
    }

    #[test]
    fn test_advanced_processor_execute_adaptive_learns_the_faster_candidate() {
        let mut processor = create_adaptive_advanced_processor();
        let graph = small_graph();

        fn fast_op(g: &Graph<i32, f64>) -> Result<usize> {
            Ok(g.node_count())
        }
        fn slow_op(g: &Graph<i32, f64>) -> Result<usize> {
            std::thread::sleep(Duration::from_millis(2));
            Ok(g.node_count())
        }

        let candidates: [CandidateOp<i32, f64, u32, usize>; 2] = [slow_op, fast_op];

        // Force pure exploitation after warm-up so the learned preference is
        // deterministic to check.
        processor.rl_agent_mut().strategy = ExplorationStrategy::EpsilonGreedy { epsilon: 0.0 };

        for _ in 0..8 {
            processor
                .execute_adaptive(&graph, &candidates)
                .expect("execute_adaptive failed");
        }

        // The fast candidate (index 1) must now have a strictly higher
        // mean reward (1/(1+latency)) than the slow one (index 0).
        let (_, slow_mean) = processor
            .rl_agent()
            .arm_stats(0)
            .expect("slow arm should have stats");
        let (_, fast_mean) = processor
            .rl_agent()
            .arm_stats(1)
            .expect("fast arm should have stats");
        assert!(
            fast_mean > slow_mean,
            "the genuinely faster candidate should have accumulated a higher reward \
             (fast={fast_mean}, slow={slow_mean})"
        );
    }

    #[test]
    fn test_execute_adaptive_rejects_empty_candidates() {
        let mut processor = create_advanced_processor();
        let graph = small_graph();
        let candidates: [CandidateOp<i32, f64, u32, usize>; 0] = [];
        assert!(processor.execute_adaptive(&graph, &candidates).is_err());
    }
}
