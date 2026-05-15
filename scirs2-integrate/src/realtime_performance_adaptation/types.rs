//! Type definitions for real-time performance adaptation system
//!
//! This module contains all the core type definitions, structs, enums, and
//! basic implementations for the real-time performance monitoring and
//! adaptive optimization system.

#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]

use crate::common::IntegrateFloat;
use crate::error::{IntegrateError, IntegrateResult};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

// ================================================================================================
// MAIN COMPONENT TYPES
// ================================================================================================

/// Real-time adaptive performance optimization system
pub struct RealTimeAdaptiveOptimizer<F: IntegrateFloat> {
    /// Performance monitoring engine
    pub performance_monitor: Arc<Mutex<PerformanceMonitoringEngine>>,
    /// Adaptive algorithm selector
    pub algorithm_selector: Arc<RwLock<AdaptiveAlgorithmSelector<F>>>,
    /// Dynamic resource manager
    pub resource_manager: Arc<Mutex<DynamicResourceManager>>,
    /// Predictive performance model
    pub performance_predictor: Arc<Mutex<PerformancePredictor<F>>>,
    /// Machine learning optimizer
    pub ml_optimizer: Arc<Mutex<MachineLearningOptimizer<F>>>,
    /// Anomaly detector
    pub anomaly_detector: Arc<Mutex<PerformanceAnomalyDetector>>,
    /// Configuration adaptation engine
    pub config_adapter: Arc<Mutex<ConfigurationAdapter<F>>>,
}

/// Performance monitoring and metrics collection engine
pub struct PerformanceMonitoringEngine {
    /// Real-time metrics collection
    pub metrics_collector: MetricsCollector,
    /// Performance history database
    pub performance_history: PerformanceHistory,
    /// System resource monitor
    pub system_monitor: SystemResourceMonitor,
    /// Network performance monitor (for distributed computing)
    pub network_monitor: NetworkPerformanceMonitor,
}

/// Adaptive algorithm selection based on real-time performance
pub struct AdaptiveAlgorithmSelector<F: IntegrateFloat> {
    /// Available algorithm registry
    pub algorithm_registry: AlgorithmRegistry<F>,
    /// Performance-based selection criteria
    pub selection_criteria: SelectionCriteria,
    /// Algorithm switching policies
    pub switching_policies: SwitchingPolicies,
    /// Performance prediction models for each algorithm
    pub algorithm_models: HashMap<String, AlgorithmPerformanceModel<F>>,
}

/// Dynamic resource allocation and management
pub struct DynamicResourceManager {
    /// CPU resource manager
    pub cpu_manager: CpuResourceManager,
    /// Memory resource manager
    pub memory_manager: MemoryResourceManager,
    /// GPU resource manager
    pub gpu_manager: GpuResourceManager,
    /// Network resource manager
    pub network_manager: NetworkResourceManager,
    /// Load balancing strategies
    pub load_balancer: LoadBalancer,
}

/// Predictive performance modeling system
pub struct PerformancePredictor<F: IntegrateFloat> {
    /// Performance model registry
    pub model_registry: ModelRegistry<F>,
    /// Feature engineering pipeline
    pub feature_engineering: FeatureEngineering<F>,
    /// Model training and validation
    pub model_trainer: ModelTrainer<F>,
    /// Prediction accuracy tracker
    pub accuracy_tracker: PredictionAccuracyTracker,
}

/// Machine learning-based optimization engine
pub struct MachineLearningOptimizer<F: IntegrateFloat> {
    /// Reinforcement learning agent
    pub rl_agent: ReinforcementLearningAgent<F>,
    /// Bayesian optimization engine
    pub bayesian_optimizer: BayesianOptimizer<F>,
    /// Neural architecture search
    pub nas_engine: NeuralArchitectureSearch<F>,
    /// Hyperparameter optimization
    pub hyperopt_engine: HyperparameterOptimizer<F>,
}

/// Performance anomaly detection and recovery
pub struct PerformanceAnomalyDetector {
    /// Statistical anomaly detection
    pub statistical_detector: StatisticalAnomalyDetector,
    /// Machine learning anomaly detection
    pub ml_detector: MLAnomalyDetector,
    /// System health monitoring
    pub health_monitor: SystemHealthMonitor,
    /// Automatic recovery mechanisms
    pub recovery_manager: AutomaticRecoveryManager,
}

/// Configuration adaptation engine
pub struct ConfigurationAdapter<F: IntegrateFloat> {
    /// Parameter adaptation rules
    pub adaptation_rules: AdaptationRules<F>,
    /// Configuration space explorer
    pub config_explorer: ConfigurationSpaceExplorer<F>,
    /// Constraint satisfaction engine
    pub constraint_solver: ConstraintSatisfactionEngine<F>,
    /// Multi-objective optimization
    pub multi_objective_optimizer: MultiObjectiveOptimizer<F>,
}

// ================================================================================================
// CORE PERFORMANCE TYPES
// ================================================================================================

/// Real-time performance metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Timestamp of measurement
    pub timestamp: Instant,
    /// Execution time per step
    pub step_time: Duration,
    /// Throughput (steps per second)
    pub throughput: f64,
    /// Memory usage (bytes)
    pub memory_usage: usize,
    /// CPU utilization (percentage)
    pub cpu_utilization: f64,
    /// GPU utilization (percentage)
    pub gpu_utilization: f64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Network bandwidth utilization
    pub network_bandwidth: f64,
    /// Error estimation accuracy
    pub error_accuracy: f64,
    /// Solver convergence rate
    pub convergence_rate: f64,
}

/// Algorithm performance characteristics
#[derive(Debug, Clone)]
pub struct AlgorithmCharacteristics<F: IntegrateFloat> {
    /// Algorithm name
    pub name: String,
    /// Computational complexity
    pub complexity: ComputationalComplexity,
    /// Memory requirements
    pub memory_requirements: MemoryRequirements,
    /// Numerical stability properties
    pub stability: NumericalStability,
    /// Parallelization potential
    pub parallelism: ParallelismCharacteristics,
    /// Accuracy characteristics
    pub accuracy: AccuracyCharacteristics<F>,
}

/// Performance adaptation strategy
#[derive(Debug, Clone)]
pub struct AdaptationStrategy<F: IntegrateFloat> {
    /// Target performance metrics
    pub target_metrics: TargetMetrics,
    /// Adaptation triggers
    pub triggers: AdaptationTriggers,
    /// Optimization objectives
    pub objectives: OptimizationObjectives<F>,
    /// Constraint specifications
    pub constraints: PerformanceConstraints,
}

/// Performance analysis result
#[derive(Debug, Clone)]
pub struct PerformanceAnalysis {
    pub average_throughput: f64,
    pub average_cpu_utilization: f64,
    pub average_memory_usage: usize,
    pub performance_trend: PerformanceTrend,
    pub bottlenecks: Vec<PerformanceBottleneck>,
}

// ================================================================================================
// ENUMS
// ================================================================================================

/// Performance trend analysis
#[derive(Debug, Clone)]
pub enum PerformanceTrend {
    Improving,
    Stable,
    Degrading,
}

/// Performance bottleneck types
#[derive(Debug, Clone)]
pub enum PerformanceBottleneck {
    CPU,
    Memory,
    Network,
    Storage,
    GPU,
}

/// Anomaly types
#[derive(Debug, Clone)]
pub enum AnomalyType {
    PerformanceDegradation,
    ResourceSpike,
    UnexpectedBehavior,
    SystemInstability,
}

/// Anomaly severity levels
#[derive(Debug, Clone, PartialEq, PartialOrd, Default)]
pub enum AnomalySeverity {
    #[default]
    Low,
    Medium,
    High,
    Critical,
}

// ================================================================================================
// OPTIMIZATION RESULT TYPES
// ================================================================================================

/// Optimization recommendations
#[derive(Debug, Clone)]
pub struct OptimizationRecommendations<F: IntegrateFloat> {
    pub algorithm_changes: Vec<AlgorithmRecommendation<F>>,
    pub parameter_adjustments: Vec<ParameterAdjustment<F>>,
    pub resource_reallocations: Vec<ResourceReallocation>,
    pub performance_predictions: PerformanceImprovement,
}

/// Algorithm switch recommendation
#[derive(Debug, Clone)]
pub struct AlgorithmSwitchRecommendation<F: IntegrateFloat> {
    pub from_algorithm: String,
    pub to_algorithm: AlgorithmCandidate<F>,
    pub expected_gain: f64,
    pub switching_cost: f64,
    pub confidence: f64,
}

/// Algorithm candidate
#[derive(Debug, Clone)]
pub struct AlgorithmCandidate<F: IntegrateFloat> {
    pub name: String,
    pub expected_performance_gain: f64,
    pub confidence: f64,
    pub characteristics: AlgorithmCharacteristics<F>,
}

/// Algorithm recommendation
#[derive(Debug, Clone)]
pub struct AlgorithmRecommendation<F: IntegrateFloat> {
    pub algorithm: String,
    pub reason: String,
    pub expected_improvement: f64,
    pub parameters: HashMap<String, F>,
}

/// Parameter adjustment recommendation
#[derive(Debug, Clone)]
pub struct ParameterAdjustment<F: IntegrateFloat> {
    pub parameter_name: String,
    pub current_value: F,
    pub recommended_value: F,
    pub adjustment_reason: String,
}

/// Resource reallocation recommendation
#[derive(Debug, Clone, Default)]
pub struct ResourceReallocation {
    pub resource_type: String,
    pub current_allocation: f64,
    pub recommended_allocation: f64,
    pub expected_benefit: f64,
}

/// Performance improvement prediction
#[derive(Debug, Clone, Default)]
pub struct PerformanceImprovement {
    pub expected_throughput_gain: f64,
    pub expected_memory_reduction: f64,
    pub expected_energy_savings: f64,
    pub confidence: f64,
}

// ================================================================================================
// ANOMALY DETECTION TYPES
// ================================================================================================

/// Anomaly analysis result
#[derive(Debug, Clone)]
pub struct AnomalyAnalysisResult {
    pub anomalies_detected: Vec<PerformanceAnomaly>,
    pub analysis: AnomalyAnalysis,
    pub recovery_plan: Option<RecoveryPlan>,
    pub recovery_executed: bool,
}

/// Performance anomaly
#[derive(Debug, Clone)]
pub struct PerformanceAnomaly {
    pub anomaly_type: AnomalyType,
    pub severity: AnomalySeverity,
    pub detected_at: Instant,
    pub affected_metrics: Vec<String>,
}

/// Anomaly analysis result
#[derive(Debug, Clone, Default)]
pub struct AnomalyAnalysis {
    pub severity: AnomalySeverity,
    pub root_cause: String,
    pub affected_components: Vec<String>,
    pub recommended_actions: Vec<String>,
}

/// Recovery plan for anomalies
#[derive(Debug, Clone, Default)]
pub struct RecoveryPlan {
    pub recovery_steps: Vec<String>,
    pub estimated_recovery_time: Duration,
    pub rollback_plan: Vec<String>,
    pub monitoring_requirements: Vec<String>,
}

// ================================================================================================
// MONITORING COMPONENT TYPES
// ================================================================================================

/// Metrics collection engine
#[derive(Debug, Clone, Default)]
pub struct MetricsCollector {
    pub collection_interval: Duration,
    pub metric_buffer: Vec<PerformanceMetrics>,
    pub active_collectors: Vec<String>,
}

/// Performance history database
#[derive(Debug, Clone, Default)]
pub struct PerformanceHistory {
    pub metrics_history: VecDeque<PerformanceMetrics>,
    pub max_history_size: usize,
    pub aggregated_stats: HashMap<String, f64>,
}

/// System resource monitor
#[derive(Debug, Clone, Default)]
pub struct SystemResourceMonitor {
    pub cpu_monitors: Vec<CpuMonitor>,
    pub memory_monitor: MemoryMonitor,
    pub disk_monitor: DiskMonitor,
}

/// Network performance monitor
#[derive(Debug, Clone, Default)]
pub struct NetworkPerformanceMonitor {
    pub bandwidth_monitor: BandwidthMonitor,
    pub latency_monitor: LatencyMonitor,
    pub packet_loss_monitor: PacketLossMonitor,
}

// ================================================================================================
// ALGORITHM SELECTION TYPES
// ================================================================================================

/// Algorithm registry for adaptive selection
#[derive(Debug, Clone)]
pub struct AlgorithmRegistry<F: IntegrateFloat> {
    pub available_algorithms: HashMap<String, AlgorithmCharacteristics<F>>,
    pub performance_models: HashMap<String, AlgorithmPerformanceModel<F>>,
    pub selection_history: Vec<String>,
}

/// Selection criteria for algorithm switching
#[derive(Debug, Clone, Default)]
pub struct SelectionCriteria {
    pub performance_weight: f64,
    pub accuracy_weight: f64,
    pub stability_weight: f64,
    pub memory_weight: f64,
}

/// Algorithm switching policies
#[derive(Debug, Clone, Default)]
pub struct SwitchingPolicies {
    pub switch_threshold: f64,
    pub cooldown_period: Duration,
    pub max_switches_per_hour: usize,
}

/// Algorithm performance model for prediction
#[derive(Debug, Clone)]
pub struct AlgorithmPerformanceModel<F: IntegrateFloat> {
    pub model_type: String,
    pub parameters: HashMap<String, F>,
    pub accuracy: f64,
    pub last_updated: Instant,
}

// ================================================================================================
// RESOURCE MANAGEMENT TYPES
// ================================================================================================

/// CPU resource manager
#[derive(Debug, Clone, Default)]
pub struct CpuResourceManager {
    pub cpu_allocation: HashMap<usize, f64>, // core_id -> utilization
    pub thermal_state: ThermalState,
    pub frequency_scaling: FrequencyScaling,
}

/// Memory resource manager
#[derive(Debug, Clone, Default)]
pub struct MemoryResourceManager {
    pub memory_pools: Vec<MemoryPool>,
    pub allocation_strategy: AllocationStrategy,
    pub gc_policy: GarbageCollectionPolicy,
}

/// GPU resource manager
#[derive(Debug, Clone, Default)]
pub struct GpuResourceManager {
    pub gpu_devices: Vec<GpuDevice>,
    pub memory_allocation: HashMap<usize, usize>, // device_id -> allocated_bytes
    pub compute_allocation: HashMap<usize, f64>,  // device_id -> utilization
}

/// Network resource manager
#[derive(Debug, Clone, Default)]
pub struct NetworkResourceManager {
    pub bandwidth_allocation: BandwidthAllocation,
    pub connection_pool: ConnectionPool,
    pub load_balancing: NetworkLoadBalancing,
}

/// Load balancer for distributed computing
#[derive(Debug, Clone, Default)]
pub struct LoadBalancer {
    pub balancing_strategy: String,
    pub node_weights: HashMap<String, f64>,
    pub current_load: HashMap<String, f64>,
}

// ================================================================================================
// PREDICTION TYPES
// ================================================================================================

/// Model registry for performance prediction
#[derive(Debug, Clone)]
pub struct ModelRegistry<F: IntegrateFloat> {
    pub models: HashMap<String, AlgorithmPerformanceModel<F>>,
    pub default_model: AlgorithmPerformanceModel<F>,
}

/// Feature engineering pipeline
#[derive(Debug, Clone)]
pub struct FeatureEngineering<F: IntegrateFloat> {
    pub feature_extractors: Vec<String>,
    pub normalization_params: HashMap<String, F>,
    pub feature_importance: HashMap<String, f64>,
}

/// Model trainer for performance prediction
#[derive(Debug, Clone)]
pub struct ModelTrainer<F: IntegrateFloat> {
    pub training_algorithm: String,
    pub hyperparameters: HashMap<String, f64>,
    pub cross_validation_folds: usize,
    pub phantom: std::marker::PhantomData<F>,
}

/// Prediction accuracy tracker
#[derive(Debug, Clone, Default)]
pub struct PredictionAccuracyTracker {
    pub mse: f64,
    pub mae: f64,
    pub r_squared: f64,
    pub prediction_count: usize,
}

// ================================================================================================
// MACHINE LEARNING TYPES
// ================================================================================================

/// Reinforcement learning agent
#[derive(Debug, Clone)]
pub struct ReinforcementLearningAgent<F: IntegrateFloat> {
    pub agent_type: String,
    pub learning_rate: f64,
    pub exploration_rate: f64,
    pub phantom: std::marker::PhantomData<F>,
}

/// Bayesian optimization engine
#[derive(Debug, Clone)]
pub struct BayesianOptimizer<F: IntegrateFloat> {
    pub acquisition_function: String,
    pub kernel_type: String,
    pub num_iterations: usize,
    pub phantom: std::marker::PhantomData<F>,
}

/// Neural architecture search engine
#[derive(Debug, Clone)]
pub struct NeuralArchitectureSearch<F: IntegrateFloat> {
    pub search_strategy: String,
    pub architecture_space: String,
    pub evaluation_budget: usize,
    pub phantom: std::marker::PhantomData<F>,
}

/// Hyperparameter optimizer
#[derive(Debug, Clone)]
pub struct HyperparameterOptimizer<F: IntegrateFloat> {
    pub optimization_algorithm: String,
    pub search_space: HashMap<String, (f64, f64)>,
    pub max_evaluations: usize,
    pub phantom: std::marker::PhantomData<F>,
}

// ================================================================================================
// ANOMALY DETECTION COMPONENTS
// ================================================================================================

/// Statistical anomaly detector
#[derive(Debug, Clone, Default)]
pub struct StatisticalAnomalyDetector {
    pub detection_algorithms: Vec<String>,
    pub thresholds: HashMap<String, f64>,
    pub confidence_interval: f64,
}

/// Machine learning anomaly detector
#[derive(Debug, Clone, Default)]
pub struct MLAnomalyDetector {
    pub model_type: String,
    pub training_data: Vec<PerformanceMetrics>,
    pub detection_threshold: f64,
}

/// System health monitor
#[derive(Debug, Clone, Default)]
pub struct SystemHealthMonitor {
    pub health_score: f64,
    pub critical_components: Vec<String>,
    pub alert_thresholds: HashMap<String, f64>,
}

/// Automatic recovery manager
#[derive(Debug, Clone, Default)]
pub struct AutomaticRecoveryManager {
    pub recovery_strategies: HashMap<String, RecoveryStrategy>,
    pub recovery_history: Vec<RecoveryEvent>,
    pub enabled: bool,
}

// ================================================================================================
// CONFIGURATION ADAPTATION TYPES
// ================================================================================================

/// Adaptation rules for configuration
#[derive(Debug, Clone)]
pub struct AdaptationRules<F: IntegrateFloat> {
    pub rules: Vec<String>,
    pub rule_weights: HashMap<String, F>,
    pub activation_thresholds: HashMap<String, f64>,
}

/// Configuration space explorer
#[derive(Debug, Clone)]
pub struct ConfigurationSpaceExplorer<F: IntegrateFloat> {
    pub exploration_strategy: String,
    pub space_dimensions: usize,
    pub explored_configurations: Vec<String>,
    pub phantom: std::marker::PhantomData<F>,
}

/// Constraint satisfaction engine
#[derive(Debug, Clone)]
pub struct ConstraintSatisfactionEngine<F: IntegrateFloat> {
    pub constraint_solver: String,
    pub constraints: Vec<String>,
    pub satisfaction_tolerance: F,
}

/// Multi-objective optimizer
#[derive(Debug, Clone)]
pub struct MultiObjectiveOptimizer<F: IntegrateFloat> {
    pub algorithm: String,
    pub pareto_front_size: usize,
    pub objectives: Vec<String>,
    pub phantom: std::marker::PhantomData<F>,
}

// ================================================================================================
// DETAILED SYSTEM COMPONENT TYPES
// ================================================================================================

/// CPU monitor
#[derive(Debug, Clone, Default)]
pub struct CpuMonitor {
    pub core_id: usize,
    pub utilization: f64,
    pub temperature: f64,
}

/// Memory monitor
#[derive(Debug, Clone, Default)]
pub struct MemoryMonitor {
    pub total_memory: usize,
    pub used_memory: usize,
    pub swap_usage: usize,
}

/// Disk monitor
#[derive(Debug, Clone, Default)]
pub struct DiskMonitor {
    pub read_iops: f64,
    pub write_iops: f64,
    pub utilization: f64,
}

/// Bandwidth monitor
#[derive(Debug, Clone, Default)]
pub struct BandwidthMonitor {
    pub inbound_bandwidth: f64,
    pub outbound_bandwidth: f64,
    pub peak_bandwidth: f64,
}

/// Latency monitor
#[derive(Debug, Clone, Default)]
pub struct LatencyMonitor {
    pub avg_latency: Duration,
    pub p99_latency: Duration,
    pub jitter: Duration,
}

/// Packet loss monitor
#[derive(Debug, Clone, Default)]
pub struct PacketLossMonitor {
    pub loss_rate: f64,
    pub retransmission_rate: f64,
}

/// Thermal state
#[derive(Debug, Clone, Default)]
pub struct ThermalState {
    pub temperature: f64,
    pub throttling_active: bool,
}

/// Frequency scaling
#[derive(Debug, Clone, Default)]
pub struct FrequencyScaling {
    pub current_frequency: f64,
    pub target_frequency: f64,
    pub scaling_governor: String,
}

/// Memory pool
#[derive(Debug, Clone, Default)]
pub struct MemoryPool {
    pub pool_id: usize,
    pub size: usize,
    pub allocation_type: String,
}

/// Allocation strategy
#[derive(Debug, Clone, Default)]
pub struct AllocationStrategy {
    pub strategy_type: String,
    pub parameters: HashMap<String, f64>,
}

/// Garbage collection policy
#[derive(Debug, Clone, Default)]
pub struct GarbageCollectionPolicy {
    pub gc_type: String,
    pub threshold: f64,
}

/// GPU device
#[derive(Debug, Clone, Default)]
pub struct GpuDevice {
    pub device_id: usize,
    pub name: String,
    pub memory_size: usize,
    pub compute_units: usize,
}

/// Bandwidth allocation
#[derive(Debug, Clone, Default)]
pub struct BandwidthAllocation {
    pub total_bandwidth: f64,
    pub allocated_bandwidth: HashMap<String, f64>,
}

/// Connection pool
#[derive(Debug, Clone, Default)]
pub struct ConnectionPool {
    pub max_connections: usize,
    pub active_connections: usize,
    pub connection_timeout: Duration,
}

/// Network load balancing
#[derive(Debug, Clone, Default)]
pub struct NetworkLoadBalancing {
    pub algorithm: String,
    pub weights: HashMap<String, f64>,
}

/// Recovery strategy
#[derive(Debug, Clone, Default)]
pub struct RecoveryStrategy {
    pub strategy_type: String,
    pub steps: Vec<String>,
    pub timeout: Duration,
}

/// Recovery event
#[derive(Debug, Clone)]
pub struct RecoveryEvent {
    pub timestamp: Instant,
    pub event_type: String,
    pub success: bool,
}

// ================================================================================================
// ALGORITHM CHARACTERISTICS TYPES
// ================================================================================================

/// Computational complexity characteristics
#[derive(Debug, Clone, Default)]
pub struct ComputationalComplexity {
    pub time_complexity: String,
    pub space_complexity: String,
    pub arithmetic_operations_per_step: usize,
}

/// Memory requirements specification
#[derive(Debug, Clone, Default)]
pub struct MemoryRequirements {
    pub base_memory: usize,
    pub scaling_factor: f64,
    pub peak_memory_multiplier: f64,
}

/// Numerical stability properties
#[derive(Debug, Clone, Default)]
pub struct NumericalStability {
    pub stability_region: String,
    pub condition_number_sensitivity: f64,
    pub error_propagation_factor: f64,
}

/// Parallelism characteristics
#[derive(Debug, Clone, Default)]
pub struct ParallelismCharacteristics {
    pub parallel_efficiency: f64,
    pub scaling_factor: f64,
    pub communication_overhead: f64,
}

/// Accuracy characteristics
#[derive(Debug, Clone)]
pub struct AccuracyCharacteristics<F: IntegrateFloat> {
    pub local_error_order: usize,
    pub global_error_order: usize,
    pub error_constant: F,
}

// ================================================================================================
// STRATEGY AND TARGET TYPES
// ================================================================================================

/// Target performance metrics
#[derive(Debug, Clone, Default)]
pub struct TargetMetrics {
    pub min_throughput: f64,
    pub max_memory_usage: usize,
    pub max_execution_time: Duration,
    pub min_accuracy: f64,
}

/// Adaptation triggers
#[derive(Debug, Clone, Default)]
pub struct AdaptationTriggers {
    pub performance_degradation_threshold: f64,
    pub memory_pressure_threshold: f64,
    pub error_increase_threshold: f64,
    pub timeout_threshold: Duration,
}

/// Optimization objectives
#[derive(Debug, Clone)]
pub struct OptimizationObjectives<F: IntegrateFloat> {
    pub primary_objective: String,
    pub weight_performance: F,
    pub weight_accuracy: F,
    pub weight_memory: F,
}

/// Performance constraints
#[derive(Debug, Clone, Default)]
pub struct PerformanceConstraints {
    pub max_memory: usize,
    pub max_execution_time: Duration,
    pub min_accuracy: f64,
    pub power_budget: f64,
}

// ================================================================================================
// OPTIMIZATION PARAMETER TYPES
// ================================================================================================

/// Parameter space definition
#[derive(Debug, Clone)]
pub struct ParameterSpace<F: IntegrateFloat> {
    pub continuous_params: HashMap<String, (F, F)>,
    pub discrete_params: HashMap<String, Vec<String>>,
    pub categorical_params: HashMap<String, Vec<String>>,
}

/// Objective function for optimization
#[derive(Debug, Clone)]
pub struct ObjectiveFunction<F: IntegrateFloat> {
    pub function_type: String,
    pub minimize: bool,
    pub constraints: Vec<String>,
    pub phantom: std::marker::PhantomData<F>,
}

/// Optimal parameters result
#[derive(Debug, Clone)]
pub struct OptimalParameters<F: IntegrateFloat> {
    pub parameters: HashMap<String, F>,
    pub objective_value: f64,
    pub constraint_violations: Vec<String>,
    pub optimization_time: Duration,
}

/// Optimal configuration result
#[derive(Debug, Clone)]
pub struct OptimalConfiguration<F: IntegrateFloat> {
    pub algorithm: String,
    pub parameters: HashMap<String, F>,
    pub expected_performance: f64,
    pub confidence: f64,
}

// ================================================================================================
// PROBLEM AND RESOURCE TYPES
// ================================================================================================

/// Problem characteristics
#[derive(Debug, Clone, Default)]
pub struct ProblemCharacteristics {
    pub problem_size: usize,
    pub sparsity: f64,
    pub stiffness: f64,
    pub nonlinearity: f64,
    pub problem_type: String,
}

/// Resource allocation specification
#[derive(Debug, Clone, Default)]
pub struct ResourceAllocation {
    pub cpu_cores: usize,
    pub memory_gb: f64,
    pub gpu_count: usize,
    pub network_bandwidth: f64,
}

/// Performance target specification
#[derive(Debug, Clone, Default)]
pub struct PerformanceTarget {
    pub target_throughput: f64,
    pub target_latency: Duration,
    pub target_accuracy: f64,
    pub resource_budget: ResourceAllocation,
}

/// Resource reallocation plan
#[derive(Debug, Clone, Default)]
pub struct ResourceReallocationPlan {
    pub cpu_reallocation: HashMap<usize, f64>,
    pub memory_reallocation: HashMap<String, usize>,
    pub gpu_reallocation: HashMap<usize, f64>,
    pub estimated_improvement: f64,
}

/// Workload prediction
#[derive(Debug, Clone, Default)]
pub struct WorkloadPrediction {
    pub predicted_load: f64,
    pub load_variance: f64,
    pub time_horizon: Duration,
    pub confidence: f64,
}

/// Predictive optimization plan
#[derive(Debug, Clone, Default)]
pub struct PredictiveOptimizationPlan<F: IntegrateFloat> {
    pub optimized_parameters: HashMap<String, F>,
    pub predicted_performance: f64,
    pub adaptation_schedule: Vec<(Duration, String)>,
    pub confidence: f64,
}

// ================================================================================================
// ANALYSIS TYPES
// ================================================================================================

/// Utilization analysis
#[derive(Debug, Clone, Default)]
pub struct UtilizationAnalysis {
    pub cpu_utilization: f64,
    pub memory_utilization: f64,
    pub gpu_utilization: f64,
    pub network_utilization: f64,
    pub bottlenecks: Vec<String>,
}

/// Resource bottleneck
#[derive(Debug, Clone, Default)]
pub struct ResourceBottleneck {
    pub resource_type: String,
    pub severity: f64,
    pub impact: String,
    pub recommended_action: String,
}

/// Performance prediction
#[derive(Debug, Clone, Default)]
pub struct PerformancePrediction {
    pub metric_name: String,
    pub predicted_value: f64,
    pub confidence_interval: (f64, f64),
    pub prediction_horizon: Duration,
}

// ================================================================================================
// DEFAULT IMPLEMENTATIONS
// ================================================================================================

impl Default for RecoveryEvent {
    fn default() -> Self {
        Self {
            timestamp: Instant::now(),
            event_type: String::new(),
            success: false,
        }
    }
}

impl<F: IntegrateFloat> Default for AlgorithmRegistry<F> {
    fn default() -> Self {
        Self {
            available_algorithms: HashMap::new(),
            performance_models: HashMap::new(),
            selection_history: Vec::new(),
        }
    }
}

impl<F: IntegrateFloat> Default for AlgorithmPerformanceModel<F> {
    fn default() -> Self {
        Self {
            model_type: "linear".to_string(),
            parameters: HashMap::new(),
            accuracy: 0.8,
            last_updated: Instant::now(),
        }
    }
}

impl<F: IntegrateFloat> Default for AccuracyCharacteristics<F> {
    fn default() -> Self {
        Self {
            local_error_order: 4,
            global_error_order: 4,
            error_constant: F::from(1e-6).unwrap_or(F::zero()),
        }
    }
}

impl<F: IntegrateFloat> Default for OptimizationObjectives<F> {
    fn default() -> Self {
        Self {
            primary_objective: "balanced".to_string(),
            weight_performance: F::from(0.4).unwrap_or(F::zero()),
            weight_accuracy: F::from(0.4).unwrap_or(F::zero()),
            weight_memory: F::from(0.2).unwrap_or(F::zero()),
        }
    }
}

impl<F: IntegrateFloat> Default for ModelRegistry<F> {
    fn default() -> Self {
        Self {
            models: HashMap::new(),
            default_model: AlgorithmPerformanceModel::default(),
        }
    }
}

impl<F: IntegrateFloat> Default for FeatureEngineering<F> {
    fn default() -> Self {
        Self {
            feature_extractors: Vec::new(),
            normalization_params: HashMap::new(),
            feature_importance: HashMap::new(),
        }
    }
}

impl<F: IntegrateFloat> Default for ModelTrainer<F> {
    fn default() -> Self {
        Self {
            training_algorithm: String::from("gradient_descent"),
            hyperparameters: HashMap::new(),
            cross_validation_folds: 5,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<F: IntegrateFloat> Default for ReinforcementLearningAgent<F> {
    fn default() -> Self {
        Self {
            agent_type: String::from("q_learning"),
            learning_rate: 0.01,
            exploration_rate: 0.1,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<F: IntegrateFloat> Default for BayesianOptimizer<F> {
    fn default() -> Self {
        Self {
            acquisition_function: String::from("expected_improvement"),
            kernel_type: String::from("rbf"),
            num_iterations: 100,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<F: IntegrateFloat> Default for NeuralArchitectureSearch<F> {
    fn default() -> Self {
        Self {
            search_strategy: String::from("evolutionary"),
            architecture_space: String::from("feedforward"),
            evaluation_budget: 1000,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<F: IntegrateFloat> Default for HyperparameterOptimizer<F> {
    fn default() -> Self {
        Self {
            optimization_algorithm: String::from("random_search"),
            search_space: HashMap::new(),
            max_evaluations: 100,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<F: IntegrateFloat + Default> Default for AdaptationRules<F> {
    fn default() -> Self {
        Self {
            rules: Vec::new(),
            rule_weights: HashMap::new(),
            activation_thresholds: HashMap::new(),
        }
    }
}

impl<F: IntegrateFloat> Default for ConfigurationSpaceExplorer<F> {
    fn default() -> Self {
        Self {
            exploration_strategy: String::from("grid_search"),
            space_dimensions: 10,
            explored_configurations: Vec::new(),
            phantom: std::marker::PhantomData,
        }
    }
}

impl<F: IntegrateFloat + Default> Default for ConstraintSatisfactionEngine<F> {
    fn default() -> Self {
        Self {
            constraint_solver: String::from("backtracking"),
            constraints: Vec::new(),
            satisfaction_tolerance: F::default(),
        }
    }
}

impl<F: IntegrateFloat> Default for MultiObjectiveOptimizer<F> {
    fn default() -> Self {
        Self {
            algorithm: String::from("nsga2"),
            pareto_front_size: 100,
            objectives: Vec::new(),
            phantom: std::marker::PhantomData,
        }
    }
}

impl<F: IntegrateFloat> Default for ParameterSpace<F> {
    fn default() -> Self {
        Self {
            continuous_params: HashMap::new(),
            discrete_params: HashMap::new(),
            categorical_params: HashMap::new(),
        }
    }
}

impl<F: IntegrateFloat> Default for ObjectiveFunction<F> {
    fn default() -> Self {
        Self {
            function_type: "performance".to_string(),
            minimize: false, // Maximize performance by default
            constraints: Vec::new(),
            phantom: std::marker::PhantomData,
        }
    }
}

impl<F: IntegrateFloat> Default for OptimalParameters<F> {
    fn default() -> Self {
        Self {
            parameters: HashMap::new(),
            objective_value: 0.0,
            constraint_violations: Vec::new(),
            optimization_time: Duration::from_secs(0),
        }
    }
}

impl<F: IntegrateFloat> Default for OptimalConfiguration<F> {
    fn default() -> Self {
        Self {
            algorithm: String::new(),
            parameters: HashMap::new(),
            expected_performance: 0.0,
            confidence: 0.0,
        }
    }
}

impl<F: IntegrateFloat> Default for AlgorithmRecommendation<F> {
    fn default() -> Self {
        Self {
            algorithm: String::new(),
            reason: String::new(),
            expected_improvement: 0.0,
            parameters: HashMap::new(),
        }
    }
}

impl<F: IntegrateFloat + Default> Default for ParameterAdjustment<F> {
    fn default() -> Self {
        Self {
            parameter_name: String::new(),
            current_value: F::default(),
            recommended_value: F::default(),
            adjustment_reason: String::new(),
        }
    }
}

// ================================================================================================
// BASIC IMPLEMENTATIONS
// ================================================================================================

impl PerformanceMetrics {
    /// Create a new PerformanceMetrics instance
    pub fn new(
        timestamp: Instant,
        step_time: Duration,
        throughput: f64,
        memory_usage: usize,
        cpu_utilization: f64,
        gpu_utilization: f64,
        cache_hit_rate: f64,
        network_bandwidth: f64,
        error_accuracy: f64,
        convergence_rate: f64,
    ) -> Self {
        Self {
            timestamp,
            step_time,
            throughput,
            memory_usage,
            cpu_utilization,
            gpu_utilization,
            cache_hit_rate,
            network_bandwidth,
            error_accuracy,
            convergence_rate,
        }
    }
}

impl AnomalyAnalysis {
    pub fn normal() -> Self {
        AnomalyAnalysis {
            severity: AnomalySeverity::Low,
            root_cause: "No anomaly detected".to_string(),
            affected_components: Vec::new(),
            recommended_actions: Vec::new(),
        }
    }

    pub fn anomalous(count: usize) -> Self {
        AnomalyAnalysis {
            severity: if count > 5 {
                AnomalySeverity::High
            } else {
                AnomalySeverity::Medium
            },
            root_cause: format!("{} anomalies detected", count),
            affected_components: Vec::new(),
            recommended_actions: vec!["Investigate performance metrics".to_string()],
        }
    }
}

impl MetricsCollector {
    /// Create a new `MetricsCollector` with sensible defaults.
    ///
    /// Initialises the collector with a 100 ms sampling interval and three
    /// built-in collector channels ("cpu", "memory", "gpu").
    pub fn new() -> IntegrateResult<Self> {
        Ok(Self {
            collection_interval: Duration::from_millis(100),
            active_collectors: vec!["cpu".to_string(), "memory".to_string(), "gpu".to_string()],
            metric_buffer: Vec::new(),
        })
    }
}

impl PerformanceHistory {
    /// Create a new `PerformanceHistory` with capacity pre-allocated.
    pub fn new() -> IntegrateResult<Self> {
        Ok(Self {
            metrics_history: VecDeque::with_capacity(1000),
            max_history_size: 1000,
            aggregated_stats: HashMap::new(),
        })
    }
}

impl SystemResourceMonitor {
    /// Create a new `SystemResourceMonitor`.
    ///
    /// Detects the number of logical CPU cores available to the process (via
    /// `std::thread::available_parallelism`) and registers one `CpuMonitor`
    /// per core so subsequent samples are correctly dimensioned.
    pub fn new() -> IntegrateResult<Self> {
        let core_count = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let cpu_monitors = (0..core_count)
            .map(|id| CpuMonitor {
                core_id: id,
                ..Default::default()
            })
            .collect();
        Ok(Self {
            cpu_monitors,
            memory_monitor: MemoryMonitor::default(),
            disk_monitor: DiskMonitor::default(),
        })
    }
}

impl NetworkPerformanceMonitor {
    /// Create a new `NetworkPerformanceMonitor` with per-component defaults.
    ///
    /// The latency monitor is initialised with a 1 ms baseline average/p99
    /// so that the zero-duration sentinel is distinguishable from a real sample.
    pub fn new() -> IntegrateResult<Self> {
        Ok(Self {
            bandwidth_monitor: BandwidthMonitor::default(),
            latency_monitor: LatencyMonitor {
                avg_latency: Duration::from_millis(1),
                p99_latency: Duration::from_millis(1),
                jitter: Duration::ZERO,
            },
            packet_loss_monitor: PacketLossMonitor::default(),
        })
    }
}

impl CpuResourceManager {
    /// Create a new `CpuResourceManager`.
    ///
    /// Pre-allocates one entry in `cpu_allocation` for each logical core so
    /// callers can update utilisation per-core without additional inserts.
    pub fn new() -> IntegrateResult<Self> {
        let core_count = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let cpu_allocation = (0..core_count).map(|id| (id, 0.0_f64)).collect();
        Ok(Self {
            cpu_allocation,
            thermal_state: ThermalState::default(),
            frequency_scaling: FrequencyScaling {
                current_frequency: 1.0,
                target_frequency: 1.0,
                scaling_governor: "performance".to_string(),
            },
        })
    }
}

impl MemoryResourceManager {
    /// Create a new `MemoryResourceManager`.
    ///
    /// Starts with a single general-purpose pool and the `BestFit` strategy
    /// to minimise fragmentation under diverse workload patterns.
    pub fn new() -> IntegrateResult<Self> {
        Ok(Self {
            memory_pools: vec![MemoryPool {
                pool_id: 0,
                size: 0,
                allocation_type: "general".to_string(),
            }],
            allocation_strategy: AllocationStrategy {
                strategy_type: "best_fit".to_string(),
                parameters: HashMap::new(),
            },
            gc_policy: GarbageCollectionPolicy {
                gc_type: "threshold".to_string(),
                threshold: 0.8,
            },
        })
    }
}

impl GpuResourceManager {
    /// Create a new `GpuResourceManager`.
    ///
    /// Registers a single placeholder GPU device (device 0).  When real GPU
    /// discovery is available this constructor should be updated accordingly.
    pub fn new() -> IntegrateResult<Self> {
        let default_device = GpuDevice {
            device_id: 0,
            name: "default_gpu".to_string(),
            memory_size: 0,
            compute_units: 0,
        };
        let mut memory_allocation = HashMap::new();
        memory_allocation.insert(0usize, 0usize);
        let mut compute_allocation = HashMap::new();
        compute_allocation.insert(0usize, 0.0_f64);
        Ok(Self {
            gpu_devices: vec![default_device],
            memory_allocation,
            compute_allocation,
        })
    }
}

impl NetworkResourceManager {
    /// Create a new `NetworkResourceManager` with a 1 Gbps default bandwidth
    /// budget and a connection pool sized for 128 concurrent connections.
    pub fn new() -> IntegrateResult<Self> {
        Ok(Self {
            bandwidth_allocation: BandwidthAllocation {
                total_bandwidth: 1_000_000_000.0, // 1 Gbps
                allocated_bandwidth: HashMap::new(),
            },
            connection_pool: ConnectionPool {
                max_connections: 128,
                active_connections: 0,
                connection_timeout: Duration::from_secs(30),
            },
            load_balancing: NetworkLoadBalancing {
                algorithm: "round_robin".to_string(),
                weights: HashMap::new(),
            },
        })
    }
}

impl LoadBalancer {
    /// Create a new `LoadBalancer` using round-robin as the default strategy.
    pub fn new() -> IntegrateResult<Self> {
        Ok(Self {
            balancing_strategy: "round_robin".to_string(),
            node_weights: HashMap::new(),
            current_load: HashMap::new(),
        })
    }
}

impl StatisticalAnomalyDetector {
    /// Create a new `StatisticalAnomalyDetector` with standard algorithm
    /// defaults.
    ///
    /// Registers z-score and IQR detectors and sets a 95 % confidence interval
    /// with per-metric thresholds calibrated for typical performance workloads.
    pub fn new() -> IntegrateResult<Self> {
        let mut thresholds = HashMap::new();
        thresholds.insert("cpu_utilization".to_string(), 0.95);
        thresholds.insert("memory_usage".to_string(), 0.90);
        thresholds.insert("throughput".to_string(), 0.85);
        Ok(Self {
            detection_algorithms: vec!["z_score".to_string(), "iqr".to_string()],
            thresholds,
            confidence_interval: 0.95,
        })
    }
}

impl MLAnomalyDetector {
    /// Create a new `MLAnomalyDetector` using an isolation-forest base model.
    ///
    /// Starts with an empty training set; call the training routine before
    /// running inference.  Detection threshold is set conservatively at 0.8
    /// to minimise false positives during warm-up.
    pub fn new() -> IntegrateResult<Self> {
        Ok(Self {
            model_type: "isolation_forest".to_string(),
            training_data: Vec::new(),
            detection_threshold: 0.8,
        })
    }
}

impl SystemHealthMonitor {
    /// Create a new `SystemHealthMonitor` with a fully-healthy initial state.
    ///
    /// Sets `health_score` to `1.0` and registers default alert thresholds for
    /// the four core resource dimensions so alerts fire correctly even before
    /// the first monitoring cycle.
    pub fn new() -> IntegrateResult<Self> {
        let mut alert_thresholds = HashMap::new();
        alert_thresholds.insert("cpu_utilization".to_string(), 0.90);
        alert_thresholds.insert("memory_utilization".to_string(), 0.85);
        alert_thresholds.insert("gpu_utilization".to_string(), 0.95);
        alert_thresholds.insert("network_utilization".to_string(), 0.80);
        Ok(Self {
            health_score: 1.0,
            critical_components: Vec::new(),
            alert_thresholds,
        })
    }
}

impl AutomaticRecoveryManager {
    /// Create a new `AutomaticRecoveryManager` in the *enabled* state.
    ///
    /// Registers default recovery strategies for CPU, memory, and GPU
    /// resource over-use so the system can self-heal without additional
    /// configuration.
    pub fn new() -> IntegrateResult<Self> {
        let mut recovery_strategies = HashMap::new();
        recovery_strategies.insert(
            "cpu_overload".to_string(),
            RecoveryStrategy {
                strategy_type: "throttle".to_string(),
                steps: vec![
                    "reduce_parallelism".to_string(),
                    "defer_background_tasks".to_string(),
                ],
                timeout: Duration::from_secs(30),
            },
        );
        recovery_strategies.insert(
            "memory_pressure".to_string(),
            RecoveryStrategy {
                strategy_type: "evict".to_string(),
                steps: vec!["trigger_gc".to_string(), "release_caches".to_string()],
                timeout: Duration::from_secs(60),
            },
        );
        recovery_strategies.insert(
            "gpu_failure".to_string(),
            RecoveryStrategy {
                strategy_type: "fallback".to_string(),
                steps: vec!["switch_to_cpu".to_string()],
                timeout: Duration::from_secs(10),
            },
        );
        Ok(Self {
            recovery_strategies,
            recovery_history: Vec::new(),
            enabled: true,
        })
    }
}

// ================================================================================================
// CONSTRUCTOR TESTS
// ================================================================================================
#[cfg(test)]
mod constructor_tests {
    use super::*;

    // --- monitoring constructors ---

    #[test]
    fn test_metrics_collector_new_initialises_collectors() {
        let mc = MetricsCollector::new().expect("constructor must succeed");
        assert!(
            !mc.active_collectors.is_empty(),
            "active_collectors must be pre-populated"
        );
        assert_eq!(mc.collection_interval, Duration::from_millis(100));
    }

    #[test]
    fn test_system_resource_monitor_new_detects_cores() {
        let mon = SystemResourceMonitor::new().expect("constructor must succeed");
        assert!(
            !mon.cpu_monitors.is_empty(),
            "must register at least one CPU monitor"
        );
        // Each CpuMonitor core_id must match its index
        for (i, m) in mon.cpu_monitors.iter().enumerate() {
            assert_eq!(m.core_id, i);
        }
    }

    #[test]
    fn test_network_monitor_new_latency_not_zero() {
        let nm = NetworkPerformanceMonitor::new().expect("constructor must succeed");
        assert_ne!(
            nm.latency_monitor.avg_latency,
            Duration::ZERO,
            "baseline latency must be set to 1 ms sentinel"
        );
    }

    // --- resource manager constructors ---

    #[test]
    fn test_cpu_resource_manager_new_allocates_per_core_entries() {
        let mgr = CpuResourceManager::new().expect("constructor must succeed");
        assert!(
            !mgr.cpu_allocation.is_empty(),
            "must pre-allocate cpu_allocation entries"
        );
        assert_eq!(mgr.frequency_scaling.scaling_governor, "performance");
    }

    #[test]
    fn test_memory_resource_manager_new_has_pool_and_strategy() {
        let mgr = MemoryResourceManager::new().expect("constructor must succeed");
        assert!(!mgr.memory_pools.is_empty());
        assert_eq!(mgr.allocation_strategy.strategy_type, "best_fit");
        assert!((mgr.gc_policy.threshold - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_gpu_resource_manager_new_registers_device_zero() {
        let mgr = GpuResourceManager::new().expect("constructor must succeed");
        assert!(!mgr.gpu_devices.is_empty());
        assert_eq!(mgr.gpu_devices[0].device_id, 0);
        assert!(mgr.memory_allocation.contains_key(&0usize));
    }

    #[test]
    fn test_network_resource_manager_new_bandwidth_and_pool() {
        let mgr = NetworkResourceManager::new().expect("constructor must succeed");
        assert!(mgr.bandwidth_allocation.total_bandwidth > 0.0);
        assert_eq!(mgr.connection_pool.max_connections, 128);
        assert_eq!(mgr.load_balancing.algorithm, "round_robin");
    }

    #[test]
    fn test_load_balancer_new_strategy_set() {
        let lb = LoadBalancer::new().expect("constructor must succeed");
        assert_eq!(lb.balancing_strategy, "round_robin");
    }

    // --- anomaly detection / health / recovery constructors ---

    #[test]
    fn test_statistical_anomaly_detector_new_has_algorithms_and_thresholds() {
        let det = StatisticalAnomalyDetector::new().expect("constructor must succeed");
        assert!(!det.detection_algorithms.is_empty());
        assert!(!det.thresholds.is_empty());
        assert!((det.confidence_interval - 0.95).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ml_anomaly_detector_new_model_type_set() {
        let det = MLAnomalyDetector::new().expect("constructor must succeed");
        assert_eq!(det.model_type, "isolation_forest");
        assert!((det.detection_threshold - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_system_health_monitor_new_starts_healthy() {
        let mon = SystemHealthMonitor::new().expect("constructor must succeed");
        assert!(
            (mon.health_score - 1.0).abs() < f64::EPSILON,
            "initial health must be 1.0"
        );
        assert!(
            !mon.alert_thresholds.is_empty(),
            "default alert thresholds must be set"
        );
    }

    #[test]
    fn test_automatic_recovery_manager_new_enabled_with_strategies() {
        let mgr = AutomaticRecoveryManager::new().expect("constructor must succeed");
        assert!(mgr.enabled, "recovery manager must be enabled on creation");
        assert!(
            !mgr.recovery_strategies.is_empty(),
            "default strategies must be registered"
        );
        assert!(mgr.recovery_strategies.contains_key("cpu_overload"));
        assert!(mgr.recovery_strategies.contains_key("memory_pressure"));
        assert!(mgr.recovery_strategies.contains_key("gpu_failure"));
    }
}

impl<F: IntegrateFloat> AlgorithmRegistry<F> {
    pub fn new() -> Self {
        Default::default()
    }
}

// ================================================================================================
// ADDITIONAL MISSING TYPES
// ================================================================================================

/// Predicted performance metrics result  
#[derive(Debug, Clone, Default)]
pub struct PredictedPerformance {
    pub expected_throughput: f64,
    pub expected_cpu_usage: f64,
    pub expected_memory_usage: usize,
    pub confidence_interval: (f64, f64),
}

/// Optimization result from ML-based optimization
#[derive(Debug, Clone, Default)]
pub struct OptimizationResult<F: IntegrateFloat> {
    pub optimized_parameters: HashMap<String, F>,
    pub improvement_factor: f64,
    pub confidence_score: f64,
}

/// Current resource utilization snapshot
#[derive(Debug, Clone, Default)]
pub struct ResourceUtilization {
    pub cpu_utilization: f64,
    pub memory_utilization: f64,
    pub gpu_utilization: f64,
    pub network_utilization: f64,
    pub storage_utilization: f64,
}

impl<F: IntegrateFloat> OptimizationRecommendations<F> {
    pub fn new() -> Self {
        OptimizationRecommendations {
            algorithm_changes: Vec::new(),
            parameter_adjustments: Vec::new(),
            resource_reallocations: Vec::new(),
            performance_predictions: PerformanceImprovement::default(),
        }
    }
}
