//! Neural Architecture Search Results and Statistics
//!
//! This module contains all result types, evaluation metrics, statistics tracking,
//! and performance measurement functionality for the Neural Architecture Search system.

use std::collections::HashMap;
use std::time::{Duration, Instant};
use num_traits::Float;
use crate::learned_optimizers::few_shot_optimizer::EvaluationMetric;

use crate::neural_architecture_search::{
    ParetoFront,
};

// Define missing types as placeholders since they're not available
#[derive(Debug, Clone)]
pub struct OptimizerArchitecture<T: Float> {
    pub components: Vec<String>,
    pub parameters: HashMap<String, T>,
}

#[derive(Debug, Clone)]
pub struct OptimizerComponent<T: Float> {
    pub name: String,
    pub parameters: HashMap<String, T>,
}

#[derive(Debug, Clone)]
pub struct ConvergenceData<T: Float> {
    pub iteration: usize,
    pub best_score: T,
    pub convergence_rate: T,
    pub stability_measure: T,
}

/// Search result for tracking individual architecture evaluations
#[derive(Debug, Clone)]
pub struct SearchResult<T: Float> {
    /// Generated architecture
    pub architecture: OptimizerArchitecture<T>,

    /// Evaluation results
    pub evaluation_results: EvaluationResults<T>,

    /// Generation number
    pub generation: usize,

    /// Search time (seconds)
    pub search_time: f64,

    /// Resource usage
    pub resource_usage: ResourceUsage<T>,

    /// Architecture encoding
    pub encoding: ArchitectureEncoding,

    /// Additional metadata
    pub metadata: SearchResultMetadata,
}

/// Evaluation results for a single architecture
#[derive(Debug, Clone)]
pub struct EvaluationResults<T: Float> {
    /// Metric scores
    pub metric_scores: HashMap<EvaluationMetric, T>,

    /// Overall score
    pub overall_score: T,

    /// Statistical confidence intervals
    pub confidence_intervals: HashMap<EvaluationMetric, (T, T)>,

    /// Evaluation time
    pub evaluation_time: Duration,

    /// Success flag
    pub success: bool,

    /// Error message (if failed)
    pub error_message: Option<String>,

    /// Cross-validation results
    pub cv_results: Option<CrossValidationResults<T>>,

    /// Benchmark-specific results
    pub benchmark_results: HashMap<String, BenchmarkResult<T>>,

    /// Performance trajectory during training
    pub training_trajectory: Vec<TrainingSnapshot<T>>,
}

/// Cross-validation results
#[derive(Debug, Clone)]
pub struct CrossValidationResults<T: Float> {
    /// Results for each fold
    pub fold_results: Vec<T>,

    /// Mean score across folds
    pub mean_score: T,

    /// Standard deviation across folds
    pub std_score: T,

    /// Confidence interval for mean
    pub confidence_interval: (T, T),

    /// Statistical significance tests
    pub statistical_tests: HashMap<String, StatisticalTestResult<T>>,
}

/// Statistical test result
#[derive(Debug, Clone)]
pub struct StatisticalTestResult<T: Float> {
    /// Test statistic value
    pub test_statistic: T,

    /// P-value
    pub p_value: T,

    /// Critical value
    pub critical_value: T,

    /// Reject null hypothesis flag
    pub reject_null: bool,

    /// Effect size
    pub effect_size: Option<T>,
}

/// Benchmark-specific evaluation result
#[derive(Debug, Clone)]
pub struct BenchmarkResult<T: Float> {
    /// Benchmark name
    pub benchmark_name: String,

    /// Primary metric score
    pub primary_score: T,

    /// All metric scores
    pub all_scores: HashMap<EvaluationMetric, T>,

    /// Evaluation time
    pub evaluation_time: Duration,

    /// Memory usage peak
    pub peak_memory_usage: usize,

    /// Success flag
    pub success: bool,

    /// Benchmark-specific metadata
    pub metadata: HashMap<String, String>,
}

/// Training snapshot during evaluation
#[derive(Debug, Clone)]
pub struct TrainingSnapshot<T: Float> {
    /// Epoch number
    pub epoch: usize,

    /// Training loss
    pub training_loss: T,

    /// Validation loss
    pub validation_loss: Option<T>,

    /// Training metrics
    pub training_metrics: HashMap<EvaluationMetric, T>,

    /// Validation metrics
    pub validation_metrics: HashMap<EvaluationMetric, T>,

    /// Learning rate
    pub learning_rate: T,

    /// Timestamp
    pub timestamp: Duration,

    /// Memory usage at this point
    pub memory_usage: usize,
}

/// Resource usage tracking
#[derive(Debug, Clone)]
pub struct ResourceUsage<T: Float> {
    /// Memory usage (GB)
    pub memory_gb: T,

    /// CPU time (seconds)
    pub cpu_time_seconds: T,

    /// GPU time (seconds)
    pub gpu_time_seconds: T,

    /// Energy consumption (kWh)
    pub energy_kwh: T,

    /// Network I/O (GB)
    pub network_io_gb: T,

    /// Disk I/O (GB)
    pub disk_io_gb: T,

    /// Peak memory usage
    pub peak_memory_gb: T,

    /// Resource efficiency score
    pub efficiency_score: T,
}

/// Architecture encoding for reproducibility and analysis
#[derive(Debug, Clone)]
pub struct ArchitectureEncoding {
    /// Encoding type
    pub encoding_type: EncodingType,

    /// Binary encoding
    pub binary_encoding: Option<Vec<u8>>,

    /// String encoding
    pub string_encoding: Option<String>,

    /// Graph encoding
    pub graph_encoding: Option<GraphEncoding>,

    /// Hash for quick comparison
    pub hash: u64,

    /// Encoding metadata
    pub metadata: EncodingMetadata,
}

/// Architecture encoding types
#[derive(Debug, Clone, Copy)]
pub enum EncodingType {
    Binary,
    String,
    Graph,
    Neural,
    Hybrid,
}

/// Graph-based architecture encoding
#[derive(Debug, Clone)]
pub struct GraphEncoding {
    /// Nodes (components)
    pub nodes: Vec<NodeEncoding>,

    /// Edges (connections)
    pub edges: Vec<EdgeEncoding>,

    /// Graph properties
    pub properties: GraphProperties,
}

/// Node encoding in graph representation
#[derive(Debug, Clone)]
pub struct NodeEncoding {
    /// Node ID
    pub id: usize,

    /// Component type
    pub component_type: String,

    /// Component parameters
    pub parameters: HashMap<String, f64>,

    /// Node properties
    pub properties: HashMap<String, String>,
}

/// Edge encoding in graph representation
#[derive(Debug, Clone)]
pub struct EdgeEncoding {
    /// Source node ID
    pub source: usize,

    /// Target node ID
    pub target: usize,

    /// Connection type
    pub connection_type: String,

    /// Edge weight
    pub weight: f64,

    /// Edge properties
    pub properties: HashMap<String, String>,
}

/// Graph properties
#[derive(Debug, Clone)]
pub struct GraphProperties {
    /// Number of nodes
    pub num_nodes: usize,

    /// Number of edges
    pub num_edges: usize,

    /// Graph density
    pub density: f64,

    /// Average degree
    pub average_degree: f64,

    /// Is directed
    pub is_directed: bool,

    /// Has cycles
    pub has_cycles: bool,
}

/// Encoding metadata
#[derive(Debug, Clone)]
pub struct EncodingMetadata {
    /// Encoding version
    pub version: String,

    /// Creation timestamp
    pub created_at: Instant,

    /// Encoding size (bytes)
    pub size_bytes: usize,

    /// Compression ratio
    pub compression_ratio: Option<f64>,

    /// Encoding quality score
    pub quality_score: Option<f64>,
}

/// Search result metadata
#[derive(Debug, Clone)]
pub struct SearchResultMetadata {
    /// Search strategy used
    pub search_strategy: String,

    /// Controller type used
    pub controller_type: String,

    /// Evaluation method
    pub evaluation_method: String,

    /// Search phase (for progressive search)
    pub search_phase: Option<String>,

    /// Parent architectures (for evolutionary methods)
    pub parent_architectures: Vec<u64>,

    /// Mutation/crossover information
    pub operation_info: Option<OperationInfo>,

    /// Additional tags
    pub tags: HashMap<String, String>,
}

/// Operation information for evolutionary/genetic methods
#[derive(Debug, Clone)]
pub struct OperationInfo {
    /// Operation type (mutation, crossover, etc.)
    pub operation_type: String,

    /// Operation parameters
    pub parameters: HashMap<String, f64>,

    /// Success flag
    pub success: bool,

    /// Operation time
    pub operation_time: Duration,
}

/// Search statistics tracking
#[derive(Debug, Clone)]
pub struct SearchStatistics<T: Float> {
    /// Total architectures evaluated
    pub total_architectures_evaluated: usize,

    /// Successful evaluations
    pub successful_evaluations: usize,

    /// Failed evaluations
    pub failed_evaluations: usize,

    /// Best score found
    pub best_score: Option<T>,

    /// Best architecture hash
    pub best_architecture_hash: Option<u64>,

    /// Average evaluation time
    pub average_evaluation_time: Duration,

    /// Total search time
    pub total_search_time: Duration,

    /// Score improvement over time
    pub score_history: Vec<T>,

    /// Resource utilization statistics
    pub resource_stats: ResourceStatistics<T>,

    /// Convergence metrics
    pub convergence_metrics: ConvergenceMetrics<T>,

    /// Diversity metrics
    pub diversity_metrics: DiversityMetrics<T>,

    /// Search efficiency metrics
    pub efficiency_metrics: EfficiencyMetrics<T>,
}

/// Resource utilization statistics
#[derive(Debug, Clone)]
pub struct ResourceStatistics<T: Float> {
    /// Total memory used (GB)
    pub total_memory_gb: T,

    /// Total CPU time (hours)
    pub total_cpu_hours: T,

    /// Total GPU time (hours)
    pub total_gpu_hours: T,

    /// Total energy consumed (kWh)
    pub total_energy_kwh: T,

    /// Peak memory usage (GB)
    pub peak_memory_gb: T,

    /// Average resource utilization
    pub average_utilization: T,

    /// Resource efficiency score
    pub efficiency_score: T,
}

/// Convergence metrics
#[derive(Debug, Clone)]
pub struct ConvergenceMetrics<T: Float> {
    /// Generations to convergence
    pub generations_to_convergence: Option<usize>,

    /// Convergence rate
    pub convergence_rate: Option<T>,

    /// Plateau detection
    pub plateau_generations: usize,

    /// Score variance over last N generations
    pub recent_score_variance: T,

    /// Improvement rate (scores per generation)
    pub improvement_rate: T,

    /// Stagnation detection
    pub is_stagnating: bool,
}

/// Diversity metrics
#[derive(Debug, Clone)]
pub struct DiversityMetrics<T: Float> {
    /// Population diversity score
    pub population_diversity: T,

    /// Architecture uniqueness ratio
    pub uniqueness_ratio: T,

    /// Genotypic diversity
    pub genotypic_diversity: T,

    /// Phenotypic diversity
    pub phenotypic_diversity: T,

    /// Diversity trends over time
    pub diversity_history: Vec<T>,

    /// Diversity maintenance score
    pub diversity_maintenance: T,
}

/// Search efficiency metrics
#[derive(Debug, Clone)]
pub struct EfficiencyMetrics<T: Float> {
    /// Evaluations per unit improvement
    pub evaluations_per_improvement: T,

    /// Time to find best solution
    pub time_to_best: Duration,

    /// Resource cost per unit improvement
    pub resource_cost_per_improvement: T,

    /// Search efficiency score
    pub efficiency_score: T,

    /// Pareto efficiency (for multi-objective)
    pub pareto_efficiency: Option<T>,

    /// Hypervolume indicator (for multi-objective)
    pub hypervolume: Option<T>,
}

/// Final search results
#[derive(Debug, Clone)]
pub struct SearchResults<T: Float> {
    /// Best architectures found
    pub best_architectures: Vec<SearchResult<T>>,

    /// Pareto front (for multi-objective optimization)
    pub pareto_front: Option<ParetoFront<T>>,

    /// Complete search history
    pub search_history: Vec<SearchResult<T>>,

    /// Search statistics
    pub search_statistics: SearchStatistics<T>,

    /// Final evaluation summary
    pub evaluation_summary: EvaluationSummary<T>,

    /// Search configuration used
    pub search_configuration: SearchConfigSummary,

    /// Recommendations for future searches
    pub recommendations: Vec<SearchRecommendation>,
}

/// Evaluation summary
#[derive(Debug, Clone)]
pub struct EvaluationSummary<T: Float> {
    /// Total evaluations performed
    pub total_evaluations: usize,

    /// Evaluation success rate
    pub success_rate: T,

    /// Best score achieved
    pub best_score: T,

    /// Score distribution statistics
    pub score_statistics: ScoreStatistics<T>,

    /// Benchmark performance summary
    pub benchmark_summary: HashMap<String, BenchmarkSummary<T>>,

    /// Resource usage summary
    pub resource_summary: ResourceSummary<T>,
}

/// Score distribution statistics
#[derive(Debug, Clone)]
pub struct ScoreStatistics<T: Float> {
    /// Mean score
    pub mean: T,

    /// Median score
    pub median: T,

    /// Standard deviation
    pub std_dev: T,

    /// Minimum score
    pub min: T,

    /// Maximum score
    pub max: T,

    /// Percentiles (25th, 75th, 90th, 95th, 99th)
    pub percentiles: HashMap<usize, T>,

    /// Score distribution histogram
    pub histogram: Vec<(T, usize)>,
}

/// Benchmark performance summary
#[derive(Debug, Clone)]
pub struct BenchmarkSummary<T: Float> {
    /// Benchmark name
    pub benchmark_name: String,

    /// Best score on this benchmark
    pub best_score: T,

    /// Average score on this benchmark
    pub average_score: T,

    /// Number of successful evaluations
    pub successful_evaluations: usize,

    /// Evaluation time statistics
    pub time_statistics: TimeStatistics,

    /// Score improvement over time
    pub improvement_trajectory: Vec<T>,
}

/// Time statistics
#[derive(Debug, Clone)]
pub struct TimeStatistics {
    /// Mean time
    pub mean: Duration,

    /// Median time
    pub median: Duration,

    /// Standard deviation
    pub std_dev: Duration,

    /// Minimum time
    pub min: Duration,

    /// Maximum time
    pub max: Duration,
}

/// Resource usage summary
#[derive(Debug, Clone)]
pub struct ResourceSummary<T: Float> {
    /// Total resources consumed
    pub total_consumption: ResourceUsage<T>,

    /// Average resource usage per evaluation
    pub average_per_evaluation: ResourceUsage<T>,

    /// Peak resource usage
    pub peak_usage: ResourceUsage<T>,

    /// Resource efficiency metrics
    pub efficiency_metrics: ResourceEfficiencyMetrics<T>,
}

/// Resource efficiency metrics
#[derive(Debug, Clone)]
pub struct ResourceEfficiencyMetrics<T: Float> {
    /// Memory efficiency (useful work per GB)
    pub memory_efficiency: T,

    /// CPU efficiency (useful work per CPU hour)
    pub cpu_efficiency: T,

    /// GPU efficiency (useful work per GPU hour)
    pub gpu_efficiency: T,

    /// Energy efficiency (useful work per kWh)
    pub energy_efficiency: T,

    /// Overall resource efficiency score
    pub overall_efficiency: T,
}

/// Search configuration summary for results
#[derive(Debug, Clone)]
pub struct SearchConfigSummary {
    /// Search strategy used
    pub search_strategy: String,

    /// Population size
    pub population_size: usize,

    /// Search budget
    pub search_budget: usize,

    /// Evaluation configuration summary
    pub evaluation_config: String,

    /// Multi-objective configuration
    pub multi_objective_config: Option<String>,

    /// Resource constraints summary
    pub resource_constraints: String,

    /// Key hyperparameters
    pub key_hyperparameters: HashMap<String, String>,
}

/// Search recommendations for future improvements
#[derive(Debug, Clone)]
pub struct SearchRecommendation {
    /// Recommendation type
    pub recommendation_type: RecommendationType,

    /// Recommendation description
    pub description: String,

    /// Priority level
    pub priority: RecommendationPriority,

    /// Expected improvement
    pub expected_improvement: Option<f64>,

    /// Implementation effort
    pub implementation_effort: ImplementationEffort,

    /// Supporting evidence
    pub evidence: Vec<String>,
}

/// Types of search recommendations
#[derive(Debug, Clone, Copy)]
pub enum RecommendationType {
    SearchStrategyChange,
    PopulationSizeAdjustment,
    BudgetReallocation,
    EvaluationMethodImprovement,
    ResourceOptimization,
    HyperparameterTuning,
    SearchSpaceModification,
    DiversityImprovement,
    ConvergenceAcceleration,
}

/// Recommendation priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RecommendationPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Implementation effort estimates
#[derive(Debug, Clone, Copy)]
pub enum ImplementationEffort {
    Minimal,
    Low,
    Medium,
    High,
    Extensive,
}

// Default implementations
impl<T: Float> Default for SearchStatistics<T> {
    fn default() -> Self {
        Self {
            total_architectures_evaluated: 0,
            successful_evaluations: 0,
            failed_evaluations: 0,
            best_score: None,
            best_architecture_hash: None,
            average_evaluation_time: Duration::from_secs(0),
            total_search_time: Duration::from_secs(0),
            score_history: Vec::new(),
            resource_stats: ResourceStatistics::default(),
            convergence_metrics: ConvergenceMetrics::default(),
            diversity_metrics: DiversityMetrics::default(),
            efficiency_metrics: EfficiencyMetrics::default(),
        }
    }
}

impl<T: Float> Default for ResourceStatistics<T> {
    fn default() -> Self {
        Self {
            total_memory_gb: T::zero(),
            total_cpu_hours: T::zero(),
            total_gpu_hours: T::zero(),
            total_energy_kwh: T::zero(),
            peak_memory_gb: T::zero(),
            average_utilization: T::zero(),
            efficiency_score: T::zero(),
        }
    }
}

impl<T: Float> Default for ConvergenceMetrics<T> {
    fn default() -> Self {
        Self {
            generations_to_convergence: None,
            convergence_rate: None,
            plateau_generations: 0,
            recent_score_variance: T::zero(),
            improvement_rate: T::zero(),
            is_stagnating: false,
        }
    }
}

impl<T: Float> Default for DiversityMetrics<T> {
    fn default() -> Self {
        Self {
            population_diversity: T::one(),
            uniqueness_ratio: T::one(),
            genotypic_diversity: T::one(),
            phenotypic_diversity: T::one(),
            diversity_history: Vec::new(),
            diversity_maintenance: T::one(),
        }
    }
}

impl<T: Float> Default for EfficiencyMetrics<T> {
    fn default() -> Self {
        Self {
            evaluations_per_improvement: T::one(),
            time_to_best: Duration::from_secs(0),
            resource_cost_per_improvement: T::zero(),
            efficiency_score: T::zero(),
            pareto_efficiency: None,
            hypervolume: None,
        }
    }
}

impl<T: Float> Default for ResourceUsage<T> {
    fn default() -> Self {
        Self {
            memory_gb: T::zero(),
            cpu_time_seconds: T::zero(),
            gpu_time_seconds: T::zero(),
            energy_kwh: T::zero(),
            network_io_gb: T::zero(),
            disk_io_gb: T::zero(),
            peak_memory_gb: T::zero(),
            efficiency_score: T::zero(),
        }
    }
}

impl Default for ArchitectureEncoding {
    fn default() -> Self {
        Self {
            encoding_type: EncodingType::String,
            binary_encoding: None,
            string_encoding: None,
            graph_encoding: None,
            hash: 0,
            metadata: EncodingMetadata::default(),
        }
    }
}

impl Default for EncodingMetadata {
    fn default() -> Self {
        Self {
            version: "1.0".to_string(),
            created_at: Instant::now(),
            size_bytes: 0,
            compression_ratio: None,
            quality_score: None,
        }
    }
}

impl Default for SearchResultMetadata {
    fn default() -> Self {
        Self {
            search_strategy: "Unknown".to_string(),
            controller_type: "Unknown".to_string(),
            evaluation_method: "Unknown".to_string(),
            search_phase: None,
            parent_architectures: Vec::new(),
            operation_info: None,
            tags: HashMap::new(),
        }
    }
}

impl Default for GraphProperties {
    fn default() -> Self {
        Self {
            num_nodes: 0,
            num_edges: 0,
            density: 0.0,
            average_degree: 0.0,
            is_directed: true,
            has_cycles: false,
        }
    }
}

impl Default for TimeStatistics {
    fn default() -> Self {
        Self {
            mean: Duration::from_secs(0),
            median: Duration::from_secs(0),
            std_dev: Duration::from_secs(0),
            min: Duration::from_secs(0),
            max: Duration::from_secs(0),
        }
    }
}

/// Utility functions for results analysis
impl<T: Float> SearchStatistics<T> {
    /// Calculate search efficiency score
    pub fn calculate_efficiency_score(&self) -> T {
        if self.total_architectures_evaluated == 0 {
            return T::zero();
        }

        let success_rate = num_traits::cast::cast(self.successful_evaluations).unwrap_or_else(|| T::zero()) /
                          num_traits::cast::cast(self.total_architectures_evaluated).unwrap_or_else(|| T::zero());

        let time_efficiency = if self.total_search_time.as_secs() > 0 {
            num_traits::cast::cast(self.successful_evaluations).unwrap_or_else(|| T::zero()) /
            T::from(self.total_search_time.as_secs()).unwrap()
        } else {
            T::zero()
        };

        (success_rate + time_efficiency) / num_traits::cast::cast(2.0).unwrap_or_else(|| T::zero())
    }

    /// Check if search has converged
    pub fn has_converged(&self, patience: usize, min_improvement: T) -> bool {
        if self.score_history.len() < patience {
            return false;
        }

        let recent_scores = &self.score_history[self.score_history.len() - patience..];
        let max_score = recent_scores.iter().fold(T::neg_infinity(), |a, &b| a.max(b));
        let min_score = recent_scores.iter().fold(T::infinity(), |a, &b| a.min(b));

        max_score - min_score < min_improvement
    }

    /// Get improvement trend
    pub fn get_improvement_trend(&self, window_size: usize) -> Option<T> {
        if self.score_history.len() < window_size * 2 {
            return None;
        }

        let recent_scores = &self.score_history[self.score_history.len() - window_size..];
        let older_scores = &self.score_history[self.score_history.len() - window_size * 2..self.score_history.len() - window_size];

        let recent_mean = recent_scores.iter().fold(T::zero(), |a, &b| a + b) / num_traits::cast::cast(window_size).unwrap_or_else(|| T::zero());
        let older_mean = older_scores.iter().fold(T::zero(), |a, &b| a + b) / num_traits::cast::cast(window_size).unwrap_or_else(|| T::zero());

        Some(recent_mean - older_mean)
    }
}

impl<T: Float> ResourceUsage<T> {
    /// Calculate total resource cost
    pub fn total_cost(&self, costs: &ResourceCosts<T>) -> T {
        self.memory_gb * costs.memory_cost_per_gb +
        self.cpu_time_seconds * costs.cpu_cost_per_second +
        self.gpu_time_seconds * costs.gpu_cost_per_second +
        self.energy_kwh * costs.energy_cost_per_kwh
    }

    /// Check if usage exceeds limits
    pub fn exceeds_limits(&self, limits: &ResourceLimits<T>) -> bool {
        self.memory_gb > limits.max_memory_gb ||
        self.cpu_time_seconds > limits.max_cpu_seconds ||
        self.gpu_time_seconds > limits.max_gpu_seconds ||
        self.energy_kwh > limits.max_energy_kwh
    }
}

/// Resource cost structure
#[derive(Debug, Clone)]
pub struct ResourceCosts<T: Float> {
    pub memory_cost_per_gb: T,
    pub cpu_cost_per_second: T,
    pub gpu_cost_per_second: T,
    pub energy_cost_per_kwh: T,
}

/// Resource limit structure
#[derive(Debug, Clone)]
pub struct ResourceLimits<T: Float> {
    pub max_memory_gb: T,
    pub max_cpu_seconds: T,
    pub max_gpu_seconds: T,
    pub max_energy_kwh: T,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_statistics_default() {
        let stats = SearchStatistics::<f32>::default();
        assert_eq!(stats.total_architectures_evaluated, 0);
        assert_eq!(stats.successful_evaluations, 0);
        assert_eq!(stats.failed_evaluations, 0);
        assert!(stats.best_score.is_none());
    }

    #[test]
    fn test_efficiency_score_calculation() {
        let mut stats = SearchStatistics::<f32>::default();
        stats.total_architectures_evaluated = 100;
        stats.successful_evaluations = 80;
        stats.total_search_time = Duration::from_secs(3600);

        let efficiency = stats.calculate_efficiency_score();
        assert!(efficiency > 0.0);
        assert!(efficiency <= 1.0);
    }

    #[test]
    fn test_convergence_detection() {
        let mut stats = SearchStatistics::<f32>::default();
        stats.score_history = vec![0.5, 0.6, 0.65, 0.66, 0.67, 0.67, 0.67, 0.68];

        let converged = stats.has_converged(5, 0.05);
        assert!(converged);

        let not_converged = stats.has_converged(5, 0.001);
        assert!(!not_converged);
    }

    #[test]
    fn test_improvement_trend() {
        let mut stats = SearchStatistics::<f32>::default();
        stats.score_history = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

        let trend = stats.get_improvement_trend(2);
        assert!(trend.is_some());
        assert!(trend.unwrap() > 0.0); // Positive trend
    }

    #[test]
    fn test_resource_usage_calculations() {
        let usage = ResourceUsage {
            memory_gb: 10.0,
            cpu_time_seconds: 3600.0,
            gpu_time_seconds: 1800.0,
            energy_kwh: 5.0,
            network_io_gb: 1.0,
            disk_io_gb: 2.0,
            peak_memory_gb: 12.0,
            efficiency_score: 0.8,
        };

        let costs = ResourceCosts {
            memory_cost_per_gb: 0.1,
            cpu_cost_per_second: 0.001,
            gpu_cost_per_second: 0.01,
            energy_cost_per_kwh: 0.2,
        };

        let total_cost = usage.total_cost(&costs);
        assert!(total_cost > 0.0);

        let limits = ResourceLimits {
            max_memory_gb: 8.0,
            max_cpu_seconds: 7200.0,
            max_gpu_seconds: 3600.0,
            max_energy_kwh: 10.0,
        };

        let exceeds = usage.exceeds_limits(&limits);
        assert!(exceeds); // Memory exceeds limit
    }

    #[test]
    fn test_architecture_encoding() {
        let encoding = ArchitectureEncoding::default();
        assert_eq!(encoding.hash, 0);
        assert!(matches!(encoding.encoding_type, EncodingType::String));
    }
}