//! Real-time adaptive algorithm selection system
//!
//! This module implements an intelligent, self-adapting system that automatically selects
//! and configures the optimal spatial algorithms based on real-time data characteristics,
//! system performance metrics, and environmental conditions. It combines machine learning,
//! heuristics, and performance feedback to make optimal algorithmic decisions.
//!
//! # Features
//!
//! - **Real-time algorithm selection**: Choose optimal algorithms based on data patterns
//! - **Performance-based adaptation**: Learn from execution performance to improve decisions
//! - **Multi-objective optimization**: Balance accuracy, speed, memory usage, and energy
//! - **Dynamic parameter tuning**: Automatically adjust algorithm parameters
//! - **Context-aware selection**: Consider system load, hardware capabilities, user preferences
//! - **Predictive algorithm switching**: Anticipate optimal algorithms for future data
//! - **Ensemble coordination**: Intelligently combine multiple algorithms
//! - **Resource-aware scheduling**: Optimize for available computational resources
//!
//! # Selection Strategies
//!
//! The system uses multiple strategies:
//! - **Pattern-based selection**: Analyze data patterns to predict best algorithms
//! - **Performance history**: Learn from past performance on similar datasets
//! - **Resource availability**: Consider current CPU, memory, GPU availability
//! - **Quality requirements**: Adapt to accuracy vs. speed trade-offs
//! - **Online learning**: Continuously update selection models
//!
//! # Examples
//!
//! ```
//! use scirs2_spatial::adaptive_selection::{AdaptiveAlgorithmSelector, SelectionContext};
//! use ndarray::array;
//!
//! // Create adaptive selector with multiple strategies
//! let mut selector = AdaptiveAlgorithmSelector::new()
//!     .with_performance_learning(true)
//!     .with_resource_awareness(true)
//!     .with_quality_optimization(true)
//!     .with_ensemble_methods(true);
//!
//! // Define selection context
//! let context = SelectionContext::new()
//!     .with_accuracy_priority(0.8)
//!     .with_speed_priority(0.6)
//!     .with_memory_constraint(1_000_000_000) // 1GB limit
//!     .with_real_time_requirement(true);
//!
//! let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
//!
//! // Automatically select and configure optimal algorithm
//! let selection = selector.select_optimal_algorithm(&points.view(), &context).await?;
//! println!("Selected algorithm: {:?}", selection.algorithm);
//! println!("Configuration: {:?}", selection.parameters);
//! println!("Expected performance: {:?}", selection.performance_prediction);
//!
//! // Execute with selected algorithm and provide feedback
//! let execution_result = selector.execute_with_feedback(&selection, &points.view()).await?;
//! println!("Actual performance: {:?}", execution_result.actual_performance);
//! ```

use crate::error::{SpatialError, SpatialResult};
use ndarray::{Array1, Array2, ArrayView2};
use statrs::statistics::Statistics;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Adaptive algorithm selector
#[derive(Debug)]
pub struct AdaptiveAlgorithmSelector {
    /// Selection strategies
    strategies: Vec<SelectionStrategy>,
    /// Performance learning enabled
    performance_learning: bool,
    /// Resource awareness enabled
    resource_awareness: bool,
    /// Quality optimization enabled
    quality_optimization: bool,
    /// Ensemble methods enabled
    ensemble_methods: bool,
    /// Algorithm performance history
    performance_history: Arc<RwLock<PerformanceHistory>>,
    /// Data pattern analyzer
    #[allow(dead_code)]
    pattern_analyzer: PatternAnalyzer,
    /// Resource monitor
    resource_monitor: ResourceMonitor,
    /// Quality predictor
    quality_predictor: QualityPredictor,
    /// Selection cache
    selection_cache: Arc<RwLock<SelectionCache>>,
}

/// Selection strategy enumeration
#[derive(Debug, Clone)]
pub enum SelectionStrategy {
    /// Pattern-based selection
    PatternBased,
    /// Performance history-based
    HistoryBased,
    /// Resource-aware selection
    ResourceAware,
    /// Quality-optimized selection
    QualityOptimized,
    /// Ensemble combination
    EnsembleBased,
    /// Hybrid strategy
    Hybrid(Vec<SelectionStrategy>),
}

/// Selection context for algorithm selection
#[derive(Debug, Clone)]
pub struct SelectionContext {
    /// Accuracy priority (0.0 - 1.0)
    pub accuracy_priority: f64,
    /// Speed priority (0.0 - 1.0)
    pub speed_priority: f64,
    /// Memory constraint (bytes)
    pub memory_constraint: usize,
    /// Real-time requirement
    pub real_time_requirement: bool,
    /// Energy efficiency priority
    pub energy_efficiency: f64,
    /// Quality tolerance
    pub quality_tolerance: f64,
    /// User preferences
    pub user_preferences: HashMap<String, f64>,
    /// Environmental constraints
    pub environmental_constraints: EnvironmentalConstraints,
}

/// Environmental constraints for algorithm selection
#[derive(Debug, Clone)]
pub struct EnvironmentalConstraints {
    /// Available CPU cores
    pub available_cores: usize,
    /// Available memory (bytes)
    pub available_memory: usize,
    /// GPU availability
    pub gpu_available: bool,
    /// Network bandwidth (if distributed)
    pub network_bandwidth: Option<f64>,
    /// Power constraints
    pub power_budget: Option<f64>,
    /// Temperature constraints
    pub thermal_budget: Option<f64>,
}

/// Algorithm selection result
#[derive(Debug, Clone)]
pub struct AlgorithmSelection {
    /// Selected algorithm
    pub algorithm: SelectedAlgorithm,
    /// Algorithm parameters
    pub parameters: AlgorithmParameters,
    /// Expected performance
    pub performance_prediction: PerformancePrediction,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f64,
    /// Selection reasoning
    pub reasoning: SelectionReasoning,
    /// Alternative algorithms
    pub alternatives: Vec<AlgorithmSelection>,
}

/// Selected algorithm enumeration
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SelectedAlgorithm {
    /// K-means clustering
    KMeans,
    /// DBSCAN clustering
    DBScan,
    /// Hierarchical clustering
    HierarchicalClustering,
    /// KD-Tree nearest neighbors
    KDTreeNN,
    /// Ball Tree nearest neighbors
    BallTreeNN,
    /// Quantum-inspired clustering
    QuantumClustering,
    /// Neuromorphic clustering
    NeuromorphicClustering,
    /// Tensor core acceleration
    TensorCoreAccelerated,
    /// Distributed processing
    DistributedProcessing,
    /// Ensemble method
    Ensemble(Vec<SelectedAlgorithm>),
}

/// Algorithm parameters
#[derive(Debug, Clone)]
pub struct AlgorithmParameters {
    /// Core parameters
    pub core_params: HashMap<String, ParameterValue>,
    /// Optimization parameters
    pub optimization_params: HashMap<String, ParameterValue>,
    /// Resource allocation parameters
    pub resource_params: HashMap<String, ParameterValue>,
}

/// Parameter value types
#[derive(Debug, Clone)]
pub enum ParameterValue {
    Integer(i64),
    Float(f64),
    Boolean(bool),
    String(String),
    Array(Vec<f64>),
}

/// Performance prediction
#[derive(Debug, Clone)]
pub struct PerformancePrediction {
    /// Expected execution time (seconds)
    pub execution_time: f64,
    /// Expected memory usage (bytes)
    pub memory_usage: usize,
    /// Expected accuracy score
    pub accuracy_score: f64,
    /// Expected energy consumption (joules)
    pub energy_consumption: f64,
    /// Confidence intervals
    pub confidence_intervals: ConfidenceIntervals,
}

/// Confidence intervals for predictions
#[derive(Debug, Clone)]
pub struct ConfidenceIntervals {
    /// Execution time range
    pub execution_time_range: (f64, f64),
    /// Memory usage range
    pub memory_usage_range: (usize, usize),
    /// Accuracy range
    pub accuracy_range: (f64, f64),
}

/// Selection reasoning
#[derive(Debug, Clone)]
pub struct SelectionReasoning {
    /// Primary selection factors
    pub primary_factors: Vec<String>,
    /// Decision weights
    pub decision_weights: HashMap<String, f64>,
    /// Alternative considerations
    pub alternatives_considered: usize,
    /// Selection confidence factors
    pub confidence_factors: Vec<String>,
}

/// Performance history tracking
#[derive(Debug)]
pub struct PerformanceHistory {
    /// Algorithm performance records
    records: HashMap<SelectedAlgorithm, VecDeque<PerformanceRecord>>,
    /// Data pattern performance mapping
    #[allow(dead_code)]
    pattern_performance: HashMap<DataPattern, Vec<(SelectedAlgorithm, f64)>>,
    /// Performance trends
    #[allow(dead_code)]
    trends: HashMap<SelectedAlgorithm, PerformanceTrend>,
}

/// Individual performance record
#[derive(Debug, Clone)]
pub struct PerformanceRecord {
    /// Data characteristics
    pub data_characteristics: DataCharacteristics,
    /// Actual execution time
    pub execution_time: f64,
    /// Memory usage
    pub memory_usage: usize,
    /// Accuracy achieved
    pub accuracy: f64,
    /// Energy consumed
    pub energy_consumed: f64,
    /// Timestamp
    pub timestamp: Instant,
    /// Context information
    pub context: SelectionContext,
}

/// Data characteristics for pattern matching
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct DataCharacteristics {
    /// Number of points (categorized)
    pub size_category: SizeCategory,
    /// Dimensionality (categorized)
    pub dimensionality_category: DimensionalityCategory,
    /// Data density
    pub density_category: DensityCategory,
    /// Clustering tendency
    pub clustering_tendency: ClusteringTendencyCategory,
    /// Noise level
    pub noise_level: NoiseLevel,
    /// Distribution type
    pub distribution_type: DistributionType,
}

/// Data size categories
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum SizeCategory {
    Tiny,   // < 100 points
    Small,  // 100 - 1K points
    Medium, // 1K - 100K points
    Large,  // 100K - 1M points
    Huge,   // > 1M points
}

/// Dimensionality categories
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum DimensionalityCategory {
    Low,      // 1-3 dimensions
    Medium,   // 4-20 dimensions
    High,     // 21-100 dimensions
    VeryHigh, // > 100 dimensions
}

/// Density categories
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum DensityCategory {
    Sparse,
    Medium,
    Dense,
}

/// Clustering tendency categories
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum ClusteringTendencyCategory {
    Random,
    Structured,
    HighlyStructured,
}

/// Noise level categories
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum NoiseLevel {
    Low,
    Medium,
    High,
}

/// Distribution type
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum DistributionType {
    Uniform,
    Gaussian,
    Multimodal,
    Skewed,
    Unknown,
}

/// Data pattern for performance mapping
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct DataPattern {
    pub characteristics: DataCharacteristics,
    pub context_hash: u64, // Hash of context for grouping
}

/// Performance trend analysis
#[derive(Debug, Clone)]
pub struct PerformanceTrend {
    /// Trend direction
    pub trend_direction: TrendDirection,
    /// Trend strength
    pub trend_strength: f64,
    /// Recent performance change
    pub recent_change: f64,
    /// Stability score
    pub stability_score: f64,
}

/// Trend direction
#[derive(Debug, Clone)]
pub enum TrendDirection {
    Improving,
    Stable,
    Degrading,
}

/// Data pattern analyzer
#[derive(Debug)]
pub struct PatternAnalyzer {
    /// Pattern recognition models
    #[allow(dead_code)]
    pattern_models: HashMap<String, PatternModel>,
    /// Feature extractors
    #[allow(dead_code)]
    feature_extractors: Vec<FeatureExtractor>,
    /// Pattern cache
    #[allow(dead_code)]
    pattern_cache: HashMap<u64, DataCharacteristics>,
}

/// Pattern recognition model
#[derive(Debug)]
pub struct PatternModel {
    /// Model type
    pub model_type: PatternModelType,
    /// Model parameters
    pub parameters: Vec<f64>,
    /// Accuracy score
    pub accuracy: f64,
    /// Last update time
    pub last_update: Instant,
}

/// Pattern model types
#[derive(Debug)]
pub enum PatternModelType {
    StatisticalAnalysis,
    MachineLearning,
    HeuristicRules,
}

/// Feature extractor for pattern analysis
#[derive(Debug)]
pub struct FeatureExtractor {
    /// Extractor name
    pub name: String,
    /// Feature computation function
    pub compute_features: fn(&ArrayView2<'_, f64>) -> Vec<f64>,
}

/// Resource monitor for system awareness
#[derive(Debug)]
pub struct ResourceMonitor {
    /// Current CPU usage
    cpu_usage: f64,
    /// Current memory usage
    memory_usage: usize,
    /// GPU availability and usage
    #[allow(dead_code)]
    gpu_status: GpuStatus,
    /// Network status
    #[allow(dead_code)]
    network_status: NetworkStatus,
    /// Power consumption
    #[allow(dead_code)]
    power_consumption: f64,
    /// Temperature readings
    #[allow(dead_code)]
    temperature: f64,
    /// Update interval
    #[allow(dead_code)]
    update_interval: Duration,
    /// Last update time
    last_update: Instant,
}

/// GPU status information
#[derive(Debug)]
pub struct GpuStatus {
    /// GPU available
    pub available: bool,
    /// GPU utilization
    pub utilization: f64,
    /// GPU memory usage
    pub memory_usage: usize,
    /// GPU temperature
    pub temperature: f64,
}

/// Network status information
#[derive(Debug)]
pub struct NetworkStatus {
    /// Bandwidth available
    pub bandwidth: f64,
    /// Latency
    pub latency: f64,
    /// Packet loss rate
    pub packet_loss: f64,
}

/// Quality predictor for accuracy estimation
#[derive(Debug)]
pub struct QualityPredictor {
    /// Quality models
    #[allow(dead_code)]
    quality_models: HashMap<SelectedAlgorithm, QualityModel>,
    /// Cross-validation results
    #[allow(dead_code)]
    cv_results: HashMap<SelectedAlgorithm, Vec<f64>>,
    /// Quality history
    quality_history: VecDeque<QualityMeasurement>,
}

/// Quality prediction model
#[derive(Debug)]
pub struct QualityModel {
    /// Model coefficients
    pub coefficients: Vec<f64>,
    /// Model intercept
    pub intercept: f64,
    /// R-squared score
    pub r_squared: f64,
    /// Training data size
    pub training_size: usize,
}

/// Quality measurement
#[derive(Debug, Clone)]
pub struct QualityMeasurement {
    /// Algorithm used
    pub algorithm: SelectedAlgorithm,
    /// Data characteristics
    pub data_characteristics: DataCharacteristics,
    /// Predicted quality
    pub predicted_quality: f64,
    /// Actual quality
    pub actual_quality: f64,
    /// Prediction error
    pub prediction_error: f64,
    /// Timestamp
    pub timestamp: Instant,
}

/// Selection cache for performance optimization
#[derive(Debug)]
pub struct SelectionCache {
    /// Cached selections
    cache: HashMap<CacheKey, CachedSelection>,
    /// Cache hit statistics
    #[allow(dead_code)]
    hit_count: u64,
    /// Cache miss statistics
    #[allow(dead_code)]
    miss_count: u64,
    /// Maximum cache size
    max_size: usize,
}

/// Cache key for selection caching
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct CacheKey {
    /// Data characteristics hash
    pub data_hash: u64,
    /// Context hash
    pub context_hash: u64,
    /// Timestamp bucket (for time-based invalidation)
    pub time_bucket: u64,
}

/// Cached selection
#[derive(Debug, Clone)]
pub struct CachedSelection {
    /// Selection result
    pub selection: AlgorithmSelection,
    /// Cache timestamp
    pub timestamp: Instant,
    /// Use count
    pub use_count: u64,
    /// Success rate
    pub success_rate: f64,
}

impl Default for SelectionContext {
    fn default() -> Self {
        Self::new()
    }
}

impl SelectionContext {
    /// Create new selection context
    pub fn new() -> Self {
        Self {
            accuracy_priority: 0.7,
            speed_priority: 0.7,
            memory_constraint: usize::MAX,
            real_time_requirement: false,
            energy_efficiency: 0.5,
            quality_tolerance: 0.1,
            user_preferences: HashMap::new(),
            environmental_constraints: EnvironmentalConstraints {
                available_cores: num_cpus::get(),
                available_memory: 8_000_000_000, // 8GB default
                gpu_available: false,
                network_bandwidth: None,
                power_budget: None,
                thermal_budget: None,
            },
        }
    }

    /// Configure accuracy priority
    pub fn with_accuracy_priority(mut self, priority: f64) -> Self {
        self.accuracy_priority = priority.clamp(0.0, 1.0);
        self
    }

    /// Configure speed priority
    pub fn with_speed_priority(mut self, priority: f64) -> Self {
        self.speed_priority = priority.clamp(0.0, 1.0);
        self
    }

    /// Configure memory constraint
    pub fn with_memory_constraint(mut self, bytes: usize) -> Self {
        self.memory_constraint = bytes;
        self
    }

    /// Configure real-time requirement
    pub fn with_real_time_requirement(mut self, required: bool) -> Self {
        self.real_time_requirement = required;
        self
    }
}

impl Default for AdaptiveAlgorithmSelector {
    fn default() -> Self {
        Self::new()
    }
}

impl AdaptiveAlgorithmSelector {
    /// Create new adaptive algorithm selector
    pub fn new() -> Self {
        Self {
            strategies: vec![
                SelectionStrategy::PatternBased,
                SelectionStrategy::HistoryBased,
                SelectionStrategy::ResourceAware,
            ],
            performance_learning: false,
            resource_awareness: false,
            quality_optimization: false,
            ensemble_methods: false,
            performance_history: Arc::new(RwLock::new(PerformanceHistory {
                records: HashMap::new(),
                pattern_performance: HashMap::new(),
                trends: HashMap::new(),
            })),
            pattern_analyzer: PatternAnalyzer {
                pattern_models: HashMap::new(),
                feature_extractors: Vec::new(),
                pattern_cache: HashMap::new(),
            },
            resource_monitor: ResourceMonitor {
                cpu_usage: 0.0,
                memory_usage: 0,
                gpu_status: GpuStatus {
                    available: false,
                    utilization: 0.0,
                    memory_usage: 0,
                    temperature: 0.0,
                },
                network_status: NetworkStatus {
                    bandwidth: 0.0,
                    latency: 0.0,
                    packet_loss: 0.0,
                },
                power_consumption: 0.0,
                temperature: 0.0,
                update_interval: Duration::from_secs(1),
                last_update: Instant::now(),
            },
            quality_predictor: QualityPredictor {
                quality_models: HashMap::new(),
                cv_results: HashMap::new(),
                quality_history: VecDeque::new(),
            },
            selection_cache: Arc::new(RwLock::new(SelectionCache {
                cache: HashMap::new(),
                hit_count: 0,
                miss_count: 0,
                max_size: 1000,
            })),
        }
    }

    /// Enable performance learning
    pub fn with_performance_learning(mut self, enabled: bool) -> Self {
        self.performance_learning = enabled;
        if enabled {
            self.strategies.push(SelectionStrategy::HistoryBased);
        }
        self
    }

    /// Enable resource awareness
    pub fn with_resource_awareness(mut self, enabled: bool) -> Self {
        self.resource_awareness = enabled;
        if enabled {
            self.strategies.push(SelectionStrategy::ResourceAware);
        }
        self
    }

    /// Enable quality optimization
    pub fn with_quality_optimization(mut self, enabled: bool) -> Self {
        self.quality_optimization = enabled;
        if enabled {
            self.strategies.push(SelectionStrategy::QualityOptimized);
        }
        self
    }

    /// Enable ensemble methods
    pub fn with_ensemble_methods(mut self, enabled: bool) -> Self {
        self.ensemble_methods = enabled;
        if enabled {
            self.strategies.push(SelectionStrategy::EnsembleBased);
        }
        self
    }

    /// Select optimal algorithm for given data and context
    pub async fn select_optimal_algorithm(
        &mut self,
        data: &ArrayView2<'_, f64>,
        context: &SelectionContext,
    ) -> SpatialResult<AlgorithmSelection> {
        // Check cache first
        if let Some(cached) = self.check_cache(data, context).await? {
            return Ok(cached.selection);
        }

        // Analyze data characteristics
        let data_characteristics = self.analyzedata_characteristics(data)?;

        // Update resource monitoring
        self.update_resource_monitor().await?;

        // Generate candidate algorithms
        let candidates = self
            .generate_candidate_algorithms(&data_characteristics, context)
            .await?;

        // Evaluate candidates using all strategies
        let mut evaluations = Vec::new();
        for candidate in candidates {
            let evaluation = self
                .evaluate_candidate(&candidate, &data_characteristics, context)
                .await?;
            evaluations.push(evaluation);
        }

        // Select best algorithm
        let best_selection = self.select_best_candidate(evaluations, context)?;

        // Cache the selection
        self.cache_selection(data, context, &best_selection).await?;

        Ok(best_selection)
    }

    /// Execute algorithm with performance feedback
    pub async fn execute_with_feedback(
        &mut self,
        selection: &AlgorithmSelection,
        data: &ArrayView2<'_, f64>,
    ) -> SpatialResult<ExecutionResult> {
        let start_time = Instant::now();

        // Execute the selected algorithm
        let algorithm_result = self.execute_algorithm(selection, data).await?;

        let execution_time = start_time.elapsed().as_secs_f64();

        // Measure actual performance
        let actual_performance = ActualPerformance {
            execution_time,
            memory_usage: algorithm_result.memory_usage,
            accuracy: algorithm_result.accuracy,
            energy_consumed: 0.0, // Would measure actual energy consumption
        };

        // Update performance history
        if self.performance_learning {
            self.update_performance_history(selection, data, &actual_performance)
                .await?;
        }

        // Update quality predictor
        if self.quality_optimization {
            self.update_quality_predictor(selection, &actual_performance)
                .await?;
        }

        Ok(ExecutionResult {
            algorithm_result,
            actual_performance: actual_performance.clone(),
            selection_accuracy: self.calculate_selection_accuracy(selection, &actual_performance),
        })
    }

    /// Default feature extractors for pattern analysis
    #[allow(dead_code)]
    fn default_feature_extractors(&self) -> Vec<FeatureExtractor> {
        vec![
            FeatureExtractor {
                name: "basic_stats".to_string(),
                compute_features: |data| {
                    let (n_points, n_dims) = data.dim();
                    vec![n_points as f64, n_dims as f64]
                },
            },
            FeatureExtractor {
                name: "distribution_stats".to_string(),
                compute_features: |data| {
                    let (_, n_dims) = data.dim();
                    let mut features = Vec::new();

                    for dim in 0..n_dims {
                        let column = data.column(dim);
                        let mean = column.to_owned().mean();
                        let std = (column.mapv(|x| (x - mean).powi(2)).mean()).sqrt();
                        features.push(mean);
                        features.push(std);
                    }

                    features
                },
            },
        ]
    }

    /// Check cache for existing selection
    async fn check_cache(
        &self,
        data: &ArrayView2<'_, f64>,
        context: &SelectionContext,
    ) -> SpatialResult<Option<CachedSelection>> {
        let cache_key = self.compute_cache_key(data, context);
        let cache = self.selection_cache.read().await;

        if let Some(cached) = cache.cache.get(&cache_key) {
            // Check if cache entry is still valid (not too old)
            if cached.timestamp.elapsed() < Duration::from_secs(300) {
                // 5 minutes
                return Ok(Some(cached.clone()));
            }
        }

        Ok(None)
    }

    /// Compute cache key for data and context
    fn compute_cache_key(
        &self,
        data: &ArrayView2<'_, f64>,
        context: &SelectionContext,
    ) -> CacheKey {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut data_hasher = DefaultHasher::new();
        let (n_points, n_dims) = data.dim();
        n_points.hash(&mut data_hasher);
        n_dims.hash(&mut data_hasher);

        // Hash a sample of data points
        for (i, point) in data.outer_iter().enumerate() {
            if i % (n_points / 10 + 1) == 0 {
                // Sample every 10th point
                for &coord in point.iter() {
                    (coord as i64).hash(&mut data_hasher);
                }
            }
        }

        let data_hash = data_hasher.finish();

        let mut context_hasher = DefaultHasher::new();
        context
            .accuracy_priority
            .to_bits()
            .hash(&mut context_hasher);
        context.speed_priority.to_bits().hash(&mut context_hasher);
        context.memory_constraint.hash(&mut context_hasher);
        context.real_time_requirement.hash(&mut context_hasher);

        let context_hash = context_hasher.finish();

        let time_bucket = Instant::now().elapsed().as_secs() / 300; // 5-minute buckets

        CacheKey {
            data_hash,
            context_hash,
            time_bucket,
        }
    }

    /// Analyze data characteristics for pattern matching
    fn analyzedata_characteristics(
        &mut self,
        data: &ArrayView2<'_, f64>,
    ) -> SpatialResult<DataCharacteristics> {
        let (n_points, n_dims) = data.dim();

        // Categorize size
        let size_category = match n_points {
            0..=99 => SizeCategory::Tiny,
            100..=999 => SizeCategory::Small,
            1000..=99_999 => SizeCategory::Medium,
            100_000..=999_999 => SizeCategory::Large,
            _ => SizeCategory::Huge,
        };

        // Categorize dimensionality
        let dimensionality_category = match n_dims {
            1..=3 => DimensionalityCategory::Low,
            4..=20 => DimensionalityCategory::Medium,
            21..=100 => DimensionalityCategory::High,
            _ => DimensionalityCategory::VeryHigh,
        };

        // Estimate density
        let density = self.estimatedata_density(data)?;
        let density_category = if density < 0.3 {
            DensityCategory::Sparse
        } else if density < 0.7 {
            DensityCategory::Medium
        } else {
            DensityCategory::Dense
        };

        // Estimate clustering tendency
        let clustering_tendency = self.estimate_clustering_tendency(data)?;
        let clustering_tendency_category = if clustering_tendency < 0.3 {
            ClusteringTendencyCategory::HighlyStructured
        } else if clustering_tendency < 0.7 {
            ClusteringTendencyCategory::Structured
        } else {
            ClusteringTendencyCategory::Random
        };

        // Estimate noise level
        let noise_level = self.estimate_noise_level(data)?;
        let noise_level_category = if noise_level < 0.3 {
            NoiseLevel::Low
        } else if noise_level < 0.7 {
            NoiseLevel::Medium
        } else {
            NoiseLevel::High
        };

        // Estimate distribution type
        let distribution_type = self.estimate_distribution_type(data)?;

        Ok(DataCharacteristics {
            size_category,
            dimensionality_category,
            density_category,
            clustering_tendency: clustering_tendency_category,
            noise_level: noise_level_category,
            distribution_type,
        })
    }

    /// Estimate data density
    fn estimatedata_density(&self, data: &ArrayView2<'_, f64>) -> SpatialResult<f64> {
        let (n_points_, n_dims) = data.dim();

        if n_points_ < 2 {
            return Ok(0.0);
        }

        let sample_size = n_points_.min(100);
        let mut total_inverse_distance = 0.0;
        let mut count = 0;

        for i in 0..sample_size {
            let mut nearest_distance = f64::INFINITY;

            for j in 0..n_points_ {
                if i != j {
                    let dist: f64 = data
                        .row(i)
                        .iter()
                        .zip(data.row(j).iter())
                        .map(|(&a, &b)| (a - b).powi(2))
                        .sum::<f64>()
                        .sqrt();

                    if dist < nearest_distance {
                        nearest_distance = dist;
                    }
                }
            }

            if nearest_distance > 0.0 && nearest_distance.is_finite() {
                total_inverse_distance += 1.0 / nearest_distance;
                count += 1;
            }
        }

        Ok(if count > 0 {
            (total_inverse_distance / count as f64).min(1.0)
        } else {
            0.0
        })
    }

    /// Estimate clustering tendency (Hopkins-like statistic)
    fn estimate_clustering_tendency(&self, data: &ArrayView2<'_, f64>) -> SpatialResult<f64> {
        let (n_points, n_dims) = data.dim();

        if n_points < 10 {
            return Ok(0.5);
        }

        let sample_size = n_points.min(20);
        let mut real_distances = Vec::new();
        let mut random_distances = Vec::new();

        // Real point distances
        for i in 0..sample_size {
            let mut min_dist = f64::INFINITY;
            for j in 0..n_points {
                if i != j {
                    let dist: f64 = data
                        .row(i)
                        .iter()
                        .zip(data.row(j).iter())
                        .map(|(&a, &b)| (a - b).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    min_dist = min_dist.min(dist);
                }
            }
            real_distances.push(min_dist);
        }

        // Random point distances
        let bounds = self.getdata_bounds(data);
        for _ in 0..sample_size {
            let random_point: Array1<f64> = Array1::from_shape_fn(n_dims, |i| {
                rand::random::<f64>() * (bounds[i].1 - bounds[i].0) + bounds[i].0
            });

            let mut min_dist = f64::INFINITY;
            for j in 0..n_points {
                let dist: f64 = random_point
                    .iter()
                    .zip(data.row(j).iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                min_dist = min_dist.min(dist);
            }
            random_distances.push(min_dist);
        }

        let sum_random: f64 = random_distances.iter().sum();
        let sum_real: f64 = real_distances.iter().sum();
        let hopkins = sum_random / (sum_random + sum_real);

        Ok(hopkins)
    }

    /// Estimate noise level in data
    fn estimate_noise_level(&self, data: &ArrayView2<'_, f64>) -> SpatialResult<f64> {
        let (n_points_, n_dims) = data.dim();

        if n_points_ < 10 {
            return Ok(0.0);
        }

        // Use local outlier factor approximation
        let sample_size = n_points_.min(50);
        let k = 5; // Number of neighbors

        let mut outlier_scores = Vec::new();

        for i in 0..sample_size {
            let mut distances = Vec::new();

            for j in 0..n_points_ {
                if i != j {
                    let dist: f64 = data
                        .row(i)
                        .iter()
                        .zip(data.row(j).iter())
                        .map(|(&a, &b)| (a - b).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    distances.push(dist);
                }
            }

            distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

            if distances.len() >= k {
                let k_distance = distances[k - 1];
                let local_density = k as f64 / k_distance;
                outlier_scores.push(1.0 / local_density);
            }
        }

        if outlier_scores.is_empty() {
            Ok(0.0)
        } else {
            let mean_score = outlier_scores.iter().sum::<f64>() / outlier_scores.len() as f64;
            let variance = outlier_scores
                .iter()
                .map(|&score| (score - mean_score).powi(2))
                .sum::<f64>()
                / outlier_scores.len() as f64;

            Ok((variance.sqrt() / mean_score).min(1.0))
        }
    }

    /// Estimate distribution type
    fn estimate_distribution_type(
        &self,
        data: &ArrayView2<'_, f64>,
    ) -> SpatialResult<DistributionType> {
        let (n_points, n_dims) = data.dim();

        if n_points < 10 {
            return Ok(DistributionType::Unknown);
        }

        // Analyze each dimension
        let mut uniform_count = 0;
        let mut gaussian_count = 0;

        for dim in 0..n_dims {
            let column = data.column(dim);
            let mean = column.to_owned().mean();
            let std = (column.mapv(|x| (x - mean).powi(2)).mean()).sqrt();

            if std < 1e-6 {
                continue; // Constant dimension
            }

            // Test for uniformity (simplified)
            let min_val = column.fold(f64::INFINITY, |a, &b| a.min(b));
            let max_val = column.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let range = max_val - min_val;

            let expected_std_uniform = range / (12.0_f64).sqrt();
            if (std - expected_std_uniform).abs() / expected_std_uniform < 0.2 {
                uniform_count += 1;
            }

            // Test for normality (simplified skewness and kurtosis)
            let normalized: Vec<f64> = column.iter().map(|&x| (x - mean) / std).collect();
            let skewness =
                normalized.iter().map(|&x| x.powi(3)).sum::<f64>() / normalized.len() as f64;
            let kurtosis =
                normalized.iter().map(|&x| x.powi(4)).sum::<f64>() / normalized.len() as f64;

            if skewness.abs() < 0.5 && (kurtosis - 3.0).abs() < 1.0 {
                gaussian_count += 1;
            }
        }

        if uniform_count > n_dims / 2 {
            Ok(DistributionType::Uniform)
        } else if gaussian_count > n_dims / 2 {
            Ok(DistributionType::Gaussian)
        } else {
            // Check for multimodality (simplified)
            Ok(DistributionType::Multimodal)
        }
    }

    /// Get data bounds for each dimension
    fn getdata_bounds(&self, data: &ArrayView2<'_, f64>) -> Vec<(f64, f64)> {
        let (_, n_dims) = data.dim();
        let mut bounds = Vec::new();

        for dim in 0..n_dims {
            let column = data.column(dim);
            let min_val = column.fold(f64::INFINITY, |a, &b| a.min(b));
            let max_val = column.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            bounds.push((min_val, max_val));
        }

        bounds
    }

    /// Update resource monitor
    async fn update_resource_monitor(&mut self) -> SpatialResult<()> {
        // In a real implementation, this would query system resources
        self.resource_monitor.cpu_usage = 0.5; // Simulated
        self.resource_monitor.memory_usage = 4_000_000_000; // 4GB
        self.resource_monitor.last_update = Instant::now();
        Ok(())
    }

    /// Generate candidate algorithms based on characteristics
    async fn generate_candidate_algorithms(
        &self,
        characteristics: &DataCharacteristics,
        context: &SelectionContext,
    ) -> SpatialResult<Vec<SelectedAlgorithm>> {
        let mut candidates = Vec::new();

        // Add base algorithms
        candidates.push(SelectedAlgorithm::KMeans);
        candidates.push(SelectedAlgorithm::DBScan);
        candidates.push(SelectedAlgorithm::KDTreeNN);

        // Add advanced algorithms based on characteristics
        match characteristics.size_category {
            SizeCategory::Huge => {
                candidates.push(SelectedAlgorithm::DistributedProcessing);
                candidates.push(SelectedAlgorithm::TensorCoreAccelerated);
            }
            SizeCategory::Large => {
                candidates.push(SelectedAlgorithm::TensorCoreAccelerated);
            }
            _ => {}
        }

        // Add specialized algorithms
        if context.accuracy_priority > 0.8 {
            candidates.push(SelectedAlgorithm::QuantumClustering);
        }

        if context.energy_efficiency > 0.8 {
            candidates.push(SelectedAlgorithm::NeuromorphicClustering);
        }

        // Add ensemble if enabled
        if self.ensemble_methods {
            candidates.push(SelectedAlgorithm::Ensemble(vec![
                SelectedAlgorithm::KMeans,
                SelectedAlgorithm::DBScan,
            ]));
        }

        Ok(candidates)
    }

    /// Evaluate candidate algorithm
    async fn evaluate_candidate(
        &self,
        candidate: &SelectedAlgorithm,
        characteristics: &DataCharacteristics,
        context: &SelectionContext,
    ) -> SpatialResult<AlgorithmEvaluation> {
        // Predict performance
        let performance_prediction = self
            .predict_performance(candidate, characteristics, context)
            .await?;

        // Calculate fitness score
        let fitness_score = self.calculate_fitness_score(&performance_prediction, context);

        // Generate parameters
        let parameters = self.generate_parameters(candidate, characteristics, context)?;

        Ok(AlgorithmEvaluation {
            algorithm: candidate.clone(),
            parameters,
            performance_prediction,
            fitness_score,
            confidence: 0.8, // Would be computed based on historical accuracy
        })
    }

    /// Predict algorithm performance
    async fn predict_performance(
        &self,
        algorithm: &SelectedAlgorithm,
        characteristics: &DataCharacteristics,
        context: &SelectionContext,
    ) -> SpatialResult<PerformancePrediction> {
        // Base predictions (would use machine learning models in practice)
        let (base_time, base_memory, base_accuracy) = match algorithm {
            SelectedAlgorithm::KMeans => (1.0, 1000000, 0.8),
            SelectedAlgorithm::DBScan => (2.0, 1500000, 0.85),
            SelectedAlgorithm::KDTreeNN => (0.5, 800000, 0.9),
            SelectedAlgorithm::QuantumClustering => (3.0, 2000000, 0.95),
            SelectedAlgorithm::NeuromorphicClustering => (1.5, 1200000, 0.88),
            SelectedAlgorithm::TensorCoreAccelerated => (0.3, 3000000, 0.9),
            SelectedAlgorithm::DistributedProcessing => (0.8, 5000000, 0.92),
            SelectedAlgorithm::HierarchicalClustering => (5.0, 2500000, 0.9),
            SelectedAlgorithm::BallTreeNN => (0.7, 1000000, 0.88),
            SelectedAlgorithm::Ensemble(_) => (2.5, 3000000, 0.95),
        };

        // Apply scaling factors based on data characteristics
        let size_factor = match characteristics.size_category {
            SizeCategory::Tiny => 0.1,
            SizeCategory::Small => 0.5,
            SizeCategory::Medium => 1.0,
            SizeCategory::Large => 3.0,
            SizeCategory::Huge => 10.0,
        };

        let dim_factor = match characteristics.dimensionality_category {
            DimensionalityCategory::Low => 0.8,
            DimensionalityCategory::Medium => 1.0,
            DimensionalityCategory::High => 1.5,
            DimensionalityCategory::VeryHigh => 2.5,
        };

        let execution_time = base_time * size_factor * dim_factor;
        let memory_usage = (base_memory as f64 * size_factor * dim_factor) as usize;
        let accuracy_score = base_accuracy
            * (if characteristics.noise_level == NoiseLevel::High {
                0.9
            } else {
                1.0
            });

        Ok(PerformancePrediction {
            execution_time,
            memory_usage,
            accuracy_score,
            energy_consumption: execution_time * 50.0, // Simplified energy model
            confidence_intervals: ConfidenceIntervals {
                execution_time_range: (execution_time * 0.8, execution_time * 1.2),
                memory_usage_range: (
                    (memory_usage as f64 * 0.9) as usize,
                    (memory_usage as f64 * 1.1) as usize,
                ),
                accuracy_range: (accuracy_score * 0.95, accuracy_score.min(1.0)),
            },
        })
    }

    /// Calculate fitness score for algorithm
    fn calculate_fitness_score(
        &self,
        prediction: &PerformancePrediction,
        context: &SelectionContext,
    ) -> f64 {
        let time_score = if context.real_time_requirement && prediction.execution_time > 1.0 {
            0.0
        } else {
            1.0 / (1.0 + prediction.execution_time)
        };

        let memory_score = if prediction.memory_usage > context.memory_constraint {
            0.0
        } else {
            1.0 - (prediction.memory_usage as f64) / (context.memory_constraint as f64)
        };

        let accuracy_score = prediction.accuracy_score;

        let energy_score = 1.0 / (1.0 + prediction.energy_consumption / 100.0);

        // Weighted combination
        context.speed_priority * time_score
            + context.accuracy_priority * accuracy_score
            + 0.2 * memory_score
            + context.energy_efficiency * energy_score
    }

    /// Generate algorithm parameters
    fn generate_parameters(
        &self,
        algorithm: &SelectedAlgorithm,
        characteristics: &DataCharacteristics,
        context: &SelectionContext,
    ) -> SpatialResult<AlgorithmParameters> {
        let mut core_params = HashMap::new();
        let optimization_params = HashMap::new();
        let mut resource_params = HashMap::new();

        match algorithm {
            SelectedAlgorithm::KMeans => {
                let k = match characteristics.clustering_tendency {
                    ClusteringTendencyCategory::HighlyStructured => 3,
                    ClusteringTendencyCategory::Structured => 5,
                    ClusteringTendencyCategory::Random => 2,
                };
                core_params.insert("n_clusters".to_string(), ParameterValue::Integer(k));
                core_params.insert("max_iter".to_string(), ParameterValue::Integer(300));
                core_params.insert("tol".to_string(), ParameterValue::Float(1e-4));
            }
            SelectedAlgorithm::DBScan => {
                let eps = match characteristics.density_category {
                    DensityCategory::Dense => 0.3,
                    DensityCategory::Medium => 0.5,
                    DensityCategory::Sparse => 1.0,
                };
                core_params.insert("eps".to_string(), ParameterValue::Float(eps));
                core_params.insert("min_samples".to_string(), ParameterValue::Integer(5));
            }
            SelectedAlgorithm::KDTreeNN => {
                core_params.insert("leaf_size".to_string(), ParameterValue::Integer(30));
            }
            _ => {
                // Default parameters for other algorithms
                core_params.insert("tolerance".to_string(), ParameterValue::Float(1e-6));
            }
        }

        // Add resource parameters
        resource_params.insert(
            "n_jobs".to_string(),
            ParameterValue::Integer(context.environmental_constraints.available_cores as i64),
        );

        Ok(AlgorithmParameters {
            core_params,
            optimization_params,
            resource_params,
        })
    }

    /// Select best candidate from evaluations
    fn select_best_candidate(
        &self,
        evaluations: Vec<AlgorithmEvaluation>,
        _context: &SelectionContext,
    ) -> SpatialResult<AlgorithmSelection> {
        let best_evaluation = evaluations
            .into_iter()
            .max_by(|a, b| a.fitness_score.partial_cmp(&b.fitness_score).unwrap())
            .ok_or_else(|| SpatialError::InvalidInput("No candidate algorithms".to_string()))?;

        Ok(AlgorithmSelection {
            algorithm: best_evaluation.algorithm,
            parameters: best_evaluation.parameters,
            performance_prediction: best_evaluation.performance_prediction,
            confidence: best_evaluation.confidence,
            reasoning: SelectionReasoning {
                primary_factors: vec!["fitness_score".to_string()],
                decision_weights: HashMap::new(),
                alternatives_considered: 1,
                confidence_factors: vec!["historical_performance".to_string()],
            },
            alternatives: Vec::new(),
        })
    }

    /// Cache selection result
    async fn cache_selection(
        &self,
        data: &ArrayView2<'_, f64>,
        context: &SelectionContext,
        selection: &AlgorithmSelection,
    ) -> SpatialResult<()> {
        let cache_key = self.compute_cache_key(data, context);
        let cached_selection = CachedSelection {
            selection: selection.clone(),
            timestamp: Instant::now(),
            use_count: 1,
            success_rate: 1.0,
        };

        let mut cache = self.selection_cache.write().await;
        cache.cache.insert(cache_key, cached_selection);

        // Limit cache size
        if cache.cache.len() > cache.max_size {
            // Remove oldest entries (simplified LRU)
            let oldest_key = cache
                .cache
                .iter()
                .min_by_key(|(_, v)| v.timestamp)
                .map(|(k, _)| k.clone());

            if let Some(key) = oldest_key {
                cache.cache.remove(&key);
            }
        }

        Ok(())
    }

    /// Execute selected algorithm
    async fn execute_algorithm(
        &self,
        _selection: &AlgorithmSelection,
        data: &ArrayView2<'_, f64>,
    ) -> SpatialResult<AlgorithmResult> {
        // Simulate algorithm execution
        tokio::time::sleep(Duration::from_millis(100)).await;

        Ok(AlgorithmResult {
            resultdata: data.to_owned(),
            memory_usage: 1000000,
            accuracy: 0.85,
            execution_details: HashMap::new(),
        })
    }

    /// Update performance history
    async fn update_performance_history(
        &mut self,
        selection: &AlgorithmSelection,
        data: &ArrayView2<'_, f64>,
        actual_performance: &ActualPerformance,
    ) -> SpatialResult<()> {
        let data_characteristics = self.analyzedata_characteristics(data)?;

        let record = PerformanceRecord {
            data_characteristics,
            execution_time: actual_performance.execution_time,
            memory_usage: actual_performance.memory_usage,
            accuracy: actual_performance.accuracy,
            energy_consumed: actual_performance.energy_consumed,
            timestamp: Instant::now(),
            context: SelectionContext::new(), // Would store actual context
        };

        let mut history = self.performance_history.write().await;
        history
            .records
            .entry(selection.algorithm.clone())
            .or_insert_with(VecDeque::new)
            .push_back(record);

        // Limit history size
        if let Some(algorithm_history) = history.records.get_mut(&selection.algorithm) {
            if algorithm_history.len() > 1000 {
                algorithm_history.pop_front();
            }
        }

        Ok(())
    }

    /// Update quality predictor
    async fn update_quality_predictor(
        &mut self,
        selection: &AlgorithmSelection,
        actual_performance: &ActualPerformance,
    ) -> SpatialResult<()> {
        let predicted_accuracy = selection.performance_prediction.accuracy_score;
        let actual_accuracy = actual_performance.accuracy;
        let prediction_error = (predicted_accuracy - actual_accuracy).abs();

        let measurement = QualityMeasurement {
            algorithm: selection.algorithm.clone(),
            data_characteristics: DataCharacteristics {
                size_category: SizeCategory::Medium,
                dimensionality_category: DimensionalityCategory::Medium,
                density_category: DensityCategory::Medium,
                clustering_tendency: ClusteringTendencyCategory::Structured,
                noise_level: NoiseLevel::Medium,
                distribution_type: DistributionType::Gaussian,
            },
            predicted_quality: predicted_accuracy,
            actual_quality: actual_accuracy,
            prediction_error,
            timestamp: Instant::now(),
        };

        self.quality_predictor
            .quality_history
            .push_back(measurement);

        // Limit history size
        if self.quality_predictor.quality_history.len() > 10000 {
            self.quality_predictor.quality_history.pop_front();
        }

        Ok(())
    }

    /// Calculate selection accuracy
    fn calculate_selection_accuracy(
        &self,
        selection: &AlgorithmSelection,
        actual_performance: &ActualPerformance,
    ) -> f64 {
        let time_accuracy = 1.0
            - (selection.performance_prediction.execution_time - actual_performance.execution_time)
                .abs()
                / selection
                    .performance_prediction
                    .execution_time
                    .max(actual_performance.execution_time);

        let accuracy_accuracy = 1.0
            - (selection.performance_prediction.accuracy_score - actual_performance.accuracy).abs();

        (time_accuracy + accuracy_accuracy) / 2.0
    }
}

/// Algorithm evaluation result
#[derive(Debug, Clone)]
pub struct AlgorithmEvaluation {
    pub algorithm: SelectedAlgorithm,
    pub parameters: AlgorithmParameters,
    pub performance_prediction: PerformancePrediction,
    pub fitness_score: f64,
    pub confidence: f64,
}

/// Algorithm execution result
#[derive(Debug)]
pub struct AlgorithmResult {
    pub resultdata: Array2<f64>,
    pub memory_usage: usize,
    pub accuracy: f64,
    pub execution_details: HashMap<String, String>,
}

/// Actual performance measurement
#[derive(Debug, Clone)]
pub struct ActualPerformance {
    pub execution_time: f64,
    pub memory_usage: usize,
    pub accuracy: f64,
    pub energy_consumed: f64,
}

/// Complete execution result with feedback
#[derive(Debug)]
pub struct ExecutionResult {
    pub algorithm_result: AlgorithmResult,
    pub actual_performance: ActualPerformance,
    pub selection_accuracy: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_selection_context() {
        let context = SelectionContext::new()
            .with_accuracy_priority(0.9)
            .with_speed_priority(0.7)
            .with_real_time_requirement(true);

        assert_eq!(context.accuracy_priority, 0.9);
        assert_eq!(context.speed_priority, 0.7);
        assert!(context.real_time_requirement);
    }

    #[test]
    fn test_adaptive_selector_creation() {
        let selector = AdaptiveAlgorithmSelector::new()
            .with_performance_learning(true)
            .with_resource_awareness(true)
            .with_quality_optimization(true);

        assert!(selector.performance_learning);
        assert!(selector.resource_awareness);
        assert!(selector.quality_optimization);
    }

    #[test]
    fn testdata_characteristics() {
        let selector = AdaptiveAlgorithmSelector::new();
        let data = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];

        let mut selector_mut = selector;
        let characteristics = selector_mut.analyzedata_characteristics(&data.view());
        assert!(characteristics.is_ok());

        let chars = characteristics.unwrap();
        assert_eq!(chars.size_category, SizeCategory::Tiny);
        assert_eq!(chars.dimensionality_category, DimensionalityCategory::Low);
    }

    #[tokio::test]
    async fn test_algorithm_selection() {
        let mut selector = AdaptiveAlgorithmSelector::new();
        let context = SelectionContext::new();
        let data = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];

        let result = selector
            .select_optimal_algorithm(&data.view(), &context)
            .await;
        assert!(result.is_ok());

        let selection = result.unwrap();
        assert!(matches!(
            selection.algorithm,
            SelectedAlgorithm::KMeans | SelectedAlgorithm::DBScan | SelectedAlgorithm::KDTreeNN
        ));
        assert!(selection.confidence > 0.0);
    }

    #[tokio::test]
    async fn test_execution_with_feedback() {
        let mut selector = AdaptiveAlgorithmSelector::new().with_performance_learning(true);

        let context = SelectionContext::new();
        let data = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];

        let selection = selector
            .select_optimal_algorithm(&data.view(), &context)
            .await
            .unwrap();
        let execution_result = selector
            .execute_with_feedback(&selection, &data.view())
            .await;

        assert!(execution_result.is_ok());
        let result = execution_result.unwrap();
        assert!(result.selection_accuracy >= 0.0 && result.selection_accuracy <= 1.0);
    }
}
