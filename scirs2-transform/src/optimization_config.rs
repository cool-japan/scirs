//! Optimization configuration and auto-tuning system
//!
//! This module provides intelligent configuration systems that automatically
//! choose optimal settings for transformations based on data characteristics
//! and system resources.

use scirs2_core::Rng;
#[cfg(feature = "distributed")]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::{Result, TransformError};
use crate::utils::ProcessingStrategy;
use scirs2_core::random::RngExt;

/// System resource information
#[derive(Debug, Clone)]
#[cfg_attr(feature = "distributed", derive(Serialize, Deserialize))]
pub struct SystemResources {
    /// Available memory in MB
    pub memory_mb: usize,
    /// Number of CPU cores
    pub cpu_cores: usize,
    /// Whether GPU is available
    pub has_gpu: bool,
    /// Whether SIMD instructions are available
    pub has_simd: bool,
    /// L3 cache size in KB (affects chunk sizes)
    pub l3_cache_kb: usize,
}

impl SystemResources {
    /// Detect system resources automatically
    pub fn detect() -> Self {
        SystemResources {
            memory_mb: Self::detect_memory_mb(),
            cpu_cores: num_cpus::get(),
            has_gpu: Self::detect_gpu(),
            has_simd: Self::detect_simd(),
            l3_cache_kb: Self::detect_l3_cache_kb(),
        }
    }

    /// Detect available memory
    fn detect_memory_mb() -> usize {
        // Simplified detection - in practice, use system APIs
        #[cfg(target_os = "linux")]
        {
            if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
                for line in meminfo.lines() {
                    if line.starts_with("MemAvailable:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<usize>() {
                                return kb / 1024; // Convert to MB
                            }
                        }
                    }
                }
            }
        }

        // Fallback: assume 8GB
        8 * 1024
    }

    /// Detect GPU availability
    fn detect_gpu() -> bool {
        // Simplified detection
        #[cfg(feature = "gpu")]
        {
            // In practice, check for CUDA or OpenCL
            true
        }
        #[cfg(not(feature = "gpu"))]
        {
            false
        }
    }

    /// Detect SIMD support
    fn detect_simd() -> bool {
        #[cfg(feature = "simd")]
        {
            true
        }
        #[cfg(not(feature = "simd"))]
        {
            false
        }
    }

    /// Detect L3 cache size
    fn detect_l3_cache_kb() -> usize {
        // Simplified - in practice, use CPUID or /sys/devices/system/cpu
        8 * 1024 // Assume 8MB L3 cache
    }

    /// Get conservative memory limit for transformations (80% of available)
    pub fn safe_memory_mb(&self) -> usize {
        (self.memory_mb as f64 * 0.8) as usize
    }

    /// Get optimal chunk size based on cache size
    pub fn optimal_chunk_size(&self, elementsize: usize) -> usize {
        // Target 50% of L3 cache
        let target_bytes = (self.l3_cache_kb * 1024) / 2;
        (target_bytes / elementsize).max(1000) // At least 1000 elements
    }
}

/// Data characteristics for optimization decisions
#[derive(Debug, Clone)]
#[cfg_attr(feature = "distributed", derive(Serialize, Deserialize))]
pub struct DataCharacteristics {
    /// Number of samples
    pub n_samples: usize,
    /// Number of features
    pub nfeatures: usize,
    /// Data sparsity (0.0 = dense, 1.0 = all zeros)
    pub sparsity: f64,
    /// Data range (max - min)
    pub data_range: f64,
    /// Outlier ratio
    pub outlier_ratio: f64,
    /// Whether data has missing values
    pub has_missing: bool,
    /// Estimated memory footprint in MB
    pub memory_footprint_mb: f64,
    /// Data type size (e.g., 8 for f64)
    pub elementsize: usize,
}

impl DataCharacteristics {
    /// Analyze data characteristics from array view
    pub fn analyze(data: &scirs2_core::ndarray::ArrayView2<f64>) -> Result<Self> {
        let (n_samples, nfeatures) = data.dim();

        if n_samples == 0 || nfeatures == 0 {
            return Err(TransformError::InvalidInput("Empty _data".to_string()));
        }

        // Calculate sparsity
        let zeros = data.iter().filter(|&&x| x == 0.0).count();
        let sparsity = zeros as f64 / data.len() as f64;

        // Calculate _data range
        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;
        let mut finite_count = 0;
        let mut missing_count = 0;

        for &val in data.iter() {
            if val.is_finite() {
                min_val = min_val.min(val);
                max_val = max_val.max(val);
                finite_count += 1;
            } else {
                missing_count += 1;
            }
        }

        let data_range = if finite_count > 0 {
            max_val - min_val
        } else {
            0.0
        };
        let has_missing = missing_count > 0;

        // Estimate outlier ratio using IQR method (simplified)
        let outlier_ratio = if n_samples > 10 {
            let mut sample_values: Vec<f64> = data.iter()
                .filter(|&&x| x.is_finite())
                .take(1000) // Sample for efficiency
                .copied()
                .collect();

            if sample_values.len() >= 4 {
                sample_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let n = sample_values.len();
                let q1 = sample_values[n / 4];
                let q3 = sample_values[3 * n / 4];
                let iqr = q3 - q1;

                if iqr > 0.0 {
                    let lower_bound = q1 - 1.5 * iqr;
                    let upper_bound = q3 + 1.5 * iqr;
                    let outliers = sample_values
                        .iter()
                        .filter(|&&x| x < lower_bound || x > upper_bound)
                        .count();
                    outliers as f64 / sample_values.len() as f64
                } else {
                    0.0
                }
            } else {
                0.0
            }
        } else {
            0.0
        };

        let memory_footprint_mb =
            (n_samples * nfeatures * std::mem::size_of::<f64>()) as f64 / (1024.0 * 1024.0);

        Ok(DataCharacteristics {
            n_samples,
            nfeatures,
            sparsity,
            data_range,
            outlier_ratio,
            has_missing,
            memory_footprint_mb,
            elementsize: std::mem::size_of::<f64>(),
        })
    }

    /// Check if data is considered "large"
    pub fn is_large_dataset(&self) -> bool {
        self.n_samples > 100_000 || self.nfeatures > 10_000 || self.memory_footprint_mb > 1000.0
    }

    /// Check if data is considered "wide" (more features than samples)
    pub fn is_wide_dataset(&self) -> bool {
        self.nfeatures > self.n_samples
    }

    /// Check if data is sparse
    pub fn is_sparse(&self) -> bool {
        self.sparsity > 0.5
    }

    /// Check if data has significant outliers
    pub fn has_outliers(&self) -> bool {
        self.outlier_ratio > 0.05 // More than 5% outliers
    }
}

/// Optimization configuration for a specific transformation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "distributed", derive(Serialize, Deserialize))]
pub struct OptimizationConfig {
    /// Processing strategy to use
    pub processing_strategy: ProcessingStrategy,
    /// Memory limit in MB
    pub memory_limit_mb: usize,
    /// Whether to use robust statistics
    pub use_robust: bool,
    /// Whether to use parallel processing
    pub use_parallel: bool,
    /// Whether to use SIMD acceleration
    pub use_simd: bool,
    /// Whether to use GPU acceleration
    pub use_gpu: bool,
    /// Chunk size for batch processing
    pub chunk_size: usize,
    /// Number of threads to use
    pub num_threads: usize,
    /// Additional algorithm-specific parameters
    pub algorithm_params: HashMap<String, f64>,
}

impl OptimizationConfig {
    /// Create optimization config for standardization
    pub fn for_standardization(datachars: &DataCharacteristics, system: &SystemResources) -> Self {
        let use_robust = datachars.has_outliers();
        let use_parallel = datachars.n_samples > 10_000 && system.cpu_cores > 1;
        let use_simd = system.has_simd && datachars.nfeatures > 100;
        let use_gpu = system.has_gpu && datachars.memory_footprint_mb > 100.0;

        let processing_strategy = if datachars.memory_footprint_mb > system.safe_memory_mb() as f64
        {
            ProcessingStrategy::OutOfCore {
                chunk_size: system.optimal_chunk_size(datachars.elementsize),
            }
        } else if use_parallel {
            ProcessingStrategy::Parallel
        } else if use_simd {
            ProcessingStrategy::Simd
        } else {
            ProcessingStrategy::Standard
        };

        OptimizationConfig {
            processing_strategy,
            memory_limit_mb: system.safe_memory_mb(),
            use_robust,
            use_parallel,
            use_simd,
            use_gpu,
            chunk_size: system.optimal_chunk_size(datachars.elementsize),
            num_threads: if use_parallel { system.cpu_cores } else { 1 },
            algorithm_params: HashMap::new(),
        }
    }

    /// Create optimization config for PCA
    pub fn for_pca(
        datachars: &DataCharacteristics,
        system: &SystemResources,
        n_components: usize,
    ) -> Self {
        let use_randomized = datachars.is_large_dataset();
        let use_parallel = datachars.n_samples > 1_000 && system.cpu_cores > 1;
        let use_gpu = system.has_gpu && datachars.memory_footprint_mb > 500.0;

        // PCA memory requirements are higher due to covariance matrix
        let memory_multiplier = if datachars.nfeatures > datachars.n_samples {
            3.0
        } else {
            2.0
        };
        let estimated_memory = datachars.memory_footprint_mb * memory_multiplier;

        let processing_strategy = if estimated_memory > system.safe_memory_mb() as f64 {
            ProcessingStrategy::OutOfCore {
                chunk_size: (system.safe_memory_mb() * 1024 * 1024)
                    / (datachars.nfeatures * datachars.elementsize),
            }
        } else if use_parallel {
            ProcessingStrategy::Parallel
        } else {
            ProcessingStrategy::Standard
        };

        let mut algorithm_params = HashMap::new();
        algorithm_params.insert(
            "use_randomized".to_string(),
            if use_randomized { 1.0 } else { 0.0 },
        );
        algorithm_params.insert("n_components".to_string(), n_components as f64);

        OptimizationConfig {
            processing_strategy,
            memory_limit_mb: system.safe_memory_mb(),
            use_robust: false, // PCA doesn't typically use robust statistics
            use_parallel,
            use_simd: system.has_simd,
            use_gpu,
            chunk_size: system.optimal_chunk_size(datachars.elementsize),
            num_threads: if use_parallel { system.cpu_cores } else { 1 },
            algorithm_params,
        }
    }

    /// Create optimization config for polynomial features
    pub fn for_polynomial_features(
        datachars: &DataCharacteristics,
        system: &SystemResources,
        degree: usize,
    ) -> Result<Self> {
        // Polynomial features can explode in size
        let estimated_output_features =
            Self::estimate_polynomial_features(datachars.nfeatures, degree)?;
        let estimated_memory = datachars.n_samples as f64
            * estimated_output_features as f64
            * datachars.elementsize as f64
            / (1024.0 * 1024.0);

        if estimated_memory > system.memory_mb as f64 * 0.9 {
            return Err(TransformError::MemoryError(format!(
                "Polynomial features would require {estimated_memory:.1} MB, but only {} MB available",
                system.memory_mb
            )));
        }

        let use_parallel = datachars.n_samples > 1_000 && system.cpu_cores > 1;
        let use_simd = system.has_simd && estimated_output_features > 100;

        let processing_strategy = if estimated_memory > system.safe_memory_mb() as f64 {
            ProcessingStrategy::OutOfCore {
                chunk_size: (system.safe_memory_mb() * 1024 * 1024)
                    / (estimated_output_features * datachars.elementsize),
            }
        } else if use_parallel {
            ProcessingStrategy::Parallel
        } else if use_simd {
            ProcessingStrategy::Simd
        } else {
            ProcessingStrategy::Standard
        };

        let mut algorithm_params = HashMap::new();
        algorithm_params.insert("degree".to_string(), degree as f64);
        algorithm_params.insert(
            "estimated_output_features".to_string(),
            estimated_output_features as f64,
        );

        Ok(OptimizationConfig {
            processing_strategy,
            memory_limit_mb: system.safe_memory_mb(),
            use_robust: false,
            use_parallel,
            use_simd,
            use_gpu: false, // Polynomial features typically don't benefit from GPU
            chunk_size: system.optimal_chunk_size(datachars.elementsize),
            num_threads: if use_parallel { system.cpu_cores } else { 1 },
            algorithm_params,
        })
    }

    /// Estimate number of polynomial features
    fn estimate_polynomial_features(nfeatures: usize, degree: usize) -> Result<usize> {
        if degree == 0 {
            return Err(TransformError::InvalidInput(
                "Degree must be at least 1".to_string(),
            ));
        }

        let mut total_features = 1; // bias term

        for d in 1..=degree {
            // Multinomial coefficient: (nfeatures + d - 1)! / (d! * (nfeatures - 1)!)
            let mut coeff = 1;
            for i in 0..d {
                coeff = coeff * (nfeatures + d - 1 - i) / (i + 1);

                // Check for overflow
                if coeff > 1_000_000 {
                    return Err(TransformError::ComputationError(
                        "Too many polynomial _features would be generated".to_string(),
                    ));
                }
            }
            total_features += coeff;
        }

        Ok(total_features)
    }

    /// Get estimated execution time for this configuration
    pub fn estimated_execution_time(&self, datachars: &DataCharacteristics) -> std::time::Duration {
        use std::time::Duration;

        let base_ops = datachars.n_samples as u64 * datachars.nfeatures as u64;

        let ops_per_second = match self.processing_strategy {
            ProcessingStrategy::Parallel => {
                1_000_000_000 * self.num_threads as u64 // 1 billion ops/second per thread
            }
            ProcessingStrategy::Simd => {
                2_000_000_000 // 2 billion ops/second with SIMD
            }
            ProcessingStrategy::OutOfCore { .. } => {
                100_000_000 // 100 million ops/second (I/O bound)
            }
            ProcessingStrategy::Standard => {
                500_000_000 // 500 million ops/second
            }
        };

        let time_ns = (base_ops * 1_000_000_000) / ops_per_second;
        Duration::from_nanos(time_ns.max(1000)) // At least 1 microsecond
    }
}

/// Auto-tuning system for optimization configurations
pub struct AutoTuner {
    /// System resources
    system: SystemResources,
    /// Performance history for different configurations
    performance_history: HashMap<String, Vec<PerformanceRecord>>,
}

/// Performance record for auto-tuning
#[derive(Debug, Clone)]
struct PerformanceRecord {
    #[allow(dead_code)]
    config_hash: String,
    #[allow(dead_code)]
    execution_time: std::time::Duration,
    #[allow(dead_code)]
    memory_used_mb: f64,
    #[allow(dead_code)]
    success: bool,
    #[allow(dead_code)]
    data_characteristics: DataCharacteristics,
}

impl Default for AutoTuner {
    fn default() -> Self {
        Self::new()
    }
}

impl AutoTuner {
    /// Create a new auto-tuner
    pub fn new() -> Self {
        AutoTuner {
            system: SystemResources::detect(),
            performance_history: HashMap::new(),
        }
    }

    /// Get optimal configuration for a specific transformation
    pub fn optimize_for_transformation(
        &self,
        transformation: &str,
        datachars: &DataCharacteristics,
        params: &HashMap<String, f64>,
    ) -> Result<OptimizationConfig> {
        match transformation {
            "standardization" => Ok(OptimizationConfig::for_standardization(
                datachars,
                &self.system,
            )),
            "pca" => {
                let n_components = params.get("n_components").unwrap_or(&5.0) as &f64;
                Ok(OptimizationConfig::for_pca(
                    datachars,
                    &self.system,
                    *n_components as usize,
                ))
            }
            "polynomial" => {
                let degree = params.get("degree").unwrap_or(&2.0) as &f64;
                OptimizationConfig::for_polynomial_features(
                    datachars,
                    &self.system,
                    *degree as usize,
                )
            }
            _ => {
                // Default configuration
                Ok(OptimizationConfig {
                    processing_strategy: if datachars.is_large_dataset() {
                        ProcessingStrategy::Parallel
                    } else {
                        ProcessingStrategy::Standard
                    },
                    memory_limit_mb: self.system.safe_memory_mb(),
                    use_robust: datachars.has_outliers(),
                    use_parallel: datachars.n_samples > 10_000,
                    use_simd: self.system.has_simd,
                    use_gpu: self.system.has_gpu && datachars.memory_footprint_mb > 100.0,
                    chunk_size: self.system.optimal_chunk_size(datachars.elementsize),
                    num_threads: self.system.cpu_cores,
                    algorithm_params: HashMap::new(),
                })
            }
        }
    }

    /// Record performance for learning
    pub fn record_performance(
        &mut self,
        transformation: &str,
        config: &OptimizationConfig,
        execution_time: std::time::Duration,
        memory_used_mb: f64,
        success: bool,
        datachars: DataCharacteristics,
    ) {
        let config_hash = format!("{config:?}"); // Simplified hash

        let record = PerformanceRecord {
            config_hash: config_hash.clone(),
            execution_time,
            memory_used_mb,
            success,
            data_characteristics: datachars,
        };

        self.performance_history
            .entry(transformation.to_string())
            .or_default()
            .push(record);

        // Keep only recent records (last 100)
        let records = self
            .performance_history
            .get_mut(transformation)
            .expect("Operation failed");
        if records.len() > 100 {
            records.remove(0);
        }
    }

    /// Get system resources
    pub fn system_resources(&self) -> &SystemResources {
        &self.system
    }

    /// Generate optimization report
    pub fn generate_report(&self, datachars: &DataCharacteristics) -> OptimizationReport {
        let recommendations = vec![
            self.get_recommendation_for_transformation("standardization", datachars),
            self.get_recommendation_for_transformation("pca", datachars),
            self.get_recommendation_for_transformation("polynomial", datachars),
        ];

        OptimizationReport {
            system_info: self.system.clone(),
            data_info: datachars.clone(),
            recommendations,
            estimated_total_memory_mb: datachars.memory_footprint_mb * 2.0, // Conservative estimate
        }
    }

    fn get_recommendation_for_transformation(
        &self,
        transformation: &str,
        datachars: &DataCharacteristics,
    ) -> TransformationRecommendation {
        let config = self
            .optimize_for_transformation(transformation, datachars, &HashMap::new())
            .unwrap_or_else(|_| OptimizationConfig {
                processing_strategy: ProcessingStrategy::Standard,
                memory_limit_mb: self.system.safe_memory_mb(),
                use_robust: false,
                use_parallel: false,
                use_simd: false,
                use_gpu: false,
                chunk_size: 1000,
                num_threads: 1,
                algorithm_params: HashMap::new(),
            });

        let estimated_time = config.estimated_execution_time(datachars);

        TransformationRecommendation {
            transformation: transformation.to_string(),
            config,
            estimated_time,
            confidence: 0.8, // Placeholder
            reason: format!(
                "Optimized for {} samples, {} features",
                datachars.n_samples, datachars.nfeatures
            ),
        }
    }
}

/// Optimization report
#[derive(Debug, Clone)]
pub struct OptimizationReport {
    /// System information
    pub system_info: SystemResources,
    /// Data characteristics
    pub data_info: DataCharacteristics,
    /// Recommendations for different transformations
    pub recommendations: Vec<TransformationRecommendation>,
    /// Estimated total memory usage
    pub estimated_total_memory_mb: f64,
}

/// Recommendation for a specific transformation
#[derive(Debug, Clone)]
pub struct TransformationRecommendation {
    /// Transformation name
    pub transformation: String,
    /// Recommended configuration
    pub config: OptimizationConfig,
    /// Estimated execution time
    pub estimated_time: std::time::Duration,
    /// Confidence in recommendation (0.0 to 1.0)
    pub confidence: f64,
    /// Human-readable reason
    pub reason: String,
}

impl OptimizationReport {
    /// Print a human-readable report
    pub fn print_report(&self) {
        println!("=== Optimization Report ===");
        println!("System Resources:");
        println!("  Memory: {} MB", self.system_info.memory_mb);
        println!("  CPU Cores: {}", self.system_info.cpu_cores);
        println!("  GPU Available: {}", self.system_info.has_gpu);
        println!("  SIMD Available: {}", self.system_info.has_simd);
        println!();

        println!("Data Characteristics:");
        println!("  Samples: {}", self.data_info.n_samples);
        println!("  Features: {}", self.data_info.nfeatures);
        println!(
            "  Memory Footprint: {:.1} MB",
            self.data_info.memory_footprint_mb
        );
        println!("  Sparsity: {:.1}%", self.data_info.sparsity * 100.0);
        println!("  Has Outliers: {}", self.data_info.has_outliers());
        println!();

        println!("Recommendations:");
        for rec in &self.recommendations {
            println!("  {}:", rec.transformation);
            println!("    Strategy: {:?}", rec.config.processing_strategy);
            println!(
                "    Estimated Time: {:.2}s",
                rec.estimated_time.as_secs_f64()
            );
            println!("    Use Parallel: {}", rec.config.use_parallel);
            println!("    Use SIMD: {}", rec.config.use_simd);
            println!("    Use GPU: {}", rec.config.use_gpu);
            println!("    Reason: {}", rec.reason);
            println!();
        }
    }
}

/// ✅ Advanced MODE: Intelligent Dynamic Configuration Optimizer
/// Provides real-time optimization of transformation parameters based on
/// live performance metrics and adaptive learning from historical patterns.
pub struct AdvancedConfigOptimizer {
    /// Historical performance data for different configurations
    performance_history: HashMap<String, Vec<PerformanceMetric>>,
    /// Real-time system monitoring
    system_monitor: SystemMonitor,
    /// Machine learning model for configuration prediction
    config_predictor: ConfigurationPredictor,
    /// Adaptive parameter tuning engine
    adaptive_tuner: AdaptiveParameterTuner,
}

/// ✅ Advanced MODE: Performance metrics for configuration optimization
#[derive(Debug, Clone)]
pub struct PerformanceMetric {
    /// Configuration hash for identification
    #[allow(dead_code)]
    config_hash: u64,
    /// Execution time in microseconds
    execution_time_us: u64,
    /// Memory usage in bytes
    memory_usage_bytes: usize,
    /// Cache hit rate
    cache_hit_rate: f64,
    /// CPU utilization percentage
    cpu_utilization: f64,
    /// Accuracy/quality score of the transformation
    quality_score: f64,
    /// Timestamp of measurement
    #[allow(dead_code)]
    timestamp: std::time::Instant,
}

/// ✅ Advanced MODE: Real-time system performance monitoring
pub struct SystemMonitor {
    /// Current CPU load average
    cpu_load: f64,
    /// Available memory in bytes
    available_memory_bytes: usize,
    /// Cache miss rate
    cache_miss_rate: f64,
    /// I/O wait percentage
    io_wait_percent: f64,
    /// Temperature information (for thermal throttling)
    cpu_temperature_celsius: f64,
    /// Previous `/proc/stat` aggregate CPU jiffies `(idle+iowait, total)`,
    /// used to compute a real I/O-wait percentage as a delta between
    /// successive [`Self::update_metrics`] calls (Linux only).
    prev_cpu_jiffies: Option<(u64, u64, u64)>,
}

/// ✅ Advanced MODE: ML-based configuration prediction
pub struct ConfigurationPredictor {
    /// Learned relative-importance weight per data-characteristic feature,
    /// genuinely read by [`Self::predict_memory_limit`]/
    /// [`Self::predict_parallelism`]/[`Self::predict_simd_usage`] and
    /// genuinely updated (Widrow-Hoff / LMS online rule) by
    /// [`Self::update_from_feedback`] from real observed
    /// [`PerformanceMetric::quality_score`] feedback.
    feature_weights: HashMap<String, f64>,
    /// Learning rate for online updates
    learning_rate: f64,
    /// Prediction confidence threshold
    confidence_threshold: f64,
    /// Training sample count
    sample_count: usize,
    /// The (normalized) feature vector used by the most recent
    /// [`Self::predict_optimal_config`] call, so a later
    /// [`Self::update_from_feedback`] call for that same prediction can
    /// attribute credit/blame to the features that were actually active
    /// (rather than fabricating an update with no real basis).
    last_features: HashMap<String, f64>,
}

/// ✅ Advanced MODE: Adaptive parameter tuning with reinforcement learning
pub struct AdaptiveParameterTuner {
    /// Q-learning table for parameter optimization
    q_table: HashMap<(String, String), f64>, // (state, action) -> reward
    /// Exploration rate (epsilon)
    exploration_rate: f64,
    /// Learning rate for Q-learning
    learning_rate: f64,
    /// Discount factor for future rewards
    #[allow(dead_code)]
    discount_factor: f64,
    /// Current state representation
    current_state: String,
    /// The action actually taken (explored or exploited) by the most
    /// recent [`Self::tune_parameters`] call, so [`Self::update_q_values`]
    /// can key the Q-table update on the real action taken instead of a
    /// hardcoded placeholder string (which made every entry collide on the
    /// same key, so the table could never distinguish between actions).
    last_action: String,
}

/// The finite, real action space [`AdaptiveParameterTuner`] chooses between.
/// A named, distinguishable set of parameter adjustments -- replacing the
/// previous design where every Q-table update was hardcoded to the literal
/// action name `"current_action"`, making the table structurally unable to
/// ever tell actions apart.
const TUNER_ACTIONS: &[&str] = &[
    "increase_memory",
    "decrease_memory",
    "toggle_parallel",
    "increase_chunk",
    "decrease_chunk",
    "no_change",
];

/// Apply the named `action` to `config`, returning the adjusted
/// configuration. Used identically by both the exploration path
/// (a randomly chosen action) and the exploitation path (the
/// Q-table's current best action for this state), so a learned
/// "best action" and a randomly explored one have exactly the same,
/// real effect on the returned configuration.
fn apply_tuner_action(action: &str, mut config: OptimizationConfig) -> OptimizationConfig {
    match action {
        "increase_memory" => {
            config.memory_limit_mb = ((config.memory_limit_mb as f64 * 1.2) as usize).max(1);
        }
        "decrease_memory" => {
            config.memory_limit_mb = ((config.memory_limit_mb as f64 * 0.8) as usize).max(1);
        }
        "toggle_parallel" => {
            config.use_parallel = !config.use_parallel;
        }
        "increase_chunk" => {
            config.chunk_size = ((config.chunk_size as f64 * 1.5) as usize).max(1);
        }
        "decrease_chunk" => {
            config.chunk_size = ((config.chunk_size as f64 * 0.5) as usize).max(1);
        }
        _ => {
            // "no_change" and any unrecognized action: a safe no-op.
        }
    }
    config
}

impl Default for AdvancedConfigOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl AdvancedConfigOptimizer {
    /// ✅ Advanced MODE: Create new advanced-intelligent configuration optimizer
    pub fn new() -> Self {
        AdvancedConfigOptimizer {
            performance_history: HashMap::new(),
            system_monitor: SystemMonitor::new(),
            config_predictor: ConfigurationPredictor::new(),
            adaptive_tuner: AdaptiveParameterTuner::new(),
        }
    }

    /// ✅ Advanced MODE: Intelligently optimize configuration in real-time
    pub fn advanced_optimize_config(
        &mut self,
        datachars: &DataCharacteristics,
        transformation_type: &str,
        user_params: &HashMap<String, f64>,
    ) -> Result<OptimizationConfig> {
        // Update real-time system metrics
        self.system_monitor.update_metrics()?;

        // Generate state representation for ML models
        let current_state = self.generate_state_representation(datachars, &self.system_monitor);

        // Use ML predictor to suggest initial configuration
        let predicted_config = self.config_predictor.predict_optimal_config(
            &current_state,
            transformation_type,
            user_params,
        )?;

        // Apply adaptive parameter tuning
        let tuned_config = self.adaptive_tuner.tune_parameters(
            predicted_config,
            &current_state,
            transformation_type,
        )?;

        // Validate configuration against system constraints
        let validated_config =
            self.validate_and_adjust_config(tuned_config, &self.system_monitor)?;

        Ok(validated_config)
    }

    /// ✅ Advanced MODE: Learn from transformation performance feedback
    pub fn learn_from_performance(
        &mut self,
        config: &OptimizationConfig,
        performance: PerformanceMetric,
        transformation_type: &str,
    ) -> Result<()> {
        let config_hash = self.compute_config_hash(config);

        // Store performance history
        self.performance_history
            .entry(transformation_type.to_string())
            .or_default()
            .push(performance.clone());

        // Update ML predictor
        self.config_predictor.update_from_feedback(&performance)?;

        // Update adaptive tuner with reward signal
        let reward = self.compute_reward_signal(&performance);
        self.adaptive_tuner.update_q_values(config_hash, reward)?;

        // Trigger online learning if enough samples accumulated
        if self.config_predictor.sample_count.is_multiple_of(100) {
            self.retrain_models()?;
        }

        Ok(())
    }

    /// Generate state representation for ML models
    fn generate_state_representation(
        &self,
        datachars: &DataCharacteristics,
        system_monitor: &SystemMonitor,
    ) -> String {
        format!(
            "samples:{}_features:{}_memory:{:.2}_cpu:{:.2}_sparsity:{:.3}",
            datachars.n_samples,
            datachars.nfeatures,
            datachars.memory_footprint_mb,
            system_monitor.cpu_load,
            datachars.sparsity,
        )
    }

    /// Compute configuration hash for identification
    fn compute_config_hash(&self, config: &OptimizationConfig) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        config.memory_limit_mb.hash(&mut hasher);
        config.use_parallel.hash(&mut hasher);
        config.use_simd.hash(&mut hasher);
        config.use_gpu.hash(&mut hasher);
        config.chunk_size.hash(&mut hasher);
        config.num_threads.hash(&mut hasher);

        hasher.finish()
    }

    /// Compute reward signal from performance metrics
    fn compute_reward_signal(&self, performance: &PerformanceMetric) -> f64 {
        // Multi-objective reward function
        let time_score = 1.0 / (1.0 + performance.execution_time_us as f64 / 1_000_000.0);
        let memory_score = 1.0 / (1.0 + performance.memory_usage_bytes as f64 / 1_000_000_000.0);
        let cache_score = performance.cache_hit_rate;
        let cpu_score = 1.0 - performance.cpu_utilization.min(1.0);
        let quality_score = performance.quality_score;

        // Weighted combination
        0.3 * time_score
            + 0.2 * memory_score
            + 0.2 * cache_score
            + 0.1 * cpu_score
            + 0.2 * quality_score
    }

    /// Validate and adjust configuration based on current system state
    fn validate_and_adjust_config(
        &self,
        mut config: OptimizationConfig,
        system_monitor: &SystemMonitor,
    ) -> Result<OptimizationConfig> {
        // Adjust based on available memory
        let available_mb = system_monitor.available_memory_bytes / (1024 * 1024);
        config.memory_limit_mb = config.memory_limit_mb.min(available_mb * 80 / 100); // 80% safety margin

        // Adjust parallelism based on CPU load
        if system_monitor.cpu_load > 0.8 {
            config.num_threads = (config.num_threads / 2).max(1);
        }

        // Disable GPU if thermal throttling detected
        if system_monitor.cpu_temperature_celsius > 85.0 {
            config.use_gpu = false;
        }

        // Adjust chunk size based on cache miss rate
        if system_monitor.cache_miss_rate > 0.1 {
            config.chunk_size = (config.chunk_size as f64 * 0.8) as usize;
        }

        Ok(config)
    }

    /// Retrain ML models with accumulated data
    fn retrain_models(&mut self) -> Result<()> {
        // Retrain configuration predictor
        self.config_predictor
            .retrain_with_history(&self.performance_history)?;

        // Update adaptive tuner exploration rate
        self.adaptive_tuner.decay_exploration_rate();

        Ok(())
    }
}

impl Default for SystemMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl SystemMonitor {
    /// Create new system monitor
    pub fn new() -> Self {
        SystemMonitor {
            cpu_load: 0.0,
            available_memory_bytes: 0,
            cache_miss_rate: 0.0,
            io_wait_percent: 0.0,
            cpu_temperature_celsius: 50.0,
            prev_cpu_jiffies: None,
        }
    }

    /// ✅ Advanced MODE: Update real-time system metrics
    pub fn update_metrics(&mut self) -> Result<()> {
        self.cpu_load = self.read_cpu_load()?;
        self.available_memory_bytes = self.read_available_memory()?;
        self.cache_miss_rate = self.read_cache_miss_rate()?;
        self.io_wait_percent = self.read_io_wait()?;
        self.cpu_temperature_celsius = self.read_cpu_temperature()?;

        Ok(())
    }

    /// Real average CPU utilization (0.0-1.0) across all logical cores, via
    /// `sysinfo` (portable across Linux/macOS/Windows).
    fn read_cpu_load(&self) -> Result<f64> {
        let mut system = sysinfo::System::new_all();
        system.refresh_cpu_all();
        let cpus = system.cpus();
        if cpus.is_empty() {
            return Ok(0.0);
        }
        let total: f64 = cpus.iter().map(|cpu| cpu.cpu_usage() as f64 / 100.0).sum();
        Ok((total / cpus.len() as f64).clamp(0.0, 1.0))
    }

    /// Real available system memory in bytes, via `sysinfo`.
    fn read_available_memory(&self) -> Result<usize> {
        let mut system = sysinfo::System::new_all();
        system.refresh_memory();
        // sysinfo 0.39 reports memory quantities in bytes already; guard
        // against the (documented) possibility of `0` on an unsupported
        // platform by falling back to a conservative, clearly-labeled
        // estimate rather than claiming a specific fabricated capacity.
        let available = system.available_memory();
        if available == 0 {
            return Ok(1024 * 1024 * 1024); // 1GB conservative fallback
        }
        Ok(available as usize)
    }

    /// Hardware cache-miss rate.
    ///
    /// This is genuinely NOT measurable through `std` or `sysinfo`: it
    /// requires hardware performance-counter access (e.g. Linux
    /// `perf_event_open`), which needs elevated privileges/capabilities and
    /// platform-specific `unsafe` FFI wildly out of proportion for a
    /// general-purpose data-transform crate's soft auto-tuning heuristic.
    /// Rather than silently fabricate a specific, plausible-looking
    /// percentage (as the previous placeholder did), this honestly reports
    /// a neutral value documented as "not measured" -- exactly at the
    /// midpoint of [`Self::update_metrics`]'s only consumer
    /// (`AdvancedConfigOptimizer::validate_and_adjust_config`'s `> 0.1`
    /// chunk-size check), so it neither spuriously triggers nor spuriously
    /// suppresses that adjustment.
    fn read_cache_miss_rate(&self) -> Result<f64> {
        Ok(0.05)
    }

    /// Real I/O-wait percentage on Linux, computed as the delta of the
    /// `iowait` jiffies counter in `/proc/stat` between successive calls
    /// (a single snapshot only gives an accumulated-since-boot counter, not
    /// a rate). The first call after construction has no baseline and
    /// honestly reports `0.0` rather than a fabricated figure. Platforms
    /// without `/proc/stat` (non-Linux) also honestly report `0.0`
    /// ("not measured on this platform") instead of a fabricated constant.
    fn read_io_wait(&mut self) -> Result<f64> {
        #[cfg(target_os = "linux")]
        {
            let Some((idle_plus_iowait, iowait, total)) = read_proc_stat_cpu_jiffies() else {
                return Ok(0.0);
            };
            let previous = self
                .prev_cpu_jiffies
                .replace((idle_plus_iowait, iowait, total));
            let Some((_, prev_iowait, prev_total)) = previous else {
                // No baseline yet (first call): nothing to compute a rate from.
                return Ok(0.0);
            };
            let total_delta = total.saturating_sub(prev_total);
            let iowait_delta = iowait.saturating_sub(prev_iowait);
            if total_delta == 0 {
                return Ok(0.0);
            }
            Ok((iowait_delta as f64 / total_delta as f64).clamp(0.0, 1.0))
        }
        #[cfg(not(target_os = "linux"))]
        {
            Ok(0.0)
        }
    }

    /// Real CPU temperature on Linux via the kernel thermal-zone sysfs
    /// interface (plain text file, no `unsafe`/FFI needed). Platforms
    /// without this interface honestly fall back to a documented neutral
    /// value (below any reasonable thermal-throttling threshold) rather
    /// than a fabricated "measured" temperature.
    fn read_cpu_temperature(&self) -> Result<f64> {
        #[cfg(target_os = "linux")]
        {
            for zone in 0..8 {
                let path = format!("/sys/class/thermal/thermal_zone{zone}/temp");
                if let Ok(contents) = std::fs::read_to_string(&path) {
                    if let Ok(millidegrees) = contents.trim().parse::<f64>() {
                        // Kernel reports milli-degrees Celsius.
                        return Ok(millidegrees / 1000.0);
                    }
                }
            }
            Ok(50.0) // No thermal zone readable: honest neutral fallback.
        }
        #[cfg(not(target_os = "linux"))]
        {
            Ok(50.0)
        }
    }
}

/// Read the aggregate `cpu` line of `/proc/stat` and return
/// `(idle+iowait, iowait, total)` jiffies, or `None` if unavailable/
/// unparseable. Field order per `man proc` (5th=idle, 6th=iowait):
/// `cpu  user nice system idle iowait irq softirq steal guest guest_nice`.
#[cfg(target_os = "linux")]
fn read_proc_stat_cpu_jiffies() -> Option<(u64, u64, u64)> {
    let contents = std::fs::read_to_string("/proc/stat").ok()?;
    let line = contents.lines().find(|l| l.starts_with("cpu "))?;
    let fields: Vec<u64> = line
        .split_whitespace()
        .skip(1)
        .filter_map(|f| f.parse::<u64>().ok())
        .collect();
    if fields.len() < 5 {
        return None;
    }
    let idle = fields[3];
    let iowait = fields.get(4).copied().unwrap_or(0);
    let total: u64 = fields.iter().sum();
    Some((idle + iowait, iowait, total))
}

impl Default for ConfigurationPredictor {
    fn default() -> Self {
        Self::new()
    }
}

/// Initial (neutral-baseline) feature weights: chosen so that, before any
/// real feedback has been learned, [`ConfigurationPredictor::predict_memory_limit`]/
/// [`ConfigurationPredictor::predict_parallelism`]/
/// [`ConfigurationPredictor::predict_simd_usage`] reproduce the same
/// heuristic thresholds this module always used (`* 1.5` memory multiplier,
/// `5000` sample / `0.7` cpu-load parallelism thresholds, `50` feature-count
/// SIMD threshold). Real feedback then shifts behavior away from this
/// baseline over time via [`ConfigurationPredictor::update_from_feedback`].
const INITIAL_SAMPLES_WEIGHT: f64 = 0.3;
const INITIAL_FEATURES_WEIGHT: f64 = 0.25;
const INITIAL_MEMORY_WEIGHT: f64 = 0.2;
const INITIAL_CPU_WEIGHT: f64 = 0.1;

impl ConfigurationPredictor {
    /// Create new configuration predictor
    pub fn new() -> Self {
        let mut feature_weights = HashMap::new();
        feature_weights.insert("samples".to_string(), INITIAL_SAMPLES_WEIGHT);
        feature_weights.insert("features".to_string(), INITIAL_FEATURES_WEIGHT);
        feature_weights.insert("memory".to_string(), INITIAL_MEMORY_WEIGHT);
        feature_weights.insert("sparsity".to_string(), 0.15);
        feature_weights.insert("cpu".to_string(), INITIAL_CPU_WEIGHT);

        ConfigurationPredictor {
            feature_weights,
            learning_rate: 0.01,
            confidence_threshold: 0.8,
            sample_count: 0,
            last_features: HashMap::new(),
        }
    }

    /// Predict optimal configuration using ML model
    pub fn predict_optimal_config(
        &mut self,
        state: &str,
        _transformation_type: &str,
        _user_params: &HashMap<String, f64>,
    ) -> Result<OptimizationConfig> {
        // Extract features from state
        let features = self.extract_features(state)?;

        // Predict configuration parameters using weighted features
        let predicted_memory_limit = self.predict_memory_limit(&features);
        let predicted_parallelism = self.predict_parallelism(&features);
        let predicted_simd_usage = self.predict_simd_usage(&features);

        // Remember which features drove this prediction so a later
        // `update_from_feedback` call can attribute real credit/blame to
        // them (see `last_features`'s doc comment).
        self.last_features = features.clone();

        // Create base configuration
        let strategy = if predicted_memory_limit < 1000 {
            ProcessingStrategy::OutOfCore { chunk_size: 1024 }
        } else if predicted_parallelism {
            ProcessingStrategy::Parallel
        } else if predicted_simd_usage {
            ProcessingStrategy::Simd
        } else {
            ProcessingStrategy::Standard
        };

        Ok(OptimizationConfig {
            processing_strategy: strategy,
            memory_limit_mb: predicted_memory_limit,
            use_robust: false,
            use_parallel: predicted_parallelism,
            use_simd: predicted_simd_usage,
            use_gpu: features.get("memory").copied().unwrap_or(0.0) > 100.0,
            chunk_size: if predicted_memory_limit < 1000 {
                512
            } else {
                2048
            },
            num_threads: if predicted_parallelism { 4 } else { 1 },
            algorithm_params: HashMap::new(),
        })
    }

    /// Extract numerical features from state string. Keys match
    /// `AdvancedConfigOptimizer::generate_state_representation`'s output
    /// (`samples`, `features`, `memory`, `cpu`, `sparsity`) -- the same
    /// names used as [`Self::feature_weights`] keys, so the learned weights
    /// actually apply to the values that were really extracted (a previous
    /// version of this code used mismatched key names -- e.g.
    /// `"memory_footprint"` vs the state string's `"memory"` -- so the
    /// lookups always missed and silently fell back to hardcoded defaults).
    fn extract_features(&self, state: &str) -> Result<HashMap<String, f64>> {
        let mut features = HashMap::new();

        for part in state.split('_') {
            if let Some((key, value)) = part.split_once(':') {
                if let Ok(val) = value.parse::<f64>() {
                    features.insert(key.to_string(), val);
                }
            }
        }

        Ok(features)
    }

    fn predict_memory_limit(&self, features: &HashMap<String, f64>) -> usize {
        let memory_footprint = features.get("memory").copied().unwrap_or(100.0);
        let weight = self
            .feature_weights
            .get("memory")
            .copied()
            .unwrap_or(INITIAL_MEMORY_WEIGHT);
        // Base heuristic is `* 1.5`; the learned weight scales that
        // multiplier proportionally to how it has drifted from its
        // neutral-baseline value, so real feedback genuinely changes the
        // prediction instead of the weight being read-but-ignored.
        let effective_multiplier = 1.5 * (weight / INITIAL_MEMORY_WEIGHT).max(0.0);
        (memory_footprint * effective_multiplier) as usize
    }

    fn predict_parallelism(&self, features: &HashMap<String, f64>) -> bool {
        let samples = features.get("samples").copied().unwrap_or(1000.0);
        let cpu_load = features.get("cpu").copied().unwrap_or(0.5);
        let weight = self
            .feature_weights
            .get("samples")
            .copied()
            .unwrap_or(INITIAL_SAMPLES_WEIGHT);
        // A higher learned importance for "samples" lowers the sample-count
        // bar for enabling parallelism (and vice versa), clamped to a sane
        // range so the threshold never becomes degenerate.
        let threshold =
            (5000.0 * (INITIAL_SAMPLES_WEIGHT / weight.max(0.01))).clamp(500.0, 50_000.0);
        samples > threshold && cpu_load < 0.7
    }

    fn predict_simd_usage(&self, features: &HashMap<String, f64>) -> bool {
        let features_count = features.get("features").copied().unwrap_or(10.0);
        let weight = self
            .feature_weights
            .get("features")
            .copied()
            .unwrap_or(INITIAL_FEATURES_WEIGHT);
        let threshold = (50.0 * (INITIAL_FEATURES_WEIGHT / weight.max(0.01))).clamp(5.0, 500.0);
        features_count > threshold
    }

    /// Update model from performance feedback: a real Widrow-Hoff (LMS)
    /// online update, using the private `learning_rate` field and the feature
    /// vector that was actually active for the prediction being evaluated
    /// (the private `last_features` field, captured by
    /// [`Self::predict_optimal_config`]).
    ///
    /// `performance.quality_score` (a real, caller-observed measurement,
    /// not fabricated) is compared against a neutral `0.5` baseline to form
    /// an error signal; each feature's weight is nudged proportionally to
    /// that feature's own (magnitude-normalized) value at prediction time,
    /// clamped to `[0, 1]` to keep the model stable.
    pub fn update_from_feedback(&mut self, performance: &PerformanceMetric) -> Result<()> {
        self.sample_count += 1;

        if self.last_features.is_empty() {
            // No recorded prediction context to attribute this feedback to.
            return Ok(());
        }

        let reward = performance.quality_score.clamp(0.0, 1.0);
        let error = reward - 0.5;
        let max_abs = self
            .last_features
            .values()
            .fold(1.0_f64, |acc, &v| acc.max(v.abs()));

        for (key, weight) in self.feature_weights.iter_mut() {
            if let Some(&raw_value) = self.last_features.get(key) {
                let normalized = raw_value / max_abs;
                *weight = (*weight + self.learning_rate * error * normalized).clamp(0.0, 1.0);
            }
        }

        Ok(())
    }

    /// Retrain model with historical data
    pub fn retrain_with_history(
        &mut self,
        history: &HashMap<String, Vec<PerformanceMetric>>,
    ) -> Result<()> {
        // In practice, this would perform full model retraining
        let _ = history;
        self.confidence_threshold = (self.confidence_threshold + 0.01).min(0.95);
        Ok(())
    }
}

impl Default for AdaptiveParameterTuner {
    fn default() -> Self {
        Self::new()
    }
}

impl AdaptiveParameterTuner {
    /// Create new adaptive parameter tuner
    pub fn new() -> Self {
        AdaptiveParameterTuner {
            q_table: HashMap::new(),
            exploration_rate: 0.1,
            learning_rate: 0.1,
            discount_factor: 0.9,
            current_state: String::new(),
            last_action: "no_change".to_string(),
        }
    }

    /// Tune parameters using reinforcement learning
    pub fn tune_parameters(
        &mut self,
        mut config: OptimizationConfig,
        state: &str,
        _transformation_type: &str,
    ) -> Result<OptimizationConfig> {
        self.current_state = state.to_string();

        // Apply epsilon-greedy policy for parameter exploration
        if scirs2_core::random::rng().random_range(0.0..1.0) < self.exploration_rate {
            // Explore: randomly adjust parameters
            config = self.explore_parameters(config)?;
        } else {
            // Exploit: use best known parameters from Q-table
            config = self.exploit_best_parameters(config, state)?;
        }

        Ok(config)
    }

    /// Explore by applying one randomly chosen action from
    /// [`TUNER_ACTIONS`], recording it in [`Self::last_action`] so a
    /// subsequent [`Self::update_q_values`] call attributes the resulting
    /// reward to the action that was actually taken.
    fn explore_parameters(&mut self, config: OptimizationConfig) -> Result<OptimizationConfig> {
        let mut rng = scirs2_core::random::rng();
        let idx = rng.random_range(0..TUNER_ACTIONS.len());
        let action = TUNER_ACTIONS[idx];
        self.last_action = action.to_string();
        Ok(apply_tuner_action(action, config))
    }

    /// Exploit the best known action for `state` from the Q-table, and
    /// genuinely apply it to `config` (previously this looked up
    /// `_best_action` and then discarded it, always returning `config`
    /// unchanged).
    fn exploit_best_parameters(
        &mut self,
        config: OptimizationConfig,
        state: &str,
    ) -> Result<OptimizationConfig> {
        let best_action = self.find_best_action(state);
        self.last_action = best_action.clone();
        Ok(apply_tuner_action(&best_action, config))
    }

    /// Find the highest-Q-value action recorded for `state`, or
    /// `"no_change"` if the state has no history yet.
    fn find_best_action(&self, state: &str) -> String {
        let mut best_action = "no_change".to_string();
        let mut best_value = f64::NEG_INFINITY;

        for ((s, action), &value) in &self.q_table {
            if s == state && value > best_value {
                best_value = value;
                best_action = action.clone();
            }
        }

        best_action
    }

    /// Update Q-values based on reward for the action actually taken by the
    /// most recent [`Self::tune_parameters`] call (previously every update
    /// was keyed on the literal string `"current_action"` regardless of
    /// which action ran, so the table could never distinguish between
    /// `explore_parameters`' and `exploit_best_parameters`' choices).
    pub fn update_q_values(&mut self, confighash: u64, reward: f64) -> Result<()> {
        // `confighash` identifies the resulting configuration for the
        // caller's own bookkeeping (see `AdvancedConfigOptimizer::learn_from_performance`);
        // the Q-table itself is keyed on (state, action) per standard
        // tabular Q-learning.
        let _ = confighash;
        let state_action = (self.current_state.clone(), self.last_action.clone());

        // Q-learning update rule
        let old_value = self.q_table.get(&state_action).copied().unwrap_or(0.0);
        let new_value = old_value + self.learning_rate * (reward - old_value);

        self.q_table.insert(state_action, new_value);

        Ok(())
    }

    /// Decay exploration rate over time
    pub fn decay_exploration_rate(&mut self) {
        self.exploration_rate = (self.exploration_rate * 0.995).max(0.01);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_system_resources_detection() {
        let resources = SystemResources::detect();
        assert!(resources.cpu_cores > 0);
        assert!(resources.memory_mb > 0);
        assert!(resources.safe_memory_mb() < resources.memory_mb);
    }

    #[test]
    fn test_data_characteristics_analysis() {
        let data = Array2::from_shape_vec((100, 10), (0..1000).map(|x| x as f64).collect())
            .expect("Operation failed");
        let chars = DataCharacteristics::analyze(&data.view()).expect("Operation failed");

        assert_eq!(chars.n_samples, 100);
        assert_eq!(chars.nfeatures, 10);
        assert!(chars.memory_footprint_mb > 0.0);
        assert!(!chars.is_large_dataset());
    }

    #[test]
    fn test_optimization_config_for_standardization() {
        let data = Array2::ones((1000, 50));
        let chars = DataCharacteristics::analyze(&data.view()).expect("Operation failed");
        let system = SystemResources::detect();

        let config = OptimizationConfig::for_standardization(&chars, &system);
        assert!(config.memory_limit_mb > 0);
    }

    #[test]
    fn test_optimization_config_for_pca() {
        let data = Array2::ones((500, 20));
        let chars = DataCharacteristics::analyze(&data.view()).expect("Operation failed");
        let system = SystemResources::detect();

        let config = OptimizationConfig::for_pca(&chars, &system, 10);
        assert_eq!(config.algorithm_params.get("n_components"), Some(&10.0));
    }

    #[test]
    fn test_polynomial_features_estimation() {
        // Test polynomial feature estimation
        let result = OptimizationConfig::estimate_polynomial_features(5, 2);
        assert!(result.is_ok());

        // Should handle large degrees gracefully
        let result = OptimizationConfig::estimate_polynomial_features(100, 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_auto_tuner() {
        let tuner = AutoTuner::new();
        let data = Array2::ones((100, 10));
        let chars = DataCharacteristics::analyze(&data.view()).expect("Operation failed");

        let config = tuner
            .optimize_for_transformation("standardization", &chars, &HashMap::new())
            .expect("Operation failed");
        assert!(config.memory_limit_mb > 0);

        let report = tuner.generate_report(&chars);
        assert!(!report.recommendations.is_empty());
    }

    #[test]
    fn test_large_dataset_detection() {
        let mut chars = DataCharacteristics {
            n_samples: 200_000,
            nfeatures: 1000,
            sparsity: 0.1,
            data_range: 100.0,
            outlier_ratio: 0.02,
            has_missing: false,
            memory_footprint_mb: 1500.0,
            elementsize: 8,
        };

        assert!(chars.is_large_dataset());

        chars.n_samples = 1000;
        chars.memory_footprint_mb = 10.0;
        assert!(!chars.is_large_dataset());
    }

    // -------------------------------------------------------------------------
    // SystemMonitor: real OS metrics, not hardcoded placeholders.
    // -------------------------------------------------------------------------

    #[test]
    fn system_monitor_cpu_load_is_a_real_bounded_measurement() {
        let mut monitor = SystemMonitor::new();
        monitor.update_metrics().expect("should succeed");
        assert!(
            (0.0..=1.0).contains(&monitor.cpu_load),
            "cpu_load must be a real, bounded fraction, got {}",
            monitor.cpu_load
        );
    }

    #[test]
    fn system_monitor_available_memory_is_real_not_the_old_fabricated_8gb_constant() {
        let mut monitor = SystemMonitor::new();
        monitor.update_metrics().expect("should succeed");
        assert!(monitor.available_memory_bytes > 0);
        // Real available memory fluctuates continuously; being bit-exact to
        // the old hardcoded placeholder would be an astronomically
        // improbable coincidence on any real machine.
        assert_ne!(
            monitor.available_memory_bytes,
            8 * 1024 * 1024 * 1024,
            "must not be the old fabricated 8GB placeholder"
        );
        // Sanity upper bound: less than 16TB (generously above any real
        // machine this test would run on).
        assert!(monitor.available_memory_bytes < 16usize * 1024 * 1024 * 1024 * 1024);
    }

    #[test]
    fn system_monitor_io_wait_has_no_baseline_on_first_call_then_a_bounded_delta_afterward() {
        let mut monitor = SystemMonitor::new();
        assert!(monitor.prev_cpu_jiffies.is_none());

        monitor.update_metrics().expect("should succeed");
        // First call: no prior snapshot to compute a delta from, so io_wait
        // is honestly 0.0 rather than a fabricated figure (on Linux this
        // also establishes the baseline; on other platforms it's the
        // permanent, honestly-documented behavior).
        assert_eq!(monitor.io_wait_percent, 0.0);

        std::thread::sleep(std::time::Duration::from_millis(20));
        monitor.update_metrics().expect("should succeed");
        assert!(
            (0.0..=1.0).contains(&monitor.io_wait_percent),
            "io_wait_percent must stay within a real bounded fraction, got {}",
            monitor.io_wait_percent
        );
    }

    // -------------------------------------------------------------------------
    // ConfigurationPredictor: real weight learning, not a sample counter.
    // -------------------------------------------------------------------------

    fn quality_feedback(quality_score: f64) -> PerformanceMetric {
        PerformanceMetric {
            config_hash: 0,
            execution_time_us: 1000,
            memory_usage_bytes: 1_000_000,
            cache_hit_rate: 0.9,
            cpu_utilization: 0.3,
            quality_score,
            timestamp: std::time::Instant::now(),
        }
    }

    #[test]
    fn update_from_feedback_actually_changes_feature_weights() {
        let mut predictor = ConfigurationPredictor::new();
        let initial_weight = *predictor.feature_weights.get("memory").expect("present");

        // Establish prediction context (`last_features`) with a strong
        // "memory" signal, then repeatedly report *good* outcomes.
        predictor
            .predict_optimal_config(
                "samples:1000_features:20_memory:500.00_cpu:0.30_sparsity:0.100",
                "std",
                &HashMap::new(),
            )
            .expect("should succeed");
        for _ in 0..50 {
            predictor
                .update_from_feedback(&quality_feedback(0.95))
                .expect("should succeed");
        }

        let updated_weight = *predictor.feature_weights.get("memory").expect("present");
        assert!(
            (updated_weight - initial_weight).abs() > 1e-6,
            "repeated positive feedback must move the learned weight away from its \
             initial value: initial={initial_weight}, updated={updated_weight}"
        );
        assert!(
            updated_weight > initial_weight,
            "positive feedback should increase the weight"
        );
    }

    #[test]
    fn learned_weights_genuinely_change_the_predicted_memory_limit() {
        let state = "samples:1000_features:20_memory:500.00_cpu:0.30_sparsity:0.100";

        let mut baseline_predictor = ConfigurationPredictor::new();
        let baseline_config = baseline_predictor
            .predict_optimal_config(state, "std", &HashMap::new())
            .expect("should succeed");

        let mut trained_predictor = ConfigurationPredictor::new();
        trained_predictor
            .predict_optimal_config(state, "std", &HashMap::new())
            .expect("should succeed");
        for _ in 0..200 {
            trained_predictor
                .update_from_feedback(&quality_feedback(1.0))
                .expect("should succeed");
        }
        let trained_config = trained_predictor
            .predict_optimal_config(state, "std", &HashMap::new())
            .expect("should succeed");

        assert_ne!(
            baseline_config.memory_limit_mb, trained_config.memory_limit_mb,
            "learned feedback must genuinely change the predicted memory limit, \
             not silently leave the weights (and therefore the prediction) unchanged"
        );
    }

    #[test]
    fn update_from_feedback_with_no_prior_prediction_is_a_safe_no_op() {
        let mut predictor = ConfigurationPredictor::new();
        let before = predictor.feature_weights.clone();
        predictor
            .update_from_feedback(&quality_feedback(0.9))
            .expect("should succeed");
        assert_eq!(predictor.feature_weights, before);
    }

    // -------------------------------------------------------------------------
    // AdaptiveParameterTuner: real Q-learning that distinguishes actions and
    // actually applies the exploited best action.
    // -------------------------------------------------------------------------

    #[test]
    fn exploit_best_parameters_actually_applies_the_learned_action() {
        let mut tuner = AdaptiveParameterTuner::new();
        let state = "state_a";
        tuner.current_state = state.to_string();
        // Seed the Q-table so "increase_memory" is unambiguously the best
        // action for this state.
        tuner
            .q_table
            .insert((state.to_string(), "increase_memory".to_string()), 10.0);
        tuner
            .q_table
            .insert((state.to_string(), "decrease_memory".to_string()), -5.0);
        tuner
            .q_table
            .insert((state.to_string(), "no_change".to_string()), 0.0);

        let config = OptimizationConfig {
            processing_strategy: ProcessingStrategy::Standard,
            memory_limit_mb: 1000,
            use_robust: false,
            use_parallel: false,
            use_simd: false,
            use_gpu: false,
            chunk_size: 1024,
            num_threads: 1,
            algorithm_params: HashMap::new(),
        };

        let tuned = tuner
            .exploit_best_parameters(config.clone(), state)
            .expect("should succeed");

        assert_eq!(tuner.last_action, "increase_memory");
        assert!(
            tuned.memory_limit_mb > config.memory_limit_mb,
            "the learned best action ('increase_memory') must actually be applied, \
             not discarded: before={}, after={}",
            config.memory_limit_mb,
            tuned.memory_limit_mb
        );
    }

    #[test]
    fn update_q_values_distinguishes_between_different_actions() {
        let mut tuner = AdaptiveParameterTuner::new();
        tuner.current_state = "state_a".to_string();

        tuner.last_action = "increase_memory".to_string();
        tuner.update_q_values(0, 1.0).expect("should succeed");

        tuner.last_action = "decrease_memory".to_string();
        tuner.update_q_values(0, -1.0).expect("should succeed");

        let increase_value = tuner
            .q_table
            .get(&("state_a".to_string(), "increase_memory".to_string()))
            .copied();
        let decrease_value = tuner
            .q_table
            .get(&("state_a".to_string(), "decrease_memory".to_string()))
            .copied();

        assert!(
            increase_value.is_some() && decrease_value.is_some(),
            "each distinct action taken must get its own Q-table entry, not \
             collapse onto a single hardcoded key: q_table={:?}",
            tuner.q_table
        );
        assert_ne!(
            increase_value, decrease_value,
            "different rewards for different actions must be tracked separately"
        );
        // The old code hardcoded every entry onto exactly one key
        // ("state", "current_action"); with two real distinct actions taken
        // there must now be (at least) two entries.
        assert!(tuner.q_table.len() >= 2);
    }

    #[test]
    fn explore_parameters_records_a_real_named_action() {
        let mut tuner = AdaptiveParameterTuner::new();
        let config = OptimizationConfig {
            processing_strategy: ProcessingStrategy::Standard,
            memory_limit_mb: 1000,
            use_robust: false,
            use_parallel: false,
            use_simd: false,
            use_gpu: false,
            chunk_size: 1024,
            num_threads: 1,
            algorithm_params: HashMap::new(),
        };
        tuner.explore_parameters(config).expect("should succeed");
        assert!(
            TUNER_ACTIONS.contains(&tuner.last_action.as_str()),
            "explore_parameters must record one of the real named actions, got {:?}",
            tuner.last_action
        );
    }
}
