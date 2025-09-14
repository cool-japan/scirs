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
    pub fn analyze(data: &ndarray::ArrayView2<f64>) -> Result<Self> {
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
        let records = self.performance_history.get_mut(transformation).unwrap();
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
}

/// ✅ Advanced MODE: ML-based configuration prediction
pub struct ConfigurationPredictor {
    /// Feature weights for different data characteristics
    #[allow(dead_code)]
    feature_weights: HashMap<String, f64>,
    /// Learning rate for online updates
    #[allow(dead_code)]
    learning_rate: f64,
    /// Prediction confidence threshold
    confidence_threshold: f64,
    /// Training sample count
    sample_count: usize,
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
        if self.config_predictor.sample_count % 100 == 0 {
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
        }
    }

    /// ✅ Advanced MODE: Update real-time system metrics
    pub fn update_metrics(&mut self) -> Result<()> {
        // In production, these would read from actual system APIs
        self.cpu_load = self.read_cpu_load()?;
        self.available_memory_bytes = self.read_available_memory()?;
        self.cache_miss_rate = self.read_cache_miss_rate()?;
        self.io_wait_percent = self.read_io_wait()?;
        self.cpu_temperature_celsius = self.read_cpu_temperature()?;

        Ok(())
    }

    fn read_cpu_load(&self) -> Result<f64> {
        // Simplified implementation - in practice, read from /proc/loadavg or similar
        Ok(0.5) // Placeholder
    }

    fn read_available_memory(&self) -> Result<usize> {
        // Simplified implementation - in practice, read from /proc/meminfo
        Ok(8 * 1024 * 1024 * 1024) // 8GB placeholder
    }

    fn read_cache_miss_rate(&self) -> Result<f64> {
        // Simplified implementation - in practice, read from perf counters
        Ok(0.05) // 5% cache miss rate placeholder
    }

    fn read_io_wait(&self) -> Result<f64> {
        // Simplified implementation - in practice, read from /proc/stat
        Ok(0.02) // 2% I/O wait placeholder
    }

    fn read_cpu_temperature(&self) -> Result<f64> {
        // Simplified implementation - in practice, read from thermal zones
        Ok(55.0) // 55°C placeholder
    }
}

impl Default for ConfigurationPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl ConfigurationPredictor {
    /// Create new configuration predictor
    pub fn new() -> Self {
        let mut feature_weights = HashMap::new();
        feature_weights.insert("n_samples".to_string(), 0.3);
        feature_weights.insert("nfeatures".to_string(), 0.25);
        feature_weights.insert("memory_footprint".to_string(), 0.2);
        feature_weights.insert("sparsity".to_string(), 0.15);
        feature_weights.insert("cpu_load".to_string(), 0.1);

        ConfigurationPredictor {
            feature_weights,
            learning_rate: 0.01,
            confidence_threshold: 0.8,
            sample_count: 0,
        }
    }

    /// Predict optimal configuration using ML model
    pub fn predict_optimal_config(
        &self,
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
            use_gpu: features.get("memory_footprint").unwrap_or(&0.0) > &100.0,
            chunk_size: if predicted_memory_limit < 1000 {
                512
            } else {
                2048
            },
            num_threads: if predicted_parallelism { 4 } else { 1 },
            algorithm_params: HashMap::new(),
        })
    }

    /// Extract numerical features from state string
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
        let memory_footprint = features.get("memory_footprint").unwrap_or(&100.0);
        (memory_footprint * 1.5) as usize
    }

    fn predict_parallelism(&self, features: &HashMap<String, f64>) -> bool {
        let samples = features.get("samples").unwrap_or(&1000.0);
        let cpu_load = features.get("cpu").unwrap_or(&0.5);
        samples > &5000.0 && cpu_load < &0.7
    }

    fn predict_simd_usage(&self, features: &HashMap<String, f64>) -> bool {
        let features_count = features.get("features").unwrap_or(&10.0);
        features_count > &50.0
    }

    /// Update model from performance feedback
    pub fn update_from_feedback(&mut self, performance: &PerformanceMetric) -> Result<()> {
        self.sample_count += 1;
        // In practice, this would update model weights based on _performance
        Ok(())
    }

    /// Retrain model with historical data
    pub fn retrain_with_history(
        &mut self,
        history: &HashMap<String, Vec<PerformanceMetric>>,
    ) -> Result<()> {
        // In practice, this would perform full model retraining
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
        if rand::rng().gen_range(0.0..1.0) < self.exploration_rate {
            // Explore: randomly adjust parameters
            config = self.explore_parameters(config)?;
        } else {
            // Exploit: use best known parameters from Q-table
            config = self.exploit_best_parameters(config, state)?;
        }

        Ok(config)
    }

    /// Explore by randomly adjusting parameters
    fn explore_parameters(&self, mut config: OptimizationConfig) -> Result<OptimizationConfig> {
        let mut rng = rand::rng();

        // Randomly adjust memory limit (±20%)
        let memory_factor = rng.gen_range(0.8..1.2);
        config.memory_limit_mb = (config.memory_limit_mb as f64 * memory_factor) as usize;

        // Randomly toggle parallelism
        if rng.gen_range(0.0..1.0) < 0.3 {
            config.use_parallel = !config.use_parallel;
        }

        // Randomly adjust chunk size (±50%)
        let chunk_factor = rng.gen_range(0.5..1.5);
        config.chunk_size = (config.chunk_size as f64 * chunk_factor) as usize;

        Ok(config)
    }

    /// Exploit best known parameters from Q-table
    fn exploit_best_parameters(
        &self,
        config: OptimizationConfig,
        state: &str,
    ) -> Result<OptimizationConfig> {
        // Find best action for current state from Q-table
        let _best_action = self.find_best_action(state);

        // In practice, this would apply the best known parameter adjustments
        // For now, return the original config
        Ok(config)
    }

    /// Find best action for given state
    fn find_best_action(&self, state: &str) -> String {
        let mut best_action = "default".to_string();
        let mut best_value = f64::NEG_INFINITY;

        for ((s, action), &value) in &self.q_table {
            if s == state && value > best_value {
                best_value = value;
                best_action = action.clone();
            }
        }

        best_action
    }

    /// Update Q-values based on reward
    pub fn update_q_values(&mut self, confighash: u64, reward: f64) -> Result<()> {
        let state_action = (self.current_state.clone(), "current_action".to_string());

        // Q-learning update rule
        let old_value = self.q_table.get(&state_action).unwrap_or(&0.0);
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
    use ndarray::Array2;

    #[test]
    fn test_system_resources_detection() {
        let resources = SystemResources::detect();
        assert!(resources.cpu_cores > 0);
        assert!(resources.memory_mb > 0);
        assert!(resources.safe_memory_mb() < resources.memory_mb);
    }

    #[test]
    fn test_data_characteristics_analysis() {
        let data =
            Array2::from_shape_vec((100, 10), (0..1000).map(|x| x as f64).collect()).unwrap();
        let chars = DataCharacteristics::analyze(&data.view()).unwrap();

        assert_eq!(chars.n_samples, 100);
        assert_eq!(chars.nfeatures, 10);
        assert!(chars.memory_footprint_mb > 0.0);
        assert!(!chars.is_large_dataset());
    }

    #[test]
    fn test_optimization_config_for_standardization() {
        let data = Array2::ones((1000, 50));
        let chars = DataCharacteristics::analyze(&data.view()).unwrap();
        let system = SystemResources::detect();

        let config = OptimizationConfig::for_standardization(&chars, &system);
        assert!(config.memory_limit_mb > 0);
    }

    #[test]
    fn test_optimization_config_for_pca() {
        let data = Array2::ones((500, 20));
        let chars = DataCharacteristics::analyze(&data.view()).unwrap();
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
        let chars = DataCharacteristics::analyze(&data.view()).unwrap();

        let config = tuner
            .optimize_for_transformation("standardization", &chars, &HashMap::new())
            .unwrap();
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
}
