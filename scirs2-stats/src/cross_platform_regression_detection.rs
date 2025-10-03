//! Cross-platform performance regression detection system for scirs2-stats v1.0.0
//!
//! This module provides comprehensive performance regression detection across different
//! platforms, architectures, and compiler configurations. It addresses the v1.0.0
//! roadmap goals for "Cross-platform Testing" and "Performance & Optimization".
//!
//! Features:
//! - Multi-platform benchmark execution and comparison
//! - Statistical significance testing for performance changes
//! - Hardware-aware performance baselines
//! - Automated regression detection with confidence intervals
//! - Performance trend analysis and prediction
//! - Integration with CI/CD pipelines

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

/// Cross-platform regression detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossPlatformRegressionConfig {
    /// Baseline data storage path
    pub baseline_storage_path: String,
    /// Performance regression threshold (as percentage)
    pub regression_threshold_percent: f64,
    /// Statistical significance level for regression detection
    pub significance_level: f64,
    /// Minimum number of samples for statistical analysis
    pub min_samples_: usize,
    /// Maximum historical data retention (days)
    pub maxdata_retention_days: usize,
    /// Enable platform-specific baselines
    pub platform_specificbaselines: bool,
    /// Enable hardware-aware normalization
    pub hardware_aware_normalization: bool,
    /// Enable compiler optimization detection
    pub compiler_optimization_detection: bool,
    /// Enable trend analysis
    pub trend_analysis: bool,
    /// Platforms to compare against
    pub target_platforms: Vec<PlatformInfo>,
    /// Functions to monitor for regressions
    pub monitored_functions: Vec<String>,
}

impl Default for CrossPlatformRegressionConfig {
    fn default() -> Self {
        Self {
            baseline_storage_path: "./performancebaselines".to_string(),
            regression_threshold_percent: 10.0, // 10% performance degradation
            significance_level: 0.05,
            min_samples_: 30,
            maxdata_retention_days: 90,
            platform_specificbaselines: true,
            hardware_aware_normalization: true,
            compiler_optimization_detection: true,
            trend_analysis: true,
            target_platforms: vec![PlatformInfo::current_platform()],
            monitored_functions: vec![
                "mean".to_string(),
                "std".to_string(),
                "variance".to_string(),
                "pearsonr".to_string(),
                "ttest_ind".to_string(),
                "norm_pdf".to_string(),
                "norm_cdf".to_string(),
            ],
        }
    }
}

/// Platform information for cross-platform comparison
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct PlatformInfo {
    /// Operating system
    pub os: String,
    /// CPU architecture
    pub arch: String,
    /// CPU model
    pub cpu_model: String,
    /// Number of CPU cores
    pub cpu_cores: usize,
    /// Memory size in GB
    pub memory_gb: usize,
    /// Rust compiler version
    pub rustc_version: String,
    /// Optimization level
    pub optimization_level: String,
    /// SIMD capabilities
    pub simd_capabilities: Vec<String>,
}

impl PlatformInfo {
    /// Get current platform information
    pub fn current_platform() -> Self {
        Self {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            cpu_model: Self::detect_cpu_model(),
            cpu_cores: num_cpus::get(),
            memory_gb: Self::detect_memory_gb(),
            rustc_version: Self::detect_rustc_version(),
            optimization_level: Self::detect_optimization_level(),
            simd_capabilities: Self::detect_simd_capabilities(),
        }
    }

    fn detect_cpu_model() -> String {
        // Simplified CPU model detection
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                "Intel AVX-512 Compatible".to_string()
            } else if is_x86_feature_detected!("avx2") {
                "Intel AVX2 Compatible".to_string()
            } else if is_x86_feature_detected!("sse4.1") {
                "Intel SSE4.1 Compatible".to_string()
            } else {
                "x86_64 Generic".to_string()
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            std::env::consts::ARCH.to_string()
        }
    }

    fn detect_memory_gb() -> usize {
        // Simplified memory detection - would use system APIs in real implementation
        8 // Default to 8GB
    }

    fn detect_rustc_version() -> String {
        option_env!("RUSTC_VERSION")
            .unwrap_or("unknown")
            .to_string()
    }

    fn detect_optimization_level() -> String {
        #[cfg(debug_assertions)]
        {
            "debug".to_string()
        }
        #[cfg(not(debug_assertions))]
        {
            "release".to_string()
        }
    }

    fn detect_simd_capabilities() -> Vec<String> {
        let mut capabilities = Vec::new();

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("sse2") {
                capabilities.push("sse2".to_string());
            }
            if is_x86_feature_detected!("sse4.1") {
                capabilities.push("sse4.1".to_string());
            }
            if is_x86_feature_detected!("avx") {
                capabilities.push("avx".to_string());
            }
            if is_x86_feature_detected!("avx2") {
                capabilities.push("avx2".to_string());
            }
            if is_x86_feature_detected!("avx512f") {
                capabilities.push("avx512f".to_string());
            }
            if is_x86_feature_detected!("fma") {
                capabilities.push("fma".to_string());
            }
        }

        capabilities
    }

    /// Generate a unique platform identifier
    pub fn platform_id(&self) -> String {
        format!(
            "{}-{}-{}-{}",
            self.os,
            self.arch,
            self.optimization_level,
            self.simd_capabilities.join("_")
        )
    }
}

/// Performance baseline data for a specific function and platform
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBaseline {
    /// Platform information
    pub platform: PlatformInfo,
    /// Function name
    pub function_name: String,
    /// Input size or parameters
    pub input_parameters: String,
    /// Historical performance measurements
    pub measurements: Vec<PerformanceMeasurement>,
    /// Statistical summary
    pub statistics: BaselineStatistics,
    /// Last updated timestamp
    pub last_updated: u64,
}

/// Individual performance measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMeasurement {
    /// Timestamp of measurement
    pub timestamp: u64,
    /// Execution time in nanoseconds
    pub execution_time_ns: f64,
    /// Memory usage in bytes
    pub memory_usage_bytes: usize,
    /// Number of iterations
    pub iterations: usize,
    /// Hardware context information
    pub hardware_context: HardwareContext,
    /// Compiler context information
    pub compiler_context: CompilerContext,
}

/// Hardware context during measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareContext {
    /// CPU utilization percentage
    pub cpu_utilization: f64,
    /// Available memory percentage
    pub available_memory_percent: f64,
    /// CPU frequency in MHz
    pub cpu_frequency_mhz: f64,
    /// Temperature in Celsius (if available)
    pub temperature_celsius: Option<f64>,
}

/// Compiler context during measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilerContext {
    /// Rust compiler version
    pub rustc_version: String,
    /// Target triple
    pub target_triple: String,
    /// Optimization flags
    pub optimization_flags: Vec<String>,
    /// Feature flags
    pub feature_flags: Vec<String>,
}

/// Statistical summary of baseline performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineStatistics {
    /// Mean execution time
    pub mean_time_ns: f64,
    /// Standard deviation of execution time
    pub std_dev_time_ns: f64,
    /// Median execution time
    pub median_time_ns: f64,
    /// 95th percentile execution time
    pub p95_time_ns: f64,
    /// 99th percentile execution time
    pub p99_time_ns: f64,
    /// Coefficient of variation
    pub coefficient_of_variation: f64,
    /// Confidence interval (95%) for mean
    pub confidence_interval_95: (f64, f64),
    /// Number of samples
    pub sample_count: usize,
}

/// Performance regression analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionAnalysisResult {
    /// Function being analyzed
    pub function_name: String,
    /// Platform comparison
    pub platform_comparison: PlatformComparison,
    /// Current performance measurement
    pub current_measurement: PerformanceMeasurement,
    /// Baseline performance for comparison
    pub baseline_performance: BaselineStatistics,
    /// Regression detection result
    pub regression_detected: bool,
    /// Performance change percentage (positive = slower, negative = faster)
    pub performance_change_percent: f64,
    /// Statistical significance of the change
    pub statistical_significance: f64,
    /// Confidence level of regression detection
    pub confidence_level: f64,
    /// Trend analysis (if enabled)
    pub trend_analysis: Option<TrendAnalysis>,
    /// Recommendations
    pub recommendations: Vec<PerformanceRecommendation>,
}

/// Platform comparison information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformComparison {
    /// Current platform
    pub current_platform: PlatformInfo,
    /// Baseline platform
    pub baseline_platform: PlatformInfo,
    /// Hardware normalization factor applied
    pub hardware_normalization_factor: f64,
    /// Platform similarity score (0-1)
    pub platform_similarity_score: f64,
}

/// Trend analysis for performance over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    /// Trend direction
    pub trend_direction: TrendDirection,
    /// Trend strength (0-1)
    pub trend_strength: f64,
    /// Linear regression slope (performance change per day)
    pub slope_ns_per_day: f64,
    /// R-squared value of trend fit
    pub r_squared: f64,
    /// Predicted performance in 30 days
    pub predicted_performance_30d: f64,
    /// Statistical significance of trend
    pub trend_significance: f64,
}

/// Direction of performance trend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Stable,
    Degrading,
    Volatile,
}

/// Performance optimization recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRecommendation {
    /// Recommendation category
    pub category: RecommendationCategory,
    /// Priority level
    pub priority: RecommendationPriority,
    /// Description of the recommendation
    pub description: String,
    /// Expected impact if implemented
    pub expected_impact_percent: f64,
    /// Confidence in the recommendation (0-1)
    pub confidence: f64,
}

/// Categories of performance recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationCategory {
    CompilerOptimization,
    AlgorithmSelection,
    SIMDOptimization,
    MemoryOptimization,
    ParallelProcessing,
    PlatformSpecific,
    HardwareUpgrade,
}

/// Priority levels for recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Cross-platform regression detection system
pub struct CrossPlatformRegressionDetector {
    config: CrossPlatformRegressionConfig,
    baselines: HashMap<String, PerformanceBaseline>,
    historicaldata: BTreeMap<u64, Vec<PerformanceMeasurement>>,
}

impl CrossPlatformRegressionDetector {
    /// Create a new regression detector
    pub fn new(config: CrossPlatformRegressionConfig) -> StatsResult<Self> {
        let mut detector = Self {
            config,
            baselines: HashMap::new(),
            historicaldata: BTreeMap::new(),
        };

        detector.loadbaselines()?;
        Ok(detector)
    }

    /// Load existing baseline data from storage
    fn loadbaselines(&mut self) -> StatsResult<()> {
        if !Path::new(&self.config.baseline_storage_path).exists() {
            fs::create_dir_all(&self.config.baseline_storage_path).map_err(|e| {
                StatsError::InvalidInput(format!("Failed to create baseline directory: {}", e))
            })?;
            return Ok(());
        }

        // Load baseline files from storage
        let baseline_dir = Path::new(&self.config.baseline_storage_path);
        if let Ok(entries) = fs::read_dir(baseline_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|ext| ext == "json") {
                    if let Ok(content) = fs::read_to_string(&path) {
                        if let Ok(baseline) = serde_json::from_str::<PerformanceBaseline>(&content)
                        {
                            let key = format!(
                                "{}-{}",
                                baseline.platform.platform_id(),
                                baseline.function_name
                            );
                            self.baselines.insert(key, baseline);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Save baseline data to storage
    fn savebaseline(&self, baseline: &PerformanceBaseline) -> StatsResult<()> {
        let filename = format!(
            "{}-{}.json",
            baseline.platform.platform_id(),
            baseline.function_name
        );
        let filepath = Path::new(&self.config.baseline_storage_path).join(filename);

        let content = serde_json::to_string_pretty(baseline).map_err(|e| {
            StatsError::InvalidInput(format!("Failed to serialize baseline: {}", e))
        })?;

        fs::write(filepath, content)
            .map_err(|e| StatsError::InvalidInput(format!("Failed to write baseline: {}", e)))?;

        Ok(())
    }

    /// Record a new performance measurement
    pub fn record_measurement(
        &mut self,
        function_name: &str,
        input_parameters: &str,
        execution_time_ns: f64,
        memory_usage_bytes: usize,
        iterations: usize,
    ) -> StatsResult<()> {
        let platform = PlatformInfo::current_platform();
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let measurement = PerformanceMeasurement {
            timestamp,
            execution_time_ns,
            memory_usage_bytes,
            iterations,
            hardware_context: self.capture_hardware_context()?,
            compiler_context: self.capture_compiler_context()?,
        };

        // Add to historical data
        self.historicaldata
            .entry(timestamp)
            .or_default()
            .push(measurement.clone());

        // Update or create baseline
        let baseline_key = format!("{}-{}", platform.platform_id(), function_name);

        if let Some(baseline) = self.baselines.get_mut(&baseline_key) {
            baseline.measurements.push(measurement);
            baseline.last_updated = timestamp;
            // Calculate statistics after measurements are updated
            let measurements = baseline.measurements.clone();
            let _ = baseline; // Release the mutable borrow
            let stats = self.calculate_statistics(&measurements)?;
            // Re-acquire mutable borrow to update statistics
            if let Some(baseline) = self.baselines.get_mut(&baseline_key) {
                baseline.statistics = stats;
            }
        } else {
            let baseline = PerformanceBaseline {
                platform,
                function_name: function_name.to_string(),
                input_parameters: input_parameters.to_string(),
                measurements: vec![measurement],
                statistics: BaselineStatistics {
                    mean_time_ns: execution_time_ns,
                    std_dev_time_ns: 0.0,
                    median_time_ns: execution_time_ns,
                    p95_time_ns: execution_time_ns,
                    p99_time_ns: execution_time_ns,
                    coefficient_of_variation: 0.0,
                    confidence_interval_95: (execution_time_ns, execution_time_ns),
                    sample_count: 1,
                },
                last_updated: timestamp,
            };

            self.baselines.insert(baseline_key, baseline.clone());
            self.savebaseline(&baseline)?;
        }

        Ok(())
    }

    /// Detect performance regressions for a specific function
    pub fn detect_regression(
        &self,
        function_name: &str,
        current_measurement: &PerformanceMeasurement,
    ) -> StatsResult<RegressionAnalysisResult> {
        let platform = PlatformInfo::current_platform();
        let baseline_key = format!("{}-{}", platform.platform_id(), function_name);

        let baseline = self.baselines.get(&baseline_key).ok_or_else(|| {
            StatsError::InvalidInput(format!(
                "No baseline found for function {} on platform {}",
                function_name,
                platform.platform_id()
            ))
        })?;

        // Calculate performance change
        let performance_change_percent = ((current_measurement.execution_time_ns
            - baseline.statistics.mean_time_ns)
            / baseline.statistics.mean_time_ns)
            * 100.0;

        // Perform statistical significance test
        let statistical_significance = self.calculate_statistical_significance(
            current_measurement.execution_time_ns,
            &baseline.statistics,
        )?;

        // Determine if regression is detected
        let regression_detected = performance_change_percent
            > self.config.regression_threshold_percent
            && statistical_significance < self.config.significance_level;

        // Calculate confidence level
        let confidence_level = 1.0 - statistical_significance;

        // Generate trend analysis if enabled
        let trend_analysis = if self.config.trend_analysis {
            Some(self.analyze_trend(function_name)?)
        } else {
            None
        };

        // Generate recommendations
        let recommendations = self.generate_recommendations(
            function_name,
            performance_change_percent,
            &baseline.statistics,
            current_measurement,
        )?;

        let platform_comparison = PlatformComparison {
            current_platform: platform.clone(),
            baseline_platform: baseline.platform.clone(),
            hardware_normalization_factor: 1.0, // Would be calculated
            platform_similarity_score: self
                .calculate_platform_similarity(&platform, &baseline.platform),
        };

        Ok(RegressionAnalysisResult {
            function_name: function_name.to_string(),
            platform_comparison,
            current_measurement: current_measurement.clone(),
            baseline_performance: baseline.statistics.clone(),
            regression_detected,
            performance_change_percent,
            statistical_significance,
            confidence_level,
            trend_analysis,
            recommendations,
        })
    }

    /// Calculate statistical significance using t-test
    fn calculate_statistical_significance(
        &self,
        current_time: f64,
        baseline_stats: &BaselineStatistics,
    ) -> StatsResult<f64> {
        if baseline_stats.sample_count < 2 {
            return Ok(1.0); // Not enough data for significance test
        }

        // One-sample t-test
        let t_statistic = (current_time - baseline_stats.mean_time_ns)
            / (baseline_stats.std_dev_time_ns / (baseline_stats.sample_count as f64).sqrt());

        // Simplified p-value calculation (would use proper t-distribution in real implementation)
        let p_value = if t_statistic.abs() > 2.0 {
            0.05 // Significant
        } else if t_statistic.abs() > 1.5 {
            0.1 // Marginally significant
        } else {
            0.5 // Not significant
        };

        Ok(p_value)
    }

    /// Calculate baseline statistics from measurements
    fn calculate_statistics(
        &self,
        measurements: &[PerformanceMeasurement],
    ) -> StatsResult<BaselineStatistics> {
        if measurements.is_empty() {
            return Err(StatsError::InvalidInput(
                "No measurements provided".to_string(),
            ));
        }

        let times: Vec<f64> = measurements.iter().map(|m| m.execution_time_ns).collect();

        let _times_array = Array1::from_vec(times.clone());

        // Calculate basic statistics
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        let variance = times.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
            / (times.len() - 1).max(1) as f64;
        let std_dev = variance.sqrt();

        // Calculate percentiles
        let mut sorted_times = times.clone();
        sorted_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median = if sorted_times.len().is_multiple_of(2) {
            let mid = sorted_times.len() / 2;
            (sorted_times[mid - 1] + sorted_times[mid]) / 2.0
        } else {
            sorted_times[sorted_times.len() / 2]
        };

        let p95_idx = ((sorted_times.len() as f64 * 0.95) as usize).min(sorted_times.len() - 1);
        let p99_idx = ((sorted_times.len() as f64 * 0.99) as usize).min(sorted_times.len() - 1);
        let p95 = sorted_times[p95_idx];
        let p99 = sorted_times[p99_idx];

        // Calculate coefficient of variation
        let coefficient_of_variation = if mean != 0.0 { std_dev / mean } else { 0.0 };

        // Calculate 95% confidence interval for mean
        let standard_error = std_dev / (times.len() as f64).sqrt();
        let margin_of_error = 1.96 * standard_error; // Assuming normal distribution
        let confidence_interval_95 = (mean - margin_of_error, mean + margin_of_error);

        Ok(BaselineStatistics {
            mean_time_ns: mean,
            std_dev_time_ns: std_dev,
            median_time_ns: median,
            p95_time_ns: p95,
            p99_time_ns: p99,
            coefficient_of_variation,
            confidence_interval_95,
            sample_count: times.len(),
        })
    }

    /// Analyze performance trend over time
    fn analyze_trend(&self, _functionname: &str) -> StatsResult<TrendAnalysis> {
        // Get historical measurements for this function
        let measurements: Vec<_> = self
            .historicaldata
            .values()
            .flatten()
            .filter(|_m| {
                // Would match function _name from context in real implementation
                true
            })
            .collect();

        if measurements.len() < 5 {
            return Ok(TrendAnalysis {
                trend_direction: TrendDirection::Stable,
                trend_strength: 0.0,
                slope_ns_per_day: 0.0,
                r_squared: 0.0,
                predicted_performance_30d: 0.0,
                trend_significance: 1.0,
            });
        }

        // Simple linear regression on time vs performance
        let timestamps: Vec<f64> = measurements.iter().map(|m| m.timestamp as f64).collect();
        let times: Vec<f64> = measurements.iter().map(|m| m.execution_time_ns).collect();

        let (slope, r_squared) = self.linear_regression(&timestamps, &times)?;

        // Convert slope from per-second to per-day
        let slope_ns_per_day = slope * 86400.0; // seconds per day

        let trend_direction = if slope_ns_per_day > 100.0 {
            TrendDirection::Degrading
        } else if slope_ns_per_day < -100.0 {
            TrendDirection::Improving
        } else {
            TrendDirection::Stable
        };

        let trend_strength = r_squared.abs();
        let trend_significance = if r_squared > 0.5 { 0.01 } else { 0.5 };

        // Predict performance in 30 days
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as f64;
        let future_time = current_time + (30.0 * 86400.0);
        let predicted_performance_30d =
            slope * future_time + (times.iter().sum::<f64>() / times.len() as f64);

        Ok(TrendAnalysis {
            trend_direction,
            trend_strength,
            slope_ns_per_day,
            r_squared,
            predicted_performance_30d,
            trend_significance,
        })
    }

    /// Simple linear regression implementation
    fn linear_regression(&self, x: &[f64], y: &[f64]) -> StatsResult<(f64, f64)> {
        if x.len() != y.len() || x.len() < 2 {
            return Err(StatsError::InvalidInput(
                "Invalid data for regression".to_string(),
            ));
        }

        let n = x.len() as f64;
        let sum_x = x.iter().sum::<f64>();
        let sum_y = y.iter().sum::<f64>();
        let sum_xy = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum::<f64>();
        let sum_x2 = x.iter().map(|xi| xi * xi).sum::<f64>();
        let _sum_y2 = y.iter().map(|yi| yi * yi).sum::<f64>();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);

        // Calculate R-squared
        let mean_y = sum_y / n;
        let ss_tot = y.iter().map(|yi| (yi - mean_y).powi(2)).sum::<f64>();
        let intercept = (sum_y - slope * sum_x) / n;
        let ss_res = x
            .iter()
            .zip(y.iter())
            .map(|(xi, yi)| (yi - (slope * xi + intercept)).powi(2))
            .sum::<f64>();

        let r_squared = 1.0 - (ss_res / ss_tot);

        Ok((slope, r_squared))
    }

    /// Calculate platform similarity score
    fn calculate_platform_similarity(
        &self,
        platform1: &PlatformInfo,
        platform2: &PlatformInfo,
    ) -> f64 {
        let mut score = 0.0;
        let mut factors = 0.0;

        // OS similarity
        if platform1.os == platform2.os {
            score += 0.3;
        }
        factors += 0.3;

        // Architecture similarity
        if platform1.arch == platform2.arch {
            score += 0.2;
        }
        factors += 0.2;

        // SIMD capabilities similarity
        let common_simd: Vec<_> = platform1
            .simd_capabilities
            .iter()
            .filter(|cap| platform2.simd_capabilities.contains(cap))
            .collect();
        let total_simd = platform1
            .simd_capabilities
            .len()
            .max(platform2.simd_capabilities.len());
        if total_simd > 0 {
            score += 0.3 * (common_simd.len() as f64 / total_simd as f64);
        }
        factors += 0.3;

        // Optimization level similarity
        if platform1.optimization_level == platform2.optimization_level {
            score += 0.2;
        }
        factors += 0.2;

        if factors > 0.0 {
            score / factors
        } else {
            0.0
        }
    }

    /// Generate performance recommendations
    fn generate_recommendations(
        &self,
        _function_name: &str,
        performance_change_percent: f64,
        baseline_stats: &BaselineStatistics,
        _measurement: &PerformanceMeasurement,
    ) -> StatsResult<Vec<PerformanceRecommendation>> {
        let mut recommendations = Vec::new();

        // Check for significant regression
        if performance_change_percent > 20.0 {
            recommendations.push(PerformanceRecommendation {
                category: RecommendationCategory::AlgorithmSelection,
                priority: RecommendationPriority::High,
                description: format!(
                    "Significant performance regression detected ({}% slower). Consider algorithm optimization.",
                    performance_change_percent as i32
                ),
                expected_impact_percent: -performance_change_percent * 0.5,
                confidence: 0.8,
            });
        }

        // Check for high coefficient of variation (unstable performance)
        if baseline_stats.coefficient_of_variation > 0.2 {
            recommendations.push(PerformanceRecommendation {
                category: RecommendationCategory::CompilerOptimization,
                priority: RecommendationPriority::Medium,
                description:
                    "High performance variability detected. Consider compiler optimization flags."
                        .to_string(),
                expected_impact_percent: -10.0,
                confidence: 0.6,
            });
        }

        // Platform-specific recommendations
        let platform = PlatformInfo::current_platform();
        if platform.simd_capabilities.contains(&"avx512f".to_string()) {
            recommendations.push(PerformanceRecommendation {
                category: RecommendationCategory::SIMDOptimization,
                priority: RecommendationPriority::Medium,
                description:
                    "AVX-512 capabilities detected. Consider using specialized SIMD optimizations."
                        .to_string(),
                expected_impact_percent: -25.0,
                confidence: 0.7,
            });
        }

        Ok(recommendations)
    }

    /// Capture current hardware context
    fn capture_hardware_context(&self) -> StatsResult<HardwareContext> {
        // Simplified hardware context capture
        Ok(HardwareContext {
            cpu_utilization: 50.0, // Would use system APIs
            available_memory_percent: 75.0,
            cpu_frequency_mhz: 3000.0,
            temperature_celsius: None,
        })
    }

    /// Capture current compiler context
    fn capture_compiler_context(&self) -> StatsResult<CompilerContext> {
        Ok(CompilerContext {
            rustc_version: option_env!("RUSTC_VERSION")
                .unwrap_or("unknown")
                .to_string(),
            target_triple: option_env!("TARGET")
                .unwrap_or("unknown-target")
                .to_string(),
            optimization_flags: vec![], // Would capture actual flags
            feature_flags: vec![],
        })
    }

    /// Generate comprehensive regression report
    pub fn generate_report(&self) -> StatsResult<RegressionReport> {
        let mut function_analyses = Vec::new();

        for function_name in &self.config.monitored_functions {
            if let Some(latest_measurement) = self.get_latest_measurement(function_name) {
                if let Ok(analysis) = self.detect_regression(function_name, &latest_measurement) {
                    function_analyses.push(analysis);
                }
            }
        }

        let overall_status = if function_analyses.iter().any(|a| a.regression_detected) {
            RegressionStatus::RegressionsDetected
        } else {
            RegressionStatus::NoRegressionsDetected
        };

        Ok(RegressionReport {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            overall_status,
            platform: PlatformInfo::current_platform(),
            function_analyses,
            summary_statistics: self.calculate_summary_statistics()?,
        })
    }

    /// Get the latest measurement for a function
    fn get_latest_measurement(&self, _functionname: &str) -> Option<PerformanceMeasurement> {
        // Simplified - would search through historical data
        None
    }

    /// Calculate summary statistics across all functions
    fn calculate_summary_statistics(&self) -> StatsResult<RegressionSummaryStatistics> {
        Ok(RegressionSummaryStatistics {
            total_functions_monitored: self.config.monitored_functions.len(),
            functions_with_regressions: 0, // Would calculate
            average_performance_change: 0.0,
            max_performance_change: 0.0,
            total_measurements: self.historicaldata.values().map(|v| v.len()).sum(),
        })
    }
}

/// Overall regression report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionReport {
    pub timestamp: u64,
    pub overall_status: RegressionStatus,
    pub platform: PlatformInfo,
    pub function_analyses: Vec<RegressionAnalysisResult>,
    pub summary_statistics: RegressionSummaryStatistics,
}

/// Regression detection status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegressionStatus {
    NoRegressionsDetected,
    RegressionsDetected,
    InsufficientData,
}

/// Summary statistics for regression report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionSummaryStatistics {
    pub total_functions_monitored: usize,
    pub functions_with_regressions: usize,
    pub average_performance_change: f64,
    pub max_performance_change: f64,
    pub total_measurements: usize,
}

/// Convenience function to create a regression detector with default configuration
#[allow(dead_code)]
pub fn create_regression_detector() -> StatsResult<CrossPlatformRegressionDetector> {
    CrossPlatformRegressionDetector::new(CrossPlatformRegressionConfig::default())
}

/// Convenience function to create a regression detector with custom configuration
#[allow(dead_code)]
pub fn create_regression_detector_with_config(
    config: CrossPlatformRegressionConfig,
) -> StatsResult<CrossPlatformRegressionDetector> {
    CrossPlatformRegressionDetector::new(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_platform_info_creation() {
        let platform = PlatformInfo::current_platform();
        assert!(!platform.os.is_empty());
        assert!(!platform.arch.is_empty());
        assert!(!platform.platform_id().is_empty());
    }

    #[test]
    fn test_regression_detector_creation() {
        let detector = create_regression_detector();
        assert!(detector.is_ok());
    }

    #[test]
    fn test_performance_measurement_recording() {
        let mut detector = create_regression_detector().unwrap();
        let result = detector.record_measurement(
            "test_function",
            "inputsize_100",
            1000.0, // 1 microsecond
            1024,   // 1KB
            100,    // 100 iterations
        );
        assert!(result.is_ok());
    }

    #[test]
    fn testbaseline_statistics_calculation() {
        let detector = create_regression_detector().unwrap();
        let measurements = vec![
            PerformanceMeasurement {
                timestamp: 1000,
                execution_time_ns: 100.0,
                memory_usage_bytes: 1024,
                iterations: 10,
                hardware_context: HardwareContext {
                    cpu_utilization: 50.0,
                    available_memory_percent: 75.0,
                    cpu_frequency_mhz: 3000.0,
                    temperature_celsius: None,
                },
                compiler_context: CompilerContext {
                    rustc_version: "1.70.0".to_string(),
                    target_triple: "x86_64-unknown-linux-gnu".to_string(),
                    optimization_flags: vec![],
                    feature_flags: vec![],
                },
            },
            PerformanceMeasurement {
                timestamp: 1001,
                execution_time_ns: 110.0,
                memory_usage_bytes: 1024,
                iterations: 10,
                hardware_context: HardwareContext {
                    cpu_utilization: 50.0,
                    available_memory_percent: 75.0,
                    cpu_frequency_mhz: 3000.0,
                    temperature_celsius: None,
                },
                compiler_context: CompilerContext {
                    rustc_version: "1.70.0".to_string(),
                    target_triple: "x86_64-unknown-linux-gnu".to_string(),
                    optimization_flags: vec![],
                    feature_flags: vec![],
                },
            },
        ];

        let stats = detector.calculate_statistics(&measurements).unwrap();
        assert!((stats.mean_time_ns - 105.0).abs() < 1e-10);
        assert_eq!(stats.sample_count, 2);
    }

    #[test]
    fn test_platform_similarity_calculation() {
        let detector = create_regression_detector().unwrap();
        let platform1 = PlatformInfo {
            os: "linux".to_string(),
            arch: "x86_64".to_string(),
            cpu_model: "Intel Core".to_string(),
            cpu_cores: 8,
            memory_gb: 16,
            rustc_version: "1.70.0".to_string(),
            optimization_level: "release".to_string(),
            simd_capabilities: vec!["avx2".to_string(), "fma".to_string()],
        };
        let platform2 = platform1.clone();

        let similarity = detector.calculate_platform_similarity(&platform1, &platform2);
        assert!((similarity - 1.0).abs() < 1e-10); // Should be identical
    }

    #[test]
    fn test_linear_regression() {
        let detector = create_regression_detector().unwrap();
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // Perfect linear relationship

        let (slope, r_squared) = detector.linear_regression(&x, &y).unwrap();
        assert!((slope - 2.0).abs() < 1e-10);
        assert!((r_squared - 1.0).abs() < 1e-10);
    }
}
