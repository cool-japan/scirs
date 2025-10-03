//! # Adaptive Advanced Optimizer - Dynamic Performance Tuning
//!
//! This module provides an adaptive optimization system that continuously monitors
//! and improves Advanced mode performance based on runtime characteristics.
//! It combines machine learning, hardware profiling, and adaptive algorithms
//! to achieve optimal performance for any workload.

use scirs2_core::ndarray::{Array1, Dimension};
use scirs2_core::numeric::Float;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use crate::advanced_fusion_algorithms::AdvancedConfig;
use crate::error::{NdimageError, NdimageResult};

/// Adaptive optimization system for Advanced mode operations
#[derive(Debug)]
pub struct AdaptiveAdvancedOptimizer {
    /// Performance history database
    performancehistory: Arc<RwLock<HashMap<String, VecDeque<PerformanceSnapshot>>>>,
    /// Machine learning model for performance prediction
    ml_predictor: Arc<Mutex<PerformancePredictionModel>>,
    /// Hardware characteristics profiler
    hardware_profiler: Arc<Mutex<HardwareProfiler>>,
    /// Adaptive parameter controller
    parameter_controller: Arc<Mutex<ParameterController>>,
    /// Real-time monitoring system
    monitor: Arc<Mutex<RealTimeMonitor>>,
    /// Configuration
    config: AdaptiveOptimizerConfig,
}

#[derive(Debug, Clone)]
pub struct AdaptiveOptimizerConfig {
    /// Learning rate for adaptive adjustments
    pub learning_rate: f64,
    /// History window size for trend analysis
    pub history_window_size: usize,
    /// Minimum improvement threshold for parameter changes
    pub improvement_threshold: f64,
    /// Maximum parameter adjustment per iteration
    pub max_adjustment_rate: f64,
    /// Enable predictive optimization
    pub enable_prediction: bool,
    /// Monitoring sampling rate (Hz)
    pub monitoring_rate: f64,
}

impl Default for AdaptiveOptimizerConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            history_window_size: 100,
            improvement_threshold: 0.05, // 5% improvement threshold
            max_adjustment_rate: 0.1,    // 10% max adjustment
            enable_prediction: true,
            monitoring_rate: 1000.0, // 1kHz monitoring
        }
    }
}

/// Performance snapshot for trend analysis
#[derive(Debug, Clone)]
pub struct PerformanceSnapshot {
    /// Timestamp of measurement
    pub timestamp: Instant,
    /// Operation type identifier
    pub operation_type: String,
    /// Input data characteristics
    pub data_characteristics: DataCharacteristics,
    /// Performance metrics
    pub metrics: PerformanceMetrics,
    /// Configuration used
    pub config_used: AdvancedConfig,
}

#[derive(Debug, Clone)]
pub struct DataCharacteristics {
    /// Data dimensions
    pub dimensions: Vec<usize>,
    /// Data type size
    pub element_size: usize,
    /// Estimated complexity
    pub complexity_score: f64,
    /// Memory access pattern
    pub access_pattern: AccessPattern,
}

#[derive(Debug, Clone)]
pub enum AccessPattern {
    Sequential,
    Random,
    Strided { stride: usize },
    Blocked { block_size: usize },
}

#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Execution time
    pub execution_time: Duration,
    /// Memory throughput (bytes/sec)
    pub memory_throughput: f64,
    /// FLOPS achieved
    pub flops: f64,
    /// Cache hit ratio
    pub cache_hit_ratio: f64,
    /// CPU utilization
    pub cpu_utilization: f64,
    /// Power consumption (watts)
    pub power_consumption: f64,
}

/// Machine learning model for performance prediction
#[derive(Debug)]
pub struct PerformancePredictionModel {
    /// Feature weights for linear model
    feature_weights: Array1<f64>,
    /// Bias term
    bias: f64,
    /// Model accuracy history
    accuracyhistory: VecDeque<f64>,
    /// Training data
    training_data: Vec<(Array1<f64>, f64)>,
}

impl PerformancePredictionModel {
    pub fn new() -> Self {
        Self {
            feature_weights: Array1::zeros(16), // 16 features initially
            bias: 0.0,
            accuracyhistory: VecDeque::new(),
            training_data: Vec::new(),
        }
    }

    /// Predict performance based on data characteristics
    pub fn predict_performance(&self, features: &Array1<f64>) -> f64 {
        let prediction = features.dot(&self.feature_weights) + self.bias;
        prediction.max(0.0) // Ensure non-negative performance
    }

    /// Update model with new training data
    pub fn update_model(&mut self, features: Array1<f64>, target: f64) -> NdimageResult<()> {
        self.training_data.push((features, target));

        // Keep only recent training data to adapt to changing conditions
        if self.training_data.len() > 1000 {
            self.training_data.remove(0);
        }

        // Retrain model using gradient descent
        self.retrain_model()?;

        Ok(())
    }

    fn retrain_model(&mut self) -> NdimageResult<()> {
        if self.training_data.is_empty() {
            return Ok(());
        }

        let learning_rate = 0.001;
        let epochs = 10;

        for _ in 0..epochs {
            for (features, target) in &self.training_data {
                let prediction = self.predict_performance(features);
                let error = prediction - target;

                // Update weights using gradient descent
                for (i, weight) in self.feature_weights.iter_mut().enumerate() {
                    *weight -= learning_rate * error * features[i];
                }
                self.bias -= learning_rate * error;
            }
        }

        Ok(())
    }
}

/// Hardware characteristics profiler
#[derive(Debug)]
pub struct HardwareProfiler {
    /// CPU cache sizes (L1, L2, L3)
    cache_sizes: Vec<usize>,
    /// Memory bandwidth
    memory_bandwidth: f64,
    /// CPU frequency
    cpu_frequency: f64,
    /// Number of cores
    num_cores: usize,
    /// SIMD capabilities
    simd_capabilities: SIMDCapabilities,
}

#[derive(Debug)]
pub struct SIMDCapabilities {
    pub avx512: bool,
    pub avx2: bool,
    pub sse4: bool,
    pub vector_width: usize,
}

impl HardwareProfiler {
    pub fn new() -> Self {
        Self {
            cache_sizes: vec![32768, 262144, 8388608], // Default L1, L2, L3 sizes
            memory_bandwidth: 25.6e9,                  // 25.6 GB/s default
            cpu_frequency: 3.0e9,                      // 3 GHz default
            num_cores: num_cpus::get(),
            simd_capabilities: Self::detect_simd_capabilities(),
        }
    }

    fn detect_simd_capabilities() -> SIMDCapabilities {
        // In a real implementation, this would use CPU feature detection
        SIMDCapabilities {
            avx512: false, // Conservative default
            avx2: true,    // Assume AVX2 support
            sse4: true,
            vector_width: 256, // 256-bit vectors for AVX2
        }
    }

    /// Estimate optimal parameters based on hardware characteristics
    pub fn suggest_optimal_parameters(&self, data_size: usize) -> OptimalParameters {
        let cache_friendly_tile_size = (self.cache_sizes[0] / 8).min(1024); // L1 cache friendly
        let parallel_threshold = data_size / self.num_cores;

        OptimalParameters {
            tile_size: cache_friendly_tile_size,
            parallel_threshold,
            simd_enabled: self.simd_capabilities.avx2,
            vector_width: self.simd_capabilities.vector_width,
        }
    }
}

#[derive(Debug, Clone)]
pub struct OptimalParameters {
    pub tile_size: usize,
    pub parallel_threshold: usize,
    pub simd_enabled: bool,
    pub vector_width: usize,
}

/// Parameter controller for adaptive adjustments
#[derive(Debug)]
pub struct ParameterController {
    /// Current parameter values
    current_parameters: HashMap<String, f64>,
    /// Parameter bounds
    parameter_bounds: HashMap<String, (f64, f64)>,
    /// Adjustment history
    adjustmenthistory: VecDeque<ParameterAdjustment>,
}

#[derive(Debug, Clone)]
pub struct ParameterAdjustment {
    pub parameter_name: String,
    pub old_value: f64,
    pub new_value: f64,
    pub performance_impact: f64,
    pub timestamp: Instant,
}

impl ParameterController {
    pub fn new() -> Self {
        let mut bounds = HashMap::new();
        bounds.insert("quantum_coherence_time".to_string(), (0.1, 10.0));
        bounds.insert("consciousness_depth".to_string(), (1.0, 32.0));
        bounds.insert("meta_learning_rate".to_string(), (0.001, 0.1));
        bounds.insert("advanced_dimensions".to_string(), (4.0, 64.0));
        bounds.insert("temporal_window".to_string(), (8.0, 256.0));

        Self {
            current_parameters: HashMap::new(),
            parameter_bounds: bounds,
            adjustmenthistory: VecDeque::new(),
        }
    }

    /// Adjust parameters based on performance feedback
    pub fn adjust_parameters(
        &mut self,
        performance_impact: f64,
        config: &AdaptiveOptimizerConfig,
    ) -> NdimageResult<HashMap<String, f64>> {
        let mut adjustments = HashMap::new();

        for (param_name, &current_value) in &self.current_parameters {
            if let Some(&(min_val, max_val)) = self.parameter_bounds.get(param_name) {
                // Calculate adjustment based on performance feedback
                let adjustment_factor = if performance_impact > config.improvement_threshold {
                    1.0 + config.learning_rate
                } else if performance_impact < -config.improvement_threshold {
                    1.0 - config.learning_rate
                } else {
                    1.0 // No adjustment needed
                };

                let new_value = (current_value * adjustment_factor).clamp(min_val, max_val);

                // Apply maximum adjustment rate limit
                let max_change = current_value * config.max_adjustment_rate;
                let limited_new_value = if new_value > current_value {
                    (current_value + max_change).min(new_value)
                } else {
                    (current_value - max_change).max(new_value)
                };

                if (limited_new_value - current_value).abs() > 1e-6 {
                    adjustments.insert(param_name.clone(), limited_new_value);

                    // Record adjustment
                    self.adjustmenthistory.push_back(ParameterAdjustment {
                        parameter_name: param_name.clone(),
                        old_value: current_value,
                        new_value: limited_new_value,
                        performance_impact,
                        timestamp: Instant::now(),
                    });
                }
            }
        }

        // Limit history size
        if self.adjustmenthistory.len() > 1000 {
            self.adjustmenthistory.pop_front();
        }

        Ok(adjustments)
    }
}

/// Real-time monitoring system
#[derive(Debug)]
pub struct RealTimeMonitor {
    /// Current operation being monitored
    current_operation: Option<String>,
    /// Monitoring start time
    start_time: Option<Instant>,
    /// Real-time metrics
    metrics: PerformanceMetrics,
    /// Monitoring active flag
    active: bool,
}

impl RealTimeMonitor {
    pub fn new() -> Self {
        Self {
            current_operation: None,
            start_time: None,
            metrics: PerformanceMetrics {
                execution_time: Duration::from_secs(0),
                memory_throughput: 0.0,
                flops: 0.0,
                cache_hit_ratio: 0.0,
                cpu_utilization: 0.0,
                power_consumption: 0.0,
            },
            active: false,
        }
    }

    /// Start monitoring an operation
    pub fn start_monitoring(&mut self, operation_name: String) {
        self.current_operation = Some(operation_name);
        self.start_time = Some(Instant::now());
        self.active = true;
    }

    /// Stop monitoring and return metrics
    pub fn stop_monitoring(&mut self) -> Option<PerformanceMetrics> {
        if !self.active {
            return None;
        }

        if let Some(start_time) = self.start_time {
            self.metrics.execution_time = start_time.elapsed();
        }

        self.active = false;
        self.current_operation = None;
        self.start_time = None;

        Some(self.metrics.clone())
    }
}

impl AdaptiveAdvancedOptimizer {
    /// Create a new adaptive optimizer
    pub fn new(config: AdaptiveOptimizerConfig) -> Self {
        Self {
            performancehistory: Arc::new(RwLock::new(HashMap::new())),
            ml_predictor: Arc::new(Mutex::new(PerformancePredictionModel::new())),
            hardware_profiler: Arc::new(Mutex::new(HardwareProfiler::new())),
            parameter_controller: Arc::new(Mutex::new(ParameterController::new())),
            monitor: Arc::new(Mutex::new(RealTimeMonitor::new())),
            config,
        }
    }

    /// Optimize configuration for a specific operation
    pub fn optimize_configuration(
        &self,
        operation_type: &str,
        data_characteristics: &DataCharacteristics,
        base_config: &AdvancedConfig,
    ) -> NdimageResult<AdvancedConfig> {
        // Get hardware-optimized parameters
        let hardware_profiler = self.hardware_profiler.lock().map_err(|_| {
            NdimageError::ComputationError("Failed to acquire hardware profiler lock".into())
        })?;

        let data_size = data_characteristics.dimensions.iter().product::<usize>();
        let optimal_params = hardware_profiler.suggest_optimal_parameters(data_size);

        // Get ML prediction for performance
        let mut optimized_config = base_config.clone();

        if self.config.enable_prediction {
            let ml_predictor = self.ml_predictor.lock().map_err(|_| {
                NdimageError::ComputationError("Failed to acquire ML predictor lock".into())
            })?;

            let features = self.extractfeatures(data_characteristics, &optimal_params);
            let predicted_performance = ml_predictor.predict_performance(&features);

            // Adjust configuration based on prediction
            if predicted_performance < 0.5 {
                // Poor predicted performance, use conservative settings
                optimized_config.advanced_dimensions = optimized_config.advanced_dimensions.min(8);
                optimized_config.consciousness_depth = optimized_config.consciousness_depth.min(4);
            } else {
                // Good predicted performance, use aggressive settings
                optimized_config.advanced_dimensions = optimized_config.advanced_dimensions.max(16);
                optimized_config.consciousness_depth = optimized_config.consciousness_depth.max(8);
            }
        }

        // Apply hardware-specific optimizations
        optimized_config.temporal_window = optimal_params
            .tile_size
            .min(optimized_config.temporal_window);

        Ok(optimized_config)
    }

    /// Record performance results for learning
    pub fn record_performance(
        &self,
        operation_type: String,
        data_characteristics: DataCharacteristics,
        config_used: AdvancedConfig,
        metrics: PerformanceMetrics,
    ) -> NdimageResult<()> {
        let snapshot = PerformanceSnapshot {
            timestamp: Instant::now(),
            operation_type: operation_type.clone(),
            data_characteristics: data_characteristics.clone(),
            metrics: metrics.clone(),
            config_used,
        };

        // Add to performance history
        let mut history = self.performancehistory.write().map_err(|_| {
            NdimageError::ComputationError("Failed to acquire performance history lock".into())
        })?;

        let operationhistory = history.entry(operation_type).or_insert_with(VecDeque::new);
        operationhistory.push_back(snapshot);

        // Limit history size
        if operationhistory.len() > self.config.history_window_size {
            operationhistory.pop_front();
        }

        // Update ML model
        if self.config.enable_prediction {
            let mut ml_predictor = self.ml_predictor.lock().map_err(|_| {
                NdimageError::ComputationError("Failed to acquire ML predictor lock".into())
            })?;

            let hardware_profiler = self.hardware_profiler.lock().map_err(|_| {
                NdimageError::ComputationError("Failed to acquire hardware profiler lock".into())
            })?;

            let data_size = data_characteristics.dimensions.iter().product::<usize>();
            let optimal_params = hardware_profiler.suggest_optimal_parameters(data_size);
            let features = self.extractfeatures(&data_characteristics, &optimal_params);

            // Use execution time as performance target (lower is better, so invert)
            let performance_score = 1.0 / (metrics.execution_time.as_secs_f64() + 1e-6);
            ml_predictor.update_model(features, performance_score)?;
        }

        Ok(())
    }

    /// Extract features for machine learning
    fn extractfeatures(
        &self,
        data_chars: &DataCharacteristics,
        optimal_params: &OptimalParameters,
    ) -> Array1<f64> {
        let mut features = Array1::zeros(16);

        // Data characteristics features
        features[0] = data_chars.dimensions.len() as f64; // Number of dimensions
        features[1] = data_chars.dimensions.iter().product::<usize>() as f64; // Total size
        features[2] = data_chars.element_size as f64;
        features[3] = data_chars.complexity_score;

        // Access pattern features
        features[4] = match data_chars.access_pattern {
            AccessPattern::Sequential => 1.0,
            AccessPattern::Random => 2.0,
            AccessPattern::Strided { .. } => 3.0,
            AccessPattern::Blocked { .. } => 4.0,
        };

        // Hardware features
        features[5] = optimal_params.tile_size as f64;
        features[6] = optimal_params.parallel_threshold as f64;
        features[7] = if optimal_params.simd_enabled {
            1.0
        } else {
            0.0
        };
        features[8] = optimal_params.vector_width as f64;

        // Shape-based features
        if !data_chars.dimensions.is_empty() {
            features[9] = data_chars.dimensions[0] as f64;
            if data_chars.dimensions.len() > 1 {
                features[10] = data_chars.dimensions[1] as f64;
            }
            if data_chars.dimensions.len() > 2 {
                features[11] = data_chars.dimensions[2] as f64;
            }
        }

        // Derived features
        features[12] = features[1].log2(); // Log of total size
        features[13] = data_chars.dimensions.len() as f64 * features[2]; // Dimension * element size
        features[14] = features[1] / features[5]; // Size / tile size ratio
        features[15] = 1.0; // Bias feature

        features
    }

    /// Get performance analysis for an operation type
    pub fn get_performance_analysis(
        &self,
        operation_type: &str,
    ) -> NdimageResult<PerformanceAnalysis> {
        let history = self.performancehistory.read().map_err(|_| {
            NdimageError::ComputationError("Failed to acquire performance history lock".into())
        })?;

        let snapshots = history.get(operation_type).ok_or_else(|| {
            NdimageError::ComputationError(format!(
                "No performance history for operation: {}",
                operation_type
            ))
        })?;

        if snapshots.is_empty() {
            return Err(NdimageError::ComputationError(
                "No performance data available".into(),
            ));
        }

        let mut execution_times: Vec<f64> = snapshots
            .iter()
            .map(|s| s.metrics.execution_time.as_secs_f64())
            .collect();

        execution_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mean_time = execution_times.iter().sum::<f64>() / execution_times.len() as f64;
        let median_time = execution_times[execution_times.len() / 2];
        let min_time = execution_times[0];
        let max_time = execution_times[execution_times.len() - 1];

        // Calculate trend (simple linear regression)
        let trend = self.calculate_performance_trend(snapshots);

        Ok(PerformanceAnalysis {
            operation_type: operation_type.to_string(),
            total_samples: snapshots.len(),
            mean_execution_time: Duration::from_secs_f64(mean_time),
            median_execution_time: Duration::from_secs_f64(median_time),
            min_execution_time: Duration::from_secs_f64(min_time),
            max_execution_time: Duration::from_secs_f64(max_time),
            performance_trend: trend,
            optimization_opportunities: self.identify_optimization_opportunities(snapshots),
        })
    }

    fn calculate_performance_trend(&self, snapshots: &VecDeque<PerformanceSnapshot>) -> f64 {
        if snapshots.len() < 2 {
            return 0.0;
        }

        let x_values: Vec<f64> = (0..snapshots.len()).map(|i| i as f64).collect();
        let y_values: Vec<f64> = snapshots
            .iter()
            .map(|s| s.metrics.execution_time.as_secs_f64())
            .collect();

        // Simple linear regression slope
        let n = x_values.len() as f64;
        let sum_x = x_values.iter().sum::<f64>();
        let sum_y = y_values.iter().sum::<f64>();
        let sum_xy = x_values
            .iter()
            .zip(y_values.iter())
            .map(|(x, y)| x * y)
            .sum::<f64>();
        let sum_x2 = x_values.iter().map(|x| x * x).sum::<f64>();

        (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
    }

    fn identify_optimization_opportunities(
        &self,
        snapshots: &VecDeque<PerformanceSnapshot>,
    ) -> Vec<String> {
        let mut opportunities = Vec::new();

        // Analyze cache hit ratios
        let avg_cache_hit_ratio = snapshots
            .iter()
            .map(|s| s.metrics.cache_hit_ratio)
            .sum::<f64>()
            / snapshots.len() as f64;

        if avg_cache_hit_ratio < 0.8 {
            opportunities
                .push("Consider adjusting tile sizes for better cache locality".to_string());
        }

        // Analyze CPU utilization
        let avg_cpu_utilization = snapshots
            .iter()
            .map(|s| s.metrics.cpu_utilization)
            .sum::<f64>()
            / snapshots.len() as f64;

        if avg_cpu_utilization < 0.6 {
            opportunities.push("CPU underutilized - consider increasing parallelism".to_string());
        }

        // Analyze memory throughput
        let max_memory_throughput = snapshots
            .iter()
            .map(|s| s.metrics.memory_throughput)
            .fold(0.0, f64::max);

        let avg_memory_throughput = snapshots
            .iter()
            .map(|s| s.metrics.memory_throughput)
            .sum::<f64>()
            / snapshots.len() as f64;

        if avg_memory_throughput < max_memory_throughput * 0.5 {
            opportunities
                .push("Memory bandwidth underutilized - optimize data access patterns".to_string());
        }

        opportunities
    }
}

#[derive(Debug)]
pub struct PerformanceAnalysis {
    pub operation_type: String,
    pub total_samples: usize,
    pub mean_execution_time: Duration,
    pub median_execution_time: Duration,
    pub min_execution_time: Duration,
    pub max_execution_time: Duration,
    pub performance_trend: f64,
    pub optimization_opportunities: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_optimizer_creation() {
        let config = AdaptiveOptimizerConfig::default();
        let optimizer = AdaptiveAdvancedOptimizer::new(config);

        // Test basic creation
        assert!(optimizer.performancehistory.read().unwrap().is_empty());
    }

    #[test]
    fn test_performance_prediction() {
        let mut model = PerformancePredictionModel::new();

        let features = Array1::ones(16);
        let prediction = model.predict_performance(&features);
        assert!(prediction >= 0.0);

        // Test model update
        let result = model.update_model(features.clone(), 1.0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_hardware_profiler() {
        let profiler = HardwareProfiler::new();
        let params = profiler.suggest_optimal_parameters(1000000);

        assert!(params.tile_size > 0);
        assert!(params.parallel_threshold > 0);
    }

    #[test]
    fn test_parameter_controller() {
        let mut controller = ParameterController::new();
        let config = AdaptiveOptimizerConfig::default();

        // Initialize some parameters
        controller
            .current_parameters
            .insert("test_param".to_string(), 1.0);
        controller
            .parameter_bounds
            .insert("test_param".to_string(), (0.1, 10.0));

        let adjustments = controller.adjust_parameters(0.1, &config);
        assert!(adjustments.is_ok());
    }
}
