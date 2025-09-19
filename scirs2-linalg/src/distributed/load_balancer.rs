//! Adaptive load balancing for distributed operations

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Adaptive load balancer with machine learning capabilities
#[derive(Debug)]
pub struct AdaptiveLoadBalancer {
    /// Historical performance data
    performance_history: Vec<NodePerformanceRecord>,
    /// Current load distribution
    current_loads: HashMap<usize, f64>,
    /// Predictive model for load balancing
    prediction_model: LoadPredictionModel,
    /// Dynamic rebalancing parameters
    rebalancing_config: RebalancingConfig,
}

/// Performance record for a compute node
#[derive(Debug, Clone)]
pub struct NodePerformanceRecord {
    node_id: usize,
    timestamp: Instant,
    operations_per_second: f64,
    memory_usage: f64,
    network_latency: f64,
    cpu_utilization: f64,
    gpu_utilization: Option<f64>,
    workload_type: WorkloadType,
}

/// Predictive model for load balancing decisions
#[derive(Debug)]
pub struct LoadPredictionModel {
    /// Linear regression coefficients
    coefficients: HashMap<String, f64>,
    /// Prediction accuracy metrics
    accuracy_metrics: ModelAccuracyMetrics,
    /// Training data
    training_data: Vec<LoadPredictionSample>,
    /// Model update frequency
    update_frequency: usize,
    /// Last model update
    last_update: Instant,
}

/// Sample for load prediction training
#[derive(Debug, Clone)]
pub struct LoadPredictionSample {
    features: HashMap<String, f64>,
    actual_performance: f64,
    prediction: Option<f64>,
    error: Option<f64>,
}

/// Model accuracy metrics
#[derive(Debug, Default)]
pub struct ModelAccuracyMetrics {
    mean_absolute_error: f64,
    root_mean_square_error: f64,
    r_squared: f64,
    samples_count: usize,
}

/// Configuration for dynamic rebalancing
#[derive(Debug, Clone)]
pub struct RebalancingConfig {
    /// Minimum imbalance threshold to trigger rebalancing
    imbalance_threshold: f64,
    /// Maximum rebalancing frequency (operations)
    max_rebalance_frequency: usize,
    /// Cost threshold for beneficial rebalancing
    cost_benefit_threshold: f64,
    /// Enable predictive rebalancing
    predictive_rebalancing: bool,
}

/// Workload type for performance modeling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorkloadType {
    MatrixMultiplication,
    Decomposition,
    LinearSolve,
    Eigenvalue,
    FFT,
    ElementWise,
    Reduction,
}

impl AdaptiveLoadBalancer {
    /// Create a new adaptive load balancer
    pub fn new(config: RebalancingConfig) -> Self {
        Self {
            performance_history: Vec::new(),
            current_loads: HashMap::new(),
            prediction_model: LoadPredictionModel::new(),
            rebalancing_config: config,
        }
    }

    /// Record performance data for a node
    pub fn record_performance(&mut self, record: NodePerformanceRecord) {
        self.performance_history.push(record.clone());
        self.update_current_load(record.node_id, record.operations_per_second);

        // Update prediction model if enough data collected
        if self.performance_history.len() % self.prediction_model.update_frequency == 0 {
            self.update_prediction_model();
        }
    }

    /// Update current load for a node
    fn update_current_load(&mut self, node_id: usize, load: f64) {
        self.current_loads.insert(node_id, load);
    }

    /// Update the prediction model
    fn update_prediction_model(&mut self) {
        // Simplified model update - in practice would use proper ML algorithms
        self.prediction_model.last_update = Instant::now();

        // Calculate accuracy metrics from recent predictions
        self.calculate_accuracy_metrics();
    }

    /// Calculate model accuracy metrics
    fn calculate_accuracy_metrics(&mut self) {
        let recent_samples: Vec<_> = self.prediction_model.training_data
            .iter()
            .filter(|s| s.prediction.is_some() && s.error.is_some())
            .collect();

        if recent_samples.is_empty() {
            return;
        }

        let mae = recent_samples.iter()
            .map(|s| s.error.unwrap().abs())
            .sum::<f64>() / recent_samples.len() as f64;

        let rmse = (recent_samples.iter()
            .map(|s| s.error.unwrap().powi(2))
            .sum::<f64>() / recent_samples.len() as f64).sqrt();

        self.prediction_model.accuracy_metrics.mean_absolute_error = mae;
        self.prediction_model.accuracy_metrics.root_mean_square_error = rmse;
        self.prediction_model.accuracy_metrics.samples_count = recent_samples.len();
    }

    /// Predict performance for a workload
    pub fn predict_performance(&self, node_id: usize, workload: WorkloadType) -> f64 {
        // Simplified prediction - would use actual ML model in practice
        self.current_loads.get(&node_id).copied().unwrap_or(1.0)
    }

    /// Check if rebalancing is needed
    pub fn should_rebalance(&self) -> bool {
        let loads: Vec<f64> = self.current_loads.values().copied().collect();
        if loads.is_empty() {
            return false;
        }

        let max_load = loads.iter().fold(0.0_f64, |a, &b| a.max(b));
        let min_load = loads.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        if max_load == 0.0 {
            return false;
        }

        let imbalance = (max_load - min_load) / max_load;
        imbalance > self.rebalancing_config.imbalance_threshold
    }

    /// Get current load distribution
    pub fn get_load_distribution(&self) -> &HashMap<usize, f64> {
        &self.current_loads
    }
}

impl LoadPredictionModel {
    fn new() -> Self {
        Self {
            coefficients: HashMap::new(),
            accuracy_metrics: ModelAccuracyMetrics::default(),
            training_data: Vec::new(),
            update_frequency: 100,
            last_update: Instant::now(),
        }
    }
}

impl Default for RebalancingConfig {
    fn default() -> Self {
        Self {
            imbalance_threshold: 0.2,
            max_rebalance_frequency: 10,
            cost_benefit_threshold: 0.1,
            predictive_rebalancing: true,
        }
    }
}