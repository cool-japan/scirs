//! Performance analysis and monitoring for optimization coordinator
//!
//! This module provides comprehensive performance analysis capabilities including
//! real-time performance monitoring, benchmarking, model tracking, and
//! performance prediction with confidence estimation.

use super::config::*;
use crate::OptimizerError as OptimError;
use ndarray::{Array1, Array2};
use num_traits::Float;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::time::{Duration, Instant, SystemTime};

/// Result type for performance operations
type Result<T> = std::result::Result<T, OptimError>;

/// Performance analyzer for optimization monitoring
#[derive(Debug)]
pub struct PerformanceAnalyzer<T: Float + Send + Sync + Debug> {
    /// Configuration
    config: PerformanceMonitoringConfig,

    /// Performance snapshots history
    snapshots: VecDeque<PerformanceSnapshot<T>>,

    /// Performance model for prediction
    performance_model: PerformanceModel<T>,

    /// Confidence estimator
    confidence_estimator: ConfidenceEstimator<T>,

    /// Benchmark results
    benchmark_results: Vec<BenchmarkResult<T>>,

    /// Performance context tracking
    context_tracker: PerformanceContextTracker<T>,

    /// Real-time metrics
    current_metrics: PerformanceMetrics<T>,

    /// Performance thresholds
    thresholds: PerformanceThresholds<T>,
}

/// Performance snapshot for point-in-time analysis
#[derive(Debug, Clone)]
pub struct PerformanceSnapshot<T: Float> {
    /// Timestamp of snapshot
    pub timestamp: SystemTime,

    /// Training step/epoch
    pub step: usize,

    /// Loss value
    pub loss: T,

    /// Accuracy metric
    pub accuracy: Option<T>,

    /// Training time for this step
    pub training_time: Duration,

    /// Memory usage
    pub memory_usage: usize,

    /// GPU utilization
    pub gpu_utilization: Option<f32>,

    /// Learning rate
    pub learning_rate: T,

    /// Gradient norm
    pub gradient_norm: Option<T>,

    /// Performance metrics
    pub metrics: PerformanceMetrics<T>,

    /// Resource usage
    pub resource_usage: ResourceUsage<T>,

    /// Performance context
    pub context: PerformanceContext<T>,
}

/// Comprehensive performance metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics<T: Float> {
    /// Primary loss value
    pub loss: T,

    /// Validation loss
    pub val_loss: Option<T>,

    /// Training accuracy
    pub accuracy: Option<T>,

    /// Validation accuracy
    pub val_accuracy: Option<T>,

    /// Learning rate
    pub learning_rate: T,

    /// Gradient norm
    pub gradient_norm: Option<T>,

    /// Parameter norm
    pub parameter_norm: Option<T>,

    /// Training throughput (samples/second)
    pub throughput: Option<T>,

    /// Convergence rate
    pub convergence_rate: Option<T>,

    /// Custom metrics
    pub custom_metrics: HashMap<String, T>,
}

/// Performance context information
#[derive(Debug, Clone)]
pub struct PerformanceContext<T: Float> {
    /// Model architecture
    pub model_architecture: String,

    /// Dataset information
    pub dataset_info: String,

    /// Batch size
    pub batch_size: usize,

    /// Optimizer type
    pub optimizer_type: String,

    /// Training regime
    pub training_regime: String,

    /// Hardware configuration
    pub hardware_config: String,

    /// Environment variables
    pub environment: HashMap<String, String>,

    /// Hyperparameters
    pub hyperparameters: HashMap<String, T>,
}

/// Resource usage tracking
#[derive(Debug, Clone)]
pub struct ResourceUsage<T: Float> {
    /// CPU usage percentage
    pub cpu_usage: f32,

    /// Memory usage in bytes
    pub memory_usage: usize,

    /// GPU memory usage in bytes
    pub gpu_memory_usage: Option<usize>,

    /// GPU utilization percentage
    pub gpu_utilization: Option<f32>,

    /// Disk I/O rate
    pub disk_io_rate: Option<T>,

    /// Network I/O rate
    pub network_io_rate: Option<T>,

    /// Power consumption (watts)
    pub power_consumption: Option<f32>,
}

/// Performance prediction model
#[derive(Debug)]
pub struct PerformanceModel<T: Float + Send + Sync + Debug> {
    /// Model type
    model_type: ModelType,

    /// Model parameters
    parameters: ModelParameters<T>,

    /// Training history
    training_history: Vec<PerformanceSnapshot<T>>,

    /// Validation metrics
    validation_metrics: ValidationMetrics<T>,

    /// Feature importance
    feature_importance: HashMap<String, T>,

    /// Model version
    version: String,

    /// Last update timestamp
    last_update: SystemTime,
}

/// Performance model types
#[derive(Debug, Clone)]
pub enum ModelType {
    /// Linear regression
    LinearRegression,

    /// Polynomial regression
    PolynomialRegression { degree: usize },

    /// Exponential model
    ExponentialModel,

    /// Neural network
    NeuralNetwork { hidden_layers: Vec<usize> },

    /// Ensemble model
    Ensemble { models: Vec<ModelType> },

    /// Custom model
    Custom { name: String },
}

/// Model parameters
#[derive(Debug, Clone)]
pub struct ModelParameters<T: Float> {
    /// Weight coefficients
    pub weights: Array1<T>,

    /// Bias term
    pub bias: T,

    /// Regularization parameters
    pub regularization: RegularizationParameters<T>,

    /// Learning parameters
    pub learning_params: HashMap<String, T>,
}

/// Regularization parameters
#[derive(Debug, Clone)]
pub struct RegularizationParameters<T: Float> {
    /// L1 regularization strength
    pub l1_lambda: T,

    /// L2 regularization strength
    pub l2_lambda: T,

    /// Dropout rate
    pub dropout_rate: Option<T>,

    /// Early stopping parameters
    pub early_stopping: EarlyStoppingConfig<T>,
}

/// Validation metrics for model performance
#[derive(Debug, Clone)]
pub struct ValidationMetrics<T: Float> {
    /// Mean squared error
    pub mse: T,

    /// Mean absolute error
    pub mae: T,

    /// R-squared score
    pub r2_score: T,

    /// Cross-validation scores
    pub cv_scores: Vec<T>,

    /// Prediction intervals
    pub prediction_intervals: Vec<(T, T)>,

    /// Confidence scores
    pub confidence_scores: Vec<T>,
}

/// Confidence estimator for performance predictions
#[derive(Debug)]
pub struct ConfidenceEstimator<T: Float + Send + Sync + Debug> {
    /// Confidence method
    method: ConfidenceMethod,

    /// Calibration points
    calibration_points: Vec<CalibrationPoint<T>>,

    /// Confidence thresholds
    thresholds: ConfidenceThresholds<T>,

    /// Historical accuracy
    historical_accuracy: VecDeque<T>,

    /// Uncertainty estimates
    uncertainty_estimates: HashMap<String, T>,
}

/// Confidence estimation methods
#[derive(Debug, Clone)]
pub enum ConfidenceMethod {
    /// Bootstrap sampling
    Bootstrap { n_samples: usize },

    /// Bayesian inference
    Bayesian,

    /// Ensemble variance
    EnsembleVariance,

    /// Prediction intervals
    PredictionIntervals { alpha: f64 },

    /// Conformal prediction
    ConformalPrediction,
}

/// Calibration point for confidence estimation
#[derive(Debug, Clone)]
pub struct CalibrationPoint<T: Float> {
    /// Predicted confidence
    pub predicted_confidence: T,

    /// Actual accuracy
    pub actual_accuracy: T,

    /// Sample count
    pub sample_count: usize,

    /// Timestamp
    pub timestamp: SystemTime,
}

/// Confidence thresholds
#[derive(Debug, Clone)]
pub struct ConfidenceThresholds<T: Float> {
    /// Low confidence threshold
    pub low_threshold: T,

    /// Medium confidence threshold
    pub medium_threshold: T,

    /// High confidence threshold
    pub high_threshold: T,

    /// Critical confidence threshold
    pub critical_threshold: T,
}

/// Benchmark result
#[derive(Debug, Clone)]
pub struct BenchmarkResult<T: Float> {
    /// Benchmark name
    pub name: String,

    /// Dataset used
    pub dataset: String,

    /// Performance metrics
    pub metrics: PerformanceMetrics<T>,

    /// Benchmark conditions
    pub conditions: BenchmarkConditions<T>,

    /// Execution time
    pub execution_time: Duration,

    /// Timestamp
    pub timestamp: SystemTime,

    /// Comparison results
    pub comparisons: HashMap<String, T>,

    /// Statistical significance
    pub statistical_significance: Option<T>,
}

/// Benchmark execution conditions
#[derive(Debug, Clone)]
pub struct BenchmarkConditions<T: Float> {
    /// Hardware configuration
    pub hardware: String,

    /// Software versions
    pub software_versions: HashMap<String, String>,

    /// Environment settings
    pub environment: HashMap<String, String>,

    /// Random seed
    pub random_seed: u64,

    /// Number of runs
    pub num_runs: usize,

    /// Warm-up iterations
    pub warmup_iterations: usize,

    /// Configuration parameters
    pub config_params: HashMap<String, T>,
}

/// Performance context tracker
#[derive(Debug)]
pub struct PerformanceContextTracker<T: Float + Send + Sync + Debug> {
    /// Current context
    current_context: PerformanceContext<T>,

    /// Context history
    context_history: VecDeque<PerformanceContext<T>>,

    /// Context changes
    context_changes: Vec<ContextChange<T>>,

    /// Impact analysis
    impact_analysis: HashMap<String, T>,
}

/// Context change tracking
#[derive(Debug, Clone)]
pub struct ContextChange<T: Float> {
    /// Change timestamp
    pub timestamp: SystemTime,

    /// Changed field
    pub field: String,

    /// Previous value
    pub previous_value: String,

    /// New value
    pub new_value: String,

    /// Performance impact
    pub impact: Option<T>,

    /// Change reason
    pub reason: String,
}

impl<T: Float + Send + Sync + Debug + Default + Clone> PerformanceAnalyzer<T> {
    /// Create new performance analyzer
    pub fn new(config: PerformanceMonitoringConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            snapshots: VecDeque::with_capacity(config.max_snapshots),
            performance_model: PerformanceModel::new(ModelType::LinearRegression)?,
            confidence_estimator: ConfidenceEstimator::new(ConfidenceMethod::Bootstrap { n_samples: 1000 }),
            benchmark_results: Vec::new(),
            context_tracker: PerformanceContextTracker::new(),
            current_metrics: PerformanceMetrics::default(),
            thresholds: config.performance_thresholds,
        })
    }

    /// Record performance snapshot
    pub fn record_snapshot(&mut self, snapshot: PerformanceSnapshot<T>) -> Result<()> {
        // Update current metrics
        self.current_metrics = snapshot.metrics.clone();

        // Add to history
        if self.snapshots.len() >= self.config.max_snapshots {
            self.snapshots.pop_front();
        }
        self.snapshots.push_back(snapshot.clone());

        // Update performance model
        self.performance_model.update_with_snapshot(&snapshot)?;

        // Update confidence estimator
        self.confidence_estimator.update(&snapshot)?;

        // Check thresholds
        self.check_performance_thresholds(&snapshot)?;

        Ok(())
    }

    /// Get current performance metrics
    pub fn get_current_metrics(&self) -> &PerformanceMetrics<T> {
        &self.current_metrics
    }

    /// Get performance history
    pub fn get_performance_history(&self) -> &VecDeque<PerformanceSnapshot<T>> {
        &self.snapshots
    }

    /// Predict future performance
    pub fn predict_performance(&self, steps_ahead: usize) -> Result<Vec<PerformanceSnapshot<T>>> {
        self.performance_model.predict(steps_ahead)
    }

    /// Get confidence estimate for predictions
    pub fn get_prediction_confidence(&self, predictions: &[PerformanceSnapshot<T>]) -> Result<Vec<T>> {
        self.confidence_estimator.estimate_confidence(predictions)
    }

    /// Run benchmark
    pub fn run_benchmark(&mut self, benchmark_name: &str, dataset: &str) -> Result<BenchmarkResult<T>> {
        let start_time = Instant::now();

        // Create benchmark conditions
        let conditions = BenchmarkConditions {
            hardware: "unknown".to_string(),
            software_versions: HashMap::new(),
            environment: HashMap::new(),
            random_seed: 42,
            num_runs: 5,
            warmup_iterations: 3,
            config_params: HashMap::new(),
        };

        // Run benchmark (simplified implementation)
        let metrics = self.current_metrics.clone();

        let result = BenchmarkResult {
            name: benchmark_name.to_string(),
            dataset: dataset.to_string(),
            metrics,
            conditions,
            execution_time: start_time.elapsed(),
            timestamp: SystemTime::now(),
            comparisons: HashMap::new(),
            statistical_significance: None,
        };

        self.benchmark_results.push(result.clone());
        Ok(result)
    }

    /// Analyze performance trends
    pub fn analyze_trends(&self, window_size: usize) -> Result<PerformanceTrendAnalysis<T>> {
        if self.snapshots.len() < window_size {
            return Err(OptimError::InsufficientData(
                "Not enough snapshots for trend analysis".to_string()
            ));
        }

        let recent_snapshots: Vec<_> = self.snapshots.iter().rev().take(window_size).collect();

        // Calculate trend metrics
        let mut loss_trend = Vec::new();
        let mut accuracy_trend = Vec::new();

        for snapshot in &recent_snapshots {
            loss_trend.push(snapshot.loss);
            if let Some(acc) = snapshot.accuracy {
                accuracy_trend.push(acc);
            }
        }

        Ok(PerformanceTrendAnalysis {
            loss_trend,
            accuracy_trend,
            convergence_rate: self.calculate_convergence_rate(&recent_snapshots)?,
            stability_score: self.calculate_stability_score(&recent_snapshots)?,
            improvement_rate: self.calculate_improvement_rate(&recent_snapshots)?,
        })
    }

    /// Check performance thresholds
    fn check_performance_thresholds(&self, snapshot: &PerformanceSnapshot<T>) -> Result<()> {
        // Check loss threshold
        if let Some(max_loss) = self.thresholds.max_loss {
            if snapshot.loss > max_loss {
                log::warn!("Loss {} exceeds threshold {}", snapshot.loss, max_loss);
            }
        }

        // Check accuracy threshold
        if let (Some(accuracy), Some(min_accuracy)) = (snapshot.accuracy, self.thresholds.min_accuracy) {
            if accuracy < min_accuracy {
                log::warn!("Accuracy {} below threshold {}", accuracy, min_accuracy);
            }
        }

        // Check training time threshold
        if let Some(max_time) = self.thresholds.max_training_time {
            if snapshot.training_time > max_time {
                log::warn!("Training time {:?} exceeds threshold {:?}",
                          snapshot.training_time, max_time);
            }
        }

        Ok(())
    }

    /// Calculate convergence rate
    fn calculate_convergence_rate(&self, snapshots: &[&PerformanceSnapshot<T>]) -> Result<T> {
        if snapshots.len() < 2 {
            return Ok(T::zero());
        }

        let first_loss = snapshots.last().unwrap().loss;
        let last_loss = snapshots.first().unwrap().loss;
        let steps = T::from(snapshots.len()).unwrap();

        Ok((first_loss - last_loss) / steps)
    }

    /// Calculate stability score
    fn calculate_stability_score(&self, snapshots: &[&PerformanceSnapshot<T>]) -> Result<T> {
        if snapshots.is_empty() {
            return Ok(T::zero());
        }

        let losses: Vec<T> = snapshots.iter().map(|s| s.loss).collect();
        let mean = losses.iter().fold(T::zero(), |acc, &x| acc + x) / T::from(losses.len()).unwrap();

        let variance = losses.iter()
            .map(|&x| (x - mean) * (x - mean))
            .fold(T::zero(), |acc, x| acc + x) / T::from(losses.len()).unwrap();

        Ok(T::one() / (T::one() + variance.sqrt()))
    }

    /// Calculate improvement rate
    fn calculate_improvement_rate(&self, snapshots: &[&PerformanceSnapshot<T>]) -> Result<T> {
        if snapshots.len() < 2 {
            return Ok(T::zero());
        }

        let mut improvement_count = 0;
        for window in snapshots.windows(2) {
            if window[1].loss < window[0].loss {
                improvement_count += 1;
            }
        }

        Ok(num_traits::cast::cast(improvement_count).unwrap_or_else(|| T::zero()) / T::from(snapshots.len() - 1).unwrap())
    }
}

/// Performance trend analysis results
#[derive(Debug, Clone)]
pub struct PerformanceTrendAnalysis<T: Float> {
    /// Loss trend values
    pub loss_trend: Vec<T>,

    /// Accuracy trend values
    pub accuracy_trend: Vec<T>,

    /// Convergence rate
    pub convergence_rate: T,

    /// Stability score (0-1)
    pub stability_score: T,

    /// Improvement rate (0-1)
    pub improvement_rate: T,
}

impl<T: Float + Send + Sync + Debug + Default + Clone> PerformanceModel<T> {
    /// Create new performance model
    pub fn new(model_type: ModelType) -> Result<Self> {
        Ok(Self {
            model_type,
            parameters: ModelParameters::default(),
            training_history: Vec::new(),
            validation_metrics: ValidationMetrics::default(),
            feature_importance: HashMap::new(),
            version: "1.0".to_string(),
            last_update: SystemTime::now(),
        })
    }

    /// Update model with new snapshot
    pub fn update_with_snapshot(&mut self, snapshot: &PerformanceSnapshot<T>) -> Result<()> {
        self.training_history.push(snapshot.clone());
        self.last_update = SystemTime::now();

        // Retrain model if enough data
        if self.training_history.len() >= 10 {
            self.retrain()?;
        }

        Ok(())
    }

    /// Predict future performance
    pub fn predict(&self, steps_ahead: usize) -> Result<Vec<PerformanceSnapshot<T>>> {
        let mut predictions = Vec::new();

        if self.training_history.is_empty() {
            return Ok(predictions);
        }

        let last_snapshot = self.training_history.last().unwrap();

        for i in 1..=steps_ahead {
            let predicted_loss = self.predict_loss(i)?;
            let predicted_accuracy = self.predict_accuracy(i)?;

            let prediction = PerformanceSnapshot {
                timestamp: SystemTime::now(),
                step: last_snapshot.step + i,
                loss: predicted_loss,
                accuracy: predicted_accuracy,
                training_time: last_snapshot.training_time,
                memory_usage: last_snapshot.memory_usage,
                gpu_utilization: last_snapshot.gpu_utilization,
                learning_rate: last_snapshot.learning_rate,
                gradient_norm: last_snapshot.gradient_norm,
                metrics: PerformanceMetrics::default(),
                resource_usage: last_snapshot.resource_usage.clone(),
                context: last_snapshot.context.clone(),
            };

            predictions.push(prediction);
        }

        Ok(predictions)
    }

    /// Retrain the model
    fn retrain(&mut self) -> Result<()> {
        // Simplified retraining logic
        log::info!("Retraining performance model with {} samples", self.training_history.len());

        // Update validation metrics
        self.validation_metrics = self.calculate_validation_metrics()?;

        Ok(())
    }

    /// Predict loss for future step
    fn predict_loss(&self, steps_ahead: usize) -> Result<T> {
        if self.training_history.is_empty() {
            return Ok(T::zero());
        }

        // Simple linear extrapolation
        let recent_losses: Vec<T> = self.training_history.iter()
            .rev()
            .take(10)
            .map(|s| s.loss)
            .collect();

        if recent_losses.len() < 2 {
            return Ok(recent_losses[0]);
        }

        let trend = (recent_losses[0] - recent_losses[recent_losses.len() - 1]) /
                   T::from(recent_losses.len() - 1).unwrap();

        Ok(recent_losses[0] + trend * num_traits::cast::cast(steps_ahead).unwrap_or_else(|| T::zero()))
    }

    /// Predict accuracy for future step
    fn predict_accuracy(&self, _steps_ahead: usize) -> Result<Option<T>> {
        // Simplified accuracy prediction
        if let Some(last_accuracy) = self.training_history.last().and_then(|s| s.accuracy) {
            Ok(Some(last_accuracy))
        } else {
            Ok(None)
        }
    }

    /// Calculate validation metrics
    fn calculate_validation_metrics(&self) -> Result<ValidationMetrics<T>> {
        Ok(ValidationMetrics::default())
    }
}

impl<T: Float + Send + Sync + Debug + Default + Clone> ConfidenceEstimator<T> {
    /// Create new confidence estimator
    pub fn new(method: ConfidenceMethod) -> Self {
        Self {
            method,
            calibration_points: Vec::new(),
            thresholds: ConfidenceThresholds::default(),
            historical_accuracy: VecDeque::new(),
            uncertainty_estimates: HashMap::new(),
        }
    }

    /// Update with new snapshot
    pub fn update(&mut self, _snapshot: &PerformanceSnapshot<T>) -> Result<()> {
        // Update confidence estimation
        Ok(())
    }

    /// Estimate confidence for predictions
    pub fn estimate_confidence(&self, _predictions: &[PerformanceSnapshot<T>]) -> Result<Vec<T>> {
        // Simplified confidence estimation
        Ok(vec![num_traits::cast::cast(0.8).unwrap_or_else(|| T::zero()); _predictions.len()])
    }
}

impl<T: Float + Send + Sync + Debug + Default + Clone> PerformanceContextTracker<T> {
    /// Create new context tracker
    pub fn new() -> Self {
        Self {
            current_context: PerformanceContext::default(),
            context_history: VecDeque::new(),
            context_changes: Vec::new(),
            impact_analysis: HashMap::new(),
        }
    }

    /// Update context
    pub fn update_context(&mut self, new_context: PerformanceContext<T>) -> Result<()> {
        // Track changes
        let changes = self.detect_changes(&self.current_context, &new_context);
        self.context_changes.extend(changes);

        // Update history
        self.context_history.push_back(self.current_context.clone());
        if self.context_history.len() > 100 {
            self.context_history.pop_front();
        }

        self.current_context = new_context;
        Ok(())
    }

    /// Detect context changes
    fn detect_changes(&self, old: &PerformanceContext<T>, new: &PerformanceContext<T>) -> Vec<ContextChange<T>> {
        let mut changes = Vec::new();

        if old.batch_size != new.batch_size {
            changes.push(ContextChange {
                timestamp: SystemTime::now(),
                field: "batch_size".to_string(),
                previous_value: old.batch_size.to_string(),
                new_value: new.batch_size.to_string(),
                impact: None,
                reason: "configuration_change".to_string(),
            });
        }

        // Add more change detection logic here

        changes
    }
}

// Default implementations
impl<T: Float> Default for PerformanceMetrics<T> {
    fn default() -> Self {
        Self {
            loss: T::zero(),
            val_loss: None,
            accuracy: None,
            val_accuracy: None,
            learning_rate: T::from(0.001).unwrap_or(T::zero()),
            gradient_norm: None,
            parameter_norm: None,
            throughput: None,
            convergence_rate: None,
            custom_metrics: HashMap::new(),
        }
    }
}

impl<T: Float> Default for PerformanceContext<T> {
    fn default() -> Self {
        Self {
            model_architecture: "unknown".to_string(),
            dataset_info: "unknown".to_string(),
            batch_size: 32,
            optimizer_type: "adam".to_string(),
            training_regime: "standard".to_string(),
            hardware_config: "unknown".to_string(),
            environment: HashMap::new(),
            hyperparameters: HashMap::new(),
        }
    }
}

impl<T: Float> Default for ResourceUsage<T> {
    fn default() -> Self {
        Self {
            cpu_usage: 0.0,
            memory_usage: 0,
            gpu_memory_usage: None,
            gpu_utilization: None,
            disk_io_rate: None,
            network_io_rate: None,
            power_consumption: None,
        }
    }
}

impl<T: Float> Default for ModelParameters<T> {
    fn default() -> Self {
        Self {
            weights: Array1::zeros(0),
            bias: T::zero(),
            regularization: RegularizationParameters::default(),
            learning_params: HashMap::new(),
        }
    }
}

impl<T: Float> Default for RegularizationParameters<T> {
    fn default() -> Self {
        Self {
            l1_lambda: T::zero(),
            l2_lambda: T::from(0.01).unwrap_or(T::zero()),
            dropout_rate: Some(T::from(0.1).unwrap_or(T::zero())),
            early_stopping: EarlyStoppingConfig::default(),
        }
    }
}

impl<T: Float> Default for ValidationMetrics<T> {
    fn default() -> Self {
        Self {
            mse: T::zero(),
            mae: T::zero(),
            r2_score: T::zero(),
            cv_scores: Vec::new(),
            prediction_intervals: Vec::new(),
            confidence_scores: Vec::new(),
        }
    }
}

impl<T: Float> Default for ConfidenceThresholds<T> {
    fn default() -> Self {
        Self {
            low_threshold: T::from(0.5).unwrap_or(T::zero()),
            medium_threshold: T::from(0.7).unwrap_or(T::zero()),
            high_threshold: T::from(0.85).unwrap_or(T::zero()),
            critical_threshold: T::from(0.95).unwrap_or(T::zero()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_analyzer_creation() {
        let config = PerformanceMonitoringConfig::default();
        let analyzer = PerformanceAnalyzer::<f32>::new(config);
        assert!(analyzer.is_ok());
    }

    #[test]
    fn test_performance_snapshot_recording() {
        let config = PerformanceMonitoringConfig::default();
        let mut analyzer = PerformanceAnalyzer::<f32>::new(config).unwrap();

        let snapshot = PerformanceSnapshot {
            timestamp: SystemTime::now(),
            step: 1,
            loss: 0.5,
            accuracy: Some(0.8),
            training_time: Duration::from_secs(1),
            memory_usage: 1000000,
            gpu_utilization: Some(0.7),
            learning_rate: 0.001,
            gradient_norm: Some(0.1),
            metrics: PerformanceMetrics::default(),
            resource_usage: ResourceUsage::default(),
            context: PerformanceContext::default(),
        };

        let result = analyzer.record_snapshot(snapshot);
        assert!(result.is_ok());
        assert_eq!(analyzer.get_performance_history().len(), 1);
    }

    #[test]
    fn test_performance_model_prediction() {
        let model = PerformanceModel::<f32>::new(ModelType::LinearRegression);
        assert!(model.is_ok());

        let model = model.unwrap();
        let predictions = model.predict(5);
        assert!(predictions.is_ok());
    }

    #[test]
    fn test_confidence_estimator() {
        let estimator = ConfidenceEstimator::<f32>::new(
            ConfidenceMethod::Bootstrap { n_samples: 100 }
        );

        let predictions = vec![PerformanceSnapshot {
            timestamp: SystemTime::now(),
            step: 1,
            loss: 0.5,
            accuracy: Some(0.8),
            training_time: Duration::from_secs(1),
            memory_usage: 1000000,
            gpu_utilization: Some(0.7),
            learning_rate: 0.001,
            gradient_norm: Some(0.1),
            metrics: PerformanceMetrics::default(),
            resource_usage: ResourceUsage::default(),
            context: PerformanceContext::default(),
        }];

        let confidence = estimator.estimate_confidence(&predictions);
        assert!(confidence.is_ok());
    }

    #[test]
    fn test_default_implementations() {
        let metrics = PerformanceMetrics::<f32>::default();
        assert_eq!(metrics.loss, 0.0);

        let context = PerformanceContext::<f32>::default();
        assert_eq!(context.batch_size, 32);

        let resource_usage = ResourceUsage::<f32>::default();
        assert_eq!(resource_usage.cpu_usage, 0.0);
    }
}