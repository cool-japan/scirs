//! Performance tracking and prediction for streaming optimization
//!
//! This module provides comprehensive performance monitoring, trend analysis,
//! and prediction capabilities for streaming optimization scenarios, including
//! real-time metrics collection, statistical analysis, and predictive modeling.

use super::config::*;
use super::optimizer::{Adaptation, AdaptationPriority, AdaptationType, StreamingDataPoint};
use super::resource_management::ResourceUsage;

use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::iter::Sum;
use std::time::{Duration, Instant};

/// Performance snapshot representing metrics at a specific point in time
#[derive(Debug, Clone)]
pub struct PerformanceSnapshot<A: Float> {
    /// Timestamp when snapshot was taken
    pub timestamp: Instant,
    /// Primary loss metric
    pub loss: A,
    /// Accuracy metric (if applicable)
    pub accuracy: Option<A>,
    /// Convergence rate
    pub convergence_rate: Option<A>,
    /// Gradient norm
    pub gradient_norm: Option<A>,
    /// Parameter update magnitude
    pub parameter_update_magnitude: Option<A>,
    /// Data quality statistics
    pub data_statistics: DataStatistics<A>,
    /// Resource usage at snapshot time
    pub resource_usage: ResourceUsage,
    /// Custom performance metrics
    pub custom_metrics: HashMap<String, A>,
}

/// Data quality and distribution statistics
#[derive(Debug, Clone, Default)]
pub struct DataStatistics<A: Float> {
    /// Number of samples in this batch
    pub sample_count: usize,
    /// Feature-wise means
    pub feature_means: ndarray::Array1<A>,
    /// Feature-wise standard deviations
    pub feature_stds: ndarray::Array1<A>,
    /// Average data quality score
    pub average_quality: A,
    /// Timestamp of statistics computation
    pub timestamp: Instant,
}

/// Performance metrics for tracking and analysis
#[derive(Debug, Clone)]
pub enum PerformanceMetric<A: Float> {
    /// Loss function value
    Loss(A),
    /// Classification/regression accuracy
    Accuracy(A),
    /// Rate of convergence
    ConvergenceRate(A),
    /// Gradient magnitude
    GradientNorm(A),
    /// Learning rate effectiveness
    LearningRateEffectiveness(A),
    /// Resource utilization efficiency
    ResourceEfficiency(A),
    /// Data quality score
    DataQuality(A),
    /// Custom metric with name and value
    Custom(String, A),
}

/// Context information for performance evaluation
#[derive(Debug, Clone)]
pub struct PerformanceContext<A: Float> {
    /// Current learning rate
    pub learning_rate: A,
    /// Current batch size
    pub batch_size: usize,
    /// Current buffer size
    pub buffer_size: usize,
    /// Recent drift detection status
    pub drift_detected: bool,
    /// Resource constraints
    pub resource_constraints: ResourceUsage,
    /// Time since last adaptation
    pub time_since_adaptation: Duration,
}

/// Performance tracker for streaming optimization
pub struct PerformanceTracker<A: Float> {
    /// Configuration for performance tracking
    config: PerformanceConfig,
    /// Performance history
    performance_history: VecDeque<PerformanceSnapshot<A>>,
    /// Trend analyzer
    trend_analyzer: PerformanceTrendAnalyzer<A>,
    /// Performance predictor
    predictor: PerformancePredictor<A>,
    /// Performance baseline
    baseline: Option<PerformanceSnapshot<A>>,
    /// Current performance context
    current_context: Option<PerformanceContext<A>>,
    /// Performance improvement tracker
    improvement_tracker: PerformanceImprovementTracker<A>,
    /// Anomaly detector for performance
    performance_anomaly_detector: PerformanceAnomalyDetector<A>,
}

/// Trend analysis for performance metrics
pub struct PerformanceTrendAnalyzer<A: Float> {
    /// Window size for trend analysis
    window_size: usize,
    /// Current trends for different metrics
    trends: HashMap<String, TrendData<A>>,
    /// Trend computation methods
    trend_methods: Vec<TrendMethod>,
}

/// Trend data for a specific metric
#[derive(Debug, Clone)]
pub struct TrendData<A: Float> {
    /// Linear trend slope
    pub slope: A,
    /// Trend correlation coefficient
    pub correlation: A,
    /// Trend volatility
    pub volatility: A,
    /// Trend confidence
    pub confidence: A,
    /// Recent values used for trend calculation
    pub recent_values: VecDeque<A>,
    /// Last update timestamp
    pub last_update: Instant,
}

/// Methods for trend calculation
#[derive(Debug, Clone)]
pub enum TrendMethod {
    /// Linear regression
    LinearRegression,
    /// Moving average
    MovingAverage { window: usize },
    /// Exponential smoothing
    ExponentialSmoothing { alpha: f64 },
    /// Seasonal decomposition
    SeasonalDecomposition,
}

/// Performance predictor using various forecasting methods
pub struct PerformancePredictor<A: Float> {
    /// Prediction methods to use
    prediction_methods: Vec<PredictionMethod>,
    /// Historical predictions for accuracy tracking
    prediction_history: VecDeque<PredictionResult<A>>,
    /// Model accuracy scores
    model_accuracies: HashMap<String, A>,
    /// Ensemble weights for combining predictions
    ensemble_weights: HashMap<String, A>,
}

/// Prediction methods for performance forecasting
#[derive(Debug, Clone)]
pub enum PredictionMethod {
    /// Linear extrapolation
    Linear,
    /// Exponential smoothing
    Exponential { alpha: f64, beta: f64 },
    /// ARIMA model
    ARIMA { p: usize, d: usize, q: usize },
    /// Neural network
    NeuralNetwork { hidden_layers: Vec<usize> },
    /// Ensemble of multiple methods
    Ensemble,
}

/// Result of performance prediction
#[derive(Debug, Clone)]
pub struct PredictionResult<A: Float> {
    /// Predicted metric value
    pub predicted_value: A,
    /// Prediction confidence interval
    pub confidence_interval: (A, A),
    /// Prediction method used
    pub method: String,
    /// Steps ahead predicted
    pub steps_ahead: usize,
    /// Prediction timestamp
    pub timestamp: Instant,
    /// Actual value (filled in later for accuracy assessment)
    pub actual_value: Option<A>,
}

/// Performance improvement tracking
pub struct PerformanceImprovementTracker<A: Float> {
    /// Baseline performance metrics
    baseline_metrics: HashMap<String, A>,
    /// Current improvement rates
    improvement_rates: HashMap<String, A>,
    /// Improvement history
    improvement_history: VecDeque<ImprovementEvent<A>>,
    /// Plateau detection
    plateau_detector: PlateauDetector<A>,
}

/// Performance improvement event
#[derive(Debug, Clone)]
pub struct ImprovementEvent<A: Float> {
    /// Event timestamp
    pub timestamp: Instant,
    /// Metric that improved
    pub metric_name: String,
    /// Improvement magnitude
    pub improvement: A,
    /// Improvement rate (per unit time)
    pub improvement_rate: A,
    /// Context when improvement occurred
    pub context: String,
}

/// Plateau detection for performance metrics
pub struct PlateauDetector<A: Float> {
    /// Window size for plateau detection
    window_size: usize,
    /// Plateau threshold (minimum change for non-plateau)
    plateau_threshold: A,
    /// Recent performance values
    recent_values: VecDeque<A>,
    /// Current plateau status
    is_plateau: bool,
    /// Plateau duration
    plateau_duration: Duration,
    /// Last significant change timestamp
    last_significant_change: Option<Instant>,
}

/// Anomaly detection for performance metrics
pub struct PerformanceAnomalyDetector<A: Float> {
    /// Anomaly detection threshold (standard deviations)
    threshold: A,
    /// Historical statistics for anomaly detection
    historical_stats: HashMap<String, MetricStatistics<A>>,
    /// Recent anomalies detected
    recent_anomalies: VecDeque<PerformanceAnomaly<A>>,
    /// Adaptive threshold adjustment
    adaptive_threshold: bool,
}

/// Statistics for a performance metric
#[derive(Debug, Clone)]
pub struct MetricStatistics<A: Float> {
    /// Running mean
    pub mean: A,
    /// Running variance
    pub variance: A,
    /// Minimum observed value
    pub min_value: A,
    /// Maximum observed value
    pub max_value: A,
    /// Number of observations
    pub count: usize,
    /// Last update timestamp
    pub last_update: Instant,
}

/// Performance anomaly event
#[derive(Debug, Clone)]
pub struct PerformanceAnomaly<A: Float> {
    /// Anomaly timestamp
    pub timestamp: Instant,
    /// Affected metric
    pub metric_name: String,
    /// Observed value
    pub observed_value: A,
    /// Expected value range
    pub expected_range: (A, A),
    /// Anomaly severity
    pub severity: AnomalySeverity,
    /// Anomaly type
    pub anomaly_type: AnomalyType,
}

/// Severity levels for performance anomalies
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum AnomalySeverity {
    /// Minor anomaly
    Minor,
    /// Moderate anomaly requiring attention
    Moderate,
    /// Major anomaly requiring intervention
    Major,
    /// Critical anomaly requiring immediate action
    Critical,
}

/// Types of performance anomalies
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnomalyType {
    /// Value significantly higher than expected
    High,
    /// Value significantly lower than expected
    Low,
    /// Sudden change in trend
    TrendChange,
    /// Unexpected oscillation
    Oscillation,
    /// Performance degradation
    Degradation,
    /// Performance plateau
    Plateau,
}

impl<A: Float + Default + Clone + std::iter::Sum> PerformanceTracker<A> {
    /// Creates a new performance tracker
    pub fn new(config: &StreamingConfig) -> Result<Self, String> {
        let performance_config = config.performance_config.clone();

        let trend_analyzer = PerformanceTrendAnalyzer::new(performance_config.trend_window_size);
        let predictor = PerformancePredictor::new();
        let improvement_tracker = PerformanceImprovementTracker::new();
        let performance_anomaly_detector = PerformanceAnomalyDetector::new(2.0); // 2 sigma threshold

        Ok(Self {
            config: performance_config,
            performance_history: VecDeque::with_capacity(1000),
            trend_analyzer,
            predictor,
            baseline: None,
            current_context: None,
            improvement_tracker,
            performance_anomaly_detector,
        })
    }

    /// Adds a new performance snapshot
    pub fn add_performance(&mut self, snapshot: PerformanceSnapshot<A>) -> Result<(), String> {
        // Store in history
        if self.performance_history.len() >= self.config.history_size {
            self.performance_history.pop_front();
        }
        self.performance_history.push_back(snapshot.clone());

        // Set baseline if this is the first measurement
        if self.baseline.is_none() {
            self.baseline = Some(snapshot.clone());
        }

        // Update trend analysis
        if self.config.enable_trend_analysis {
            self.trend_analyzer.update(&snapshot)?;
        }

        // Update improvement tracking
        self.improvement_tracker.update(&snapshot)?;

        // Check for performance anomalies
        let anomalies = self
            .performance_anomaly_detector
            .check_for_anomalies(&snapshot)?;
        if !anomalies.is_empty() {
            // Handle detected anomalies
            self.handle_performance_anomalies(&anomalies)?;
        }

        // Update predictions if enabled
        if self.config.enable_prediction {
            self.predictor.update_with_actual(&snapshot)?;
        }

        Ok(())
    }

    /// Gets recent performance snapshots
    pub fn get_recent_performance(&self, count: usize) -> Vec<PerformanceSnapshot<A>> {
        self.performance_history
            .iter()
            .rev()
            .take(count)
            .cloned()
            .collect()
    }

    /// Gets recent loss values for trend analysis
    pub fn get_recent_losses(&self, count: usize) -> Vec<A> {
        self.performance_history
            .iter()
            .rev()
            .take(count)
            .map(|snapshot| snapshot.loss)
            .collect()
    }

    /// Predicts future performance
    pub fn predict_performance(
        &mut self,
        steps_ahead: usize,
    ) -> Result<PredictionResult<A>, String> {
        if !self.config.enable_prediction {
            return Err("Performance prediction is disabled".to_string());
        }

        self.predictor
            .predict(steps_ahead, &self.performance_history)
    }

    /// Gets current performance trends
    pub fn get_performance_trends(&self) -> HashMap<String, TrendData<A>> {
        self.trend_analyzer.get_current_trends()
    }

    /// Computes adaptation for performance thresholds
    pub fn apply_threshold_adaptation(&mut self, adaptation: &Adaptation<A>) -> Result<(), String> {
        if adaptation.adaptation_type == AdaptationType::PerformanceThreshold {
            // Adjust anomaly detection thresholds
            let new_threshold = self.performance_anomaly_detector.threshold + adaptation.magnitude;
            self.performance_anomaly_detector
                .update_threshold(new_threshold);
        }
        Ok(())
    }

    /// Handles detected performance anomalies
    fn handle_performance_anomalies(
        &mut self,
        anomalies: &[PerformanceAnomaly<A>],
    ) -> Result<(), String> {
        for anomaly in anomalies {
            match anomaly.severity {
                AnomalySeverity::Critical | AnomalySeverity::Major => {
                    // Log critical anomalies for immediate attention
                    println!("Critical performance anomaly detected: {:?}", anomaly);
                }
                _ => {
                    // Store for analysis
                    self.performance_anomaly_detector
                        .recent_anomalies
                        .push_back(anomaly.clone());
                }
            }
        }
        Ok(())
    }

    /// Resets performance tracking
    pub fn reset(&mut self) -> Result<(), String> {
        self.performance_history.clear();
        self.baseline = None;
        self.current_context = None;
        self.trend_analyzer.reset();
        self.predictor.reset();
        self.improvement_tracker.reset();
        self.performance_anomaly_detector.reset();
        Ok(())
    }

    /// Gets diagnostic information
    pub fn get_diagnostics(&self) -> PerformanceDiagnostics {
        PerformanceDiagnostics {
            history_size: self.performance_history.len(),
            baseline_set: self.baseline.is_some(),
            trends_available: !self.trend_analyzer.trends.is_empty(),
            anomalies_detected: self.performance_anomaly_detector.recent_anomalies.len(),
            plateau_detected: self.improvement_tracker.plateau_detector.is_plateau,
            prediction_accuracy: self.predictor.get_average_accuracy(),
        }
    }
}

impl<A: Float + Default + Clone> PerformanceTrendAnalyzer<A> {
    fn new(window_size: usize) -> Self {
        Self {
            window_size,
            trends: HashMap::new(),
            trend_methods: vec![
                TrendMethod::LinearRegression,
                TrendMethod::MovingAverage {
                    window: window_size / 2,
                },
                TrendMethod::ExponentialSmoothing { alpha: 0.3 },
            ],
        }
    }

    fn update(&mut self, snapshot: &PerformanceSnapshot<A>) -> Result<(), String> {
        // Update trends for different metrics
        self.update_metric_trend("loss", snapshot.loss)?;

        if let Some(accuracy) = snapshot.accuracy {
            self.update_metric_trend("accuracy", accuracy)?;
        }

        if let Some(convergence) = snapshot.convergence_rate {
            self.update_metric_trend("convergence", convergence)?;
        }

        self.compute_trends()?;
        Ok(())
    }

    fn update_metric_trend(&mut self, metric_name: &str, value: A) -> Result<(), String> {
        let trend_data = self
            .trends
            .entry(metric_name.to_string())
            .or_insert_with(|| TrendData {
                slope: A::zero(),
                correlation: A::zero(),
                volatility: A::zero(),
                confidence: A::zero(),
                recent_values: VecDeque::with_capacity(self.window_size),
                last_update: Instant::now(),
            });

        if trend_data.recent_values.len() >= self.window_size {
            trend_data.recent_values.pop_front();
        }
        trend_data.recent_values.push_back(value);
        trend_data.last_update = Instant::now();

        Ok(())
    }

    fn compute_trends(&mut self) -> Result<(), String> {
        for (metric_name, trend_data) in &mut self.trends {
            if trend_data.recent_values.len() >= 3 {
                trend_data.slope = self.compute_slope(&trend_data.recent_values)?;
                trend_data.correlation = self.compute_correlation(&trend_data.recent_values)?;
                trend_data.volatility = self.compute_volatility(&trend_data.recent_values)?;
                trend_data.confidence = self.compute_confidence(&trend_data.recent_values)?;
            }
        }
        Ok(())
    }

    fn compute_slope(&self, values: &VecDeque<A>) -> Result<A, String> {
        if values.len() < 2 {
            return Ok(A::zero());
        }

        let n = A::from(values.len()).unwrap();
        let sum_x = (A::one()..=n).sum::<A>();
        let sum_y = values.iter().cloned().sum::<A>();
        let sum_xy = values
            .iter()
            .enumerate()
            .map(|(i, &y)| A::from(i + 1).unwrap() * y)
            .sum::<A>();
        let sum_x_squared = (A::one()..=n).map(|x| x * x).sum::<A>();

        let denominator = n * sum_x_squared - sum_x * sum_x;
        if denominator == A::zero() {
            return Ok(A::zero());
        }

        let slope = (n * sum_xy - sum_x * sum_y) / denominator;
        Ok(slope)
    }

    fn compute_correlation(&self, values: &VecDeque<A>) -> Result<A, String> {
        if values.len() < 2 {
            return Ok(A::zero());
        }

        // Simplified correlation with time index
        let n = values.len();
        let time_values: Vec<A> = (1..=n).map(|i| A::from(i).unwrap()).collect();
        let value_vec: Vec<A> = values.iter().cloned().collect();

        let mean_time = time_values.iter().cloned().sum::<A>() / A::from(n).unwrap();
        let mean_value = value_vec.iter().cloned().sum::<A>() / A::from(n).unwrap();

        let numerator = time_values
            .iter()
            .zip(value_vec.iter())
            .map(|(&t, &v)| (t - mean_time) * (v - mean_value))
            .sum::<A>();

        let time_variance = time_values
            .iter()
            .map(|&t| (t - mean_time) * (t - mean_time))
            .sum::<A>();

        let value_variance = value_vec
            .iter()
            .map(|&v| (v - mean_value) * (v - mean_value))
            .sum::<A>();

        let denominator = (time_variance * value_variance).sqrt();
        if denominator == A::zero() {
            return Ok(A::zero());
        }

        Ok(numerator / denominator)
    }

    fn compute_volatility(&self, values: &VecDeque<A>) -> Result<A, String> {
        if values.len() < 2 {
            return Ok(A::zero());
        }

        let mean = values.iter().cloned().sum::<A>() / A::from(values.len()).unwrap();
        let variance = values.iter().map(|&v| (v - mean) * (v - mean)).sum::<A>()
            / A::from(values.len()).unwrap();

        Ok(variance.sqrt())
    }

    fn compute_confidence(&self, values: &VecDeque<A>) -> Result<A, String> {
        // Simple confidence based on trend consistency
        if values.len() < 3 {
            return Ok(A::zero());
        }

        let slope = self.compute_slope(values)?;
        let correlation = self.compute_correlation(values)?;

        // Confidence increases with stronger correlation and consistent slope direction
        let confidence = correlation.abs() * (A::one() - (slope.abs() / (slope.abs() + A::one())));
        Ok(confidence)
    }

    fn get_current_trends(&self) -> HashMap<String, TrendData<A>> {
        self.trends.clone()
    }

    fn reset(&mut self) {
        self.trends.clear();
    }
}

impl<A: Float + Default + Clone> PerformancePredictor<A> {
    fn new() -> Self {
        Self {
            prediction_methods: vec![
                PredictionMethod::Linear,
                PredictionMethod::Exponential {
                    alpha: 0.3,
                    beta: 0.1,
                },
            ],
            prediction_history: VecDeque::with_capacity(1000),
            model_accuracies: HashMap::new(),
            ensemble_weights: HashMap::new(),
        }
    }

    fn predict(
        &mut self,
        steps_ahead: usize,
        history: &VecDeque<PerformanceSnapshot<A>>,
    ) -> Result<PredictionResult<A>, String> {
        if history.len() < 2 {
            return Err("Insufficient history for prediction".to_string());
        }

        // Extract loss values for prediction
        let loss_values: Vec<A> = history.iter().map(|s| s.loss).collect();

        // Use linear prediction for simplicity
        let predicted_value = self.linear_prediction(&loss_values, steps_ahead)?;

        // Estimate confidence interval (simplified)
        let recent_volatility = self.compute_recent_volatility(&loss_values)?;
        let confidence_interval = (
            predicted_value - recent_volatility,
            predicted_value + recent_volatility,
        );

        let prediction = PredictionResult {
            predicted_value,
            confidence_interval,
            method: "linear".to_string(),
            steps_ahead,
            timestamp: Instant::now(),
            actual_value: None,
        };

        // Store prediction for later accuracy assessment
        if self.prediction_history.len() >= 1000 {
            self.prediction_history.pop_front();
        }
        self.prediction_history.push_back(prediction.clone());

        Ok(prediction)
    }

    fn linear_prediction(&self, values: &[A], steps_ahead: usize) -> Result<A, String> {
        if values.len() < 2 {
            return Ok(A::zero());
        }

        // Simple linear extrapolation using last two points
        let n = values.len();
        let x1 = A::from(n - 1).unwrap();
        let y1 = values[n - 1];
        let x2 = A::from(n).unwrap();
        let y2 = values[n - 1]; // Use same point for stability

        // Use trend from last few points
        if n >= 3 {
            let slope = (values[n - 1] - values[n - 3]) / A::from(2).unwrap();
            let predicted = values[n - 1] + slope * A::from(steps_ahead).unwrap();
            Ok(predicted)
        } else {
            Ok(values[n - 1])
        }
    }

    fn exponential_prediction(&self, values: &[A], steps_ahead: usize) -> Result<A, String> {
        if values.is_empty() {
            return Ok(A::zero());
        }

        // Simple exponential smoothing
        let alpha = A::from(0.3).unwrap();
        let mut forecast = values[0];

        for &value in values.iter().skip(1) {
            forecast = alpha * value + (A::one() - alpha) * forecast;
        }

        // Project forward (simplified)
        for _ in 0..steps_ahead {
            forecast = forecast * A::from(0.99).unwrap(); // Assume slight improvement
        }

        Ok(forecast)
    }

    fn compute_recent_volatility(&self, values: &[A]) -> Result<A, String> {
        if values.len() < 2 {
            return Ok(A::zero());
        }

        let recent_count = values.len().min(10);
        let recent_values = &values[values.len() - recent_count..];

        let mean = recent_values.iter().cloned().sum::<A>() / A::from(recent_count).unwrap();
        let variance = recent_values
            .iter()
            .map(|&v| (v - mean) * (v - mean))
            .sum::<A>()
            / A::from(recent_count).unwrap();

        Ok(variance.sqrt())
    }

    fn update_with_actual(&mut self, snapshot: &PerformanceSnapshot<A>) -> Result<(), String> {
        // Update prediction accuracy by matching actual values with predictions
        for prediction in &mut self.prediction_history {
            if prediction.actual_value.is_none() {
                let time_diff = snapshot.timestamp.duration_since(prediction.timestamp);
                let expected_duration = Duration::from_secs(prediction.steps_ahead as u64 * 10); // Assume 10s per step

                if time_diff >= expected_duration {
                    prediction.actual_value = Some(snapshot.loss);
                    self.update_accuracy_metrics(prediction)?;
                }
            }
        }

        Ok(())
    }

    fn update_accuracy_metrics(&mut self, prediction: &PredictionResult<A>) -> Result<(), String> {
        if let Some(actual) = prediction.actual_value {
            let error = (prediction.predicted_value - actual).abs();
            let relative_error = error / actual.max(A::from(1e-8).unwrap());

            // Update accuracy for this method
            let accuracy = A::one() - relative_error.min(A::one());
            let method_name = &prediction.method;

            self.model_accuracies.insert(method_name.clone(), accuracy);
        }

        Ok(())
    }

    fn get_average_accuracy(&self) -> f64 {
        if self.model_accuracies.is_empty() {
            return 0.0;
        }

        let sum: A = self.model_accuracies.values().cloned().sum();
        let avg = sum / A::from(self.model_accuracies.len()).unwrap();
        avg.to_f64().unwrap_or(0.0)
    }

    fn reset(&mut self) {
        self.prediction_history.clear();
        self.model_accuracies.clear();
        self.ensemble_weights.clear();
    }
}

impl<A: Float + Default + Clone + Sum> PerformanceImprovementTracker<A> {
    fn new() -> Self {
        Self {
            baseline_metrics: HashMap::new(),
            improvement_rates: HashMap::new(),
            improvement_history: VecDeque::with_capacity(1000),
            plateau_detector: PlateauDetector::new(50, A::from(0.01).unwrap()),
        }
    }

    fn update(&mut self, snapshot: &PerformanceSnapshot<A>) -> Result<(), String> {
        // Update baseline if not set
        if self.baseline_metrics.is_empty() {
            self.baseline_metrics
                .insert("loss".to_string(), snapshot.loss);
            if let Some(accuracy) = snapshot.accuracy {
                self.baseline_metrics
                    .insert("accuracy".to_string(), accuracy);
            }
        }

        // Check for improvements
        if let Some(&baseline_loss) = self.baseline_metrics.get("loss") {
            if snapshot.loss < baseline_loss {
                let improvement = baseline_loss - snapshot.loss;
                let improvement_event = ImprovementEvent {
                    timestamp: snapshot.timestamp,
                    metric_name: "loss".to_string(),
                    improvement,
                    improvement_rate: improvement / A::from(1.0).unwrap(), // Simplified rate
                    context: "optimization_step".to_string(),
                };

                if self.improvement_history.len() >= 1000 {
                    self.improvement_history.pop_front();
                }
                self.improvement_history.push_back(improvement_event);

                // Update baseline
                self.baseline_metrics
                    .insert("loss".to_string(), snapshot.loss);
            }
        }

        // Update plateau detector
        self.plateau_detector.update(snapshot.loss);

        Ok(())
    }

    fn reset(&mut self) {
        self.baseline_metrics.clear();
        self.improvement_rates.clear();
        self.improvement_history.clear();
        self.plateau_detector.reset();
    }
}

impl<A: Float + Default + Clone> PlateauDetector<A> {
    fn new(window_size: usize, threshold: A) -> Self {
        Self {
            window_size,
            plateau_threshold: threshold,
            recent_values: VecDeque::with_capacity(window_size),
            is_plateau: false,
            plateau_duration: Duration::ZERO,
            last_significant_change: None,
        }
    }

    fn update(&mut self, value: A) {
        if self.recent_values.len() >= self.window_size {
            self.recent_values.pop_front();
        }
        self.recent_values.push_back(value);

        if self.recent_values.len() >= self.window_size {
            self.detect_plateau();
        }
    }

    fn detect_plateau(&mut self) {
        if self.recent_values.len() < 2 {
            return;
        }

        let max_val = self.recent_values.iter().cloned().fold(A::zero(), A::max);
        let min_val = self.recent_values.iter().cloned().fold(A::zero(), A::min);
        let range = max_val - min_val;

        let was_plateau = self.is_plateau;
        self.is_plateau = range < self.plateau_threshold;

        if self.is_plateau && !was_plateau {
            self.plateau_duration = Duration::ZERO;
        } else if self.is_plateau {
            self.plateau_duration += Duration::from_secs(1); // Simplified
        } else if !self.is_plateau {
            self.last_significant_change = Some(Instant::now());
            self.plateau_duration = Duration::ZERO;
        }
    }

    fn reset(&mut self) {
        self.recent_values.clear();
        self.is_plateau = false;
        self.plateau_duration = Duration::ZERO;
        self.last_significant_change = None;
    }
}

impl<A: Float + Default + Clone + Sum> PerformanceAnomalyDetector<A> {
    fn new(threshold: f64) -> Self {
        Self {
            threshold: A::from(threshold).unwrap(),
            historical_stats: HashMap::new(),
            recent_anomalies: VecDeque::with_capacity(100),
            adaptive_threshold: true,
        }
    }

    fn check_for_anomalies(
        &mut self,
        snapshot: &PerformanceSnapshot<A>,
    ) -> Result<Vec<PerformanceAnomaly<A>>, String> {
        let mut anomalies = Vec::new();

        // Check loss anomaly
        let loss_anomaly = self.check_metric_anomaly("loss", snapshot.loss, snapshot.timestamp)?;
        if let Some(anomaly) = loss_anomaly {
            anomalies.push(anomaly);
        }

        // Check accuracy anomaly if available
        if let Some(accuracy) = snapshot.accuracy {
            let accuracy_anomaly =
                self.check_metric_anomaly("accuracy", accuracy, snapshot.timestamp)?;
            if let Some(anomaly) = accuracy_anomaly {
                anomalies.push(anomaly);
            }
        }

        Ok(anomalies)
    }

    fn check_metric_anomaly(
        &mut self,
        metric_name: &str,
        value: A,
        timestamp: Instant,
    ) -> Result<Option<PerformanceAnomaly<A>>, String> {
        // Update statistics for this metric
        let stats = self
            .historical_stats
            .entry(metric_name.to_string())
            .or_insert_with(|| MetricStatistics {
                mean: value,
                variance: A::zero(),
                min_value: value,
                max_value: value,
                count: 0,
                last_update: timestamp,
            });

        // Update running statistics
        stats.count += 1;
        let delta = value - stats.mean;
        stats.mean = stats.mean + delta / A::from(stats.count).unwrap();
        let delta2 = value - stats.mean;
        stats.variance = stats.variance + delta * delta2;
        stats.min_value = stats.min_value.min(value);
        stats.max_value = stats.max_value.max(value);
        stats.last_update = timestamp;

        // Check for anomaly after sufficient samples
        if stats.count >= 10 {
            let std_dev = (stats.variance / A::from(stats.count - 1).unwrap()).sqrt();
            let z_score = (value - stats.mean) / std_dev.max(A::from(1e-8).unwrap());

            if z_score.abs() > self.threshold {
                let severity = if z_score.abs() > A::from(3.0).unwrap() {
                    AnomalySeverity::Critical
                } else if z_score.abs() > A::from(2.5).unwrap() {
                    AnomalySeverity::Major
                } else {
                    AnomalySeverity::Moderate
                };

                let anomaly_type = if z_score > A::zero() {
                    AnomalyType::High
                } else {
                    AnomalyType::Low
                };

                let expected_range = (
                    stats.mean - self.threshold * std_dev,
                    stats.mean + self.threshold * std_dev,
                );

                let anomaly = PerformanceAnomaly {
                    timestamp,
                    metric_name: metric_name.to_string(),
                    observed_value: value,
                    expected_range,
                    severity,
                    anomaly_type,
                };

                return Ok(Some(anomaly));
            }
        }

        Ok(None)
    }

    fn update_threshold(&mut self, new_threshold: A) {
        self.threshold = new_threshold;
    }

    fn reset(&mut self) {
        self.historical_stats.clear();
        self.recent_anomalies.clear();
    }
}

/// Diagnostic information for performance tracking
#[derive(Debug, Clone)]
pub struct PerformanceDiagnostics {
    pub history_size: usize,
    pub baseline_set: bool,
    pub trends_available: bool,
    pub anomalies_detected: usize,
    pub plateau_detected: bool,
    pub prediction_accuracy: f64,
}
