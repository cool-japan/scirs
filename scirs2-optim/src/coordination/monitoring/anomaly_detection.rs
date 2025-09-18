//! Anomaly detection for optimization processes
//!
//! This module provides comprehensive anomaly detection capabilities for optimization
//! workflows, including statistical outlier detection, ML-based anomaly classification,
//! and real-time alerting systems.

#![allow(dead_code)]

use std::collections::{HashMap, VecDeque};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use num_traits::Float;

/// Anomaly classification types
#[derive(Debug, Clone, PartialEq)]
pub enum AnomalyType {
    StatisticalOutlier,
    TrendAnomaly,
    PerformanceAnomaly,
    ConvergenceAnomaly,
    ResourceAnomaly,
    PatternAnomaly,
    SeasonalAnomaly,
    SystemAnomaly,
}

/// Severity levels for anomalies
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum AnomalySeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Anomaly detection result
#[derive(Debug, Clone)]
pub struct AnomalyResult<T: Float> {
    pub is_anomaly: bool,
    pub anomaly_type: AnomalyType,
    pub severity: AnomalySeverity,
    pub confidence: T,
    pub anomaly_score: T,
    pub timestamp: Instant,
    pub context: AnomalyContext<T>,
    pub suggested_actions: Vec<String>,
}

/// Context information for anomalies
#[derive(Debug, Clone)]
pub struct AnomalyContext<T: Float> {
    pub baseline_mean: T,
    pub baseline_std: T,
    pub current_value: T,
    pub deviation_magnitude: T,
    pub trend_deviation: T,
    pub pattern_match_score: T,
    pub historical_frequency: T,
}

/// Anomaly alert with detailed information
#[derive(Debug, Clone)]
pub struct AnomalyAlert<T: Float> {
    pub id: String,
    pub anomaly_type: AnomalyType,
    pub severity: AnomalySeverity,
    pub timestamp: Instant,
    pub message: String,
    pub confidence: T,
    pub data_point: T,
    pub context: AnomalyContext<T>,
    pub suggested_actions: Vec<String>,
    pub acknowledged: bool,
    pub resolved: bool,
}

/// Configuration for anomaly detection
#[derive(Debug, Clone)]
pub struct AnomalyConfig<T: Float> {
    pub statistical_threshold: T,
    pub trend_sensitivity: T,
    pub pattern_window: usize,
    pub baseline_window: usize,
    pub min_data_points: usize,
    pub confidence_threshold: T,
    pub enable_adaptive_thresholds: bool,
    pub seasonal_analysis: bool,
    pub outlier_methods: Vec<OutlierMethod>,
    pub alert_cooldown: Duration,
}

impl<T: Float> Default for AnomalyConfig<T> {
    fn default() -> Self {
        Self {
            statistical_threshold: num_traits::cast::cast(2.5).unwrap_or_else(|| T::zero()),
            trend_sensitivity: num_traits::cast::cast(0.1).unwrap_or_else(|| T::zero()),
            pattern_window: 50,
            baseline_window: 100,
            min_data_points: 10,
            confidence_threshold: num_traits::cast::cast(0.8).unwrap_or_else(|| T::zero()),
            enable_adaptive_thresholds: true,
            seasonal_analysis: false,
            outlier_methods: vec![
                OutlierMethod::ZScore,
                OutlierMethod::IQR,
                OutlierMethod::IsolationForest,
                OutlierMethod::LocalOutlierFactor,
            ],
            alert_cooldown: Duration::from_secs(60),
        }
    }
}

/// Available outlier detection methods
#[derive(Debug, Clone, PartialEq)]
pub enum OutlierMethod {
    ZScore,
    ModifiedZScore,
    IQR,
    Hampel,
    IsolationForest,
    LocalOutlierFactor,
    OneClassSVM,
    DBSCAN,
}

/// Primary anomaly detector
pub struct AnomalyDetector<T: Float> {
    config: AnomalyConfig<T>,
    analyzer: AnomalyAnalyzer<T>,
    classifier: AnomalyClassifier<T>,
    outlier_detector: OutlierDetector<T>,
    reporter: AnomalyReporter<T>,
    data_history: VecDeque<(Instant, T)>,
    baseline_stats: BaselineStats<T>,
    pattern_memory: PatternMemory<T>,
    adaptive_thresholds: AdaptiveThresholds<T>,
    _phantom: PhantomData<T>,
}

impl<T: Float> AnomalyDetector<T> {
    pub fn new(config: AnomalyConfig<T>) -> Self {
        Self {
            analyzer: AnomalyAnalyzer::new(config.clone()),
            classifier: AnomalyClassifier::new(config.clone()),
            outlier_detector: OutlierDetector::new(config.clone()),
            reporter: AnomalyReporter::new(config.clone()),
            data_history: VecDeque::with_capacity(config.baseline_window * 2),
            baseline_stats: BaselineStats::new(),
            pattern_memory: PatternMemory::new(config.pattern_window),
            adaptive_thresholds: AdaptiveThresholds::new(),
            config,
            _phantom: PhantomData,
        }
    }

    pub fn detect_anomaly(&mut self, value: T) -> AnomalyResult<T> {
        let timestamp = Instant::now();
        self.data_history.push_back((timestamp, value));
        
        if self.data_history.len() > self.config.baseline_window * 2 {
            self.data_history.pop_front();
        }

        // Update baseline statistics
        self.update_baseline_stats();

        // Skip detection if insufficient data
        if self.data_history.len() < self.config.min_data_points {
            return AnomalyResult {
                is_anomaly: false,
                anomaly_type: AnomalyType::StatisticalOutlier,
                severity: AnomalySeverity::Low,
                confidence: T::zero(),
                anomaly_score: T::zero(),
                timestamp,
                context: self.create_context(value),
                suggested_actions: vec![],
            };
        }

        // Multi-method anomaly detection
        let statistical_result = self.statistical_anomaly_detection(value);
        let trend_result = self.trend_anomaly_detection(value);
        let pattern_result = self.pattern_anomaly_detection(value);
        let outlier_result = self.outlier_detector.detect_outlier(value, &self.data_history);

        // Combine results using weighted ensemble
        let combined_result = self.combine_detection_results(
            statistical_result,
            trend_result,
            pattern_result,
            outlier_result,
            timestamp,
            value,
        );

        // Update pattern memory
        self.pattern_memory.update(value, combined_result.is_anomaly);

        // Update adaptive thresholds
        if self.config.enable_adaptive_thresholds {
            self.adaptive_thresholds.update(value, combined_result.is_anomaly);
        }

        combined_result
    }

    fn update_baseline_stats(&mut self) {
        if self.data_history.is_empty() {
            return;
        }

        let values: Vec<T> = self.data_history.iter().map(|(_, v)| *v).collect();
        
        // Compute baseline window statistics
        let baseline_window = self.config.baseline_window.min(values.len());
        let baseline_values = if values.len() > baseline_window {
            &values[values.len() - baseline_window..]
        } else {
            &values
        };

        self.baseline_stats.update(baseline_values);
    }

    fn statistical_anomaly_detection(&self, value: T) -> AnomalyResult<T> {
        let z_score = if self.baseline_stats.std_dev > T::epsilon() {
            (value - self.baseline_stats.mean).abs() / self.baseline_stats.std_dev
        } else {
            T::zero()
        };

        let threshold = if self.config.enable_adaptive_thresholds {
            self.adaptive_thresholds.get_statistical_threshold()
        } else {
            self.config.statistical_threshold
        };

        let is_anomaly = z_score > threshold;
        let confidence = if is_anomaly {
            (z_score / threshold).min(T::one())
        } else {
            T::zero()
        };

        let severity = self.determine_severity(z_score, threshold);

        AnomalyResult {
            is_anomaly,
            anomaly_type: AnomalyType::StatisticalOutlier,
            severity,
            confidence,
            anomaly_score: z_score,
            timestamp: Instant::now(),
            context: self.create_context(value),
            suggested_actions: self.get_statistical_actions(z_score, threshold),
        }
    }

    fn trend_anomaly_detection(&self, value: T) -> AnomalyResult<T> {
        if self.data_history.len() < 3 {
            return self.create_no_anomaly_result(value, AnomalyType::TrendAnomaly);
        }

        let recent_values: Vec<T> = self.data_history.iter()
            .rev()
            .take(10)
            .map(|(_, v)| *v)
            .collect();

        let trend = self.compute_trend(&recent_values);
        let expected_value = self.predict_next_value(&recent_values, trend);
        let trend_deviation = (value - expected_value).abs();

        let threshold = self.baseline_stats.std_dev * self.config.trend_sensitivity;
        let is_anomaly = trend_deviation > threshold && threshold > T::epsilon();
        
        let confidence = if is_anomaly && threshold > T::epsilon() {
            (trend_deviation / threshold).min(T::one())
        } else {
            T::zero()
        };

        let severity = self.determine_severity(trend_deviation, threshold);

        AnomalyResult {
            is_anomaly,
            anomaly_type: AnomalyType::TrendAnomaly,
            severity,
            confidence,
            anomaly_score: trend_deviation,
            timestamp: Instant::now(),
            context: self.create_context(value),
            suggested_actions: self.get_trend_actions(trend_deviation, threshold),
        }
    }

    fn pattern_anomaly_detection(&self, value: T) -> AnomalyResult<T> {
        let pattern_score = self.pattern_memory.compute_pattern_score(value);
        let threshold = num_traits::cast::cast(0.3).unwrap_or_else(|| T::zero()); // Pattern similarity threshold
        
        let is_anomaly = pattern_score < threshold;
        let confidence = if is_anomaly {
            T::one() - pattern_score
        } else {
            T::zero()
        };

        let severity = if pattern_score < num_traits::cast::cast(0.1).unwrap_or_else(|| T::zero()) {
            AnomalySeverity::High
        } else if pattern_score < num_traits::cast::cast(0.2).unwrap_or_else(|| T::zero()) {
            AnomalySeverity::Medium
        } else {
            AnomalySeverity::Low
        };

        AnomalyResult {
            is_anomaly,
            anomaly_type: AnomalyType::PatternAnomaly,
            severity,
            confidence,
            anomaly_score: T::one() - pattern_score,
            timestamp: Instant::now(),
            context: self.create_context(value),
            suggested_actions: self.get_pattern_actions(pattern_score),
        }
    }

    fn combine_detection_results(
        &self,
        statistical: AnomalyResult<T>,
        trend: AnomalyResult<T>,
        pattern: AnomalyResult<T>,
        outlier: AnomalyResult<T>,
        timestamp: Instant,
        value: T,
    ) -> AnomalyResult<T> {
        // Weighted combination of results
        let weights = [num_traits::cast::cast(0.3).unwrap_or_else(|| T::zero()), num_traits::cast::cast(0.25).unwrap_or_else(|| T::zero()), num_traits::cast::cast(0.25).unwrap_or_else(|| T::zero()), num_traits::cast::cast(0.2).unwrap_or_else(|| T::zero())];
        let results = [&statistical, &trend, &pattern, &outlier];

        let combined_confidence = results.iter()
            .zip(weights.iter())
            .map(|(result, &weight)| result.confidence * weight)
            .fold(T::zero(), |acc, x| acc + x);

        let combined_score = results.iter()
            .zip(weights.iter())
            .map(|(result, &weight)| result.anomaly_score * weight)
            .fold(T::zero(), |acc, x| acc + x);

        let is_anomaly = combined_confidence > self.config.confidence_threshold;

        // Select the most significant anomaly type
        let anomaly_type = results.iter()
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
            .map(|r| r.anomaly_type.clone())
            .unwrap_or(AnomalyType::StatisticalOutlier);

        let severity = self.determine_severity(combined_score, self.config.statistical_threshold);

        let mut suggested_actions = Vec::new();
        for result in &results {
            if result.is_anomaly {
                suggested_actions.extend(result.suggested_actions.clone());
            }
        }

        AnomalyResult {
            is_anomaly,
            anomaly_type,
            severity,
            confidence: combined_confidence,
            anomaly_score: combined_score,
            timestamp,
            context: self.create_context(value),
            suggested_actions,
        }
    }

    fn create_no_anomaly_result(&self, value: T, anomaly_type: AnomalyType) -> AnomalyResult<T> {
        AnomalyResult {
            is_anomaly: false,
            anomaly_type,
            severity: AnomalySeverity::Low,
            confidence: T::zero(),
            anomaly_score: T::zero(),
            timestamp: Instant::now(),
            context: self.create_context(value),
            suggested_actions: vec![],
        }
    }

    fn create_context(&self, value: T) -> AnomalyContext<T> {
        AnomalyContext {
            baseline_mean: self.baseline_stats.mean,
            baseline_std: self.baseline_stats.std_dev,
            current_value: value,
            deviation_magnitude: (value - self.baseline_stats.mean).abs(),
            trend_deviation: T::zero(), // Computed separately
            pattern_match_score: self.pattern_memory.compute_pattern_score(value),
            historical_frequency: T::zero(), // Computed separately
        }
    }

    fn compute_trend(&self, values: &[T]) -> T {
        if values.len() < 2 {
            return T::zero();
        }

        let n = T::from(values.len()).unwrap();
        let sum_x = (T::zero()..n).fold(T::zero(), |acc, i| acc + i);
        let sum_y = values.iter().fold(T::zero(), |acc, &y| acc + y);
        let sum_xy = values.iter().enumerate()
            .fold(T::zero(), |acc, (i, &y)| acc + num_traits::cast::cast(i).unwrap_or_else(|| T::zero()) * y);
        let sum_x2 = (T::zero()..n).fold(T::zero(), |acc, i| acc + i * i);

        let denominator = n * sum_x2 - sum_x * sum_x;
        if denominator.abs() < T::epsilon() {
            return T::zero();
        }

        (n * sum_xy - sum_x * sum_y) / denominator
    }

    fn predict_next_value(&self, values: &[T], trend: T) -> T {
        if values.is_empty() {
            return T::zero();
        }

        let last_value = values[values.len() - 1];
        last_value + trend
    }

    fn determine_severity(&self, score: T, threshold: T) -> AnomalySeverity {
        if threshold < T::epsilon() {
            return AnomalySeverity::Low;
        }

        let ratio = score / threshold;
        if ratio > num_traits::cast::cast(3.0).unwrap_or_else(|| T::zero()) {
            AnomalySeverity::Critical
        } else if ratio > num_traits::cast::cast(2.0).unwrap_or_else(|| T::zero()) {
            AnomalySeverity::High
        } else if ratio > num_traits::cast::cast(1.5).unwrap_or_else(|| T::zero()) {
            AnomalySeverity::Medium
        } else {
            AnomalySeverity::Low
        }
    }

    fn get_statistical_actions(&self, z_score: T, threshold: T) -> Vec<String> {
        let mut actions = Vec::new();
        
        if z_score > threshold * num_traits::cast::cast(2.0).unwrap_or_else(|| T::zero()) {
            actions.push("Investigate data source for potential errors".to_string());
            actions.push("Check for system or measurement anomalies".to_string());
        } else if z_score > threshold {
            actions.push("Monitor closely for recurring pattern".to_string());
            actions.push("Consider adjusting optimization parameters".to_string());
        }

        actions
    }

    fn get_trend_actions(&self, deviation: T, threshold: T) -> Vec<String> {
        let mut actions = Vec::new();
        
        if deviation > threshold {
            actions.push("Analyze trend disruption causes".to_string());
            actions.push("Consider adaptive optimization strategies".to_string());
            actions.push("Review recent parameter changes".to_string());
        }

        actions
    }

    fn get_pattern_actions(&self, pattern_score: T) -> Vec<String> {
        let mut actions = Vec::new();
        
        if pattern_score < num_traits::cast::cast(0.2).unwrap_or_else(|| T::zero()) {
            actions.push("Investigate unusual behavioral patterns".to_string());
            actions.push("Check for optimization convergence issues".to_string());
        }

        actions
    }

    pub fn get_config(&self) -> &AnomalyConfig<T> {
        &self.config
    }

    pub fn update_config(&mut self, config: AnomalyConfig<T>) {
        self.config = config;
    }

    pub fn reset(&mut self) {
        self.data_history.clear();
        self.baseline_stats = BaselineStats::new();
        self.pattern_memory = PatternMemory::new(self.config.pattern_window);
        self.adaptive_thresholds = AdaptiveThresholds::new();
    }
}

/// Baseline statistics computation
#[derive(Debug)]
struct BaselineStats<T: Float> {
    mean: T,
    std_dev: T,
    min: T,
    max: T,
    median: T,
}

impl<T: Float> BaselineStats<T> {
    fn new() -> Self {
        Self {
            mean: T::zero(),
            std_dev: T::zero(),
            min: T::infinity(),
            max: T::neg_infinity(),
            median: T::zero(),
        }
    }

    fn update(&mut self, values: &[T]) {
        if values.is_empty() {
            return;
        }

        self.mean = values.iter().fold(T::zero(), |acc, &x| acc + x) / T::from(values.len()).unwrap();
        
        let variance = values.iter()
            .map(|&x| (x - self.mean) * (x - self.mean))
            .fold(T::zero(), |acc, x| acc + x) / T::from(values.len()).unwrap();
        
        self.std_dev = variance.sqrt();
        
        self.min = values.iter().fold(T::infinity(), |acc, &x| acc.min(x));
        self.max = values.iter().fold(T::neg_infinity(), |acc, &x| acc.max(x));
        
        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = sorted_values.len() / 2;
        self.median = if sorted_values.len() % 2 == 0 {
            (sorted_values[mid - 1] + sorted_values[mid]) / num_traits::cast::cast(2.0).unwrap_or_else(|| T::zero())
        } else {
            sorted_values[mid]
        };
    }
}

/// Pattern memory for anomaly detection
#[derive(Debug)]
struct PatternMemory<T: Float> {
    patterns: VecDeque<T>,
    window_size: usize,
    pattern_weights: Vec<T>,
}

impl<T: Float> PatternMemory<T> {
    fn new(window_size: usize) -> Self {
        Self {
            patterns: VecDeque::with_capacity(window_size),
            window_size,
            pattern_weights: vec![T::one(); window_size],
        }
    }

    fn update(&mut self, value: T, is_anomaly: bool) {
        self.patterns.push_back(value);
        if self.patterns.len() > self.window_size {
            self.patterns.pop_front();
        }

        // Update pattern weights based on anomaly feedback
        if is_anomaly && !self.pattern_weights.is_empty() {
            let last_idx = self.pattern_weights.len() - 1;
            self.pattern_weights[last_idx] = self.pattern_weights[last_idx] * num_traits::cast::cast(0.9).unwrap_or_else(|| T::zero());
        }
    }

    fn compute_pattern_score(&self, value: T) -> T {
        if self.patterns.is_empty() {
            return T::one();
        }

        // Compute similarity to historical patterns
        let similarities: Vec<T> = self.patterns.iter()
            .map(|&pattern| {
                let diff = (value - pattern).abs();
                (-diff).exp()
            })
            .collect();

        let weighted_similarity = similarities.iter()
            .zip(self.pattern_weights.iter())
            .map(|(&sim, &weight)| sim * weight)
            .fold(T::zero(), |acc, x| acc + x);

        let total_weight = self.pattern_weights.iter().fold(T::zero(), |acc, &w| acc + w);
        
        if total_weight > T::epsilon() {
            weighted_similarity / total_weight
        } else {
            T::zero()
        }
    }
}

/// Adaptive threshold management
#[derive(Debug)]
struct AdaptiveThresholds<T: Float> {
    statistical_threshold: T,
    trend_threshold: T,
    pattern_threshold: T,
    adaptation_rate: T,
    false_positive_count: usize,
    false_negative_count: usize,
}

impl<T: Float> AdaptiveThresholds<T> {
    fn new() -> Self {
        Self {
            statistical_threshold: num_traits::cast::cast(2.5).unwrap_or_else(|| T::zero()),
            trend_threshold: num_traits::cast::cast(0.1).unwrap_or_else(|| T::zero()),
            pattern_threshold: num_traits::cast::cast(0.3).unwrap_or_else(|| T::zero()),
            adaptation_rate: num_traits::cast::cast(0.01).unwrap_or_else(|| T::zero()),
            false_positive_count: 0,
            false_negative_count: 0,
        }
    }

    fn update(&mut self, _value: T, detected_anomaly: bool) {
        // Simplified adaptive threshold update
        if detected_anomaly {
            self.statistical_threshold = self.statistical_threshold * (T::one() + self.adaptation_rate);
        } else {
            self.statistical_threshold = self.statistical_threshold * (T::one() - self.adaptation_rate / num_traits::cast::cast(2.0).unwrap_or_else(|| T::zero()));
        }

        // Keep thresholds within reasonable bounds
        self.statistical_threshold = self.statistical_threshold
            .max(num_traits::cast::cast(1.5).unwrap_or_else(|| T::zero()))
            .min(num_traits::cast::cast(4.0).unwrap_or_else(|| T::zero()));
    }

    fn get_statistical_threshold(&self) -> T {
        self.statistical_threshold
    }
}

/// Advanced anomaly analyzer
pub struct AnomalyAnalyzer<T: Float> {
    config: AnomalyConfig<T>,
    time_series_analyzer: TimeSeriesAnalyzer<T>,
    frequency_analyzer: FrequencyAnalyzer<T>,
    _phantom: PhantomData<T>,
}

impl<T: Float> AnomalyAnalyzer<T> {
    pub fn new(config: AnomalyConfig<T>) -> Self {
        Self {
            time_series_analyzer: TimeSeriesAnalyzer::new(config.clone()),
            frequency_analyzer: FrequencyAnalyzer::new(config.clone()),
            config,
            _phantom: PhantomData,
        }
    }

    pub fn analyze_time_series(&self, data: &[(Instant, T)]) -> Vec<AnomalyResult<T>> {
        self.time_series_analyzer.analyze(data)
    }

    pub fn analyze_frequency_domain(&self, values: &[T]) -> Vec<AnomalyResult<T>> {
        self.frequency_analyzer.analyze(values)
    }

    pub fn seasonal_decomposition(&self, data: &[(Instant, T)]) -> SeasonalDecomposition<T> {
        self.time_series_analyzer.seasonal_decomposition(data)
    }
}

/// Time series analysis for anomaly detection
struct TimeSeriesAnalyzer<T: Float> {
    config: AnomalyConfig<T>,
    _phantom: PhantomData<T>,
}

impl<T: Float> TimeSeriesAnalyzer<T> {
    fn new(config: AnomalyConfig<T>) -> Self {
        Self {
            config,
            _phantom: PhantomData,
        }
    }

    fn analyze(&self, data: &[(Instant, T)]) -> Vec<AnomalyResult<T>> {
        let mut results = Vec::new();
        
        if data.len() < self.config.min_data_points {
            return results;
        }

        // STL decomposition for trend and seasonal analysis
        let decomposition = self.seasonal_decomposition(data);
        
        // Detect anomalies in residuals
        for (i, &(timestamp, value)) in data.iter().enumerate() {
            if i < decomposition.residuals.len() {
                let residual = decomposition.residuals[i];
                let threshold = self.compute_residual_threshold(&decomposition.residuals);
                
                if residual.abs() > threshold {
                    results.push(AnomalyResult {
                        is_anomaly: true,
                        anomaly_type: AnomalyType::SeasonalAnomaly,
                        severity: self.determine_severity(residual.abs(), threshold),
                        confidence: (residual.abs() / threshold).min(T::one()),
                        anomaly_score: residual.abs(),
                        timestamp,
                        context: AnomalyContext {
                            baseline_mean: T::zero(),
                            baseline_std: threshold,
                            current_value: value,
                            deviation_magnitude: residual.abs(),
                            trend_deviation: T::zero(),
                            pattern_match_score: T::zero(),
                            historical_frequency: T::zero(),
                        },
                        suggested_actions: vec!["Investigate seasonal pattern disruption".to_string()],
                    });
                }
            }
        }

        results
    }

    fn seasonal_decomposition(&self, data: &[(Instant, T)]) -> SeasonalDecomposition<T> {
        let values: Vec<T> = data.iter().map(|(_, v)| *v).collect();
        
        if values.len() < 10 {
            return SeasonalDecomposition {
                trend: values.clone(),
                seasonal: vec![T::zero(); values.len()],
                residuals: vec![T::zero(); values.len()],
            };
        }

        // Simplified seasonal decomposition
        let window_size = (values.len() / 4).max(3);
        let mut trend = Vec::with_capacity(values.len());
        
        // Moving average for trend
        for i in 0..values.len() {
            let start = i.saturating_sub(window_size / 2);
            let end = (i + window_size / 2 + 1).min(values.len());
            let window_mean = values[start..end].iter().fold(T::zero(), |acc, &x| acc + x) 
                            / num_traits::cast::cast(end - start).unwrap_or_else(|| T::zero());
            trend.push(window_mean);
        }

        // Detrend
        let detrended: Vec<T> = values.iter().zip(trend.iter())
            .map(|(&val, &tr)| val - tr)
            .collect();

        // Simple seasonal component (placeholder)
        let seasonal = vec![T::zero(); values.len()];
        
        // Residuals
        let residuals: Vec<T> = detrended.iter().zip(seasonal.iter())
            .map(|(&det, &seas)| det - seas)
            .collect();

        SeasonalDecomposition {
            trend,
            seasonal,
            residuals,
        }
    }

    fn compute_residual_threshold(&self, residuals: &[T]) -> T {
        if residuals.is_empty() {
            return T::zero();
        }

        let mean = residuals.iter().fold(T::zero(), |acc, &x| acc + x) / T::from(residuals.len()).unwrap();
        let variance = residuals.iter()
            .map(|&x| (x - mean) * (x - mean))
            .fold(T::zero(), |acc, x| acc + x) / T::from(residuals.len()).unwrap();

        variance.sqrt() * self.config.statistical_threshold
    }

    fn determine_severity(&self, score: T, threshold: T) -> AnomalySeverity {
        if threshold < T::epsilon() {
            return AnomalySeverity::Low;
        }

        let ratio = score / threshold;
        if ratio > num_traits::cast::cast(3.0).unwrap_or_else(|| T::zero()) {
            AnomalySeverity::Critical
        } else if ratio > num_traits::cast::cast(2.0).unwrap_or_else(|| T::zero()) {
            AnomalySeverity::High
        } else if ratio > num_traits::cast::cast(1.5).unwrap_or_else(|| T::zero()) {
            AnomalySeverity::Medium
        } else {
            AnomalySeverity::Low
        }
    }
}

/// Seasonal decomposition result
#[derive(Debug)]
struct SeasonalDecomposition<T: Float> {
    trend: Vec<T>,
    seasonal: Vec<T>,
    residuals: Vec<T>,
}

/// Frequency domain analysis for anomaly detection
struct FrequencyAnalyzer<T: Float> {
    config: AnomalyConfig<T>,
    _phantom: PhantomData<T>,
}

impl<T: Float> FrequencyAnalyzer<T> {
    fn new(config: AnomalyConfig<T>) -> Self {
        Self {
            config,
            _phantom: PhantomData,
        }
    }

    fn analyze(&self, values: &[T]) -> Vec<AnomalyResult<T>> {
        let mut results = Vec::new();
        
        if values.len() < self.config.min_data_points {
            return results;
        }

        // Simplified frequency analysis using autocorrelation
        let autocorr = self.compute_autocorrelation(values);
        let anomaly_threshold = num_traits::cast::cast(0.1).unwrap_or_else(|| T::zero());

        for (lag, &corr) in autocorr.iter().enumerate() {
            if corr.abs() > anomaly_threshold && lag > 0 {
                results.push(AnomalyResult {
                    is_anomaly: true,
                    anomaly_type: AnomalyType::PatternAnomaly,
                    severity: AnomalySeverity::Medium,
                    confidence: corr.abs(),
                    anomaly_score: corr.abs(),
                    timestamp: Instant::now(),
                    context: AnomalyContext {
                        baseline_mean: T::zero(),
                        baseline_std: T::zero(),
                        current_value: T::zero(),
                        deviation_magnitude: corr.abs(),
                        trend_deviation: T::zero(),
                        pattern_match_score: corr.abs(),
                        historical_frequency: num_traits::cast::cast(lag).unwrap_or_else(|| T::zero()),
                    },
                    suggested_actions: vec![
                        format!("Investigate periodic pattern with lag {}", lag)
                    ],
                });
            }
        }

        results
    }

    fn compute_autocorrelation(&self, values: &[T]) -> Vec<T> {
        let n = values.len();
        let mut autocorr = vec![T::zero(); n / 2];
        
        let mean = values.iter().fold(T::zero(), |acc, &x| acc + x) / num_traits::cast::cast(n).unwrap_or_else(|| T::zero());
        
        for lag in 0..autocorr.len() {
            let mut sum = T::zero();
            let mut count = 0;
            
            for i in lag..n {
                sum = sum + (values[i] - mean) * (values[i - lag] - mean);
                count += 1;
            }
            
            if count > 0 {
                autocorr[lag] = sum / num_traits::cast::cast(count).unwrap_or_else(|| T::zero());
            }
        }

        // Normalize by lag-0 autocorrelation
        if autocorr[0] > T::epsilon() {
            for corr in &mut autocorr {
                *corr = *corr / autocorr[0];
            }
        }

        autocorr
    }
}

/// Anomaly classification system
pub struct AnomalyClassifier<T: Float> {
    config: AnomalyConfig<T>,
    classification_models: HashMap<AnomalyType, ClassificationModel<T>>,
    _phantom: PhantomData<T>,
}

impl<T: Float> AnomalyClassifier<T> {
    pub fn new(config: AnomalyConfig<T>) -> Self {
        let mut classifier = Self {
            config,
            classification_models: HashMap::new(),
            _phantom: PhantomData,
        };
        
        classifier.initialize_models();
        classifier
    }

    fn initialize_models(&mut self) {
        // Initialize classification models for different anomaly types
        let anomaly_types = [
            AnomalyType::StatisticalOutlier,
            AnomalyType::TrendAnomaly,
            AnomalyType::PerformanceAnomaly,
            AnomalyType::ConvergenceAnomaly,
            AnomalyType::ResourceAnomaly,
            AnomalyType::PatternAnomaly,
            AnomalyType::SeasonalAnomaly,
            AnomalyType::SystemAnomaly,
        ];

        for anomaly_type in &anomaly_types {
            self.classification_models.insert(
                anomaly_type.clone(),
                ClassificationModel::new(anomaly_type.clone()),
            );
        }
    }

    pub fn classify_anomaly(&self, features: &AnomalyFeatures<T>) -> AnomalyType {
        let mut best_score = T::zero();
        let mut best_type = AnomalyType::StatisticalOutlier;

        for (anomaly_type, model) in &self.classification_models {
            let score = model.classify(features);
            if score > best_score {
                best_score = score;
                best_type = anomaly_type.clone();
            }
        }

        best_type
    }

    pub fn update_model(&mut self, anomaly_type: &AnomalyType, features: &AnomalyFeatures<T>, label: bool) {
        if let Some(model) = self.classification_models.get_mut(anomaly_type) {
            model.update(features, label);
        }
    }
}

/// Features extracted for anomaly classification
#[derive(Debug, Clone)]
pub struct AnomalyFeatures<T: Float> {
    pub statistical_score: T,
    pub trend_score: T,
    pub pattern_score: T,
    pub volatility: T,
    pub magnitude: T,
    pub frequency_features: Vec<T>,
    pub temporal_features: Vec<T>,
}

/// Simple classification model
struct ClassificationModel<T: Float> {
    anomaly_type: AnomalyType,
    weights: Vec<T>,
    bias: T,
    learning_rate: T,
    _phantom: PhantomData<T>,
}

impl<T: Float> ClassificationModel<T> {
    fn new(anomaly_type: AnomalyType) -> Self {
        Self {
            anomaly_type,
            weights: vec![num_traits::cast::cast(0.1).unwrap_or_else(|| T::zero()); 7], // 7 basic features
            bias: T::zero(),
            learning_rate: num_traits::cast::cast(0.01).unwrap_or_else(|| T::zero()),
            _phantom: PhantomData,
        }
    }

    fn classify(&self, features: &AnomalyFeatures<T>) -> T {
        let feature_vec = vec![
            features.statistical_score,
            features.trend_score,
            features.pattern_score,
            features.volatility,
            features.magnitude,
            features.frequency_features.get(0).copied().unwrap_or(T::zero()),
            features.temporal_features.get(0).copied().unwrap_or(T::zero()),
        ];

        let score = feature_vec.iter()
            .zip(self.weights.iter())
            .map(|(&f, &w)| f * w)
            .fold(T::zero(), |acc, x| acc + x) + self.bias;

        // Sigmoid activation
        T::one() / (T::one() + (-score).exp())
    }

    fn update(&mut self, features: &AnomalyFeatures<T>, label: bool) {
        let feature_vec = vec![
            features.statistical_score,
            features.trend_score,
            features.pattern_score,
            features.volatility,
            features.magnitude,
            features.frequency_features.get(0).copied().unwrap_or(T::zero()),
            features.temporal_features.get(0).copied().unwrap_or(T::zero()),
        ];

        let prediction = self.classify(features);
        let target = if label { T::one() } else { T::zero() };
        let error = prediction - target;

        // Gradient descent update
        for (weight, &feature) in self.weights.iter_mut().zip(feature_vec.iter()) {
            *weight = *weight - self.learning_rate * error * feature;
        }
        
        self.bias = self.bias - self.learning_rate * error;
    }
}

/// Outlier detection using multiple methods
pub struct OutlierDetector<T: Float> {
    config: AnomalyConfig<T>,
    _phantom: PhantomData<T>,
}

impl<T: Float> OutlierDetector<T> {
    pub fn new(config: AnomalyConfig<T>) -> Self {
        Self {
            config,
            _phantom: PhantomData,
        }
    }

    pub fn detect_outlier(&self, value: T, history: &VecDeque<(Instant, T)>) -> AnomalyResult<T> {
        let values: Vec<T> = history.iter().map(|(_, v)| *v).collect();
        
        if values.len() < self.config.min_data_points {
            return AnomalyResult {
                is_anomaly: false,
                anomaly_type: AnomalyType::StatisticalOutlier,
                severity: AnomalySeverity::Low,
                confidence: T::zero(),
                anomaly_score: T::zero(),
                timestamp: Instant::now(),
                context: AnomalyContext {
                    baseline_mean: T::zero(),
                    baseline_std: T::zero(),
                    current_value: value,
                    deviation_magnitude: T::zero(),
                    trend_deviation: T::zero(),
                    pattern_match_score: T::zero(),
                    historical_frequency: T::zero(),
                },
                suggested_actions: vec![],
            };
        }

        let mut method_results = Vec::new();

        // Apply configured outlier detection methods
        for method in &self.config.outlier_methods {
            let result = match method {
                OutlierMethod::ZScore => self.zscore_detection(value, &values),
                OutlierMethod::ModifiedZScore => self.modified_zscore_detection(value, &values),
                OutlierMethod::IQR => self.iqr_detection(value, &values),
                OutlierMethod::Hampel => self.hampel_detection(value, &values),
                OutlierMethod::IsolationForest => self.isolation_forest_detection(value, &values),
                OutlierMethod::LocalOutlierFactor => self.lof_detection(value, &values),
                OutlierMethod::OneClassSVM => self.svm_detection(value, &values),
                OutlierMethod::DBSCAN => self.dbscan_detection(value, &values),
            };
            
            method_results.push(result);
        }

        // Combine results using majority voting
        self.combine_outlier_results(method_results, value)
    }

    fn zscore_detection(&self, value: T, values: &[T]) -> (bool, T, T) {
        let mean = values.iter().fold(T::zero(), |acc, &x| acc + x) / T::from(values.len()).unwrap();
        let variance = values.iter()
            .map(|&x| (x - mean) * (x - mean))
            .fold(T::zero(), |acc, x| acc + x) / T::from(values.len()).unwrap();
        let std_dev = variance.sqrt();

        if std_dev < T::epsilon() {
            return (false, T::zero(), T::zero());
        }

        let z_score = (value - mean).abs() / std_dev;
        let is_outlier = z_score > self.config.statistical_threshold;
        let confidence = if is_outlier {
            (z_score / self.config.statistical_threshold).min(T::one())
        } else {
            T::zero()
        };

        (is_outlier, confidence, z_score)
    }

    fn modified_zscore_detection(&self, value: T, values: &[T]) -> (bool, T, T) {
        // Using median absolute deviation (MAD)
        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let median = if sorted_values.len() % 2 == 0 {
            let mid = sorted_values.len() / 2;
            (sorted_values[mid - 1] + sorted_values[mid]) / num_traits::cast::cast(2.0).unwrap_or_else(|| T::zero())
        } else {
            sorted_values[sorted_values.len() / 2]
        };

        let mad = {
            let deviations: Vec<T> = values.iter()
                .map(|&x| (x - median).abs())
                .collect();
            let mut sorted_deviations = deviations;
            sorted_deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            if sorted_deviations.len() % 2 == 0 {
                let mid = sorted_deviations.len() / 2;
                (sorted_deviations[mid - 1] + sorted_deviations[mid]) / num_traits::cast::cast(2.0).unwrap_or_else(|| T::zero())
            } else {
                sorted_deviations[sorted_deviations.len() / 2]
            }
        };

        if mad < T::epsilon() {
            return (false, T::zero(), T::zero());
        }

        let modified_z = num_traits::cast::cast(0.6745).unwrap_or_else(|| T::zero()) * (value - median).abs() / mad;
        let is_outlier = modified_z > num_traits::cast::cast(3.5).unwrap_or_else(|| T::zero());
        let confidence = if is_outlier {
            (modified_z / num_traits::cast::cast(3.5).unwrap_or_else(|| T::zero())).min(T::one())
        } else {
            T::zero()
        };

        (is_outlier, confidence, modified_z)
    }

    fn iqr_detection(&self, value: T, values: &[T]) -> (bool, T, T) {
        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let n = sorted_values.len();
        let q1_idx = n / 4;
        let q3_idx = 3 * n / 4;
        
        let q1 = sorted_values[q1_idx];
        let q3 = sorted_values[q3_idx];
        let iqr = q3 - q1;
        
        let lower_bound = q1 - num_traits::cast::cast(1.5).unwrap_or_else(|| T::zero()) * iqr;
        let upper_bound = q3 + num_traits::cast::cast(1.5).unwrap_or_else(|| T::zero()) * iqr;
        
        let is_outlier = value < lower_bound || value > upper_bound;
        let score = if value < lower_bound {
            (lower_bound - value) / iqr
        } else if value > upper_bound {
            (value - upper_bound) / iqr
        } else {
            T::zero()
        };

        let confidence = if is_outlier { score.min(T::one()) } else { T::zero() };
        
        (is_outlier, confidence, score)
    }

    fn hampel_detection(&self, value: T, values: &[T]) -> (bool, T, T) {
        // Simplified Hampel identifier
        if values.len() < 3 {
            return (false, T::zero(), T::zero());
        }

        let window_size = 7.min(values.len());
        let recent_values = &values[values.len().saturating_sub(window_size)..];
        
        let mut sorted_values = recent_values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let median = if sorted_values.len() % 2 == 0 {
            let mid = sorted_values.len() / 2;
            (sorted_values[mid - 1] + sorted_values[mid]) / num_traits::cast::cast(2.0).unwrap_or_else(|| T::zero())
        } else {
            sorted_values[sorted_values.len() / 2]
        };

        let mad = {
            let deviations: Vec<T> = recent_values.iter()
                .map(|&x| (x - median).abs())
                .collect();
            let mut sorted_deviations = deviations;
            sorted_deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            if sorted_deviations.len() % 2 == 0 {
                let mid = sorted_deviations.len() / 2;
                (sorted_deviations[mid - 1] + sorted_deviations[mid]) / num_traits::cast::cast(2.0).unwrap_or_else(|| T::zero())
            } else {
                sorted_deviations[sorted_deviations.len() / 2]
            }
        };

        if mad < T::epsilon() {
            return (false, T::zero(), T::zero());
        }

        let hampel_score = (value - median).abs() / mad;
        let threshold = num_traits::cast::cast(3.0).unwrap_or_else(|| T::zero());
        let is_outlier = hampel_score > threshold;
        let confidence = if is_outlier {
            (hampel_score / threshold).min(T::one())
        } else {
            T::zero()
        };

        (is_outlier, confidence, hampel_score)
    }

    fn isolation_forest_detection(&self, value: T, values: &[T]) -> (bool, T, T) {
        // Simplified isolation forest approximation
        let mut random_scores = Vec::new();
        let num_trees = 10;
        let subsample_size = (values.len() / 2).max(8);
        
        for _ in 0..num_trees {
            let score = self.isolation_tree_score(value, values, subsample_size, 0);
            random_scores.push(score);
        }

        let avg_score = random_scores.iter().fold(T::zero(), |acc, &x| acc + x) / T::from(random_scores.len()).unwrap();
        let threshold = num_traits::cast::cast(0.6).unwrap_or_else(|| T::zero());
        let is_outlier = avg_score > threshold;
        let confidence = if is_outlier { avg_score } else { T::zero() };

        (is_outlier, confidence, avg_score)
    }

    fn isolation_tree_score(&self, value: T, values: &[T], max_depth: usize, current_depth: usize) -> T {
        if current_depth >= max_depth || values.len() <= 1 {
            return num_traits::cast::cast(current_depth).unwrap_or_else(|| T::zero()) / num_traits::cast::cast(max_depth).unwrap_or_else(|| T::zero());
        }

        let min_val = values.iter().fold(T::infinity(), |acc, &x| acc.min(x));
        let max_val = values.iter().fold(T::neg_infinity(), |acc, &x| acc.max(x));
        
        if (max_val - min_val).abs() < T::epsilon() {
            return num_traits::cast::cast(current_depth).unwrap_or_else(|| T::zero()) / num_traits::cast::cast(max_depth).unwrap_or_else(|| T::zero());
        }

        let split_point = min_val + (max_val - min_val) / num_traits::cast::cast(2.0).unwrap_or_else(|| T::zero());
        
        if value <= split_point {
            let left_values: Vec<T> = values.iter().filter(|&&x| x <= split_point).cloned().collect();
            if !left_values.is_empty() {
                self.isolation_tree_score(value, &left_values, max_depth, current_depth + 1)
            } else {
                num_traits::cast::cast(current_depth + 1).unwrap_or_else(|| T::zero()) / num_traits::cast::cast(max_depth).unwrap_or_else(|| T::zero())
            }
        } else {
            let right_values: Vec<T> = values.iter().filter(|&&x| x > split_point).cloned().collect();
            if !right_values.is_empty() {
                self.isolation_tree_score(value, &right_values, max_depth, current_depth + 1)
            } else {
                num_traits::cast::cast(current_depth + 1).unwrap_or_else(|| T::zero()) / num_traits::cast::cast(max_depth).unwrap_or_else(|| T::zero())
            }
        }
    }

    fn lof_detection(&self, value: T, values: &[T]) -> (bool, T, T) {
        // Simplified Local Outlier Factor
        let k = 5.min(values.len());
        if k == 0 {
            return (false, T::zero(), T::zero());
        }

        let mut distances: Vec<T> = values.iter()
            .map(|&x| (x - value).abs())
            .collect();
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let k_distance = distances[k.min(distances.len() - 1)];
        let neighbors: Vec<T> = values.iter()
            .filter(|&&x| (x - value).abs() <= k_distance)
            .cloned()
            .collect();

        if neighbors.is_empty() {
            return (false, T::zero(), T::zero());
        }

        // Simplified density estimation
        let local_density = T::from(neighbors.len()).unwrap() / (k_distance + T::epsilon());
        let neighbor_densities: Vec<T> = neighbors.iter()
            .map(|&neighbor| {
                let neighbor_distances: Vec<T> = values.iter()
                    .map(|&x| (x - neighbor).abs())
                    .collect();
                let mut sorted_neighbor_distances = neighbor_distances;
                sorted_neighbor_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let neighbor_k_distance = sorted_neighbor_distances[k.min(sorted_neighbor_distances.len() - 1)];
                num_traits::cast::cast(k).unwrap_or_else(|| T::zero()) / (neighbor_k_distance + T::epsilon())
            })
            .collect();

        let avg_neighbor_density = neighbor_densities.iter().fold(T::zero(), |acc, &x| acc + x) 
                                 / T::from(neighbor_densities.len()).unwrap();
        
        let lof_score = if local_density > T::epsilon() {
            avg_neighbor_density / local_density
        } else {
            T::one()
        };

        let threshold = num_traits::cast::cast(1.5).unwrap_or_else(|| T::zero());
        let is_outlier = lof_score > threshold;
        let confidence = if is_outlier {
            ((lof_score - T::one()) / threshold).min(T::one())
        } else {
            T::zero()
        };

        (is_outlier, confidence, lof_score)
    }

    fn svm_detection(&self, value: T, values: &[T]) -> (bool, T, T) {
        // Simplified One-Class SVM approximation using distance to centroid
        let mean = values.iter().fold(T::zero(), |acc, &x| acc + x) / T::from(values.len()).unwrap();
        let distances: Vec<T> = values.iter().map(|&x| (x - mean).abs()).collect();
        let avg_distance = distances.iter().fold(T::zero(), |acc, &x| acc + x) / T::from(distances.len()).unwrap();
        
        let current_distance = (value - mean).abs();
        let score = current_distance / (avg_distance + T::epsilon());
        let threshold = num_traits::cast::cast(2.0).unwrap_or_else(|| T::zero());
        let is_outlier = score > threshold;
        let confidence = if is_outlier {
            (score / threshold).min(T::one())
        } else {
            T::zero()
        };

        (is_outlier, confidence, score)
    }

    fn dbscan_detection(&self, value: T, values: &[T]) -> (bool, T, T) {
        // Simplified DBSCAN-based outlier detection
        let eps = self.compute_eps(values);
        let min_points = 3;
        
        let neighbors: Vec<T> = values.iter()
            .filter(|&&x| (x - value).abs() <= eps)
            .cloned()
            .collect();

        let is_outlier = neighbors.len() < min_points;
        let score = num_traits::cast::cast(min_points).unwrap_or_else(|| T::zero()) / (T::from(neighbors.len()).unwrap() + T::one());
        let confidence = if is_outlier { score.min(T::one()) } else { T::zero() };

        (is_outlier, confidence, score)
    }

    fn compute_eps(&self, values: &[T]) -> T {
        if values.len() < 2 {
            return T::one();
        }

        let mut distances = Vec::new();
        for i in 0..values.len() {
            for j in i+1..values.len() {
                distances.push((values[i] - values[j]).abs());
            }
        }

        distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let percentile_90 = distances[(distances.len() * 9 / 10).min(distances.len() - 1)];
        
        percentile_90 / num_traits::cast::cast(2.0).unwrap_or_else(|| T::zero())
    }

    fn combine_outlier_results(&self, results: Vec<(bool, T, T)>, value: T) -> AnomalyResult<T> {
        if results.is_empty() {
            return AnomalyResult {
                is_anomaly: false,
                anomaly_type: AnomalyType::StatisticalOutlier,
                severity: AnomalySeverity::Low,
                confidence: T::zero(),
                anomaly_score: T::zero(),
                timestamp: Instant::now(),
                context: AnomalyContext {
                    baseline_mean: T::zero(),
                    baseline_std: T::zero(),
                    current_value: value,
                    deviation_magnitude: T::zero(),
                    trend_deviation: T::zero(),
                    pattern_match_score: T::zero(),
                    historical_frequency: T::zero(),
                },
                suggested_actions: vec![],
            };
        }

        let outlier_count = results.iter().filter(|(is_outlier, _, _)| *is_outlier).count();
        let total_confidence = results.iter().map(|(_, confidence, _)| *confidence).fold(T::zero(), |acc, x| acc + x);
        let total_score = results.iter().map(|(_, _, score)| *score).fold(T::zero(), |acc, x| acc + x);

        let avg_confidence = total_confidence / T::from(results.len()).unwrap();
        let avg_score = total_score / T::from(results.len()).unwrap();
        
        // Use majority voting
        let is_anomaly = num_traits::cast::cast(outlier_count).unwrap_or_else(|| T::zero()) > T::from(results.len()).unwrap() / num_traits::cast::cast(2.0).unwrap_or_else(|| T::zero());
        
        let severity = if avg_score > num_traits::cast::cast(3.0).unwrap_or_else(|| T::zero()) {
            AnomalySeverity::Critical
        } else if avg_score > num_traits::cast::cast(2.0).unwrap_or_else(|| T::zero()) {
            AnomalySeverity::High
        } else if avg_score > num_traits::cast::cast(1.5).unwrap_or_else(|| T::zero()) {
            AnomalySeverity::Medium
        } else {
            AnomalySeverity::Low
        };

        AnomalyResult {
            is_anomaly,
            anomaly_type: AnomalyType::StatisticalOutlier,
            severity,
            confidence: avg_confidence,
            anomaly_score: avg_score,
            timestamp: Instant::now(),
            context: AnomalyContext {
                baseline_mean: T::zero(),
                baseline_std: T::zero(),
                current_value: value,
                deviation_magnitude: avg_score,
                trend_deviation: T::zero(),
                pattern_match_score: T::zero(),
                historical_frequency: T::zero(),
            },
            suggested_actions: vec![
                "Review data collection process".to_string(),
                "Check for measurement errors".to_string(),
                "Investigate potential system issues".to_string(),
            ],
        }
    }
}

/// Anomaly reporting and alerting system
pub struct AnomalyReporter<T: Float> {
    config: AnomalyConfig<T>,
    alerts: VecDeque<AnomalyAlert<T>>,
    alert_history: HashMap<AnomalyType, Vec<Instant>>,
    last_alert_times: HashMap<AnomalyType, Instant>,
    _phantom: PhantomData<T>,
}

impl<T: Float> AnomalyReporter<T> {
    pub fn new(config: AnomalyConfig<T>) -> Self {
        Self {
            config,
            alerts: VecDeque::with_capacity(1000),
            alert_history: HashMap::new(),
            last_alert_times: HashMap::new(),
            _phantom: PhantomData,
        }
    }

    pub fn report_anomaly(&mut self, result: &AnomalyResult<T>) -> Option<AnomalyAlert<T>> {
        if !result.is_anomaly {
            return None;
        }

        // Check cooldown period
        let now = Instant::now();
        if let Some(&last_alert) = self.last_alert_times.get(&result.anomaly_type) {
            if now.duration_since(last_alert) < self.config.alert_cooldown {
                return None; // Still in cooldown period
            }
        }

        let alert_id = format!("anomaly_{}_{}", 
            self.anomaly_type_to_string(&result.anomaly_type),
            now.elapsed().as_nanos()
        );

        let alert = AnomalyAlert {
            id: alert_id,
            anomaly_type: result.anomaly_type.clone(),
            severity: result.severity.clone(),
            timestamp: now,
            message: self.generate_alert_message(result),
            confidence: result.confidence,
            data_point: result.context.current_value,
            context: result.context.clone(),
            suggested_actions: result.suggested_actions.clone(),
            acknowledged: false,
            resolved: false,
        };

        self.alerts.push_back(alert.clone());
        if self.alerts.len() > 1000 {
            self.alerts.pop_front();
        }

        // Update alert history
        self.alert_history.entry(result.anomaly_type.clone())
            .or_insert_with(Vec::new)
            .push(now);
        
        self.last_alert_times.insert(result.anomaly_type.clone(), now);

        Some(alert)
    }

    fn generate_alert_message(&self, result: &AnomalyResult<T>) -> String {
        match result.anomaly_type {
            AnomalyType::StatisticalOutlier => {
                format!("Statistical outlier detected: value {:.4} deviates by {:.2} standard deviations",
                    result.context.current_value.to_f64().unwrap_or(0.0),
                    (result.context.deviation_magnitude / result.context.baseline_std).to_f64().unwrap_or(0.0))
            },
            AnomalyType::TrendAnomaly => {
                format!("Trend anomaly detected: unexpected deviation of {:.4} from expected trend",
                    result.context.trend_deviation.to_f64().unwrap_or(0.0))
            },
            AnomalyType::PatternAnomaly => {
                format!("Pattern anomaly detected: pattern match score {:.4} below threshold",
                    result.context.pattern_match_score.to_f64().unwrap_or(0.0))
            },
            _ => {
                format!("{:?} anomaly detected with confidence {:.4}",
                    result.anomaly_type,
                    result.confidence.to_f64().unwrap_or(0.0))
            }
        }
    }

    fn anomaly_type_to_string(&self, anomaly_type: &AnomalyType) -> &str {
        match anomaly_type {
            AnomalyType::StatisticalOutlier => "statistical",
            AnomalyType::TrendAnomaly => "trend",
            AnomalyType::PerformanceAnomaly => "performance",
            AnomalyType::ConvergenceAnomaly => "convergence",
            AnomalyType::ResourceAnomaly => "resource",
            AnomalyType::PatternAnomaly => "pattern",
            AnomalyType::SeasonalAnomaly => "seasonal",
            AnomalyType::SystemAnomaly => "system",
        }
    }

    pub fn acknowledge_alert(&mut self, alert_id: &str) -> bool {
        for alert in &mut self.alerts {
            if alert.id == alert_id {
                alert.acknowledged = true;
                return true;
            }
        }
        false
    }

    pub fn resolve_alert(&mut self, alert_id: &str) -> bool {
        for alert in &mut self.alerts {
            if alert.id == alert_id {
                alert.resolved = true;
                return true;
            }
        }
        false
    }

    pub fn get_active_alerts(&self) -> Vec<&AnomalyAlert<T>> {
        self.alerts.iter()
            .filter(|alert| !alert.resolved)
            .collect()
    }

    pub fn get_unacknowledged_alerts(&self) -> Vec<&AnomalyAlert<T>> {
        self.alerts.iter()
            .filter(|alert| !alert.acknowledged && !alert.resolved)
            .collect()
    }

    pub fn get_alert_statistics(&self) -> HashMap<AnomalyType, usize> {
        let mut stats = HashMap::new();
        for alert in &self.alerts {
            *stats.entry(alert.anomaly_type.clone()).or_insert(0) += 1;
        }
        stats
    }

    pub fn clear_resolved_alerts(&mut self) {
        self.alerts.retain(|alert| !alert.resolved);
    }

    pub fn generate_summary_report(&self) -> String {
        let total_alerts = self.alerts.len();
        let active_alerts = self.get_active_alerts().len();
        let unacknowledged_alerts = self.get_unacknowledged_alerts().len();
        
        let mut report = format!(
            "Anomaly Detection Summary:\n\
             - Total Alerts: {}\n\
             - Active Alerts: {}\n\
             - Unacknowledged Alerts: {}\n\
             \n\
             Alert Breakdown by Type:\n",
            total_alerts, active_alerts, unacknowledged_alerts
        );

        let stats = self.get_alert_statistics();
        for (anomaly_type, count) in stats {
            report.push_str(&format!("- {:?}: {}\n", anomaly_type, count));
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anomaly_detector_basic() {
        let config = AnomalyConfig::<f64>::default();
        let mut detector = AnomalyDetector::new(config);

        // Test with normal values
        let normal_values = vec![1.0, 1.1, 0.9, 1.2, 0.8, 1.0, 1.1];
        for value in normal_values {
            let result = detector.detect_anomaly(value);
            println!("Value: {}, Anomaly: {}, Confidence: {:.4}", 
                     value, result.is_anomaly, result.confidence);
        }

        // Test with clear outlier
        let outlier_result = detector.detect_anomaly(10.0);
        assert!(outlier_result.is_anomaly);
        assert!(outlier_result.confidence > 0.5);
    }

    #[test]
    fn test_outlier_methods() {
        let config = AnomalyConfig::<f64>::default();
        let detector = OutlierDetector::new(config);
        
        let mut history = VecDeque::new();
        let normal_values = vec![1.0, 2.0, 1.5, 2.5, 1.8, 2.2, 1.7, 2.1];
        for (i, value) in normal_values.iter().enumerate() {
            history.push_back((Instant::now(), *value));
        }

        let outlier_result = detector.detect_outlier(10.0, &history);
        assert!(outlier_result.is_anomaly);
    }

    #[test]
    fn test_anomaly_reporter() {
        let config = AnomalyConfig::<f64>::default();
        let mut reporter = AnomalyReporter::new(config);

        let anomaly_result = AnomalyResult {
            is_anomaly: true,
            anomaly_type: AnomalyType::StatisticalOutlier,
            severity: AnomalySeverity::High,
            confidence: 0.9,
            anomaly_score: 3.5,
            timestamp: Instant::now(),
            context: AnomalyContext {
                baseline_mean: 1.0,
                baseline_std: 0.2,
                current_value: 5.0,
                deviation_magnitude: 4.0,
                trend_deviation: 0.0,
                pattern_match_score: 0.1,
                historical_frequency: 0.0,
            },
            suggested_actions: vec!["Investigate data source".to_string()],
        };

        let alert = reporter.report_anomaly(&anomaly_result);
        assert!(alert.is_some());
        
        let active_alerts = reporter.get_active_alerts();
        assert_eq!(active_alerts.len(), 1);
    }
}