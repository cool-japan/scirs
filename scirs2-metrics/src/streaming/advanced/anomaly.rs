//! Anomaly detection for streaming data
//!
//! This module provides various anomaly detection algorithms for identifying
//! outliers and unusual patterns in streaming data.

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use super::core::AnomalyDetectionAlgorithm;
use crate::error::{MetricsError, Result};
use scirs2_core::numeric::Float;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Anomaly detector for streaming data
#[derive(Debug, Clone)]
pub struct AnomalyDetector<F: Float + std::fmt::Debug + Send + Sync> {
    algorithm: AnomalyDetectionAlgorithm,
    history_buffer: VecDeque<F>,
    anomaly_scores: VecDeque<F>,
    threshold: F,
    pub detected_anomalies: VecDeque<Anomaly<F>>,
    pub statistics: AnomalyStatistics<F>,
}

/// Detected anomaly
#[derive(Debug, Clone)]
pub struct Anomaly<F: Float + std::fmt::Debug> {
    pub timestamp: Instant,
    pub value: F,
    pub score: F,
    pub anomaly_type: AnomalyType,
    pub confidence: F,
    pub context: HashMap<String, String>,
}

/// Types of anomalies
#[derive(Debug, Clone)]
pub enum AnomalyType {
    PointAnomaly,
    ContextualAnomaly,
    CollectiveAnomaly,
    ConceptDrift,
    DataQualityIssue,
    Unknown,
}

/// Anomaly detection statistics
#[derive(Debug, Clone)]
pub struct AnomalyStatistics<F: Float + std::fmt::Debug> {
    pub total_anomalies: usize,
    pub anomalies_by_type: HashMap<String, usize>,
    pub false_positive_rate: F,
    pub detection_latency: Duration,
    pub last_anomaly: Option<Instant>,
}

impl<F: Float + std::iter::Sum + std::fmt::Debug + Send + Sync> AnomalyDetector<F> {
    pub fn new(algorithm: AnomalyDetectionAlgorithm) -> Result<Self> {
        let threshold = match &algorithm {
            AnomalyDetectionAlgorithm::ZScore { threshold } => F::from(*threshold).unwrap(),
            AnomalyDetectionAlgorithm::IsolationForest { contamination: _ } => {
                F::from(0.5).unwrap()
            }
            _ => F::from(3.0).unwrap(),
        };

        Ok(Self {
            algorithm,
            history_buffer: VecDeque::with_capacity(1000), // Bounded for memory efficiency
            anomaly_scores: VecDeque::with_capacity(1000), // Bounded for memory efficiency
            threshold,
            detected_anomalies: VecDeque::with_capacity(500), // Keep recent anomalies only
            statistics: AnomalyStatistics {
                total_anomalies: 0,
                anomalies_by_type: HashMap::new(),
                false_positive_rate: F::zero(),
                detection_latency: Duration::from_millis(0),
                last_anomaly: None,
            },
        })
    }

    pub fn detect(&mut self, error: F) -> Result<Anomaly<F>> {
        let detection_start = Instant::now();

        // Add to history with memory management
        self.history_buffer.push_back(error);
        if self.history_buffer.len() > 1000 {
            self.history_buffer.pop_front();
        }

        // Detect anomaly based on algorithm
        let (is_anomaly, score, anomaly_type) = match &self.algorithm {
            AnomalyDetectionAlgorithm::ZScore { threshold } => {
                self.detect_zscore_anomaly(error, *threshold)?
            }
            AnomalyDetectionAlgorithm::IsolationForest { contamination } => {
                self.detect_isolation_forest_anomaly(error, *contamination)?
            }
            AnomalyDetectionAlgorithm::LocalOutlierFactor { n_neighbors } => {
                self.detect_lof_anomaly(error, *n_neighbors)?
            }
            _ => {
                // Default to z-score
                self.detect_zscore_anomaly(error, 3.0)?
            }
        };

        // Record anomaly score
        self.anomaly_scores.push_back(score);
        if self.anomaly_scores.len() > 1000 {
            self.anomaly_scores.pop_front();
        }

        if is_anomaly {
            let detection_latency = detection_start.elapsed();

            let anomaly = Anomaly {
                timestamp: Instant::now(),
                value: error,
                score,
                anomaly_type: anomaly_type.clone(),
                confidence: score / self.threshold,
                context: self.build_anomaly_context(error)?,
            };

            // Add to detected anomalies with memory management
            self.detected_anomalies.push_back(anomaly.clone());
            if self.detected_anomalies.len() > 500 {
                self.detected_anomalies.pop_front();
            }

            // Update statistics
            self.statistics.total_anomalies += 1;
            self.statistics.detection_latency = detection_latency;
            self.statistics.last_anomaly = Some(Instant::now());

            let type_name = format!("{anomaly_type:?}");
            *self
                .statistics
                .anomalies_by_type
                .entry(type_name)
                .or_insert(0) += 1;

            return Ok(anomaly);
        }

        Err(MetricsError::ComputationError(
            "No anomaly detected".to_string(),
        ))
    }

    fn detect_zscore_anomaly(&self, error: F, threshold: f64) -> Result<(bool, F, AnomalyType)> {
        if self.history_buffer.len() < 10 {
            return Ok((false, F::zero(), AnomalyType::Unknown));
        }

        // Calculate running statistics efficiently
        let mean = self.history_buffer.iter().cloned().sum::<F>()
            / F::from(self.history_buffer.len()).unwrap();

        let variance = self
            .history_buffer
            .iter()
            .map(|&x| (x - mean) * (x - mean))
            .sum::<F>()
            / F::from(self.history_buffer.len() - 1).unwrap();

        let std_dev = variance.sqrt();

        let z_score = if std_dev > F::zero() {
            (error - mean).abs() / std_dev
        } else {
            F::zero()
        };

        let threshold_f = F::from(threshold).unwrap();
        let is_anomaly = z_score > threshold_f;

        let anomaly_type = if is_anomaly {
            if z_score > threshold_f * F::from(2.0).unwrap() {
                AnomalyType::PointAnomaly
            } else {
                AnomalyType::ContextualAnomaly
            }
        } else {
            AnomalyType::Unknown
        };

        Ok((is_anomaly, z_score, anomaly_type))
    }

    fn detect_isolation_forest_anomaly(
        &self,
        error: F,
        contamination: f64,
    ) -> Result<(bool, F, AnomalyType)> {
        // Simplified isolation forest implementation
        if self.history_buffer.len() < 20 {
            return Ok((false, F::zero(), AnomalyType::Unknown));
        }

        // Calculate isolation score based on value position in sorted data
        let mut sorted_values: Vec<F> = self.history_buffer.iter().cloned().collect();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let position = sorted_values
            .iter()
            .position(|&x| x >= error)
            .unwrap_or(sorted_values.len());
        let relative_position = position as f64 / sorted_values.len() as f64;

        // Anomalies are typically at the extremes
        let isolation_score = F::from(1.0 - (relative_position - 0.5).abs() * 2.0).unwrap();
        let contamination_threshold = F::from(1.0 - contamination).unwrap();

        let is_anomaly = isolation_score > contamination_threshold;
        let anomaly_type = if is_anomaly {
            AnomalyType::PointAnomaly
        } else {
            AnomalyType::Unknown
        };

        Ok((is_anomaly, isolation_score, anomaly_type))
    }

    fn detect_lof_anomaly(&self, error: F, n_neighbors: usize) -> Result<(bool, F, AnomalyType)> {
        // Simplified LOF implementation
        if self.history_buffer.len() < n_neighbors * 2 {
            return Ok((false, F::zero(), AnomalyType::Unknown));
        }

        // Calculate local outlier factor based on k-nearest neighbors
        let mut distances: Vec<(F, usize)> = self
            .history_buffer
            .iter()
            .enumerate()
            .map(|(i, &value)| ((value - error).abs(), i))
            .collect();

        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Get k nearest neighbors
        let k_distance = if distances.len() > n_neighbors {
            distances[n_neighbors].0
        } else {
            distances.last().unwrap().0
        };

        // Simple LOF approximation
        let lof_score = if k_distance > F::zero() {
            F::from(2.0).unwrap() / (F::one() + k_distance)
        } else {
            F::one()
        };

        let is_anomaly = lof_score > F::from(1.5).unwrap();
        let anomaly_type = if is_anomaly {
            AnomalyType::ContextualAnomaly
        } else {
            AnomalyType::Unknown
        };

        Ok((is_anomaly, lof_score, anomaly_type))
    }

    fn build_anomaly_context(&self, error: F) -> Result<HashMap<String, String>> {
        let mut context = HashMap::new();

        context.insert(
            "buffer_size".to_string(),
            self.history_buffer.len().to_string(),
        );
        context.insert(
            "error_value".to_string(),
            format!("{:.6}", error.to_f64().unwrap_or(0.0)),
        );

        if !self.history_buffer.is_empty() {
            let min_val = self
                .history_buffer
                .iter()
                .cloned()
                .fold(F::infinity(), F::min);
            let max_val = self
                .history_buffer
                .iter()
                .cloned()
                .fold(F::neg_infinity(), F::max);
            context.insert(
                "buffer_min".to_string(),
                format!("{:.6}", min_val.to_f64().unwrap_or(0.0)),
            );
            context.insert(
                "buffer_max".to_string(),
                format!("{:.6}", max_val.to_f64().unwrap_or(0.0)),
            );
        }

        Ok(context)
    }

    pub fn reset(&mut self) {
        self.history_buffer.clear();
        self.anomaly_scores.clear();
        self.detected_anomalies.clear();
        self.statistics.total_anomalies = 0;
        self.statistics.anomalies_by_type.clear();
        self.statistics.last_anomaly = None;
    }

    /// Get recent anomaly rate
    pub fn get_recent_anomaly_rate(&self, window: usize) -> f64 {
        if self.anomaly_scores.len() < window {
            return 0.0;
        }

        let recent_scores: Vec<_> = self.anomaly_scores.iter().rev().take(window).collect();
        let anomaly_count = recent_scores
            .iter()
            .filter(|&&score| score > &self.threshold)
            .count();

        anomaly_count as f64 / window as f64
    }

    /// Get anomaly score statistics
    pub fn get_anomaly_score_stats(&self) -> Option<(f64, f64, f64)> {
        if self.anomaly_scores.is_empty() {
            return None;
        }

        let scores: Vec<f64> = self
            .anomaly_scores
            .iter()
            .map(|x| x.to_f64().unwrap_or(0.0))
            .collect();

        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        let min = scores.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = scores.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        Some((mean, min, max))
    }
}
