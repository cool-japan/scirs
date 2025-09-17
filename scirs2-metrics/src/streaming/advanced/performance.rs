//! Performance monitoring for streaming metrics
//!
//! This module provides comprehensive performance tracking and degradation detection
//! for streaming machine learning systems.

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use super::core::AlertSeverity;
use super::window_management::StreamingStatistics;
use crate::error::Result;
use num_traits::Float;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Performance monitor for streaming metrics
#[derive(Debug, Clone)]
pub struct PerformanceMonitor<F: Float + std::fmt::Debug> {
    monitoring_interval: Duration,
    last_monitoring: Instant,
    performance_history: VecDeque<PerformanceSnapshot<F>>,
    current_metrics: HashMap<String, F>,
    baseline_metrics: HashMap<String, F>,
    performance_thresholds: HashMap<String, F>,
    degradation_alerts: VecDeque<PerformanceDegradation>,
}

/// Performance snapshot
#[derive(Debug, Clone)]
pub struct PerformanceSnapshot<F: Float + std::fmt::Debug> {
    pub timestamp: Instant,
    pub accuracy: F,
    pub precision: F,
    pub recall: F,
    pub f1_score: F,
    pub processing_time: Duration,
    pub memory_usage: usize,
    pub window_size: usize,
    pub samples_processed: usize,
}

/// Performance degradation alert
#[derive(Debug, Clone)]
pub struct PerformanceDegradation {
    pub timestamp: Instant,
    pub metric_name: String,
    pub current_value: f64,
    pub baseline_value: f64,
    pub degradation_percentage: f64,
    pub severity: AlertSeverity,
}

impl<F: Float + std::fmt::Debug + Send + Sync> PerformanceMonitor<F> {
    pub fn new(interval: Duration) -> Self {
        let mut thresholds = HashMap::new();
        thresholds.insert("accuracy".to_string(), F::from(0.8).unwrap()); // 80% accuracy threshold
        thresholds.insert("precision".to_string(), F::from(0.75).unwrap());
        thresholds.insert("recall".to_string(), F::from(0.75).unwrap());
        thresholds.insert("f1_score".to_string(), F::from(0.75).unwrap());

        Self {
            monitoring_interval: interval,
            last_monitoring: Instant::now(),
            performance_history: VecDeque::with_capacity(1000), // Bounded capacity for memory efficiency
            current_metrics: HashMap::new(),
            baseline_metrics: HashMap::new(),
            performance_thresholds: thresholds,
            degradation_alerts: VecDeque::with_capacity(100), // Limited alert history
        }
    }

    pub fn should_monitor(&self) -> bool {
        self.last_monitoring.elapsed() >= self.monitoring_interval
    }

    pub fn take_snapshot(&mut self, stats: &StreamingStatistics<F>) -> Result<()> {
        let now = Instant::now();

        // Create performance snapshot
        let snapshot = PerformanceSnapshot {
            timestamp: now,
            accuracy: stats.current_accuracy,
            precision: F::zero(), // Would be calculated from confusion matrix
            recall: F::zero(),    // Would be calculated from confusion matrix
            f1_score: F::zero(),  // Would be calculated from confusion matrix
            processing_time: now.duration_since(
                self.performance_history
                    .back()
                    .map(|p| p.timestamp)
                    .unwrap_or_else(|| now - Duration::from_millis(1)),
            ),
            memory_usage: std::mem::size_of::<StreamingStatistics<F>>(),
            window_size: 1000, // Would come from actual window manager
            samples_processed: stats.total_samples,
        };

        // Add to history with memory management
        self.performance_history.push_back(snapshot.clone());
        if self.performance_history.len() > 1000 {
            self.performance_history.pop_front();
        }

        // Update current metrics
        self.current_metrics
            .insert("accuracy".to_string(), stats.current_accuracy);
        self.current_metrics
            .insert("error_rate".to_string(), stats.error_rate);
        self.current_metrics.insert(
            "moving_average_accuracy".to_string(),
            stats.moving_average_accuracy,
        );

        // Set baseline if this is first measurement
        if self.baseline_metrics.is_empty() {
            self.baseline_metrics = self.current_metrics.clone();
        }

        // Check for performance degradation
        self.check_performance_degradation()?;

        self.last_monitoring = now;
        Ok(())
    }

    fn check_performance_degradation(&mut self) -> Result<()> {
        for (metric_name, &current_value) in &self.current_metrics {
            if let Some(&baseline_value) = self.baseline_metrics.get(metric_name) {
                if let Some(&threshold) = self.performance_thresholds.get(metric_name) {
                    let current_f64 = current_value.to_f64().unwrap_or(0.0);
                    let baseline_f64 = baseline_value.to_f64().unwrap_or(0.0);
                    let threshold_f64 = threshold.to_f64().unwrap_or(0.0);

                    // Check if current performance is below threshold
                    if current_f64 < threshold_f64 {
                        let degradation_percentage = if baseline_f64 > 0.0 {
                            ((baseline_f64 - current_f64) / baseline_f64) * 100.0
                        } else {
                            0.0
                        };

                        let severity = if degradation_percentage > 50.0 {
                            AlertSeverity::Critical
                        } else if degradation_percentage > 25.0 {
                            AlertSeverity::High
                        } else if degradation_percentage > 10.0 {
                            AlertSeverity::Medium
                        } else {
                            AlertSeverity::Low
                        };

                        let degradation = PerformanceDegradation {
                            timestamp: Instant::now(),
                            metric_name: metric_name.clone(),
                            current_value: current_f64,
                            baseline_value: baseline_f64,
                            degradation_percentage,
                            severity,
                        };

                        self.degradation_alerts.push_back(degradation);
                        if self.degradation_alerts.len() > 100 {
                            self.degradation_alerts.pop_front();
                        }
                    }
                }
            }
        }
        Ok(())
    }

    pub fn reset(&mut self) {
        self.performance_history.clear();
        self.current_metrics.clear();
        self.baseline_metrics.clear();
        self.degradation_alerts.clear();
        self.last_monitoring = Instant::now();
    }

    /// Get recent performance trends
    pub fn get_performance_trend(&self, metric_name: &str, window: usize) -> Option<(f64, f64)> {
        if self.performance_history.len() < window {
            return None;
        }

        let recent_snapshots: Vec<_> = self.performance_history.iter().rev().take(window).collect();

        let values: Vec<f64> = recent_snapshots
            .iter()
            .map(|snapshot| match metric_name {
                "accuracy" => snapshot.accuracy.to_f64().unwrap_or(0.0),
                "precision" => snapshot.precision.to_f64().unwrap_or(0.0),
                "recall" => snapshot.recall.to_f64().unwrap_or(0.0),
                "f1_score" => snapshot.f1_score.to_f64().unwrap_or(0.0),
                _ => 0.0,
            })
            .collect();

        if values.is_empty() {
            return None;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;

        Some((mean, variance))
    }

    /// Get current performance summary
    pub fn get_performance_summary(&self) -> HashMap<String, f64> {
        self.current_metrics
            .iter()
            .map(|(k, v)| (k.clone(), v.to_f64().unwrap_or(0.0)))
            .collect()
    }

    /// Get degradation alerts
    pub fn get_degradation_alerts(&self) -> &VecDeque<PerformanceDegradation> {
        &self.degradation_alerts
    }

    /// Update performance thresholds
    pub fn set_threshold(&mut self, metric_name: String, threshold: F) {
        self.performance_thresholds.insert(metric_name, threshold);
    }

    /// Get performance history
    pub fn get_performance_history(&self) -> &VecDeque<PerformanceSnapshot<F>> {
        &self.performance_history
    }
}
