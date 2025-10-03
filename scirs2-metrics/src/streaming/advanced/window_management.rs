//! Adaptive window management for streaming metrics
//!
//! This module provides intelligent window sizing strategies that adapt based on
//! concept drift, performance degradation, and other streaming data characteristics.

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use super::anomaly::Anomaly;
use super::core::WindowAdaptationStrategy;
use crate::error::Result;
use scirs2_core::numeric::Float;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Adaptive window manager
#[derive(Debug, Clone)]
pub struct AdaptiveWindowManager<F: Float + std::fmt::Debug> {
    current_window_size: usize,
    base_window_size: usize,
    min_window_size: usize,
    max_window_size: usize,
    adaptation_strategy: WindowAdaptationStrategy,
    performance_history: VecDeque<F>,
    adaptation_history: VecDeque<WindowAdaptation>,
    last_adaptation: Option<Instant>,
    adaptation_cooldown: Duration,
}

/// Window adaptation record
#[derive(Debug, Clone)]
pub struct WindowAdaptation {
    pub timestamp: Instant,
    pub old_size: usize,
    pub new_size: usize,
    pub trigger: AdaptationTrigger,
    pub performance_before: f64,
    pub performance_after: Option<f64>,
}

/// Triggers for window adaptation
#[derive(Debug, Clone)]
pub enum AdaptationTrigger {
    DriftDetected,
    PerformanceDegradation { threshold: f64 },
    AnomalyDetected,
    Manual,
    Scheduled,
    MLRecommendation { confidence: f64 },
}

/// Current streaming statistics (simplified for this module)
#[derive(Debug, Clone)]
pub struct StreamingStatistics<F: Float + std::fmt::Debug> {
    pub total_samples: usize,
    pub correct_predictions: usize,
    pub current_accuracy: F,
    pub moving_average_accuracy: F,
    pub error_rate: F,
    pub drift_detected: bool,
    pub anomalies_detected: usize,
    pub processing_rate: F, // samples per second
    pub memory_usage: usize,
    pub last_update: Instant,
}

impl<F: Float + std::fmt::Debug> StreamingStatistics<F> {
    pub fn reset(&mut self) {
        self.total_samples = 0;
        self.correct_predictions = 0;
        self.current_accuracy = F::zero();
        self.moving_average_accuracy = F::zero();
        self.error_rate = F::zero();
        self.drift_detected = false;
        self.anomalies_detected = 0;
        self.processing_rate = F::zero();
        self.memory_usage = 0;
        self.last_update = Instant::now();
    }
}

impl<F: Float + std::fmt::Debug + Send + Sync> AdaptiveWindowManager<F> {
    pub fn new(
        base_size: usize,
        min_size: usize,
        max_size: usize,
        strategy: WindowAdaptationStrategy,
    ) -> Self {
        Self {
            current_window_size: base_size,
            base_window_size: base_size,
            min_window_size: min_size,
            max_window_size: max_size,
            adaptation_strategy: strategy,
            performance_history: VecDeque::with_capacity(100), // Limit memory usage
            adaptation_history: VecDeque::with_capacity(50),   // Keep adaptation history bounded
            last_adaptation: None,
            adaptation_cooldown: Duration::from_secs(60),
        }
    }

    pub fn consider_adaptation(
        &mut self,
        stats: &StreamingStatistics<F>,
        drift_detected: bool,
        anomaly: Option<&Anomaly<F>>,
    ) -> Result<Option<WindowAdaptation>> {
        // Check cooldown period to prevent thrashing
        if let Some(last_adapt) = self.last_adaptation {
            if last_adapt.elapsed() < self.adaptation_cooldown {
                return Ok(None);
            }
        }

        // Record current performance
        self.performance_history.push_back(stats.current_accuracy);
        if self.performance_history.len() > 100 {
            self.performance_history.pop_front();
        }

        let current_performance = stats.current_accuracy.to_f64().unwrap_or(0.0);
        let old_size = self.current_window_size;
        let mut should_adapt = false;
        let mut trigger = AdaptationTrigger::Manual;

        // Determine if adaptation is needed based on strategy
        match &self.adaptation_strategy {
            WindowAdaptationStrategy::Fixed => {
                // No adaptation for fixed strategy
                return Ok(None);
            }
            WindowAdaptationStrategy::DriftBased => {
                if drift_detected {
                    should_adapt = true;
                    trigger = AdaptationTrigger::DriftDetected;
                }
            }
            WindowAdaptationStrategy::PerformanceBased { target_accuracy } => {
                if current_performance < *target_accuracy {
                    should_adapt = true;
                    trigger = AdaptationTrigger::PerformanceDegradation {
                        threshold: *target_accuracy,
                    };
                }
            }
            WindowAdaptationStrategy::ExponentialDecay { decay_rate } => {
                // Gradually reduce window size based on decay rate
                let new_size = (self.current_window_size as f64 * (1.0 - decay_rate)) as usize;
                if new_size >= self.min_window_size && new_size != self.current_window_size {
                    self.current_window_size = new_size;
                    should_adapt = true;
                    trigger = AdaptationTrigger::Scheduled;
                }
            }
            WindowAdaptationStrategy::Hybrid {
                strategies,
                weights,
            } => {
                // Combine multiple strategies with weights
                let mut adaptation_score = 0.0;
                for (strategy, weight) in strategies.iter().zip(weights.iter()) {
                    let score =
                        self.evaluate_strategy_score(strategy, stats, drift_detected, anomaly)?;
                    adaptation_score += score * weight;
                }
                if adaptation_score > 0.5 {
                    should_adapt = true;
                    trigger = AdaptationTrigger::MLRecommendation {
                        confidence: adaptation_score,
                    };
                }
            }
            WindowAdaptationStrategy::MLBased { .. } => {
                // ML-based adaptation using performance history
                if self.should_adapt_ml_based()? {
                    should_adapt = true;
                    trigger = AdaptationTrigger::MLRecommendation { confidence: 0.8 };
                }
            }
        }

        // Check for anomaly-triggered adaptation
        if anomaly.is_some() && !should_adapt {
            should_adapt = true;
            trigger = AdaptationTrigger::AnomalyDetected;
        }

        if should_adapt {
            let new_size = self.calculate_new_window_size(stats, drift_detected, anomaly)?;

            if new_size != self.current_window_size {
                self.current_window_size = new_size;
                self.last_adaptation = Some(Instant::now());

                let adaptation = WindowAdaptation {
                    timestamp: Instant::now(),
                    old_size,
                    new_size,
                    trigger,
                    performance_before: current_performance,
                    performance_after: None, // Will be updated later
                };

                self.adaptation_history.push_back(adaptation.clone());
                if self.adaptation_history.len() > 50 {
                    self.adaptation_history.pop_front();
                }

                return Ok(Some(adaptation));
            }
        }

        Ok(None)
    }

    fn evaluate_strategy_score(
        &self,
        strategy: &WindowAdaptationStrategy,
        stats: &StreamingStatistics<F>,
        drift_detected: bool,
        _anomaly: Option<&Anomaly<F>>,
    ) -> Result<f64> {
        let score = match strategy {
            WindowAdaptationStrategy::DriftBased => {
                if drift_detected {
                    1.0
                } else {
                    0.0
                }
            }
            WindowAdaptationStrategy::PerformanceBased { target_accuracy } => {
                let current = stats.current_accuracy.to_f64().unwrap_or(0.0);
                if current < *target_accuracy {
                    (*target_accuracy - current) / target_accuracy
                } else {
                    0.0
                }
            }
            _ => 0.0,
        };
        Ok(score)
    }

    fn should_adapt_ml_based(&self) -> Result<bool> {
        if self.performance_history.len() < 10 {
            return Ok(false);
        }

        // Simple trend analysis: check if performance is consistently declining
        let hist_len = self.performance_history.len();
        let recent: Vec<_> = self
            .performance_history
            .range((hist_len - 5)..)
            .cloned()
            .collect();
        let older: Vec<_> = self
            .performance_history
            .range((hist_len - 10)..(hist_len - 5))
            .cloned()
            .collect();

        let recent_avg = recent
            .iter()
            .map(|x| x.to_f64().unwrap_or(0.0))
            .sum::<f64>()
            / recent.len() as f64;
        let older_avg =
            older.iter().map(|x| x.to_f64().unwrap_or(0.0)).sum::<f64>() / older.len() as f64;

        // Adapt if performance declined by more than 5%
        Ok(recent_avg < older_avg * 0.95)
    }

    fn calculate_new_window_size(
        &self,
        stats: &StreamingStatistics<F>,
        drift_detected: bool,
        anomaly: Option<&Anomaly<F>>,
    ) -> Result<usize> {
        let current_accuracy = stats.current_accuracy.to_f64().unwrap_or(0.0);

        let mut size_multiplier = 1.0;

        // Adjust based on different factors
        if drift_detected {
            // Reduce window size to adapt faster to new concept
            size_multiplier *= 0.7;
        }

        if anomaly.is_some() {
            // Slightly reduce window to be more sensitive
            size_multiplier *= 0.9;
        }

        if current_accuracy < 0.6 {
            // Poor performance: reduce window size
            size_multiplier *= 0.8;
        } else if current_accuracy > 0.9 {
            // Good performance: can afford larger window
            size_multiplier *= 1.2;
        }

        // Apply variance based on recent performance stability
        if self.performance_history.len() > 5 {
            let recent_values: Vec<f64> = self
                .performance_history
                .iter()
                .rev()
                .take(5)
                .map(|x| x.to_f64().unwrap_or(0.0))
                .collect();

            let mean = recent_values.iter().sum::<f64>() / recent_values.len() as f64;
            let variance = recent_values
                .iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>()
                / recent_values.len() as f64;

            if variance > 0.01 {
                // High variance: smaller window for responsiveness
                size_multiplier *= 0.9;
            }
        }

        let new_size = ((self.current_window_size as f64) * size_multiplier) as usize;
        Ok(new_size.clamp(self.min_window_size, self.max_window_size))
    }

    pub fn adapt_for_drift(&mut self) -> Result<()> {
        // Aggressive adaptation for drift: reduce to minimum effective size
        let emergency_size = (self.min_window_size * 3).min(self.current_window_size / 2);
        self.current_window_size = emergency_size.max(self.min_window_size);
        self.last_adaptation = Some(Instant::now());

        let adaptation = WindowAdaptation {
            timestamp: Instant::now(),
            old_size: self.current_window_size,
            new_size: emergency_size,
            trigger: AdaptationTrigger::DriftDetected,
            performance_before: 0.0, // Will be updated
            performance_after: None,
        };

        self.adaptation_history.push_back(adaptation);
        if self.adaptation_history.len() > 50 {
            self.adaptation_history.pop_front();
        }

        Ok(())
    }

    pub fn reset(&mut self) {
        self.current_window_size = self.base_window_size;
        self.performance_history.clear();
        self.adaptation_history.clear();
        self.last_adaptation = None;
    }

    /// Get current window size
    pub fn get_current_size(&self) -> usize {
        self.current_window_size
    }

    /// Get adaptation history for analysis
    pub fn get_adaptation_history(&self) -> &VecDeque<WindowAdaptation> {
        &self.adaptation_history
    }

    /// Get performance history
    pub fn get_performance_history(&self) -> Vec<f64> {
        self.performance_history
            .iter()
            .map(|x| x.to_f64().unwrap_or(0.0))
            .collect()
    }
}
