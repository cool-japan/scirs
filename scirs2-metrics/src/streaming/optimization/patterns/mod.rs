//! Streaming patterns and windowing strategies
//!
//! This module provides comprehensive windowing and batching patterns:
//! - Adaptive windowing strategies
//! - Dynamic batching algorithms
//! - Buffer management patterns
//! - Stream partitioning strategies

pub mod windowing;
pub mod batching;
pub mod buffering;
pub mod partitioning;

pub use windowing::*;
pub use batching::*;
pub use buffering::*;
pub use partitioning::*;

use crate::error::{MetricsError, Result};
use scirs2_core::numeric::Float;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::{Duration, Instant, SystemTime};

pub use super::config::{WindowAdaptationStrategy, AdaptationTrigger, BatchingStrategy, BufferConfig, EvictionPolicy};

/// Adaptive window manager for streaming data
#[derive(Debug)]
pub struct AdaptiveWindowManager<F: Float + std::fmt::Debug> {
    /// Current window size
    current_window_size: usize,
    /// Minimum window size
    min_window_size: usize,
    /// Maximum window size
    max_window_size: usize,
    /// Adaptation strategy
    strategy: WindowAdaptationStrategy,
    /// Window data
    window_data: VecDeque<WindowDataPoint<F>>,
    /// Adaptation history
    adaptation_history: VecDeque<WindowAdaptation>,
    /// Performance tracker
    performance_tracker: WindowPerformanceTracker<F>,
    /// Last adaptation time
    last_adaptation: Instant,
    /// Adaptation triggers
    adaptation_triggers: Vec<AdaptationTrigger>,
}

impl<F: Float + std::fmt::Debug + Send + Sync> AdaptiveWindowManager<F> {
    /// Create a new adaptive window manager
    pub fn new(
        min_size: usize,
        max_size: usize,
        initial_size: usize,
        strategy: WindowAdaptationStrategy,
    ) -> Self {
        Self {
            current_window_size: initial_size.max(min_size).min(max_size),
            min_window_size: min_size,
            max_window_size: max_size,
            strategy,
            window_data: VecDeque::new(),
            adaptation_history: VecDeque::new(),
            performance_tracker: WindowPerformanceTracker::new(),
            last_adaptation: Instant::now(),
            adaptation_triggers: vec![],
        }
    }

    /// Add a data point to the window
    pub fn add_data_point(&mut self, value: F, timestamp: SystemTime) -> Result<WindowUpdateResult<F>> {
        let data_point = WindowDataPoint {
            value,
            timestamp,
            weight: F::one(),
        };

        self.window_data.push_back(data_point);

        // Maintain window size
        while self.window_data.len() > self.current_window_size {
            self.window_data.pop_front();
        }

        // Update performance metrics
        self.performance_tracker.update(&self.window_data);

        // Check for adaptation triggers
        let adaptation_needed = self.check_adaptation_triggers()?;

        let mut result = WindowUpdateResult {
            window_size: self.current_window_size,
            adaptation_performed: false,
            new_window_size: None,
            performance_metrics: self.performance_tracker.get_current_metrics(),
            trigger_reason: None,
        };

        if adaptation_needed.is_some() {
            let trigger = adaptation_needed.unwrap();
            let new_size = self.adapt_window_size(&trigger)?;
            
            if new_size != self.current_window_size {
                self.current_window_size = new_size;
                self.record_adaptation(trigger.clone(), new_size);
                
                result.adaptation_performed = true;
                result.new_window_size = Some(new_size);
                result.trigger_reason = Some(trigger);
                
                // Adjust window data to new size
                while self.window_data.len() > self.current_window_size {
                    self.window_data.pop_front();
                }
            }
        }

        Ok(result)
    }

    /// Check if adaptation triggers are met
    fn check_adaptation_triggers(&self) -> Result<Option<AdaptationTrigger>> {
        for trigger in &self.adaptation_triggers {
            match trigger {
                AdaptationTrigger::Time(duration) => {
                    if self.last_adaptation.elapsed() >= *duration {
                        return Ok(Some(trigger.clone()));
                    }
                }
                AdaptationTrigger::SampleCount(count) => {
                    if self.window_data.len() >= *count {
                        return Ok(Some(trigger.clone()));
                    }
                }
                AdaptationTrigger::Performance { accuracy_threshold, latency_threshold } => {
                    let metrics = self.performance_tracker.get_current_metrics();
                    if metrics.accuracy < *accuracy_threshold || 
                       metrics.processing_latency > *latency_threshold {
                        return Ok(Some(trigger.clone()));
                    }
                }
                AdaptationTrigger::Drift { confidence } => {
                    // This would typically be triggered by external drift detection
                    // For now, we'll use a simple heuristic based on variance
                    let metrics = self.performance_tracker.get_current_metrics();
                    if metrics.variance > F::from(*confidence).unwrap() {
                        return Ok(Some(trigger.clone()));
                    }
                }
                AdaptationTrigger::Manual => {
                    // Manual triggers would be set externally
                    continue;
                }
                AdaptationTrigger::Combined(triggers) => {
                    // Check if all combined triggers are met
                    let mut all_met = true;
                    for sub_trigger in triggers {
                        if self.check_single_trigger(sub_trigger)?.is_none() {
                            all_met = false;
                            break;
                        }
                    }
                    if all_met {
                        return Ok(Some(trigger.clone()));
                    }
                }
            }
        }
        Ok(None)
    }

    /// Check a single trigger condition
    fn check_single_trigger(&self, trigger: &AdaptationTrigger) -> Result<Option<AdaptationTrigger>> {
        match trigger {
            AdaptationTrigger::Time(duration) => {
                if self.last_adaptation.elapsed() >= *duration {
                    Ok(Some(trigger.clone()))
                } else {
                    Ok(None)
                }
            }
            AdaptationTrigger::SampleCount(count) => {
                if self.window_data.len() >= *count {
                    Ok(Some(trigger.clone()))
                } else {
                    Ok(None)
                }
            }
            _ => Ok(None), // Simplified for other trigger types
        }
    }

    /// Adapt window size based on strategy and trigger
    fn adapt_window_size(&self, trigger: &AdaptationTrigger) -> Result<usize> {
        let current_metrics = self.performance_tracker.get_current_metrics();
        
        let new_size = match &self.strategy {
            WindowAdaptationStrategy::Fixed => self.current_window_size,
            
            WindowAdaptationStrategy::ExponentialDecay { decay_rate } => {
                let decay = F::from(*decay_rate).unwrap();
                let adapted_size = F::from(self.current_window_size).unwrap() * decay;
                adapted_size.to_usize().unwrap_or(self.current_window_size)
            }
            
            WindowAdaptationStrategy::PerformanceBased { target_accuracy } => {
                if current_metrics.accuracy < *target_accuracy {
                    // Increase window size to improve accuracy
                    (self.current_window_size as f64 * 1.2) as usize
                } else {
                    // Decrease window size to improve responsiveness
                    (self.current_window_size as f64 * 0.9) as usize
                }
            }
            
            WindowAdaptationStrategy::DriftBased => {
                match trigger {
                    AdaptationTrigger::Drift { confidence } => {
                        if *confidence > 0.8 {
                            // High confidence drift - reduce window size significantly
                            (self.current_window_size as f64 * 0.5) as usize
                        } else {
                            // Low confidence drift - reduce window size moderately
                            (self.current_window_size as f64 * 0.8) as usize
                        }
                    }
                    _ => {
                        // Gradual adaptation
                        if current_metrics.variance > F::from(0.1).unwrap() {
                            (self.current_window_size as f64 * 0.9) as usize
                        } else {
                            (self.current_window_size as f64 * 1.1) as usize
                        }
                    }
                }
            }
            
            WindowAdaptationStrategy::Hybrid { strategies, weights } => {
                let mut weighted_size = 0.0;
                let mut total_weight = 0.0;
                
                for (strategy, weight) in strategies.iter().zip(weights.iter()) {
                    let temp_manager = AdaptiveWindowManager {
                        current_window_size: self.current_window_size,
                        min_window_size: self.min_window_size,
                        max_window_size: self.max_window_size,
                        strategy: strategy.clone(),
                        window_data: self.window_data.clone(),
                        adaptation_history: VecDeque::new(),
                        performance_tracker: self.performance_tracker.clone(),
                        last_adaptation: self.last_adaptation,
                        adaptation_triggers: vec![],
                    };
                    
                    let size = temp_manager.adapt_window_size(trigger)?;
                    weighted_size += size as f64 * weight;
                    total_weight += weight;
                }
                
                if total_weight > 0.0 {
                    (weighted_size / total_weight) as usize
                } else {
                    self.current_window_size
                }
            }
            
            WindowAdaptationStrategy::MLBased { model_type: _ } => {
                // Placeholder for ML-based adaptation
                // This would use a trained model to predict optimal window size
                self.current_window_size
            }
        };

        // Ensure new size is within bounds
        Ok(new_size.max(self.min_window_size).min(self.max_window_size))
    }

    /// Record an adaptation event
    fn record_adaptation(&mut self, trigger: AdaptationTrigger, new_size: usize) {
        let adaptation = WindowAdaptation {
            timestamp: SystemTime::now(),
            old_size: self.current_window_size,
            new_size,
            trigger,
            performance_before: self.performance_tracker.get_current_metrics(),
        };

        self.adaptation_history.push_back(adaptation);
        
        // Keep only recent adaptations
        while self.adaptation_history.len() > 100 {
            self.adaptation_history.pop_front();
        }

        self.last_adaptation = Instant::now();
    }

    /// Get current window statistics
    pub fn get_window_statistics(&self) -> WindowStatistics<F> {
        if self.window_data.is_empty() {
            return WindowStatistics {
                size: self.current_window_size,
                actual_size: 0,
                mean: F::zero(),
                variance: F::zero(),
                min_value: F::zero(),
                max_value: F::zero(),
                age: Duration::from_secs(0),
            };
        }

        let values: Vec<F> = self.window_data.iter().map(|p| p.value).collect();
        let mean = values.iter().copied().fold(F::zero(), |acc, x| acc + x) / F::from(values.len()).unwrap();
        
        let variance = values.iter()
            .map(|&x| {
                let diff = x - mean;
                diff * diff
            })
            .fold(F::zero(), |acc, x| acc + x) / F::from(values.len()).unwrap();

        let min_value = values.iter().copied().fold(F::infinity(), |acc, x| acc.min(x));
        let max_value = values.iter().copied().fold(F::neg_infinity(), |acc, x| acc.max(x));

        let oldest_timestamp = self.window_data.front().unwrap().timestamp;
        let age = SystemTime::now().duration_since(oldest_timestamp).unwrap_or(Duration::from_secs(0));

        WindowStatistics {
            size: self.current_window_size,
            actual_size: self.window_data.len(),
            mean,
            variance,
            min_value,
            max_value,
            age,
        }
    }

    /// Add an adaptation trigger
    pub fn add_trigger(&mut self, trigger: AdaptationTrigger) {
        self.adaptation_triggers.push(trigger);
    }

    /// Remove all adaptation triggers
    pub fn clear_triggers(&mut self) {
        self.adaptation_triggers.clear();
    }

    /// Get adaptation history
    pub fn get_adaptation_history(&self) -> Vec<WindowAdaptation> {
        self.adaptation_history.iter().cloned().collect()
    }

    /// Get current window size
    pub fn get_current_size(&self) -> usize {
        self.current_window_size
    }

    /// Manually set window size
    pub fn set_window_size(&mut self, size: usize) -> Result<()> {
        let new_size = size.max(self.min_window_size).min(self.max_window_size);
        
        if new_size != self.current_window_size {
            let trigger = AdaptationTrigger::Manual;
            self.record_adaptation(trigger, new_size);
            self.current_window_size = new_size;
            
            // Adjust window data to new size
            while self.window_data.len() > self.current_window_size {
                self.window_data.pop_front();
            }
        }
        
        Ok(())
    }
}

/// Data point in a window
#[derive(Debug, Clone)]
pub struct WindowDataPoint<F: Float + std::fmt::Debug> {
    /// Data value
    pub value: F,
    /// Timestamp
    pub timestamp: SystemTime,
    /// Weight for weighted calculations
    pub weight: F,
}

/// Window adaptation record
#[derive(Debug, Clone)]
pub struct WindowAdaptation {
    /// Adaptation timestamp
    pub timestamp: SystemTime,
    /// Old window size
    pub old_size: usize,
    /// New window size
    pub new_size: usize,
    /// Adaptation trigger
    pub trigger: AdaptationTrigger,
    /// Performance metrics before adaptation
    pub performance_before: WindowPerformanceMetrics<f64>,
}

/// Window update result
#[derive(Debug, Clone)]
pub struct WindowUpdateResult<F: Float + std::fmt::Debug> {
    /// Current window size
    pub window_size: usize,
    /// Whether adaptation was performed
    pub adaptation_performed: bool,
    /// New window size if adapted
    pub new_window_size: Option<usize>,
    /// Current performance metrics
    pub performance_metrics: WindowPerformanceMetrics<F>,
    /// Trigger reason if adapted
    pub trigger_reason: Option<AdaptationTrigger>,
}

/// Window statistics
#[derive(Debug, Clone)]
pub struct WindowStatistics<F: Float + std::fmt::Debug> {
    /// Configured window size
    pub size: usize,
    /// Actual number of data points
    pub actual_size: usize,
    /// Mean value
    pub mean: F,
    /// Variance
    pub variance: F,
    /// Minimum value
    pub min_value: F,
    /// Maximum value
    pub max_value: F,
    /// Age of oldest data point
    pub age: Duration,
}

/// Window performance tracker
#[derive(Debug, Clone)]
pub struct WindowPerformanceTracker<F: Float + std::fmt::Debug> {
    /// Current metrics
    current_metrics: WindowPerformanceMetrics<F>,
    /// Metrics history
    metrics_history: VecDeque<WindowPerformanceMetrics<F>>,
    /// Last update time
    last_update: Instant,
}

impl<F: Float + std::fmt::Debug> WindowPerformanceTracker<F> {
    /// Create a new performance tracker
    pub fn new() -> Self {
        Self {
            current_metrics: WindowPerformanceMetrics::default(),
            metrics_history: VecDeque::new(),
            last_update: Instant::now(),
        }
    }

    /// Update performance metrics based on window data
    pub fn update(&mut self, window_data: &VecDeque<WindowDataPoint<F>>) {
        if window_data.is_empty() {
            return;
        }

        let values: Vec<F> = window_data.iter().map(|p| p.value).collect();
        let mean = values.iter().copied().fold(F::zero(), |acc, x| acc + x) / F::from(values.len()).unwrap();
        
        let variance = values.iter()
            .map(|&x| {
                let diff = x - mean;
                diff * diff
            })
            .fold(F::zero(), |acc, x| acc + x) / F::from(values.len()).unwrap();

        let processing_time = self.last_update.elapsed();

        self.current_metrics = WindowPerformanceMetrics {
            accuracy: F::from(0.95).unwrap(), // Placeholder - would be calculated based on predictions
            variance,
            processing_latency: processing_time,
            throughput: F::from(values.len()).unwrap() / F::from(processing_time.as_secs_f64()).unwrap(),
            memory_usage: F::from(values.len() * std::mem::size_of::<F>()).unwrap(),
        };

        // Store metrics history
        self.metrics_history.push_back(self.current_metrics.clone());
        while self.metrics_history.len() > 1000 {
            self.metrics_history.pop_front();
        }

        self.last_update = Instant::now();
    }

    /// Get current performance metrics
    pub fn get_current_metrics(&self) -> WindowPerformanceMetrics<F> {
        self.current_metrics.clone()
    }

    /// Get metrics history
    pub fn get_metrics_history(&self) -> Vec<WindowPerformanceMetrics<F>> {
        self.metrics_history.iter().cloned().collect()
    }
}

/// Window performance metrics
#[derive(Debug, Clone)]
pub struct WindowPerformanceMetrics<F: Float + std::fmt::Debug> {
    /// Accuracy metric
    pub accuracy: F,
    /// Variance in the data
    pub variance: F,
    /// Processing latency
    pub processing_latency: Duration,
    /// Throughput (samples per second)
    pub throughput: F,
    /// Memory usage
    pub memory_usage: F,
}

impl<F: Float + std::fmt::Debug> Default for WindowPerformanceMetrics<F> {
    fn default() -> Self {
        Self {
            accuracy: F::one(),
            variance: F::zero(),
            processing_latency: Duration::from_millis(0),
            throughput: F::zero(),
            memory_usage: F::zero(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_window_manager_creation() {
        let manager = AdaptiveWindowManager::<f64>::new(
            10, 
            1000, 
            100, 
            WindowAdaptationStrategy::Fixed
        );
        
        assert_eq!(manager.get_current_size(), 100);
    }

    #[test]
    fn test_window_data_point_addition() {
        let mut manager = AdaptiveWindowManager::<f64>::new(
            10, 
            1000, 
            100, 
            WindowAdaptationStrategy::Fixed
        );
        
        let result = manager.add_data_point(1.0, SystemTime::now()).unwrap();
        assert_eq!(result.window_size, 100);
        assert!(!result.adaptation_performed);
    }

    #[test]
    fn test_window_size_bounds() {
        let mut manager = AdaptiveWindowManager::<f64>::new(
            10, 
            100, 
            50, 
            WindowAdaptationStrategy::Fixed
        );
        
        // Try to set size below minimum
        manager.set_window_size(5).unwrap();
        assert_eq!(manager.get_current_size(), 10);
        
        // Try to set size above maximum
        manager.set_window_size(200).unwrap();
        assert_eq!(manager.get_current_size(), 100);
    }

    #[test]
    fn test_window_statistics() {
        let mut manager = AdaptiveWindowManager::<f64>::new(
            10, 
            1000, 
            5, 
            WindowAdaptationStrategy::Fixed
        );
        
        // Add some data points
        for i in 0..5 {
            manager.add_data_point(i as f64, SystemTime::now()).unwrap();
        }
        
        let stats = manager.get_window_statistics();
        assert_eq!(stats.actual_size, 5);
        assert_eq!(stats.mean, 2.0); // (0+1+2+3+4)/5
    }

    #[test]
    fn test_adaptation_triggers() {
        let mut manager = AdaptiveWindowManager::<f64>::new(
            10, 
            1000, 
            100, 
            WindowAdaptationStrategy::PerformanceBased { target_accuracy: 0.9 }
        );
        
        manager.add_trigger(AdaptationTrigger::SampleCount(50));
        assert_eq!(manager.adaptation_triggers.len(), 1);
        
        manager.clear_triggers();
        assert_eq!(manager.adaptation_triggers.len(), 0);
    }

    #[test]
    fn test_performance_tracker() {
        let mut tracker = WindowPerformanceTracker::<f64>::new();
        let mut window_data = VecDeque::new();
        
        // Add some test data
        for i in 0..10 {
            window_data.push_back(WindowDataPoint {
                value: i as f64,
                timestamp: SystemTime::now(),
                weight: 1.0,
            });
        }
        
        tracker.update(&window_data);
        let metrics = tracker.get_current_metrics();
        
        assert!(metrics.throughput > 0.0);
        assert!(metrics.variance > 0.0);
    }
}