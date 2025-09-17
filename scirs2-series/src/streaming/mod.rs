//! Real-time streaming time series analysis module
//!
//! This module provides capabilities for analyzing time series data in real-time,
//! including online learning algorithms, streaming forecasting, incremental statistics,
//! change point detection, anomaly detection, and pattern matching.
//!
//! # Module Structure
//!
//! - `config` - Configuration types and change point definitions
//! - `statistics` - Online statistics and exponential weighted moving averages
//! - `change_detection` - Change point detection algorithms
//! - `online_learning` - Adaptive learning algorithms (regression, ARIMA)
//! - `forecasting` - Real-time forecasting with exponential smoothing
//! - `memory_management` - Memory-efficient utilities, anomaly detection, and pattern matching

use ndarray::{Array1, Array2};
use num_traits::{Float, FromPrimitive};
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::time::{Duration, Instant};

use crate::error::{Result, TimeSeriesError};
use statrs::statistics::Statistics;

// Declare sub-modules
pub mod change_detection;
pub mod config;
pub mod forecasting;
pub mod memory_management;
pub mod online_learning;
pub mod statistics;

// Re-export all public types for backward compatibility
pub use change_detection::CusumDetector;
pub use config::{ChangePoint, ChangeType, StreamConfig};
pub use forecasting::{ModelState, StreamingForecaster};
pub use memory_management::{
    CircularBuffer, MultiSeriesAnalyzer, PatternMatch, StreamingAnomalyDetector,
    StreamingPatternMatcher,
};
pub use online_learning::{AdaptiveARIMA, AdaptiveLinearRegression};
pub use statistics::{OnlineStats, EWMA};

/// Core streaming time series analyzer
///
/// This is the main interface for streaming time series analysis, providing
/// a unified API for real-time analysis capabilities.
#[derive(Debug)]
pub struct StreamingAnalyzer<F: Float + Debug> {
    /// Configuration parameters
    config: StreamConfig,
    /// Online statistics tracker
    stats: OnlineStats<F>,
    /// Exponential weighted moving average
    ewma: EWMA<F>,
    /// Change point detector
    cusum: CusumDetector<F>,
    /// Recent observations buffer
    buffer: VecDeque<F>,
    /// Last update timestamp
    last_update: Instant,
}

impl<F: Float + Debug + Clone + FromPrimitive> StreamingAnalyzer<F> {
    /// Create new streaming analyzer with default configuration
    pub fn new(config: StreamConfig) -> Result<Self> {
        let ewma = EWMA::new(F::from(0.1).unwrap())?;
        let cusum = CusumDetector::new(
            F::from(config.change_detection_threshold).unwrap(),
            F::from(0.5).unwrap(),
        );

        let window_size = config.window_size;

        Ok(Self {
            config,
            stats: OnlineStats::new(),
            ewma,
            cusum,
            buffer: VecDeque::with_capacity(window_size),
            last_update: Instant::now(),
        })
    }

    /// Add new observation to the analyzer
    pub fn add_observation(&mut self, value: F) -> Result<()> {
        // Update statistics
        self.stats.update(value);
        self.ewma.update(value);

        // Add to buffer
        if self.buffer.len() >= self.config.window_size {
            self.buffer.pop_front();
        }
        self.buffer.push_back(value);

        // Update change detection
        self.cusum.update(value);

        self.last_update = Instant::now();
        Ok(())
    }

    /// Get current statistics
    pub fn get_stats(&self) -> &OnlineStats<F> {
        &self.stats
    }

    /// Get current EWMA value
    pub fn get_ewma(&self) -> Option<F> {
        self.ewma.value()
    }

    /// Check for change points
    pub fn detect_change(&self) -> bool {
        self.cusum.is_change_detected()
    }

    /// Get change detection signal value
    pub fn get_change_signal(&self) -> (F, F) {
        self.cusum.get_signals()
    }

    /// Get recent observations buffer
    pub fn get_buffer(&self) -> &VecDeque<F> {
        &self.buffer
    }

    /// Get configuration
    pub fn get_config(&self) -> &StreamConfig {
        &self.config
    }

    /// Get time since last update
    pub fn time_since_update(&self) -> Duration {
        self.last_update.elapsed()
    }

    /// Reset the analyzer state
    pub fn reset(&mut self) {
        self.stats = OnlineStats::new();
        self.ewma = EWMA::new(F::from(0.1).unwrap()).unwrap();
        self.cusum = CusumDetector::new(
            F::from(self.config.change_detection_threshold).unwrap(),
            F::from(0.5).unwrap(),
        );
        self.buffer.clear();
        self.last_update = Instant::now();
    }

    /// Perform automatic memory cleanup if threshold is reached
    pub fn cleanup_memory(&mut self) {
        if self.buffer.len() > self.config.memory_threshold {
            let target_size = self.config.memory_threshold / 2;
            let to_remove = self.buffer.len() - target_size;

            for _ in 0..to_remove {
                self.buffer.pop_front();
            }
        }

        self.last_update = Instant::now();
    }

    /// Get change points (compatibility method)
    pub fn get_change_points(&self) -> Vec<ChangePoint> {
        // This is a simple implementation - in the original it might have been more sophisticated
        if self.detect_change() {
            vec![ChangePoint {
                index: self.stats.count(),
                timestamp: Some(Instant::now()),
                confidence: self
                    .cusum
                    .get_signals()
                    .0
                    .max(self.cusum.get_signals().1)
                    .to_f64()
                    .unwrap_or(0.0),
                change_type: ChangeType::MeanShift,
            }]
        } else {
            Vec::new()
        }
    }

    /// Check if a value is an outlier (simple implementation)
    pub fn is_outlier(&self, value: F) -> bool {
        if self.stats.count() < 10 {
            return false; // Not enough data
        }

        let mean = self.stats.mean();
        let std_dev = self.stats.std_dev();
        let z_score = ((value - mean) / std_dev).abs();

        // Consider outlier if z-score > 3
        z_score > F::from(3.0).unwrap()
    }

    /// Simple forecast method (compatibility)
    pub fn forecast(&self, _steps: usize) -> Result<Array1<F>> {
        // Simple implementation using EWMA as forecast
        if let Some(ewma_value) = self.ewma.value() {
            Ok(Array1::from_elem(_steps, ewma_value))
        } else {
            Ok(Array1::zeros(_steps))
        }
    }

    /// Get observation count
    pub fn observation_count(&self) -> usize {
        self.stats.count()
    }

    /// Get buffer size
    pub fn buffer_size(&self) -> usize {
        self.buffer.len()
    }

    /// Get time since last update (alias for compatibility)
    pub fn time_since_last_update(&self) -> Duration {
        self.time_since_update()
    }
}

// Re-export the adaptive module types (maintaining backward compatibility)
pub mod adaptive {
    pub use super::online_learning::{AdaptiveARIMA, AdaptiveLinearRegression};
}

// Re-export the advanced module types (maintaining backward compatibility)
pub mod advanced {
    pub use super::forecasting::{ModelState, StreamingForecaster};
    pub use super::memory_management::{
        CircularBuffer, PatternMatch, StreamingAnomalyDetector, StreamingPatternMatcher,
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_streaming_analyzer_basic() {
        let config = StreamConfig::default();
        let mut analyzer = StreamingAnalyzer::<f64>::new(config).unwrap();

        // Add some observations
        for i in 1..=10 {
            analyzer.add_observation(i as f64).unwrap();
        }

        // Check statistics
        let stats = analyzer.get_stats();
        assert_eq!(stats.count(), 10);
        assert_abs_diff_eq!(stats.mean(), 5.5, epsilon = 1e-10);
        assert_abs_diff_eq!(stats.min(), 1.0);
        assert_abs_diff_eq!(stats.max(), 10.0);
    }

    #[test]
    fn test_streaming_analyzer_ewma() {
        let config = StreamConfig::default();
        let mut analyzer = StreamingAnalyzer::<f64>::new(config).unwrap();

        analyzer.add_observation(10.0).unwrap();
        let ewma1 = analyzer.get_ewma().unwrap();
        assert_abs_diff_eq!(ewma1, 10.0);

        analyzer.add_observation(20.0).unwrap();
        let ewma2 = analyzer.get_ewma().unwrap();
        // EWMA should be between 10 and 20
        assert!(ewma2 > 10.0);
        assert!(ewma2 < 20.0);
    }

    #[test]
    fn test_streaming_analyzer_buffer() {
        let mut config = StreamConfig::default();
        config.window_size = 3;
        let mut analyzer = StreamingAnalyzer::<f64>::new(config).unwrap();

        // Add more data than buffer size
        for i in 1..=5 {
            analyzer.add_observation(i as f64).unwrap();
        }

        let buffer = analyzer.get_buffer();
        assert_eq!(buffer.len(), 3);
        // Should contain last 3 observations
        assert_eq!(*buffer.get(0).unwrap(), 3.0);
        assert_eq!(*buffer.get(1).unwrap(), 4.0);
        assert_eq!(*buffer.get(2).unwrap(), 5.0);
    }

    #[test]
    fn test_memory_cleanup() {
        let mut config = StreamConfig::default();
        config.window_size = 1000;
        config.memory_threshold = 50;
        let mut analyzer = StreamingAnalyzer::<f64>::new(config).unwrap();

        // Add many observations to exceed memory threshold
        for i in 1..=100 {
            analyzer.add_observation(i as f64).unwrap();
        }

        analyzer.cleanup_memory();

        // Buffer should be reduced to half the threshold
        assert_eq!(analyzer.get_buffer().len(), 25);
    }
}
