//! Configuration types for streaming time series analysis
//!
//! This module defines configuration structures and enums used throughout
//! the streaming time series analysis framework.

use std::time::Instant;

/// Configuration for streaming analysis
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Maximum window size for online calculations
    pub window_size: usize,
    /// Minimum number of observations before starting analysis
    pub min_observations: usize,
    /// Update frequency for model parameters
    pub update_frequency: usize,
    /// Memory threshold for automatic cleanup
    pub memory_threshold: usize,
    /// Enable adaptive windowing
    pub adaptive_windowing: bool,
    /// Detection threshold for change points
    pub change_detection_threshold: f64,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            window_size: 1000,
            min_observations: 10,
            update_frequency: 10,
            memory_threshold: 10000,
            adaptive_windowing: false,
            change_detection_threshold: 3.0,
        }
    }
}

/// Real-time change point detection result
#[derive(Debug, Clone)]
pub struct ChangePoint {
    /// Index of the change point
    pub index: usize,
    /// Timestamp of the change point
    pub timestamp: Option<Instant>,
    /// Confidence score (higher = more confident)
    pub confidence: f64,
    /// Type of change detected
    pub change_type: ChangeType,
}

/// Types of changes that can be detected
#[derive(Debug, Clone)]
pub enum ChangeType {
    /// Change in mean
    MeanShift,
    /// Change in variance
    VarianceShift,
    /// Change in trend
    TrendChange,
    /// Change in seasonality
    SeasonalityChange,
    /// General structural break
    StructuralBreak,
}
