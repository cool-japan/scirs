//! Metrics-related callbacks
//!
//! This module provides callbacks for computing and tracking metrics during training.

mod scirs_metrics;

// Re-export ScirsMetricsCallback
#[allow(unused_imports)]
pub use scirs_metrics::ScirsMetricsCallback;
