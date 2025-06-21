//! Statistical functions for ndarray arrays
//!
//! This module provides statistical functions for ndarray arrays, similar to `NumPy`'s
//! statistical functions.

// Module declarations
mod correlation;
mod descriptive;
mod distribution;
mod hypothesis;

// Re-exports from descriptive statistics module
pub use descriptive::{
    max, max_2d, mean, mean_2d, median, median_2d, min, min_2d, percentile, percentile_2d,
    product_2d, std_dev, std_dev_2d, sum_2d, variance, variance_2d,
};

// Re-exports from distribution-related module
pub use distribution::{
    bincount, digitize, histogram, histogram2d, quantile, Histogram2dResult, HistogramResult,
};

// Re-exports from correlation module
pub use correlation::{corrcoef, cov};

// Re-exports from hypothesis testing module
// (No functions yet)

// Re-export any other types that were previously exposed
