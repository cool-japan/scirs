//! Incremental statistics and metrics for streaming data
//!
//! This module provides online algorithms for computing statistics on streaming data,
//! including Welford's algorithm for variance and exponentially weighted moving averages.

use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use crate::error::{Result, TimeSeriesError};

/// Online statistics tracker using Welford's algorithm
#[derive(Debug, Clone)]
pub struct OnlineStats<F: Float> {
    count: usize,
    mean: F,
    m2: F, // For variance calculation
    min_val: F,
    max_val: F,
    sum: F,
    sum_squares: F,
}

impl<F: Float + Debug> OnlineStats<F> {
    /// Create new online statistics tracker
    pub fn new() -> Self {
        Self {
            count: 0,
            mean: F::zero(),
            m2: F::zero(),
            min_val: F::infinity(),
            max_val: F::neg_infinity(),
            sum: F::zero(),
            sum_squares: F::zero(),
        }
    }

    /// Update statistics with new observation
    pub fn update(&mut self, value: F) {
        self.count += 1;
        self.sum = self.sum + value;
        self.sum_squares = self.sum_squares + value * value;

        if value < self.min_val {
            self.min_val = value;
        }
        if value > self.max_val {
            self.max_val = value;
        }

        // Welford's online algorithm for mean and variance
        let delta = value - self.mean;
        self.mean = self.mean + delta / F::from(self.count).unwrap();
        let delta2 = value - self.mean;
        self.m2 = self.m2 + delta * delta2;
    }

    /// Get current mean
    pub fn mean(&self) -> F {
        self.mean
    }

    /// Get current variance
    pub fn variance(&self) -> F {
        if self.count < 2 {
            F::zero()
        } else {
            self.m2 / F::from(self.count - 1).unwrap()
        }
    }

    /// Get current standard deviation
    pub fn std_dev(&self) -> F {
        self.variance().sqrt()
    }

    /// Get current minimum
    pub fn min(&self) -> F {
        self.min_val
    }

    /// Get current maximum
    pub fn max(&self) -> F {
        self.max_val
    }

    /// Get current count
    pub fn count(&self) -> usize {
        self.count
    }

    /// Get current sum
    pub fn sum(&self) -> F {
        self.sum
    }
}

impl<F: Float + Debug> Default for OnlineStats<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Exponentially Weighted Moving Average (EWMA) tracker
#[derive(Debug, Clone)]
pub struct EWMA<F: Float> {
    alpha: F,
    current_value: Option<F>,
    variance: Option<F>,
}

impl<F: Float + Debug> EWMA<F> {
    /// Create new EWMA tracker
    pub fn new(alpha: F) -> Result<Self> {
        if alpha <= F::zero() || alpha > F::one() {
            return Err(TimeSeriesError::InvalidParameter {
                name: "alpha".to_string(),
                message: "Alpha must be between 0 and 1".to_string(),
            });
        }

        Ok(Self {
            alpha,
            current_value: None,
            variance: None,
        })
    }

    /// Update EWMA with new observation
    pub fn update(&mut self, value: F) {
        match self.current_value {
            None => {
                self.current_value = Some(value);
                self.variance = Some(F::zero());
            }
            Some(prev) => {
                let new_value = self.alpha * value + (F::one() - self.alpha) * prev;
                self.current_value = Some(new_value);

                // Update variance estimate
                let error = value - new_value;
                let new_variance = self.alpha * error * error
                    + (F::one() - self.alpha) * self.variance.unwrap_or(F::zero());
                self.variance = Some(new_variance);
            }
        }
    }

    /// Get current EWMA value
    pub fn value(&self) -> Option<F> {
        self.current_value
    }

    /// Get current variance estimate
    pub fn variance(&self) -> Option<F> {
        self.variance
    }

    /// Check if value is an outlier based on EWMA
    pub fn is_outlier(&self, value: F, threshold: F) -> bool {
        if let (Some(ewma), Some(var)) = (self.current_value, self.variance) {
            let std_dev = var.sqrt();
            let z_score = (value - ewma).abs() / std_dev;
            z_score > threshold
        } else {
            false
        }
    }
}
