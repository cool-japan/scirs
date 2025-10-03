//! Change point detection algorithms for streaming data
//!
//! This module provides algorithms for detecting changes in streaming time series data,
//! including CUSUM (Cumulative Sum) control charts for real-time change detection.

use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;
use std::time::Instant;

use super::config::{ChangePoint, ChangeType};

/// Cumulative Sum (CUSUM) change point detector
#[derive(Debug, Clone)]
pub struct CusumDetector<F: Float> {
    mean_estimate: F,
    threshold: F,
    cusum_pos: F,
    cusum_neg: F,
    count: usize,
    drift: F,
}

impl<F: Float + Debug> CusumDetector<F> {
    /// Create new CUSUM detector
    pub fn new(threshold: F, drift: F) -> Self {
        Self {
            mean_estimate: F::zero(),
            threshold,
            cusum_pos: F::zero(),
            cusum_neg: F::zero(),
            count: 0,
            drift,
        }
    }

    /// Update CUSUM with new observation
    pub fn update(&mut self, value: F) -> Option<ChangePoint> {
        self.count += 1;

        // Update mean estimate
        let delta = value - self.mean_estimate;
        self.mean_estimate = self.mean_estimate + delta / F::from(self.count).unwrap();

        // Update CUSUM statistics
        let diff = value - self.mean_estimate;
        self.cusum_pos = F::max(F::zero(), self.cusum_pos + diff - self.drift);
        self.cusum_neg = F::max(F::zero(), self.cusum_neg - diff - self.drift);

        // Check for change point
        if self.cusum_pos > self.threshold {
            self.reset();
            Some(ChangePoint {
                index: self.count,
                timestamp: Some(Instant::now()),
                confidence: self.cusum_pos.to_f64().unwrap_or(0.0),
                change_type: ChangeType::MeanShift,
            })
        } else if self.cusum_neg > self.threshold {
            self.reset();
            Some(ChangePoint {
                index: self.count,
                timestamp: Some(Instant::now()),
                confidence: self.cusum_neg.to_f64().unwrap_or(0.0),
                change_type: ChangeType::MeanShift,
            })
        } else {
            None
        }
    }

    /// Reset CUSUM statistics after change point detection
    fn reset(&mut self) {
        self.cusum_pos = F::zero();
        self.cusum_neg = F::zero();
    }

    /// Check if a change has been detected based on current signals
    pub fn is_change_detected(&self) -> bool {
        self.cusum_pos > self.threshold || self.cusum_neg > self.threshold
    }

    /// Get current CUSUM signals (positive, negative)
    pub fn get_signals(&self) -> (F, F) {
        (self.cusum_pos, self.cusum_neg)
    }

    /// Get current mean estimate
    pub fn get_mean_estimate(&self) -> F {
        self.mean_estimate
    }

    /// Get threshold
    pub fn get_threshold(&self) -> F {
        self.threshold
    }
}
