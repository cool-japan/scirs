//! Adaptive progress tracking
//!
//! This module provides adaptive algorithms for intelligent progress tracking,
//! including dynamic update rates and predictive ETA calculations.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Adaptive update rate controller
pub struct AdaptiveController {
    /// Recent update intervals
    update_intervals: VecDeque<Duration>,
    /// Recent processing speeds
    processing_speeds: VecDeque<f64>,
    /// Minimum allowed update interval
    min_interval: Duration,
    /// Maximum allowed update interval
    max_interval: Duration,
    /// Target update frequency (updates per second)
    target_frequency: f64,
}

impl AdaptiveController {
    /// Create a new adaptive controller
    pub fn new(min_interval: Duration, max_interval: Duration) -> Self {
        Self {
            update_intervals: VecDeque::with_capacity(20),
            processing_speeds: VecDeque::with_capacity(20),
            min_interval,
            max_interval,
            target_frequency: 2.0, // 2 updates per second by default
        }
    }

    /// Set the target update frequency
    pub fn set_target_frequency(&mut self, frequency: f64) {
        self.target_frequency = frequency.clamp(0.1, 10.0); // Clamp between 0.1 and 10 Hz
    }

    /// Record an update interval
    pub fn record_update(&mut self, interval: Duration, processing_speed: f64) {
        self.update_intervals.push_back(interval);
        self.processing_speeds.push_back(processing_speed);

        // Keep only recent measurements
        if self.update_intervals.len() > 20 {
            self.update_intervals.pop_front();
            self.processing_speeds.pop_front();
        }
    }

    /// Calculate the next optimal update interval
    pub fn next_interval(&self, current_progress: f64) -> Duration {
        // Base interval from target frequency
        let base_interval = Duration::from_secs_f64(1.0 / self.target_frequency);

        // Adjust based on progress position
        let position_factor = self.calculate_position_factor(current_progress);

        // Adjust based on processing speed stability
        let stability_factor = self.calculate_stability_factor();

        // Combine factors
        let adjusted_interval = base_interval.mul_f64(position_factor * stability_factor);

        // Clamp to min/max bounds
        adjusted_interval
            .max(self.min_interval)
            .min(self.max_interval)
    }

    /// Calculate position-based adjustment factor
    fn calculate_position_factor(&self, progress: f64) -> f64 {
        // Update more frequently at start and end, less in the middle
        // This creates a U-shaped curve
        let normalized_progress = progress.clamp(0.0, 1.0);
        let middle_distance = (0.5 - normalized_progress).abs() * 2.0; // 0 at middle, 1 at edges

        // Factor ranges from 0.5 (at middle) to 1.0 (at edges)
        0.5 + 0.5 * middle_distance
    }

    /// Calculate stability-based adjustment factor
    fn calculate_stability_factor(&self) -> f64 {
        if self.processing_speeds.len() < 2 {
            return 1.0;
        }

        // Calculate coefficient of variation for processing speeds
        let mean: f64 =
            self.processing_speeds.iter().sum::<f64>() / self.processing_speeds.len() as f64;

        if mean <= 0.0 {
            return 1.0;
        }

        let variance: f64 = self
            .processing_speeds
            .iter()
            .map(|&speed| (speed - mean).powi(2))
            .sum::<f64>()
            / self.processing_speeds.len() as f64;

        let std_dev = variance.sqrt();
        let cv = std_dev / mean; // Coefficient of variation

        // If processing is stable (low CV), we can update less frequently
        // If processing is unstable (high CV), we should update more frequently
        // Factor ranges from 0.5 (very stable) to 2.0 (very unstable)
        (1.0 + cv).clamp(0.5, 2.0)
    }

    /// Get average processing speed
    pub fn average_speed(&self) -> f64 {
        if self.processing_speeds.is_empty() {
            return 0.0;
        }

        self.processing_speeds.iter().sum::<f64>() / self.processing_speeds.len() as f64
    }

    /// Predict if processing speed is increasing or decreasing
    pub fn speed_trend(&self) -> SpeedTrend {
        if self.processing_speeds.len() < 3 {
            return SpeedTrend::Stable;
        }

        let recent_half = self.processing_speeds.len() / 2;
        let early_speeds: f64 =
            self.processing_speeds.iter().take(recent_half).sum::<f64>() / recent_half as f64;

        let late_speeds: f64 = self.processing_speeds.iter().skip(recent_half).sum::<f64>()
            / (self.processing_speeds.len() - recent_half) as f64;

        let change_ratio = if early_speeds > 0.0 {
            late_speeds / early_speeds
        } else {
            1.0
        };

        if change_ratio > 1.1 {
            SpeedTrend::Increasing
        } else if change_ratio < 0.9 {
            SpeedTrend::Decreasing
        } else {
            SpeedTrend::Stable
        }
    }
}

/// Processing speed trend
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpeedTrend {
    /// Processing speed is increasing
    Increasing,
    /// Processing speed is decreasing
    Decreasing,
    /// Processing speed is stable
    Stable,
}

/// Predictive ETA calculator
pub struct PredictiveETA {
    /// Historical progress measurements
    progress_history: VecDeque<(Instant, u64)>,
    /// Historical speed measurements
    speed_history: VecDeque<f64>,
    /// Maximum history length
    max_history: usize,
}

impl PredictiveETA {
    /// Create a new predictive ETA calculator
    pub fn new() -> Self {
        Self {
            progress_history: VecDeque::with_capacity(50),
            speed_history: VecDeque::with_capacity(50),
            max_history: 50,
        }
    }

    /// Record a progress measurement
    pub fn record_progress(&mut self, time: Instant, processed: u64, speed: f64) {
        self.progress_history.push_back((time, processed));
        self.speed_history.push_back(speed);

        // Maintain maximum history size
        if self.progress_history.len() > self.max_history {
            self.progress_history.pop_front();
            self.speed_history.pop_front();
        }
    }

    /// Calculate predictive ETA using multiple methods
    pub fn calculate_eta(&self, current_processed: u64, total: u64) -> Duration {
        if current_processed >= total || self.progress_history.len() < 2 {
            return Duration::from_secs(0);
        }

        let remaining = total - current_processed;

        // Method 1: Linear regression on recent progress
        let linear_eta = self.linear_regression_eta(remaining);

        // Method 2: Exponential smoothing of speeds
        let smoothed_eta = self.exponential_smoothing_eta(remaining);

        // Method 3: Simple moving average
        let average_eta = self.moving_average_eta(remaining);

        // Combine methods with weights based on data quality
        let weights = self.calculate_method_weights();

        let combined_seconds = linear_eta.as_secs_f64() * weights.0
            + smoothed_eta.as_secs_f64() * weights.1
            + average_eta.as_secs_f64() * weights.2;

        Duration::from_secs_f64(combined_seconds.max(0.0))
    }

    /// Calculate ETA using linear regression
    fn linear_regression_eta(&self, remaining: u64) -> Duration {
        if self.progress_history.len() < 3 {
            return self.moving_average_eta(remaining);
        }

        // Perform simple linear regression on time vs progress
        let n = self.progress_history.len() as f64;
        let start_time = self.progress_history[0].0;

        let (sum_x, sum_y, sum_xy, sum_x2) = self.progress_history.iter().fold(
            (0.0, 0.0, 0.0, 0.0),
            |(sx, sy, sxy, sx2), &(time, progress)| {
                let x = time.duration_since(start_time).as_secs_f64();
                let y = progress as f64;
                (sx + x, sy + y, sxy + x * y, sx2 + x * x)
            },
        );

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);

        if slope > 0.0 {
            let remaining_time = remaining as f64 / slope;
            Duration::from_secs_f64(remaining_time)
        } else {
            self.moving_average_eta(remaining)
        }
    }

    /// Calculate ETA using exponential smoothing
    fn exponential_smoothing_eta(&self, remaining: u64) -> Duration {
        if self.speed_history.is_empty() {
            return Duration::from_secs(0);
        }

        // Apply exponential smoothing to recent speeds
        let alpha = 0.3; // Smoothing factor
        let mut smoothed_speed = self.speed_history[0];

        for &speed in self.speed_history.iter().skip(1) {
            smoothed_speed = alpha * speed + (1.0 - alpha) * smoothed_speed;
        }

        if smoothed_speed > 0.0 {
            Duration::from_secs_f64(remaining as f64 / smoothed_speed)
        } else {
            Duration::from_secs(0)
        }
    }

    /// Calculate ETA using moving average
    fn moving_average_eta(&self, remaining: u64) -> Duration {
        if self.speed_history.is_empty() {
            return Duration::from_secs(0);
        }

        let recent_count = (self.speed_history.len() / 2).clamp(1, 10);
        let recent_speeds: Vec<_> = self.speed_history.iter().rev().take(recent_count).collect();

        let avg_speed: f64 =
            recent_speeds.iter().map(|&&s| s).sum::<f64>() / recent_speeds.len() as f64;

        if avg_speed > 0.0 {
            Duration::from_secs_f64(remaining as f64 / avg_speed)
        } else {
            Duration::from_secs(0)
        }
    }

    /// Calculate weights for combining different ETA methods
    fn calculate_method_weights(&self) -> (f64, f64, f64) {
        let history_len = self.progress_history.len();

        if history_len < 3 {
            // Not enough data for linear regression, rely on others
            (0.0, 0.4, 0.6)
        } else if history_len < 10 {
            // Some data, but still prefer simpler methods
            (0.3, 0.4, 0.3)
        } else {
            // Enough data for reliable linear regression
            (0.5, 0.3, 0.2)
        }
    }
}

impl Default for PredictiveETA {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_controller_creation() {
        let controller =
            AdaptiveController::new(Duration::from_millis(100), Duration::from_secs(1));

        assert_eq!(controller.min_interval, Duration::from_millis(100));
        assert_eq!(controller.max_interval, Duration::from_secs(1));
    }

    #[test]
    fn test_adaptive_controller_position_factor() {
        let controller =
            AdaptiveController::new(Duration::from_millis(100), Duration::from_secs(1));

        // Test U-shaped curve - more frequent updates at start and end
        let start_factor = controller.calculate_position_factor(0.0);
        let middle_factor = controller.calculate_position_factor(0.5);
        let end_factor = controller.calculate_position_factor(1.0);

        assert!(start_factor > middle_factor);
        assert!(end_factor > middle_factor);
        assert!((start_factor - end_factor).abs() < 0.1);
    }

    #[test]
    fn test_speed_trend_detection() {
        let mut controller =
            AdaptiveController::new(Duration::from_millis(100), Duration::from_secs(1));

        // Simulate increasing speeds
        for i in 1..=10 {
            controller.record_update(Duration::from_millis(100), i as f64 * 10.0);
        }

        assert_eq!(controller.speed_trend(), SpeedTrend::Increasing);
    }

    #[test]
    fn test_predictive_eta() {
        let mut eta_calc = PredictiveETA::new();
        let start_time = Instant::now();

        // Simulate steady progress
        for i in 1..=10 {
            let time = start_time + Duration::from_secs(i);
            eta_calc.record_progress(time, i * 10, 10.0);
        }

        let eta = eta_calc.calculate_eta(50, 100);

        // Should estimate roughly 5 seconds remaining (50 items at 10 items/sec)
        assert!((eta.as_secs_f64() - 5.0).abs() < 2.0);
    }
}
