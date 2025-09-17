//! Concept drift detection algorithms for streaming data
//!
//! This module provides implementations of various drift detection algorithms
//! including ADWIN, DDM, and Page-Hinkley test.

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use super::core::{ConceptDriftDetector, DriftDetectionResult, DriftStatistics, DriftStatus};
use crate::error::{MetricsError, Result};
use num_traits::Float;
use std::collections::{HashMap, VecDeque};
use std::time::SystemTime;

/// ADWIN drift detector implementation
#[derive(Debug, Clone)]
pub struct AdwinDetector<F: Float + std::fmt::Debug> {
    confidence: f64,
    window: VecDeque<F>,
    total_sum: F,
    width: usize,
    variance: F,
    bucket_number: usize,
    last_bucket_row: usize,
    buckets: Vec<Bucket<F>>,
    drift_count: usize,
    warning_count: usize,
    samples_count: usize,
}

/// Bucket for ADWIN algorithm - optimized for memory efficiency
#[derive(Debug, Clone)]
struct Bucket<F: Float + std::fmt::Debug> {
    _maxbuckets: usize,
    sum: Vec<F>,
    variance: Vec<F>,
    width: Vec<usize>,
    used_buckets: usize,
}

impl<F: Float + std::fmt::Debug + Send + Sync> Bucket<F> {
    fn new(_maxbuckets: usize) -> Self {
        Self {
            _maxbuckets,
            sum: vec![F::zero(); _maxbuckets],
            variance: vec![F::zero(); _maxbuckets],
            width: vec![0; _maxbuckets],
            used_buckets: 0,
        }
    }

    /// Add a new bucket with optimized memory management
    fn add_bucket(&mut self, sum: F, variance: F, width: usize) -> Result<()> {
        if self.used_buckets >= self._maxbuckets {
            // Compress by merging oldest buckets
            self.compress_oldest_buckets();
        }

        if self.used_buckets < self._maxbuckets {
            self.sum[self.used_buckets] = sum;
            self.variance[self.used_buckets] = variance;
            self.width[self.used_buckets] = width;
            self.used_buckets += 1;
            Ok(())
        } else {
            Err(MetricsError::ComputationError(
                "Cannot add bucket: maximum capacity reached".to_string(),
            ))
        }
    }

    /// Compress oldest buckets to save memory
    fn compress_oldest_buckets(&mut self) {
        if self.used_buckets >= 2 {
            // Merge first two buckets
            self.sum[0] = self.sum[0] + self.sum[1];
            self.variance[0] = self.variance[0] + self.variance[1];
            self.width[0] = self.width[0] + self.width[1];

            // Shift remaining buckets down
            for i in 1..(self.used_buckets - 1) {
                self.sum[i] = self.sum[i + 1];
                self.variance[i] = self.variance[i + 1];
                self.width[i] = self.width[i + 1];
            }
            self.used_buckets -= 1;
        }
    }

    /// Get total statistics efficiently
    fn get_total(&self) -> (F, F, usize) {
        let mut total_sum = F::zero();
        let mut total_variance = F::zero();
        let mut total_width = 0;

        for i in 0..self.used_buckets {
            total_sum = total_sum + self.sum[i];
            total_variance = total_variance + self.variance[i];
            total_width += self.width[i];
        }

        (total_sum, total_variance, total_width)
    }

    /// Clear all buckets
    fn clear(&mut self) {
        for i in 0..self.used_buckets {
            self.sum[i] = F::zero();
            self.variance[i] = F::zero();
            self.width[i] = 0;
        }
        self.used_buckets = 0;
    }
}

impl<F: Float + std::iter::Sum + std::fmt::Debug + Send + Sync> AdwinDetector<F> {
    pub fn new(confidence: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&confidence) {
            return Err(MetricsError::InvalidInput(
                "Confidence must be between 0 and 1".to_string(),
            ));
        }

        Ok(Self {
            confidence,
            window: VecDeque::with_capacity(1000),
            total_sum: F::zero(),
            width: 0,
            variance: F::zero(),
            bucket_number: 0,
            last_bucket_row: 0,
            buckets: vec![Bucket::new(5)], // Start with 5 buckets per row
            drift_count: 0,
            warning_count: 0,
            samples_count: 0,
        })
    }

    /// Optimized window management with efficient memory usage
    fn compress_buckets(&mut self) {
        // Implement bucket compression to maintain memory efficiency
        if self.bucket_number >= self.buckets[0]._maxbuckets {
            // Merge oldest buckets to save memory
            for bucket in &mut self.buckets {
                if bucket.used_buckets > 1 {
                    // Merge two oldest buckets
                    bucket.sum[0] = bucket.sum[0] + bucket.sum[1];
                    bucket.variance[0] = bucket.variance[0] + bucket.variance[1];
                    bucket.width[0] = bucket.width[0] + bucket.width[1];

                    // Shift remaining buckets
                    for i in 1..(bucket.used_buckets - 1) {
                        bucket.sum[i] = bucket.sum[i + 1];
                        bucket.variance[i] = bucket.variance[i + 1];
                        bucket.width[i] = bucket.width[i + 1];
                    }
                    bucket.used_buckets -= 1;
                }
            }
        }
    }

    /// Efficient cut detection using statistical bounds
    fn detect_change(&mut self) -> bool {
        if self.width < 2 {
            return false;
        }

        let mut change_detected = false;
        let delta = F::from((1.0 / self.confidence).ln() / 2.0).unwrap();

        // Check for significant difference in subwindows
        for cut_point in 1..self.width {
            let w0 = cut_point;
            let w1 = self.width - cut_point;

            if w0 >= 5 && w1 >= 5 {
                // Minimum subwindow size
                let mean0 = self.calculate_subwindow_mean(0, cut_point);
                let mean1 = self.calculate_subwindow_mean(cut_point, self.width);

                let var0 = self.calculate_subwindow_variance(0, cut_point, mean0);
                let var1 = self.calculate_subwindow_variance(cut_point, self.width, mean1);

                let epsilon =
                    (delta * (var0 / F::from(w0).unwrap() + var1 / F::from(w1).unwrap())).sqrt();

                if (mean0 - mean1).abs() > epsilon {
                    // Change detected - remove old data
                    self.remove_subwindow(0, cut_point);
                    change_detected = true;
                    break;
                }
            }
        }

        change_detected
    }

    fn calculate_subwindow_mean(&self, start: usize, end: usize) -> F {
        if start >= end || end > self.window.len() {
            return F::zero();
        }

        let sum = self.window.range(start..end).cloned().sum::<F>();
        sum / F::from(end - start).unwrap()
    }

    fn calculate_subwindow_variance(&self, start: usize, end: usize, mean: F) -> F {
        if start >= end || end > self.window.len() {
            return F::zero();
        }

        let variance = self
            .window
            .range(start..end)
            .map(|&x| (x - mean) * (x - mean))
            .sum::<F>();
        variance / F::from(end - start).unwrap()
    }

    fn remove_subwindow(&mut self, start: usize, end: usize) {
        for _ in start..end {
            if let Some(removed) = self.window.pop_front() {
                self.total_sum = self.total_sum - removed;
                self.width -= 1;
            }
        }
        // Recalculate variance efficiently
        self.update_variance();
    }

    fn update_variance(&mut self) {
        if self.width < 2 {
            self.variance = F::zero();
            return;
        }

        let mean = self.total_sum / F::from(self.width).unwrap();
        self.variance = self
            .window
            .iter()
            .map(|&x| (x - mean) * (x - mean))
            .sum::<F>()
            / F::from(self.width - 1).unwrap();
    }
}

impl<F: Float + std::iter::Sum + std::fmt::Debug + Send + Sync> ConceptDriftDetector<F>
    for AdwinDetector<F>
{
    fn update(&mut self, prediction_correct: bool, error: F) -> Result<DriftDetectionResult> {
        self.samples_count += 1;

        // Add error value to window
        self.window.push_back(error);
        self.total_sum = self.total_sum + error;
        self.width += 1;

        // Compress buckets if needed for memory efficiency
        if self.width % 100 == 0 {
            self.compress_buckets();
        }

        // Detect concept drift
        let change_detected = self.detect_change();

        let status = if change_detected {
            self.drift_count += 1;
            DriftStatus::Drift
        } else {
            DriftStatus::Stable
        };

        let mut statistics = HashMap::new();
        statistics.insert("window_size".to_string(), self.width as f64);
        statistics.insert("total_drifts".to_string(), self.drift_count as f64);
        statistics.insert("confidence".to_string(), self.confidence);

        Ok(DriftDetectionResult {
            status,
            confidence: self.confidence,
            change_point: if change_detected {
                Some(self.samples_count)
            } else {
                None
            },
            statistics,
        })
    }

    fn get_status(&self) -> DriftStatus {
        if self.drift_count > 0 && self.samples_count > 0 {
            // Consider recent drift activity
            let recent_drift_rate = self.drift_count as f64 / (self.samples_count as f64 / 100.0);
            if recent_drift_rate > 1.0 {
                DriftStatus::Drift
            } else if recent_drift_rate > 0.1 {
                DriftStatus::Warning
            } else {
                DriftStatus::Stable
            }
        } else {
            DriftStatus::Stable
        }
    }

    fn reset(&mut self) {
        self.window.clear();
        self.total_sum = F::zero();
        self.width = 0;
        self.variance = F::zero();
        self.bucket_number = 0;
        self.buckets.clear();
        self.buckets.push(Bucket::new(5));
        self.samples_count = 0;
        // Keep drift_count and warning_count for historical tracking
    }

    fn get_config(&self) -> HashMap<String, f64> {
        let mut config = HashMap::new();
        config.insert("confidence".to_string(), self.confidence);
        config.insert("max_window_size".to_string(), self.window.capacity() as f64);
        config
    }

    fn get_statistics(&self) -> DriftStatistics<F> {
        let current_error_rate = if self.width > 0 {
            self.total_sum / F::from(self.width).unwrap()
        } else {
            F::zero()
        };

        DriftStatistics {
            samples_since_reset: self.samples_count,
            warnings_count: self.warning_count,
            drifts_count: self.drift_count,
            current_error_rate,
            baseline_error_rate: if self.width > 10 {
                // Use first 10% as baseline
                let baseline_size = self.width / 10;
                self.window.iter().take(baseline_size).cloned().sum::<F>()
                    / F::from(baseline_size).unwrap()
            } else {
                current_error_rate
            },
            drift_score: self.variance,
            last_detection_time: if self.drift_count > 0 {
                Some(SystemTime::now())
            } else {
                None
            },
        }
    }
}

/// DDM (Drift Detection Method) implementation
#[derive(Debug, Clone)]
pub struct DdmDetector<F: Float + std::fmt::Debug> {
    warning_level: f64,
    drift_level: f64,
    min_instances: usize,
    num_errors: usize,
    num_instances: usize,
    p_min: F,
    s_min: F,
    p_last: F,
    s_last: F,
    status: DriftStatus,
    warning_count: usize,
    drift_count: usize,
}

impl<F: Float + std::fmt::Debug + Send + Sync> DdmDetector<F> {
    pub fn new(warning_level: f64, drift_level: f64) -> Self {
        Self {
            warning_level,
            drift_level,
            min_instances: 30,
            num_errors: 0,
            num_instances: 0,
            p_min: F::infinity(),
            s_min: F::infinity(),
            p_last: F::zero(),
            s_last: F::zero(),
            status: DriftStatus::Stable,
            warning_count: 0,
            drift_count: 0,
        }
    }
}

impl<F: Float + std::fmt::Debug + Send + Sync + std::iter::Sum> ConceptDriftDetector<F>
    for DdmDetector<F>
{
    fn update(&mut self, prediction_correct: bool, _error: F) -> Result<DriftDetectionResult> {
        self.num_instances += 1;
        if !prediction_correct {
            self.num_errors += 1;
        }

        if self.num_instances >= self.min_instances {
            let p = F::from(self.num_errors).unwrap() / F::from(self.num_instances).unwrap();
            let s = (p * (F::one() - p) / F::from(self.num_instances).unwrap()).sqrt();

            self.p_last = p;
            self.s_last = s;

            if p + s < self.p_min + self.s_min {
                self.p_min = p;
                self.s_min = s;
            }

            let warning_threshold = F::from(self.warning_level).unwrap();
            let drift_threshold = F::from(self.drift_level).unwrap();

            if p + s > self.p_min + warning_threshold * self.s_min {
                if p + s > self.p_min + drift_threshold * self.s_min {
                    self.status = DriftStatus::Drift;
                    self.drift_count += 1;
                } else {
                    self.status = DriftStatus::Warning;
                    self.warning_count += 1;
                }
            } else {
                self.status = DriftStatus::Stable;
            }
        }

        Ok(DriftDetectionResult {
            status: self.status.clone(),
            confidence: 0.8,
            change_point: if self.status == DriftStatus::Drift {
                Some(self.num_instances)
            } else {
                None
            },
            statistics: HashMap::new(),
        })
    }

    fn get_status(&self) -> DriftStatus {
        self.status.clone()
    }

    fn reset(&mut self) {
        self.num_errors = 0;
        self.num_instances = 0;
        self.p_min = F::infinity();
        self.s_min = F::infinity();
        self.status = DriftStatus::Stable;
    }

    fn get_config(&self) -> HashMap<String, f64> {
        let mut config = HashMap::new();
        config.insert("warning_level".to_string(), self.warning_level);
        config.insert("drift_level".to_string(), self.drift_level);
        config
    }

    fn get_statistics(&self) -> DriftStatistics<F> {
        DriftStatistics {
            samples_since_reset: self.num_instances,
            warnings_count: self.warning_count,
            drifts_count: self.drift_count,
            current_error_rate: if self.num_instances > 0 {
                F::from(self.num_errors).unwrap() / F::from(self.num_instances).unwrap()
            } else {
                F::zero()
            },
            baseline_error_rate: F::zero(),
            drift_score: self.p_last + self.s_last,
            last_detection_time: None,
        }
    }
}

/// Page-Hinkley test implementation
#[derive(Debug, Clone)]
pub struct PageHinkleyDetector<F: Float + std::fmt::Debug> {
    threshold: f64,
    alpha: f64,
    cumulative_sum: F,
    min_cumulative_sum: F,
    status: DriftStatus,
    samples_count: usize,
    drift_count: usize,
    warning_count: usize,
}

impl<F: Float + std::fmt::Debug + Send + Sync> PageHinkleyDetector<F> {
    pub fn new(threshold: f64, alpha: f64) -> Self {
        Self {
            threshold,
            alpha,
            cumulative_sum: F::zero(),
            min_cumulative_sum: F::zero(),
            status: DriftStatus::Stable,
            samples_count: 0,
            drift_count: 0,
            warning_count: 0,
        }
    }
}

impl<F: Float + std::fmt::Debug + Send + Sync + std::iter::Sum> ConceptDriftDetector<F>
    for PageHinkleyDetector<F>
{
    fn update(&mut self, prediction_correct: bool, _error: F) -> Result<DriftDetectionResult> {
        self.samples_count += 1;

        let x = if prediction_correct {
            F::zero()
        } else {
            F::one()
        };
        let mu = F::from(self.alpha).unwrap();

        self.cumulative_sum = self.cumulative_sum + x - mu;

        if self.cumulative_sum < self.min_cumulative_sum {
            self.min_cumulative_sum = self.cumulative_sum;
        }

        let ph_value = self.cumulative_sum - self.min_cumulative_sum;
        let threshold = F::from(self.threshold).unwrap();

        self.status = if ph_value > threshold {
            self.drift_count += 1;
            DriftStatus::Drift
        } else {
            DriftStatus::Stable
        };

        Ok(DriftDetectionResult {
            status: self.status.clone(),
            confidence: 0.7,
            change_point: if self.status == DriftStatus::Drift {
                Some(self.samples_count)
            } else {
                None
            },
            statistics: HashMap::new(),
        })
    }

    fn get_status(&self) -> DriftStatus {
        self.status.clone()
    }

    fn reset(&mut self) {
        self.cumulative_sum = F::zero();
        self.min_cumulative_sum = F::zero();
        self.status = DriftStatus::Stable;
        self.samples_count = 0;
    }

    fn get_config(&self) -> HashMap<String, f64> {
        let mut config = HashMap::new();
        config.insert("threshold".to_string(), self.threshold);
        config.insert("alpha".to_string(), self.alpha);
        config
    }

    fn get_statistics(&self) -> DriftStatistics<F> {
        DriftStatistics {
            samples_since_reset: self.samples_count,
            warnings_count: self.warning_count,
            drifts_count: self.drift_count,
            current_error_rate: F::zero(),
            baseline_error_rate: F::zero(),
            drift_score: self.cumulative_sum - self.min_cumulative_sum,
            last_detection_time: None,
        }
    }
}
