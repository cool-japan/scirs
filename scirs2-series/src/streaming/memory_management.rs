//! Memory management and multi-series utilities for streaming analysis
//!
//! This module provides memory-efficient data structures, anomaly detection,
//! pattern matching, and utilities for managing multiple time series streams.

use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::{Float, FromPrimitive};
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;

use super::config::StreamConfig;
use super::statistics::OnlineStats;
use crate::error::{Result, TimeSeriesError};
use crate::streaming::StreamingAnalyzer;

/// Multi-series streaming analyzer for handling multiple time series simultaneously
#[derive(Debug)]
pub struct MultiSeriesAnalyzer<F: Float + Debug> {
    analyzers: HashMap<String, StreamingAnalyzer<F>>,
    config: StreamConfig,
}

impl<F: Float + Debug + Clone + FromPrimitive> MultiSeriesAnalyzer<F> {
    /// Create new multi-series analyzer
    pub fn new(config: StreamConfig) -> Self {
        Self {
            analyzers: HashMap::new(),
            config,
        }
    }

    /// Add new time series to track
    pub fn add_series(&mut self, seriesid: String) -> Result<()> {
        let analyzer = StreamingAnalyzer::new(self.config.clone())?;
        self.analyzers.insert(seriesid, analyzer);
        Ok(())
    }

    /// Add observation to specific series
    pub fn add_observation(&mut self, seriesid: &str, value: F) -> Result<()> {
        if let Some(analyzer) = self.analyzers.get_mut(seriesid) {
            analyzer.add_observation(value)
        } else {
            Err(TimeSeriesError::InvalidInput(format!(
                "Series '{seriesid}' not found"
            )))
        }
    }

    /// Get analyzer for specific series
    pub fn get_analyzer(&self, seriesid: &str) -> Option<&StreamingAnalyzer<F>> {
        self.analyzers.get(seriesid)
    }

    /// Get mutable analyzer for specific series
    pub fn get_analyzer_mut(&mut self, seriesid: &str) -> Option<&mut StreamingAnalyzer<F>> {
        self.analyzers.get_mut(seriesid)
    }

    /// Get all series IDs
    pub fn get_series_ids(&self) -> Vec<String> {
        self.analyzers.keys().cloned().collect()
    }

    /// Remove series
    pub fn remove_series(&mut self, seriesid: &str) -> bool {
        self.analyzers.remove(seriesid).is_some()
    }

    /// Get cross-series correlation (simplified)
    pub fn get_correlation(&self, series1: &str, series2: &str) -> Result<F> {
        let analyzer1 = self.analyzers.get(series1).ok_or_else(|| {
            TimeSeriesError::InvalidInput(format!("Series '{series1}' not found"))
        })?;

        let analyzer2 = self.analyzers.get(series2).ok_or_else(|| {
            TimeSeriesError::InvalidInput(format!("Series '{series2}' not found"))
        })?;

        let buffer1 = analyzer1.get_buffer();
        let buffer2 = analyzer2.get_buffer();

        let min_len = std::cmp::min(buffer1.len(), buffer2.len());
        if min_len < 2 {
            return Ok(F::zero());
        }

        // Calculate Pearson correlation
        let mean1 = buffer1
            .iter()
            .take(min_len)
            .cloned()
            .fold(F::zero(), |acc, x| acc + x)
            / F::from(min_len).unwrap();
        let mean2 = buffer2
            .iter()
            .take(min_len)
            .cloned()
            .fold(F::zero(), |acc, x| acc + x)
            / F::from(min_len).unwrap();

        let mut numerator = F::zero();
        let mut sum1_sq = F::zero();
        let mut sum2_sq = F::zero();

        for i in 0..min_len {
            let diff1 = buffer1[i] - mean1;
            let diff2 = buffer2[i] - mean2;
            numerator = numerator + diff1 * diff2;
            sum1_sq = sum1_sq + diff1 * diff1;
            sum2_sq = sum2_sq + diff2 * diff2;
        }

        let denominator = (sum1_sq * sum2_sq).sqrt();
        if denominator > F::epsilon() {
            Ok(numerator / denominator)
        } else {
            Ok(F::zero())
        }
    }
}

/// Online anomaly detection using Isolation Forest-like approach
#[derive(Debug)]
pub struct StreamingAnomalyDetector<F: Float + Debug> {
    /// Recent feature vectors for comparison
    feature_buffer: VecDeque<Vec<F>>,
    /// Maximum buffer size
    max_buffer_size: usize,
    /// Anomaly threshold
    threshold: F,
    /// Feature extractors
    window_size: usize,
    /// Number of features to extract
    num_features: usize,
}

impl<F: Float + Debug + Clone> StreamingAnomalyDetector<F> {
    /// Create new anomaly detector
    pub fn new(
        max_buffer_size: usize,
        threshold: F,
        window_size: usize,
        num_features: usize,
    ) -> Self {
        Self {
            feature_buffer: VecDeque::with_capacity(max_buffer_size),
            max_buffer_size,
            threshold,
            window_size,
            num_features,
        }
    }

    /// Extract features from a time series window
    fn extract_features(&self, window: &[F]) -> Vec<F> {
        if window.is_empty() {
            return vec![F::zero(); self.num_features];
        }

        let mut features = Vec::with_capacity(self.num_features);
        let n = F::from(window.len()).unwrap();

        // Feature 1: Mean
        let mean = window.iter().fold(F::zero(), |acc, &x| acc + x) / n;
        features.push(mean);

        // Feature 2: Standard deviation
        let variance = window
            .iter()
            .map(|&x| (x - mean) * (x - mean))
            .fold(F::zero(), |acc, x| acc + x)
            / n;
        features.push(variance.sqrt());

        // Feature 3: Skewness (simplified)
        let skewness = window
            .iter()
            .map(|&x| {
                let normalized = (x - mean) / variance.sqrt();
                normalized * normalized * normalized
            })
            .fold(F::zero(), |acc, x| acc + x)
            / n;
        features.push(skewness);

        // Feature 4: Range
        let min_val = window.iter().fold(F::infinity(), |acc, &x| acc.min(x));
        let max_val = window.iter().fold(F::neg_infinity(), |acc, &x| acc.max(x));
        features.push(max_val - min_val);

        // Feature 5: Trend (slope of linear regression)
        if window.len() > 1 {
            let x_mean = F::from(window.len() - 1).unwrap() / F::from(2).unwrap();
            let mut num = F::zero();
            let mut den = F::zero();

            for (i, &y) in window.iter().enumerate() {
                let x = F::from(i).unwrap();
                num = num + (x - x_mean) * (y - mean);
                den = den + (x - x_mean) * (x - x_mean);
            }

            let slope = if den > F::zero() {
                num / den
            } else {
                F::zero()
            };
            features.push(slope);
        } else {
            features.push(F::zero());
        }

        features
    }

    /// Update detector with new window and check for anomalies
    pub fn update(&mut self, window: &[F]) -> Result<bool> {
        if window.len() < self.window_size {
            return Ok(false); // Not enough data
        }

        let features = self.extract_features(&window[window.len() - self.window_size..]);

        if self.feature_buffer.is_empty() {
            // First observation - just store
            if self.feature_buffer.len() >= self.max_buffer_size {
                self.feature_buffer.pop_front();
            }
            self.feature_buffer.push_back(features);
            return Ok(false);
        }

        // Calculate isolation score (simplified)
        let mut min_distance = F::infinity();
        for stored_features in &self.feature_buffer {
            let distance = features
                .iter()
                .zip(stored_features.iter())
                .map(|(&a, &b)| (a - b) * (a - b))
                .fold(F::zero(), |acc, x| acc + x)
                .sqrt();
            min_distance = min_distance.min(distance);
        }

        // Add current features to buffer
        if self.feature_buffer.len() >= self.max_buffer_size {
            self.feature_buffer.pop_front();
        }
        self.feature_buffer.push_back(features);

        // Check if anomaly (isolated point)
        Ok(min_distance > self.threshold)
    }

    /// Update threshold based on recent observations
    pub fn adapt_threshold(&mut self, factor: F) {
        if self.feature_buffer.len() > 2 {
            // Calculate average distance between recent features
            let mut total_distance = F::zero();
            let mut count = 0;

            for i in 0..self.feature_buffer.len() {
                for j in i + 1..self.feature_buffer.len() {
                    let distance = self.feature_buffer[i]
                        .iter()
                        .zip(self.feature_buffer[j].iter())
                        .map(|(&a, &b)| (a - b) * (a - b))
                        .fold(F::zero(), |acc, x| acc + x)
                        .sqrt();
                    total_distance = total_distance + distance;
                    count += 1;
                }
            }

            if count > 0 {
                let avg_distance = total_distance / F::from(count).unwrap();
                self.threshold = avg_distance * factor;
            }
        }
    }
}

/// Online pattern matching for streaming time series
#[derive(Debug)]
pub struct StreamingPatternMatcher<F: Float + Debug> {
    /// Template patterns to match against
    patterns: Vec<Vec<F>>,
    /// Pattern names
    pattern_names: Vec<String>,
    /// Recent data buffer for pattern matching
    buffer: VecDeque<F>,
    /// Maximum buffer size
    max_buffer_size: usize,
    /// Matching threshold (normalized correlation)
    threshold: F,
}

impl<F: Float + Debug + Clone> StreamingPatternMatcher<F> {
    /// Create new pattern matcher
    pub fn new(_max_buffersize: usize, threshold: F) -> Self {
        Self {
            patterns: Vec::new(),
            pattern_names: Vec::new(),
            buffer: VecDeque::with_capacity(_max_buffersize),
            max_buffer_size: _max_buffersize,
            threshold,
        }
    }

    /// Add a pattern to match against
    pub fn add_pattern(&mut self, pattern: Vec<F>, name: String) -> Result<()> {
        if pattern.is_empty() {
            return Err(TimeSeriesError::InvalidInput(
                "Pattern cannot be empty".to_string(),
            ));
        }
        self.patterns.push(pattern);
        self.pattern_names.push(name);
        Ok(())
    }

    /// Update buffer and check for pattern matches
    pub fn update(&mut self, value: F) -> Vec<PatternMatch> {
        // Add to buffer
        if self.buffer.len() >= self.max_buffer_size {
            self.buffer.pop_front();
        }
        self.buffer.push_back(value);

        let mut matches = Vec::new();

        // Check each pattern
        for (i, pattern) in self.patterns.iter().enumerate() {
            if self.buffer.len() >= pattern.len() {
                let recent_data: Vec<F> = self
                    .buffer
                    .iter()
                    .rev()
                    .take(pattern.len())
                    .rev()
                    .cloned()
                    .collect();

                if let Ok(correlation) = self.normalized_correlation(&recent_data, pattern) {
                    if correlation >= self.threshold {
                        matches.push(PatternMatch {
                            pattern_name: self.pattern_names[i].clone(),
                            correlation: correlation.to_f64().unwrap(),
                            start_index: self.buffer.len() - pattern.len(),
                            pattern_length: pattern.len(),
                        });
                    }
                }
            }
        }

        matches
    }

    /// Calculate normalized correlation between two sequences
    fn normalized_correlation(&self, a: &[F], b: &[F]) -> Result<F> {
        if a.len() != b.len() || a.is_empty() {
            return Err(TimeSeriesError::InvalidInput(
                "Sequences must have the same non-zero length".to_string(),
            ));
        }

        let n = F::from(a.len()).unwrap();

        // Calculate means
        let mean_a = a.iter().fold(F::zero(), |acc, &x| acc + x) / n;
        let mean_b = b.iter().fold(F::zero(), |acc, &x| acc + x) / n;

        // Calculate correlation components
        let mut num = F::zero();
        let mut den_a = F::zero();
        let mut den_b = F::zero();

        for (&val_a, &val_b) in a.iter().zip(b.iter()) {
            let diff_a = val_a - mean_a;
            let diff_b = val_b - mean_b;

            num = num + diff_a * diff_b;
            den_a = den_a + diff_a * diff_a;
            den_b = den_b + diff_b * diff_b;
        }

        let denominator = (den_a * den_b).sqrt();
        if denominator > F::zero() {
            Ok(num / denominator)
        } else {
            Ok(F::zero())
        }
    }
}

/// Pattern match result
#[derive(Debug, Clone)]
pub struct PatternMatch {
    /// Name of the matched pattern
    pub pattern_name: String,
    /// Correlation coefficient with the pattern
    pub correlation: f64,
    /// Starting index in the time series
    pub start_index: usize,
    /// Length of the matched pattern
    pub pattern_length: usize,
}

/// Memory-efficient circular buffer for streaming data
#[derive(Debug)]
pub struct CircularBuffer<F: Float> {
    /// Internal buffer
    buffer: Vec<F>,
    /// Current write position
    position: usize,
    /// Maximum capacity
    capacity: usize,
    /// Whether buffer is full
    is_full: bool,
}

impl<F: Float + Debug + Clone + Default> CircularBuffer<F> {
    /// Create new circular buffer
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: vec![F::default(); capacity],
            position: 0,
            capacity,
            is_full: false,
        }
    }

    /// Add new value to buffer
    pub fn push(&mut self, value: F) {
        self.buffer[self.position] = value;
        self.position = (self.position + 1) % self.capacity;

        if self.position == 0 {
            self.is_full = true;
        }
    }

    /// Get current size of buffer
    pub fn len(&self) -> usize {
        if self.is_full {
            self.capacity
        } else {
            self.position
        }
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        !self.is_full && self.position == 0
    }

    /// Get slice of recent n values
    pub fn recent(&self, n: usize) -> Vec<F> {
        let available = self.len();
        let take = n.min(available);
        let mut result = Vec::with_capacity(take);

        if self.is_full {
            // Buffer is full, need to handle wrap-around
            let start_pos = (self.position + self.capacity - take) % self.capacity;

            if start_pos + take <= self.capacity {
                // No wrap-around needed
                result.extend_from_slice(&self.buffer[start_pos..start_pos + take]);
            } else {
                // Need to handle wrap-around
                let first_part = self.capacity - start_pos;
                result.extend_from_slice(&self.buffer[start_pos..]);
                result.extend_from_slice(&self.buffer[..take - first_part]);
            }
        } else {
            // Buffer not full, simple case
            let start = self.position.saturating_sub(take);
            result.extend_from_slice(&self.buffer[start..self.position]);
        }

        result
    }

    /// Get all values in chronological order
    pub fn to_vec(&self) -> Vec<F> {
        self.recent(self.len())
    }

    /// Calculate statistics over recent window
    pub fn window_stats(&self, windowsize: usize) -> OnlineStats<F> {
        let recent_data = self.recent(windowsize);
        let mut stats = OnlineStats::new();

        for value in recent_data {
            stats.update(value);
        }

        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anomaly_detector() {
        let mut detector = StreamingAnomalyDetector::new(100, 2.0, 10, 5);

        // Add normal data
        let normal_data: Vec<f64> = (0..20).map(|x| x as f64).collect();

        for window in normal_data.windows(10) {
            let is_anomaly = detector.update(window).unwrap();
            assert!(!is_anomaly, "Normal data should not be anomalous");
        }

        // Add anomalous data
        let mut anomalous_data = normal_data.clone();
        anomalous_data.extend(vec![1000.0; 10]); // Clear anomaly

        let result = detector
            .update(&anomalous_data[anomalous_data.len() - 10..])
            .unwrap();
        assert!(result, "Clear anomaly should be detected");
    }

    #[test]
    fn test_pattern_matcher() {
        let mut matcher = StreamingPatternMatcher::new(100, 0.8);

        // Add a simple pattern
        let pattern = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        matcher
            .add_pattern(pattern.clone(), "triangle".to_string())
            .unwrap();

        // Add matching data
        for &value in &pattern {
            let matches = matcher.update(value);
            if !matches.is_empty() {
                assert_eq!(matches[0].pattern_name, "triangle");
                assert!(matches[0].correlation >= 0.8);
            }
        }
    }

    #[test]
    fn test_circular_buffer() {
        let mut buffer = CircularBuffer::new(5);

        // Add data
        for i in 1..=3 {
            buffer.push(i as f64);
        }

        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.recent(2), vec![2.0, 3.0]);

        // Fill buffer completely
        for i in 4..=7 {
            buffer.push(i as f64);
        }

        assert_eq!(buffer.len(), 5);
        assert_eq!(buffer.to_vec(), vec![3.0, 4.0, 5.0, 6.0, 7.0]);
    }
}
