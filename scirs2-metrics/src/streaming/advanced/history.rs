//! History buffer for streaming data
//!
//! This module provides efficient buffering of historical data points
//! for streaming metrics analysis.

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use super::core::DataPoint;
use num_traits::Float;
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

/// History buffer for storing past data
#[derive(Debug, Clone)]
pub struct HistoryBuffer<F: Float + std::fmt::Debug> {
    max_size: usize,
    data: VecDeque<DataPoint<F>>,
    timestamps: VecDeque<Instant>,
    metadata: VecDeque<HashMap<String, String>>,
}

impl<F: Float + std::fmt::Debug + Send + Sync + std::iter::Sum> HistoryBuffer<F> {
    pub fn new(max_size: usize) -> Self {
        Self {
            max_size,
            data: VecDeque::with_capacity(max_size),
            timestamps: VecDeque::with_capacity(max_size),
            metadata: VecDeque::with_capacity(max_size),
        }
    }

    pub fn add_data_point(&mut self, data_point: DataPoint<F>) {
        self.add_data_point_with_metadata(data_point, HashMap::new());
    }

    pub fn add_data_point_with_metadata(
        &mut self,
        data_point: DataPoint<F>,
        metadata: HashMap<String, String>,
    ) {
        if self.data.len() >= self.max_size {
            self.data.pop_front();
            self.timestamps.pop_front();
            self.metadata.pop_front();
        }

        self.data.push_back(data_point);
        self.timestamps.push_back(Instant::now());
        self.metadata.push_back(metadata);
    }

    pub fn get_recent_data(&self, n: usize) -> Vec<&DataPoint<F>> {
        self.data.iter().rev().take(n).collect()
    }

    pub fn get_data_in_range(&self, start: Instant, end: Instant) -> Vec<&DataPoint<F>> {
        self.data
            .iter()
            .zip(self.timestamps.iter())
            .filter(|(_, &timestamp)| timestamp >= start && timestamp <= end)
            .map(|(data, _)| data)
            .collect()
    }

    pub fn get_error_history(&self, n: usize) -> Vec<F> {
        self.data.iter().rev().take(n).map(|dp| dp.error).collect()
    }

    pub fn get_prediction_accuracy_history(&self, n: usize) -> Vec<bool> {
        self.data
            .iter()
            .rev()
            .take(n)
            .map(|dp| (dp.true_value - dp.predicted_value).abs() < F::from(0.001).unwrap())
            .collect()
    }

    pub fn calculate_moving_average(&self, window: usize) -> Option<F> {
        if self.data.len() < window {
            return None;
        }

        let recent_errors: Vec<F> = self
            .data
            .iter()
            .rev()
            .take(window)
            .map(|dp| dp.error)
            .collect();

        let sum = recent_errors.iter().cloned().sum::<F>();
        Some(sum / F::from(window).unwrap())
    }

    pub fn calculate_moving_variance(&self, window: usize) -> Option<F> {
        if self.data.len() < window {
            return None;
        }

        let recent_errors: Vec<F> = self
            .data
            .iter()
            .rev()
            .take(window)
            .map(|dp| dp.error)
            .collect();

        let mean = recent_errors.iter().cloned().sum::<F>() / F::from(window).unwrap();
        let variance = recent_errors
            .iter()
            .map(|&x| (x - mean) * (x - mean))
            .sum::<F>()
            / F::from(window - 1).unwrap();

        Some(variance)
    }

    pub fn get_confidence_history(&self, n: usize) -> Vec<F> {
        self.data
            .iter()
            .rev()
            .take(n)
            .map(|dp| dp.confidence)
            .collect()
    }

    pub fn find_outliers(&self, threshold: F) -> Vec<(usize, &DataPoint<F>)> {
        if self.data.len() < 10 {
            return Vec::new();
        }

        let errors: Vec<F> = self.data.iter().map(|dp| dp.error).collect();
        let mean = errors.iter().cloned().sum::<F>() / F::from(errors.len()).unwrap();
        let variance = errors.iter().map(|&x| (x - mean) * (x - mean)).sum::<F>()
            / F::from(errors.len() - 1).unwrap();
        let std_dev = variance.sqrt();

        self.data
            .iter()
            .enumerate()
            .filter(|(_, dp)| (dp.error - mean).abs() > threshold * std_dev)
            .collect()
    }

    pub fn clear(&mut self) {
        self.data.clear();
        self.timestamps.clear();
        self.metadata.clear();
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn get_capacity(&self) -> usize {
        self.max_size
    }

    pub fn get_memory_usage(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.data.capacity() * std::mem::size_of::<DataPoint<F>>()
            + self.timestamps.capacity() * std::mem::size_of::<Instant>()
            + self.metadata.capacity() * std::mem::size_of::<HashMap<String, String>>()
    }
}
