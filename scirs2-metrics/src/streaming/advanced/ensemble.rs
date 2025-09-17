//! Ensemble metrics for streaming data
//!
//! This module provides ensemble-based metric aggregation for combining
//! multiple streaming metrics for improved robustness.

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use super::core::{EnsembleAggregation, StreamingMetric};
use crate::error::Result;
use num_traits::Float;
use std::collections::HashMap;

/// Ensemble of different metrics
pub struct MetricEnsemble<F: Float + std::fmt::Debug> {
    base_metrics: HashMap<String, Box<dyn StreamingMetric<F> + Send + Sync>>,
    weights: HashMap<String, F>,
    aggregation_strategy: EnsembleAggregation,
    consensus_threshold: F,
}

impl<F: Float + std::fmt::Debug> std::fmt::Debug for MetricEnsemble<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetricEnsemble")
            .field(
                "base_metrics",
                &format!("{} metrics", self.base_metrics.len()),
            )
            .field("weights", &self.weights)
            .field("aggregation_strategy", &self.aggregation_strategy)
            .field("consensus_threshold", &self.consensus_threshold)
            .finish()
    }
}

impl<F: Float + std::fmt::Debug + Send + Sync + std::iter::Sum> MetricEnsemble<F> {
    pub fn new() -> Self {
        Self {
            base_metrics: HashMap::new(),
            weights: HashMap::new(),
            aggregation_strategy: EnsembleAggregation::WeightedAverage,
            consensus_threshold: F::from(0.7).unwrap(),
        }
    }

    pub fn add_metric(
        &mut self,
        name: String,
        metric: Box<dyn StreamingMetric<F> + Send + Sync>,
        weight: F,
    ) {
        self.base_metrics.insert(name.clone(), metric);
        self.weights.insert(name, weight);
    }

    pub fn update(&mut self, true_value: F, predicted_value: F) -> Result<()> {
        for metric in self.base_metrics.values_mut() {
            metric.update(true_value, predicted_value)?;
        }
        Ok(())
    }

    pub fn get_ensemble_value(&self) -> F {
        match &self.aggregation_strategy {
            EnsembleAggregation::WeightedAverage => self.get_weighted_average(),
            EnsembleAggregation::Majority => self.get_majority_value(),
            EnsembleAggregation::Maximum => self.get_maximum_value(),
            EnsembleAggregation::Minimum => self.get_minimum_value(),
            EnsembleAggregation::Median => self.get_median_value(),
            EnsembleAggregation::Stacking { .. } => self.get_stacking_value(),
        }
    }

    fn get_weighted_average(&self) -> F {
        let mut weighted_sum = F::zero();
        let mut total_weight = F::zero();

        for (name, metric) in &self.base_metrics {
            if let Some(&weight) = self.weights.get(name) {
                weighted_sum = weighted_sum + metric.get_value() * weight;
                total_weight = total_weight + weight;
            }
        }

        if total_weight > F::zero() {
            weighted_sum / total_weight
        } else {
            F::zero()
        }
    }

    fn get_majority_value(&self) -> F {
        // Simple majority voting - return most common value
        let values: Vec<F> = self.base_metrics.values().map(|m| m.get_value()).collect();
        if values.is_empty() {
            return F::zero();
        }

        // For continuous values, use median as proxy for majority
        self.get_median_value()
    }

    fn get_maximum_value(&self) -> F {
        self.base_metrics
            .values()
            .map(|m| m.get_value())
            .fold(F::neg_infinity(), F::max)
    }

    fn get_minimum_value(&self) -> F {
        self.base_metrics
            .values()
            .map(|m| m.get_value())
            .fold(F::infinity(), F::min)
    }

    fn get_median_value(&self) -> F {
        let mut values: Vec<F> = self.base_metrics.values().map(|m| m.get_value()).collect();
        if values.is_empty() {
            return F::zero();
        }

        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let len = values.len();

        if len % 2 == 0 {
            (values[len / 2 - 1] + values[len / 2]) / F::from(2.0).unwrap()
        } else {
            values[len / 2]
        }
    }

    fn get_stacking_value(&self) -> F {
        // Simplified stacking - return weighted average for now
        self.get_weighted_average()
    }

    pub fn get_consensus(&self) -> F {
        // Calculate consensus level among ensemble members
        let values: Vec<F> = self.base_metrics.values().map(|m| m.get_value()).collect();
        if values.len() < 2 {
            return F::one();
        }

        let mean = values.iter().cloned().sum::<F>() / F::from(values.len()).unwrap();
        let variance = values.iter().map(|&x| (x - mean) * (x - mean)).sum::<F>()
            / F::from(values.len()).unwrap();

        // Higher consensus when variance is low
        F::one() / (F::one() + variance)
    }

    pub fn set_aggregation_strategy(&mut self, strategy: EnsembleAggregation) {
        self.aggregation_strategy = strategy;
    }

    pub fn set_consensus_threshold(&mut self, threshold: F) {
        self.consensus_threshold = threshold;
    }

    pub fn reset(&mut self) {
        for metric in self.base_metrics.values_mut() {
            metric.reset();
        }
    }

    pub fn get_metric_names(&self) -> Vec<String> {
        self.base_metrics.keys().cloned().collect()
    }

    pub fn get_individual_values(&self) -> HashMap<String, F> {
        self.base_metrics
            .iter()
            .map(|(name, metric)| (name.clone(), metric.get_value()))
            .collect()
    }
}
