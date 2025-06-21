//! Automated model selection based on multiple metrics
//!
//! This module provides utilities for automatically selecting the best model
//! from a set of candidates based on multiple evaluation metrics.
//!
//! # Features
//!
//! - **Multi-metric evaluation**: Combine multiple metrics with custom weights
//! - **Flexible scoring**: Support different aggregation strategies
//! - **Pareto optimal selection**: Find models that are not dominated by others
//! - **Cross-validation integration**: Work with CV results for robust selection
//! - **Custom criteria**: Define custom selection criteria
//!
//! # Examples
//!
//! ## Basic Model Selection
//!
//! ```
//! use scirs2_metrics::selection::{ModelSelector, SelectionCriteria};
//! use std::collections::HashMap;
//!
//! // Define models and their metric scores
//! let mut model_scores = HashMap::new();
//! model_scores.insert("model_a".to_string(), vec![("accuracy", 0.85), ("precision", 0.82)]);
//! model_scores.insert("model_b".to_string(), vec![("accuracy", 0.80), ("precision", 0.90)]);
//! model_scores.insert("model_c".to_string(), vec![("accuracy", 0.88), ("precision", 0.85)]);
//!
//! // Create selector with weighted criteria
//! let mut selector = ModelSelector::new();
//! selector.add_metric("accuracy", 0.6, true)  // 60% weight, higher is better
//!         .add_metric("precision", 0.4, true); // 40% weight, higher is better
//!
//! // Select best model
//! let best_model = selector.select_best(&model_scores).unwrap();
//! println!("Best model: {}", best_model);
//! ```
//!
//! ## Pareto Optimal Selection
//!
//! ```
//! use scirs2_metrics::selection::ModelSelector;
//! use std::collections::HashMap;
//!
//! let mut model_scores = HashMap::new();
//! model_scores.insert("model_a".to_string(), vec![("accuracy", 0.85), ("speed", 100.0)]);
//! model_scores.insert("model_b".to_string(), vec![("accuracy", 0.80), ("speed", 200.0)]);
//! model_scores.insert("model_c".to_string(), vec![("accuracy", 0.90), ("speed", 50.0)]);
//!
//! let mut selector = ModelSelector::new();
//! selector
//!     .add_metric("accuracy", 1.0, true)   // higher is better
//!     .add_metric("speed", 1.0, true);     // higher is better (faster inference)
//!
//! let pareto_optimal = selector.find_pareto_optimal(&model_scores);
//! println!("Pareto optimal models: {:?}", pareto_optimal);
//! ```

use crate::error::{MetricsError, Result};
use std::collections::HashMap;
use std::fmt;

/// Represents a metric with its weight and optimization direction
#[derive(Debug, Clone)]
pub struct MetricCriterion {
    /// Name of the metric
    pub name: String,
    /// Weight of the metric in the final score (0.0 to 1.0)
    pub weight: f64,
    /// Whether higher values are better
    pub higher_is_better: bool,
}

/// Aggregation strategies for combining multiple metrics
#[derive(Debug, Clone, Copy)]
pub enum AggregationStrategy {
    /// Weighted sum of normalized scores
    WeightedSum,
    /// Weighted geometric mean
    WeightedGeometricMean,
    /// Weighted harmonic mean
    WeightedHarmonicMean,
    /// Minimum score across all metrics (conservative)
    MinScore,
    /// Maximum score across all metrics (optimistic)
    MaxScore,
}

/// Model selection criteria configuration
#[derive(Debug, Clone)]
pub struct SelectionCriteria {
    /// List of metrics to consider
    pub metrics: Vec<MetricCriterion>,
    /// Strategy for aggregating metric scores
    pub aggregation: AggregationStrategy,
    /// Minimum threshold that must be met for each metric
    pub thresholds: HashMap<String, f64>,
}

impl Default for SelectionCriteria {
    fn default() -> Self {
        Self {
            metrics: Vec::new(),
            aggregation: AggregationStrategy::WeightedSum,
            thresholds: HashMap::new(),
        }
    }
}

/// Main model selector that evaluates and ranks models
pub struct ModelSelector {
    criteria: SelectionCriteria,
}

impl Default for ModelSelector {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelSelector {
    /// Creates a new model selector
    pub fn new() -> Self {
        Self {
            criteria: SelectionCriteria::default(),
        }
    }

    /// Adds a metric to the selection criteria
    pub fn add_metric(&mut self, name: &str, weight: f64, higher_is_better: bool) -> &mut Self {
        self.criteria.metrics.push(MetricCriterion {
            name: name.to_string(),
            weight,
            higher_is_better,
        });
        self
    }

    /// Sets the aggregation strategy
    pub fn with_aggregation(&mut self, strategy: AggregationStrategy) -> &mut Self {
        self.criteria.aggregation = strategy;
        self
    }

    /// Adds a threshold for a specific metric
    pub fn add_threshold(&mut self, metric_name: &str, threshold: f64) -> &mut Self {
        self.criteria
            .thresholds
            .insert(metric_name.to_string(), threshold);
        self
    }

    /// Selects the best model from a set of candidates
    pub fn select_best(&self, model_scores: &HashMap<String, Vec<(&str, f64)>>) -> Result<String> {
        if model_scores.is_empty() {
            return Err(MetricsError::InvalidInput("No models provided".to_string()));
        }

        let rankings = self.rank_models(model_scores)?;

        rankings
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(model_name, _)| model_name)
            .ok_or_else(|| MetricsError::ComputationError("No valid models found".to_string()))
    }

    /// Ranks all models and returns them sorted by score (descending)
    pub fn rank_models(
        &self,
        model_scores: &HashMap<String, Vec<(&str, f64)>>,
    ) -> Result<Vec<(String, f64)>> {
        let mut rankings = Vec::new();

        for (model_name, scores) in model_scores {
            if let Ok(aggregated_score) = self.compute_aggregated_score(scores) {
                if self.meets_thresholds(scores) {
                    rankings.push((model_name.clone(), aggregated_score));
                }
            }
        }

        rankings.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(rankings)
    }

    /// Finds Pareto optimal models (not dominated by any other model)
    pub fn find_pareto_optimal(
        &self,
        model_scores: &HashMap<String, Vec<(&str, f64)>>,
    ) -> Vec<String> {
        let mut pareto_optimal = Vec::new();

        for (model_name, scores) in model_scores {
            let mut is_dominated = false;

            for (other_name, other_scores) in model_scores {
                if model_name == other_name {
                    continue;
                }

                if self.dominates(other_scores, scores) {
                    is_dominated = true;
                    break;
                }
            }

            if !is_dominated {
                pareto_optimal.push(model_name.clone());
            }
        }

        pareto_optimal
    }

    /// Computes the aggregated score for a model based on the selection criteria
    fn compute_aggregated_score(&self, scores: &[(&str, f64)]) -> Result<f64> {
        let score_map: HashMap<&str, f64> = scores.iter().cloned().collect();

        // Normalize scores for each metric
        let mut normalized_scores = Vec::new();
        let mut total_weight = 0.0;

        for criterion in &self.criteria.metrics {
            if let Some(&score) = score_map.get(criterion.name.as_str()) {
                let normalized = if criterion.higher_is_better {
                    score
                } else {
                    -score // Flip for minimization metrics
                };

                normalized_scores.push((normalized, criterion.weight));
                total_weight += criterion.weight;
            }
        }

        if normalized_scores.is_empty() {
            return Err(MetricsError::InvalidInput(
                "No matching metrics found".to_string(),
            ));
        }

        // Normalize weights
        for (_, weight) in &mut normalized_scores {
            *weight /= total_weight;
        }

        // Apply aggregation strategy
        let aggregated = match self.criteria.aggregation {
            AggregationStrategy::WeightedSum => normalized_scores
                .iter()
                .map(|(score, weight)| score * weight)
                .sum(),
            AggregationStrategy::WeightedGeometricMean => {
                let product: f64 = normalized_scores
                    .iter()
                    .map(|(score, weight)| score.abs().powf(*weight))
                    .product();
                product
            }
            AggregationStrategy::WeightedHarmonicMean => {
                let weighted_reciprocal_sum: f64 = normalized_scores
                    .iter()
                    .map(|(score, weight)| weight / score.abs())
                    .sum();
                total_weight / weighted_reciprocal_sum
            }
            AggregationStrategy::MinScore => normalized_scores
                .iter()
                .map(|(score, _)| *score)
                .fold(f64::INFINITY, f64::min),
            AggregationStrategy::MaxScore => normalized_scores
                .iter()
                .map(|(score, _)| *score)
                .fold(f64::NEG_INFINITY, f64::max),
        };

        Ok(aggregated)
    }

    /// Checks if a model meets all threshold requirements
    fn meets_thresholds(&self, scores: &[(&str, f64)]) -> bool {
        let score_map: HashMap<&str, f64> = scores.iter().cloned().collect();

        for (metric_name, threshold) in &self.criteria.thresholds {
            if let Some(&score) = score_map.get(metric_name.as_str()) {
                // Find the metric criterion to check optimization direction
                if let Some(criterion) = self
                    .criteria
                    .metrics
                    .iter()
                    .find(|c| c.name == *metric_name)
                {
                    let meets_threshold = if criterion.higher_is_better {
                        score >= *threshold
                    } else {
                        score <= *threshold
                    };

                    if !meets_threshold {
                        return false;
                    }
                }
            } else {
                // Metric not found, consider as not meeting threshold
                return false;
            }
        }

        true
    }

    /// Checks if model A dominates model B (Pareto dominance)
    fn dominates(&self, scores_a: &[(&str, f64)], scores_b: &[(&str, f64)]) -> bool {
        let map_a: HashMap<&str, f64> = scores_a.iter().cloned().collect();
        let map_b: HashMap<&str, f64> = scores_b.iter().cloned().collect();

        let mut at_least_one_better = false;

        for criterion in &self.criteria.metrics {
            let metric_name = criterion.name.as_str();

            if let (Some(&score_a), Some(&score_b)) =
                (map_a.get(metric_name), map_b.get(metric_name))
            {
                let a_better_than_b = if criterion.higher_is_better {
                    score_a > score_b
                } else {
                    score_a < score_b
                };

                let a_worse_than_b = if criterion.higher_is_better {
                    score_a < score_b
                } else {
                    score_a > score_b
                };

                if a_worse_than_b {
                    return false; // A is worse in at least one metric
                }

                if a_better_than_b {
                    at_least_one_better = true;
                }
            }
        }

        at_least_one_better
    }
}

/// Represents the result of model selection with detailed information
#[derive(Debug, Clone)]
pub struct SelectionResult {
    /// Name of the selected model
    pub selected_model: String,
    /// Final aggregated score
    pub score: f64,
    /// All models ranked by score
    pub rankings: Vec<(String, f64)>,
    /// Pareto optimal models
    pub pareto_optimal: Vec<String>,
}

impl fmt::Display for SelectionResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Model Selection Results")?;
        writeln!(f, "======================")?;
        writeln!(
            f,
            "Selected Model: {} (Score: {:.4})",
            self.selected_model, self.score
        )?;
        writeln!(f)?;

        writeln!(f, "Complete Rankings:")?;
        writeln!(f, "------------------")?;
        for (i, (model, score)) in self.rankings.iter().enumerate() {
            writeln!(f, "{}: {} ({:.4})", i + 1, model, score)?;
        }

        writeln!(f)?;
        writeln!(f, "Pareto Optimal Models: {:?}", self.pareto_optimal)?;

        Ok(())
    }
}

/// Builder for creating complex model selection scenarios
pub struct ModelSelectionBuilder {
    selector: ModelSelector,
}

impl ModelSelectionBuilder {
    /// Creates a new builder
    pub fn new() -> Self {
        Self {
            selector: ModelSelector::new(),
        }
    }

    /// Adds a metric with weight and direction
    pub fn metric(mut self, name: &str, weight: f64, higher_is_better: bool) -> Self {
        self.selector.add_metric(name, weight, higher_is_better);
        self
    }

    /// Sets the aggregation strategy
    pub fn aggregation(mut self, strategy: AggregationStrategy) -> Self {
        self.selector.with_aggregation(strategy);
        self
    }

    /// Adds a threshold for a metric
    pub fn threshold(mut self, metric_name: &str, threshold: f64) -> Self {
        self.selector.add_threshold(metric_name, threshold);
        self
    }

    /// Builds the selector and performs complete model selection
    pub fn select(
        self,
        model_scores: &HashMap<String, Vec<(&str, f64)>>,
    ) -> Result<SelectionResult> {
        let selected_model = self.selector.select_best(model_scores)?;
        let rankings = self.selector.rank_models(model_scores)?;
        let pareto_optimal = self.selector.find_pareto_optimal(model_scores);

        let score = rankings
            .iter()
            .find(|(name, _)| name == &selected_model)
            .map(|(_, score)| *score)
            .unwrap_or(0.0);

        Ok(SelectionResult {
            selected_model,
            score,
            rankings,
            pareto_optimal,
        })
    }
}

impl Default for ModelSelectionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_scores() -> HashMap<String, Vec<(&'static str, f64)>> {
        let mut scores = HashMap::new();
        scores.insert(
            "model_a".to_string(),
            vec![("accuracy", 0.85), ("precision", 0.82), ("speed", 100.0)],
        );
        scores.insert(
            "model_b".to_string(),
            vec![("accuracy", 0.80), ("precision", 0.90), ("speed", 200.0)],
        );
        scores.insert(
            "model_c".to_string(),
            vec![("accuracy", 0.88), ("precision", 0.85), ("speed", 150.0)],
        );
        scores
    }

    #[test]
    fn test_basic_selection() {
        let scores = create_test_scores();

        let mut selector = ModelSelector::new();
        selector
            .add_metric("accuracy", 0.6, true)
            .add_metric("precision", 0.4, true);

        let best = selector.select_best(&scores).unwrap();
        assert!(!best.is_empty());
    }

    #[test]
    fn test_ranking() {
        let scores = create_test_scores();

        let mut selector = ModelSelector::new();
        selector
            .add_metric("accuracy", 0.5, true)
            .add_metric("precision", 0.5, true);

        let rankings = selector.rank_models(&scores).unwrap();
        assert_eq!(rankings.len(), 3);

        // Rankings should be sorted by score (descending)
        for i in 1..rankings.len() {
            assert!(rankings[i - 1].1 >= rankings[i].1);
        }
    }

    #[test]
    fn test_pareto_optimal() {
        let scores = create_test_scores();

        let mut selector = ModelSelector::new();
        selector
            .add_metric("accuracy", 1.0, true)
            .add_metric("speed", 1.0, true);

        let pareto = selector.find_pareto_optimal(&scores);
        assert!(!pareto.is_empty());
    }

    #[test]
    fn test_thresholds() {
        let scores = create_test_scores();

        let mut selector = ModelSelector::new();
        selector
            .add_metric("accuracy", 1.0, true)
            .add_threshold("accuracy", 0.87); // Only model_c meets this

        let rankings = selector.rank_models(&scores).unwrap();
        assert_eq!(rankings.len(), 1);
        assert_eq!(rankings[0].0, "model_c");
    }

    #[test]
    fn test_different_aggregation_strategies() {
        let scores = create_test_scores();

        let strategies = [
            AggregationStrategy::WeightedSum,
            AggregationStrategy::WeightedGeometricMean,
            AggregationStrategy::MinScore,
            AggregationStrategy::MaxScore,
        ];

        for strategy in &strategies {
            let mut selector = ModelSelector::new();
            selector
                .add_metric("accuracy", 0.5, true)
                .add_metric("precision", 0.5, true)
                .with_aggregation(*strategy);

            let best = selector.select_best(&scores).unwrap();
            assert!(!best.is_empty());
        }
    }

    #[test]
    fn test_builder_pattern() {
        let scores = create_test_scores();

        let result = ModelSelectionBuilder::new()
            .metric("accuracy", 0.6, true)
            .metric("precision", 0.4, true)
            .threshold("accuracy", 0.8)
            .aggregation(AggregationStrategy::WeightedSum)
            .select(&scores)
            .unwrap();

        assert!(!result.selected_model.is_empty());
        assert!(!result.rankings.is_empty());
        assert!(!result.pareto_optimal.is_empty());
    }

    #[test]
    fn test_empty_models() {
        let scores = HashMap::new();
        let selector = ModelSelector::new();

        assert!(selector.select_best(&scores).is_err());
    }

    #[test]
    fn test_minimization_metrics() {
        let mut scores = HashMap::new();
        scores.insert("model_a".to_string(), vec![("error", 0.1), ("time", 5.0)]);
        scores.insert("model_b".to_string(), vec![("error", 0.2), ("time", 3.0)]);

        let mut selector = ModelSelector::new();
        selector
            .add_metric("error", 0.7, false)    // lower is better
            .add_metric("time", 0.3, false); // lower is better

        let best = selector.select_best(&scores).unwrap();
        assert!(!best.is_empty());
    }
}
