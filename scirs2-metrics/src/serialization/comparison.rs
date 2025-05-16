//! Metric comparison utilities
//!
//! This module provides tools for comparing metric results between runs.

use chrono::Duration;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::{MetricCollection, MetricResult};
use crate::error::Result;

/// Comparison result between metric values
///
/// This struct represents the result of comparing two metric values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricComparison {
    /// Name of the metric
    pub name: String,
    /// First value
    pub value1: f64,
    /// Second value
    pub value2: f64,
    /// Absolute difference (value2 - value1)
    pub absolute_diff: f64,
    /// Relative difference ((value2 - value1) / value1)
    pub relative_diff: f64,
    /// Whether the difference is significant
    pub is_significant: bool,
    /// Optional threshold used for significance determination
    pub threshold: Option<f64>,
}

/// Collection comparison result
///
/// This struct represents the result of comparing two metric collections.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionComparison {
    /// Name of the first collection
    pub name1: String,
    /// Name of the second collection
    pub name2: String,
    /// Comparison results for each metric
    pub metric_comparisons: Vec<MetricComparison>,
    /// Summary statistics
    pub summary: ComparisonSummary,
}

/// Summary statistics for a collection comparison
///
/// This struct represents summary statistics for a collection comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonSummary {
    /// Number of metrics compared
    pub total_metrics: usize,
    /// Number of metrics that improved
    pub improved: usize,
    /// Number of metrics that degraded
    pub degraded: usize,
    /// Number of metrics that remained unchanged
    pub unchanged: usize,
    /// Number of metrics only in the first collection
    pub only_in_first: usize,
    /// Number of metrics only in the second collection
    pub only_in_second: usize,
    /// Average absolute difference
    pub avg_absolute_diff: f64,
    /// Average relative difference
    pub avg_relative_diff: f64,
}

/// Compare two metric results
///
/// # Arguments
///
/// * `metric1` - First metric result
/// * `metric2` - Second metric result
/// * `threshold` - Optional threshold for significance determination
///
/// # Returns
///
/// * A MetricComparison
pub fn compare_metrics(
    metric1: &MetricResult,
    metric2: &MetricResult,
    threshold: Option<f64>,
) -> MetricComparison {
    let value1 = metric1.value;
    let value2 = metric2.value;

    let absolute_diff = value2 - value1;
    let relative_diff = if value1 != 0.0 {
        absolute_diff / value1
    } else if value2 == 0.0 {
        0.0
    } else {
        1.0
    };

    let is_significant = if let Some(t) = threshold {
        relative_diff.abs() > t
    } else {
        false
    };

    MetricComparison {
        name: metric1.name.clone(),
        value1,
        value2,
        absolute_diff,
        relative_diff,
        is_significant,
        threshold,
    }
}

/// Compare two metric collections
///
/// # Arguments
///
/// * `collection1` - First collection
/// * `collection2` - Second collection
/// * `threshold` - Optional threshold for significance determination
///
/// # Returns
///
/// * A CollectionComparison
pub fn compare_collections(
    collection1: &MetricCollection,
    collection2: &MetricCollection,
    threshold: Option<f64>,
) -> CollectionComparison {
    // Create maps for faster lookup
    let mut metrics1_map = HashMap::new();
    for metric in &collection1.metrics {
        metrics1_map.insert(metric.name.clone(), metric);
    }

    let mut metrics2_map = HashMap::new();
    for metric in &collection2.metrics {
        metrics2_map.insert(metric.name.clone(), metric);
    }

    // Compare metrics that exist in both collections
    let mut metric_comparisons = Vec::new();

    let mut improved = 0;
    let mut degraded = 0;
    let mut unchanged = 0;
    let mut only_in_first = 0;
    let mut only_in_second = 0;

    let mut total_absolute_diff = 0.0;
    let mut total_relative_diff = 0.0;

    // Process all metrics from collection1
    for metric1 in &collection1.metrics {
        if let Some(metric2) = metrics2_map.get(&metric1.name) {
            // Metric exists in both collections
            let comparison = compare_metrics(metric1, metric2, threshold);

            // Update counters
            if comparison.absolute_diff > 0.0 {
                improved += 1;
            } else if comparison.absolute_diff < 0.0 {
                degraded += 1;
            } else {
                unchanged += 1;
            }

            total_absolute_diff += comparison.absolute_diff.abs();
            total_relative_diff += comparison.relative_diff.abs();

            metric_comparisons.push(comparison);
        } else {
            // Metric only in collection1
            only_in_first += 1;
        }
    }

    // Find metrics only in collection2
    for metric2 in &collection2.metrics {
        if !metrics1_map.contains_key(&metric2.name) {
            only_in_second += 1;
        }
    }

    // Calculate averages
    let compared_count = metric_comparisons.len();
    let avg_absolute_diff = if compared_count > 0 {
        total_absolute_diff / compared_count as f64
    } else {
        0.0
    };

    let avg_relative_diff = if compared_count > 0 {
        total_relative_diff / compared_count as f64
    } else {
        0.0
    };

    // Create summary
    let summary = ComparisonSummary {
        total_metrics: compared_count,
        improved,
        degraded,
        unchanged,
        only_in_first,
        only_in_second,
        avg_absolute_diff,
        avg_relative_diff,
    };

    CollectionComparison {
        name1: collection1.name.clone(),
        name2: collection2.name.clone(),
        metric_comparisons,
        summary,
    }
}

/// Find metrics that differ significantly between collections
///
/// # Arguments
///
/// * `collection1` - First collection
/// * `collection2` - Second collection
/// * `threshold` - Threshold for significance determination
///
/// # Returns
///
/// * Vector of significantly different metrics
pub fn find_significant_differences(
    collection1: &MetricCollection,
    collection2: &MetricCollection,
    threshold: f64,
) -> Vec<MetricComparison> {
    let comparison = compare_collections(collection1, collection2, Some(threshold));

    comparison
        .metric_comparisons
        .into_iter()
        .filter(|comp| comp.is_significant)
        .collect()
}

/// Combine multiple metric collections into one
///
/// # Arguments
///
/// * `collections` - Collections to combine
/// * `name` - Name for the combined collection
/// * `description` - Optional description for the combined collection
///
/// # Returns
///
/// * A combined MetricCollection
pub fn combine_collections(
    collections: &[MetricCollection],
    name: &str,
    description: Option<&str>,
) -> MetricCollection {
    let mut combined = MetricCollection::new(name, description);

    for collection in collections {
        for metric in &collection.metrics {
            // Create a new metric with original metadata plus collection source
            let mut metadata = metric.metadata.clone().unwrap_or_default();

            if metadata.additional_metadata.is_none() {
                metadata.additional_metadata = Some(HashMap::new());
            }

            if let Some(ref mut additional) = metadata.additional_metadata {
                additional.insert("source_collection".to_string(), collection.name.clone());
                additional.insert(
                    "source_timestamp".to_string(),
                    collection.created_at.to_string(),
                );
            }

            let new_metric = MetricResult {
                name: metric.name.clone(),
                value: metric.value,
                additional_values: metric.additional_values.clone(),
                timestamp: metric.timestamp,
                metadata: Some(metadata),
            };

            combined.add_metric(new_metric);
        }
    }

    combined
}

/// Get metrics from a collection by filtering
///
/// # Arguments
///
/// * `collection` - Collection to filter
/// * `filter_fn` - Function to filter metrics
///
/// # Returns
///
/// * A new collection with filtered metrics
pub fn filter_metrics<F>(collection: &MetricCollection, filter_fn: F) -> MetricCollection
where
    F: Fn(&MetricResult) -> bool,
{
    let mut filtered = MetricCollection::new(
        &format!("{} (filtered)", collection.name),
        collection.description.as_deref(),
    );

    for metric in &collection.metrics {
        if filter_fn(metric) {
            filtered.add_metric(metric.clone());
        }
    }

    filtered
}

/// Get metrics from a collection by name pattern
///
/// # Arguments
///
/// * `collection` - Collection to filter
/// * `pattern` - Pattern to match metric names against
///
/// # Returns
///
/// * A new collection with filtered metrics
pub fn filter_by_name(collection: &MetricCollection, pattern: &str) -> MetricCollection {
    filter_metrics(collection, |metric| metric.name.contains(pattern))
}

/// Get metrics from a collection by time range
///
/// # Arguments
///
/// * `collection` - Collection to filter
/// * `start_age` - Oldest age to include (Duration from now)
/// * `end_age` - Newest age to include (Duration from now)
///
/// # Returns
///
/// * A new collection with filtered metrics
pub fn filter_by_time_range(
    collection: &MetricCollection,
    start_age: Duration,
    end_age: Duration,
) -> Result<MetricCollection> {
    let now = chrono::Utc::now();

    Ok(filter_metrics(collection, |metric| {
        let age = now - metric.timestamp;
        age >= end_age && age <= start_age
    }))
}
