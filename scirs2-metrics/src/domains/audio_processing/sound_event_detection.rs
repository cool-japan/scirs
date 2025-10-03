//! Sound event detection metrics
//!
//! This module provides metrics for evaluating sound event detection systems,
//! including event-based, segment-based, and class-wise evaluation metrics.

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use crate::error::{MetricsError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Sound event detection metrics
#[derive(Debug, Clone)]
pub struct SoundEventDetectionMetrics {
    /// Event-based metrics
    event_based: EventBasedMetrics,
    /// Segment-based metrics
    segment_based: SegmentBasedMetrics,
    /// Class-wise metrics
    class_wise: ClassWiseEventMetrics,
}

/// Event-based detection metrics
#[derive(Debug, Clone, Default)]
pub struct EventBasedMetrics {
    /// Error rate
    error_rate: f64,
    /// F1 score
    f1_score: f64,
    /// Precision
    precision: f64,
    /// Recall
    recall: f64,
    /// Deletion rate
    deletion_rate: f64,
    /// Insertion rate
    insertion_rate: f64,
}

/// Segment-based detection metrics
#[derive(Debug, Clone, Default)]
pub struct SegmentBasedMetrics {
    /// F1 score
    f1_score: f64,
    /// Precision
    precision: f64,
    /// Recall
    recall: f64,
    /// Equal error rate
    equal_error_rate: f64,
}

/// Class-wise event detection metrics
#[derive(Debug, Clone, Default)]
pub struct ClassWiseEventMetrics {
    /// Per-class F1 scores
    class_f1_scores: HashMap<String, f64>,
    /// Per-class precision
    class_precision: HashMap<String, f64>,
    /// Per-class recall
    class_recall: HashMap<String, f64>,
    /// Average metrics across classes
    macro_averaged: EventBasedMetrics,
}

/// Sound event detection results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoundEventResults {
    /// Event-based F1 score
    pub event_f1: f64,
    /// Segment-based F1 score
    pub segment_f1: f64,
    /// Error rate
    pub error_rate: f64,
    /// Equal error rate
    pub equal_error_rate: f64,
}

/// Sound event structure
#[derive(Debug, Clone, PartialEq)]
pub struct SoundEvent {
    /// Event class/label
    pub label: String,
    /// Start time in seconds
    pub onset: f64,
    /// End time in seconds
    pub offset: f64,
    /// Confidence score (optional)
    pub confidence: Option<f64>,
}

impl SoundEventDetectionMetrics {
    /// Create new sound event detection metrics
    pub fn new() -> Self {
        Self {
            event_based: EventBasedMetrics::default(),
            segment_based: SegmentBasedMetrics::default(),
            class_wise: ClassWiseEventMetrics::default(),
        }
    }

    /// Evaluate sound event detection performance
    pub fn evaluate_events(
        &mut self,
        reference_events: &[SoundEvent],
        predicted_events: &[SoundEvent],
        tolerance: f64,
    ) -> Result<SoundEventResults> {
        // Compute event-based metrics
        self.event_based
            .evaluate_event_based(reference_events, predicted_events, tolerance)?;

        // Compute segment-based metrics
        self.segment_based
            .evaluate_segment_based(reference_events, predicted_events, tolerance)?;

        // Compute class-wise metrics
        self.class_wise
            .evaluate_class_wise(reference_events, predicted_events, tolerance)?;

        Ok(SoundEventResults {
            event_f1: self.event_based.f1_score,
            segment_f1: self.segment_based.f1_score,
            error_rate: self.event_based.error_rate,
            equal_error_rate: self.segment_based.equal_error_rate,
        })
    }

    /// Get event-based F1 score
    pub fn get_event_f1(&self) -> f64 {
        self.event_based.f1_score
    }

    /// Get segment-based F1 score
    pub fn get_segment_f1(&self) -> f64 {
        self.segment_based.f1_score
    }

    /// Get class-wise F1 scores
    pub fn get_class_f1_scores(&self) -> &HashMap<String, f64> {
        &self.class_wise.class_f1_scores
    }
}

impl EventBasedMetrics {
    /// Evaluate event-based metrics
    pub fn evaluate_event_based(
        &mut self,
        reference_events: &[SoundEvent],
        predicted_events: &[SoundEvent],
        tolerance: f64,
    ) -> Result<()> {
        let mut true_positives = 0;
        let mut false_positives = 0;

        // Match predicted events to reference events
        let mut matched_references = vec![false; reference_events.len()];

        for pred_event in predicted_events {
            let mut matched = false;

            for (i, ref_event) in reference_events.iter().enumerate() {
                if !matched_references[i] && self.events_match(ref_event, pred_event, tolerance) {
                    true_positives += 1;
                    matched_references[i] = true;
                    matched = true;
                    break;
                }
            }

            if !matched {
                false_positives += 1;
            }
        }

        // Count unmatched reference events as false negatives
        let false_negatives = matched_references
            .iter()
            .filter(|&&matched| !matched)
            .count();

        // Compute metrics
        self.precision = if true_positives + false_positives > 0 {
            true_positives as f64 / (true_positives + false_positives) as f64
        } else {
            0.0
        };

        self.recall = if true_positives + false_negatives > 0 {
            true_positives as f64 / (true_positives + false_negatives) as f64
        } else {
            0.0
        };

        self.f1_score = if self.precision + self.recall > 0.0 {
            2.0 * self.precision * self.recall / (self.precision + self.recall)
        } else {
            0.0
        };

        self.error_rate = if !reference_events.is_empty() {
            (false_positives + false_negatives) as f64 / reference_events.len() as f64
        } else {
            0.0
        };

        self.deletion_rate = if !reference_events.is_empty() {
            false_negatives as f64 / reference_events.len() as f64
        } else {
            0.0
        };

        self.insertion_rate = if !reference_events.is_empty() {
            false_positives as f64 / reference_events.len() as f64
        } else {
            0.0
        };

        Ok(())
    }

    /// Check if two events match within tolerance
    fn events_match(
        &self,
        ref_event: &SoundEvent,
        pred_event: &SoundEvent,
        tolerance: f64,
    ) -> bool {
        // Events must have the same label
        if ref_event.label != pred_event.label {
            return false;
        }

        // Check temporal overlap with tolerance
        let ref_start = ref_event.onset;
        let ref_end = ref_event.offset;
        let pred_start = pred_event.onset;
        let pred_end = pred_event.offset;

        // Events match if they have sufficient temporal overlap
        let overlap_start = pred_start.max(ref_start);
        let overlap_end = pred_end.min(ref_end);
        let overlap_duration = (overlap_end - overlap_start).max(0.0);

        let ref_duration = ref_end - ref_start;
        let pred_duration = pred_end - pred_start;

        // Require at least 50% overlap relative to the shorter event
        let min_duration = ref_duration.min(pred_duration);
        overlap_duration >= min_duration * 0.5 && (pred_start - ref_start).abs() <= tolerance
    }
}

impl SegmentBasedMetrics {
    /// Evaluate segment-based metrics
    pub fn evaluate_segment_based(
        &mut self,
        reference_events: &[SoundEvent],
        predicted_events: &[SoundEvent],
        tolerance: f64,
    ) -> Result<()> {
        // Create time grid for segment-based evaluation
        let time_resolution = 0.01; // 10ms resolution
        let max_time = reference_events
            .iter()
            .chain(predicted_events.iter())
            .map(|e| e.offset)
            .fold(0.0f64, f64::max);

        if max_time <= 0.0 {
            self.f1_score = 0.0;
            self.precision = 0.0;
            self.recall = 0.0;
            return Ok(());
        }

        let num_segments = (max_time / time_resolution).ceil() as usize;

        // Create reference and prediction segment labels
        let mut ref_segments = vec![false; num_segments];
        let mut pred_segments = vec![false; num_segments];

        // Fill reference segments
        for event in reference_events {
            let start_idx = (event.onset / time_resolution) as usize;
            let end_idx = ((event.offset / time_resolution) as usize).min(num_segments);

            for i in start_idx..end_idx {
                ref_segments[i] = true;
            }
        }

        // Fill prediction segments
        for event in predicted_events {
            let start_idx = (event.onset / time_resolution) as usize;
            let end_idx = ((event.offset / time_resolution) as usize).min(num_segments);

            for i in start_idx..end_idx {
                pred_segments[i] = true;
            }
        }

        // Compute segment-based metrics
        let mut true_positives = 0;
        let mut false_positives = 0;
        let mut false_negatives = 0;

        for (ref_seg, pred_seg) in ref_segments.iter().zip(pred_segments.iter()) {
            match (ref_seg, pred_seg) {
                (true, true) => true_positives += 1,
                (false, true) => false_positives += 1,
                (true, false) => false_negatives += 1,
                (false, false) => {} // True negative
            }
        }

        self.precision = if true_positives + false_positives > 0 {
            true_positives as f64 / (true_positives + false_positives) as f64
        } else {
            0.0
        };

        self.recall = if true_positives + false_negatives > 0 {
            true_positives as f64 / (true_positives + false_negatives) as f64
        } else {
            0.0
        };

        self.f1_score = if self.precision + self.recall > 0.0 {
            2.0 * self.precision * self.recall / (self.precision + self.recall)
        } else {
            0.0
        };

        // Equal Error Rate would require probability scores
        self.equal_error_rate = 0.0; // Placeholder

        Ok(())
    }
}

impl ClassWiseEventMetrics {
    /// Evaluate class-wise metrics
    pub fn evaluate_class_wise(
        &mut self,
        reference_events: &[SoundEvent],
        predicted_events: &[SoundEvent],
        tolerance: f64,
    ) -> Result<()> {
        // Get all unique class labels
        let mut all_labels = std::collections::HashSet::new();
        for event in reference_events.iter().chain(predicted_events.iter()) {
            all_labels.insert(event.label.clone());
        }

        let mut class_metrics = Vec::new();

        for label in &all_labels {
            // Filter events for this class
            let ref_class_events: Vec<_> = reference_events
                .iter()
                .filter(|e| &e.label == label)
                .cloned()
                .collect();

            let pred_class_events: Vec<_> = predicted_events
                .iter()
                .filter(|e| &e.label == label)
                .cloned()
                .collect();

            // Compute metrics for this class
            let mut class_event_metrics = EventBasedMetrics::default();
            class_event_metrics.evaluate_event_based(
                &ref_class_events,
                &pred_class_events,
                tolerance,
            )?;

            // Store class-specific metrics
            self.class_f1_scores
                .insert(label.clone(), class_event_metrics.f1_score);
            self.class_precision
                .insert(label.clone(), class_event_metrics.precision);
            self.class_recall
                .insert(label.clone(), class_event_metrics.recall);

            class_metrics.push(class_event_metrics);
        }

        // Compute macro-averaged metrics
        if !class_metrics.is_empty() {
            self.macro_averaged.precision =
                class_metrics.iter().map(|m| m.precision).sum::<f64>() / class_metrics.len() as f64;
            self.macro_averaged.recall =
                class_metrics.iter().map(|m| m.recall).sum::<f64>() / class_metrics.len() as f64;
            self.macro_averaged.f1_score =
                class_metrics.iter().map(|m| m.f1_score).sum::<f64>() / class_metrics.len() as f64;
            self.macro_averaged.error_rate =
                class_metrics.iter().map(|m| m.error_rate).sum::<f64>()
                    / class_metrics.len() as f64;
        }

        Ok(())
    }
}

impl Default for SoundEventDetectionMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl SoundEvent {
    /// Create new sound event
    pub fn new(label: String, onset: f64, offset: f64) -> Self {
        Self {
            label,
            onset,
            offset,
            confidence: None,
        }
    }

    /// Create new sound event with confidence
    pub fn new_with_confidence(label: String, onset: f64, offset: f64, confidence: f64) -> Self {
        Self {
            label,
            onset,
            offset,
            confidence: Some(confidence),
        }
    }

    /// Get event duration
    pub fn duration(&self) -> f64 {
        self.offset - self.onset
    }

    /// Check if event overlaps with another event
    pub fn overlaps_with(&self, other: &SoundEvent) -> bool {
        !(self.offset <= other.onset || other.offset <= self.onset)
    }
}
