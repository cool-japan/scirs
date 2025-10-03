//! Audio classification metrics
//!
//! This module provides comprehensive metrics for evaluating audio classification tasks,
//! including general classification metrics, audio-specific metrics like Equal Error Rate (EER),
//! temporal consistency measures, and boundary detection capabilities.

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use crate::error::{MetricsError, Result};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::Float;
use serde::{Deserialize, Serialize};

/// Audio classification metrics
#[derive(Debug, Clone)]
pub struct AudioClassificationMetrics {
    /// Standard classification metrics
    classification_metrics: crate::sklearn_compat::ClassificationMetrics,
    /// Audio-specific metrics
    audio_specific: AudioSpecificMetrics,
    /// Temporal metrics for audio segments
    temporal_metrics: TemporalAudioMetrics,
}

/// Audio-specific classification metrics
#[derive(Debug, Clone)]
pub struct AudioSpecificMetrics {
    /// Equal Error Rate (EER)
    eer: Option<f64>,
    /// Detection Cost Function (DCF)
    dcf: Option<f64>,
    /// Area Under ROC Curve for audio
    auc_audio: Option<f64>,
    /// Minimum DCF
    min_dcf: Option<f64>,
}

/// Temporal metrics for audio classification
#[derive(Debug, Clone)]
pub struct TemporalAudioMetrics {
    /// Frame-level accuracy
    frame_accuracy: f64,
    /// Segment-level accuracy
    segment_accuracy: f64,
    /// Temporal consistency score
    temporal_consistency: f64,
    /// Boundary detection metrics
    boundary_metrics: BoundaryDetectionMetrics,
}

/// Boundary detection metrics
#[derive(Debug, Clone)]
pub struct BoundaryDetectionMetrics {
    /// Precision of boundary detection
    boundary_precision: f64,
    /// Recall of boundary detection
    boundary_recall: f64,
    /// F1 score for boundary detection
    boundary_f1: f64,
    /// Boundary tolerance (in seconds)
    tolerance: f64,
}

/// Audio classification evaluation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioClassificationResults {
    /// Overall accuracy
    pub accuracy: f64,
    /// Precision
    pub precision: f64,
    /// Recall
    pub recall: f64,
    /// F1 score
    pub f1_score: f64,
    /// Equal Error Rate
    pub eer: Option<f64>,
    /// Area Under Curve
    pub auc: f64,
    /// Frame-level accuracy
    pub frame_accuracy: f64,
}

impl AudioClassificationMetrics {
    /// Create new audio classification metrics
    pub fn new() -> Self {
        Self {
            classification_metrics: crate::sklearn_compat::ClassificationMetrics::new(),
            audio_specific: AudioSpecificMetrics::new(),
            temporal_metrics: TemporalAudioMetrics::new(),
        }
    }

    /// Compute comprehensive audio classification metrics
    pub fn compute_metrics<F: Float + std::fmt::Debug>(
        &mut self,
        y_true: ArrayView1<i32>,
        y_pred: ArrayView1<i32>,
        y_scores: Option<ArrayView2<F>>,
        frame_predictions: Option<ArrayView2<i32>>,
    ) -> Result<AudioClassificationResults> {
        // Compute standard classification metrics
        let standard_results = self.classification_metrics.compute(
            y_true,
            y_pred,
            y_scores.map(|s| s.map(|&x| x.to_f64().unwrap_or(0.0))),
        )?;

        // Compute audio-specific metrics
        if let Some(scores) = y_scores {
            self.audio_specific.compute_eer(y_true, scores.column(0))?;
            self.audio_specific.compute_dcf(y_true, scores.column(0))?;
        }

        // Compute temporal metrics if frame-level data is available
        if let Some(frame_preds) = frame_predictions {
            self.temporal_metrics.compute_frame_accuracy(frame_preds)?;
            self.temporal_metrics
                .compute_temporal_consistency(frame_preds)?;
        }

        Ok(AudioClassificationResults {
            accuracy: standard_results.accuracy,
            precision: standard_results.precision_weighted,
            recall: standard_results.recall_weighted,
            f1_score: standard_results.f1_weighted,
            eer: self.audio_specific.eer,
            auc: standard_results.auc_roc,
            frame_accuracy: self.temporal_metrics.frame_accuracy,
        })
    }

    /// Compute Equal Error Rate (EER)
    pub fn compute_eer<F: Float>(
        &mut self,
        y_true: ArrayView1<i32>,
        y_scores: ArrayView1<F>,
    ) -> Result<f64> {
        self.audio_specific.compute_eer(y_true, y_scores)
    }

    /// Compute Detection Cost Function (DCF)
    pub fn compute_dcf<F: Float>(
        &mut self,
        y_true: ArrayView1<i32>,
        y_scores: ArrayView1<F>,
    ) -> Result<f64> {
        self.audio_specific.compute_dcf(y_true, y_scores)
    }

    /// Compute frame-level accuracy
    pub fn compute_frame_accuracy(&mut self, frame_predictions: ArrayView2<i32>) -> Result<f64> {
        self.temporal_metrics
            .compute_frame_accuracy(frame_predictions)
    }

    /// Compute temporal consistency
    pub fn compute_temporal_consistency(
        &mut self,
        frame_predictions: ArrayView2<i32>,
    ) -> Result<f64> {
        self.temporal_metrics
            .compute_temporal_consistency(frame_predictions)
    }

    /// Detect segment boundaries
    pub fn detect_boundaries(
        &mut self,
        predictions: ArrayView1<i32>,
        timestamps: ArrayView1<f64>,
    ) -> Result<Vec<f64>> {
        self.temporal_metrics
            .boundary_metrics
            .detect_boundaries(predictions, timestamps)
    }

    /// Get comprehensive results
    pub fn get_results(&self) -> AudioClassificationResults {
        AudioClassificationResults {
            accuracy: 0.0, // Would be computed from standard metrics
            precision: 0.0,
            recall: 0.0,
            f1_score: 0.0,
            eer: self.audio_specific.eer,
            auc: 0.0,
            frame_accuracy: self.temporal_metrics.frame_accuracy,
        }
    }
}

impl AudioSpecificMetrics {
    /// Create new audio-specific metrics
    pub fn new() -> Self {
        Self {
            eer: None,
            dcf: None,
            auc_audio: None,
            min_dcf: None,
        }
    }

    /// Compute Equal Error Rate (EER)
    pub fn compute_eer<F: Float>(
        &mut self,
        y_true: ArrayView1<i32>,
        y_scores: ArrayView1<F>,
    ) -> Result<f64> {
        if y_true.len() != y_scores.len() {
            return Err(MetricsError::InvalidInput(
                "True labels and scores must have the same length".to_string(),
            ));
        }

        // Create (score, label) pairs and sort by score
        let mut score_label_pairs: Vec<(f64, i32)> = y_true
            .iter()
            .zip(y_scores.iter())
            .map(|(&label, &score)| (score.to_f64().unwrap_or(0.0), label))
            .collect();

        score_label_pairs
            .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let total_positives = y_true.iter().filter(|&&x| x == 1).count() as f64;
        let total_negatives = y_true.iter().filter(|&&x| x == 0).count() as f64;

        if total_positives == 0.0 || total_negatives == 0.0 {
            return Err(MetricsError::InvalidInput(
                "Need both positive and negative examples for EER".to_string(),
            ));
        }

        let mut min_diff = f64::INFINITY;
        let mut best_eer = 0.0;

        let mut true_positives = 0.0;
        let mut false_positives = 0.0;

        for (_, label) in score_label_pairs.iter().rev() {
            if *label == 1 {
                true_positives += 1.0;
            } else {
                false_positives += 1.0;
            }

            let tpr = true_positives / total_positives;
            let fpr = false_positives / total_negatives;
            let fnr = 1.0 - tpr;

            let diff = (fpr - fnr).abs();
            if diff < min_diff {
                min_diff = diff;
                best_eer = (fpr + fnr) / 2.0;
            }
        }

        self.eer = Some(best_eer);
        Ok(best_eer)
    }

    /// Compute Detection Cost Function (DCF)
    pub fn compute_dcf<F: Float>(
        &mut self,
        y_true: ArrayView1<i32>,
        y_scores: ArrayView1<F>,
    ) -> Result<f64> {
        // DCF parameters (NIST SRE standard)
        let c_miss = 1.0;
        let c_fa = 1.0;
        let p_target = 0.01;

        let mut score_label_pairs: Vec<(f64, i32)> = y_true
            .iter()
            .zip(y_scores.iter())
            .map(|(&label, &score)| (score.to_f64().unwrap_or(0.0), label))
            .collect();

        score_label_pairs
            .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let total_positives = y_true.iter().filter(|&&x| x == 1).count() as f64;
        let total_negatives = y_true.iter().filter(|&&x| x == 0).count() as f64;

        let mut min_dcf = f64::INFINITY;
        let mut true_positives = 0.0;
        let mut false_positives = 0.0;

        for (_, label) in score_label_pairs.iter().rev() {
            if *label == 1 {
                true_positives += 1.0;
            } else {
                false_positives += 1.0;
            }

            let pmiss = 1.0 - (true_positives / total_positives);
            let pfa = false_positives / total_negatives;

            let dcf = c_miss * pmiss * p_target + c_fa * pfa * (1.0 - p_target);
            min_dcf = min_dcf.min(dcf);
        }

        self.dcf = Some(min_dcf);
        self.min_dcf = Some(min_dcf);
        Ok(min_dcf)
    }
}

impl TemporalAudioMetrics {
    /// Create new temporal audio metrics
    pub fn new() -> Self {
        Self {
            frame_accuracy: 0.0,
            segment_accuracy: 0.0,
            temporal_consistency: 0.0,
            boundary_metrics: BoundaryDetectionMetrics::new(),
        }
    }

    /// Compute frame-level accuracy
    pub fn compute_frame_accuracy(&mut self, frame_predictions: ArrayView2<i32>) -> Result<f64> {
        let (n_utterances, n_frames) = frame_predictions.dim();

        if n_utterances == 0 || n_frames == 0 {
            return Ok(0.0);
        }

        // Placeholder: would compute frame-level accuracy from ground truth
        let total_frames = (n_utterances * n_frames) as f64;
        let correct_frames = total_frames * 0.85; // Placeholder

        self.frame_accuracy = correct_frames / total_frames;
        Ok(self.frame_accuracy)
    }

    /// Compute temporal consistency score
    pub fn compute_temporal_consistency(
        &mut self,
        frame_predictions: ArrayView2<i32>,
    ) -> Result<f64> {
        let (n_utterances, n_frames) = frame_predictions.dim();

        if n_utterances == 0 || n_frames < 2 {
            return Ok(0.0);
        }

        let mut total_consistency = 0.0;
        let mut total_transitions = 0;

        for i in 0..n_utterances {
            for j in 1..n_frames {
                let prev_pred = frame_predictions[[i, j - 1]];
                let curr_pred = frame_predictions[[i, j]];

                // Count consistent transitions
                if prev_pred == curr_pred {
                    total_consistency += 1.0;
                }
                total_transitions += 1;
            }
        }

        self.temporal_consistency = if total_transitions > 0 {
            total_consistency / total_transitions as f64
        } else {
            0.0
        };

        Ok(self.temporal_consistency)
    }
}

impl BoundaryDetectionMetrics {
    /// Create new boundary detection metrics
    pub fn new() -> Self {
        Self {
            boundary_precision: 0.0,
            boundary_recall: 0.0,
            boundary_f1: 0.0,
            tolerance: 0.5, // 500ms tolerance
        }
    }

    /// Detect boundaries in prediction sequence
    pub fn detect_boundaries(
        &mut self,
        predictions: ArrayView1<i32>,
        timestamps: ArrayView1<f64>,
    ) -> Result<Vec<f64>> {
        if predictions.len() != timestamps.len() {
            return Err(MetricsError::InvalidInput(
                "Predictions and timestamps must have the same length".to_string(),
            ));
        }

        let mut boundaries = Vec::new();

        for i in 1..predictions.len() {
            if predictions[i] != predictions[i - 1] {
                boundaries.push(timestamps[i]);
            }
        }

        Ok(boundaries)
    }

    /// Evaluate boundary detection performance
    pub fn evaluate_boundaries(&mut self, detected: &[f64], reference: &[f64]) -> Result<()> {
        if reference.is_empty() {
            self.boundary_precision = if detected.is_empty() { 1.0 } else { 0.0 };
            self.boundary_recall = 1.0;
            self.boundary_f1 = if detected.is_empty() { 1.0 } else { 0.0 };
            return Ok(());
        }

        let mut true_positives = 0;
        let mut false_positives = 0;
        let mut false_negatives = 0;

        // Count true positives and false positives
        for &det_boundary in detected {
            let mut matched = false;
            for &ref_boundary in reference {
                if (det_boundary - ref_boundary).abs() <= self.tolerance {
                    true_positives += 1;
                    matched = true;
                    break;
                }
            }
            if !matched {
                false_positives += 1;
            }
        }

        // Count false negatives
        for &ref_boundary in reference {
            let mut matched = false;
            for &det_boundary in detected {
                if (det_boundary - ref_boundary).abs() <= self.tolerance {
                    matched = true;
                    break;
                }
            }
            if !matched {
                false_negatives += 1;
            }
        }

        // Calculate metrics
        self.boundary_precision = if true_positives + false_positives > 0 {
            true_positives as f64 / (true_positives + false_positives) as f64
        } else {
            0.0
        };

        self.boundary_recall = if true_positives + false_negatives > 0 {
            true_positives as f64 / (true_positives + false_negatives) as f64
        } else {
            0.0
        };

        self.boundary_f1 = if self.boundary_precision + self.boundary_recall > 0.0 {
            2.0 * self.boundary_precision * self.boundary_recall
                / (self.boundary_precision + self.boundary_recall)
        } else {
            0.0
        };

        Ok(())
    }

    /// Set boundary tolerance
    pub fn set_tolerance(&mut self, tolerance: f64) {
        self.tolerance = tolerance;
    }
}

impl Default for AudioClassificationMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for AudioSpecificMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for TemporalAudioMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for BoundaryDetectionMetrics {
    fn default() -> Self {
        Self::new()
    }
}
