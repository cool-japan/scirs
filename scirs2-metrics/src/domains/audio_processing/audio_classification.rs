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
    /// Cache of the last full result computed by [`Self::compute_metrics`],
    /// so [`Self::get_results`] can return real values instead of fabricated
    /// zeros.
    last_results: Option<AudioClassificationResults>,
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
            last_results: None,
        }
    }

    /// Compute comprehensive audio classification metrics
    ///
    /// `frame_ground_truth`, when provided alongside `frame_predictions`,
    /// enables a real frame-level accuracy computation (see
    /// [`TemporalAudioMetrics::compute_frame_accuracy`]); without it, frame
    /// accuracy is left at its last-known value rather than fabricated.
    pub fn compute_metrics<F: Float + std::fmt::Debug>(
        &mut self,
        y_true: ArrayView1<i32>,
        y_pred: ArrayView1<i32>,
        y_scores: Option<ArrayView2<F>>,
        frame_predictions: Option<ArrayView2<i32>>,
        frame_ground_truth: Option<ArrayView2<i32>>,
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
            if let Some(frame_truth) = frame_ground_truth {
                self.temporal_metrics
                    .compute_frame_accuracy(frame_preds, frame_truth)?;
            }
            self.temporal_metrics
                .compute_temporal_consistency(frame_preds)?;
        }

        let results = AudioClassificationResults {
            accuracy: standard_results.accuracy,
            precision: standard_results.precision_weighted,
            recall: standard_results.recall_weighted,
            f1_score: standard_results.f1_weighted,
            eer: self.audio_specific.eer,
            auc: standard_results.auc_roc,
            frame_accuracy: self.temporal_metrics.frame_accuracy,
        };
        self.last_results = Some(results.clone());
        Ok(results)
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

    /// Compute frame-level accuracy against real ground-truth frame labels
    pub fn compute_frame_accuracy(
        &mut self,
        frame_predictions: ArrayView2<i32>,
        frame_ground_truth: ArrayView2<i32>,
    ) -> Result<f64> {
        self.temporal_metrics
            .compute_frame_accuracy(frame_predictions, frame_ground_truth)
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

    /// Get the comprehensive results from the last call to
    /// [`Self::compute_metrics`].
    ///
    /// Returns honest all-zero placeholders (documented, not fabricated)
    /// only if `compute_metrics` has never been called yet.
    pub fn get_results(&self) -> AudioClassificationResults {
        self.last_results
            .clone()
            .unwrap_or(AudioClassificationResults {
                accuracy: 0.0,
                precision: 0.0,
                recall: 0.0,
                f1_score: 0.0,
                eer: self.audio_specific.eer,
                auc: 0.0,
                frame_accuracy: self.temporal_metrics.frame_accuracy,
            })
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

    /// Compute frame-level accuracy as the real fraction of frames where
    /// `frame_predictions` agrees with `frame_ground_truth`.
    pub fn compute_frame_accuracy(
        &mut self,
        frame_predictions: ArrayView2<i32>,
        frame_ground_truth: ArrayView2<i32>,
    ) -> Result<f64> {
        if frame_predictions.dim() != frame_ground_truth.dim() {
            return Err(MetricsError::InvalidInput(format!(
                "frame_predictions {:?} and frame_ground_truth {:?} must have the same shape",
                frame_predictions.dim(),
                frame_ground_truth.dim()
            )));
        }

        let (n_utterances, n_frames) = frame_predictions.dim();

        if n_utterances == 0 || n_frames == 0 {
            return Ok(0.0);
        }

        let total_frames = (n_utterances * n_frames) as f64;
        let correct_frames = frame_predictions
            .iter()
            .zip(frame_ground_truth.iter())
            .filter(|(pred, truth)| pred == truth)
            .count() as f64;

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

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn frame_accuracy_matches_hand_computed_value_on_non_constant_data() {
        // 2 utterances x 5 frames; deliberately non-constant so a fabricated
        // constant (e.g. the old hardcoded 0.85) cannot coincidentally pass.
        let predictions =
            Array2::from_shape_vec((2, 5), vec![1, 0, 1, 1, 0, 0, 0, 1, 0, 1]).expect("shape");
        let ground_truth =
            Array2::from_shape_vec((2, 5), vec![1, 0, 0, 1, 1, 0, 1, 1, 0, 0]).expect("shape");
        // Mismatches at flat indices 2, 4, 6, 9 -> 6 correct out of 10 -> 0.6
        let mut temporal = TemporalAudioMetrics::new();
        let acc = temporal
            .compute_frame_accuracy(predictions.view(), ground_truth.view())
            .expect("shapes match");
        assert!(
            (acc - 0.6).abs() < 1e-9,
            "expected 0.6 (6/10 correct), got {acc}"
        );
        // Must not be the old fabricated constant.
        assert!((acc - 0.85).abs() > 1e-9);
    }

    #[test]
    fn frame_accuracy_is_perfect_when_predictions_match_ground_truth_exactly() {
        let data = Array2::from_shape_vec((3, 4), vec![1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0])
            .expect("shape");
        let mut temporal = TemporalAudioMetrics::new();
        let acc = temporal
            .compute_frame_accuracy(data.view(), data.view())
            .expect("shapes match");
        assert!((acc - 1.0).abs() < 1e-9);
    }

    #[test]
    fn frame_accuracy_is_zero_when_every_frame_disagrees() {
        let predictions = Array2::from_shape_vec((1, 4), vec![1, 1, 1, 1]).expect("shape");
        let ground_truth = Array2::from_shape_vec((1, 4), vec![0, 0, 0, 0]).expect("shape");
        let mut temporal = TemporalAudioMetrics::new();
        let acc = temporal
            .compute_frame_accuracy(predictions.view(), ground_truth.view())
            .expect("shapes match");
        assert!((acc - 0.0).abs() < 1e-9);
    }

    #[test]
    fn frame_accuracy_rejects_mismatched_shapes() {
        let predictions = Array2::from_shape_vec((1, 4), vec![1, 1, 1, 1]).expect("shape");
        let ground_truth = Array2::from_shape_vec((1, 3), vec![0, 0, 0]).expect("shape");
        let mut temporal = TemporalAudioMetrics::new();
        let result = temporal.compute_frame_accuracy(predictions.view(), ground_truth.view());
        assert!(result.is_err());
    }

    #[test]
    fn compute_metrics_threads_frame_ground_truth_into_real_frame_accuracy() {
        let y_true = scirs2_core::ndarray::array![0, 1, 1, 0];
        let y_pred = scirs2_core::ndarray::array![0, 1, 0, 0];
        let frame_predictions =
            Array2::from_shape_vec((2, 3), vec![1, 0, 1, 0, 1, 1]).expect("shape");
        let frame_truth = Array2::from_shape_vec((2, 3), vec![1, 0, 0, 0, 1, 0]).expect("shape");
        // Mismatches at flat indices 2 and 5 -> 4/6 correct
        let mut metrics = AudioClassificationMetrics::new();
        let results = metrics
            .compute_metrics::<f64>(
                y_true.view(),
                y_pred.view(),
                None,
                Some(frame_predictions.view()),
                Some(frame_truth.view()),
            )
            .expect("computation should succeed");
        assert!(
            (results.frame_accuracy - 4.0 / 6.0).abs() < 1e-9,
            "expected 4/6, got {}",
            results.frame_accuracy
        );

        // get_results() must reflect the same real, cached computation
        let cached = metrics.get_results();
        assert!((cached.frame_accuracy - 4.0 / 6.0).abs() < 1e-9);
        assert!((cached.accuracy - results.accuracy).abs() < 1e-9);
    }

    #[test]
    fn get_results_before_any_computation_is_honest_zero_not_fabricated() {
        let metrics = AudioClassificationMetrics::new();
        let results = metrics.get_results();
        assert_eq!(results.accuracy, 0.0);
        assert_eq!(results.frame_accuracy, 0.0);
    }
}
