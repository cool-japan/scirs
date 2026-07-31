//! Robotic perception metrics
//!
//! This module provides metrics for evaluating robotic perception systems,
//! including object detection, scene understanding, and sensor fusion.

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use super::core::{BoundingBox, RealTimePerformanceMetrics};
use crate::error::{MetricsError, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// A single 3D object detection: bounding box, confidence, and class label.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Detection3D {
    /// Predicted bounding box.
    pub bbox: BoundingBox,
    /// Detector confidence in `[0, 1]`.
    pub confidence: f64,
    /// Predicted class label.
    pub class_id: usize,
}

/// A single 3D ground-truth object annotation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundTruth3D {
    /// Ground-truth bounding box.
    pub bbox: BoundingBox,
    /// Ground-truth class label.
    pub class_id: usize,
}

/// 3D Intersection-over-Union between two axis-aligned bounding boxes.
/// Returns `0.0` when the boxes don't overlap or are both degenerate
/// (zero-volume).
fn iou_3d(a: &BoundingBox, b: &BoundingBox) -> f64 {
    let Some(inter) = a.intersection(b) else {
        return 0.0;
    };
    let inter_vol = inter.volume();
    let union_vol = a.volume() + b.volume() - inter_vol;
    if union_vol <= 0.0 {
        0.0
    } else {
        inter_vol / union_vol
    }
}

/// Robotic perception evaluation metrics
#[derive(Debug, Clone)]
pub struct RoboticPerceptionMetrics {
    /// Object detection performance
    pub object_detection: ObjectDetectionMetrics,
    /// Scene understanding capabilities
    pub scene_understanding: SceneUnderstandingMetrics,
    /// Sensor fusion quality
    pub sensor_fusion: SensorFusionMetrics,
    /// Float-time performance
    pub real_time_performance: RealTimePerformanceMetrics,
}

/// Object detection evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectDetectionMetrics {
    /// Detection accuracy (mAP)
    pub detection_accuracy: f64,
    /// False positive rate
    pub false_positive_rate: f64,
    /// False negative rate
    pub false_negative_rate: f64,
    /// Localization accuracy
    pub localization_accuracy: f64,
    /// Detection latency
    pub detection_latency: Duration,
}

/// Scene understanding evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneUnderstandingMetrics {
    /// Semantic segmentation accuracy
    pub segmentation_accuracy: f64,
    /// Depth estimation accuracy
    pub depth_accuracy: f64,
    /// Scene classification accuracy
    pub classification_accuracy: f64,
    /// Spatial relationship understanding
    pub spatial_understanding: f64,
}

/// Sensor fusion quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorFusionMetrics {
    /// Fusion accuracy improvement
    pub accuracy_improvement: f64,
    /// Sensor agreement score
    pub sensor_agreement: f64,
    /// Uncertainty quantification quality
    pub uncertainty_quality: f64,
    /// Robustness to sensor failures
    pub failure_robustness: f64,
}

impl RoboticPerceptionMetrics {
    /// Create new robotic perception metrics
    pub fn new() -> Self {
        Self {
            object_detection: ObjectDetectionMetrics::default(),
            scene_understanding: SceneUnderstandingMetrics::default(),
            sensor_fusion: SensorFusionMetrics::default(),
            real_time_performance: RealTimePerformanceMetrics::default(),
        }
    }

    /// Evaluate 3D object detection quality via real greedy IoU matching and
    /// per-class average precision (mean average precision, mAP).
    ///
    /// For each class: predictions are sorted by confidence (descending) and
    /// greedily matched to the highest-IoU unmatched ground-truth box of the
    /// same class (a match requires `iou >= iou_threshold`); this produces a
    /// precision/recall curve, integrated via the standard all-points
    /// interpolation to give that class's Average Precision. `mAP` is the
    /// mean AP over classes that have at least one ground-truth instance.
    pub fn evaluate_object_detection(
        &mut self,
        predictions: &[Detection3D],
        ground_truth: &[GroundTruth3D],
        iou_threshold: f64,
        mean_detection_latency: Duration,
    ) -> Result<ObjectDetectionMetrics> {
        if ground_truth.is_empty() {
            return Err(MetricsError::InvalidInput(
                "ground_truth must not be empty".to_string(),
            ));
        }

        let mut classes: Vec<usize> = ground_truth.iter().map(|g| g.class_id).collect();
        classes.extend(predictions.iter().map(|p| p.class_id));
        classes.sort_unstable();
        classes.dedup();

        let mut aps = Vec::new();
        let mut total_tp = 0usize;
        let mut total_fp = 0usize;
        let mut matched_ious = Vec::new();

        for &class in &classes {
            let gt_indices: Vec<usize> = (0..ground_truth.len())
                .filter(|&i| ground_truth[i].class_id == class)
                .collect();
            if gt_indices.is_empty() {
                continue; // AP is only defined for classes with ground truth
            }
            let num_gt = gt_indices.len();
            let mut gt_matched = vec![false; ground_truth.len()];

            let mut preds: Vec<&Detection3D> =
                predictions.iter().filter(|p| p.class_id == class).collect();
            preds.sort_by(|a, b| {
                b.confidence
                    .partial_cmp(&a.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let mut tp_flags = Vec::with_capacity(preds.len());
            for pred in &preds {
                let mut best_iou = 0.0;
                let mut best_gt: Option<usize> = None;
                for &gi in &gt_indices {
                    if gt_matched[gi] {
                        continue;
                    }
                    let iou = iou_3d(&pred.bbox, &ground_truth[gi].bbox);
                    if iou > best_iou {
                        best_iou = iou;
                        best_gt = Some(gi);
                    }
                }
                if best_iou >= iou_threshold {
                    if let Some(gi) = best_gt {
                        gt_matched[gi] = true;
                        tp_flags.push(true);
                        matched_ious.push(best_iou);
                        total_tp += 1;
                        continue;
                    }
                }
                tp_flags.push(false);
                total_fp += 1;
            }

            // Precision/recall curve at each prediction rank.
            let mut cum_tp = 0usize;
            let mut cum_fp = 0usize;
            let mut precisions = Vec::with_capacity(tp_flags.len());
            let mut recalls = Vec::with_capacity(tp_flags.len());
            for &is_tp in &tp_flags {
                if is_tp {
                    cum_tp += 1;
                } else {
                    cum_fp += 1;
                }
                precisions.push(cum_tp as f64 / (cum_tp + cum_fp) as f64);
                recalls.push(cum_tp as f64 / num_gt as f64);
            }

            aps.push(average_precision(&precisions, &recalls));
        }

        let detection_accuracy = if aps.is_empty() {
            0.0
        } else {
            aps.iter().sum::<f64>() / aps.len() as f64
        };

        let total_predictions = predictions.len();
        let false_positive_rate = if total_predictions == 0 {
            0.0
        } else {
            total_fp as f64 / total_predictions as f64
        };
        let total_fn = ground_truth.len().saturating_sub(total_tp);
        let false_negative_rate = total_fn as f64 / ground_truth.len() as f64;
        let localization_accuracy = if matched_ious.is_empty() {
            0.0
        } else {
            matched_ious.iter().sum::<f64>() / matched_ious.len() as f64
        };

        let result = ObjectDetectionMetrics {
            detection_accuracy,
            false_positive_rate,
            false_negative_rate,
            localization_accuracy,
            detection_latency: mean_detection_latency,
        };
        self.object_detection = result.clone();
        Ok(result)
    }

    /// Evaluate scene-understanding quality from real per-sample predictions.
    ///
    /// - `predicted_segmentation`/`ground_truth_segmentation`: per-pixel (or
    ///   per-voxel) class labels; `segmentation_accuracy` is the fraction
    ///   that match.
    /// - `predicted_depth`/`ground_truth_depth`: per-pixel depth estimates
    ///   (must be strictly positive); `depth_accuracy` is the standard
    ///   monocular-depth "delta < 1.25" threshold accuracy from Eigen et al.
    ///   (2014): the fraction of pixels with `max(pred/gt, gt/pred) < 1.25`.
    /// - `scene_class_correct`: one bool per scene classified.
    /// - `spatial_relations_correct`: one bool per pairwise spatial judgment
    ///   (e.g. "is A left of B").
    pub fn evaluate_scene_understanding(
        &mut self,
        predicted_segmentation: &[usize],
        ground_truth_segmentation: &[usize],
        predicted_depth: &[f64],
        ground_truth_depth: &[f64],
        scene_class_correct: &[bool],
        spatial_relations_correct: &[bool],
    ) -> Result<SceneUnderstandingMetrics> {
        if predicted_segmentation.len() != ground_truth_segmentation.len() {
            return Err(MetricsError::InvalidInput(
                "predicted_segmentation and ground_truth_segmentation must have the same length"
                    .to_string(),
            ));
        }
        if predicted_depth.len() != ground_truth_depth.len() {
            return Err(MetricsError::InvalidInput(
                "predicted_depth and ground_truth_depth must have the same length".to_string(),
            ));
        }
        if predicted_segmentation.is_empty() || predicted_depth.is_empty() {
            return Err(MetricsError::InvalidInput(
                "segmentation and depth arrays must not be empty".to_string(),
            ));
        }
        if predicted_depth
            .iter()
            .chain(ground_truth_depth.iter())
            .any(|&d| d <= 0.0)
        {
            return Err(MetricsError::InvalidInput(
                "depth values must be strictly positive".to_string(),
            ));
        }

        let seg_matches = predicted_segmentation
            .iter()
            .zip(ground_truth_segmentation.iter())
            .filter(|(p, g)| p == g)
            .count();
        let segmentation_accuracy = seg_matches as f64 / predicted_segmentation.len() as f64;

        let depth_within_threshold = predicted_depth
            .iter()
            .zip(ground_truth_depth.iter())
            .filter(|(&p, &g)| (p / g).max(g / p) < 1.25)
            .count();
        let depth_accuracy = depth_within_threshold as f64 / predicted_depth.len() as f64;

        let classification_accuracy = if scene_class_correct.is_empty() {
            1.0
        } else {
            scene_class_correct.iter().filter(|&&c| c).count() as f64
                / scene_class_correct.len() as f64
        };

        let spatial_understanding = if spatial_relations_correct.is_empty() {
            1.0
        } else {
            spatial_relations_correct.iter().filter(|&&c| c).count() as f64
                / spatial_relations_correct.len() as f64
        };

        let result = SceneUnderstandingMetrics {
            segmentation_accuracy,
            depth_accuracy,
            classification_accuracy,
            spatial_understanding,
        };
        self.scene_understanding = result.clone();
        Ok(result)
    }

    /// Evaluate multi-sensor fusion quality from real per-sample sensor
    /// readings.
    ///
    /// - `sensor_a_readings`/`sensor_b_readings`: independent single-sensor
    ///   measurements of the same quantity; `sensor_agreement` reflects how
    ///   closely they track each other.
    /// - `fused_readings`/`ground_truth`: the fusion algorithm's output and
    ///   the true value; `accuracy_improvement` compares the fused error
    ///   against `sensor_a`'s standalone error (negative if fusion made
    ///   things worse -- reported honestly either way).
    /// - `predicted_uncertainties`: the fusion algorithm's own per-sample
    ///   uncertainty estimate; `uncertainty_quality` is the empirical
    ///   coverage fraction (`|fused - truth| <= predicted_uncertainty`).
    /// - `failure_trials`: `(sensor_failed, task_succeeded)` pairs from
    ///   trials that simulate a sensor dropout; `failure_robustness` is the
    ///   success rate restricted to trials where a failure was injected
    ///   (vacuously `1.0` if none were).
    pub fn evaluate_sensor_fusion(
        &mut self,
        sensor_a_readings: &[f64],
        sensor_b_readings: &[f64],
        fused_readings: &[f64],
        ground_truth: &[f64],
        predicted_uncertainties: &[f64],
        failure_trials: &[(bool, bool)],
    ) -> Result<SensorFusionMetrics> {
        let n = sensor_a_readings.len();
        if n != sensor_b_readings.len()
            || n != fused_readings.len()
            || n != ground_truth.len()
            || n != predicted_uncertainties.len()
        {
            return Err(MetricsError::InvalidInput(
                "all sensor/fusion input arrays must have the same length".to_string(),
            ));
        }
        if n == 0 {
            return Err(MetricsError::InvalidInput(
                "sensor readings must not be empty".to_string(),
            ));
        }

        let sensor_agreement = {
            let mean_abs_diff = sensor_a_readings
                .iter()
                .zip(sensor_b_readings.iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f64>()
                / n as f64;
            1.0 / (1.0 + mean_abs_diff)
        };

        let single_sensor_error = sensor_a_readings
            .iter()
            .zip(ground_truth.iter())
            .map(|(a, g)| (a - g).abs())
            .sum::<f64>()
            / n as f64;
        let fused_error = fused_readings
            .iter()
            .zip(ground_truth.iter())
            .map(|(f, g)| (f - g).abs())
            .sum::<f64>()
            / n as f64;
        let accuracy_improvement = if single_sensor_error > 0.0 {
            (single_sensor_error - fused_error) / single_sensor_error
        } else {
            0.0
        };

        let covered = fused_readings
            .iter()
            .zip(ground_truth.iter())
            .zip(predicted_uncertainties.iter())
            .filter(|((f, g), u)| (*f - *g).abs() <= **u)
            .count();
        let uncertainty_quality = covered as f64 / n as f64;

        let failure_robustness = {
            let injected: Vec<&(bool, bool)> = failure_trials
                .iter()
                .filter(|(failed, _)| *failed)
                .collect();
            if injected.is_empty() {
                1.0
            } else {
                injected.iter().filter(|(_, succeeded)| *succeeded).count() as f64
                    / injected.len() as f64
            }
        };

        let result = SensorFusionMetrics {
            accuracy_improvement,
            sensor_agreement,
            uncertainty_quality,
            failure_robustness,
        };
        self.sensor_fusion = result.clone();
        Ok(result)
    }
}

/// Average Precision via all-points interpolation (PASCAL VOC 2010+ style):
/// the precision envelope is made monotonically non-increasing from right to
/// left, then integrated over the recall axis.
fn average_precision(precisions: &[f64], recalls: &[f64]) -> f64 {
    if precisions.is_empty() {
        return 0.0;
    }
    let mut envelope = precisions.to_vec();
    for i in (0..envelope.len() - 1).rev() {
        envelope[i] = envelope[i].max(envelope[i + 1]);
    }

    let mut ap = 0.0;
    let mut prev_recall = 0.0;
    for i in 0..envelope.len() {
        ap += (recalls[i] - prev_recall) * envelope[i];
        prev_recall = recalls[i];
    }
    ap
}

// Default implementations
//
// These are the *neutral, not-yet-evaluated* baseline (matching the "no
// evidence of a problem" convention used by every sibling metrics struct in
// this module, e.g. `1.0` for a not-yet-measured rate/accuracy and `0.0` for
// a not-yet-observed error rate) -- not a fabricated "typical" measurement.
// Call `evaluate_object_detection` / `evaluate_scene_understanding` /
// `evaluate_sensor_fusion` to replace them with real computed values.
impl Default for ObjectDetectionMetrics {
    fn default() -> Self {
        Self {
            detection_accuracy: 1.0,
            false_positive_rate: 0.0,
            false_negative_rate: 0.0,
            localization_accuracy: 1.0,
            detection_latency: Duration::from_millis(0),
        }
    }
}

impl Default for SceneUnderstandingMetrics {
    fn default() -> Self {
        Self {
            segmentation_accuracy: 1.0,
            depth_accuracy: 1.0,
            classification_accuracy: 1.0,
            spatial_understanding: 1.0,
        }
    }
}

impl Default for SensorFusionMetrics {
    fn default() -> Self {
        Self {
            accuracy_improvement: 0.0,
            sensor_agreement: 1.0,
            uncertainty_quality: 1.0,
            failure_robustness: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bbox(x0: f64, y0: f64, z0: f64, x1: f64, y1: f64, z1: f64) -> BoundingBox {
        BoundingBox::new([x0, y0, z0], [x1, y1, z1])
    }

    #[test]
    fn object_detection_perfect_matches_give_map_one() {
        let ground_truth = vec![
            GroundTruth3D {
                bbox: bbox(0.0, 0.0, 0.0, 2.0, 2.0, 2.0),
                class_id: 0,
            },
            GroundTruth3D {
                bbox: bbox(10.0, 10.0, 10.0, 12.0, 12.0, 12.0),
                class_id: 1,
            },
        ];
        let predictions = vec![
            Detection3D {
                bbox: bbox(0.0, 0.0, 0.0, 2.0, 2.0, 2.0),
                confidence: 0.9,
                class_id: 0,
            },
            Detection3D {
                bbox: bbox(10.0, 10.0, 10.0, 12.0, 12.0, 12.0),
                confidence: 0.8,
                class_id: 1,
            },
        ];
        let mut metrics = RoboticPerceptionMetrics::new();
        let result = metrics
            .evaluate_object_detection(&predictions, &ground_truth, 0.5, Duration::from_millis(20))
            .expect("evaluation should succeed");

        assert!(
            (result.detection_accuracy - 1.0).abs() < 1e-9,
            "got {}",
            result.detection_accuracy
        );
        assert_eq!(result.false_positive_rate, 0.0);
        assert_eq!(result.false_negative_rate, 0.0);
        assert!((result.localization_accuracy - 1.0).abs() < 1e-9);
    }

    #[test]
    fn object_detection_penalizes_missed_and_spurious_detections_not_hardcoded() {
        // Two class-0 ground truths, but only one gets a matching prediction;
        // the other is a total miss, plus one spurious false-positive
        // prediction far from anything.
        let ground_truth = vec![
            GroundTruth3D {
                bbox: bbox(0.0, 0.0, 0.0, 2.0, 2.0, 2.0),
                class_id: 0,
            },
            GroundTruth3D {
                bbox: bbox(20.0, 20.0, 20.0, 22.0, 22.0, 22.0),
                class_id: 0,
            },
        ];
        let predictions = vec![
            Detection3D {
                bbox: bbox(0.0, 0.0, 0.0, 2.0, 2.0, 2.0),
                confidence: 0.9,
                class_id: 0,
            },
            Detection3D {
                bbox: bbox(50.0, 50.0, 50.0, 52.0, 52.0, 52.0),
                confidence: 0.4,
                class_id: 0,
            },
        ];
        let mut metrics = RoboticPerceptionMetrics::new();
        let result = metrics
            .evaluate_object_detection(&predictions, &ground_truth, 0.5, Duration::from_millis(20))
            .expect("evaluation should succeed");

        // Precision=1.0 at the correct high-confidence detection (recall=0.5),
        // then precision drops to 0.5 once the spurious detection is counted
        // (recall stays 0.5) -> AP = 0.5*1.0 + 0*0.5 = 0.5.
        assert!(
            (result.detection_accuracy - 0.5).abs() < 1e-9,
            "expected AP=0.5 (one hit + one miss + one false alarm), got {}",
            result.detection_accuracy
        );
        assert!(
            result.detection_accuracy < 1.0 && result.detection_accuracy > 0.0,
            "must not silently report a perfect or zero score"
        );
        assert!(
            (result.false_negative_rate - 0.5).abs() < 1e-9,
            "got {}",
            result.false_negative_rate
        );
        assert!(
            (result.false_positive_rate - 0.5).abs() < 1e-9,
            "got {}",
            result.false_positive_rate
        );
    }

    #[test]
    fn object_detection_rejects_empty_ground_truth() {
        let mut metrics = RoboticPerceptionMetrics::new();
        assert!(metrics
            .evaluate_object_detection(&[], &[], 0.5, Duration::from_millis(0))
            .is_err());
    }

    #[test]
    fn scene_understanding_computes_real_accuracies() {
        let mut metrics = RoboticPerceptionMetrics::new();
        let result = metrics
            .evaluate_scene_understanding(
                &[1, 1, 2, 2, 3],
                &[1, 2, 2, 2, 3],
                &[10.0, 5.0],
                &[10.0, 6.5],
                &[true, false, true],
                &[true, true, false, true],
            )
            .expect("evaluation should succeed");

        assert!(
            (result.segmentation_accuracy - 0.8).abs() < 1e-9,
            "got {}",
            result.segmentation_accuracy
        );
        // idx0: max(10/10,10/10)=1.0 < 1.25 (within); idx1: max(5/6.5,6.5/5)=1.3, not within.
        assert!(
            (result.depth_accuracy - 0.5).abs() < 1e-9,
            "got {}",
            result.depth_accuracy
        );
        assert!((result.classification_accuracy - 2.0 / 3.0).abs() < 1e-9);
        assert!((result.spatial_understanding - 0.75).abs() < 1e-9);
    }

    #[test]
    fn sensor_fusion_computes_real_improvement_and_coverage() {
        let mut metrics = RoboticPerceptionMetrics::new();
        let result = metrics
            .evaluate_sensor_fusion(
                &[12.0, 8.0],
                &[11.0, 9.0],
                &[10.5, 9.5],
                &[10.0, 10.0],
                &[0.6, 0.3],
                &[(true, true), (true, false), (false, true)],
            )
            .expect("evaluation should succeed");

        assert!(
            (result.sensor_agreement - 0.5).abs() < 1e-9,
            "got {}",
            result.sensor_agreement
        );
        // single-sensor error = 2.0, fused error = 0.5 -> improvement = 0.75
        assert!(
            (result.accuracy_improvement - 0.75).abs() < 1e-9,
            "got {}",
            result.accuracy_improvement
        );
        // Only the first sample's true error (0.5) falls within its predicted
        // uncertainty (0.6); the second (0.5 error, 0.3 uncertainty) does not.
        assert!(
            (result.uncertainty_quality - 0.5).abs() < 1e-9,
            "got {}",
            result.uncertainty_quality
        );
        // 2 trials injected a failure; only 1 of those still succeeded.
        assert!(
            (result.failure_robustness - 0.5).abs() < 1e-9,
            "got {}",
            result.failure_robustness
        );
    }

    #[test]
    fn sensor_fusion_rejects_mismatched_lengths() {
        let mut metrics = RoboticPerceptionMetrics::new();
        assert!(metrics
            .evaluate_sensor_fusion(&[1.0], &[1.0, 2.0], &[1.0], &[1.0], &[1.0], &[])
            .is_err());
    }
}
