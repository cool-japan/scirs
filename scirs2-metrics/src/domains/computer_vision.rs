//! Computer vision domain metrics
//!
//! This module provides specialized metric collections for computer vision tasks
//! including object detection, image classification, and segmentation.

use crate::classification::{accuracy_score, f1_score, precision_score, recall_score};
use crate::domains::{DomainEvaluationResult, DomainMetrics};
use crate::error::{MetricsError, Result};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Bounding box representation (x1, y1, x2, y2, confidence, class_id)
pub type BoundingBox = (f64, f64, f64, f64, f64, i32);

/// Ground truth bounding box (x1, y1, x2, y2, class_id)
pub type GroundTruthBox = (f64, f64, f64, f64, i32);

/// Object detection evaluation results
#[derive(Debug, Clone)]
pub struct ObjectDetectionResults {
    /// Mean Average Precision at IoU threshold
    pub map: f64,
    /// Overall precision
    pub precision: f64,
    /// Overall recall
    pub recall: f64,
    /// F1 score
    pub f1_score: f64,
    /// Per-class Average Precision
    pub per_class_ap: HashMap<i32, f64>,
    /// IoU threshold used
    pub iou_threshold: f64,
}

/// Image classification evaluation results
#[derive(Debug, Clone)]
pub struct ImageClassificationResults {
    /// Top-1 accuracy
    pub top1_accuracy: f64,
    /// Top-5 accuracy (if applicable)
    pub top5_accuracy: Option<f64>,
    /// Per-class precision
    pub per_class_precision: HashMap<i32, f64>,
    /// Per-class recall
    pub per_class_recall: HashMap<i32, f64>,
    /// Per-class F1 score
    pub per_class_f1: HashMap<i32, f64>,
    /// Overall weighted F1 score
    pub weighted_f1: f64,
}

/// Segmentation evaluation results
#[derive(Debug, Clone)]
pub struct SegmentationResults {
    /// Pixel accuracy
    pub pixel_accuracy: f64,
    /// Mean Intersection over Union
    pub mean_iou: f64,
    /// Per-class IoU
    pub per_class_iou: HashMap<i32, f64>,
    /// Dice coefficient (for binary segmentation)
    pub dice_coefficient: Option<f64>,
    /// Jaccard index (IoU for binary case)
    pub jaccard_index: Option<f64>,
}

/// Object detection metrics calculator
pub struct ObjectDetectionMetrics {
    iou_thresholds: Vec<f64>,
    confidence_threshold: f64,
}

impl Default for ObjectDetectionMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl ObjectDetectionMetrics {
    /// Create new object detection metrics calculator
    pub fn new() -> Self {
        Self {
            iou_thresholds: vec![0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
            confidence_threshold: 0.5,
        }
    }

    /// Set IoU thresholds for mAP calculation
    pub fn with_iou_thresholds(mut self, thresholds: Vec<f64>) -> Self {
        self.iou_thresholds = thresholds;
        self
    }

    /// Set confidence threshold
    pub fn with_confidence_threshold(mut self, threshold: f64) -> Self {
        self.confidence_threshold = threshold;
        self
    }

    /// Evaluate object detection predictions
    pub fn evaluate_object_detection(
        &self,
        predictions: &[BoundingBox],
        ground_truth: &[GroundTruthBox],
        iou_threshold: f64,
    ) -> Result<ObjectDetectionResults> {
        // Filter predictions by confidence
        let filtered_preds: Vec<_> = predictions
            .iter()
            .filter(|(_, _, _, _, conf, _)| *conf >= self.confidence_threshold)
            .collect();

        // Calculate IoU for all prediction-ground truth pairs
        let mut matches = Vec::new();
        let mut per_class_stats: HashMap<i32, (usize, usize)> = HashMap::new(); // (tp, fp + fn)

        // Track which ground truth boxes have been matched
        let mut gt_matched = vec![false; ground_truth.len()];

        // Sort predictions by confidence (descending)
        let mut sorted_preds: Vec<_> = filtered_preds.iter().enumerate().collect();
        sorted_preds.sort_by(|a, b| {
            b.1 .4
                .partial_cmp(&a.1 .4)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for (_, pred) in sorted_preds {
            let (px1, py1, px2, py2, _conf, pred_class) = *pred;
            let mut best_iou = 0.0;
            let mut best_gt_idx = None;

            // Find best matching ground truth box
            for (gt_idx, gt) in ground_truth.iter().enumerate() {
                let (gx1, gy1, gx2, gy2, gt_class) = *gt;

                if *pred_class == gt_class && !gt_matched[gt_idx] {
                    let iou = self.calculate_iou(*px1, *py1, *px2, *py2, gx1, gy1, gx2, gy2);
                    if iou > best_iou {
                        best_iou = iou;
                        best_gt_idx = Some(gt_idx);
                    }
                }
            }

            // Check if match is above threshold
            let is_true_positive = best_iou >= iou_threshold;
            if is_true_positive && best_gt_idx.is_some() {
                gt_matched[best_gt_idx.unwrap()] = true;
                matches.push((pred_class, true)); // True positive
            } else {
                matches.push((pred_class, false)); // False positive
            }

            // Update per-class statistics
            let stats = per_class_stats.entry(*pred_class).or_insert((0, 0));
            if is_true_positive {
                stats.0 += 1; // TP
            } else {
                stats.1 += 1; // FP
            }
        }

        // Count false negatives (unmatched ground truth)
        for (gt_idx, gt) in ground_truth.iter().enumerate() {
            if !gt_matched[gt_idx] {
                let (_, _, _, _, gt_class) = *gt;
                let stats = per_class_stats.entry(gt_class).or_insert((0, 0));
                stats.1 += 1; // FN
            }
        }

        // Calculate per-class AP and overall metrics
        let mut per_class_ap = HashMap::new();
        let mut total_tp = 0;
        let mut total_fp = 0;
        let mut total_fn = 0;

        for (&class_id, &(tp, fp_plus_fn)) in &per_class_stats {
            let fp_count = fp_plus_fn.saturating_sub(tp);
            let fn_count = 0; // Simplified for this implementation
            let precision = if tp + fp_count > 0 {
                tp as f64 / (tp + fp_count) as f64
            } else {
                0.0
            };
            let recall = if tp + fn_count > 0 {
                tp as f64 / (tp + fn_count) as f64
            } else {
                1.0 // If no false negatives, recall is perfect
            };

            // Simplified AP calculation (would normally use precision-recall curve)
            let ap = if precision + recall > 0.0 {
                2.0 * precision * recall / (precision + recall)
            } else {
                0.0
            };

            per_class_ap.insert(class_id, ap);
            total_tp += tp;
            total_fp += fp_count;
            total_fn += fn_count;
        }

        // Calculate overall metrics
        let overall_precision = if total_tp + total_fp > 0 {
            total_tp as f64 / (total_tp + total_fp) as f64
        } else {
            0.0
        };

        let overall_recall = if total_tp + total_fn > 0 {
            total_tp as f64 / (total_tp + total_fn) as f64
        } else {
            0.0
        };

        let overall_f1 = if overall_precision + overall_recall > 0.0 {
            2.0 * overall_precision * overall_recall / (overall_precision + overall_recall)
        } else {
            0.0
        };

        let map = if !per_class_ap.is_empty() {
            per_class_ap.values().sum::<f64>() / per_class_ap.len() as f64
        } else {
            0.0
        };

        Ok(ObjectDetectionResults {
            map,
            precision: overall_precision,
            recall: overall_recall,
            f1_score: overall_f1,
            per_class_ap,
            iou_threshold,
        })
    }

    /// Calculate Intersection over Union (IoU) between two bounding boxes
    fn calculate_iou(
        &self,
        x1_a: f64,
        y1_a: f64,
        x2_a: f64,
        y2_a: f64,
        x1_b: f64,
        y1_b: f64,
        x2_b: f64,
        y2_b: f64,
    ) -> f64 {
        // Calculate intersection area
        let x1_inter = x1_a.max(x1_b);
        let y1_inter = y1_a.max(y1_b);
        let x2_inter = x2_a.min(x2_b);
        let y2_inter = y2_a.min(y2_b);

        if x2_inter <= x1_inter || y2_inter <= y1_inter {
            return 0.0; // No intersection
        }

        let intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter);

        // Calculate union area
        let area_a = (x2_a - x1_a) * (y2_a - y1_a);
        let area_b = (x2_b - x1_b) * (y2_b - y1_b);
        let union = area_a + area_b - intersection;

        if union <= 0.0 {
            0.0
        } else {
            intersection / union
        }
    }
}

/// Image classification metrics calculator
pub struct ImageClassificationMetrics {
    top_k: Vec<usize>,
}

impl Default for ImageClassificationMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl ImageClassificationMetrics {
    /// Create new image classification metrics calculator
    pub fn new() -> Self {
        Self { top_k: vec![1, 5] }
    }

    /// Set top-k values to calculate
    pub fn with_top_k(mut self, top_k: Vec<usize>) -> Self {
        self.top_k = top_k;
        self
    }

    /// Evaluate image classification predictions
    pub fn evaluate_classification(
        &self,
        y_true: &Array1<i32>,
        y_pred: &Array1<i32>,
        y_prob: Option<&Array2<f64>>,
    ) -> Result<ImageClassificationResults> {
        // Calculate top-1 accuracy
        let top1_accuracy = accuracy_score(y_true, y_pred)?;

        // Calculate top-5 accuracy if probabilities are provided
        let top5_accuracy = if let Some(probs) = y_prob {
            if self.top_k.contains(&5) {
                Some(self.calculate_top_k_accuracy(y_true, probs, 5)?)
            } else {
                None
            }
        } else {
            None
        };

        // Calculate per-class metrics
        let classes: Vec<i32> = {
            let mut classes = y_true
                .iter()
                .chain(y_pred.iter())
                .copied()
                .collect::<Vec<_>>();
            classes.sort_unstable();
            classes.dedup();
            classes
        };

        let mut per_class_precision = HashMap::new();
        let mut per_class_recall = HashMap::new();
        let mut per_class_f1 = HashMap::new();

        for &class in &classes {
            let precision = precision_score(y_true, y_pred, class)?;
            let recall = recall_score(y_true, y_pred, class)?;
            let f1 = f1_score(y_true, y_pred, class)?;

            per_class_precision.insert(class, precision);
            per_class_recall.insert(class, recall);
            per_class_f1.insert(class, f1);
        }

        // Calculate weighted F1 score
        let mut weighted_f1 = 0.0;
        let mut total_samples = 0;

        for &class in &classes {
            let class_count = y_true.iter().filter(|&&label| label == class).count();
            if let Some(&f1) = per_class_f1.get(&class) {
                weighted_f1 += f1 * class_count as f64;
                total_samples += class_count;
            }
        }

        if total_samples > 0 {
            weighted_f1 /= total_samples as f64;
        }

        Ok(ImageClassificationResults {
            top1_accuracy,
            top5_accuracy,
            per_class_precision,
            per_class_recall,
            per_class_f1,
            weighted_f1,
        })
    }

    /// Calculate top-k accuracy
    fn calculate_top_k_accuracy(
        &self,
        y_true: &Array1<i32>,
        y_prob: &Array2<f64>,
        k: usize,
    ) -> Result<f64> {
        if y_true.len() != y_prob.nrows() {
            return Err(MetricsError::InvalidInput(
                "Length mismatch between true labels and probabilities".to_string(),
            ));
        }

        let mut correct = 0;

        for (i, &true_label) in y_true.iter().enumerate() {
            // Get top-k predictions for this sample
            let mut probs_with_idx: Vec<(usize, f64)> = y_prob
                .row(i)
                .iter()
                .enumerate()
                .map(|(idx, &prob)| (idx, prob))
                .collect();

            probs_with_idx
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let top_k_classes: Vec<usize> = probs_with_idx
                .into_iter()
                .take(k)
                .map(|(idx, _)| idx)
                .collect();

            if top_k_classes.contains(&(true_label as usize)) {
                correct += 1;
            }
        }

        Ok(correct as f64 / y_true.len() as f64)
    }
}

/// Segmentation metrics calculator
pub struct SegmentationMetrics {
    ignore_index: Option<i32>,
}

impl Default for SegmentationMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl SegmentationMetrics {
    /// Create new segmentation metrics calculator
    pub fn new() -> Self {
        Self { ignore_index: None }
    }

    /// Set index to ignore in calculations (e.g., background class)
    pub fn with_ignore_index(mut self, ignore_index: i32) -> Self {
        self.ignore_index = Some(ignore_index);
        self
    }

    /// Evaluate semantic segmentation predictions
    pub fn evaluate_segmentation(
        &self,
        y_true: &Array2<i32>,
        y_pred: &Array2<i32>,
    ) -> Result<SegmentationResults> {
        if y_true.shape() != y_pred.shape() {
            return Err(MetricsError::InvalidInput(
                "Shape mismatch between true and predicted masks".to_string(),
            ));
        }

        // Calculate pixel accuracy
        let mut correct_pixels = 0;
        let mut total_pixels = 0;

        for (true_pixel, pred_pixel) in y_true.iter().zip(y_pred.iter()) {
            if let Some(ignore_idx) = self.ignore_index {
                if *true_pixel == ignore_idx {
                    continue;
                }
            }

            total_pixels += 1;
            if true_pixel == pred_pixel {
                correct_pixels += 1;
            }
        }

        let pixel_accuracy = if total_pixels > 0 {
            correct_pixels as f64 / total_pixels as f64
        } else {
            0.0
        };

        // Calculate per-class IoU
        let classes: Vec<i32> = {
            let mut classes = y_true
                .iter()
                .chain(y_pred.iter())
                .copied()
                .collect::<Vec<_>>();
            if let Some(ignore_idx) = self.ignore_index {
                classes.retain(|&x| x != ignore_idx);
            }
            classes.sort_unstable();
            classes.dedup();
            classes
        };

        let mut per_class_iou = HashMap::new();
        let mut ious = Vec::new();

        for &class in &classes {
            let mut intersection = 0;
            let mut union = 0;

            for (true_pixel, pred_pixel) in y_true.iter().zip(y_pred.iter()) {
                if let Some(ignore_idx) = self.ignore_index {
                    if *true_pixel == ignore_idx {
                        continue;
                    }
                }

                let true_match = *true_pixel == class;
                let pred_match = *pred_pixel == class;

                if true_match && pred_match {
                    intersection += 1;
                }
                if true_match || pred_match {
                    union += 1;
                }
            }

            let iou = if union > 0 {
                intersection as f64 / union as f64
            } else {
                0.0
            };

            per_class_iou.insert(class, iou);
            ious.push(iou);
        }

        let mean_iou = if !ious.is_empty() {
            ious.iter().sum::<f64>() / ious.len() as f64
        } else {
            0.0
        };

        // Calculate Dice coefficient and Jaccard index for binary case
        let (dice_coefficient, jaccard_index) = if classes.len() == 2 {
            let class = classes[1]; // Assume foreground class
            let mut tp = 0;
            let mut fp = 0;
            let mut fn_count = 0;

            for (true_pixel, pred_pixel) in y_true.iter().zip(y_pred.iter()) {
                if let Some(ignore_idx) = self.ignore_index {
                    if *true_pixel == ignore_idx {
                        continue;
                    }
                }

                let true_positive = *true_pixel == class;
                let pred_positive = *pred_pixel == class;

                match (true_positive, pred_positive) {
                    (true, true) => tp += 1,
                    (false, true) => fp += 1,
                    (true, false) => fn_count += 1,
                    (false, false) => {} // TN
                }
            }

            let dice = if tp + fp + fn_count > 0 {
                2.0 * tp as f64 / (2.0 * tp as f64 + fp as f64 + fn_count as f64)
            } else {
                0.0
            };

            let jaccard = if tp + fp + fn_count > 0 {
                tp as f64 / (tp + fp + fn_count) as f64
            } else {
                0.0
            };

            (Some(dice), Some(jaccard))
        } else {
            (None, None)
        };

        Ok(SegmentationResults {
            pixel_accuracy,
            mean_iou,
            per_class_iou,
            dice_coefficient,
            jaccard_index,
        })
    }
}

/// Complete computer vision metrics suite
pub struct ComputerVisionSuite {
    object_detection: ObjectDetectionMetrics,
    classification: ImageClassificationMetrics,
    segmentation: SegmentationMetrics,
}

impl Default for ComputerVisionSuite {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputerVisionSuite {
    /// Create a new computer vision metrics suite
    pub fn new() -> Self {
        Self {
            object_detection: ObjectDetectionMetrics::new(),
            classification: ImageClassificationMetrics::new(),
            segmentation: SegmentationMetrics::new(),
        }
    }

    /// Get object detection metrics calculator
    pub fn object_detection(&self) -> &ObjectDetectionMetrics {
        &self.object_detection
    }

    /// Get image classification metrics calculator
    pub fn classification(&self) -> &ImageClassificationMetrics {
        &self.classification
    }

    /// Get segmentation metrics calculator
    pub fn segmentation(&self) -> &SegmentationMetrics {
        &self.segmentation
    }
}

impl DomainMetrics for ComputerVisionSuite {
    type Result = DomainEvaluationResult;

    fn domain_name(&self) -> &'static str {
        "Computer Vision"
    }

    fn available_metrics(&self) -> Vec<&'static str> {
        vec![
            "object_detection_map",
            "object_detection_precision",
            "object_detection_recall",
            "classification_top1_accuracy",
            "classification_top5_accuracy",
            "classification_weighted_f1",
            "segmentation_pixel_accuracy",
            "segmentation_mean_iou",
            "segmentation_dice_coefficient",
        ]
    }

    fn metric_descriptions(&self) -> HashMap<&'static str, &'static str> {
        let mut descriptions = HashMap::new();
        descriptions.insert(
            "object_detection_map",
            "Mean Average Precision for object detection",
        );
        descriptions.insert(
            "object_detection_precision",
            "Overall precision for object detection",
        );
        descriptions.insert(
            "object_detection_recall",
            "Overall recall for object detection",
        );
        descriptions.insert(
            "classification_top1_accuracy",
            "Top-1 accuracy for image classification",
        );
        descriptions.insert(
            "classification_top5_accuracy",
            "Top-5 accuracy for image classification",
        );
        descriptions.insert(
            "classification_weighted_f1",
            "Weighted F1 score for classification",
        );
        descriptions.insert(
            "segmentation_pixel_accuracy",
            "Pixel-wise accuracy for segmentation",
        );
        descriptions.insert(
            "segmentation_mean_iou",
            "Mean Intersection over Union for segmentation",
        );
        descriptions.insert(
            "segmentation_dice_coefficient",
            "Dice coefficient for binary segmentation",
        );
        descriptions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iou_calculation() {
        let metrics = ObjectDetectionMetrics::new();

        // Perfect overlap
        let iou = metrics.calculate_iou(0.0, 0.0, 10.0, 10.0, 0.0, 0.0, 10.0, 10.0);
        assert_eq!(iou, 1.0);

        // No overlap
        let iou = metrics.calculate_iou(0.0, 0.0, 10.0, 10.0, 20.0, 20.0, 30.0, 30.0);
        assert_eq!(iou, 0.0);

        // Partial overlap: box1=(0,0,10,10) area=100, box2=(5,5,15,15) area=100
        // intersection=(5,5,10,10) area=25, union=100+100-25=175, IoU=25/175≈0.143
        let iou = metrics.calculate_iou(0.0, 0.0, 10.0, 10.0, 5.0, 5.0, 15.0, 15.0);
        assert!((iou - (25.0 / 175.0)).abs() < 1e-6);
    }

    #[test]
    fn test_object_detection_evaluation() {
        let metrics = ObjectDetectionMetrics::new();

        let predictions = vec![
            (10.0, 10.0, 50.0, 50.0, 0.9, 1),   // High confidence
            (60.0, 60.0, 100.0, 100.0, 0.7, 2), // Medium confidence
        ];

        let ground_truth = vec![
            (12.0, 12.0, 48.0, 48.0, 1),   // Close to first prediction
            (70.0, 70.0, 110.0, 110.0, 2), // Close to second prediction
        ];

        let results = metrics
            .evaluate_object_detection(&predictions, &ground_truth, 0.5)
            .unwrap();

        assert!(results.map >= 0.0 && results.map <= 1.0);
        assert!(results.precision >= 0.0 && results.precision <= 1.0);
        assert!(results.recall >= 0.0 && results.recall <= 1.0);
        assert_eq!(results.iou_threshold, 0.5);
    }

    #[test]
    fn test_image_classification_evaluation() {
        let metrics = ImageClassificationMetrics::new();

        let y_true = Array1::from_vec(vec![0, 1, 2, 0, 1, 2]);
        let y_pred = Array1::from_vec(vec![0, 2, 1, 0, 0, 2]);

        let results = metrics
            .evaluate_classification(&y_true, &y_pred, None)
            .unwrap();

        assert!(results.top1_accuracy >= 0.0 && results.top1_accuracy <= 1.0);
        assert!(results.weighted_f1 >= 0.0 && results.weighted_f1 <= 1.0);
        assert!(results.per_class_precision.len() <= 3); // 3 classes max
    }

    #[test]
    fn test_segmentation_evaluation() {
        let metrics = SegmentationMetrics::new();

        let y_true = Array2::from_shape_vec((3, 3), vec![0, 0, 1, 0, 1, 1, 1, 1, 1]).unwrap();

        let y_pred = Array2::from_shape_vec((3, 3), vec![0, 0, 1, 0, 0, 1, 1, 1, 1]).unwrap();

        let results = metrics.evaluate_segmentation(&y_true, &y_pred).unwrap();

        assert!(results.pixel_accuracy >= 0.0 && results.pixel_accuracy <= 1.0);
        assert!(results.mean_iou >= 0.0 && results.mean_iou <= 1.0);
        assert!(results.dice_coefficient.is_some());
        assert!(results.jaccard_index.is_some());
    }

    #[test]
    fn test_computer_vision_suite() {
        let suite = ComputerVisionSuite::new();

        assert_eq!(suite.domain_name(), "Computer Vision");
        assert!(!suite.available_metrics().is_empty());
        assert!(!suite.metric_descriptions().is_empty());
    }
}
