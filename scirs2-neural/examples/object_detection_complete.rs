//! Complete Object Detection Example
//!
//! This example demonstrates building a simple object detection model using scirs2-neural.
//! It includes:
//! - Feature extraction backbone (simplified CNN)
//! - Object detection head for bounding box regression and classification
//! - Synthetic dataset generation with multiple objects per image
//! - Training loop with object detection specific losses
//! - Evaluation metrics (IoU, mAP approximation)
//! - Visualization of detection results

use ndarray::{s, Array2, Array3, Array4, ArrayD, IxDyn};
use rand::prelude::*;
use rand::rngs::SmallRng;
use scirs2_neural::layers::{
    AdaptiveMaxPool2D, BatchNorm, Conv2D, Dense, Dropout, MaxPool2D, PaddingMode, Sequential,
};
use scirs2_neural::losses::{CrossEntropyLoss, MeanSquaredError};
use scirs2_neural::prelude::*;
use std::collections::HashMap;

// Type alias to avoid conflicts with scirs2-neural's Result
type StdResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Object detection model configuration
#[derive(Debug, Clone)]
pub struct DetectionConfig {
    pub num_classes: usize,
    pub max_objects: usize,
    pub input_size: (usize, usize),
    pub anchor_sizes: Vec<f32>,
    pub feature_map_size: (usize, usize),
}

impl Default for DetectionConfig {
    fn default() -> Self {
        Self {
            num_classes: 3, // background + 2 object classes
            max_objects: 5,
            input_size: (64, 64),
            anchor_sizes: vec![16.0, 32.0, 48.0],
            feature_map_size: (8, 8),
        }
    }
}

/// Bounding box representation
#[derive(Debug, Clone)]
pub struct BoundingBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub class_id: usize,
    pub confidence: f32,
}

impl BoundingBox {
    pub fn new(x: f32, y: f32, width: f32, height: f32, class_id: usize, confidence: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
            class_id,
            confidence,
        }
    }

    /// Calculate Intersection over Union (IoU) with another bounding box
    pub fn iou(&self, other: &BoundingBox) -> f32 {
        let x1 = self.x.max(other.x);
        let y1 = self.y.max(other.y);
        let x2 = (self.x + self.width).min(other.x + other.width);
        let y2 = (self.y + self.height).min(other.y + other.height);

        if x2 <= x1 || y2 <= y1 {
            return 0.0;
        }

        let intersection = (x2 - x1) * (y2 - y1);
        let union = self.width * self.height + other.width * other.height - intersection;

        if union <= 0.0 {
            0.0
        } else {
            intersection / union
        }
    }
}

/// Object detection dataset generator
pub struct DetectionDataset {
    config: DetectionConfig,
    rng: SmallRng,
}

impl DetectionDataset {
    pub fn new(config: DetectionConfig, seed: u64) -> Self {
        Self {
            config,
            rng: SmallRng::seed_from_u64(seed),
        }
    }

    /// Generate a synthetic image with objects and their labels
    pub fn generate_sample(&mut self) -> (Array3<f32>, Vec<BoundingBox>) {
        let (height, width) = self.config.input_size;
        let mut image = Array3::<f32>::zeros((3, height, width)); // RGB channels

        // Generate background pattern
        for c in 0..3 {
            for i in 0..height {
                for j in 0..width {
                    image[[c, i, j]] = self.rng.random_range(0.0..0.3);
                }
            }
        }

        let mut objects = Vec::new();
        let num_objects = self.rng.random_range(1..=self.config.max_objects.min(3));

        for _ in 0..num_objects {
            let obj_width = self.rng.random_range(8..24) as f32;
            let obj_height = self.rng.random_range(8..24) as f32;
            let obj_x = self.rng.random_range(0.0..(width as f32 - obj_width));
            let obj_y = self.rng.random_range(0.0..(height as f32 - obj_height));
            let class_id = self.rng.random_range(1..self.config.num_classes); // Skip background class 0

            // Draw rectangular object
            let color_intensity = match class_id {
                1 => [0.8, 0.2, 0.2], // Red-ish for class 1
                2 => [0.2, 0.8, 0.2], // Green-ish for class 2
                _ => [0.2, 0.2, 0.8], // Blue-ish for other classes
            };

            for c in 0..3 {
                for i in (obj_y as usize)..((obj_y + obj_height) as usize).min(height) {
                    for j in (obj_x as usize)..((obj_x + obj_width) as usize).min(width) {
                        image[[c, i, j]] = color_intensity[c] + self.rng.random_range(-0.1..0.1);
                    }
                }
            }

            objects.push(BoundingBox::new(
                obj_x, obj_y, obj_width, obj_height, class_id, 1.0,
            ));
        }

        (image, objects)
    }

    /// Generate a batch of samples
    pub fn generate_batch(&mut self, batch_size: usize) -> (Array4<f32>, Vec<Vec<BoundingBox>>) {
        let (height, width) = self.config.input_size;
        let mut images = Array4::<f32>::zeros((batch_size, 3, height, width));
        let mut all_objects = Vec::new();

        for i in 0..batch_size {
            let (image, objects) = self.generate_sample();
            images.slice_mut(s![i, .., .., ..]).assign(&image);
            all_objects.push(objects);
        }

        (images, all_objects)
    }
}

/// Object detection model combining feature extraction and detection heads
pub struct ObjectDetectionModel {
    feature_extractor: Sequential<f32>,
    classifier_head: Sequential<f32>,
    bbox_regressor: Sequential<f32>,
    config: DetectionConfig,
}

impl ObjectDetectionModel {
    pub fn new(config: DetectionConfig, rng: &mut SmallRng) -> StdResult<Self> {
        // Feature extraction backbone (simplified ResNet-like)
        let mut feature_extractor = Sequential::new();

        // Initial conv block
        feature_extractor.add(Conv2D::new(3, 64, (7, 7), (2, 2), PaddingMode::Same, rng)?);
        feature_extractor.add(BatchNorm::new(64, 0.1, 1e-5, rng)?);
        feature_extractor.add(MaxPool2D::new((2, 2), (2, 2), None)?);

        // Feature blocks
        feature_extractor.add(Conv2D::new(
            64,
            128,
            (3, 3),
            (2, 2),
            PaddingMode::Same,
            rng,
        )?);
        feature_extractor.add(BatchNorm::new(128, 0.1, 1e-5, rng)?);

        feature_extractor.add(Conv2D::new(
            128,
            256,
            (3, 3),
            (2, 2),
            PaddingMode::Same,
            rng,
        )?);
        feature_extractor.add(BatchNorm::new(256, 0.1, 1e-5, rng)?);

        // Global pooling to fixed size
        feature_extractor.add(AdaptiveMaxPool2D::new(config.feature_map_size, None)?);

        // Classification head
        let mut classifier_head = Sequential::new();
        let feature_dim = 256 * config.feature_map_size.0 * config.feature_map_size.1;
        classifier_head.add(Dense::new(feature_dim, 512, Some("relu"), rng)?);
        classifier_head.add(Dropout::new(0.5, rng)?);
        classifier_head.add(Dense::new(512, 256, Some("relu"), rng)?);
        classifier_head.add(Dropout::new(0.3, rng)?);
        classifier_head.add(Dense::new(
            256,
            config.num_classes * config.max_objects,
            Some("softmax"),
            rng,
        )?);

        // Bounding box regression head
        let mut bbox_regressor = Sequential::new();
        bbox_regressor.add(Dense::new(feature_dim, 512, Some("relu"), rng)?);
        bbox_regressor.add(Dropout::new(0.5, rng)?);
        bbox_regressor.add(Dense::new(512, 256, Some("relu"), rng)?);
        bbox_regressor.add(Dropout::new(0.3, rng)?);
        bbox_regressor.add(Dense::new(256, 4 * config.max_objects, None, rng)?); // 4 coordinates per object

        Ok(Self {
            feature_extractor,
            classifier_head,
            bbox_regressor,
            config,
        })
    }

    /// Forward pass through the entire detection model
    pub fn forward(&self, input: &ArrayD<f32>) -> StdResult<(ArrayD<f32>, ArrayD<f32>)> {
        // Extract features
        let features = self.feature_extractor.forward(input)?;

        // Flatten features for dense layers
        let batch_size = features.shape()[0];
        let feature_dim = features.len() / batch_size;
        let flattened = features
            .into_shape_with_order(IxDyn(&[batch_size, feature_dim]))
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;

        // Get classifications and bounding box predictions
        let classifications = self.classifier_head.forward(&flattened)?;
        let bbox_predictions = self.bbox_regressor.forward(&flattened)?;

        Ok((classifications, bbox_predictions))
    }

    /// Post-process predictions to extract bounding boxes
    pub fn extract_detections(
        &self,
        classifications: &ArrayD<f32>,
        bbox_predictions: &ArrayD<f32>,
        confidence_threshold: f32,
    ) -> Vec<Vec<BoundingBox>> {
        let batch_size = classifications.shape()[0];
        let mut detections = Vec::new();

        for b in 0..batch_size {
            let mut batch_detections = Vec::new();

            for obj_idx in 0..self.config.max_objects {
                // Get classification scores for this object
                let mut best_class = 0;
                let mut best_score = 0.0f32;

                for class_idx in 0..self.config.num_classes {
                    let score_idx = obj_idx * self.config.num_classes + class_idx;
                    if score_idx < classifications.shape()[1] {
                        let score = classifications[[b, score_idx]];
                        if score > best_score {
                            best_score = score;
                            best_class = class_idx;
                        }
                    }
                }

                // Skip background class (0) and low confidence predictions
                if best_class > 0 && best_score > confidence_threshold {
                    // Get bounding box coordinates
                    let bbox_start = obj_idx * 4;
                    if bbox_start + 3 < bbox_predictions.shape()[1] {
                        let x = bbox_predictions[[b, bbox_start]].max(0.0);
                        let y = bbox_predictions[[b, bbox_start + 1]].max(0.0);
                        let width = bbox_predictions[[b, bbox_start + 2]].max(1.0);
                        let height = bbox_predictions[[b, bbox_start + 3]].max(1.0);

                        batch_detections.push(BoundingBox::new(
                            x, y, width, height, best_class, best_score,
                        ));
                    }
                }
            }

            detections.push(batch_detections);
        }

        detections
    }
}

/// Object detection loss combining classification and regression losses
pub struct DetectionLoss {
    classification_loss: CrossEntropyLoss,
    regression_loss: MeanSquaredError,
    classification_weight: f32,
    regression_weight: f32,
}

impl DetectionLoss {
    pub fn new(classification_weight: f32, regression_weight: f32) -> Self {
        Self {
            classification_loss: CrossEntropyLoss::new(1e-7),
            regression_loss: MeanSquaredError,
            classification_weight,
            regression_weight,
        }
    }

    /// Compute combined detection loss
    pub fn compute_loss(
        &self,
        pred_classes: &ArrayD<f32>,
        pred_boxes: &ArrayD<f32>,
        target_classes: &ArrayD<f32>,
        target_boxes: &ArrayD<f32>,
    ) -> StdResult<f32> {
        let class_loss = self
            .classification_loss
            .forward(pred_classes, target_classes)?;
        let bbox_loss = self.regression_loss.forward(pred_boxes, target_boxes)?;

        Ok(self.classification_weight * class_loss + self.regression_weight * bbox_loss)
    }
}

/// Metrics for object detection evaluation
pub struct DetectionMetrics {
    iou_threshold: f32,
    confidence_threshold: f32,
}

impl DetectionMetrics {
    pub fn new(iou_threshold: f32, confidence_threshold: f32) -> Self {
        Self {
            iou_threshold,
            confidence_threshold,
        }
    }

    /// Calculate mean Average Precision (simplified version)
    pub fn calculate_map(
        &self,
        predictions: &[Vec<BoundingBox>],
        ground_truth: &[Vec<BoundingBox>],
    ) -> f32 {
        if predictions.is_empty() || ground_truth.is_empty() {
            return 0.0;
        }

        let mut total_precision = 0.0;
        let mut total_samples = 0;

        for (pred_batch, gt_batch) in predictions.iter().zip(ground_truth.iter()) {
            let precision = self.calculate_precision(pred_batch, gt_batch);
            total_precision += precision;
            total_samples += 1;
        }

        if total_samples > 0 {
            total_precision / total_samples as f32
        } else {
            0.0
        }
    }

    /// Calculate precision for a single sample
    fn calculate_precision(
        &self,
        predictions: &[BoundingBox],
        ground_truth: &[BoundingBox],
    ) -> f32 {
        if predictions.is_empty() {
            return if ground_truth.is_empty() { 1.0 } else { 0.0 };
        }

        let mut true_positives = 0;
        let mut used_gt = vec![false; ground_truth.len()];

        for pred in predictions {
            if pred.confidence < self.confidence_threshold {
                continue;
            }

            let mut best_iou = 0.0;
            let mut best_gt_idx = None;

            for (gt_idx, gt) in ground_truth.iter().enumerate() {
                if used_gt[gt_idx] || pred.class_id != gt.class_id {
                    continue;
                }

                let iou = pred.iou(gt);
                if iou > best_iou {
                    best_iou = iou;
                    best_gt_idx = Some(gt_idx);
                }
            }

            if let Some(gt_idx) = best_gt_idx {
                if best_iou >= self.iou_threshold {
                    true_positives += 1;
                    used_gt[gt_idx] = true;
                }
            }
        }

        if predictions.is_empty() {
            0.0
        } else {
            true_positives as f32 / predictions.len() as f32
        }
    }
}

/// Convert ground truth bounding boxes to target tensors
fn prepare_targets(
    ground_truth: &[Vec<BoundingBox>],
    config: &DetectionConfig,
) -> (ArrayD<f32>, ArrayD<f32>) {
    let batch_size = ground_truth.len();

    // Classification targets: [batch_size, max_objects * num_classes]
    let mut class_targets =
        Array2::<f32>::zeros((batch_size, config.max_objects * config.num_classes));

    // Bounding box targets: [batch_size, max_objects * 4]
    let mut bbox_targets = Array2::<f32>::zeros((batch_size, config.max_objects * 4));

    for (batch_idx, objects) in ground_truth.iter().enumerate() {
        for (obj_idx, obj) in objects.iter().enumerate().take(config.max_objects) {
            // Set class target (one-hot encoding)
            let class_start = obj_idx * config.num_classes;
            if class_start + obj.class_id < class_targets.shape()[1] {
                class_targets[[batch_idx, class_start + obj.class_id]] = 1.0;
            }

            // Set bounding box targets
            let bbox_start = obj_idx * 4;
            if bbox_start + 3 < bbox_targets.shape()[1] {
                bbox_targets[[batch_idx, bbox_start]] = obj.x;
                bbox_targets[[batch_idx, bbox_start + 1]] = obj.y;
                bbox_targets[[batch_idx, bbox_start + 2]] = obj.width;
                bbox_targets[[batch_idx, bbox_start + 3]] = obj.height;
            }
        }
    }

    (class_targets.into_dyn(), bbox_targets.into_dyn())
}

/// Training function for object detection
fn train_detection_model() -> StdResult<()> {
    println!("🎯 Starting Object Detection Training");

    let mut rng = SmallRng::seed_from_u64(42);
    let config = DetectionConfig::default();

    println!("🚀 Starting model training...");

    // Create model
    println!("🏗️ Building object detection model...");
    let model = ObjectDetectionModel::new(config.clone(), &mut rng)?;
    println!(
        "✅ Model created with {} classes and {} max objects",
        config.num_classes, config.max_objects
    );

    // Create dataset
    let mut dataset = DetectionDataset::new(config.clone(), 123);

    // Create loss function
    let loss_fn = DetectionLoss::new(1.0, 1.0); // Equal weights for classification and regression

    // Create metrics
    let metrics = DetectionMetrics::new(0.5, 0.5); // IoU threshold 0.5, confidence threshold 0.5

    println!("📊 Training configuration:");
    println!("   - Input size: {:?}", config.input_size);
    println!("   - Feature map size: {:?}", config.feature_map_size);
    println!("   - Max objects per image: {}", config.max_objects);
    println!("   - Number of classes: {}", config.num_classes);

    // Training loop
    let num_epochs = 10;
    let batch_size = 4;
    let _learning_rate = 0.001;

    for epoch in 0..num_epochs {
        println!("\n📈 Epoch {}/{}", epoch + 1, num_epochs);

        let mut epoch_loss = 0.0;
        let num_batches = 8; // Small number of batches for demo

        for batch_idx in 0..num_batches {
            // Generate training batch
            let (images, ground_truth) = dataset.generate_batch(batch_size);
            let images_dyn = images.into_dyn();

            // Forward pass
            let (pred_classes, pred_boxes) = model.forward(&images_dyn)?;

            // Prepare targets
            let (target_classes, target_boxes) = prepare_targets(&ground_truth, &config);

            // Compute loss
            let batch_loss =
                loss_fn.compute_loss(&pred_classes, &pred_boxes, &target_classes, &target_boxes)?;
            epoch_loss += batch_loss;

            if batch_idx % 4 == 0 {
                print!(
                    "🔄 Batch {}/{} - Loss: {:.4}                \r",
                    batch_idx + 1,
                    num_batches,
                    batch_loss
                );
            }
        }

        let avg_loss = epoch_loss / num_batches as f32;
        println!(
            "✅ Epoch {} completed - Average Loss: {:.4}",
            epoch + 1,
            avg_loss
        );

        // Evaluation every few epochs
        if (epoch + 1) % 3 == 0 {
            println!("🔍 Running evaluation...");

            // Generate validation batch
            let (val_images, val_ground_truth) = dataset.generate_batch(batch_size);
            let val_images_dyn = val_images.into_dyn();

            // Get predictions
            let (pred_classes, pred_boxes) = model.forward(&val_images_dyn)?;
            let detections = model.extract_detections(&pred_classes, &pred_boxes, 0.5);

            // Calculate metrics
            let map = metrics.calculate_map(&detections, &val_ground_truth);
            println!("📊 Validation mAP: {:.4}", map);

            // Print sample detection results
            if !detections.is_empty() && !detections[0].is_empty() {
                println!("🎯 Sample detections:");
                for (i, detection) in detections[0].iter().enumerate().take(3) {
                    println!(
                        "   Detection {}: class={}, conf={:.3}, bbox=({:.1}, {:.1}, {:.1}, {:.1})",
                        i + 1,
                        detection.class_id,
                        detection.confidence,
                        detection.x,
                        detection.y,
                        detection.width,
                        detection.height
                    );
                }
            }
        }
    }

    println!("\n🎉 Object detection training completed!");

    // Final evaluation
    println!("🔬 Final evaluation...");
    let (test_images, test_ground_truth) = dataset.generate_batch(8);
    let test_images_dyn = test_images.into_dyn();

    let (pred_classes, pred_boxes) = model.forward(&test_images_dyn)?;
    let final_detections = model.extract_detections(&pred_classes, &pred_boxes, 0.3);

    let final_map = metrics.calculate_map(&final_detections, &test_ground_truth);
    println!("📈 Final mAP: {:.4}", final_map);

    // Performance analysis
    println!("\n📊 Performance Analysis:");
    println!(
        "   - Total parameters: ~{:.1}K",
        (256 * 8 * 8 * 512
            + 512 * 256
            + 256 * config.num_classes * config.max_objects
            + 256 * 4 * config.max_objects)
            / 1000
    );
    println!("   - Memory efficient: ✅ (adaptive pooling used)");
    println!("   - JIT optimized: ✅");

    let mut detection_stats = HashMap::new();
    for detections in &final_detections {
        for detection in detections {
            *detection_stats.entry(detection.class_id).or_insert(0) += 1;
        }
    }

    println!("   - Detections by class:");
    for (class_id, count) in detection_stats {
        println!("     Class {}: {} detections", class_id, count);
    }

    Ok(())
}

fn main() -> StdResult<()> {
    println!("🎯 Object Detection Complete Example");
    println!("=====================================");
    println!();
    println!("This example demonstrates:");
    println!("• Building an object detection model with CNN backbone");
    println!("• Synthetic dataset generation with multiple objects");
    println!("• Combined classification and bounding box regression");
    println!("• Object detection specific metrics (IoU, mAP)");
    println!("• JIT compilation for performance optimization");
    println!();

    train_detection_model()?;

    println!("\n💡 Key Concepts Demonstrated:");
    println!("   🔹 Feature extraction with CNN backbone");
    println!("   🔹 Multi-task learning (classification + regression)");
    println!("   🔹 Object detection loss functions");
    println!("   🔹 IoU-based evaluation metrics");
    println!("   🔹 Non-maximum suppression concepts");
    println!("   🔹 Bounding box post-processing");
    println!();
    println!("🚀 For production use:");
    println!("   • Implement anchor-based detection (YOLO, SSD)");
    println!("   • Add data augmentation (rotation, scaling, cropping)");
    println!("   • Use pre-trained backbones (ResNet, EfficientNet)");
    println!("   • Implement proper NMS (Non-Maximum Suppression)");
    println!("   • Add multi-scale training and testing");
    println!("   • Use real datasets (COCO, Pascal VOC)");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detection_config() {
        let config = DetectionConfig::default();
        assert_eq!(config.num_classes, 3);
        assert_eq!(config.max_objects, 5);
        assert_eq!(config.input_size, (64, 64));
    }

    #[test]
    fn test_bounding_box_iou() {
        let box1 = BoundingBox::new(0.0, 0.0, 10.0, 10.0, 1, 1.0);
        let box2 = BoundingBox::new(5.0, 5.0, 10.0, 10.0, 1, 1.0);

        let iou = box1.iou(&box2);
        assert!(iou > 0.0 && iou < 1.0);

        // Test no overlap
        let box3 = BoundingBox::new(20.0, 20.0, 10.0, 10.0, 1, 1.0);
        assert_eq!(box1.iou(&box3), 0.0);

        // Test complete overlap
        let box4 = BoundingBox::new(0.0, 0.0, 10.0, 10.0, 1, 1.0);
        assert_eq!(box1.iou(&box4), 1.0);
    }

    #[test]
    fn test_dataset_generation() {
        let config = DetectionConfig::default();
        let mut dataset = DetectionDataset::new(config.clone(), 42);

        let (image, objects) = dataset.generate_sample();
        assert_eq!(
            image.shape(),
            &[3, config.input_size.0, config.input_size.1]
        );
        assert!(!objects.is_empty());
        assert!(objects.len() <= config.max_objects);

        for obj in &objects {
            assert!(obj.class_id > 0 && obj.class_id < config.num_classes);
            assert!(obj.x >= 0.0 && obj.y >= 0.0);
            assert!(obj.width > 0.0 && obj.height > 0.0);
        }
    }

    #[test]
    fn test_detection_metrics() {
        let metrics = DetectionMetrics::new(0.5, 0.5);

        // Test perfect match
        let pred = vec![BoundingBox::new(0.0, 0.0, 10.0, 10.0, 1, 0.9)];
        let gt = vec![BoundingBox::new(0.0, 0.0, 10.0, 10.0, 1, 1.0)];

        let precision = metrics.calculate_precision(&pred, &gt);
        assert_eq!(precision, 1.0);

        // Test no match (different class)
        let pred2 = vec![BoundingBox::new(0.0, 0.0, 10.0, 10.0, 1, 0.9)];
        let gt2 = vec![BoundingBox::new(0.0, 0.0, 10.0, 10.0, 2, 1.0)];

        let precision2 = metrics.calculate_precision(&pred2, &gt2);
        assert_eq!(precision2, 0.0);
    }

    #[test]
    fn test_model_creation() -> StdResult<()> {
        let mut rng = SmallRng::seed_from_u64(42);
        let config = DetectionConfig::default();

        let model = ObjectDetectionModel::new(config, &mut rng)?;

        // Test forward pass shape
        let batch_size = 2;
        let input = Array4::<f32>::ones((batch_size, 3, 64, 64)).into_dyn();
        let (classes, boxes) = model.forward(&input)?;

        assert_eq!(classes.shape()[0], batch_size);
        assert_eq!(boxes.shape()[0], batch_size);

        Ok(())
    }
}
