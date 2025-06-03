//! Non-maximum suppression algorithms
//!
//! This module provides various non-maximum suppression techniques
//! commonly used in computer vision for peak detection and object detection.

use ndarray::{Array2, ArrayView2};
use std::cmp::Ordering;

/// Bounding box representation
#[derive(Debug, Clone, Copy)]
pub struct BoundingBox {
    /// X coordinate of top-left corner
    pub x: f32,
    /// Y coordinate of top-left corner  
    pub y: f32,
    /// Width of the box
    pub width: f32,
    /// Height of the box
    pub height: f32,
    /// Confidence score
    pub score: f32,
    /// Optional class ID
    pub class_id: Option<usize>,
}

impl BoundingBox {
    /// Create a new bounding box
    pub fn new(x: f32, y: f32, width: f32, height: f32, score: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
            score,
            class_id: None,
        }
    }

    /// Create with class ID
    pub fn with_class(
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        score: f32,
        class_id: usize,
    ) -> Self {
        Self {
            x,
            y,
            width,
            height,
            score,
            class_id: Some(class_id),
        }
    }

    /// Calculate area of the bounding box
    pub fn area(&self) -> f32 {
        self.width * self.height
    }

    /// Calculate intersection over union with another box
    pub fn iou(&self, other: &BoundingBox) -> f32 {
        let x1 = self.x.max(other.x);
        let y1 = self.y.max(other.y);
        let x2 = (self.x + self.width).min(other.x + other.width);
        let y2 = (self.y + self.height).min(other.y + other.height);

        if x2 < x1 || y2 < y1 {
            return 0.0;
        }

        let intersection = (x2 - x1) * (y2 - y1);
        let union = self.area() + other.area() - intersection;

        if union > 0.0 {
            intersection / union
        } else {
            0.0
        }
    }
}

/// Perform non-maximum suppression on bounding boxes
///
/// # Arguments
///
/// * `boxes` - Vector of bounding boxes
/// * `iou_threshold` - IoU threshold for suppression
///
/// # Returns
///
/// * Vector of indices of boxes to keep
pub fn nms_boxes(boxes: &[BoundingBox], iou_threshold: f32) -> Vec<usize> {
    if boxes.is_empty() {
        return vec![];
    }

    // Sort boxes by score in descending order
    let mut indices: Vec<usize> = (0..boxes.len()).collect();
    indices.sort_by(|&a, &b| {
        boxes[b]
            .score
            .partial_cmp(&boxes[a].score)
            .unwrap_or(Ordering::Equal)
    });

    let mut keep = Vec::new();
    let mut suppressed = vec![false; boxes.len()];

    for &idx in &indices {
        if suppressed[idx] {
            continue;
        }

        keep.push(idx);

        // Suppress boxes with high IoU
        for &other_idx in &indices {
            if other_idx != idx && !suppressed[other_idx] {
                let iou = boxes[idx].iou(&boxes[other_idx]);
                if iou > iou_threshold {
                    suppressed[other_idx] = true;
                }
            }
        }
    }

    keep
}

/// Perform class-aware NMS
///
/// Only suppress boxes of the same class
pub fn nms_boxes_class_aware(boxes: &[BoundingBox], iou_threshold: f32) -> Vec<usize> {
    if boxes.is_empty() {
        return vec![];
    }

    // Group boxes by class
    let mut class_groups: std::collections::HashMap<Option<usize>, Vec<usize>> =
        std::collections::HashMap::new();

    for (idx, box_) in boxes.iter().enumerate() {
        class_groups.entry(box_.class_id).or_default().push(idx);
    }

    let mut keep = Vec::new();

    // Apply NMS within each class
    for indices in class_groups.values() {
        let class_boxes: Vec<BoundingBox> = indices.iter().map(|&idx| boxes[idx]).collect();

        let kept_in_class = nms_boxes(&class_boxes, iou_threshold);

        for kept_idx in kept_in_class {
            keep.push(indices[kept_idx]);
        }
    }

    keep.sort_unstable();
    keep
}

/// Perform soft-NMS on bounding boxes
///
/// Instead of removing boxes, reduce their scores based on overlap
///
/// # Arguments
///
/// * `boxes` - Mutable vector of bounding boxes
/// * `iou_threshold` - IoU threshold for score reduction
/// * `sigma` - Gaussian parameter for score reduction
/// * `score_threshold` - Minimum score threshold
pub fn soft_nms(
    boxes: &mut [BoundingBox],
    iou_threshold: f32,
    sigma: f32,
    score_threshold: f32,
) -> Vec<usize> {
    if boxes.is_empty() {
        return vec![];
    }

    let mut indices: Vec<usize> = (0..boxes.len()).collect();
    let mut keep = Vec::new();

    while !indices.is_empty() {
        // Find box with highest score
        let max_idx = indices
            .iter()
            .max_by(|&&a, &&b| {
                boxes[a]
                    .score
                    .partial_cmp(&boxes[b].score)
                    .unwrap_or(Ordering::Equal)
            })
            .copied()
            .unwrap();

        let max_pos = indices.iter().position(|&x| x == max_idx).unwrap();
        indices.remove(max_pos);

        if boxes[max_idx].score >= score_threshold {
            keep.push(max_idx);
        }

        // Update scores of remaining boxes
        let indices_to_update = indices.clone();
        for &idx in &indices_to_update {
            let iou = boxes[max_idx].iou(&boxes[idx]);

            if iou > iou_threshold {
                // Gaussian weight
                boxes[idx].score *= (-iou * iou / sigma).exp();

                if boxes[idx].score < score_threshold {
                    indices.retain(|&x| x != idx);
                }
            }
        }
    }

    keep
}

/// Non-maximum suppression for 2D response maps
///
/// # Arguments
///
/// * `response` - 2D array of response values
/// * `window_size` - Size of suppression window
/// * `threshold` - Minimum response threshold
///
/// # Returns
///
/// * Vector of (x, y, response) tuples for local maxima
pub fn nms_2d(
    response: &ArrayView2<f32>,
    window_size: usize,
    threshold: f32,
) -> Vec<(usize, usize, f32)> {
    let (height, width) = response.dim();
    let half_window = window_size / 2;
    let mut peaks = Vec::new();

    for y in half_window..(height - half_window) {
        for x in half_window..(width - half_window) {
            let value = response[[y, x]];

            if value < threshold {
                continue;
            }

            // Check if it's a local maximum
            let mut is_max = true;
            'window: for dy in 0..window_size {
                for dx in 0..window_size {
                    let ny = y + dy - half_window;
                    let nx = x + dx - half_window;

                    if (ny != y || nx != x) && response[[ny, nx]] >= value {
                        is_max = false;
                        break 'window;
                    }
                }
            }

            if is_max {
                peaks.push((x, y, value));
            }
        }
    }

    peaks
}

/// Fast NMS for 2D using maximum filter
///
/// More efficient implementation using maximum filtering
pub fn fast_nms_2d(
    response: &ArrayView2<f32>,
    window_size: usize,
    threshold: f32,
) -> Vec<(usize, usize, f32)> {
    let (height, width) = response.dim();
    let half_window = window_size / 2;

    // Create maximum filtered version
    let mut max_filtered = Array2::zeros((height, width));

    for y in 0..height {
        for x in 0..width {
            let mut max_val = f32::NEG_INFINITY;

            for dy in y.saturating_sub(half_window)..=(y + half_window).min(height - 1) {
                for dx in x.saturating_sub(half_window)..=(x + half_window).min(width - 1) {
                    max_val = max_val.max(response[[dy, dx]]);
                }
            }

            max_filtered[[y, x]] = max_val;
        }
    }

    // Find peaks where original equals max filtered
    let mut peaks = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let value = response[[y, x]];
            if value >= threshold && value == max_filtered[[y, x]] {
                peaks.push((x, y, value));
            }
        }
    }

    peaks
}

/// Oriented NMS for edge detection
///
/// Performs NMS along the gradient direction
pub fn oriented_nms(
    magnitude: &ArrayView2<f32>,
    orientation: &ArrayView2<f32>,
    threshold: f32,
) -> Array2<f32> {
    let (height, width) = magnitude.dim();
    let mut suppressed = Array2::zeros((height, width));

    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let mag = magnitude[[y, x]];

            if mag < threshold {
                continue;
            }

            let angle = orientation[[y, x]];

            // Quantize angle to 8 directions
            let direction =
                ((angle + std::f32::consts::PI) / (std::f32::consts::PI / 4.0)) as i32 % 8;

            let (dx1, dy1, dx2, dy2) = match direction {
                0 | 4 => (1, 0, -1, 0),  // Horizontal
                1 | 5 => (1, 1, -1, -1), // Diagonal /
                2 | 6 => (0, 1, 0, -1),  // Vertical
                3 | 7 => (-1, 1, 1, -1), // Diagonal \
                _ => (1, 0, -1, 0),
            };

            let mag1 = magnitude[[(y as i32 + dy1) as usize, (x as i32 + dx1) as usize]];
            let mag2 = magnitude[[(y as i32 + dy2) as usize, (x as i32 + dx2) as usize]];

            if mag >= mag1 && mag >= mag2 {
                suppressed[[y, x]] = mag;
            }
        }
    }

    suppressed
}

/// Scale-space NMS
///
/// Performs NMS across multiple scales
pub fn scale_space_nms(
    responses: &[Array2<f32>],
    window_size: usize,
    threshold: f32,
) -> Vec<(usize, usize, usize, f32)> {
    let mut all_peaks = Vec::new();

    for (scale, response) in responses.iter().enumerate() {
        let peaks = nms_2d(&response.view(), window_size, threshold);

        for (x, y, value) in peaks {
            // Check if it's also a maximum across scales
            let mut is_scale_max = true;

            if scale > 0 && responses[scale - 1][[y, x]] >= value {
                is_scale_max = false;
            }

            if scale < responses.len() - 1 && responses[scale + 1][[y, x]] >= value {
                is_scale_max = false;
            }

            if is_scale_max {
                all_peaks.push((x, y, scale, value));
            }
        }
    }

    all_peaks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounding_box_iou() {
        let box1 = BoundingBox::new(0.0, 0.0, 10.0, 10.0, 0.9);
        let box2 = BoundingBox::new(5.0, 5.0, 10.0, 10.0, 0.8);

        let iou = box1.iou(&box2);
        assert!(iou > 0.0 && iou < 1.0);

        // No overlap
        let box3 = BoundingBox::new(20.0, 20.0, 10.0, 10.0, 0.7);
        assert_eq!(box1.iou(&box3), 0.0);
    }

    #[test]
    fn test_nms_boxes() {
        let boxes = vec![
            BoundingBox::new(0.0, 0.0, 10.0, 10.0, 0.9),
            BoundingBox::new(1.0, 1.0, 10.0, 10.0, 0.8),
            BoundingBox::new(20.0, 20.0, 10.0, 10.0, 0.95),
        ];

        let keep = nms_boxes(&boxes, 0.5);
        assert_eq!(keep.len(), 2); // Should keep the highest scoring and non-overlapping
    }

    #[test]
    fn test_nms_2d() {
        let mut response = Array2::zeros((10, 10));
        response[[5, 5]] = 1.0;
        response[[2, 2]] = 0.8;
        response[[2, 3]] = 0.6; // Should be suppressed

        let peaks = nms_2d(&response.view(), 3, 0.5);
        assert_eq!(peaks.len(), 2);
    }

    #[test]
    fn test_soft_nms() {
        let mut boxes = vec![
            BoundingBox::new(0.0, 0.0, 10.0, 10.0, 0.9),
            BoundingBox::new(1.0, 1.0, 10.0, 10.0, 0.8),
            BoundingBox::new(20.0, 20.0, 10.0, 10.0, 0.95),
        ];

        let keep = soft_nms(&mut boxes, 0.5, 0.5, 0.3);
        assert!(keep.len() >= 2);

        // Check that overlapping box has reduced score
        assert!(boxes[1].score < 0.8);
    }
}
