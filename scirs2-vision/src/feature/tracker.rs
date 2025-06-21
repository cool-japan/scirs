//! Feature tracking algorithms for motion analysis
//!
//! This module provides algorithms for tracking feature points across multiple frames,
//! building on optical flow and feature detection capabilities.

use crate::error::Result;
use crate::feature::{lucas_kanade_flow, LucasKanadeParams};
use image::{DynamicImage, GrayImage};
use std::collections::HashMap;

/// A tracked feature point
#[derive(Debug, Clone, Copy)]
pub struct TrackedFeature {
    /// Current position
    pub position: (f32, f32),
    /// Previous position
    pub prev_position: (f32, f32),
    /// Velocity (pixels per frame)
    pub velocity: (f32, f32),
    /// Tracking confidence (0.0 to 1.0)
    pub confidence: f32,
    /// Age of the track (number of frames)
    pub age: u32,
    /// Feature ID for consistent tracking
    pub id: u32,
}

/// Parameters for feature tracking
#[derive(Debug, Clone)]
pub struct TrackerParams {
    /// Maximum number of features to track
    pub max_features: usize,
    /// Minimum tracking confidence to maintain
    pub min_confidence: f32,
    /// Maximum distance for feature association
    pub max_distance: f32,
    /// Parameters for optical flow computation
    pub flow_params: LucasKanadeParams,
    /// Enable backwards flow check for robustness
    pub use_backwards_check: bool,
    /// Threshold for backwards flow error
    pub backwards_threshold: f32,
    /// Minimum feature quality for new detections
    pub min_feature_quality: f32,
}

impl Default for TrackerParams {
    fn default() -> Self {
        Self {
            max_features: 500,
            min_confidence: 0.3,
            max_distance: 50.0,
            flow_params: LucasKanadeParams::default(),
            use_backwards_check: true,
            backwards_threshold: 1.0,
            min_feature_quality: 0.01,
        }
    }
}

/// Lucas-Kanade feature tracker
pub struct LKTracker {
    /// Current tracked features
    features: Vec<TrackedFeature>,
    /// Parameters
    params: TrackerParams,
    /// Previous frame
    prev_frame: Option<GrayImage>,
    /// Next feature ID
    next_id: u32,
    /// Feature positions for optical flow
    feature_points: Vec<(f32, f32)>,
}

impl LKTracker {
    /// Create a new LK tracker
    ///
    /// # Arguments
    ///
    /// * `params` - Tracking parameters
    ///
    /// # Returns
    ///
    /// * New tracker instance
    ///
    /// # Example
    ///
    /// ```rust
    /// use scirs2_vision::feature::{LKTracker, TrackerParams};
    ///
    /// let tracker = LKTracker::new(TrackerParams::default());
    /// ```
    pub fn new(params: TrackerParams) -> Self {
        Self {
            features: Vec::new(),
            params,
            prev_frame: None,
            next_id: 0,
            feature_points: Vec::new(),
        }
    }

    /// Update tracker with new frame
    ///
    /// # Arguments
    ///
    /// * `frame` - New frame to process
    /// * `new_features` - Optional new features to add
    ///
    /// # Returns
    ///
    /// * Result containing updated tracked features
    ///
    /// # Example
    ///
    /// ```rust
    /// use scirs2_vision::feature::{LKTracker, TrackerParams};
    /// use image::{DynamicImage, RgbImage};
    ///
    /// # fn main() -> scirs2_vision::error::Result<()> {
    /// let mut tracker = LKTracker::new(TrackerParams::default());
    /// let frame = DynamicImage::ImageRgb8(RgbImage::new(640, 480));
    /// let features = tracker.update(&frame, None)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn update(
        &mut self,
        frame: &DynamicImage,
        new_features: Option<&[(f32, f32)]>,
    ) -> Result<&[TrackedFeature]> {
        let gray_frame = frame.to_luma8();

        if let Some(prev_frame) = self.prev_frame.clone() {
            // Track existing features
            self.track_features(&prev_frame, &gray_frame)?;
        }

        // Add new features if provided and we have capacity
        if let Some(features) = new_features {
            self.add_new_features(features);
        }

        // Update previous frame
        self.prev_frame = Some(gray_frame);

        // Update feature points for next iteration
        self.feature_points = self.features.iter().map(|f| f.position).collect();

        Ok(&self.features)
    }

    /// Track existing features using optical flow
    fn track_features(&mut self, prev_frame: &GrayImage, curr_frame: &GrayImage) -> Result<()> {
        if self.features.is_empty() {
            return Ok(());
        }

        // Prepare points for optical flow
        let points: Vec<(f32, f32)> = self.features.iter().map(|f| f.position).collect();

        // Compute forward flow
        let forward_flow = lucas_kanade_flow(
            &DynamicImage::ImageLuma8(prev_frame.clone()),
            &DynamicImage::ImageLuma8(curr_frame.clone()),
            Some(&points),
            &self.params.flow_params,
        )?;

        let mut tracked_features = Vec::new();

        for feature in self.features.iter() {
            let x = feature.position.0 as usize;
            let y = feature.position.1 as usize;

            // Check bounds
            if y >= forward_flow.nrows() || x >= forward_flow.ncols() {
                continue;
            }

            let flow_vec = &forward_flow[[y, x]];
            let new_position = (
                feature.position.0 + flow_vec.u,
                feature.position.1 + flow_vec.v,
            );

            // Check if new position is within bounds
            let (width, height) = curr_frame.dimensions();
            if new_position.0 < 0.0
                || new_position.0 >= width as f32
                || new_position.1 < 0.0
                || new_position.1 >= height as f32
            {
                continue;
            }

            let mut confidence = feature.confidence;
            let mut accept_track = true;

            // Backwards flow check for robustness
            if self.params.use_backwards_check {
                let backwards_flow = lucas_kanade_flow(
                    &DynamicImage::ImageLuma8(curr_frame.clone()),
                    &DynamicImage::ImageLuma8(prev_frame.clone()),
                    Some(&[new_position]),
                    &self.params.flow_params,
                )?;

                if let Some(back_flow) =
                    backwards_flow.get([new_position.1 as usize, new_position.0 as usize])
                {
                    let back_position =
                        (new_position.0 + back_flow.u, new_position.1 + back_flow.v);

                    let error = ((back_position.0 - feature.position.0).powi(2)
                        + (back_position.1 - feature.position.1).powi(2))
                    .sqrt();

                    if error > self.params.backwards_threshold {
                        accept_track = false;
                    } else {
                        // Update confidence based on backwards error
                        confidence *= (1.0 - error / self.params.backwards_threshold).max(0.0);
                    }
                }
            }

            if accept_track && confidence >= self.params.min_confidence {
                let velocity = (
                    new_position.0 - feature.position.0,
                    new_position.1 - feature.position.1,
                );

                tracked_features.push(TrackedFeature {
                    position: new_position,
                    prev_position: feature.position,
                    velocity,
                    confidence,
                    age: feature.age + 1,
                    id: feature.id,
                });
            }
        }

        self.features = tracked_features;
        Ok(())
    }

    /// Add new features to track
    fn add_new_features(&mut self, new_features: &[(f32, f32)]) {
        let remaining_capacity = self.params.max_features.saturating_sub(self.features.len());

        for &position in new_features.iter().take(remaining_capacity) {
            // Check if this position is too close to existing features
            let too_close = self.features.iter().any(|f| {
                let distance = ((f.position.0 - position.0).powi(2)
                    + (f.position.1 - position.1).powi(2))
                .sqrt();
                distance < self.params.max_distance / 2.0
            });

            if !too_close {
                self.features.push(TrackedFeature {
                    position,
                    prev_position: position,
                    velocity: (0.0, 0.0),
                    confidence: 1.0,
                    age: 0,
                    id: self.next_id,
                });
                self.next_id += 1;
            }
        }
    }

    /// Get current tracked features
    pub fn get_features(&self) -> &[TrackedFeature] {
        &self.features
    }

    /// Get number of tracked features
    pub fn feature_count(&self) -> usize {
        self.features.len()
    }

    /// Reset tracker
    pub fn reset(&mut self) {
        self.features.clear();
        self.prev_frame = None;
        self.next_id = 0;
        self.feature_points.clear();
    }

    /// Get feature trajectories
    ///
    /// Returns a map from feature ID to positions over time
    pub fn get_trajectories(&self) -> HashMap<u32, Vec<(f32, f32)>> {
        let mut trajectories = HashMap::new();

        for feature in &self.features {
            trajectories
                .entry(feature.id)
                .or_insert_with(Vec::new)
                .push(feature.position);
        }

        trajectories
    }

    /// Predict next positions based on current velocities
    pub fn predict_positions(&self) -> Vec<(f32, f32)> {
        self.features
            .iter()
            .map(|f| (f.position.0 + f.velocity.0, f.position.1 + f.velocity.1))
            .collect()
    }
}

/// Multi-scale feature tracker using image pyramids
pub struct PyramidTracker {
    /// Trackers for each pyramid level
    trackers: Vec<LKTracker>,
    /// Number of pyramid levels
    levels: usize,
    /// Base tracker parameters
    #[allow(dead_code)]
    base_params: TrackerParams,
}

impl PyramidTracker {
    /// Create a new pyramid tracker
    ///
    /// # Arguments
    ///
    /// * `levels` - Number of pyramid levels
    /// * `params` - Base tracking parameters
    ///
    /// # Returns
    ///
    /// * New pyramid tracker instance
    pub fn new(levels: usize, params: TrackerParams) -> Self {
        let mut trackers = Vec::new();

        for level in 0..levels {
            let scale = 2.0_f32.powi(level as i32);
            let mut level_params = params.clone();

            // Adjust parameters for this pyramid level
            level_params.max_distance /= scale;
            level_params.backwards_threshold /= scale;
            level_params.flow_params.window_size =
                (level_params.flow_params.window_size as f32 / scale).max(3.0) as usize;

            trackers.push(LKTracker::new(level_params));
        }

        Self {
            trackers,
            levels,
            base_params: params,
        }
    }

    /// Update pyramid tracker with new frame
    pub fn update(
        &mut self,
        frame: &DynamicImage,
        new_features: Option<&[(f32, f32)]>,
    ) -> Result<Vec<TrackedFeature>> {
        // Build pyramid for current frame
        let gray_frame = frame.to_luma8();
        let pyramid = self.build_pyramid(&gray_frame);

        let mut all_features = Vec::new();

        // Track at each pyramid level
        for (level, tracker) in self.trackers.iter_mut().enumerate() {
            let level_frame = DynamicImage::ImageLuma8(pyramid[level].clone());

            // Scale new features for this level
            let scaled_features = if level == 0 {
                new_features.map(|features| features.to_vec())
            } else {
                let scale = 2.0_f32.powi(level as i32);
                new_features.map(|features| {
                    features
                        .iter()
                        .map(|&(x, y)| (x / scale, y / scale))
                        .collect()
                })
            };

            let level_features = tracker.update(&level_frame, scaled_features.as_deref())?;

            // Scale features back to original resolution
            let scale = 2.0_f32.powi(level as i32);
            for feature in level_features {
                let mut scaled_feature = *feature;
                scaled_feature.position.0 *= scale;
                scaled_feature.position.1 *= scale;
                scaled_feature.prev_position.0 *= scale;
                scaled_feature.prev_position.1 *= scale;
                scaled_feature.velocity.0 *= scale;
                scaled_feature.velocity.1 *= scale;
                all_features.push(scaled_feature);
            }
        }

        Ok(all_features)
    }

    /// Build image pyramid
    fn build_pyramid(&self, img: &GrayImage) -> Vec<GrayImage> {
        let mut pyramid = vec![img.clone()];

        for _ in 1..self.levels {
            let prev = &pyramid[pyramid.len() - 1];
            let (width, height) = prev.dimensions();
            let new_width = width / 2;
            let new_height = height / 2;

            if new_width < 8 || new_height < 8 {
                break; // Stop if image gets too small
            }

            let mut downsampled = image::ImageBuffer::new(new_width, new_height);

            for y in 0..new_height {
                for x in 0..new_width {
                    // Simple 2x2 average with bounds checking
                    let x2 = x * 2;
                    let y2 = y * 2;

                    let mut sum = prev.get_pixel(x2, y2)[0] as u32;
                    let mut count = 1;

                    if x2 + 1 < width {
                        sum += prev.get_pixel(x2 + 1, y2)[0] as u32;
                        count += 1;
                    }
                    if y2 + 1 < height {
                        sum += prev.get_pixel(x2, y2 + 1)[0] as u32;
                        count += 1;
                    }
                    if x2 + 1 < width && y2 + 1 < height {
                        sum += prev.get_pixel(x2 + 1, y2 + 1)[0] as u32;
                        count += 1;
                    }

                    downsampled.put_pixel(x, y, image::Luma([(sum / count) as u8]));
                }
            }

            pyramid.push(downsampled);
        }

        pyramid
    }

    /// Reset all trackers
    pub fn reset(&mut self) {
        for tracker in &mut self.trackers {
            tracker.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Luma};

    fn create_test_image(width: u32, height: u32) -> DynamicImage {
        let img = ImageBuffer::from_fn(width, height, |x, y| {
            let val = ((x + y) % 256) as u8;
            Luma([val])
        });
        DynamicImage::ImageLuma8(img)
    }

    #[test]
    fn test_tracker_creation() {
        let tracker = LKTracker::new(TrackerParams::default());
        assert_eq!(tracker.feature_count(), 0);
    }

    #[test]
    fn test_tracker_update() {
        let mut tracker = LKTracker::new(TrackerParams::default());
        let frame = create_test_image(100, 100);

        let result = tracker.update(&frame, Some(&[(10.0, 10.0), (50.0, 50.0)]));
        assert!(result.is_ok());

        let features = result.unwrap();
        assert_eq!(features.len(), 2);
    }

    #[test]
    fn test_feature_tracking() {
        let mut tracker = LKTracker::new(TrackerParams::default());

        // First frame with features
        let frame1 = create_test_image(100, 100);
        tracker.update(&frame1, Some(&[(25.0, 25.0)])).unwrap();

        // Second frame - features should be tracked
        let frame2 = create_test_image(100, 100);
        let features = tracker.update(&frame2, None).unwrap();

        // Should maintain at least some features
        assert!(!features.is_empty());
        assert!(features[0].age > 0);
    }

    #[test]
    fn test_pyramid_tracker() {
        let mut tracker = PyramidTracker::new(3, TrackerParams::default());
        let frame = create_test_image(64, 64);

        let result = tracker.update(&frame, Some(&[(16.0, 16.0)]));
        assert!(result.is_ok());

        let features = result.unwrap();
        assert!(!features.is_empty());
    }

    #[test]
    fn test_feature_id_consistency() {
        let mut tracker = LKTracker::new(TrackerParams::default());
        let frame = create_test_image(100, 100);

        tracker.update(&frame, Some(&[(25.0, 25.0)])).unwrap();
        let features1 = tracker.get_features();
        let id1 = features1[0].id;

        tracker.update(&frame, None).unwrap();
        let features2 = tracker.get_features();

        if !features2.is_empty() {
            assert_eq!(features2[0].id, id1);
        }
    }

    #[test]
    fn test_trajectory_tracking() {
        let mut tracker = LKTracker::new(TrackerParams::default());
        let frame = create_test_image(100, 100);

        tracker.update(&frame, Some(&[(25.0, 25.0)])).unwrap();
        tracker.update(&frame, None).unwrap();

        let trajectories = tracker.get_trajectories();
        assert!(!trajectories.is_empty());
    }
}
