//! Advanced animation capabilities for clustering visualization
//!
//! This module provides sophisticated animation features for clustering algorithms,
//! including 3D animations, convergence animations, real-time streaming visualizations,
//! and export capabilities for creating videos and interactive presentations.

use ndarray::{s, Array1, Array2, Array3, ArrayView2};
use num_traits::{Float, FromPrimitive};
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use super::{EasingFunction, ScatterPlot2D, ScatterPlot3D, VisualizationConfig};
use crate::error::{ClusteringError, Result};

/// Configuration for iterative algorithm animations (like K-means convergence)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IterativeAnimationConfig {
    /// Capture frame every N iterations
    pub capture_frequency: usize,
    /// Interpolate between captured frames
    pub interpolate_frames: bool,
    /// Number of interpolation frames between captures
    pub interpolation_frames: usize,
    /// Animation speed (frames per second)
    pub fps: f32,
    /// Show convergence metrics overlay
    pub show_convergence_overlay: bool,
    /// Show iteration numbers
    pub show_iteration_numbers: bool,
    /// Highlight centroid movement
    pub highlight_centroid_movement: bool,
    /// Fade effect for old positions
    pub fade_effect: bool,
    /// Trail length for moving points
    pub trail_length: usize,
}

impl Default for IterativeAnimationConfig {
    fn default() -> Self {
        Self {
            capture_frequency: 1,
            interpolate_frames: true,
            interpolation_frames: 5,
            fps: 10.0,
            show_convergence_overlay: true,
            show_iteration_numbers: true,
            highlight_centroid_movement: true,
            fade_effect: true,
            trail_length: 3,
        }
    }
}

/// Configuration for streaming data visualizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    /// Buffer size for streaming data
    pub buffer_size: usize,
    /// Update frequency for visualization
    pub update_frequency_ms: u64,
    /// Window size for rolling statistics
    pub rolling_window_size: usize,
    /// Show data arrival animation
    pub animate_new_data: bool,
    /// Animate cluster updates
    pub animate_cluster_updates: bool,
    /// Adaptive plot bounds
    pub adaptive_bounds: bool,
    /// Show streaming statistics
    pub show_streaming_stats: bool,
    /// Data point lifetime (for fading effect)
    pub point_lifetime_ms: u64,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            buffer_size: 1000,
            update_frequency_ms: 100,
            rolling_window_size: 50,
            animate_new_data: true,
            animate_cluster_updates: true,
            adaptive_bounds: true,
            show_streaming_stats: true,
            point_lifetime_ms: 10000,
        }
    }
}

/// Animation frame for iterative algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimationFrame {
    /// Frame number
    pub frame_number: usize,
    /// Iteration number (for iterative algorithms)
    pub iteration: usize,
    /// Timestamp
    pub timestamp: f64,
    /// Data points for this frame
    pub points: Array2<f64>,
    /// Cluster labels
    pub labels: Array1<i32>,
    /// Centroids (if available)
    pub centroids: Option<Array2<f64>>,
    /// Previous centroids (for movement visualization)
    pub previous_centroids: Option<Array2<f64>>,
    /// Convergence metrics
    pub convergence_info: Option<ConvergenceInfo>,
    /// Custom annotations
    pub annotations: Vec<AnimationAnnotation>,
}

/// Convergence information for animation overlays
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceInfo {
    /// Current inertia/distortion
    pub inertia: f64,
    /// Change in inertia from previous iteration
    pub inertia_change: f64,
    /// Maximum centroid movement
    pub max_centroid_movement: f64,
    /// Number of points that changed clusters
    pub label_changes: usize,
    /// Whether algorithm has converged
    pub converged: bool,
}

/// Animation annotation for custom overlays
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimationAnnotation {
    /// Annotation type
    pub annotation_type: String,
    /// Position (2D or 3D coordinates)
    pub position: Vec<f64>,
    /// Text content
    pub text: String,
    /// Color
    pub color: String,
    /// Font size
    pub font_size: f32,
}

/// Recorder for iterative algorithm animations
pub struct IterativeAnimationRecorder {
    frames: Vec<AnimationFrame>,
    config: IterativeAnimationConfig,
    start_time: Instant,
    current_iteration: usize,
    previous_centroids: Option<Array2<f64>>,
    previous_inertia: Option<f64>,
}

impl IterativeAnimationRecorder {
    /// Create a new animation recorder
    pub fn new(config: IterativeAnimationConfig) -> Self {
        Self {
            frames: Vec::new(),
            config,
            start_time: Instant::now(),
            current_iteration: 0,
            previous_centroids: None,
            previous_inertia: None,
        }
    }

    /// Record a frame during algorithm iteration
    pub fn record_frame<F: Float + FromPrimitive + Debug>(
        &mut self,
        data: ArrayView2<F>,
        labels: &Array1<i32>,
        centroids: Option<&Array2<F>>,
        inertia: Option<f64>,
    ) -> Result<()> {
        if self.current_iteration % self.config.capture_frequency != 0 {
            self.current_iteration += 1;
            return Ok(());
        }

        let timestamp = self.start_time.elapsed().as_secs_f64();

        // Convert data to f64
        let points = data.mapv(|x| x.to_f64().unwrap_or(0.0));

        // Convert centroids to f64
        let centroids_f64 = centroids.map(|c| c.mapv(|x| x.to_f64().unwrap_or(0.0)));

        // Calculate convergence info
        let convergence_info =
            if let (Some(current_centroids), Some(current_inertia)) = (&centroids_f64, inertia) {
                let centroid_movement = if let Some(prev_centroids) = &self.previous_centroids {
                    calculate_max_centroid_movement(prev_centroids, current_centroids)
                } else {
                    0.0
                };

                let inertia_change = if let Some(prev_inertia) = self.previous_inertia {
                    prev_inertia - current_inertia
                } else {
                    0.0
                };

                Some(ConvergenceInfo {
                    inertia: current_inertia,
                    inertia_change,
                    max_centroid_movement: centroid_movement,
                    label_changes: 0, // Would need previous labels to calculate
                    converged: centroid_movement < 1e-4, // Simple convergence check
                })
            } else {
                None
            };

        let frame = AnimationFrame {
            frame_number: self.frames.len(),
            iteration: self.current_iteration,
            timestamp,
            points,
            labels: labels.clone(),
            centroids: centroids_f64.clone(),
            previous_centroids: self.previous_centroids.clone(),
            convergence_info,
            annotations: Vec::new(),
        };

        self.frames.push(frame);

        // Update state for next iteration
        self.previous_centroids = centroids_f64;
        self.previous_inertia = inertia;
        self.current_iteration += 1;

        Ok(())
    }

    /// Add custom annotation to the current frame
    pub fn add_annotation(&mut self, annotation: AnimationAnnotation) {
        if let Some(frame) = self.frames.last_mut() {
            frame.annotations.push(annotation);
        }
    }

    /// Generate interpolated frames between recorded frames
    pub fn generate_interpolated_frames(&self) -> Vec<AnimationFrame> {
        if !self.config.interpolate_frames || self.frames.len() < 2 {
            return self.frames.clone();
        }

        let mut interpolated_frames = Vec::new();

        for i in 0..self.frames.len() - 1 {
            let current_frame = &self.frames[i];
            let next_frame = &self.frames[i + 1];

            // Add current frame
            interpolated_frames.push(current_frame.clone());

            // Add interpolated frames
            for j in 1..=self.config.interpolation_frames {
                let t = j as f64 / (self.config.interpolation_frames + 1) as f64;
                let interpolated_frame =
                    match interpolate_frames(current_frame, next_frame, t, &self.config) {
                        Ok(frame) => frame,
                        Err(_) => continue, // Skip interpolation on error
                    };
                interpolated_frames.push(interpolated_frame);
            }
        }

        // Add last frame
        if let Some(last_frame) = self.frames.last() {
            interpolated_frames.push(last_frame.clone());
        }

        interpolated_frames
    }

    /// Get all recorded frames
    pub fn get_frames(&self) -> &[AnimationFrame] {
        &self.frames
    }

    /// Export animation to JSON format
    pub fn export_to_json(&self) -> Result<String> {
        #[cfg(feature = "serde")]
        {
            let frames = if self.config.interpolate_frames {
                self.generate_interpolated_frames()
            } else {
                self.frames.clone()
            };

            return serde_json::to_string_pretty(&frames).map_err(|e| {
                ClusteringError::ComputationError(format!("JSON export failed: {}", e))
            });
        }

        #[cfg(not(feature = "serde"))]
        {
            Err(ClusteringError::ComputationError(
                "JSON export requires 'serde' feature".to_string(),
            ))
        }
    }
}

/// Streaming data visualizer for real-time clustering
pub struct StreamingVisualizer {
    data_buffer: VecDeque<(Array1<f64>, i32, Instant)>,
    config: StreamingConfig,
    last_update: Instant,
    bounds: Option<(f64, f64, f64, f64, f64, f64)>, // min_x, max_x, min_y, max_y, min_z, max_z
    streaming_stats: StreamingStats,
}

/// Statistics for streaming visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingStats {
    pub total_points_processed: usize,
    pub points_per_second: f64,
    pub cluster_counts: HashMap<i32, usize>,
    pub recent_cluster_changes: usize,
    pub data_arrival_rate: f64,
}

impl StreamingVisualizer {
    /// Create a new streaming visualizer
    pub fn new(config: StreamingConfig) -> Self {
        Self {
            data_buffer: VecDeque::new(),
            config,
            last_update: Instant::now(),
            bounds: None,
            streaming_stats: StreamingStats {
                total_points_processed: 0,
                points_per_second: 0.0,
                cluster_counts: HashMap::new(),
                recent_cluster_changes: 0,
                data_arrival_rate: 0.0,
            },
        }
    }

    /// Add new data point to the stream
    pub fn add_data_point(&mut self, point: Array1<f64>, label: i32) {
        let now = Instant::now();

        // Update bounds if adaptive (before moving point)
        if self.config.adaptive_bounds {
            self.update_bounds(&point);
        }

        // Add to buffer
        self.data_buffer.push_back((point, label, now));

        // Maintain buffer size
        while self.data_buffer.len() > self.config.buffer_size {
            self.data_buffer.pop_front();
        }

        // Update statistics
        self.streaming_stats.total_points_processed += 1;
        *self
            .streaming_stats
            .cluster_counts
            .entry(label)
            .or_insert(0) += 1;

        // Clean up old points
        self.cleanup_old_points(now);
    }

    /// Add batch of data points
    pub fn add_data_batch(&mut self, points: &Array2<f64>, labels: &Array1<i32>) -> Result<()> {
        if points.nrows() != labels.len() {
            return Err(ClusteringError::InvalidInput(
                "Number of points must match number of labels".to_string(),
            ));
        }

        for i in 0..points.nrows() {
            let point = points.row(i).to_owned();
            self.add_data_point(point, labels[i]);
        }

        Ok(())
    }

    /// Check if visualization should be updated
    pub fn should_update(&self) -> bool {
        self.last_update.elapsed().as_millis() >= self.config.update_frequency_ms as u128
    }

    /// Generate current visualization frame
    pub fn generate_frame(&mut self) -> Result<StreamingFrame> {
        let now = Instant::now();

        // Calculate statistics
        let time_since_last_update = now.duration_since(self.last_update).as_secs_f64();
        if time_since_last_update > 0.0 {
            let recent_points = self
                .data_buffer
                .iter()
                .filter(|(_, _, timestamp)| now.duration_since(*timestamp).as_secs_f64() < 1.0)
                .count();
            self.streaming_stats.points_per_second =
                recent_points as f64 / time_since_last_update.min(1.0);
        }

        // Extract current data
        let current_data: Vec<_> = self.data_buffer.iter().collect();

        if current_data.is_empty() {
            return Ok(StreamingFrame {
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs_f64(),
                points: Array2::zeros((0, 0)),
                labels: Array1::zeros(0),
                point_ages: Vec::new(),
                bounds: self.bounds,
                stats: self.streaming_stats.clone(),
                new_points_mask: Vec::new(),
            });
        }

        // Determine dimensionality
        let n_dims = current_data[0].0.len();
        let n_points = current_data.len();

        // Convert to arrays
        let mut points = Array2::zeros((n_points, n_dims));
        let mut labels = Array1::zeros(n_points);
        let mut point_ages = Vec::with_capacity(n_points);
        let mut new_points_mask = Vec::with_capacity(n_points);

        for (i, (point, label, timestamp)) in current_data.iter().enumerate() {
            for j in 0..n_dims {
                points[[i, j]] = point[j];
            }
            labels[i] = *label;

            let age = now.duration_since(*timestamp).as_millis() as f64;
            point_ages.push(age);

            // Mark as new if arrived recently
            new_points_mask.push(age < 500.0); // 500ms threshold for "new"
        }

        self.last_update = now;

        Ok(StreamingFrame {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
            points,
            labels,
            point_ages,
            bounds: self.bounds,
            stats: self.streaming_stats.clone(),
            new_points_mask,
        })
    }

    /// Update adaptive bounds
    fn update_bounds(&mut self, point: &Array1<f64>) {
        let n_dims = point.len();

        if let Some(bounds) = &mut self.bounds {
            // Update existing bounds
            if n_dims >= 1 {
                bounds.0 = bounds.0.min(point[0]); // min_x
                bounds.1 = bounds.1.max(point[0]); // max_x
            }
            if n_dims >= 2 {
                bounds.2 = bounds.2.min(point[1]); // min_y
                bounds.3 = bounds.3.max(point[1]); // max_y
            }
            if n_dims >= 3 {
                bounds.4 = bounds.4.min(point[2]); // min_z
                bounds.5 = bounds.5.max(point[2]); // max_z
            }
        } else {
            // Initialize bounds
            self.bounds = Some(if n_dims >= 3 {
                (point[0], point[0], point[1], point[1], point[2], point[2])
            } else if n_dims >= 2 {
                (point[0], point[0], point[1], point[1], 0.0, 0.0)
            } else {
                (point[0], point[0], 0.0, 0.0, 0.0, 0.0)
            });
        }
    }

    /// Clean up old points based on lifetime
    fn cleanup_old_points(&mut self, now: Instant) {
        let lifetime = Duration::from_millis(self.config.point_lifetime_ms);

        while let Some((_, _, timestamp)) = self.data_buffer.front() {
            if now.duration_since(*timestamp) > lifetime {
                self.data_buffer.pop_front();
            } else {
                break;
            }
        }
    }

    /// Get current streaming statistics
    pub fn get_stats(&self) -> &StreamingStats {
        &self.streaming_stats
    }
}

/// Frame for streaming visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingFrame {
    pub timestamp: f64,
    pub points: Array2<f64>,
    pub labels: Array1<i32>,
    pub point_ages: Vec<f64>,
    pub bounds: Option<(f64, f64, f64, f64, f64, f64)>,
    pub stats: StreamingStats,
    pub new_points_mask: Vec<bool>,
}

/// Calculate maximum centroid movement between iterations
#[allow(dead_code)]
fn calculate_max_centroid_movement(
    prev_centroids: &Array2<f64>,
    current_centroids: &Array2<f64>,
) -> f64 {
    if prev_centroids.shape() != current_centroids.shape() {
        return f64::INFINITY;
    }

    let mut max_movement = 0.0;

    for i in 0..prev_centroids.nrows() {
        let mut movement = 0.0;
        for j in 0..prev_centroids.ncols() {
            let diff = current_centroids[[i, j]] - prev_centroids[[i, j]];
            movement += diff * diff;
        }
        movement = movement.sqrt();
        max_movement = max_movement.max(movement);
    }

    max_movement
}

/// Interpolate between two animation frames
#[allow(dead_code)]
fn interpolate_frames(
    frame1: &AnimationFrame,
    frame2: &AnimationFrame,
    t: f64,
    config: &IterativeAnimationConfig,
) -> Result<AnimationFrame> {
    let t = apply_easing(t, EasingFunction::EaseInOut);

    // Interpolate centroids if both frames have them
    let centroids = if let (Some(c1), Some(c2)) = (&frame1.centroids, &frame2.centroids) {
        if c1.shape() == c2.shape() {
            Some(c1 * (1.0 - t) + c2 * t)
        } else {
            Some(c2.clone()) // Fall back to destination centroids
        }
    } else {
        frame2.centroids.clone()
    };

    // Interpolate convergence info
    let convergence_info =
        if let (Some(conv1), Some(conv2)) = (&frame1.convergence_info, &frame2.convergence_info) {
            Some(ConvergenceInfo {
                inertia: conv1.inertia * (1.0 - t) + conv2.inertia * t,
                inertia_change: conv1.inertia_change * (1.0 - t) + conv2.inertia_change * t,
                max_centroid_movement: conv1.max_centroid_movement * (1.0 - t)
                    + conv2.max_centroid_movement * t,
                label_changes: if t < 0.5 {
                    conv1.label_changes
                } else {
                    conv2.label_changes
                },
                converged: conv2.converged,
            })
        } else {
            frame2.convergence_info.clone()
        };

    Ok(AnimationFrame {
        frame_number: frame1.frame_number,
        iteration: frame1.iteration,
        timestamp: frame1.timestamp * (1.0 - t) + frame2.timestamp * t,
        points: frame2.points.clone(), // Don't interpolate data points
        labels: frame2.labels.clone(),
        centroids,
        previous_centroids: frame1.centroids.clone(),
        convergence_info,
        annotations: frame2.annotations.clone(),
    })
}

/// Apply easing function to interpolation parameter
#[allow(dead_code)]
fn apply_easing(t: f64, easing: EasingFunction) -> f64 {
    let t = t.clamp(0.0, 1.0);

    match easing {
        EasingFunction::Linear => t,
        EasingFunction::EaseIn => t * t,
        EasingFunction::EaseOut => 1.0 - (1.0 - t).powi(2),
        EasingFunction::EaseInOut => {
            if t < 0.5 {
                2.0 * t * t
            } else {
                1.0 - 2.0 * (1.0 - t).powi(2)
            }
        }
        EasingFunction::Bounce => {
            if t < 1.0 / 2.75 {
                7.5625 * t * t
            } else if t < 2.0 / 2.75 {
                let t = t - 1.5 / 2.75;
                7.5625 * t * t + 0.75
            } else if t < 2.5 / 2.75 {
                let t = t - 2.25 / 2.75;
                7.5625 * t * t + 0.9375
            } else {
                let t = t - 2.625 / 2.75;
                7.5625 * t * t + 0.984375
            }
        }
        EasingFunction::Elastic => {
            if t == 0.0 || t == 1.0 {
                t
            } else {
                let p = 0.3;
                let s = p / 4.0;
                -(2.0_f64.powf(10.0 * (t - 1.0))
                    * ((t - 1.0 - s) * (2.0 * std::f64::consts::PI) / p).sin())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_animation_recorder() {
        let config = IterativeAnimationConfig::default();
        let mut recorder = IterativeAnimationRecorder::new(config);

        let data =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let labels = Array1::from_vec(vec![0, 0, 1, 1]);
        let centroids = Array2::from_shape_vec((2, 2), vec![2.0, 3.0, 6.0, 7.0]).unwrap();

        recorder
            .record_frame(data.view(), &labels, Some(&centroids), Some(10.0))
            .unwrap();

        assert_eq!(recorder.get_frames().len(), 1);
        assert_eq!(recorder.get_frames()[0].iteration, 0);
    }

    #[test]
    fn test_streaming_visualizer() {
        let config = StreamingConfig::default();
        let mut visualizer = StreamingVisualizer::new(config);

        let point = Array1::from_vec(vec![1.0, 2.0]);
        visualizer.add_data_point(point, 0);

        let frame = visualizer.generate_frame().unwrap();
        assert_eq!(frame.points.nrows(), 1);
        assert_eq!(frame.labels[0], 0);
    }

    #[test]
    fn test_easing_functions() {
        assert_eq!(apply_easing(0.0, EasingFunction::Linear), 0.0);
        assert_eq!(apply_easing(1.0, EasingFunction::Linear), 1.0);
        assert_eq!(apply_easing(0.5, EasingFunction::Linear), 0.5);

        assert!(apply_easing(0.5, EasingFunction::EaseIn) < 0.5);
        assert!(apply_easing(0.5, EasingFunction::EaseOut) > 0.5);
    }
}
