//! Interactive 3D visualization capabilities for clustering results
//!
//! This module provides advanced interactive 3D visualization features including
//! real-time manipulation, dynamic clustering updates, multi-view perspectives,
//! and immersive exploration tools for complex clustering scenarios.

use ndarray::{Array1, Array2, ArrayView2};
use num_traits::{Float, FromPrimitive};
use std::collections::HashMap;
use std::fmt::Debug;

use serde::{Deserialize, Serialize};

use super::{ColorScheme, ScatterPlot3D, VisualizationConfig};
use crate::error::{ClusteringError, Result};

/// Configuration for interactive 3D visualizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractiveConfig {
    /// Enable camera controls (rotation, zoom, pan)
    pub enable_camera_controls: bool,
    /// Enable point selection and highlighting
    pub enable_point_selection: bool,
    /// Enable cluster manipulation (drag centroids)
    pub enable_cluster_manipulation: bool,
    /// Show coordinate axes
    pub show_axes: bool,
    /// Show grid
    pub show_grid: bool,
    /// Enable real-time statistics display
    pub show_realtime_stats: bool,
    /// Enable multi-view layout
    pub multi_view: bool,
    /// Number of simultaneous views
    pub view_count: usize,
    /// Enable VR/AR mode
    pub enable_vr_mode: bool,
    /// Enable stereoscopic rendering
    pub stereoscopic: bool,
    /// Field of view for 3D perspective
    pub field_of_view: f32,
    /// Camera movement sensitivity
    pub camera_sensitivity: f32,
    /// Point highlighting on hover
    pub highlight_on_hover: bool,
    /// Show cluster boundaries in 3D
    pub show_3d_boundaries: bool,
    /// Enable temporal view (for time series clustering)
    pub temporal_view: bool,
}

impl Default for InteractiveConfig {
    fn default() -> Self {
        Self {
            enable_camera_controls: true,
            enable_point_selection: true,
            enable_cluster_manipulation: false,
            show_axes: true,
            show_grid: true,
            show_realtime_stats: true,
            multi_view: false,
            view_count: 1,
            enable_vr_mode: false,
            stereoscopic: false,
            field_of_view: 60.0,
            camera_sensitivity: 1.0,
            highlight_on_hover: true,
            show_3d_boundaries: true,
            temporal_view: false,
        }
    }
}

/// Camera state for 3D visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraState {
    /// Camera position (x, y, z)
    pub position: (f64, f64, f64),
    /// Look-at target (x, y, z)
    pub target: (f64, f64, f64),
    /// Up vector (x, y, z)
    pub up: (f64, f64, f64),
    /// Field of view in degrees
    pub fov: f32,
    /// Near clipping plane
    pub near: f64,
    /// Far clipping plane
    pub far: f64,
    /// Camera rotation (euler angles: pitch, yaw, roll)
    pub rotation: (f64, f64, f64),
    /// Zoom level
    pub zoom: f64,
}

impl Default for CameraState {
    fn default() -> Self {
        Self {
            position: (10.0, 10.0, 10.0),
            target: (0.0, 0.0, 0.0),
            up: (0.0, 1.0, 0.0),
            fov: 60.0,
            near: 0.1,
            far: 1000.0,
            rotation: (0.0, 0.0, 0.0),
            zoom: 1.0,
        }
    }
}

/// Interactive state management for 3D visualization
#[derive(Debug, Clone)]
pub struct InteractiveState {
    /// Current camera state
    pub camera: CameraState,
    /// Selected points
    pub selected_points: Vec<usize>,
    /// Highlighted points (on hover)
    pub highlighted_points: Vec<usize>,
    /// Active cluster (for manipulation)
    pub active_cluster: Option<i32>,
    /// Mouse/touch input state
    pub input_state: InputState,
    /// View bounds for each dimension
    pub view_bounds: (f64, f64, f64, f64, f64, f64),
    /// Current time (for temporal views)
    pub current_time: f64,
    /// Animation playback state
    pub animation_playing: bool,
    /// Current view mode
    pub view_mode: ViewMode,
}

/// Input state for interactive controls
#[derive(Debug, Clone)]
pub struct InputState {
    /// Mouse position (x, y)
    pub mouse_position: (f64, f64),
    /// Previous mouse position
    pub prev_mouse_position: (f64, f64),
    /// Mouse buttons pressed
    pub mouse_buttons: Vec<MouseButton>,
    /// Keyboard keys pressed
    pub keys_pressed: Vec<KeyCode>,
    /// Touch points (for multi-touch)
    pub touch_points: Vec<TouchPoint>,
    /// Gesture state
    pub gesture_state: GestureState,
}

/// Mouse button identifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
    Other(u8),
}

/// Key codes for keyboard input
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyCode {
    Space,
    Enter,
    Escape,
    ArrowUp,
    ArrowDown,
    ArrowLeft,
    ArrowRight,
    Shift,
    Ctrl,
    Alt,
    Key(char),
}

/// Touch point for multi-touch input
#[derive(Debug, Clone)]
pub struct TouchPoint {
    pub id: u64,
    pub position: (f64, f64),
    pub pressure: f64,
}

/// Gesture recognition state
#[derive(Debug, Clone)]
pub struct GestureState {
    pub is_pinching: bool,
    pub pinch_scale: f64,
    pub is_rotating: bool,
    pub rotation_angle: f64,
    pub is_panning: bool,
    pub pan_delta: (f64, f64),
}

/// 3D view modes
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ViewMode {
    /// Standard perspective view
    Perspective,
    /// Orthographic projection
    Orthographic,
    /// First-person view
    FirstPerson,
    /// Bird's eye view
    BirdsEye,
    /// Side view
    Side,
    /// Front view
    Front,
    /// Top view
    Top,
    /// Split screen (multiple views)
    SplitScreen,
    /// VR stereo view
    VRStereo,
}

/// Real-time cluster statistics for interactive display
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterStats {
    /// Cluster ID
    pub cluster_id: i32,
    /// Number of points in cluster
    pub point_count: usize,
    /// Cluster centroid
    pub centroid: Array1<f64>,
    /// Cluster diameter (maximum distance between points)
    pub diameter: f64,
    /// Average distance to centroid
    pub avg_distance_to_centroid: f64,
    /// Cluster density
    pub density: f64,
    /// Bounding box (min_x, max_x, min_y, max_y, min_z, max_z)
    pub bounding_box: (f64, f64, f64, f64, f64, f64),
    /// Cluster color
    pub color: String,
}

/// Interactive 3D visualizer
pub struct InteractiveVisualizer {
    config: InteractiveConfig,
    state: InteractiveState,
    cluster_stats: HashMap<i32, ClusterStats>,
    last_update: std::time::Instant,
}

impl InteractiveVisualizer {
    /// Create a new interactive visualizer
    pub fn new(config: InteractiveConfig) -> Self {
        Self {
            config,
            state: InteractiveState {
                camera: CameraState::default(),
                selected_points: Vec::new(),
                highlighted_points: Vec::new(),
                active_cluster: None,
                input_state: InputState {
                    mouse_position: (0.0, 0.0),
                    prev_mouse_position: (0.0, 0.0),
                    mouse_buttons: Vec::new(),
                    keys_pressed: Vec::new(),
                    touch_points: Vec::new(),
                    gesture_state: GestureState {
                        is_pinching: false,
                        pinch_scale: 1.0,
                        is_rotating: false,
                        rotation_angle: 0.0,
                        is_panning: false,
                        pan_delta: (0.0, 0.0),
                    },
                },
                view_bounds: (-10.0, 10.0, -10.0, 10.0, -10.0, 10.0),
                current_time: 0.0,
                animation_playing: false,
                view_mode: ViewMode::Perspective,
            },
            cluster_stats: HashMap::new(),
            last_update: std::time::Instant::now(),
        }
    }

    /// Update visualization with new data
    pub fn update_data<F: Float + FromPrimitive + Debug>(
        &mut self,
        data: ArrayView2<F>,
        labels: &Array1<i32>,
        centroids: Option<&Array2<F>>,
    ) -> Result<()> {
        // Calculate cluster statistics
        self.calculate_cluster_stats(data, labels, centroids)?;

        // Update view bounds if needed
        self.update_view_bounds(data);

        // Reset selection if data changed significantly
        self.validate_selections(data.nrows());

        Ok(())
    }

    /// Handle mouse input
    pub fn handle_mouse_input(&mut self, button: MouseButton, position: (f64, f64), pressed: bool) {
        self.state.input_state.prev_mouse_position = self.state.input_state.mouse_position;
        self.state.input_state.mouse_position = position;

        if pressed {
            if !self.state.input_state.mouse_buttons.contains(&button) {
                self.state.input_state.mouse_buttons.push(button);
            }
        } else {
            self.state
                .input_state
                .mouse_buttons
                .retain(|&b| b != button);
        }

        // Handle camera controls
        if self.config.enable_camera_controls {
            self.handle_camera_input();
        }
    }

    /// Handle keyboard input
    pub fn handle_keyboard_input(&mut self, key: KeyCode, pressed: bool) {
        if pressed {
            if !self.state.input_state.keys_pressed.contains(&key) {
                self.state.input_state.keys_pressed.push(key);
            }
        } else {
            self.state.input_state.keys_pressed.retain(|&k| k != key);
        }

        // Handle special key combinations
        self.handle_keyboard_shortcuts(key, pressed);
    }

    /// Handle touch input for mobile/tablet interfaces
    pub fn handle_touch_input(&mut self, touchpoints: Vec<TouchPoint>) {
        let prev_touch_count = self.state.input_state.touch_points.len();
        self.state.input_state.touch_points = touchpoints;
        let current_touch_count = self.state.input_state.touch_points.len();

        // Gesture recognition
        self.update_gesture_state(prev_touch_count, current_touch_count);

        // Handle multi-touch gestures
        if self.config.enable_camera_controls {
            self.handle_touch_gestures();
        }
    }

    /// Select points within a 3D region
    pub fn select_points_in_region(&mut self, region: BoundingBox3D) -> Vec<usize> {
        // This would be implemented with actual 3D point-in-box testing
        // For now, return empty selection
        let selected = Vec::new();
        self.state.selected_points = selected.clone();
        selected
    }

    /// Highlight points at screen coordinates
    pub fn highlight_points_at(&mut self, screenpos: (f64, f64)) -> Vec<usize> {
        // This would implement 3D picking/ray casting
        // For now, return empty highlights
        let highlighted = Vec::new();
        self.state.highlighted_points = highlighted.clone();
        highlighted
    }

    /// Get current cluster statistics
    pub fn get_cluster_stats(&self) -> &HashMap<i32, ClusterStats> {
        &self.cluster_stats
    }

    /// Get current interactive state
    pub fn get_state(&self) -> &InteractiveState {
        &self.state
    }

    /// Set camera position
    pub fn set_camera_position(&mut self, position: (f64, f64, f64)) {
        self.state.camera.position = position;
    }

    /// Set camera target
    pub fn set_camera_target(&mut self, target: (f64, f64, f64)) {
        self.state.camera.target = target;
    }

    /// Set view mode
    pub fn set_view_mode(&mut self, mode: ViewMode) {
        self.state.view_mode = mode;

        // Adjust camera for specific view modes
        match mode {
            ViewMode::BirdsEye => {
                self.state.camera.position = (0.0, 20.0, 0.0);
                self.state.camera.target = (0.0, 0.0, 0.0);
                self.state.camera.up = (0.0, 0.0, -1.0);
            }
            ViewMode::Side => {
                self.state.camera.position = (20.0, 0.0, 0.0);
                self.state.camera.target = (0.0, 0.0, 0.0);
                self.state.camera.up = (0.0, 1.0, 0.0);
            }
            ViewMode::Front => {
                self.state.camera.position = (0.0, 0.0, 20.0);
                self.state.camera.target = (0.0, 0.0, 0.0);
                self.state.camera.up = (0.0, 1.0, 0.0);
            }
            ViewMode::Top => {
                self.state.camera.position = (0.0, 20.0, 0.0);
                self.state.camera.target = (0.0, 0.0, 0.0);
                self.state.camera.up = (0.0, 0.0, -1.0);
            }
            _ => {
                // Keep current camera settings for other modes
            }
        }
    }

    /// Enable/disable animation playback
    pub fn set_animation_playing(&mut self, playing: bool) {
        self.state.animation_playing = playing;
    }

    /// Set current time for temporal views
    pub fn set_current_time(&mut self, time: f64) {
        self.state.current_time = time;
    }

    /// Generate export data for current view
    pub fn export_view_state(&self) -> Result<String> {
        #[cfg(feature = "serde")]
        {
            let export_data = InteractiveViewExport {
                camera: self.state.camera.clone(),
                view_mode: self.state.view_mode,
                cluster_stats: self.cluster_stats.clone(),
                view_bounds: self.state.view_bounds,
                current_time: self.state.current_time,
            };

            return serde_json::to_string_pretty(&export_data)
                .map_err(|e| ClusteringError::ComputationError(format!("Export failed: {}", e)));
        }

        #[cfg(not(feature = "serde"))]
        {
            Err(ClusteringError::ComputationError(
                "Export requires 'serde' feature".to_string(),
            ))
        }
    }

    /// Calculate cluster statistics
    fn calculate_cluster_stats<F: Float + FromPrimitive + Debug>(
        &mut self,
        data: ArrayView2<F>,
        labels: &Array1<i32>,
        centroids: Option<&Array2<F>>,
    ) -> Result<()> {
        self.cluster_stats.clear();

        // Get unique cluster labels
        let mut unique_labels: Vec<i32> = labels.iter().cloned().collect();
        unique_labels.sort_unstable();
        unique_labels.dedup();

        for &cluster_id in &unique_labels {
            // Find points in this cluster
            let cluster_points: Vec<usize> = labels
                .iter()
                .enumerate()
                .filter(|(_, &label)| label == cluster_id)
                .map(|(idx_, _)| idx_)
                .collect();

            if cluster_points.is_empty() {
                continue;
            }

            // Calculate centroid
            let centroid = if let Some(cents) = centroids {
                if cluster_id >= 0 && (cluster_id as usize) < cents.nrows() {
                    cents
                        .row(cluster_id as usize)
                        .mapv(|x| x.to_f64().unwrap_or(0.0))
                } else {
                    // Calculate centroid from points
                    self.calculate_centroid_from_points(data, &cluster_points)?
                }
            } else {
                self.calculate_centroid_from_points(data, &cluster_points)?
            };

            // Calculate statistics
            let (diameter, avg_distance, density, bounding_box) =
                self.calculate_cluster_metrics(data, &cluster_points, &centroid)?;

            let stats = ClusterStats {
                cluster_id,
                point_count: cluster_points.len(),
                centroid,
                diameter,
                avg_distance_to_centroid: avg_distance,
                density,
                bounding_box,
                color: format!("#{:06x}", (cluster_id.abs() as u32 * 123456) % 0xFFFFFF),
            };

            self.cluster_stats.insert(cluster_id, stats);
        }

        Ok(())
    }

    /// Calculate centroid from points
    fn calculate_centroid_from_points<F: Float + FromPrimitive + Debug>(
        &self,
        data: ArrayView2<F>,
        point_indices: &[usize],
    ) -> Result<Array1<f64>> {
        let n_features = data.ncols();
        let mut centroid = Array1::zeros(n_features);

        for &idx in point_indices {
            for j in 0..n_features {
                centroid[j] += data[[idx, j]].to_f64().unwrap_or(0.0);
            }
        }

        let count = point_indices.len() as f64;
        if count > 0.0 {
            centroid.mapv_inplace(|x| x / count);
        }

        Ok(centroid)
    }

    /// Calculate various cluster metrics
    fn calculate_cluster_metrics<F: Float + FromPrimitive + Debug>(
        &self,
        data: ArrayView2<F>,
        point_indices: &[usize],
        centroid: &Array1<f64>,
    ) -> Result<(f64, f64, f64, (f64, f64, f64, f64, f64, f64))> {
        let n_features = data.ncols();

        let mut max_distance = 0.0;
        let mut total_distance = 0.0;
        let mut min_coords = vec![f64::INFINITY; n_features];
        let mut max_coords = vec![f64::NEG_INFINITY; n_features];

        // Calculate distances and bounding box
        for &idx in point_indices {
            let mut distance_to_centroid = 0.0;

            for j in 0..n_features {
                let coord = data[[idx, j]].to_f64().unwrap_or(0.0);
                let diff = coord - centroid[j];
                distance_to_centroid += diff * diff;

                min_coords[j] = min_coords[j].min(coord);
                max_coords[j] = max_coords[j].max(coord);
            }

            distance_to_centroid = distance_to_centroid.sqrt();
            total_distance += distance_to_centroid;
        }

        // Calculate diameter (maximum pairwise distance)
        for i in 0..point_indices.len() {
            for j in (i + 1)..point_indices.len() {
                let mut distance = 0.0;
                for k in 0..n_features {
                    let diff = data[[point_indices[i], k]].to_f64().unwrap_or(0.0)
                        - data[[point_indices[j], k]].to_f64().unwrap_or(0.0);
                    distance += diff * diff;
                }
                distance = distance.sqrt();
                max_distance = max_distance.max(distance);
            }
        }

        let avg_distance = if point_indices.is_empty() {
            0.0
        } else {
            total_distance / point_indices.len() as f64
        };

        // Calculate density (points per unit volume)
        let volume = if n_features >= 3 {
            (max_coords[0] - min_coords[0])
                * (max_coords[1] - min_coords[1])
                * (max_coords[2] - min_coords[2])
        } else if n_features >= 2 {
            (max_coords[0] - min_coords[0]) * (max_coords[1] - min_coords[1])
        } else {
            max_coords[0] - min_coords[0]
        };

        let density = if volume > 0.0 {
            point_indices.len() as f64 / volume
        } else {
            0.0
        };

        let bounding_box = (
            min_coords.get(0).copied().unwrap_or(0.0),
            max_coords.get(0).copied().unwrap_or(0.0),
            min_coords.get(1).copied().unwrap_or(0.0),
            max_coords.get(1).copied().unwrap_or(0.0),
            min_coords.get(2).copied().unwrap_or(0.0),
            max_coords.get(2).copied().unwrap_or(0.0),
        );

        Ok((max_distance, avg_distance, density, bounding_box))
    }

    /// Update view bounds based on data
    fn update_view_bounds<F: Float + FromPrimitive + Debug>(&mut self, data: ArrayView2<F>) {
        let n_features = data.ncols();

        if n_features == 0 || data.nrows() == 0 {
            return;
        }

        let mut min_vals = vec![f64::INFINITY; n_features];
        let mut max_vals = vec![f64::NEG_INFINITY; n_features];

        for i in 0..data.nrows() {
            for j in 0..n_features {
                let val = data[[i, j]].to_f64().unwrap_or(0.0);
                min_vals[j] = min_vals[j].min(val);
                max_vals[j] = max_vals[j].max(val);
            }
        }

        // Add some padding
        let padding = 0.1;
        for j in 0..n_features {
            let range = max_vals[j] - min_vals[j];
            min_vals[j] -= range * padding;
            max_vals[j] += range * padding;
        }

        self.state.view_bounds = (
            min_vals.get(0).copied().unwrap_or(-10.0),
            max_vals.get(0).copied().unwrap_or(10.0),
            min_vals.get(1).copied().unwrap_or(-10.0),
            max_vals.get(1).copied().unwrap_or(10.0),
            min_vals.get(2).copied().unwrap_or(-10.0),
            max_vals.get(2).copied().unwrap_or(10.0),
        );
    }

    /// Validate point selections after data changes
    fn validate_selections(&mut self, npoints: usize) {
        self.state.selected_points.retain(|&idx| idx < npoints);
        self.state.highlighted_points.retain(|&idx| idx < npoints);
    }

    /// Handle camera input based on mouse state
    fn handle_camera_input(&mut self) {
        let mouse_delta = (
            self.state.input_state.mouse_position.0 - self.state.input_state.prev_mouse_position.0,
            self.state.input_state.mouse_position.1 - self.state.input_state.prev_mouse_position.1,
        );

        let sensitivity = self.config.camera_sensitivity as f64;

        // Rotation with left mouse button
        if self
            .state
            .input_state
            .mouse_buttons
            .contains(&MouseButton::Left)
        {
            self.state.camera.rotation.0 += mouse_delta.1 * sensitivity * 0.01;
            self.state.camera.rotation.1 += mouse_delta.0 * sensitivity * 0.01;
        }

        // Zoom with right mouse button or scroll
        if self
            .state
            .input_state
            .mouse_buttons
            .contains(&MouseButton::Right)
        {
            self.state.camera.zoom *= 1.0 + mouse_delta.1 * sensitivity * 0.01;
            self.state.camera.zoom = self.state.camera.zoom.clamp(0.1, 10.0);
        }

        // Pan with middle mouse button
        if self
            .state
            .input_state
            .mouse_buttons
            .contains(&MouseButton::Middle)
        {
            // This would update camera position based on pan delta
        }
    }

    /// Handle keyboard shortcuts
    fn handle_keyboard_shortcuts(&mut self, key: KeyCode, pressed: bool) {
        if !pressed {
            return;
        }

        match key {
            KeyCode::Space => {
                self.state.animation_playing = !self.state.animation_playing;
            }
            KeyCode::Key('1') => self.set_view_mode(ViewMode::Perspective),
            KeyCode::Key('2') => self.set_view_mode(ViewMode::Orthographic),
            KeyCode::Key('3') => self.set_view_mode(ViewMode::BirdsEye),
            KeyCode::Key('4') => self.set_view_mode(ViewMode::Side),
            KeyCode::Key('5') => self.set_view_mode(ViewMode::Front),
            KeyCode::Key('6') => self.set_view_mode(ViewMode::Top),
            KeyCode::Escape => {
                self.state.selected_points.clear();
                self.state.highlighted_points.clear();
            }
            _ => {}
        }
    }

    /// Update gesture recognition state
    fn update_gesture_state(&mut self, prev_touch_count: usize, current_touchcount: usize) {
        // Detect pinch gesture
        if current_touchcount == 2 {
            let touch1 = &self.state.input_state.touch_points[0];
            let touch2 = &self.state.input_state.touch_points[1];

            let distance = ((touch1.position.0 - touch2.position.0).powi(2)
                + (touch1.position.1 - touch2.position.1).powi(2))
            .sqrt();

            if !self.state.input_state.gesture_state.is_pinching {
                self.state.input_state.gesture_state.is_pinching = true;
                self.state.input_state.gesture_state.pinch_scale = distance;
            } else {
                let scale_factor = distance / self.state.input_state.gesture_state.pinch_scale;
                self.state.camera.zoom *= scale_factor;
                self.state.input_state.gesture_state.pinch_scale = distance;
            }
        } else {
            self.state.input_state.gesture_state.is_pinching = false;
        }
    }

    /// Handle multi-touch gestures
    fn handle_touch_gestures(&mut self) {
        // Implementation would handle pinch-to-zoom, rotation, etc.
    }
}

/// 3D bounding box for region selection
#[derive(Debug, Clone)]
pub struct BoundingBox3D {
    pub min: (f64, f64, f64),
    pub max: (f64, f64, f64),
}

/// Export format for interactive view state
#[derive(Debug, Clone, Serialize, Deserialize)]
struct InteractiveViewExport {
    camera: CameraState,
    view_mode: ViewMode,
    cluster_stats: HashMap<i32, ClusterStats>,
    view_bounds: (f64, f64, f64, f64, f64, f64),
    current_time: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_interactive_visualizer_creation() {
        let config = InteractiveConfig::default();
        let visualizer = InteractiveVisualizer::new(config);

        assert_eq!(visualizer.state.view_mode, ViewMode::Perspective);
        assert!(visualizer.cluster_stats.is_empty());
    }

    #[test]
    fn test_camera_controls() {
        let config = InteractiveConfig::default();
        let mut visualizer = InteractiveVisualizer::new(config);

        visualizer.set_camera_position((5.0, 5.0, 5.0));
        assert_eq!(visualizer.state.camera.position, (5.0, 5.0, 5.0));

        visualizer.set_camera_target((1.0, 1.0, 1.0));
        assert_eq!(visualizer.state.camera.target, (1.0, 1.0, 1.0));
    }

    #[test]
    fn test_view_mode_switching() {
        let config = InteractiveConfig::default();
        let mut visualizer = InteractiveVisualizer::new(config);

        visualizer.set_view_mode(ViewMode::BirdsEye);
        assert_eq!(visualizer.state.view_mode, ViewMode::BirdsEye);
        assert_eq!(visualizer.state.camera.position, (0.0, 20.0, 0.0));
    }

    #[test]
    fn test_cluster_stats_calculation() {
        let config = InteractiveConfig::default();
        let mut visualizer = InteractiveVisualizer::new(config);

        let data = Array2::from_shape_vec(
            (4, 3),
            vec![1.0, 2.0, 3.0, 1.1, 2.1, 3.1, 5.0, 6.0, 7.0, 5.1, 6.1, 7.1],
        )
        .unwrap();

        let labels = Array1::from_vec(vec![0, 0, 1, 1]);

        visualizer.update_data(data.view(), &labels, None).unwrap();

        let stats = visualizer.get_cluster_stats();
        assert_eq!(stats.len(), 2);
        assert!(stats.contains_key(&0));
        assert!(stats.contains_key(&1));

        let cluster_0_stats = &stats[&0];
        assert_eq!(cluster_0_stats.point_count, 2);
    }
}
