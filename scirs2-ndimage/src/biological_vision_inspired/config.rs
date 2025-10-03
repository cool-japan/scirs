//! Configuration and Core Types for Biological Vision Inspired Processing
//!
//! This module contains configuration structures, models, and core data types
//! used throughout the biological vision inspired processing system.

use scirs2_core::ndarray::{Array1, Array2, Array3, Array4};
use std::collections::{HashMap, VecDeque};

/// Configuration for biological vision algorithms
#[derive(Debug, Clone)]
pub struct BiologicalVisionConfig {
    /// Number of cortical layers
    pub cortical_layers: usize,
    /// Receptive field sizes for each layer
    pub receptive_field_sizes: Vec<usize>,
    /// Lateral inhibition strength
    pub lateral_inhibition_strength: f64,
    /// Temporal integration window
    pub temporal_window: usize,
    /// Attention focus radius
    pub attention_radius: usize,
    /// Saccade planning horizon
    pub saccade_horizon: usize,
    /// Color constancy adaptation rate
    pub color_adaptation_rate: f64,
    /// Motion prediction window
    pub motion_prediction_window: usize,
    /// Compound eye ommatidial count
    pub ommatidial_count: usize,
    /// Predictive coding error threshold
    pub prediction_error_threshold: f64,
}

impl Default for BiologicalVisionConfig {
    fn default() -> Self {
        Self {
            cortical_layers: 6,
            receptive_field_sizes: vec![3, 5, 7, 11, 15, 21],
            lateral_inhibition_strength: 0.5,
            temporal_window: 10,
            attention_radius: 50,
            saccade_horizon: 5,
            color_adaptation_rate: 0.1,
            motion_prediction_window: 8,
            ommatidial_count: 1000,
            prediction_error_threshold: 0.3,
        }
    }
}

/// Hierarchical cortical layer representation
#[derive(Debug, Clone)]
pub struct CorticalLayer {
    /// Layer level (V1, V2, V4, etc.)
    pub level: usize,
    /// Feature maps at this layer
    pub feature_maps: Array3<f64>,
    /// Receptive field size
    pub receptive_field_size: usize,
    /// Lateral connections
    pub lateral_connections: Array2<f64>,
    /// Top-down predictions
    pub top_down_predictions: Array3<f64>,
    /// Bottom-up features
    pub bottom_upfeatures: Array3<f64>,
    /// Prediction errors
    pub prediction_errors: Array3<f64>,
}

/// Retinal processing structure
#[derive(Debug, Clone)]
pub struct RetinaModel {
    /// Photoreceptor responses
    pub photoreceptors: Array2<f64>,
    /// Bipolar cells
    pub bipolar_cells: Array2<f64>,
    /// Horizontal cells (lateral inhibition)
    pub horizontal_cells: Array2<f64>,
    /// Ganglion cells (edge detection)
    pub ganglion_cells: Array2<f64>,
    /// Center-surround filters
    pub center_surround_filters: Vec<Array2<f64>>,
}

/// Compound eye structure (inspired by insects)
#[derive(Debug, Clone)]
pub struct CompoundEyeModel {
    /// Individual ommatidia
    pub ommatidia: Vec<Ommatidium>,
    /// Motion detection cells
    pub motion_detectors: Array2<f64>,
    /// Wide-field integration
    pub wide_field_neurons: Array1<f64>,
    /// Looming detection
    pub looming_detectors: Array1<f64>,
}

/// Individual ommatidium
#[derive(Debug, Clone)]
pub struct Ommatidium {
    /// Position in compound eye
    pub position: (f64, f64),
    /// Optical axis direction
    pub optical_axis: (f64, f64, f64),
    /// Photoreceptor response
    pub response: f64,
    /// Temporal response history
    pub responsehistory: VecDeque<f64>,
}

/// Attention and saccade planning system
#[derive(Debug, Clone)]
pub struct AttentionSystem {
    /// Current focus of attention
    pub attention_center: (usize, usize),
    /// Attention map (salience)
    pub attention_map: Array2<f64>,
    /// Saccade targets
    pub saccade_targets: Vec<(usize, usize)>,
    /// Inhibition of return map
    pub inhibition_of_return: Array2<f64>,
    /// Feature-based attention weights
    pub feature_attention_weights: HashMap<String, f64>,
}

/// Predictive coding system
#[derive(Debug, Clone)]
pub struct PredictiveCodingSystem {
    /// Prediction models for each layer
    pub prediction_models: Vec<Array3<f64>>,
    /// Prediction errors
    pub prediction_errors: Vec<Array3<f64>>,
    /// Temporal predictions
    pub temporal_predictions: Vec<Array4<f64>>,
    /// Confidence estimates
    pub confidence_estimates: Vec<Array3<f64>>,
}

/// Color constancy system
#[derive(Debug, Clone)]
pub struct ColorConstancySystem {
    /// Illumination estimates
    pub illumination_estimates: Array2<(f64, f64, f64)>,
    /// Surface reflectance estimates
    pub surface_reflectance: Array2<(f64, f64, f64)>,
    /// Adaptation state
    pub adaptationstate: (f64, f64, f64),
    /// Color memory
    pub color_memory: Vec<(f64, f64, f64)>,
}

/// Motion tracking structure
#[derive(Debug, Clone)]
pub struct MotionTrack {
    /// Object identifier
    pub object_id: usize,
    /// Current position
    pub position: (f64, f64),
    /// Velocity vector
    pub velocity: (f64, f64),
    /// Position history
    pub positionhistory: VecDeque<(f64, f64)>,
    /// Predicted future positions
    pub predicted_positions: Vec<(f64, f64)>,
    /// Confidence in tracking
    pub confidence: f64,
    /// Track age
    pub age: usize,
}

impl CorticalLayer {
    /// Create a new cortical layer
    pub fn new(level: usize, height: usize, width: usize, receptive_field_size: usize) -> Self {
        Self {
            level,
            feature_maps: Array3::zeros((8, height, width)), // 8 feature channels
            receptive_field_size,
            lateral_connections: Array2::zeros((height, width)),
            top_down_predictions: Array3::zeros((8, height, width)),
            bottom_upfeatures: Array3::zeros((8, height, width)),
            prediction_errors: Array3::zeros((8, height, width)),
        }
    }
}

impl RetinaModel {
    /// Create a new retina model
    pub fn new(height: usize, width: usize) -> Self {
        Self {
            photoreceptors: Array2::zeros((height, width)),
            bipolar_cells: Array2::zeros((height, width)),
            horizontal_cells: Array2::zeros((height, width)),
            ganglion_cells: Array2::zeros((height, width)),
            center_surround_filters: Vec::new(),
        }
    }
}

impl Ommatidium {
    /// Create a new ommatidium
    pub fn new(position: (f64, f64), optical_axis: (f64, f64, f64)) -> Self {
        Self {
            position,
            optical_axis,
            response: 0.0,
            responsehistory: VecDeque::new(),
        }
    }
}
