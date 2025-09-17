//! Core types and configuration for robotics metrics
//!
//! This module contains the main RoboticsMetrics structure and shared types
//! used throughout the robotics evaluation system.

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use super::super::{DomainEvaluationResult, DomainMetrics};
use crate::error::{MetricsError, Result};
use ndarray::{Array1, Array2};
use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

// Re-export from submodules
use super::motion_planning::MotionPlanningMetrics;
use super::slam_localization::SlamMetrics;
use super::manipulation::ManipulationMetrics;
use super::navigation::NavigationMetrics;
use super::human_robot_interaction::HumanRobotInteractionMetrics;
use super::multi_robot::MultiRobotMetrics;
use super::safety_reliability::SafetyReliabilityMetrics;
use super::perception::RoboticPerceptionMetrics;

/// Comprehensive robotics evaluation metrics suite
#[derive(Debug)]
pub struct RoboticsMetrics {
    /// Motion planning and control metrics
    pub motion_metrics: MotionPlanningMetrics,
    /// SLAM and localization metrics
    pub slam_metrics: SlamMetrics,
    /// Manipulation task metrics
    pub manipulation_metrics: ManipulationMetrics,
    /// Navigation metrics
    pub navigation_metrics: NavigationMetrics,
    /// Human-robot interaction metrics
    pub hri_metrics: HumanRobotInteractionMetrics,
    /// Multi-robot coordination metrics
    pub multi_robot_metrics: MultiRobotMetrics,
    /// Safety and reliability metrics
    pub safety_metrics: SafetyReliabilityMetrics,
    /// Perception metrics for robotics
    pub perception_metrics: RoboticPerceptionMetrics,
}

/// Common error statistics used across multiple robotics domains
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorStatistics {
    /// Mean error
    pub mean_error: f64,
    /// Standard deviation of error
    pub std_error: f64,
    /// Root mean square error
    pub rmse: f64,
    /// Maximum error
    pub max_error: f64,
    /// Median error
    pub median_error: f64,
    /// 95th percentile error
    pub percentile_95_error: f64,
}

/// Common drift metrics used in robotics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftMetrics {
    /// Linear drift rate (units per time)
    pub linear_drift_rate: f64,
    /// Angular drift rate (radians per time)
    pub angular_drift_rate: f64,
    /// Cumulative drift over time
    pub cumulative_drift: f64,
    /// Drift consistency (lower is better)
    pub drift_consistency: f64,
}

/// Real-time performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimePerformanceMetrics {
    /// Average processing time per frame/iteration
    pub average_processing_time: Duration,
    /// Maximum processing time
    pub max_processing_time: Duration,
    /// Processing time variance
    pub processing_time_variance: f64,
    /// Frame rate (Hz)
    pub frame_rate: f64,
    /// Percentage of real-time deadlines met
    pub real_time_compliance: f64,
    /// Memory usage statistics
    pub memory_usage: MemoryUsageMetrics,
}

/// Memory usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsageMetrics {
    /// Average memory usage (MB)
    pub average_memory_mb: f64,
    /// Peak memory usage (MB)
    pub peak_memory_mb: f64,
    /// Memory allocation rate (allocations per second)
    pub allocation_rate: f64,
    /// Memory fragmentation percentage
    pub fragmentation_percentage: f64,
}

/// Configuration parameters for robotics evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoboticsEvaluationConfig {
    /// Time window for evaluation (seconds)
    pub evaluation_window: Duration,
    /// Sampling frequency (Hz)
    pub sampling_frequency: f64,
    /// Enable detailed analysis
    pub enable_detailed_analysis: bool,
    /// Coordinate system type
    pub coordinate_system: CoordinateSystem,
    /// Evaluation mode
    pub evaluation_mode: EvaluationMode,
}

/// Coordinate system types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinateSystem {
    /// World coordinate system
    World,
    /// Robot base coordinate system
    RobotBase,
    /// Camera coordinate system
    Camera,
    /// Custom coordinate system
    Custom(String),
}

/// Evaluation modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvaluationMode {
    /// Real-time evaluation
    RealTime,
    /// Offline analysis
    Offline,
    /// Benchmark comparison
    Benchmark,
    /// Simulation mode
    Simulation,
}

/// Common trajectory point representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryPoint {
    /// Position (x, y, z)
    pub position: [f64; 3],
    /// Orientation (quaternion: w, x, y, z)
    pub orientation: [f64; 4],
    /// Linear velocity
    pub linear_velocity: [f64; 3],
    /// Angular velocity
    pub angular_velocity: [f64; 3],
    /// Timestamp
    pub timestamp: Duration,
}

/// Common pose representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pose {
    /// Position (x, y, z)
    pub position: [f64; 3],
    /// Orientation (quaternion: w, x, y, z)
    pub orientation: [f64; 4],
    /// Confidence/uncertainty
    pub confidence: f64,
}

/// Common velocity representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Velocity {
    /// Linear velocity (x, y, z)
    pub linear: [f64; 3],
    /// Angular velocity (x, y, z)
    pub angular: [f64; 3],
}

/// Common acceleration representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Acceleration {
    /// Linear acceleration (x, y, z)
    pub linear: [f64; 3],
    /// Angular acceleration (x, y, z)
    pub angular: [f64; 3],
}

/// Common force representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Force {
    /// Force vector (x, y, z)
    pub force: [f64; 3],
    /// Torque vector (x, y, z)
    pub torque: [f64; 3],
    /// Application point
    pub application_point: [f64; 3],
}

/// Common bounding box representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    /// Minimum coordinates (x, y, z)
    pub min: [f64; 3],
    /// Maximum coordinates (x, y, z)
    pub max: [f64; 3],
    /// Center point
    pub center: [f64; 3],
    /// Dimensions (width, height, depth)
    pub dimensions: [f64; 3],
}

impl Default for RoboticsEvaluationConfig {
    fn default() -> Self {
        Self {
            evaluation_window: Duration::from_secs(60),
            sampling_frequency: 30.0,
            enable_detailed_analysis: true,
            coordinate_system: CoordinateSystem::World,
            evaluation_mode: EvaluationMode::RealTime,
        }
    }
}

impl Default for ErrorStatistics {
    fn default() -> Self {
        Self {
            mean_error: 0.0,
            std_error: 0.0,
            rmse: 0.0,
            max_error: 0.0,
            median_error: 0.0,
            percentile_95_error: 0.0,
        }
    }
}

impl Default for DriftMetrics {
    fn default() -> Self {
        Self {
            linear_drift_rate: 0.0,
            angular_drift_rate: 0.0,
            cumulative_drift: 0.0,
            drift_consistency: 0.0,
        }
    }
}

impl Default for RealTimePerformanceMetrics {
    fn default() -> Self {
        Self {
            average_processing_time: Duration::from_millis(33), // ~30 FPS
            max_processing_time: Duration::from_millis(100),
            processing_time_variance: 0.0,
            frame_rate: 30.0,
            real_time_compliance: 100.0,
            memory_usage: MemoryUsageMetrics::default(),
        }
    }
}

impl Default for MemoryUsageMetrics {
    fn default() -> Self {
        Self {
            average_memory_mb: 100.0,
            peak_memory_mb: 150.0,
            allocation_rate: 1000.0,
            fragmentation_percentage: 5.0,
        }
    }
}

impl TrajectoryPoint {
    /// Create a new trajectory point
    pub fn new(
        position: [f64; 3],
        orientation: [f64; 4],
        linear_velocity: [f64; 3],
        angular_velocity: [f64; 3],
        timestamp: Duration,
    ) -> Self {
        Self {
            position,
            orientation,
            linear_velocity,
            angular_velocity,
            timestamp,
        }
    }

    /// Calculate distance to another trajectory point
    pub fn distance_to(&self, other: &TrajectoryPoint) -> f64 {
        let dx = self.position[0] - other.position[0];
        let dy = self.position[1] - other.position[1];
        let dz = self.position[2] - other.position[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

impl Pose {
    /// Create a new pose
    pub fn new(position: [f64; 3], orientation: [f64; 4], confidence: f64) -> Self {
        Self {
            position,
            orientation,
            confidence,
        }
    }

    /// Calculate distance to another pose
    pub fn distance_to(&self, other: &Pose) -> f64 {
        let dx = self.position[0] - other.position[0];
        let dy = self.position[1] - other.position[1];
        let dz = self.position[2] - other.position[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Calculate angular difference to another pose (in radians)
    pub fn angular_distance_to(&self, other: &Pose) -> f64 {
        // Simplified quaternion distance calculation
        let dot_product = self.orientation[0] * other.orientation[0] +
                         self.orientation[1] * other.orientation[1] +
                         self.orientation[2] * other.orientation[2] +
                         self.orientation[3] * other.orientation[3];
        2.0 * dot_product.abs().acos()
    }
}

impl BoundingBox {
    /// Create a new bounding box
    pub fn new(min: [f64; 3], max: [f64; 3]) -> Self {
        let center = [
            (min[0] + max[0]) / 2.0,
            (min[1] + max[1]) / 2.0,
            (min[2] + max[2]) / 2.0,
        ];
        let dimensions = [
            max[0] - min[0],
            max[1] - min[1],
            max[2] - min[2],
        ];

        Self {
            min,
            max,
            center,
            dimensions,
        }
    }

    /// Calculate volume of the bounding box
    pub fn volume(&self) -> f64 {
        self.dimensions[0] * self.dimensions[1] * self.dimensions[2]
    }

    /// Check if a point is inside the bounding box
    pub fn contains_point(&self, point: [f64; 3]) -> bool {
        point[0] >= self.min[0] && point[0] <= self.max[0] &&
        point[1] >= self.min[1] && point[1] <= self.max[1] &&
        point[2] >= self.min[2] && point[2] <= self.max[2]
    }

    /// Calculate intersection with another bounding box
    pub fn intersection(&self, other: &BoundingBox) -> Option<BoundingBox> {
        let min_x = self.min[0].max(other.min[0]);
        let min_y = self.min[1].max(other.min[1]);
        let min_z = self.min[2].max(other.min[2]);
        let max_x = self.max[0].min(other.max[0]);
        let max_y = self.max[1].min(other.max[1]);
        let max_z = self.max[2].min(other.max[2]);

        if min_x <= max_x && min_y <= max_y && min_z <= max_z {
            Some(BoundingBox::new([min_x, min_y, min_z], [max_x, max_y, max_z]))
        } else {
            None
        }
    }
}

/// Utility functions for robotics calculations
pub mod utils {
    use super::*;

    /// Calculate quaternion from Euler angles (roll, pitch, yaw)
    pub fn euler_to_quaternion(roll: f64, pitch: f64, yaw: f64) -> [f64; 4] {
        let cr = (roll * 0.5).cos();
        let sr = (roll * 0.5).sin();
        let cp = (pitch * 0.5).cos();
        let sp = (pitch * 0.5).sin();
        let cy = (yaw * 0.5).cos();
        let sy = (yaw * 0.5).sin();

        [
            cr * cp * cy + sr * sp * sy, // w
            sr * cp * cy - cr * sp * sy, // x
            cr * sp * cy + sr * cp * sy, // y
            cr * cp * sy - sr * sp * cy, // z
        ]
    }

    /// Calculate Euler angles from quaternion
    pub fn quaternion_to_euler(q: [f64; 4]) -> [f64; 3] {
        let w = q[0];
        let x = q[1];
        let y = q[2];
        let z = q[3];

        // Roll (x-axis rotation)
        let sin_r_cp = 2.0 * (w * x + y * z);
        let cos_r_cp = 1.0 - 2.0 * (x * x + y * y);
        let roll = sin_r_cp.atan2(cos_r_cp);

        // Pitch (y-axis rotation)
        let sin_p = 2.0 * (w * y - z * x);
        let pitch = if sin_p.abs() >= 1.0 {
            std::f64::consts::PI / 2.0 * sin_p.signum()
        } else {
            sin_p.asin()
        };

        // Yaw (z-axis rotation)
        let sin_y_cp = 2.0 * (w * z + x * y);
        let cos_y_cp = 1.0 - 2.0 * (y * y + z * z);
        let yaw = sin_y_cp.atan2(cos_y_cp);

        [roll, pitch, yaw]
    }

    /// Calculate statistical measures for a series of values
    pub fn calculate_statistics(values: &[f64]) -> ErrorStatistics {
        if values.is_empty() {
            return ErrorStatistics::default();
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();
        let rmse = (values.iter().map(|x| x.powi(2)).sum::<f64>() / values.len() as f64).sqrt();

        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median = if sorted_values.len() % 2 == 0 {
            (sorted_values[sorted_values.len() / 2 - 1] + sorted_values[sorted_values.len() / 2]) / 2.0
        } else {
            sorted_values[sorted_values.len() / 2]
        };

        let percentile_95_idx = ((sorted_values.len() as f64) * 0.95) as usize;
        let percentile_95 = sorted_values.get(percentile_95_idx).copied().unwrap_or(0.0);

        ErrorStatistics {
            mean_error: mean,
            std_error: std_dev,
            rmse,
            max_error: values.iter().copied().fold(0.0, f64::max),
            median_error: median,
            percentile_95_error: percentile_95,
        }
    }
}