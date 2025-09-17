//! Manipulation and grasping metrics
//!
//! This module provides metrics for evaluating robotic manipulation tasks,
//! including grasping performance, manipulation accuracy, and force control.

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use super::core::{ErrorStatistics, Force, BoundingBox};
use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Manipulation task evaluation metrics
#[derive(Debug, Clone)]
pub struct ManipulationMetrics {
    /// Grasping performance metrics
    pub grasping_metrics: GraspingMetrics,
    /// Manipulation accuracy metrics
    pub manipulation_accuracy: ManipulationAccuracyMetrics,
    /// Task completion metrics
    pub task_completion: TaskCompletionMetrics,
    /// Force and contact metrics
    pub force_metrics: ForceContactMetrics,
}

/// Grasping performance evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraspingMetrics {
    /// Grasp success rate
    pub success_rate: f64,
    /// Grasp stability score
    pub stability_score: f64,
    /// Force closure quality
    pub force_closure_quality: f64,
    /// Approach trajectory quality
    pub approach_quality: f64,
    /// Grasp planning time
    pub planning_time: Duration,
    /// Object damage rate
    pub damage_rate: f64,
}

/// Manipulation accuracy metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManipulationAccuracyMetrics {
    /// Position accuracy (mm)
    pub position_accuracy: ErrorStatistics,
    /// Orientation accuracy (degrees)
    pub orientation_accuracy: ErrorStatistics,
    /// Trajectory following accuracy
    pub trajectory_accuracy: f64,
    /// Repeatability measure
    pub repeatability: f64,
}

/// Task completion evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskCompletionMetrics {
    /// Overall success rate
    pub success_rate: f64,
    /// Task completion time
    pub completion_time: Duration,
    /// Efficiency score
    pub efficiency_score: f64,
    /// Error recovery rate
    pub error_recovery_rate: f64,
    /// Quality of final result
    pub result_quality: f64,
}

/// Force and contact analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForceContactMetrics {
    /// Force control accuracy
    pub force_accuracy: ErrorStatistics,
    /// Contact stability
    pub contact_stability: f64,
    /// Force overshoot percentage
    pub force_overshoot: f64,
    /// Contact detection accuracy
    pub contact_detection_accuracy: f64,
    /// Compliance control quality
    pub compliance_quality: f64,
}

impl ManipulationMetrics {
    /// Create new manipulation metrics
    pub fn new() -> Self {
        Self {
            grasping_metrics: GraspingMetrics::default(),
            manipulation_accuracy: ManipulationAccuracyMetrics::default(),
            task_completion: TaskCompletionMetrics::default(),
            force_metrics: ForceContactMetrics::default(),
        }
    }
}

// Default implementations
impl Default for GraspingMetrics {
    fn default() -> Self {
        Self {
            success_rate: 1.0,
            stability_score: 1.0,
            force_closure_quality: 1.0,
            approach_quality: 1.0,
            planning_time: Duration::from_millis(0),
            damage_rate: 0.0,
        }
    }
}

impl Default for ManipulationAccuracyMetrics {
    fn default() -> Self {
        Self {
            position_accuracy: ErrorStatistics::default(),
            orientation_accuracy: ErrorStatistics::default(),
            trajectory_accuracy: 1.0,
            repeatability: 1.0,
        }
    }
}

impl Default for TaskCompletionMetrics {
    fn default() -> Self {
        Self {
            success_rate: 1.0,
            completion_time: Duration::from_secs(0),
            efficiency_score: 1.0,
            error_recovery_rate: 1.0,
            result_quality: 1.0,
        }
    }
}

impl Default for ForceContactMetrics {
    fn default() -> Self {
        Self {
            force_accuracy: ErrorStatistics::default(),
            contact_stability: 1.0,
            force_overshoot: 0.0,
            contact_detection_accuracy: 1.0,
            compliance_quality: 1.0,
        }
    }
}