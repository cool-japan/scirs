//! Motion planning and trajectory evaluation metrics
//!
//! This module provides comprehensive metrics for evaluating robotic motion planning
//! and trajectory execution, including smoothness, optimality, and constraint satisfaction.

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use super::core::{TrajectoryPoint, Pose, ErrorStatistics, RealTimePerformanceMetrics};
use crate::error::Result;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::numeric::Float;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Motion planning and trajectory evaluation metrics
#[derive(Debug, Clone)]
pub struct MotionPlanningMetrics {
    /// Trajectory smoothness measures
    pub smoothness_metrics: TrajectorySmoothnessMetrics,
    /// Path optimality metrics
    pub optimality_metrics: PathOptimalityMetrics,
    /// Dynamic constraints satisfaction
    pub constraint_metrics: ConstraintSatisfactionMetrics,
    /// Execution time and efficiency
    pub efficiency_metrics: PlanningEfficiencyMetrics,
}

/// Trajectory smoothness evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectorySmoothnessMetrics {
    /// Average jerk (third derivative of position)
    pub average_jerk: f64,
    /// Maximum jerk
    pub max_jerk: f64,
    /// Acceleration variance
    pub acceleration_variance: f64,
    /// Curvature analysis
    pub curvature_metrics: CurvatureMetrics,
    /// Velocity profile smoothness
    pub velocity_smoothness: f64,
}

/// Curvature analysis for trajectories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurvatureMetrics {
    /// Average curvature
    pub average_curvature: f64,
    /// Maximum curvature
    pub max_curvature: f64,
    /// Curvature variance
    pub curvature_variance: f64,
    /// Number of sharp turns (high curvature points)
    pub sharp_turns_count: usize,
}

/// Path optimality evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathOptimalityMetrics {
    /// Path length ratio to optimal
    pub length_optimality_ratio: f64,
    /// Energy consumption ratio
    pub energy_optimality_ratio: f64,
    /// Time optimality ratio
    pub time_optimality_ratio: f64,
    /// Clearance from obstacles
    pub obstacle_clearance: ObstacleClearanceMetrics,
}

/// Obstacle clearance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObstacleClearanceMetrics {
    /// Minimum clearance distance
    pub min_clearance: f64,
    /// Average clearance distance
    pub avg_clearance: f64,
    /// Clearance variance
    pub clearance_variance: f64,
    /// Safety margin ratio
    pub safety_margin_ratio: f64,
}

/// Constraint satisfaction metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintSatisfactionMetrics {
    /// Joint limits satisfaction rate
    pub joint_limits_satisfaction: f64,
    /// Velocity limits satisfaction rate
    pub velocity_limits_satisfaction: f64,
    /// Acceleration limits satisfaction rate
    pub acceleration_limits_satisfaction: f64,
    /// Torque limits satisfaction rate
    pub torque_limits_satisfaction: f64,
    /// Collision avoidance success rate
    pub collision_avoidance_rate: f64,
}

/// Planning efficiency metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanningEfficiencyMetrics {
    /// Planning computation time
    pub planning_time: Duration,
    /// Memory usage during planning
    pub memory_usage: usize,
    /// Number of iterations required
    pub iterations_count: usize,
    /// Success rate of planning
    pub planning_success_rate: f64,
    /// Convergence speed
    pub convergence_speed: f64,
}

/// Motion planning algorithm types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PlanningAlgorithm {
    /// Rapidly-exploring Random Tree
    RRT,
    /// RRT*
    RRTStar,
    /// Probabilistic Roadmap
    PRM,
    /// A* search
    AStar,
    /// Optimal sampling-based planner
    InformedRRTStar,
    /// Artificial Potential Field
    APF,
    /// Dynamic Window Approach
    DWA,
    /// Model Predictive Control
    MPC,
    /// Custom algorithm
    Custom(String),
}

/// Planning constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanningConstraints {
    /// Joint position limits (min, max) for each joint
    pub joint_limits: Vec<(f64, f64)>,
    /// Velocity limits for each joint
    pub velocity_limits: Vec<f64>,
    /// Acceleration limits for each joint
    pub acceleration_limits: Vec<f64>,
    /// Torque limits for each joint
    pub torque_limits: Vec<f64>,
    /// Maximum allowed collision probability
    pub collision_threshold: f64,
}

/// Trajectory quality assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryQuality {
    /// Overall quality score (0-1)
    pub overall_score: f64,
    /// Smoothness score (0-1)
    pub smoothness_score: f64,
    /// Efficiency score (0-1)
    pub efficiency_score: f64,
    /// Safety score (0-1)
    pub safety_score: f64,
    /// Feasibility score (0-1)
    pub feasibility_score: f64,
}

impl MotionPlanningMetrics {
    /// Create new motion planning metrics
    pub fn new() -> Self {
        Self {
            smoothness_metrics: TrajectorySmoothnessMetrics::default(),
            optimality_metrics: PathOptimalityMetrics::default(),
            constraint_metrics: ConstraintSatisfactionMetrics::default(),
            efficiency_metrics: PlanningEfficiencyMetrics::default(),
        }
    }

    /// Evaluate trajectory smoothness
    pub fn evaluate_trajectory_smoothness<F: Float>(
        &mut self,
        trajectory: &[TrajectoryPoint],
    ) -> Result<TrajectorySmoothnessMetrics> {
        if trajectory.len() < 3 {
            return Ok(TrajectorySmoothnessMetrics::default());
        }

        let mut jerks = Vec::new();
        let mut accelerations = Vec::new();
        let mut curvatures = Vec::new();

        // Calculate derivatives
        for i in 1..trajectory.len() - 1 {
            let dt1 = (trajectory[i].timestamp - trajectory[i - 1].timestamp).as_secs_f64();
            let dt2 = (trajectory[i + 1].timestamp - trajectory[i].timestamp).as_secs_f64();

            if dt1 > 0.0 && dt2 > 0.0 {
                // Calculate acceleration
                let vel_prev = [
                    trajectory[i - 1].linear_velocity[0],
                    trajectory[i - 1].linear_velocity[1],
                    trajectory[i - 1].linear_velocity[2],
                ];
                let vel_curr = [
                    trajectory[i].linear_velocity[0],
                    trajectory[i].linear_velocity[1],
                    trajectory[i].linear_velocity[2],
                ];
                let vel_next = [
                    trajectory[i + 1].linear_velocity[0],
                    trajectory[i + 1].linear_velocity[1],
                    trajectory[i + 1].linear_velocity[2],
                ];

                let acc_curr = [
                    (vel_curr[0] - vel_prev[0]) / dt1,
                    (vel_curr[1] - vel_prev[1]) / dt1,
                    (vel_curr[2] - vel_prev[2]) / dt1,
                ];

                let acc_next = [
                    (vel_next[0] - vel_curr[0]) / dt2,
                    (vel_next[1] - vel_curr[1]) / dt2,
                    (vel_next[2] - vel_curr[2]) / dt2,
                ];

                // Calculate jerk
                let jerk = [
                    (acc_next[0] - acc_curr[0]) / dt2,
                    (acc_next[1] - acc_curr[1]) / dt2,
                    (acc_next[2] - acc_curr[2]) / dt2,
                ];

                let jerk_magnitude = (jerk[0].powi(2) + jerk[1].powi(2) + jerk[2].powi(2)).sqrt();
                let acc_magnitude = (acc_curr[0].powi(2) + acc_curr[1].powi(2) + acc_curr[2].powi(2)).sqrt();

                jerks.push(jerk_magnitude);
                accelerations.push(acc_magnitude);

                // Calculate curvature
                let vel_magnitude = (vel_curr[0].powi(2) + vel_curr[1].powi(2) + vel_curr[2].powi(2)).sqrt();
                if vel_magnitude > 1e-6 && acc_magnitude > 1e-6 {
                    // Cross product for curvature calculation
                    let cross_product = [
                        vel_curr[1] * acc_curr[2] - vel_curr[2] * acc_curr[1],
                        vel_curr[2] * acc_curr[0] - vel_curr[0] * acc_curr[2],
                        vel_curr[0] * acc_curr[1] - vel_curr[1] * acc_curr[0],
                    ];
                    let cross_magnitude = (cross_product[0].powi(2) + cross_product[1].powi(2) + cross_product[2].powi(2)).sqrt();
                    let curvature = cross_magnitude / vel_magnitude.powi(3);
                    curvatures.push(curvature);
                }
            }
        }

        // Calculate metrics
        let average_jerk = if !jerks.is_empty() {
            jerks.iter().sum::<f64>() / jerks.len() as f64
        } else {
            0.0
        };

        let max_jerk = jerks.iter().copied().fold(0.0, f64::max);

        let acceleration_variance = if accelerations.len() > 1 {
            let mean_acc = accelerations.iter().sum::<f64>() / accelerations.len() as f64;
            accelerations.iter().map(|a| (a - mean_acc).powi(2)).sum::<f64>() / accelerations.len() as f64
        } else {
            0.0
        };

        let curvature_metrics = self.calculate_curvature_metrics(&curvatures);

        // Velocity smoothness (variation in velocity magnitude)
        let velocity_smoothness = self.calculate_velocity_smoothness(trajectory);

        let smoothness_metrics = TrajectorySmoothnessMetrics {
            average_jerk,
            max_jerk,
            acceleration_variance,
            curvature_metrics,
            velocity_smoothness,
        };

        self.smoothness_metrics = smoothness_metrics.clone();
        Ok(smoothness_metrics)
    }

    /// Calculate curvature metrics
    fn calculate_curvature_metrics(&self, curvatures: &[f64]) -> CurvatureMetrics {
        if curvatures.is_empty() {
            return CurvatureMetrics::default();
        }

        let average_curvature = curvatures.iter().sum::<f64>() / curvatures.len() as f64;
        let max_curvature = curvatures.iter().copied().fold(0.0, f64::max);

        let curvature_variance = if curvatures.len() > 1 {
            curvatures.iter()
                .map(|c| (c - average_curvature).powi(2))
                .sum::<f64>() / curvatures.len() as f64
        } else {
            0.0
        };

        // Count sharp turns (curvature > threshold)
        let sharp_turn_threshold = 2.0; // rad/m (adjustable)
        let sharp_turns_count = curvatures.iter()
            .filter(|&&c| c > sharp_turn_threshold)
            .count();

        CurvatureMetrics {
            average_curvature,
            max_curvature,
            curvature_variance,
            sharp_turns_count,
        }
    }

    /// Calculate velocity smoothness
    fn calculate_velocity_smoothness(&self, trajectory: &[TrajectoryPoint]) -> f64 {
        if trajectory.len() < 2 {
            return 1.0; // Perfect smoothness for trivial cases
        }

        let mut velocity_changes = Vec::new();

        for i in 1..trajectory.len() {
            let vel_prev = [
                trajectory[i - 1].linear_velocity[0],
                trajectory[i - 1].linear_velocity[1],
                trajectory[i - 1].linear_velocity[2],
            ];
            let vel_curr = [
                trajectory[i].linear_velocity[0],
                trajectory[i].linear_velocity[1],
                trajectory[i].linear_velocity[2],
            ];

            let vel_change = [
                vel_curr[0] - vel_prev[0],
                vel_curr[1] - vel_prev[1],
                vel_curr[2] - vel_prev[2],
            ];

            let change_magnitude = (vel_change[0].powi(2) + vel_change[1].powi(2) + vel_change[2].powi(2)).sqrt();
            velocity_changes.push(change_magnitude);
        }

        // Smoothness is inversely related to velocity changes
        let average_change = velocity_changes.iter().sum::<f64>() / velocity_changes.len() as f64;
        1.0 / (1.0 + average_change) // Returns value between 0 and 1
    }

    /// Evaluate path optimality
    pub fn evaluate_path_optimality(
        &mut self,
        actual_path: &[TrajectoryPoint],
        optimal_path: Option<&[TrajectoryPoint]>,
        energy_consumption: f64,
        optimal_energy: f64,
        execution_time: Duration,
        optimal_time: Duration,
    ) -> Result<PathOptimalityMetrics> {
        // Calculate length optimality
        let actual_length = self.calculate_path_length(actual_path);
        let optimal_length = if let Some(opt_path) = optimal_path {
            self.calculate_path_length(opt_path)
        } else {
            actual_length // If no optimal path provided, assume current is optimal
        };

        let length_optimality_ratio = if optimal_length > 0.0 {
            actual_length / optimal_length
        } else {
            1.0
        };

        // Energy optimality
        let energy_optimality_ratio = if optimal_energy > 0.0 {
            energy_consumption / optimal_energy
        } else {
            1.0
        };

        // Time optimality
        let time_optimality_ratio = if optimal_time.as_secs_f64() > 0.0 {
            execution_time.as_secs_f64() / optimal_time.as_secs_f64()
        } else {
            1.0
        };

        // Calculate obstacle clearance (simplified - would need obstacle map in practice)
        let obstacle_clearance = self.calculate_obstacle_clearance(actual_path);

        let optimality_metrics = PathOptimalityMetrics {
            length_optimality_ratio,
            energy_optimality_ratio,
            time_optimality_ratio,
            obstacle_clearance,
        };

        self.optimality_metrics = optimality_metrics.clone();
        Ok(optimality_metrics)
    }

    /// Calculate path length
    fn calculate_path_length(&self, path: &[TrajectoryPoint]) -> f64 {
        if path.len() < 2 {
            return 0.0;
        }

        let mut total_length = 0.0;
        for i in 1..path.len() {
            total_length += path[i - 1].distance_to(&path[i]);
        }
        total_length
    }

    /// Calculate obstacle clearance metrics (simplified)
    fn calculate_obstacle_clearance(&self, path: &[TrajectoryPoint]) -> ObstacleClearanceMetrics {
        // In a real implementation, this would require an obstacle map
        // For now, we'll return default values
        ObstacleClearanceMetrics::default()
    }

    /// Evaluate constraint satisfaction
    pub fn evaluate_constraint_satisfaction(
        &mut self,
        trajectory: &[TrajectoryPoint],
        constraints: &PlanningConstraints,
    ) -> Result<ConstraintSatisfactionMetrics> {
        let mut violations = ConstraintViolations::default();

        // Check each trajectory point against constraints
        for point in trajectory {
            // Note: In practice, this would require joint angles and velocities
            // For simplification, we'll use placeholder calculations

            // Joint limits (placeholder - would need joint angles)
            let joint_satisfaction = 1.0; // Assume satisfied for now

            // Velocity limits (using linear velocity as proxy)
            let velocity_magnitude = (point.linear_velocity[0].powi(2) +
                                    point.linear_velocity[1].powi(2) +
                                    point.linear_velocity[2].powi(2)).sqrt();
            let velocity_satisfaction = if !constraints.velocity_limits.is_empty() {
                if velocity_magnitude <= constraints.velocity_limits[0] { 1.0 } else { 0.0 }
            } else {
                1.0
            };

            violations.joint_violations.push(1.0 - joint_satisfaction);
            violations.velocity_violations.push(1.0 - velocity_satisfaction);
        }

        let constraint_metrics = ConstraintSatisfactionMetrics {
            joint_limits_satisfaction: self.calculate_satisfaction_rate(&violations.joint_violations),
            velocity_limits_satisfaction: self.calculate_satisfaction_rate(&violations.velocity_violations),
            acceleration_limits_satisfaction: 1.0, // Placeholder
            torque_limits_satisfaction: 1.0, // Placeholder
            collision_avoidance_rate: 1.0, // Placeholder
        };

        self.constraint_metrics = constraint_metrics.clone();
        Ok(constraint_metrics)
    }

    /// Calculate satisfaction rate from violations
    fn calculate_satisfaction_rate(&self, violations: &[f64]) -> f64 {
        if violations.is_empty() {
            return 1.0;
        }

        let satisfied_count = violations.iter().filter(|&&v| v == 0.0).count();
        satisfied_count as f64 / violations.len() as f64
    }

    /// Evaluate overall trajectory quality
    pub fn evaluate_trajectory_quality(&self) -> TrajectoryQuality {
        let smoothness_score = self.calculate_smoothness_score();
        let efficiency_score = self.calculate_efficiency_score();
        let safety_score = self.calculate_safety_score();
        let feasibility_score = self.calculate_feasibility_score();

        let overall_score = (smoothness_score + efficiency_score + safety_score + feasibility_score) / 4.0;

        TrajectoryQuality {
            overall_score,
            smoothness_score,
            efficiency_score,
            safety_score,
            feasibility_score,
        }
    }

    /// Calculate smoothness score (0-1)
    fn calculate_smoothness_score(&self) -> f64 {
        // Combine various smoothness metrics
        let jerk_score = 1.0 / (1.0 + self.smoothness_metrics.average_jerk);
        let acceleration_score = 1.0 / (1.0 + self.smoothness_metrics.acceleration_variance);
        let curvature_score = 1.0 / (1.0 + self.smoothness_metrics.curvature_metrics.average_curvature);
        let velocity_score = self.smoothness_metrics.velocity_smoothness;

        (jerk_score + acceleration_score + curvature_score + velocity_score) / 4.0
    }

    /// Calculate efficiency score (0-1)
    fn calculate_efficiency_score(&self) -> f64 {
        let length_score = 1.0 / self.optimality_metrics.length_optimality_ratio.max(1.0);
        let energy_score = 1.0 / self.optimality_metrics.energy_optimality_ratio.max(1.0);
        let time_score = 1.0 / self.optimality_metrics.time_optimality_ratio.max(1.0);

        (length_score + energy_score + time_score) / 3.0
    }

    /// Calculate safety score (0-1)
    fn calculate_safety_score(&self) -> f64 {
        let clearance_score = self.optimality_metrics.obstacle_clearance.safety_margin_ratio.min(1.0);
        let collision_score = self.constraint_metrics.collision_avoidance_rate;

        (clearance_score + collision_score) / 2.0
    }

    /// Calculate feasibility score (0-1)
    fn calculate_feasibility_score(&self) -> f64 {
        let joint_score = self.constraint_metrics.joint_limits_satisfaction;
        let velocity_score = self.constraint_metrics.velocity_limits_satisfaction;
        let acceleration_score = self.constraint_metrics.acceleration_limits_satisfaction;
        let torque_score = self.constraint_metrics.torque_limits_satisfaction;

        (joint_score + velocity_score + acceleration_score + torque_score) / 4.0
    }
}

/// Constraint violations tracking
#[derive(Debug, Default)]
struct ConstraintViolations {
    pub joint_violations: Vec<f64>,
    pub velocity_violations: Vec<f64>,
    pub acceleration_violations: Vec<f64>,
    pub torque_violations: Vec<f64>,
}

// Default implementations
impl Default for TrajectorySmoothnessMetrics {
    fn default() -> Self {
        Self {
            average_jerk: 0.0,
            max_jerk: 0.0,
            acceleration_variance: 0.0,
            curvature_metrics: CurvatureMetrics::default(),
            velocity_smoothness: 1.0,
        }
    }
}

impl Default for CurvatureMetrics {
    fn default() -> Self {
        Self {
            average_curvature: 0.0,
            max_curvature: 0.0,
            curvature_variance: 0.0,
            sharp_turns_count: 0,
        }
    }
}

impl Default for PathOptimalityMetrics {
    fn default() -> Self {
        Self {
            length_optimality_ratio: 1.0,
            energy_optimality_ratio: 1.0,
            time_optimality_ratio: 1.0,
            obstacle_clearance: ObstacleClearanceMetrics::default(),
        }
    }
}

impl Default for ObstacleClearanceMetrics {
    fn default() -> Self {
        Self {
            min_clearance: 0.0,
            avg_clearance: 0.0,
            clearance_variance: 0.0,
            safety_margin_ratio: 1.0,
        }
    }
}

impl Default for ConstraintSatisfactionMetrics {
    fn default() -> Self {
        Self {
            joint_limits_satisfaction: 1.0,
            velocity_limits_satisfaction: 1.0,
            acceleration_limits_satisfaction: 1.0,
            torque_limits_satisfaction: 1.0,
            collision_avoidance_rate: 1.0,
        }
    }
}

impl Default for PlanningEfficiencyMetrics {
    fn default() -> Self {
        Self {
            planning_time: Duration::from_millis(0),
            memory_usage: 0,
            iterations_count: 0,
            planning_success_rate: 1.0,
            convergence_speed: 1.0,
        }
    }
}

impl Default for PlanningConstraints {
    fn default() -> Self {
        Self {
            joint_limits: Vec::new(),
            velocity_limits: Vec::new(),
            acceleration_limits: Vec::new(),
            torque_limits: Vec::new(),
            collision_threshold: 0.01,
        }
    }
}

impl Default for TrajectoryQuality {
    fn default() -> Self {
        Self {
            overall_score: 1.0,
            smoothness_score: 1.0,
            efficiency_score: 1.0,
            safety_score: 1.0,
            feasibility_score: 1.0,
        }
    }
}