//! Motion planning and trajectory evaluation metrics
//!
//! This module provides comprehensive metrics for evaluating robotic motion planning
//! and trajectory execution, including smoothness, optimality, and constraint satisfaction.

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use super::core::{ErrorStatistics, Pose, RealTimePerformanceMetrics, TrajectoryPoint};
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

/// A simple spherical obstacle for collision/clearance metrics.
///
/// Trajectories in this module are represented in Cartesian space
/// (`TrajectoryPoint::position`), so obstacles are modeled as bounding
/// spheres (center + radius) -- the standard simplification for
/// collision-clearance metrics when a full occupancy grid or mesh map isn't
/// available. Pass an empty obstacle slice when no obstacle map exists; all
/// obstacle-dependent metrics are then vacuously satisfied (there is nothing
/// to collide with) rather than fabricated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Obstacle {
    /// Obstacle center, in the same frame as `TrajectoryPoint::position`
    pub center: [f64; 3],
    /// Bounding sphere radius
    pub radius: f64,
}

impl Obstacle {
    /// Create a new spherical obstacle
    pub fn new(center: [f64; 3], radius: f64) -> Self {
        Self { center, radius }
    }

    /// Signed clearance from `position` to this obstacle's surface.
    /// Positive: outside the obstacle (safe). Negative: penetrating (collision).
    fn clearance(&self, position: &[f64; 3]) -> f64 {
        let dx = position[0] - self.center[0];
        let dy = position[1] - self.center[1];
        let dz = position[2] - self.center[2];
        (dx * dx + dy * dy + dz * dz).sqrt() - self.radius
    }
}

/// Nearest-obstacle clearance for a single position, or `f64::INFINITY` when
/// there are no obstacles to check against.
pub(super) fn nearest_clearance(position: &[f64; 3], obstacles: &[Obstacle]) -> f64 {
    obstacles
        .iter()
        .map(|o| o.clearance(position))
        .fold(f64::INFINITY, f64::min)
}

/// Returns `1.0` (violated) if any component of `values` exceeds the
/// corresponding limit in `limits`, `0.0` otherwise.
///
/// - `limits.len() >= 3`: each axis is checked independently against its own
///   limit (positional correspondence with `values`).
/// - `limits.len() == 1` or `2`: only an overall magnitude check against
///   `limits[0]` is meaningful (there is no way to know which axis a lone
///   limit refers to), so the Euclidean norm of `values` is compared to it.
/// - `limits.is_empty()`: no constraint was configured, so it is vacuously
///   satisfied.
fn check_axis_violation(values: &[f64; 3], limits: &[f64]) -> f64 {
    if limits.is_empty() {
        0.0
    } else if limits.len() >= 3 {
        let violated = values
            .iter()
            .zip(limits.iter())
            .any(|(&v, &limit)| v.abs() > limit);
        if violated {
            1.0
        } else {
            0.0
        }
    } else {
        let magnitude = (values[0].powi(2) + values[1].powi(2) + values[2].powi(2)).sqrt();
        if magnitude > limits[0] {
            1.0
        } else {
            0.0
        }
    }
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
                let acc_magnitude =
                    (acc_curr[0].powi(2) + acc_curr[1].powi(2) + acc_curr[2].powi(2)).sqrt();

                jerks.push(jerk_magnitude);
                accelerations.push(acc_magnitude);

                // Calculate curvature
                let vel_magnitude =
                    (vel_curr[0].powi(2) + vel_curr[1].powi(2) + vel_curr[2].powi(2)).sqrt();
                if vel_magnitude > 1e-6 && acc_magnitude > 1e-6 {
                    // Cross product for curvature calculation
                    let cross_product = [
                        vel_curr[1] * acc_curr[2] - vel_curr[2] * acc_curr[1],
                        vel_curr[2] * acc_curr[0] - vel_curr[0] * acc_curr[2],
                        vel_curr[0] * acc_curr[1] - vel_curr[1] * acc_curr[0],
                    ];
                    let cross_magnitude = (cross_product[0].powi(2)
                        + cross_product[1].powi(2)
                        + cross_product[2].powi(2))
                    .sqrt();
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
            accelerations
                .iter()
                .map(|a| (a - mean_acc).powi(2))
                .sum::<f64>()
                / accelerations.len() as f64
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
            curvatures
                .iter()
                .map(|c| (c - average_curvature).powi(2))
                .sum::<f64>()
                / curvatures.len() as f64
        } else {
            0.0
        };

        // Count sharp turns (curvature > threshold)
        let sharp_turn_threshold = 2.0; // rad/m (adjustable)
        let sharp_turns_count = curvatures
            .iter()
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

            let change_magnitude =
                (vel_change[0].powi(2) + vel_change[1].powi(2) + vel_change[2].powi(2)).sqrt();
            velocity_changes.push(change_magnitude);
        }

        // Smoothness is inversely related to velocity changes
        let average_change = velocity_changes.iter().sum::<f64>() / velocity_changes.len() as f64;
        1.0 / (1.0 + average_change) // Returns value between 0 and 1
    }

    /// Evaluate path optimality
    ///
    /// `obstacles` is used to compute real obstacle-clearance metrics; pass
    /// an empty slice if no obstacle map is available (clearance metrics
    /// then default to "no obstacles to clear", not a fabricated score).
    pub fn evaluate_path_optimality(
        &mut self,
        actual_path: &[TrajectoryPoint],
        optimal_path: Option<&[TrajectoryPoint]>,
        energy_consumption: f64,
        optimal_energy: f64,
        execution_time: Duration,
        optimal_time: Duration,
        obstacles: &[Obstacle],
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

        // Calculate real obstacle clearance from the actually-supplied obstacle set
        let obstacle_clearance = self.calculate_obstacle_clearance(actual_path, obstacles);

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

    /// Calculate real obstacle clearance metrics from the given obstacle set.
    ///
    /// Returns the default (vacuous "fully clear") metrics only when there is
    /// nothing to compute against (no obstacles or an empty path) -- not as a
    /// substitute for a real computation.
    fn calculate_obstacle_clearance(
        &self,
        path: &[TrajectoryPoint],
        obstacles: &[Obstacle],
    ) -> ObstacleClearanceMetrics {
        if obstacles.is_empty() || path.is_empty() {
            return ObstacleClearanceMetrics::default();
        }

        let clearances: Vec<f64> = path
            .iter()
            .map(|point| nearest_clearance(&point.position, obstacles))
            .collect();

        let min_clearance = clearances.iter().copied().fold(f64::INFINITY, f64::min);
        let avg_clearance = clearances.iter().sum::<f64>() / clearances.len() as f64;
        let clearance_variance = if clearances.len() > 1 {
            clearances
                .iter()
                .map(|c| (c - avg_clearance).powi(2))
                .sum::<f64>()
                / clearances.len() as f64
        } else {
            0.0
        };

        // Fraction of the path that stays outside every obstacle (no penetration)
        let safe_count = clearances.iter().filter(|&&c| c >= 0.0).count();
        let safety_margin_ratio = safe_count as f64 / clearances.len() as f64;

        ObstacleClearanceMetrics {
            min_clearance,
            avg_clearance,
            clearance_variance,
            safety_margin_ratio,
        }
    }

    /// Evaluate constraint satisfaction
    ///
    /// Computes real per-waypoint violation rates from the trajectory and the
    /// constraints/obstacles actually passed in:
    /// - **Joint limits**: this trajectory representation is Cartesian (no
    ///   joint angles are carried by `TrajectoryPoint`), so each configured
    ///   `(min, max)` pair in `constraints.joint_limits` is checked
    ///   positionally against the corresponding `position` component.
    /// - **Velocity limits**: checked against `linear_velocity`, per-axis
    ///   when 3+ limits are configured, otherwise as an overall magnitude
    ///   check (see the crate-private `check_axis_violation` helper).
    /// - **Acceleration limits**: acceleration is computed for real via a
    ///   backward finite difference of `linear_velocity` between consecutive
    ///   trajectory points, then checked the same way as velocity.
    /// - **Torque limits**: torque cannot be computed from this trajectory
    ///   representation -- no mass/inertia/applied-torque data is available
    ///   anywhere in `TrajectoryPoint` or `PlanningConstraints`. When
    ///   `constraints.torque_limits` is non-empty (the caller asked for a
    ///   torque check), this honestly reports `f64::NAN` ("not computable
    ///   from the available data") instead of a fabricated perfect score.
    ///   When empty, it is vacuously satisfied like the other constraint
    ///   types.
    /// - **Collision avoidance**: checked against `obstacles` (bounding
    ///   spheres); pass an empty slice when no obstacle map is available,
    ///   which is vacuously satisfied (nothing to collide with).
    pub fn evaluate_constraint_satisfaction(
        &mut self,
        trajectory: &[TrajectoryPoint],
        constraints: &PlanningConstraints,
        obstacles: &[Obstacle],
    ) -> Result<ConstraintSatisfactionMetrics> {
        let mut violations = ConstraintViolations::default();

        // Check each trajectory point against constraints
        for point in trajectory {
            // Joint limits: position components checked positionally against
            // the configured (min, max) ranges.
            let joint_violation = if constraints.joint_limits.is_empty() {
                0.0
            } else {
                let out_of_range = point
                    .position
                    .iter()
                    .zip(constraints.joint_limits.iter())
                    .any(|(&p, &(min, max))| p < min || p > max);
                if out_of_range {
                    1.0
                } else {
                    0.0
                }
            };

            // Velocity limits (real check against linear_velocity)
            let velocity_violation =
                check_axis_violation(&point.linear_velocity, &constraints.velocity_limits);

            // Collision avoidance against the actually-supplied obstacle set
            let collision_violation = if obstacles.is_empty() {
                0.0
            } else if nearest_clearance(&point.position, obstacles)
                < constraints.collision_threshold
            {
                1.0
            } else {
                0.0
            };

            violations.joint_violations.push(joint_violation);
            violations.velocity_violations.push(velocity_violation);
            violations.collision_violations.push(collision_violation);
        }

        // Acceleration limits: real finite-difference acceleration between
        // consecutive trajectory points (mirrors the derivative computation
        // used in `evaluate_trajectory_smoothness`).
        for i in 1..trajectory.len() {
            let dt = (trajectory[i].timestamp.as_secs_f64()
                - trajectory[i - 1].timestamp.as_secs_f64())
            .abs();
            if dt <= 0.0 {
                continue;
            }
            let accel = [
                (trajectory[i].linear_velocity[0] - trajectory[i - 1].linear_velocity[0]) / dt,
                (trajectory[i].linear_velocity[1] - trajectory[i - 1].linear_velocity[1]) / dt,
                (trajectory[i].linear_velocity[2] - trajectory[i - 1].linear_velocity[2]) / dt,
            ];
            violations
                .acceleration_violations
                .push(check_axis_violation(
                    &accel,
                    &constraints.acceleration_limits,
                ));
        }

        // Torque: genuinely not computable without a dynamics model (mass,
        // inertia, applied joint torques), none of which this trajectory
        // representation carries. Report the honest "not computed" sentinel
        // rather than a fabricated perfect score whenever a torque
        // constraint was actually configured.
        let torque_limits_satisfaction = if constraints.torque_limits.is_empty() {
            1.0
        } else {
            f64::NAN
        };

        let constraint_metrics = ConstraintSatisfactionMetrics {
            joint_limits_satisfaction: self
                .calculate_satisfaction_rate(&violations.joint_violations),
            velocity_limits_satisfaction: self
                .calculate_satisfaction_rate(&violations.velocity_violations),
            acceleration_limits_satisfaction: self
                .calculate_satisfaction_rate(&violations.acceleration_violations),
            torque_limits_satisfaction,
            collision_avoidance_rate: self
                .calculate_satisfaction_rate(&violations.collision_violations),
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

        let overall_score =
            (smoothness_score + efficiency_score + safety_score + feasibility_score) / 4.0;

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
        let curvature_score =
            1.0 / (1.0 + self.smoothness_metrics.curvature_metrics.average_curvature);
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
        let clearance_score = self
            .optimality_metrics
            .obstacle_clearance
            .safety_margin_ratio
            .min(1.0);
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
    pub collision_violations: Vec<f64>,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn point(position: [f64; 3], linear_velocity: [f64; 3], millis: u64) -> TrajectoryPoint {
        TrajectoryPoint::new(
            position,
            [1.0, 0.0, 0.0, 0.0],
            linear_velocity,
            [0.0, 0.0, 0.0],
            Duration::from_millis(millis),
        )
    }

    fn loose_constraints() -> PlanningConstraints {
        PlanningConstraints {
            joint_limits: vec![(-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)],
            velocity_limits: vec![5.0, 5.0, 5.0],
            acceleration_limits: vec![5.0, 5.0, 5.0],
            torque_limits: vec![],
            collision_threshold: 0.1,
        }
    }

    #[test]
    fn constraint_satisfaction_clean_trajectory_is_fully_satisfied() {
        let trajectory = vec![
            point([0.0, 0.0, 0.0], [0.1, 0.0, 0.0], 0),
            point([0.1, 0.0, 0.0], [0.1, 0.0, 0.0], 100),
            point([0.2, 0.0, 0.0], [0.1, 0.0, 0.0], 200),
            point([0.3, 0.0, 0.0], [0.1, 0.0, 0.0], 300),
        ];
        let mut metrics = MotionPlanningMetrics::new();
        let result = metrics
            .evaluate_constraint_satisfaction(&trajectory, &loose_constraints(), &[])
            .expect("evaluation should succeed");

        assert_eq!(result.joint_limits_satisfaction, 1.0);
        assert_eq!(result.velocity_limits_satisfaction, 1.0);
        assert_eq!(result.acceleration_limits_satisfaction, 1.0);
        assert_eq!(result.torque_limits_satisfaction, 1.0);
        assert_eq!(result.collision_avoidance_rate, 1.0);
    }

    #[test]
    fn constraint_satisfaction_detects_real_joint_limit_violation() {
        // One waypoint (x = 15) is well outside the configured [-10, 10] joint range.
        let trajectory = vec![
            point([0.0, 0.0, 0.0], [0.1, 0.0, 0.0], 0),
            point([15.0, 0.0, 0.0], [0.1, 0.0, 0.0], 100),
            point([0.2, 0.0, 0.0], [0.1, 0.0, 0.0], 200),
            point([0.3, 0.0, 0.0], [0.1, 0.0, 0.0], 300),
        ];
        let mut metrics = MotionPlanningMetrics::new();
        let result = metrics
            .evaluate_constraint_satisfaction(&trajectory, &loose_constraints(), &[])
            .expect("evaluation should succeed");

        // 3 of 4 waypoints are within range -- this must NOT be the old
        // hardcoded constant of 1.0.
        assert!((result.joint_limits_satisfaction - 0.75).abs() < 1e-9);
    }

    #[test]
    fn constraint_satisfaction_detects_real_velocity_limit_violation() {
        let trajectory = vec![
            point([0.0, 0.0, 0.0], [0.1, 0.0, 0.0], 0),
            point([0.1, 0.0, 0.0], [100.0, 0.0, 0.0], 100), // far beyond the 5.0 m/s limit
        ];
        let mut metrics = MotionPlanningMetrics::new();
        let result = metrics
            .evaluate_constraint_satisfaction(&trajectory, &loose_constraints(), &[])
            .expect("evaluation should succeed");

        assert!((result.velocity_limits_satisfaction - 0.5).abs() < 1e-9);
    }

    #[test]
    fn constraint_satisfaction_computes_real_acceleration_not_hardcoded_one() {
        // A huge velocity jump in a short time implies a huge acceleration
        // that must violate the configured 5.0 m/s^2 limit.
        let trajectory = vec![
            point([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0),
            point([0.0, 0.0, 0.0], [50.0, 0.0, 0.0], 100),
        ];
        let mut metrics = MotionPlanningMetrics::new();
        let result = metrics
            .evaluate_constraint_satisfaction(&trajectory, &loose_constraints(), &[])
            .expect("evaluation should succeed");

        assert_eq!(
            result.acceleration_limits_satisfaction, 0.0,
            "the old code hardcoded this to 1.0 regardless of the trajectory"
        );
    }

    #[test]
    fn constraint_satisfaction_reports_torque_honestly_as_not_computable() {
        let trajectory = vec![
            point([0.0, 0.0, 0.0], [0.1, 0.0, 0.0], 0),
            point([0.1, 0.0, 0.0], [0.1, 0.0, 0.0], 100),
        ];
        let mut constraints = loose_constraints();
        constraints.torque_limits = vec![50.0, 50.0, 50.0];

        let mut metrics = MotionPlanningMetrics::new();
        let result = metrics
            .evaluate_constraint_satisfaction(&trajectory, &constraints, &[])
            .expect("evaluation should succeed");

        // No mass/inertia/torque data is available anywhere in this trajectory
        // representation, so a torque check that was actually requested must
        // be reported as NaN ("not computable"), never a fabricated 1.0.
        assert!(result.torque_limits_satisfaction.is_nan());

        // But when no torque constraint was requested at all, it's vacuously satisfied.
        let unconstrained = loose_constraints();
        let mut metrics2 = MotionPlanningMetrics::new();
        let result2 = metrics2
            .evaluate_constraint_satisfaction(&trajectory, &unconstrained, &[])
            .expect("evaluation should succeed");
        assert_eq!(result2.torque_limits_satisfaction, 1.0);
    }

    #[test]
    fn constraint_satisfaction_detects_real_collision_with_obstacle() {
        let trajectory = vec![
            point([0.0, 0.0, 0.0], [0.1, 0.0, 0.0], 0),
            point([0.1, 0.0, 0.0], [0.1, 0.0, 0.0], 100),
            point([0.2, 0.0, 0.0], [0.1, 0.0, 0.0], 200),
        ];
        // A large obstacle centered on the path engulfs every waypoint.
        let blocking = vec![Obstacle::new([0.1, 0.0, 0.0], 0.5)];
        let mut metrics = MotionPlanningMetrics::new();
        let result = metrics
            .evaluate_constraint_satisfaction(&trajectory, &loose_constraints(), &blocking)
            .expect("evaluation should succeed");
        assert_eq!(
            result.collision_avoidance_rate, 0.0,
            "every waypoint is inside the obstacle's safety threshold"
        );

        // A far-away obstacle must not affect the same trajectory at all.
        let far_away = vec![Obstacle::new([1000.0, 1000.0, 1000.0], 0.1)];
        let mut metrics2 = MotionPlanningMetrics::new();
        let result2 = metrics2
            .evaluate_constraint_satisfaction(&trajectory, &loose_constraints(), &far_away)
            .expect("evaluation should succeed");
        assert_eq!(result2.collision_avoidance_rate, 1.0);
    }

    #[test]
    fn obstacle_clearance_is_computed_for_real_when_obstacles_are_supplied() {
        let path = vec![
            point([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0),
            point([1.0, 0.0, 0.0], [0.0, 0.0, 0.0], 100),
            point([2.0, 0.0, 0.0], [0.0, 0.0, 0.0], 200),
        ];
        let metrics = MotionPlanningMetrics::new();
        // No obstacles: vacuous default (fully clear).
        let clear = metrics.calculate_obstacle_clearance(&path, &[]);
        assert_eq!(clear.safety_margin_ratio, 1.0);

        // One obstacle exactly at the first waypoint blocks 1 of 3 points.
        let obstacles = vec![Obstacle::new([0.0, 0.0, 0.0], 0.5)];
        let blocked = metrics.calculate_obstacle_clearance(&path, &obstacles);
        assert!((blocked.safety_margin_ratio - 2.0 / 3.0).abs() < 1e-9);
        assert!(
            blocked.min_clearance < 0.0,
            "the first waypoint is inside the obstacle"
        );
    }

    #[test]
    fn path_optimality_uses_real_obstacle_clearance() {
        let path = vec![
            point([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0),
            point([1.0, 0.0, 0.0], [0.0, 0.0, 0.0], 100),
        ];
        let obstacles = vec![Obstacle::new([0.0, 0.0, 0.0], 5.0)]; // engulfs both points
        let mut metrics = MotionPlanningMetrics::new();
        let result = metrics
            .evaluate_path_optimality(
                &path,
                None,
                10.0,
                10.0,
                Duration::from_secs(1),
                Duration::from_secs(1),
                &obstacles,
            )
            .expect("evaluation should succeed");

        assert_eq!(result.obstacle_clearance.safety_margin_ratio, 0.0);
    }
}
