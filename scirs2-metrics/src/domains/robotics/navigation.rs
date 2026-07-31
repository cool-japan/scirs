//! Navigation and path planning metrics
//!
//! This module provides metrics for evaluating robotic navigation systems,
//! including path planning, obstacle avoidance, and goal-reaching performance.

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use super::core::{BoundingBox, Pose, TrajectoryPoint};
use super::motion_planning::{nearest_clearance, Obstacle};
use crate::error::{MetricsError, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Outcome of a single goal-reaching attempt, used to compute real
/// [`GoalReachingMetrics`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalReachingAttempt {
    /// The robot's actual final pose when the attempt ended.
    pub final_pose: Pose,
    /// The commanded goal pose.
    pub goal_pose: Pose,
    /// Wall-clock time taken to reach (or abandon) the goal.
    pub completion_time: Duration,
    /// Energy consumed during this attempt (any consistent unit).
    pub energy_used: f64,
}

/// A single path-planning trial, used to compute real [`PathPlanningMetrics`].
#[derive(Debug, Clone)]
pub struct PathPlanningAttempt {
    /// Whether the planner found a valid path at all.
    pub success: bool,
    /// The planned path (only meaningful when `success` is `true`).
    pub planned_path: Vec<TrajectoryPoint>,
    /// The known optimal path length for this scenario, if available.
    pub optimal_length: Option<f64>,
    /// Wall-clock time spent planning.
    pub planning_time: Duration,
}

/// A single environmental-change event during navigation, used to compute
/// real [`DynamicAdaptationMetrics`].
#[derive(Debug, Clone)]
pub struct AdaptationEvent {
    /// Time between the environmental change and the system's response.
    pub response_time: Duration,
    /// Whether the system triggered a replan in response.
    pub replanned: bool,
    /// Whether the robot successfully continued its task afterward (whether
    /// or not it replanned).
    pub adaptation_successful: bool,
}

/// Navigation system evaluation metrics
#[derive(Debug, Clone)]
pub struct NavigationMetrics {
    /// Path planning performance
    pub path_planning: PathPlanningMetrics,
    /// Obstacle avoidance metrics
    pub obstacle_avoidance: ObstacleAvoidanceMetrics,
    /// Goal reaching performance
    pub goal_reaching: GoalReachingMetrics,
    /// Dynamic adaptation capabilities
    pub dynamic_adaptation: DynamicAdaptationMetrics,
}

/// Path planning evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathPlanningMetrics {
    /// Planning success rate
    pub success_rate: f64,
    /// Planning time
    pub planning_time: Duration,
    /// Path optimality (length ratio to optimal)
    pub path_optimality: f64,
    /// Path smoothness score
    pub smoothness: f64,
    /// Computational efficiency
    pub computational_efficiency: f64,
}

/// Obstacle avoidance performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObstacleAvoidanceMetrics {
    /// Collision avoidance success rate
    pub collision_avoidance_rate: f64,
    /// Minimum clearance distance
    pub min_clearance: f64,
    /// Average clearance distance
    pub avg_clearance: f64,
    /// Reaction time to new obstacles
    pub reaction_time: Duration,
    /// Path deviation due to avoidance
    pub path_deviation: f64,
}

/// Goal reaching performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalReachingMetrics {
    /// Success rate
    pub success_rate: f64,
    /// Final position accuracy
    pub position_accuracy: f64,
    /// Final orientation accuracy
    pub orientation_accuracy: f64,
    /// Time to reach goal
    pub completion_time: Duration,
    /// Energy efficiency
    pub energy_efficiency: f64,
}

/// Dynamic adaptation capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicAdaptationMetrics {
    /// Response time to environmental changes
    pub adaptation_time: Duration,
    /// Replanning frequency
    pub replanning_frequency: f64,
    /// Adaptation success rate
    pub adaptation_success_rate: f64,
    /// Robustness to disturbances
    pub disturbance_robustness: f64,
}

impl NavigationMetrics {
    /// Create new navigation metrics
    pub fn new() -> Self {
        Self {
            path_planning: PathPlanningMetrics::default(),
            obstacle_avoidance: ObstacleAvoidanceMetrics::default(),
            goal_reaching: GoalReachingMetrics::default(),
            dynamic_adaptation: DynamicAdaptationMetrics::default(),
        }
    }

    /// Evaluate goal-reaching performance from a real log of navigation
    /// attempts. An attempt "succeeds" when both its final position and
    /// orientation land within the given tolerances of the goal.
    pub fn evaluate_goal_reaching(
        &mut self,
        attempts: &[GoalReachingAttempt],
        position_tolerance: f64,
        orientation_tolerance: f64,
        optimal_energy: f64,
    ) -> Result<GoalReachingMetrics> {
        if attempts.is_empty() {
            return Err(MetricsError::InvalidInput(
                "goal-reaching attempt log must not be empty".to_string(),
            ));
        }

        let total = attempts.len() as f64;
        let position_errors: Vec<f64> = attempts
            .iter()
            .map(|a| a.final_pose.distance_to(&a.goal_pose))
            .collect();
        let orientation_errors: Vec<f64> = attempts
            .iter()
            .map(|a| a.final_pose.angular_distance_to(&a.goal_pose))
            .collect();

        let successes = position_errors
            .iter()
            .zip(orientation_errors.iter())
            .filter(|(&p, &o)| p <= position_tolerance && o <= orientation_tolerance)
            .count();
        let success_rate = successes as f64 / total;

        let position_accuracy = position_errors.iter().sum::<f64>() / total;
        let orientation_accuracy = orientation_errors.iter().sum::<f64>() / total;

        let total_nanos: u128 = attempts.iter().map(|a| a.completion_time.as_nanos()).sum();
        let completion_time = Duration::from_nanos((total_nanos / attempts.len() as u128) as u64);

        let mean_energy = attempts.iter().map(|a| a.energy_used).sum::<f64>() / total;
        let energy_efficiency = if mean_energy > 0.0 {
            (optimal_energy / mean_energy).min(1.0)
        } else {
            1.0
        };

        let result = GoalReachingMetrics {
            success_rate,
            position_accuracy,
            orientation_accuracy,
            completion_time,
            energy_efficiency,
        };
        self.goal_reaching = result.clone();
        Ok(result)
    }

    /// Evaluate path-planning performance from a real log of planning trials.
    ///
    /// `path_optimality` averages `actual_length / optimal_length` over
    /// attempts that supply a known optimal length; when none do, it is
    /// honestly `1.0` (nothing to compare against) rather than fabricated.
    /// `smoothness` is derived from the actual turning angle between
    /// consecutive path segments (higher for straighter paths).
    pub fn evaluate_path_planning(
        &mut self,
        attempts: &[PathPlanningAttempt],
    ) -> Result<PathPlanningMetrics> {
        if attempts.is_empty() {
            return Err(MetricsError::InvalidInput(
                "path-planning attempt log must not be empty".to_string(),
            ));
        }

        let total = attempts.len();
        let successful: Vec<&PathPlanningAttempt> = attempts.iter().filter(|a| a.success).collect();
        let success_rate = successful.len() as f64 / total as f64;

        let total_nanos: u128 = attempts.iter().map(|a| a.planning_time.as_nanos()).sum();
        let planning_time = Duration::from_nanos((total_nanos / total as u128) as u64);
        let mean_planning_secs = planning_time.as_secs_f64();
        let computational_efficiency = 1.0 / (1.0 + mean_planning_secs);

        let mut optimality_ratios = Vec::new();
        let mut smoothness_scores = Vec::new();
        for a in &successful {
            let actual_length = path_length(&a.planned_path);
            if let Some(optimal_length) = a.optimal_length {
                if optimal_length > 0.0 {
                    optimality_ratios.push(actual_length / optimal_length);
                }
            }
            if let Some(turn) = average_turning_angle(&a.planned_path) {
                smoothness_scores.push(1.0 / (1.0 + turn));
            }
        }

        let path_optimality = if optimality_ratios.is_empty() {
            1.0
        } else {
            optimality_ratios.iter().sum::<f64>() / optimality_ratios.len() as f64
        };
        let smoothness = if smoothness_scores.is_empty() {
            1.0
        } else {
            smoothness_scores.iter().sum::<f64>() / smoothness_scores.len() as f64
        };

        let result = PathPlanningMetrics {
            success_rate,
            planning_time,
            path_optimality,
            smoothness,
            computational_efficiency,
        };
        self.path_planning = result.clone();
        Ok(result)
    }

    /// Evaluate obstacle-avoidance performance from a real executed path
    /// against a real obstacle set, comparing against the originally planned
    /// path (before any avoidance maneuvers) to measure deviation.
    ///
    /// Pass an empty `obstacles` slice when there is nothing to avoid --
    /// clearance metrics are then vacuously "fully clear" rather than
    /// fabricated. Pass an empty `planned_path` when the original plan isn't
    /// available -- `path_deviation` is then honestly `0.0` (nothing to
    /// compare against).
    pub fn evaluate_obstacle_avoidance(
        &mut self,
        actual_path: &[TrajectoryPoint],
        planned_path: &[TrajectoryPoint],
        obstacles: &[Obstacle],
        reaction_times: &[Duration],
    ) -> Result<ObstacleAvoidanceMetrics> {
        if actual_path.is_empty() {
            return Err(MetricsError::InvalidInput(
                "actual_path must not be empty".to_string(),
            ));
        }

        let (collision_avoidance_rate, min_clearance, avg_clearance) = if obstacles.is_empty() {
            (1.0, f64::INFINITY, f64::INFINITY)
        } else {
            let clearances: Vec<f64> = actual_path
                .iter()
                .map(|p| nearest_clearance(&p.position, obstacles))
                .collect();
            let safe_count = clearances.iter().filter(|&&c| c >= 0.0).count();
            let min_c = clearances.iter().copied().fold(f64::INFINITY, f64::min);
            let avg_c = clearances.iter().sum::<f64>() / clearances.len() as f64;
            (safe_count as f64 / clearances.len() as f64, min_c, avg_c)
        };

        let reaction_time = if reaction_times.is_empty() {
            Duration::from_millis(0)
        } else {
            let total_nanos: u128 = reaction_times.iter().map(|d| d.as_nanos()).sum();
            Duration::from_nanos((total_nanos / reaction_times.len() as u128) as u64)
        };

        let path_deviation = if planned_path.is_empty() {
            0.0
        } else {
            let n = actual_path.len().min(planned_path.len());
            let total: f64 = (0..n)
                .map(|i| actual_path[i].distance_to(&planned_path[i]))
                .sum();
            total / n as f64
        };

        let result = ObstacleAvoidanceMetrics {
            collision_avoidance_rate,
            min_clearance,
            avg_clearance,
            reaction_time,
            path_deviation,
        };
        self.obstacle_avoidance = result.clone();
        Ok(result)
    }

    /// Evaluate dynamic-adaptation capability from a real log of
    /// environmental-change events.
    ///
    /// `disturbance_robustness` counts an event as "handled" when the system
    /// either didn't need to replan at all, or replanned and then succeeded --
    /// i.e. the fraction of disturbances the system shrugged off or
    /// successfully recovered from.
    pub fn evaluate_dynamic_adaptation(
        &mut self,
        events: &[AdaptationEvent],
    ) -> Result<DynamicAdaptationMetrics> {
        if events.is_empty() {
            return Err(MetricsError::InvalidInput(
                "adaptation event log must not be empty".to_string(),
            ));
        }

        let total = events.len();
        let replanned: Vec<&AdaptationEvent> = events.iter().filter(|e| e.replanned).collect();
        let replanning_frequency = replanned.len() as f64 / total as f64;

        let adaptation_time = if replanned.is_empty() {
            Duration::from_millis(0)
        } else {
            let total_nanos: u128 = replanned.iter().map(|e| e.response_time.as_nanos()).sum();
            Duration::from_nanos((total_nanos / replanned.len() as u128) as u64)
        };

        let adaptation_success_rate = if replanned.is_empty() {
            1.0
        } else {
            replanned.iter().filter(|e| e.adaptation_successful).count() as f64
                / replanned.len() as f64
        };

        let handled = events
            .iter()
            .filter(|e| !e.replanned || e.adaptation_successful)
            .count();
        let disturbance_robustness = handled as f64 / total as f64;

        let result = DynamicAdaptationMetrics {
            adaptation_time,
            replanning_frequency,
            adaptation_success_rate,
            disturbance_robustness,
        };
        self.dynamic_adaptation = result.clone();
        Ok(result)
    }
}

/// Total Euclidean path length through consecutive trajectory points.
fn path_length(path: &[TrajectoryPoint]) -> f64 {
    if path.len() < 2 {
        return 0.0;
    }
    (1..path.len())
        .map(|i| path[i].distance_to(&path[i - 1]))
        .sum()
}

/// Average turning angle (radians) between consecutive path segments.
/// Returns `None` when there are fewer than 3 points (no turn to measure).
fn average_turning_angle(path: &[TrajectoryPoint]) -> Option<f64> {
    if path.len() < 3 {
        return None;
    }
    let mut angles = Vec::new();
    for i in 1..path.len() - 1 {
        let v1 = [
            path[i].position[0] - path[i - 1].position[0],
            path[i].position[1] - path[i - 1].position[1],
            path[i].position[2] - path[i - 1].position[2],
        ];
        let v2 = [
            path[i + 1].position[0] - path[i].position[0],
            path[i + 1].position[1] - path[i].position[1],
            path[i + 1].position[2] - path[i].position[2],
        ];
        let n1 = (v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]).sqrt();
        let n2 = (v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2]).sqrt();
        if n1 > 1e-9 && n2 > 1e-9 {
            let dot = (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]) / (n1 * n2);
            angles.push(dot.clamp(-1.0, 1.0).acos());
        }
    }
    if angles.is_empty() {
        None
    } else {
        Some(angles.iter().sum::<f64>() / angles.len() as f64)
    }
}

// Default implementations
impl Default for PathPlanningMetrics {
    fn default() -> Self {
        Self {
            success_rate: 1.0,
            planning_time: Duration::from_millis(0),
            path_optimality: 1.0,
            smoothness: 1.0,
            computational_efficiency: 1.0,
        }
    }
}

impl Default for ObstacleAvoidanceMetrics {
    fn default() -> Self {
        Self {
            collision_avoidance_rate: 1.0,
            min_clearance: 0.0,
            avg_clearance: 0.0,
            reaction_time: Duration::from_millis(0),
            path_deviation: 0.0,
        }
    }
}

impl Default for GoalReachingMetrics {
    fn default() -> Self {
        Self {
            success_rate: 1.0,
            position_accuracy: 0.0,
            orientation_accuracy: 0.0,
            completion_time: Duration::from_secs(0),
            energy_efficiency: 1.0,
        }
    }
}

impl Default for DynamicAdaptationMetrics {
    fn default() -> Self {
        Self {
            adaptation_time: Duration::from_millis(0),
            replanning_frequency: 0.0,
            adaptation_success_rate: 1.0,
            disturbance_robustness: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pose(x: f64, y: f64) -> Pose {
        Pose::new([x, y, 0.0], [1.0, 0.0, 0.0, 0.0], 1.0)
    }

    fn tp(x: f64, y: f64, z: f64) -> TrajectoryPoint {
        TrajectoryPoint::new(
            [x, y, z],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            Duration::from_millis(0),
        )
    }

    #[test]
    fn goal_reaching_computes_real_rates() {
        let attempts = vec![
            GoalReachingAttempt {
                final_pose: pose(0.0, 0.0),
                goal_pose: pose(0.0, 0.0),
                completion_time: Duration::from_secs(2),
                energy_used: 10.0,
            },
            GoalReachingAttempt {
                final_pose: pose(5.0, 0.0),
                goal_pose: pose(0.0, 0.0),
                completion_time: Duration::from_secs(4),
                energy_used: 20.0,
            },
        ];
        let mut metrics = NavigationMetrics::new();
        let result = metrics
            .evaluate_goal_reaching(&attempts, 0.5, 0.1, 10.0)
            .expect("evaluation should succeed");

        assert!(
            (result.success_rate - 0.5).abs() < 1e-9,
            "got {}",
            result.success_rate
        );
        assert!((result.position_accuracy - 2.5).abs() < 1e-9);
        assert_eq!(result.completion_time, Duration::from_secs(3));
        assert!(
            (result.energy_efficiency - (10.0 / 15.0)).abs() < 1e-9,
            "got {}",
            result.energy_efficiency
        );
    }

    #[test]
    fn path_planning_computes_real_optimality_and_smoothness() {
        let straight = vec![tp(0.0, 0.0, 0.0), tp(1.0, 0.0, 0.0), tp(2.0, 0.0, 0.0)];
        let l_shaped = vec![tp(0.0, 0.0, 0.0), tp(1.0, 0.0, 0.0), tp(1.0, 1.0, 0.0)];
        let attempts = vec![
            PathPlanningAttempt {
                success: true,
                planned_path: straight,
                optimal_length: Some(2.0),
                planning_time: Duration::from_millis(100),
            },
            PathPlanningAttempt {
                success: true,
                planned_path: l_shaped,
                optimal_length: Some(1.5),
                planning_time: Duration::from_millis(200),
            },
            PathPlanningAttempt {
                success: false,
                planned_path: vec![],
                optimal_length: None,
                planning_time: Duration::from_millis(50),
            },
        ];
        let mut metrics = NavigationMetrics::new();
        let result = metrics
            .evaluate_path_planning(&attempts)
            .expect("evaluation should succeed");

        assert!((result.success_rate - 2.0 / 3.0).abs() < 1e-9);
        assert_eq!(result.planning_time, Duration::from_nanos(350_000_000 / 3));
        // Straight path ratio = 2.0/2.0 = 1.0; L-shaped ratio = 2.0/1.5 = 1.3333
        assert!(
            (result.path_optimality - (1.0 + 2.0 / 1.5) / 2.0).abs() < 1e-6,
            "got {}",
            result.path_optimality
        );
        // Straight path has zero turning angle (smoothness=1.0); L-shaped has
        // a 90-degree turn (smoothness = 1/(1+pi/2)).
        let expected_smoothness = (1.0 + 1.0 / (1.0 + std::f64::consts::FRAC_PI_2)) / 2.0;
        assert!(
            (result.smoothness - expected_smoothness).abs() < 1e-6,
            "got {}",
            result.smoothness
        );
        assert!(
            result.smoothness < 1.0,
            "the L-shaped detour must be penalized, not hardcoded to 1.0"
        );
    }

    #[test]
    fn obstacle_avoidance_detects_real_penetration_and_deviation() {
        let actual_path = vec![tp(0.0, 0.0, 0.0), tp(1.0, 0.0, 0.0), tp(2.0, 0.0, 0.0)];
        let planned_path = vec![tp(0.0, 0.0, 0.0), tp(1.0, 1.0, 0.0), tp(2.0, 0.0, 0.0)];
        let obstacles = vec![Obstacle::new([1.0, 0.0, 0.0], 0.5)];
        let mut metrics = NavigationMetrics::new();
        let result = metrics
            .evaluate_obstacle_avoidance(
                &actual_path,
                &planned_path,
                &obstacles,
                &[Duration::from_millis(100), Duration::from_millis(200)],
            )
            .expect("evaluation should succeed");

        assert!((result.collision_avoidance_rate - 2.0 / 3.0).abs() < 1e-9);
        assert!((result.min_clearance - (-0.5)).abs() < 1e-9);
        assert!(
            (result.avg_clearance - 0.5 / 3.0).abs() < 1e-9,
            "got {}",
            result.avg_clearance
        );
        assert_eq!(result.reaction_time, Duration::from_millis(150));
        // Only the middle waypoint deviates (distance 1.0), others match exactly.
        assert!(
            (result.path_deviation - 1.0 / 3.0).abs() < 1e-9,
            "got {}",
            result.path_deviation
        );
    }

    #[test]
    fn obstacle_avoidance_vacuous_when_no_obstacles() {
        let path = vec![tp(0.0, 0.0, 0.0), tp(1.0, 0.0, 0.0)];
        let mut metrics = NavigationMetrics::new();
        let result = metrics
            .evaluate_obstacle_avoidance(&path, &[], &[], &[])
            .expect("evaluation should succeed");
        assert_eq!(result.collision_avoidance_rate, 1.0);
        assert_eq!(result.path_deviation, 0.0);
    }

    #[test]
    fn dynamic_adaptation_computes_real_robustness() {
        let events = vec![
            AdaptationEvent {
                response_time: Duration::from_millis(50),
                replanned: false,
                adaptation_successful: true,
            },
            AdaptationEvent {
                response_time: Duration::from_millis(200),
                replanned: true,
                adaptation_successful: true,
            },
            AdaptationEvent {
                response_time: Duration::from_millis(300),
                replanned: true,
                adaptation_successful: false,
            },
            AdaptationEvent {
                response_time: Duration::from_millis(10),
                replanned: false,
                adaptation_successful: true,
            },
        ];
        let mut metrics = NavigationMetrics::new();
        let result = metrics
            .evaluate_dynamic_adaptation(&events)
            .expect("evaluation should succeed");

        assert!((result.replanning_frequency - 0.5).abs() < 1e-9);
        assert_eq!(result.adaptation_time, Duration::from_millis(250));
        assert!((result.adaptation_success_rate - 0.5).abs() < 1e-9);
        // 3 of 4 events were either not-replanned or successfully replanned.
        assert!(
            (result.disturbance_robustness - 0.75).abs() < 1e-9,
            "got {}",
            result.disturbance_robustness
        );
    }
}
