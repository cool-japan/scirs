//! Manipulation and grasping metrics
//!
//! This module provides metrics for evaluating robotic manipulation tasks,
//! including grasping performance, manipulation accuracy, and force control.

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use super::core::{utils, BoundingBox, ErrorStatistics, Force};
use crate::error::{MetricsError, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Outcome of a single grasp attempt, used to compute real [`GraspingMetrics`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraspAttempt {
    /// Whether the grasp succeeded (object held stably through the task).
    pub success: bool,
    /// Measured/estimated force-closure quality in `[0, 1]` for this attempt
    /// (meaningful regardless of final success).
    pub force_closure_quality: f64,
    /// Post-grasp stability score in `[0, 1]`, e.g. derived from force/torque
    /// sensor variance while holding the object. Only meaningful for
    /// successful grasps.
    pub stability_score: f64,
    /// Whether the approach trajectory required a collision-avoidance
    /// correction (a clean approach has no correction).
    pub approach_collision: bool,
    /// Whether the grasped object was damaged.
    pub object_damaged: bool,
    /// Time spent planning this grasp.
    pub planning_time: Duration,
}

/// Outcome of a single manipulation task execution, used to compute real
/// [`TaskCompletionMetrics`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskOutcome {
    /// Whether the task completed successfully.
    pub success: bool,
    /// Wall-clock time taken to complete (or abandon) the task.
    pub completion_time: Duration,
    /// Measured quality of the final result in `[0, 1]` (e.g. from
    /// post-task inspection).
    pub result_quality: f64,
    /// Whether an error occurred during execution.
    pub error_occurred: bool,
    /// Whether the system recovered from the error (only meaningful when
    /// `error_occurred` is `true`).
    pub recovered_from_error: bool,
}

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

    /// Evaluate grasping performance from a real log of grasp attempts.
    ///
    /// `stability_score` and `force_closure_quality` are averaged over the
    /// *successful* attempts (a failed grasp's "stability" isn't a meaningful
    /// measurement); if no attempt succeeded, both are honestly `NaN` rather
    /// than a fabricated perfect or zero score.
    pub fn evaluate_grasping(&mut self, attempts: &[GraspAttempt]) -> Result<GraspingMetrics> {
        if attempts.is_empty() {
            return Err(MetricsError::InvalidInput(
                "grasp attempt log must not be empty".to_string(),
            ));
        }

        let total = attempts.len();
        let successes: Vec<&GraspAttempt> = attempts.iter().filter(|a| a.success).collect();
        let success_rate = successes.len() as f64 / total as f64;

        let stability_score = if successes.is_empty() {
            f64::NAN
        } else {
            successes.iter().map(|a| a.stability_score).sum::<f64>() / successes.len() as f64
        };

        let force_closure_quality = attempts
            .iter()
            .map(|a| a.force_closure_quality)
            .sum::<f64>()
            / total as f64;

        let clean_approaches = attempts.iter().filter(|a| !a.approach_collision).count();
        let approach_quality = clean_approaches as f64 / total as f64;

        let total_planning_nanos: u128 = attempts.iter().map(|a| a.planning_time.as_nanos()).sum();
        let planning_time = Duration::from_nanos((total_planning_nanos / total as u128) as u64);

        let damaged = attempts.iter().filter(|a| a.object_damaged).count();
        let damage_rate = damaged as f64 / total as f64;

        let result = GraspingMetrics {
            success_rate,
            stability_score,
            force_closure_quality,
            approach_quality,
            planning_time,
            damage_rate,
        };
        self.grasping_metrics = result.clone();
        Ok(result)
    }

    /// Evaluate manipulation accuracy from real position/orientation error
    /// samples and repeated end-effector positioning trials.
    ///
    /// - `position_errors_mm` / `orientation_errors_deg`: per-sample error
    ///   magnitudes against the commanded pose.
    /// - `trajectory_tracking_errors`: per-sample deviation from the
    ///   commanded trajectory (any consistent distance unit); folded into a
    ///   `[0, 1]` score via `1 / (1 + mean_error)`, the same "smaller error ->
    ///   score closer to 1" idiom used for velocity smoothness in
    ///   [`super::motion_planning`].
    /// - `repeated_final_positions`: the end-effector's final position
    ///   across repeated executions of the *same* commanded motion; the
    ///   spread of these positions around their centroid measures
    ///   repeatability. Fewer than 2 repeats makes repeatability
    ///   unmeasurable, honestly returned as `1.0` (vacuous: nothing to
    ///   compare) rather than a fabricated score derived from one sample.
    pub fn evaluate_manipulation_accuracy(
        &mut self,
        position_errors_mm: &[f64],
        orientation_errors_deg: &[f64],
        trajectory_tracking_errors: &[f64],
        repeated_final_positions: &[[f64; 3]],
    ) -> Result<ManipulationAccuracyMetrics> {
        if position_errors_mm.is_empty() || orientation_errors_deg.is_empty() {
            return Err(MetricsError::InvalidInput(
                "position_errors_mm and orientation_errors_deg must not be empty".to_string(),
            ));
        }

        let position_accuracy = utils::calculate_statistics(position_errors_mm);
        let orientation_accuracy = utils::calculate_statistics(orientation_errors_deg);

        let trajectory_accuracy = if trajectory_tracking_errors.is_empty() {
            1.0
        } else {
            let mean_error = trajectory_tracking_errors.iter().sum::<f64>()
                / trajectory_tracking_errors.len() as f64;
            1.0 / (1.0 + mean_error.abs())
        };

        let repeatability = if repeated_final_positions.len() < 2 {
            1.0
        } else {
            let n = repeated_final_positions.len() as f64;
            let centroid = repeated_final_positions.iter().fold([0.0; 3], |acc, p| {
                [acc[0] + p[0] / n, acc[1] + p[1] / n, acc[2] + p[2] / n]
            });
            let deviations: Vec<f64> = repeated_final_positions
                .iter()
                .map(|p| {
                    let dx = p[0] - centroid[0];
                    let dy = p[1] - centroid[1];
                    let dz = p[2] - centroid[2];
                    (dx * dx + dy * dy + dz * dz).sqrt()
                })
                .collect();
            let rmse_dev =
                (deviations.iter().map(|d| d * d).sum::<f64>() / deviations.len() as f64).sqrt();
            1.0 / (1.0 + rmse_dev)
        };

        let result = ManipulationAccuracyMetrics {
            position_accuracy,
            orientation_accuracy,
            trajectory_accuracy,
            repeatability,
        };
        self.manipulation_accuracy = result.clone();
        Ok(result)
    }

    /// Evaluate overall task completion from a real log of task outcomes.
    ///
    /// `optimal_time` is the reference (expert/planned) completion time used
    /// to normalize `efficiency_score`; pass the mean observed time itself if
    /// no external reference is available (yields `efficiency_score <= 1.0`
    /// relative to the observed average).
    pub fn evaluate_task_completion(
        &mut self,
        outcomes: &[TaskOutcome],
        optimal_time: Duration,
    ) -> Result<TaskCompletionMetrics> {
        if outcomes.is_empty() {
            return Err(MetricsError::InvalidInput(
                "task outcome log must not be empty".to_string(),
            ));
        }

        let total = outcomes.len();
        let success_rate = outcomes.iter().filter(|o| o.success).count() as f64 / total as f64;

        let total_nanos: u128 = outcomes.iter().map(|o| o.completion_time.as_nanos()).sum();
        let completion_time = Duration::from_nanos((total_nanos / total as u128) as u64);

        let efficiency_score = if completion_time.as_secs_f64() > 0.0 {
            (optimal_time.as_secs_f64() / completion_time.as_secs_f64()).min(1.0)
        } else {
            1.0
        };

        let errors: Vec<&TaskOutcome> = outcomes.iter().filter(|o| o.error_occurred).collect();
        let error_recovery_rate = if errors.is_empty() {
            1.0
        } else {
            errors.iter().filter(|o| o.recovered_from_error).count() as f64 / errors.len() as f64
        };

        let result_quality = outcomes.iter().map(|o| o.result_quality).sum::<f64>() / total as f64;

        let result = TaskCompletionMetrics {
            success_rate,
            completion_time,
            efficiency_score,
            error_recovery_rate,
            result_quality,
        };
        self.task_completion = result.clone();
        Ok(result)
    }

    /// Evaluate force/contact control from real force-error samples and
    /// contact-detection outcomes.
    ///
    /// - `force_errors`: per-sample deviation (any consistent force unit,
    ///   e.g. Newtons) between commanded and measured contact force.
    /// - `overshoot_tolerance`: samples whose absolute error exceeds this
    ///   tolerance count as an overshoot event.
    /// - `contact_detected` / `contact_ground_truth`: per-sample booleans for
    ///   whether contact was reported vs. actually occurring; used for
    ///   `contact_detection_accuracy`. Pass equal-length empty slices if no
    ///   contact-detection log is available -- accuracy is then vacuously
    ///   `1.0` (nothing to have gotten wrong) rather than fabricated.
    pub fn evaluate_force_control(
        &mut self,
        force_errors: &[f64],
        overshoot_tolerance: f64,
        contact_detected: &[bool],
        contact_ground_truth: &[bool],
    ) -> Result<ForceContactMetrics> {
        if force_errors.is_empty() {
            return Err(MetricsError::InvalidInput(
                "force_errors must not be empty".to_string(),
            ));
        }
        if contact_detected.len() != contact_ground_truth.len() {
            return Err(MetricsError::InvalidInput(format!(
                "contact_detected ({}) and contact_ground_truth ({}) must have the same length",
                contact_detected.len(),
                contact_ground_truth.len()
            )));
        }

        let force_accuracy = utils::calculate_statistics(force_errors);

        let contact_stability = 1.0 / (1.0 + force_accuracy.std_error);
        let compliance_quality = 1.0 / (1.0 + force_accuracy.mean_error.abs());

        let overshoot_count = force_errors
            .iter()
            .filter(|e| e.abs() > overshoot_tolerance)
            .count();
        let force_overshoot = overshoot_count as f64 / force_errors.len() as f64;

        let contact_detection_accuracy = if contact_detected.is_empty() {
            1.0
        } else {
            let matches = contact_detected
                .iter()
                .zip(contact_ground_truth.iter())
                .filter(|(d, g)| d == g)
                .count();
            matches as f64 / contact_detected.len() as f64
        };

        let result = ForceContactMetrics {
            force_accuracy,
            contact_stability,
            force_overshoot,
            contact_detection_accuracy,
            compliance_quality,
        };
        self.force_metrics = result.clone();
        Ok(result)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn attempt(
        success: bool,
        force_closure_quality: f64,
        stability_score: f64,
        approach_collision: bool,
        object_damaged: bool,
        planning_ms: u64,
    ) -> GraspAttempt {
        GraspAttempt {
            success,
            force_closure_quality,
            stability_score,
            approach_collision,
            object_damaged,
            planning_time: Duration::from_millis(planning_ms),
        }
    }

    #[test]
    fn grasping_computes_real_rates_not_hardcoded() {
        let attempts = vec![
            attempt(true, 0.9, 0.95, false, false, 100),
            attempt(false, 0.5, 0.0, true, false, 200),
            attempt(true, 0.8, 0.85, false, true, 150),
            attempt(true, 0.7, 0.75, false, false, 50),
        ];
        let mut metrics = ManipulationMetrics::new();
        let result = metrics
            .evaluate_grasping(&attempts)
            .expect("evaluation should succeed");

        assert!(
            (result.success_rate - 0.75).abs() < 1e-9,
            "got {}",
            result.success_rate
        );
        // Stability averaged over the 3 *successful* attempts only: (0.95+0.85+0.75)/3
        assert!(
            (result.stability_score - 0.85).abs() < 1e-9,
            "got {}",
            result.stability_score
        );
        // Force closure quality averaged over all 4 attempts.
        assert!(
            (result.force_closure_quality - 0.725).abs() < 1e-9,
            "got {}",
            result.force_closure_quality
        );
        assert!((result.approach_quality - 0.75).abs() < 1e-9);
        assert!((result.damage_rate - 0.25).abs() < 1e-9);
        assert_eq!(result.planning_time, Duration::from_millis(125));
    }

    #[test]
    fn grasping_stability_is_nan_when_nothing_succeeded() {
        let attempts = vec![
            attempt(false, 0.5, 0.0, true, false, 10),
            attempt(false, 0.4, 0.0, true, true, 10),
        ];
        let mut metrics = ManipulationMetrics::new();
        let result = metrics
            .evaluate_grasping(&attempts)
            .expect("evaluation should succeed");
        assert_eq!(result.success_rate, 0.0);
        assert!(
            result.stability_score.is_nan(),
            "stability is not measurable when every grasp failed"
        );
    }

    #[test]
    fn manipulation_accuracy_computes_real_values() {
        let mut metrics = ManipulationMetrics::new();
        let result = metrics
            .evaluate_manipulation_accuracy(
                &[1.0, 2.0, 3.0],
                &[0.5, 1.5],
                &[1.0, 1.0],
                &[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            )
            .expect("evaluation should succeed");

        assert!((result.position_accuracy.mean_error - 2.0).abs() < 1e-9);
        assert!((result.orientation_accuracy.mean_error - 1.0).abs() < 1e-9);
        // trajectory_accuracy = 1 / (1 + mean(|1.0|, |1.0|)) = 1 / 2 = 0.5
        assert!(
            (result.trajectory_accuracy - 0.5).abs() < 1e-9,
            "got {}",
            result.trajectory_accuracy
        );
        // Two repeats 2.0 apart -> centroid at 1.0, deviation 1.0 each -> repeatability = 1/(1+1) = 0.5
        assert!(
            (result.repeatability - 0.5).abs() < 1e-9,
            "got {}",
            result.repeatability
        );
    }

    #[test]
    fn manipulation_accuracy_repeatability_vacuous_with_single_sample() {
        let mut metrics = ManipulationMetrics::new();
        let result = metrics
            .evaluate_manipulation_accuracy(&[1.0], &[1.0], &[], &[[0.0, 0.0, 0.0]])
            .expect("evaluation should succeed");
        assert_eq!(result.repeatability, 1.0);
        assert_eq!(result.trajectory_accuracy, 1.0);
    }

    #[test]
    fn task_completion_computes_real_metrics() {
        let outcomes = vec![
            TaskOutcome {
                success: true,
                completion_time: Duration::from_secs(1),
                result_quality: 0.9,
                error_occurred: false,
                recovered_from_error: false,
            },
            TaskOutcome {
                success: false,
                completion_time: Duration::from_secs(2),
                result_quality: 0.3,
                error_occurred: true,
                recovered_from_error: false,
            },
            TaskOutcome {
                success: true,
                completion_time: Duration::from_millis(1500),
                result_quality: 0.8,
                error_occurred: true,
                recovered_from_error: true,
            },
        ];
        let mut metrics = ManipulationMetrics::new();
        let result = metrics
            .evaluate_task_completion(&outcomes, Duration::from_secs(1))
            .expect("evaluation should succeed");

        assert!((result.success_rate - 2.0 / 3.0).abs() < 1e-9);
        assert_eq!(result.completion_time, Duration::from_millis(1500));
        // efficiency = optimal(1s) / observed(1.5s) = 0.6667
        assert!(
            (result.efficiency_score - (1.0 / 1.5)).abs() < 1e-9,
            "got {}",
            result.efficiency_score
        );
        // 2 errors occurred, only 1 recovered -> 0.5
        assert!((result.error_recovery_rate - 0.5).abs() < 1e-9);
        assert!((result.result_quality - (2.0 / 3.0)).abs() < 1e-9);
    }

    #[test]
    fn force_control_computes_real_values_not_hardcoded() {
        let mut metrics = ManipulationMetrics::new();
        let result = metrics
            .evaluate_force_control(
                &[1.0, -1.0, 2.0, -2.0],
                1.5,
                &[true, false, true],
                &[true, true, true],
            )
            .expect("evaluation should succeed");

        assert!(result.force_accuracy.mean_error.abs() < 1e-9);
        // std_error = sqrt(((1)^2+(1)^2+(2)^2+(2)^2)/4) = sqrt(2.5)
        let expected_std = 2.5_f64.sqrt();
        assert!((result.force_accuracy.std_error - expected_std).abs() < 1e-9);
        assert!(
            (result.contact_stability - 1.0 / (1.0 + expected_std)).abs() < 1e-9,
            "got {}",
            result.contact_stability
        );
        // Only |2.0| and |-2.0| exceed the 1.5 tolerance -> 2/4 = 0.5
        assert!(
            (result.force_overshoot - 0.5).abs() < 1e-9,
            "got {}",
            result.force_overshoot
        );
        // 2 of 3 contact-detection samples match ground truth
        assert!(
            (result.contact_detection_accuracy - 2.0 / 3.0).abs() < 1e-9,
            "got {}",
            result.contact_detection_accuracy
        );
    }

    #[test]
    fn force_control_rejects_mismatched_contact_log_lengths() {
        let mut metrics = ManipulationMetrics::new();
        assert!(metrics
            .evaluate_force_control(&[1.0], 1.0, &[true], &[true, false])
            .is_err());
    }
}
