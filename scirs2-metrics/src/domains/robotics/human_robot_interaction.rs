//! Human-Robot Interaction (HRI) metrics
//!
//! This module provides metrics for evaluating human-robot interaction quality,
//! safety, communication effectiveness, and collaboration efficiency.

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use crate::error::{MetricsError, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;

fn mean_duration(durations: &[Duration]) -> Duration {
    if durations.is_empty() {
        return Duration::from_millis(0);
    }
    let total_nanos: u128 = durations.iter().map(|d| d.as_nanos()).sum();
    Duration::from_nanos((total_nanos / durations.len() as u128) as u64)
}

/// Human-Robot Interaction evaluation metrics
#[derive(Debug, Clone)]
pub struct HumanRobotInteractionMetrics {
    /// Safety metrics for HRI
    pub safety_metrics: HriSafetyMetrics,
    /// Communication effectiveness
    pub communication_metrics: CommunicationMetrics,
    /// User satisfaction measures
    pub user_satisfaction: UserSatisfactionMetrics,
    /// Collaboration efficiency
    pub collaboration_efficiency: CollaborationEfficiencyMetrics,
}

/// HRI safety evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HriSafetyMetrics {
    /// Minimum distance maintained
    pub min_safe_distance: f64,
    /// Number of safety violations
    pub safety_violations: usize,
    /// Emergency stop response time
    pub emergency_response_time: Duration,
    /// Collision avoidance success rate
    pub collision_avoidance_rate: f64,
}

/// Communication effectiveness metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationMetrics {
    /// Command understanding accuracy
    pub command_accuracy: f64,
    /// Response time to commands
    pub response_time: Duration,
    /// Feedback quality score
    pub feedback_quality: f64,
    /// Multimodal communication success
    pub multimodal_success_rate: f64,
}

/// User satisfaction assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserSatisfactionMetrics {
    /// Overall satisfaction score
    pub overall_satisfaction: f64,
    /// Ease of use rating
    pub ease_of_use: f64,
    /// Trust level in robot
    pub trust_level: f64,
    /// Task completion satisfaction
    pub task_satisfaction: f64,
}

/// Collaboration efficiency evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationEfficiencyMetrics {
    /// Task completion time improvement
    pub time_improvement: f64,
    /// Workload distribution balance
    pub workload_balance: f64,
    /// Coordination effectiveness
    pub coordination_effectiveness: f64,
    /// Human cognitive load
    pub cognitive_load: f64,
}

impl HumanRobotInteractionMetrics {
    /// Create new HRI metrics
    pub fn new() -> Self {
        Self {
            safety_metrics: HriSafetyMetrics::default(),
            communication_metrics: CommunicationMetrics::default(),
            user_satisfaction: UserSatisfactionMetrics::default(),
            collaboration_efficiency: CollaborationEfficiencyMetrics::default(),
        }
    }

    /// Evaluate HRI safety from a real time-series of human-robot distances.
    ///
    /// `human_robot_distances` are per-sample measured distances (any
    /// consistent unit matching `safe_distance_threshold`) over an
    /// interaction session. `emergency_stop_response_times` are the
    /// latencies of any emergency-stop events that were actually triggered
    /// (empty if none were -- the response time is then honestly `0`
    /// rather than a fabricated value).
    pub fn evaluate_safety(
        &mut self,
        human_robot_distances: &[f64],
        safe_distance_threshold: f64,
        emergency_stop_response_times: &[Duration],
    ) -> Result<HriSafetyMetrics> {
        if human_robot_distances.is_empty() {
            return Err(MetricsError::InvalidInput(
                "human_robot_distances must not be empty".to_string(),
            ));
        }

        let min_safe_distance = human_robot_distances
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let safety_violations = human_robot_distances
            .iter()
            .filter(|&&d| d < safe_distance_threshold)
            .count();
        let collision_avoidance_rate =
            1.0 - (safety_violations as f64 / human_robot_distances.len() as f64);
        let emergency_response_time = mean_duration(emergency_stop_response_times);

        let result = HriSafetyMetrics {
            min_safe_distance,
            safety_violations,
            emergency_response_time,
            collision_avoidance_rate,
        };
        self.safety_metrics = result.clone();
        Ok(result)
    }

    /// Evaluate communication effectiveness from real command/response logs.
    ///
    /// `feedback_ratings`/`multimodal_success` may be empty (no such
    /// sub-channel exercised in this session), in which case the
    /// corresponding sub-metric is honestly `1.0` (vacuous: nothing observed
    /// to have gone wrong) rather than fabricated.
    pub fn evaluate_communication(
        &mut self,
        commands_understood_correctly: &[bool],
        response_times: &[Duration],
        feedback_ratings: &[f64],
        multimodal_success: &[bool],
    ) -> Result<CommunicationMetrics> {
        if commands_understood_correctly.is_empty() {
            return Err(MetricsError::InvalidInput(
                "commands_understood_correctly must not be empty".to_string(),
            ));
        }

        let command_accuracy = commands_understood_correctly.iter().filter(|&&c| c).count() as f64
            / commands_understood_correctly.len() as f64;
        let response_time = mean_duration(response_times);
        let feedback_quality = if feedback_ratings.is_empty() {
            1.0
        } else {
            feedback_ratings.iter().sum::<f64>() / feedback_ratings.len() as f64
        };
        let multimodal_success_rate = if multimodal_success.is_empty() {
            1.0
        } else {
            multimodal_success.iter().filter(|&&s| s).count() as f64
                / multimodal_success.len() as f64
        };

        let result = CommunicationMetrics {
            command_accuracy,
            response_time,
            feedback_quality,
            multimodal_success_rate,
        };
        self.communication_metrics = result.clone();
        Ok(result)
    }

    /// Evaluate user satisfaction from real per-respondent survey ratings
    /// (each slice in `[0, 1]`, one entry per respondent/session; all four
    /// must have the same length).
    pub fn evaluate_user_satisfaction(
        &mut self,
        overall_ratings: &[f64],
        ease_of_use_ratings: &[f64],
        trust_ratings: &[f64],
        task_satisfaction_ratings: &[f64],
    ) -> Result<UserSatisfactionMetrics> {
        let n = overall_ratings.len();
        if n == 0
            || n != ease_of_use_ratings.len()
            || n != trust_ratings.len()
            || n != task_satisfaction_ratings.len()
        {
            return Err(MetricsError::InvalidInput(
                "all rating slices must be non-empty and the same length (one row per respondent)"
                    .to_string(),
            ));
        }

        let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;

        let result = UserSatisfactionMetrics {
            overall_satisfaction: mean(overall_ratings),
            ease_of_use: mean(ease_of_use_ratings),
            trust_level: mean(trust_ratings),
            task_satisfaction: mean(task_satisfaction_ratings),
        };
        self.user_satisfaction = result.clone();
        Ok(result)
    }

    /// Evaluate collaboration efficiency from real solo-vs-collaborative
    /// timing, workload split, coordination outcomes, and cognitive-load
    /// ratings.
    ///
    /// `human_workload_fraction` is the fraction of total task effort
    /// attributed to the human partner in `[0, 1]`; `workload_balance` scores
    /// `1.0` at a perfect 50/50 split and falls off linearly toward `0.0` at
    /// an all-or-nothing split.
    pub fn evaluate_collaboration_efficiency(
        &mut self,
        solo_task_time: Duration,
        collaborative_task_time: Duration,
        human_workload_fraction: f64,
        coordination_events_successful: &[bool],
        cognitive_load_ratings: &[f64],
    ) -> Result<CollaborationEfficiencyMetrics> {
        if coordination_events_successful.is_empty() {
            return Err(MetricsError::InvalidInput(
                "coordination_events_successful must not be empty".to_string(),
            ));
        }
        if cognitive_load_ratings.is_empty() {
            return Err(MetricsError::InvalidInput(
                "cognitive_load_ratings must not be empty".to_string(),
            ));
        }

        let time_improvement = if solo_task_time.as_secs_f64() > 0.0 {
            (solo_task_time.as_secs_f64() - collaborative_task_time.as_secs_f64())
                / solo_task_time.as_secs_f64()
        } else {
            0.0
        };

        let workload_balance = 1.0 - (human_workload_fraction - 0.5).abs() * 2.0;

        let coordination_effectiveness = coordination_events_successful
            .iter()
            .filter(|&&s| s)
            .count() as f64
            / coordination_events_successful.len() as f64;

        let cognitive_load =
            cognitive_load_ratings.iter().sum::<f64>() / cognitive_load_ratings.len() as f64;

        let result = CollaborationEfficiencyMetrics {
            time_improvement,
            workload_balance,
            coordination_effectiveness,
            cognitive_load,
        };
        self.collaboration_efficiency = result.clone();
        Ok(result)
    }
}

// Default implementations
impl Default for HriSafetyMetrics {
    fn default() -> Self {
        Self {
            min_safe_distance: 0.0,
            safety_violations: 0,
            emergency_response_time: Duration::from_millis(0),
            collision_avoidance_rate: 1.0,
        }
    }
}

impl Default for CommunicationMetrics {
    fn default() -> Self {
        Self {
            command_accuracy: 1.0,
            response_time: Duration::from_millis(0),
            feedback_quality: 1.0,
            multimodal_success_rate: 1.0,
        }
    }
}

impl Default for UserSatisfactionMetrics {
    fn default() -> Self {
        Self {
            overall_satisfaction: 1.0,
            ease_of_use: 1.0,
            trust_level: 1.0,
            task_satisfaction: 1.0,
        }
    }
}

impl Default for CollaborationEfficiencyMetrics {
    fn default() -> Self {
        Self {
            time_improvement: 0.0,
            workload_balance: 1.0,
            coordination_effectiveness: 1.0,
            cognitive_load: 0.5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safety_detects_real_violations_not_hardcoded() {
        let distances = vec![1.2, 0.8, 0.3, 1.5, 0.9];
        let mut metrics = HumanRobotInteractionMetrics::new();
        let result = metrics
            .evaluate_safety(&distances, 0.5, &[Duration::from_millis(200)])
            .expect("evaluation should succeed");

        assert!((result.min_safe_distance - 0.3).abs() < 1e-9);
        assert_eq!(result.safety_violations, 1);
        assert!(
            (result.collision_avoidance_rate - 0.8).abs() < 1e-9,
            "got {}",
            result.collision_avoidance_rate
        );
        assert_eq!(result.emergency_response_time, Duration::from_millis(200));
    }

    #[test]
    fn communication_computes_real_accuracy() {
        let understood = vec![true, true, false, true];
        let mut metrics = HumanRobotInteractionMetrics::new();
        let result = metrics
            .evaluate_communication(
                &understood,
                &[Duration::from_millis(100), Duration::from_millis(300)],
                &[0.9, 0.7],
                &[true, false],
            )
            .expect("evaluation should succeed");

        assert!(
            (result.command_accuracy - 0.75).abs() < 1e-9,
            "got {}",
            result.command_accuracy
        );
        assert_eq!(result.response_time, Duration::from_millis(200));
        assert!((result.feedback_quality - 0.8).abs() < 1e-9);
        assert!((result.multimodal_success_rate - 0.5).abs() < 1e-9);
    }

    #[test]
    fn user_satisfaction_computes_real_means() {
        let mut metrics = HumanRobotInteractionMetrics::new();
        let result = metrics
            .evaluate_user_satisfaction(&[0.9, 0.7], &[0.8, 0.6], &[1.0, 0.5], &[0.9, 0.9])
            .expect("evaluation should succeed");
        assert!((result.overall_satisfaction - 0.8).abs() < 1e-9);
        assert!((result.trust_level - 0.75).abs() < 1e-9);
    }

    #[test]
    fn user_satisfaction_rejects_mismatched_lengths() {
        let mut metrics = HumanRobotInteractionMetrics::new();
        assert!(metrics
            .evaluate_user_satisfaction(&[0.9], &[0.8, 0.6], &[1.0], &[0.9])
            .is_err());
    }

    #[test]
    fn collaboration_efficiency_computes_real_values_not_hardcoded() {
        let mut metrics = HumanRobotInteractionMetrics::new();
        let result = metrics
            .evaluate_collaboration_efficiency(
                Duration::from_secs(10),
                Duration::from_secs(6),
                0.3,
                &[true, true, false],
                &[0.4, 0.6],
            )
            .expect("evaluation should succeed");

        // (10 - 6) / 10 = 0.4
        assert!(
            (result.time_improvement - 0.4).abs() < 1e-9,
            "got {}",
            result.time_improvement
        );
        // |0.3 - 0.5| * 2 = 0.4 -> balance = 0.6
        assert!(
            (result.workload_balance - 0.6).abs() < 1e-9,
            "got {}",
            result.workload_balance
        );
        assert!((result.coordination_effectiveness - 2.0 / 3.0).abs() < 1e-9);
        assert!((result.cognitive_load - 0.5).abs() < 1e-9);
    }
}
