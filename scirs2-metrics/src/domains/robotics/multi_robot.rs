//! Multi-robot coordination metrics
//!
//! This module provides metrics for evaluating multi-robot systems,
//! including formation control, task allocation, and collective behavior.

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use crate::error::{MetricsError, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// Multi-robot coordination metrics
#[derive(Debug, Clone)]
pub struct MultiRobotMetrics {
    /// Formation control performance
    pub formation_control: FormationControlMetrics,
    /// Task allocation efficiency
    pub task_allocation: TaskAllocationMetrics,
    /// Network performance
    pub network_performance: NetworkPerformanceMetrics,
    /// Collective behavior assessment
    pub collective_behavior: CollectiveBehaviorMetrics,
}

/// Formation control evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormationControlMetrics {
    /// Formation maintenance accuracy
    pub formation_accuracy: f64,
    /// Convergence time to formation
    pub convergence_time: Duration,
    /// Formation stability
    pub stability: f64,
    /// Leader-follower coordination
    pub coordination_quality: f64,
}

/// Task allocation performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskAllocationMetrics {
    /// Allocation optimality
    pub allocation_optimality: f64,
    /// Load balancing efficiency
    pub load_balancing: f64,
    /// Adaptation to failures
    pub failure_adaptation: f64,
    /// Communication overhead
    pub communication_overhead: f64,
}

/// Network performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkPerformanceMetrics {
    /// Communication latency
    pub latency: Duration,
    /// Message loss rate
    pub message_loss_rate: f64,
    /// Bandwidth utilization
    pub bandwidth_utilization: f64,
    /// Network reliability
    pub network_reliability: f64,
}

/// Collective behavior assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectiveBehaviorMetrics {
    /// Swarm cohesion
    pub swarm_cohesion: f64,
    /// Emergent behavior quality
    pub emergent_behavior: f64,
    /// Scalability factor
    pub scalability: f64,
    /// Fault tolerance
    pub fault_tolerance: f64,
}

impl MultiRobotMetrics {
    /// Create new multi-robot metrics
    pub fn new() -> Self {
        Self {
            formation_control: FormationControlMetrics::default(),
            task_allocation: TaskAllocationMetrics::default(),
            network_performance: NetworkPerformanceMetrics::default(),
            collective_behavior: CollectiveBehaviorMetrics::default(),
        }
    }

    /// Evaluate formation-control performance from a real time series of
    /// actual vs. target per-robot positions.
    ///
    /// `actual_positions_over_time[t][i]` and `target_positions_over_time[t][i]`
    /// give robot `i`'s actual/target position at snapshot `t` (`dt` apart);
    /// both must have the same outer length, and each inner `Vec` at a given
    /// `t` must have the same robot count on both sides.
    pub fn evaluate_formation_control(
        &mut self,
        actual_positions_over_time: &[Vec<[f64; 3]>],
        target_positions_over_time: &[Vec<[f64; 3]>],
        convergence_threshold: f64,
        dt: Duration,
    ) -> Result<FormationControlMetrics> {
        if actual_positions_over_time.len() != target_positions_over_time.len() {
            return Err(MetricsError::InvalidInput(
                "actual and target position time series must have the same length".to_string(),
            ));
        }
        if actual_positions_over_time.is_empty() {
            return Err(MetricsError::InvalidInput(
                "position time series must not be empty".to_string(),
            ));
        }

        let mut per_timestep_error = Vec::with_capacity(actual_positions_over_time.len());
        for (actual, target) in actual_positions_over_time
            .iter()
            .zip(target_positions_over_time.iter())
        {
            if actual.len() != target.len() || actual.is_empty() {
                return Err(MetricsError::InvalidInput(
                    "each timestep must have a matching, non-empty robot count".to_string(),
                ));
            }
            let step_error: f64 = actual
                .iter()
                .zip(target.iter())
                .map(|(a, t)| {
                    let dx = a[0] - t[0];
                    let dy = a[1] - t[1];
                    let dz = a[2] - t[2];
                    (dx * dx + dy * dy + dz * dz).sqrt()
                })
                .sum::<f64>()
                / actual.len() as f64;
            per_timestep_error.push(step_error);
        }

        let overall_mean_error = mean(&per_timestep_error);
        let formation_accuracy = 1.0 / (1.0 + overall_mean_error);

        let error_variance = if per_timestep_error.len() > 1 {
            per_timestep_error
                .iter()
                .map(|e| (e - overall_mean_error).powi(2))
                .sum::<f64>()
                / per_timestep_error.len() as f64
        } else {
            0.0
        };
        let stability = 1.0 / (1.0 + error_variance);

        let step_to_step_change = if per_timestep_error.len() > 1 {
            let total: f64 = (1..per_timestep_error.len())
                .map(|i| (per_timestep_error[i] - per_timestep_error[i - 1]).abs())
                .sum();
            total / (per_timestep_error.len() - 1) as f64
        } else {
            0.0
        };
        let coordination_quality = 1.0 / (1.0 + step_to_step_change);

        let converged_at = per_timestep_error
            .iter()
            .position(|&e| e <= convergence_threshold);
        let convergence_time = match converged_at {
            Some(idx) => dt.mul_f64(idx as f64),
            // Never converged within the observed window: report the full
            // window length as an honest lower bound, not a fabricated time.
            None => dt.mul_f64((per_timestep_error.len() - 1) as f64),
        };

        let result = FormationControlMetrics {
            formation_accuracy,
            convergence_time,
            stability,
            coordination_quality,
        };
        self.formation_control = result.clone();
        Ok(result)
    }

    /// Evaluate task-allocation performance from real assignment costs and
    /// workload counts.
    ///
    /// `assigned_costs`/`optimal_costs` are per-task costs under the actual
    /// vs. a reference (optimal or best-known) assignment.
    /// `robot_task_counts` is the number of tasks given to each robot (for
    /// load-balancing); `communication_messages` is the total number of
    /// coordination messages exchanged while allocating `total_tasks` tasks.
    pub fn evaluate_task_allocation(
        &mut self,
        assigned_costs: &[f64],
        optimal_costs: &[f64],
        robot_task_counts: &[usize],
        reassignments_due_to_failure: usize,
        communication_messages: usize,
        total_tasks: usize,
    ) -> Result<TaskAllocationMetrics> {
        if assigned_costs.len() != optimal_costs.len() || assigned_costs.is_empty() {
            return Err(MetricsError::InvalidInput(
                "assigned_costs and optimal_costs must be non-empty and the same length"
                    .to_string(),
            ));
        }
        if robot_task_counts.is_empty() {
            return Err(MetricsError::InvalidInput(
                "robot_task_counts must not be empty".to_string(),
            ));
        }
        if total_tasks == 0 {
            return Err(MetricsError::InvalidInput(
                "total_tasks must be greater than zero".to_string(),
            ));
        }

        let total_assigned: f64 = assigned_costs.iter().sum();
        let total_optimal: f64 = optimal_costs.iter().sum();
        let allocation_optimality = if total_assigned > 0.0 {
            (total_optimal / total_assigned).min(1.0)
        } else {
            1.0
        };

        let counts_f64: Vec<f64> = robot_task_counts.iter().map(|&c| c as f64).collect();
        let mean_count = mean(&counts_f64);
        let load_balancing = if mean_count > 0.0 {
            let variance = counts_f64
                .iter()
                .map(|c| (c - mean_count).powi(2))
                .sum::<f64>()
                / counts_f64.len() as f64;
            let coefficient_of_variation = variance.sqrt() / mean_count;
            (1.0 - coefficient_of_variation).max(0.0)
        } else {
            1.0
        };

        let failure_adaptation =
            (1.0 - reassignments_due_to_failure as f64 / total_tasks as f64).max(0.0);
        let communication_overhead = communication_messages as f64 / total_tasks as f64;

        let result = TaskAllocationMetrics {
            allocation_optimality,
            load_balancing,
            failure_adaptation,
            communication_overhead,
        };
        self.task_allocation = result.clone();
        Ok(result)
    }

    /// Evaluate the inter-robot communication network from real message
    /// logs.
    pub fn evaluate_network_performance(
        &mut self,
        message_latencies: &[Duration],
        messages_sent: usize,
        messages_lost: usize,
        bytes_sent: u64,
        bandwidth_capacity_bytes_per_sec: f64,
        window: Duration,
    ) -> Result<NetworkPerformanceMetrics> {
        if messages_sent == 0 {
            return Err(MetricsError::InvalidInput(
                "messages_sent must be greater than zero".to_string(),
            ));
        }
        if messages_lost > messages_sent {
            return Err(MetricsError::InvalidInput(
                "messages_lost cannot exceed messages_sent".to_string(),
            ));
        }

        let latency = if message_latencies.is_empty() {
            Duration::from_millis(0)
        } else {
            let total_nanos: u128 = message_latencies.iter().map(|d| d.as_nanos()).sum();
            Duration::from_nanos((total_nanos / message_latencies.len() as u128) as u64)
        };

        let message_loss_rate = messages_lost as f64 / messages_sent as f64;
        let network_reliability = 1.0 - message_loss_rate;

        let bandwidth_utilization =
            if window.as_secs_f64() > 0.0 && bandwidth_capacity_bytes_per_sec > 0.0 {
                (bytes_sent as f64 / window.as_secs_f64()) / bandwidth_capacity_bytes_per_sec
            } else {
                0.0
            };

        let result = NetworkPerformanceMetrics {
            latency,
            message_loss_rate,
            bandwidth_utilization,
            network_reliability,
        };
        self.network_performance = result.clone();
        Ok(result)
    }

    /// Evaluate collective/swarm behavior from real neighbor-distance
    /// samples, a robot-count scaling study, and fault-injection trials.
    ///
    /// `emergent_behavior` compares the swarm's actual measured task
    /// performance against `robot_count * baseline_individual_performance`
    /// (what the same number of robots would achieve acting independently,
    /// with no coordination benefit); a ratio above `1.0` is genuine
    /// positive emergence. `scalability` is the performance-retention ratio
    /// between the largest- and smallest-count entries in
    /// `robot_count_scaling` (fewer than 2 distinct counts makes this
    /// unmeasurable, honestly reported as `1.0`).
    pub fn evaluate_collective_behavior(
        &mut self,
        neighbor_distances: &[f64],
        robot_count_scaling: &[(usize, f64)],
        baseline_individual_performance: f64,
        actual_collective_performance: f64,
        robot_count: usize,
        disturbance_recovery: &[bool],
    ) -> Result<CollectiveBehaviorMetrics> {
        if neighbor_distances.is_empty() {
            return Err(MetricsError::InvalidInput(
                "neighbor_distances must not be empty".to_string(),
            ));
        }

        let swarm_cohesion = 1.0 / (1.0 + mean(neighbor_distances));

        let mut sorted_scaling = robot_count_scaling.to_vec();
        sorted_scaling.sort_by_key(|(count, _)| *count);
        sorted_scaling.dedup_by_key(|(count, _)| *count);
        let scalability = if sorted_scaling.len() < 2 {
            1.0
        } else {
            let (_, perf_min_count) = sorted_scaling[0];
            let (_, perf_max_count) = sorted_scaling[sorted_scaling.len() - 1];
            if perf_min_count > 0.0 {
                perf_max_count / perf_min_count
            } else {
                1.0
            }
        };

        let expected_independent_performance = baseline_individual_performance * robot_count as f64;
        let emergent_behavior = if expected_independent_performance > 0.0 {
            actual_collective_performance / expected_independent_performance
        } else {
            f64::NAN
        };

        let fault_tolerance = if disturbance_recovery.is_empty() {
            1.0
        } else {
            disturbance_recovery.iter().filter(|&&r| r).count() as f64
                / disturbance_recovery.len() as f64
        };

        let result = CollectiveBehaviorMetrics {
            swarm_cohesion,
            emergent_behavior,
            scalability,
            fault_tolerance,
        };
        self.collective_behavior = result.clone();
        Ok(result)
    }
}

// Default implementations
impl Default for FormationControlMetrics {
    fn default() -> Self {
        Self {
            formation_accuracy: 1.0,
            convergence_time: Duration::from_secs(0),
            stability: 1.0,
            coordination_quality: 1.0,
        }
    }
}

impl Default for TaskAllocationMetrics {
    fn default() -> Self {
        Self {
            allocation_optimality: 1.0,
            load_balancing: 1.0,
            failure_adaptation: 1.0,
            communication_overhead: 0.0,
        }
    }
}

impl Default for NetworkPerformanceMetrics {
    fn default() -> Self {
        Self {
            latency: Duration::from_millis(0),
            message_loss_rate: 0.0,
            bandwidth_utilization: 0.5,
            network_reliability: 1.0,
        }
    }
}

impl Default for CollectiveBehaviorMetrics {
    fn default() -> Self {
        Self {
            swarm_cohesion: 1.0,
            emergent_behavior: 1.0,
            scalability: 1.0,
            fault_tolerance: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formation_control_finds_real_convergence_time() {
        let actual = vec![
            vec![[0.5, 0.0, 0.0], [1.5, 0.0, 0.0]],
            vec![[0.2, 0.0, 0.0], [1.2, 0.0, 0.0]],
            vec![[0.05, 0.0, 0.0], [1.05, 0.0, 0.0]],
        ];
        let target = vec![
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ];
        let mut metrics = MultiRobotMetrics::new();
        let result = metrics
            .evaluate_formation_control(&actual, &target, 0.1, Duration::from_secs(1))
            .expect("evaluation should succeed");

        assert!(
            (result.formation_accuracy - 0.8).abs() < 1e-9,
            "got {}",
            result.formation_accuracy
        );
        // Error drops below 0.1 only at the third snapshot (index 2).
        assert_eq!(result.convergence_time, Duration::from_secs(2));
        assert!(result.stability > 0.0 && result.stability < 1.0);
        assert!(result.coordination_quality > 0.0 && result.coordination_quality < 1.0);
    }

    #[test]
    fn formation_control_reports_full_window_when_never_converged() {
        let actual = vec![
            vec![[0.5, 0.0, 0.0]],
            vec![[0.2, 0.0, 0.0]],
            vec![[0.05, 0.0, 0.0]],
        ];
        let target = vec![vec![[0.0, 0.0, 0.0]]; 3];
        let mut metrics = MultiRobotMetrics::new();
        let result = metrics
            .evaluate_formation_control(&actual, &target, 0.01, Duration::from_secs(1))
            .expect("evaluation should succeed");
        assert_eq!(result.convergence_time, Duration::from_secs(2));
    }

    #[test]
    fn task_allocation_computes_real_metrics_not_hardcoded() {
        let mut metrics = MultiRobotMetrics::new();
        let result = metrics
            .evaluate_task_allocation(
                &[10.0, 20.0, 15.0],
                &[8.0, 18.0, 15.0],
                &[3, 3, 4],
                1,
                25,
                10,
            )
            .expect("evaluation should succeed");

        assert!(
            (result.allocation_optimality - 41.0 / 45.0).abs() < 1e-9,
            "got {}",
            result.allocation_optimality
        );
        assert!(result.load_balancing < 1.0 && result.load_balancing > 0.0);
        assert!((result.failure_adaptation - 0.9).abs() < 1e-9);
        assert!((result.communication_overhead - 2.5).abs() < 1e-9);
    }

    #[test]
    fn network_performance_computes_real_rates() {
        let mut metrics = MultiRobotMetrics::new();
        let result = metrics
            .evaluate_network_performance(
                &[Duration::from_millis(50), Duration::from_millis(150)],
                100,
                5,
                1_000_000,
                2_000_000.0,
                Duration::from_secs(1),
            )
            .expect("evaluation should succeed");

        assert_eq!(result.latency, Duration::from_millis(100));
        assert!((result.message_loss_rate - 0.05).abs() < 1e-9);
        assert!((result.network_reliability - 0.95).abs() < 1e-9);
        assert!((result.bandwidth_utilization - 0.5).abs() < 1e-9);
    }

    #[test]
    fn network_performance_rejects_impossible_loss_count() {
        let mut metrics = MultiRobotMetrics::new();
        assert!(metrics
            .evaluate_network_performance(&[], 10, 20, 0, 1.0, Duration::from_secs(1))
            .is_err());
    }

    #[test]
    fn collective_behavior_computes_real_scalability_and_emergence() {
        let mut metrics = MultiRobotMetrics::new();
        let result = metrics
            .evaluate_collective_behavior(
                &[1.0, 2.0, 3.0],
                &[(2, 10.0), (4, 9.0), (8, 7.0)],
                2.0,
                7.0,
                8,
                &[true, true, false, true],
            )
            .expect("evaluation should succeed");

        assert!(
            (result.swarm_cohesion - 1.0 / 3.0).abs() < 1e-9,
            "got {}",
            result.swarm_cohesion
        );
        // Performance retention from count=2 (10.0) to count=8 (7.0): 7/10 = 0.7
        assert!(
            (result.scalability - 0.7).abs() < 1e-9,
            "got {}",
            result.scalability
        );
        // 7.0 actual vs 2.0*8=16.0 expected-independent -> sub-additive, 0.4375
        assert!(
            (result.emergent_behavior - 0.4375).abs() < 1e-9,
            "got {}",
            result.emergent_behavior
        );
        assert!((result.fault_tolerance - 0.75).abs() < 1e-9);
    }

    #[test]
    fn collective_behavior_scalability_vacuous_with_one_data_point() {
        let mut metrics = MultiRobotMetrics::new();
        let result = metrics
            .evaluate_collective_behavior(&[1.0], &[(4, 5.0)], 1.0, 4.0, 4, &[])
            .expect("evaluation should succeed");
        assert_eq!(result.scalability, 1.0);
        assert_eq!(result.fault_tolerance, 1.0);
    }
}
