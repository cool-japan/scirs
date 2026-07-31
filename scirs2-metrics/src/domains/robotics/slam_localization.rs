//! SLAM and localization metrics
//!
//! This module provides metrics for evaluating Simultaneous Localization and Mapping (SLAM)
//! systems, including localization accuracy, mapping quality, and loop closure performance.

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use super::core::{utils, DriftMetrics, ErrorStatistics, Pose, RealTimePerformanceMetrics};
use crate::error::{MetricsError, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// SLAM and localization metrics
#[derive(Debug, Clone)]
pub struct SlamMetrics {
    /// Localization accuracy metrics
    pub localization_metrics: LocalizationAccuracyMetrics,
    /// Mapping quality metrics
    pub mapping_metrics: MappingQualityMetrics,
    /// Loop closure metrics
    pub loop_closure_metrics: LoopClosureMetrics,
    /// Computational efficiency
    pub computational_metrics: SlamComputationalMetrics,
}

/// Localization accuracy evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalizationAccuracyMetrics {
    /// Absolute Trajectory Error (ATE)
    pub absolute_trajectory_error: f64,
    /// Relative Pose Error (RPE)
    pub relative_pose_error: f64,
    /// Translation error statistics
    pub translation_error: ErrorStatistics,
    /// Rotation error statistics
    pub rotation_error: ErrorStatistics,
    /// Drift analysis
    pub drift_metrics: DriftMetrics,
}

/// Mapping quality evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MappingQualityMetrics {
    /// Map completeness ratio
    pub completeness: f64,
    /// Map accuracy compared to ground truth
    pub map_accuracy: f64,
    /// Feature detection rate
    pub feature_detection_rate: f64,
    /// Map consistency metrics
    pub consistency_metrics: MapConsistencyMetrics,
}

/// Map consistency evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MapConsistencyMetrics {
    /// Feature matching consistency
    pub feature_consistency: f64,
    /// Geometric consistency
    pub geometric_consistency: f64,
    /// Temporal consistency
    pub temporal_consistency: f64,
    /// Global consistency score
    pub global_consistency: f64,
}

/// Loop closure detection and quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoopClosureMetrics {
    /// Detection rate (true positives / total loops)
    pub detection_rate: f64,
    /// False positive rate
    pub false_positive_rate: f64,
    /// Loop closure accuracy
    pub closure_accuracy: f64,
    /// Time to detect loops
    pub detection_time: Duration,
    /// Graph optimization convergence
    pub optimization_convergence: f64,
}

/// SLAM computational performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlamComputationalMetrics {
    /// Float-time performance
    pub real_time_performance: RealTimePerformanceMetrics,
    /// Memory usage for maps
    pub map_memory_usage: f64,
    /// Keyframe processing time
    pub keyframe_processing_time: Duration,
    /// Graph optimization time
    pub optimization_time: Duration,
}

impl SlamMetrics {
    /// Create new SLAM metrics
    pub fn new() -> Self {
        Self {
            localization_metrics: LocalizationAccuracyMetrics::default(),
            mapping_metrics: MappingQualityMetrics::default(),
            loop_closure_metrics: LoopClosureMetrics::default(),
            computational_metrics: SlamComputationalMetrics::default(),
        }
    }

    /// Evaluate localization accuracy from an estimated trajectory against ground truth.
    ///
    /// Computes the standard SLAM/odometry accuracy measures:
    /// - **Absolute Trajectory Error (ATE)**: RMSE of per-pose position error
    ///   after pairing each estimated pose with its ground-truth counterpart.
    /// - **Relative Pose Error (RPE)**: RMSE of the difference between
    ///   consecutive-pose displacement in the estimate vs. ground truth --
    ///   captures local drift independent of any global offset.
    /// - **Drift**: cumulative final-pose error normalized by path length and
    ///   elapsed time (rate), plus a coefficient-of-variation-based
    ///   "inconsistency" score (lower is better, matching
    ///   [`DriftMetrics::drift_consistency`]'s documented convention).
    ///
    /// `estimated` and `ground_truth` must be the same non-empty length
    /// (one pose per synchronized timestep).
    pub fn evaluate_localization_accuracy(
        &mut self,
        estimated: &[Pose],
        ground_truth: &[Pose],
        elapsed_time: Duration,
    ) -> Result<LocalizationAccuracyMetrics> {
        if estimated.len() != ground_truth.len() {
            return Err(MetricsError::InvalidInput(format!(
                "estimated ({}) and ground_truth ({}) trajectories must have the same length",
                estimated.len(),
                ground_truth.len()
            )));
        }
        if estimated.is_empty() {
            return Err(MetricsError::InvalidInput(
                "estimated/ground_truth trajectories must not be empty".to_string(),
            ));
        }

        // Absolute Trajectory Error: per-pose position/rotation error statistics.
        let position_errors: Vec<f64> = estimated
            .iter()
            .zip(ground_truth.iter())
            .map(|(e, g)| e.distance_to(g))
            .collect();
        let rotation_errors: Vec<f64> = estimated
            .iter()
            .zip(ground_truth.iter())
            .map(|(e, g)| e.angular_distance_to(g))
            .collect();
        let translation_error = utils::calculate_statistics(&position_errors);
        let rotation_error = utils::calculate_statistics(&rotation_errors);
        let absolute_trajectory_error = translation_error.rmse;

        // Relative Pose Error: consecutive-frame relative-displacement error.
        let mut rpe_sq_sum = 0.0;
        let mut rpe_count = 0usize;
        for i in 1..estimated.len() {
            let est_delta = estimated[i].distance_to(&estimated[i - 1]);
            let gt_delta = ground_truth[i].distance_to(&ground_truth[i - 1]);
            let diff = est_delta - gt_delta;
            rpe_sq_sum += diff * diff;
            rpe_count += 1;
        }
        let relative_pose_error = if rpe_count > 0 {
            (rpe_sq_sum / rpe_count as f64).sqrt()
        } else {
            0.0
        };

        // Drift: normalize the cumulative final-pose error by elapsed time.
        let cumulative_drift = *position_errors.last().unwrap_or(&0.0);
        let elapsed_secs = elapsed_time.as_secs_f64();
        let linear_drift_rate = if elapsed_secs > 0.0 {
            cumulative_drift / elapsed_secs
        } else {
            0.0
        };
        let cumulative_angular_drift = *rotation_errors.last().unwrap_or(&0.0);
        let angular_drift_rate = if elapsed_secs > 0.0 {
            cumulative_angular_drift / elapsed_secs
        } else {
            0.0
        };
        // Coefficient of variation of the position error: how erratically the
        // error grows/shrinks along the trajectory (0 = perfectly steady drift).
        let drift_consistency = if translation_error.mean_error.abs() > 1e-12 {
            translation_error.std_error / translation_error.mean_error.abs()
        } else {
            0.0
        };

        let drift_metrics = DriftMetrics {
            linear_drift_rate,
            angular_drift_rate,
            cumulative_drift,
            drift_consistency,
        };

        let result = LocalizationAccuracyMetrics {
            absolute_trajectory_error,
            relative_pose_error,
            translation_error,
            rotation_error,
            drift_metrics,
        };
        self.localization_metrics = result.clone();
        Ok(result)
    }

    /// Evaluate mapping quality from occupancy-grid overlap with ground truth.
    ///
    /// `estimated_occupancy`/`ground_truth_occupancy` are equal-length,
    /// flattened boolean occupancy grids (`true` = occupied). `detected_features`
    /// out of `total_features` known landmarks/features is used for the
    /// feature-detection rate (pass `0`/`0` if no feature-tracking data is
    /// available; the rate is then vacuously `1.0` rather than fabricated).
    ///
    /// [`MapConsistencyMetrics`] cannot be computed from a single occupancy
    /// snapshot -- consistency requires comparing multiple mapping passes or
    /// loop-closure re-visits, neither of which this method receives. Those
    /// four sub-fields are honestly reported as `NaN` ("not computable from
    /// the available data") rather than a fabricated perfect score.
    pub fn evaluate_mapping_quality(
        &mut self,
        estimated_occupancy: &[bool],
        ground_truth_occupancy: &[bool],
        detected_features: usize,
        total_features: usize,
    ) -> Result<MappingQualityMetrics> {
        if estimated_occupancy.len() != ground_truth_occupancy.len() {
            return Err(MetricsError::InvalidInput(format!(
                "estimated_occupancy ({}) and ground_truth_occupancy ({}) must have the same length",
                estimated_occupancy.len(),
                ground_truth_occupancy.len()
            )));
        }
        if estimated_occupancy.is_empty() {
            return Err(MetricsError::InvalidInput(
                "occupancy grids must not be empty".to_string(),
            ));
        }

        let total = estimated_occupancy.len();
        let matching = estimated_occupancy
            .iter()
            .zip(ground_truth_occupancy.iter())
            .filter(|(e, g)| e == g)
            .count();
        let map_accuracy = matching as f64 / total as f64;

        let gt_occupied = ground_truth_occupancy.iter().filter(|&&b| b).count();
        let completeness = if gt_occupied == 0 {
            1.0
        } else {
            let recalled = estimated_occupancy
                .iter()
                .zip(ground_truth_occupancy.iter())
                .filter(|(&e, &g)| g && e)
                .count();
            recalled as f64 / gt_occupied as f64
        };

        let feature_detection_rate = if total_features == 0 {
            1.0
        } else {
            detected_features as f64 / total_features as f64
        };

        let consistency_metrics = MapConsistencyMetrics {
            feature_consistency: f64::NAN,
            geometric_consistency: f64::NAN,
            temporal_consistency: f64::NAN,
            global_consistency: f64::NAN,
        };

        let result = MappingQualityMetrics {
            completeness,
            map_accuracy,
            feature_detection_rate,
            consistency_metrics,
        };
        self.mapping_metrics = result.clone();
        Ok(result)
    }

    /// Evaluate loop closure detection quality from real detection counts.
    ///
    /// `true_positives`/`false_positives`/`false_negatives` are loop-closure
    /// detection outcomes counted against a ground-truth set of actual loop
    /// closures. `detection_times` are the wall-clock latencies of each
    /// detection event (used for the mean [`LoopClosureMetrics::detection_time`]).
    ///
    /// [`LoopClosureMetrics::optimization_convergence`] requires observing the
    /// pose-graph optimizer's iteration trace, which this method does not
    /// receive; it is honestly reported as `NaN` rather than fabricated.
    pub fn evaluate_loop_closure(
        &mut self,
        true_positives: usize,
        false_positives: usize,
        false_negatives: usize,
        detection_times: &[Duration],
    ) -> Result<LoopClosureMetrics> {
        let total_actual_loops = true_positives + false_negatives;
        let detection_rate = if total_actual_loops == 0 {
            1.0
        } else {
            true_positives as f64 / total_actual_loops as f64
        };

        let total_detections = true_positives + false_positives;
        let false_positive_rate = if total_detections == 0 {
            0.0
        } else {
            false_positives as f64 / total_detections as f64
        };

        let denom = true_positives + false_positives + false_negatives;
        let closure_accuracy = if denom == 0 {
            1.0
        } else {
            true_positives as f64 / denom as f64
        };

        let detection_time = if detection_times.is_empty() {
            Duration::from_millis(0)
        } else {
            let total_nanos: u128 = detection_times.iter().map(|d| d.as_nanos()).sum();
            Duration::from_nanos((total_nanos / detection_times.len() as u128) as u64)
        };

        let result = LoopClosureMetrics {
            detection_rate,
            false_positive_rate,
            closure_accuracy,
            detection_time,
            optimization_convergence: f64::NAN,
        };
        self.loop_closure_metrics = result.clone();
        Ok(result)
    }
}

// Default implementations
impl Default for LocalizationAccuracyMetrics {
    fn default() -> Self {
        Self {
            absolute_trajectory_error: 0.0,
            relative_pose_error: 0.0,
            translation_error: ErrorStatistics::default(),
            rotation_error: ErrorStatistics::default(),
            drift_metrics: DriftMetrics::default(),
        }
    }
}

impl Default for MappingQualityMetrics {
    fn default() -> Self {
        Self {
            completeness: 1.0,
            map_accuracy: 1.0,
            feature_detection_rate: 1.0,
            consistency_metrics: MapConsistencyMetrics::default(),
        }
    }
}

impl Default for MapConsistencyMetrics {
    fn default() -> Self {
        Self {
            feature_consistency: 1.0,
            geometric_consistency: 1.0,
            temporal_consistency: 1.0,
            global_consistency: 1.0,
        }
    }
}

impl Default for LoopClosureMetrics {
    fn default() -> Self {
        Self {
            detection_rate: 1.0,
            false_positive_rate: 0.0,
            closure_accuracy: 1.0,
            detection_time: Duration::from_millis(0),
            optimization_convergence: 1.0,
        }
    }
}

impl Default for SlamComputationalMetrics {
    fn default() -> Self {
        Self {
            real_time_performance: RealTimePerformanceMetrics::default(),
            map_memory_usage: 0.0,
            keyframe_processing_time: Duration::from_millis(0),
            optimization_time: Duration::from_millis(0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pose(x: f64, y: f64) -> Pose {
        Pose::new([x, y, 0.0], [1.0, 0.0, 0.0, 0.0], 1.0)
    }

    #[test]
    fn localization_accuracy_is_zero_for_identical_trajectories() {
        let traj = vec![
            pose(0.0, 0.0),
            pose(1.0, 0.0),
            pose(2.0, 0.0),
            pose(3.0, 0.0),
        ];
        let mut metrics = SlamMetrics::new();
        let result = metrics
            .evaluate_localization_accuracy(&traj, &traj, Duration::from_secs(3))
            .expect("equal-length trajectories should succeed");
        assert_eq!(result.absolute_trajectory_error, 0.0);
        assert_eq!(result.relative_pose_error, 0.0);
        assert_eq!(result.drift_metrics.cumulative_drift, 0.0);
    }

    #[test]
    fn localization_accuracy_computes_real_ate_not_hardcoded() {
        // Ground truth: straight line along x. Estimate: offset by a growing
        // amount at each step (a classic drifting-odometry scenario).
        let ground_truth = vec![
            pose(0.0, 0.0),
            pose(1.0, 0.0),
            pose(2.0, 0.0),
            pose(3.0, 0.0),
        ];
        let estimated = vec![
            pose(0.0, 0.0),
            pose(1.0, 0.1),
            pose(2.0, 0.2),
            pose(3.0, 0.3),
        ];
        let mut metrics = SlamMetrics::new();
        let result = metrics
            .evaluate_localization_accuracy(&estimated, &ground_truth, Duration::from_secs(3))
            .expect("evaluation should succeed");

        // Per-point position errors are exactly [0.0, 0.1, 0.2, 0.3];
        // RMSE = sqrt((0^2+0.1^2+0.2^2+0.3^2)/4) = sqrt(0.14/4) = sqrt(0.035)
        let expected_ate = (0.14_f64 / 4.0).sqrt();
        assert!(
            (result.absolute_trajectory_error - expected_ate).abs() < 1e-9,
            "expected ATE {expected_ate}, got {}",
            result.absolute_trajectory_error
        );
        assert!(
            result.absolute_trajectory_error > 0.0,
            "the old code must not silently report 0.0 for a genuinely drifting trajectory"
        );
        // Final-pose error is the largest (0.3) -> cumulative drift.
        assert!((result.drift_metrics.cumulative_drift - 0.3).abs() < 1e-9);
        assert!(result.drift_metrics.linear_drift_rate > 0.0);
    }

    #[test]
    fn localization_accuracy_rejects_mismatched_lengths() {
        let mut metrics = SlamMetrics::new();
        let short = vec![pose(0.0, 0.0)];
        let long = vec![pose(0.0, 0.0), pose(1.0, 0.0)];
        assert!(metrics
            .evaluate_localization_accuracy(&short, &long, Duration::from_secs(1))
            .is_err());
    }

    #[test]
    fn mapping_quality_computes_real_accuracy_from_occupancy_overlap() {
        let ground_truth = vec![true, true, false, false, true, false];
        // 4 out of 6 cells match ground truth (indices 0,2,3,5 wait -- compute below).
        let estimated = vec![true, false, false, true, true, false];
        // Mismatches at index 1 (true vs false) and index 3 (false vs true) => 4/6 match.
        let mut metrics = SlamMetrics::new();
        let result = metrics
            .evaluate_mapping_quality(&estimated, &ground_truth, 8, 10)
            .expect("evaluation should succeed");

        assert!(
            (result.map_accuracy - 4.0 / 6.0).abs() < 1e-9,
            "expected 4/6 accuracy, got {}",
            result.map_accuracy
        );
        assert!((result.feature_detection_rate - 0.8).abs() < 1e-9);
        // Ground truth has 3 occupied cells (0, 1, 4); estimated correctly
        // flags 2 of them (0 and 4) => completeness = 2/3.
        assert!((result.completeness - 2.0 / 3.0).abs() < 1e-9);
        // Consistency genuinely cannot be computed from a single snapshot.
        assert!(result.consistency_metrics.global_consistency.is_nan());
    }

    #[test]
    fn mapping_quality_perfect_match_gives_perfect_accuracy() {
        let grid = vec![true, false, true, true, false];
        let mut metrics = SlamMetrics::new();
        let result = metrics
            .evaluate_mapping_quality(&grid, &grid, 5, 5)
            .expect("evaluation should succeed");
        assert_eq!(result.map_accuracy, 1.0);
        assert_eq!(result.completeness, 1.0);
    }

    #[test]
    fn loop_closure_computes_real_rates_from_counts() {
        let mut metrics = SlamMetrics::new();
        // 6 real loops exist; detector found 3 (detection_rate = 0.5), plus 2
        // spurious detections (false_positive_rate = 2/5).
        let result = metrics
            .evaluate_loop_closure(
                3,
                2,
                3,
                &[Duration::from_millis(50), Duration::from_millis(150)],
            )
            .expect("evaluation should succeed");

        assert!(
            (result.detection_rate - 0.5).abs() < 1e-9,
            "got {}",
            result.detection_rate
        );
        assert!(
            (result.false_positive_rate - 0.4).abs() < 1e-9,
            "got {}",
            result.false_positive_rate
        );
        assert_eq!(result.detection_time, Duration::from_millis(100));
        assert!(
            result.optimization_convergence.is_nan(),
            "graph-optimizer convergence is not observable from detection counts alone"
        );
    }

    #[test]
    fn loop_closure_vacuous_when_no_loops_exist() {
        let mut metrics = SlamMetrics::new();
        let result = metrics
            .evaluate_loop_closure(0, 0, 0, &[])
            .expect("evaluation should succeed");
        assert_eq!(result.detection_rate, 1.0);
        assert_eq!(result.false_positive_rate, 0.0);
    }
}
