//! Safety and reliability metrics
//!
//! This module provides metrics for evaluating robotic system safety,
//! reliability, failure analysis, and maintenance requirements.

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

/// Safety and reliability evaluation metrics
#[derive(Debug, Clone)]
pub struct SafetyReliabilityMetrics {
    /// Failure analysis metrics
    pub failure_metrics: FailureMetrics,
    /// Risk assessment
    pub risk_assessment: RiskAssessmentMetrics,
    /// System redundancy evaluation
    pub redundancy_metrics: RedundancyMetrics,
    /// Maintenance and diagnostics
    pub maintenance_metrics: MaintenanceMetrics,
}

/// Failure analysis and tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureMetrics {
    /// Mean Time Between Failures (MTBF)
    pub mtbf: Duration,
    /// Mean Time To Repair (MTTR)
    pub mttr: Duration,
    /// Failure rate per operation hour
    pub failure_rate: f64,
    /// Critical failure rate
    pub critical_failure_rate: f64,
    /// System availability
    pub availability: f64,
}

/// Risk assessment metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessmentMetrics {
    /// Overall risk score
    pub overall_risk_score: f64,
    /// Safety integrity level
    pub safety_integrity_level: u8,
    /// Hazard identification coverage
    pub hazard_coverage: f64,
    /// Risk mitigation effectiveness
    pub mitigation_effectiveness: f64,
}

/// System redundancy evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedundancyMetrics {
    /// Redundancy level
    pub redundancy_level: u8,
    /// Graceful degradation capability
    pub graceful_degradation: f64,
    /// Fault detection coverage
    pub fault_detection_coverage: f64,
    /// Recovery time
    pub recovery_time: Duration,
}

/// Maintenance and diagnostics metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaintenanceMetrics {
    /// Predictive maintenance accuracy
    pub predictive_accuracy: f64,
    /// Diagnostic coverage
    pub diagnostic_coverage: f64,
    /// Maintenance efficiency
    pub maintenance_efficiency: f64,
    /// Component life prediction accuracy
    pub life_prediction_accuracy: f64,
}

impl SafetyReliabilityMetrics {
    /// Create new safety and reliability metrics
    pub fn new() -> Self {
        Self {
            failure_metrics: FailureMetrics::default(),
            risk_assessment: RiskAssessmentMetrics::default(),
            redundancy_metrics: RedundancyMetrics::default(),
            maintenance_metrics: MaintenanceMetrics::default(),
        }
    }

    /// Evaluate failure/reliability statistics from a real log of failure
    /// and repair events.
    ///
    /// `operating_intervals`/`repair_durations` are parallel, one pair per
    /// observed failure: how long the system ran before that failure, and
    /// how long the subsequent repair took. `observation_window` is the
    /// total wall-clock duration covered by the log (used to compute
    /// per-hour rates and availability).
    pub fn evaluate_failure_metrics(
        &mut self,
        operating_intervals: &[Duration],
        repair_durations: &[Duration],
        critical_failure_count: usize,
        observation_window: Duration,
    ) -> Result<FailureMetrics> {
        if operating_intervals.len() != repair_durations.len() || operating_intervals.is_empty() {
            return Err(MetricsError::InvalidInput(
                "operating_intervals and repair_durations must be non-empty and the same length"
                    .to_string(),
            ));
        }
        if critical_failure_count > operating_intervals.len() {
            return Err(MetricsError::InvalidInput(
                "critical_failure_count cannot exceed the total number of recorded failures"
                    .to_string(),
            ));
        }
        let window_hours = observation_window.as_secs_f64() / 3600.0;
        if window_hours <= 0.0 {
            return Err(MetricsError::InvalidInput(
                "observation_window must be greater than zero".to_string(),
            ));
        }

        let mtbf = mean_duration(operating_intervals);
        let mttr = mean_duration(repair_durations);
        let total_downtime_secs: f64 = repair_durations.iter().map(|d| d.as_secs_f64()).sum();

        let failure_count = operating_intervals.len();
        let failure_rate = failure_count as f64 / window_hours;
        let critical_failure_rate = critical_failure_count as f64 / window_hours;
        let availability = ((observation_window.as_secs_f64() - total_downtime_secs)
            / observation_window.as_secs_f64())
        .clamp(0.0, 1.0);

        let result = FailureMetrics {
            mtbf,
            mttr,
            failure_rate,
            critical_failure_rate,
            availability,
        };
        self.failure_metrics = result.clone();
        Ok(result)
    }

    /// Evaluate risk from a real per-hazard severity/likelihood/mitigation
    /// log.
    ///
    /// `overall_risk_score` is the mean of `severity[i] * likelihood[i]`
    /// (the standard risk = severity x likelihood formulation) across all
    /// identified hazards. `safety_integrity_level` is not derived here: it
    /// is the outcome of a formal hazard-and-risk-assessment process (e.g.
    /// IEC 61508 / ISO 26262), external to this scalar risk score, so it is
    /// accepted as an already-assessed input and passed through rather than
    /// invented from a single-number formula.
    pub fn evaluate_risk_assessment(
        &mut self,
        severity_scores: &[f64],
        likelihood_scores: &[f64],
        mitigations_effective: &[bool],
        hazards_identified: usize,
        hazards_total_known: usize,
        assessed_safety_integrity_level: u8,
    ) -> Result<RiskAssessmentMetrics> {
        if severity_scores.len() != likelihood_scores.len() || severity_scores.is_empty() {
            return Err(MetricsError::InvalidInput(
                "severity_scores and likelihood_scores must be non-empty and the same length"
                    .to_string(),
            ));
        }
        if hazards_identified > hazards_total_known || hazards_total_known == 0 {
            return Err(MetricsError::InvalidInput(
                "hazards_identified must be <= hazards_total_known, and hazards_total_known must be > 0"
                    .to_string(),
            ));
        }

        let overall_risk_score = severity_scores
            .iter()
            .zip(likelihood_scores.iter())
            .map(|(s, l)| s * l)
            .sum::<f64>()
            / severity_scores.len() as f64;

        let hazard_coverage = hazards_identified as f64 / hazards_total_known as f64;
        let mitigation_effectiveness = if mitigations_effective.is_empty() {
            1.0
        } else {
            mitigations_effective.iter().filter(|&&m| m).count() as f64
                / mitigations_effective.len() as f64
        };

        let result = RiskAssessmentMetrics {
            overall_risk_score,
            safety_integrity_level: assessed_safety_integrity_level,
            hazard_coverage,
            mitigation_effectiveness,
        };
        self.risk_assessment = result.clone();
        Ok(result)
    }

    /// Evaluate redundancy/fault-tolerance from real fault-injection trials.
    ///
    /// `redundancy_level` (the number of redundant backup units actually
    /// provisioned) is a designed architectural property, not a measured
    /// rate, so it is accepted as a direct input and passed through.
    pub fn evaluate_redundancy(
        &mut self,
        component_failures_survived: &[bool],
        fault_detections: &[bool],
        recovery_times: &[Duration],
        redundancy_level: u8,
    ) -> Result<RedundancyMetrics> {
        if component_failures_survived.is_empty() {
            return Err(MetricsError::InvalidInput(
                "component_failures_survived must not be empty".to_string(),
            ));
        }
        if fault_detections.is_empty() {
            return Err(MetricsError::InvalidInput(
                "fault_detections must not be empty".to_string(),
            ));
        }

        let graceful_degradation = component_failures_survived.iter().filter(|&&s| s).count()
            as f64
            / component_failures_survived.len() as f64;
        let fault_detection_coverage =
            fault_detections.iter().filter(|&&d| d).count() as f64 / fault_detections.len() as f64;
        let recovery_time = mean_duration(recovery_times);

        let result = RedundancyMetrics {
            redundancy_level,
            graceful_degradation,
            fault_detection_coverage,
            recovery_time,
        };
        self.redundancy_metrics = result.clone();
        Ok(result)
    }

    /// Evaluate maintenance/diagnostics quality from real prediction-vs-outcome
    /// logs.
    ///
    /// `maintenance_efficiency` compares scheduled vs. actual maintenance
    /// duration (`scheduled / actual`, capped at `1.0`).
    /// `life_prediction_accuracy` is `1 - mean(|predicted - actual| / actual)`
    /// (a MAPE-based accuracy), clamped to `[0, 1]`.
    pub fn evaluate_maintenance(
        &mut self,
        predicted_failures: &[bool],
        actual_failures: &[bool],
        diagnostics_correct: &[bool],
        scheduled_durations: &[Duration],
        actual_durations: &[Duration],
        predicted_remaining_life: &[f64],
        actual_remaining_life: &[f64],
    ) -> Result<MaintenanceMetrics> {
        if predicted_failures.len() != actual_failures.len() || predicted_failures.is_empty() {
            return Err(MetricsError::InvalidInput(
                "predicted_failures and actual_failures must be non-empty and the same length"
                    .to_string(),
            ));
        }
        if scheduled_durations.len() != actual_durations.len() || scheduled_durations.is_empty() {
            return Err(MetricsError::InvalidInput(
                "scheduled_durations and actual_durations must be non-empty and the same length"
                    .to_string(),
            ));
        }
        if predicted_remaining_life.len() != actual_remaining_life.len()
            || predicted_remaining_life.is_empty()
        {
            return Err(MetricsError::InvalidInput(
                "predicted_remaining_life and actual_remaining_life must be non-empty and the same length"
                    .to_string(),
            ));
        }
        if actual_remaining_life.iter().any(|&a| a <= 0.0) {
            return Err(MetricsError::InvalidInput(
                "actual_remaining_life values must be strictly positive".to_string(),
            ));
        }

        let predictive_accuracy = predicted_failures
            .iter()
            .zip(actual_failures.iter())
            .filter(|(p, a)| p == a)
            .count() as f64
            / predicted_failures.len() as f64;

        let diagnostic_coverage = if diagnostics_correct.is_empty() {
            1.0
        } else {
            diagnostics_correct.iter().filter(|&&d| d).count() as f64
                / diagnostics_correct.len() as f64
        };

        let efficiency_ratios: Vec<f64> = scheduled_durations
            .iter()
            .zip(actual_durations.iter())
            .map(|(sched, actual)| {
                if actual.as_secs_f64() > 0.0 {
                    (sched.as_secs_f64() / actual.as_secs_f64()).min(1.0)
                } else {
                    1.0
                }
            })
            .collect();
        let maintenance_efficiency =
            efficiency_ratios.iter().sum::<f64>() / efficiency_ratios.len() as f64;

        let mape = predicted_remaining_life
            .iter()
            .zip(actual_remaining_life.iter())
            .map(|(p, a)| (p - a).abs() / a.abs())
            .sum::<f64>()
            / predicted_remaining_life.len() as f64;
        let life_prediction_accuracy = (1.0 - mape).clamp(0.0, 1.0);

        let result = MaintenanceMetrics {
            predictive_accuracy,
            diagnostic_coverage,
            maintenance_efficiency,
            life_prediction_accuracy,
        };
        self.maintenance_metrics = result.clone();
        Ok(result)
    }
}

// Default implementations
//
// Neutral, not-yet-evaluated baselines (matching the convention used across
// this module: `1.0`/max-integrity for a not-yet-measured rate, `0.0` for a
// not-yet-observed error rate) -- not a fabricated "typical fleet" measurement.
// Call `evaluate_failure_metrics` / `evaluate_risk_assessment` /
// `evaluate_redundancy` / `evaluate_maintenance` to replace them with real
// computed values.
impl Default for FailureMetrics {
    fn default() -> Self {
        Self {
            mtbf: Duration::from_secs(0),
            mttr: Duration::from_secs(0),
            failure_rate: 0.0,
            critical_failure_rate: 0.0,
            availability: 1.0,
        }
    }
}

impl Default for RiskAssessmentMetrics {
    fn default() -> Self {
        Self {
            overall_risk_score: 0.0,
            safety_integrity_level: 0,
            hazard_coverage: 1.0,
            mitigation_effectiveness: 1.0,
        }
    }
}

impl Default for RedundancyMetrics {
    fn default() -> Self {
        Self {
            redundancy_level: 0,
            graceful_degradation: 1.0,
            fault_detection_coverage: 1.0,
            recovery_time: Duration::from_secs(0),
        }
    }
}

impl Default for MaintenanceMetrics {
    fn default() -> Self {
        Self {
            predictive_accuracy: 1.0,
            diagnostic_coverage: 1.0,
            maintenance_efficiency: 1.0,
            life_prediction_accuracy: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failure_metrics_computes_real_availability_not_hardcoded() {
        let mut metrics = SafetyReliabilityMetrics::new();
        let result = metrics
            .evaluate_failure_metrics(
                &[
                    Duration::from_secs(3600 * 10),
                    Duration::from_secs(3600 * 20),
                ],
                &[Duration::from_secs(3600), Duration::from_secs(3600 * 3)],
                1,
                Duration::from_secs(3600 * 100),
            )
            .expect("evaluation should succeed");

        assert_eq!(result.mtbf, Duration::from_secs(3600 * 15));
        assert_eq!(result.mttr, Duration::from_secs(3600 * 2));
        assert!(
            (result.failure_rate - 0.02).abs() < 1e-9,
            "got {}",
            result.failure_rate
        );
        assert!((result.critical_failure_rate - 0.01).abs() < 1e-9);
        assert!(
            (result.availability - 0.96).abs() < 1e-9,
            "got {}",
            result.availability
        );
    }

    #[test]
    fn failure_metrics_rejects_more_critical_than_total_failures() {
        let mut metrics = SafetyReliabilityMetrics::new();
        assert!(metrics
            .evaluate_failure_metrics(
                &[Duration::from_secs(10)],
                &[Duration::from_secs(1)],
                5,
                Duration::from_secs(100),
            )
            .is_err());
    }

    #[test]
    fn risk_assessment_computes_real_score_and_passes_through_sil() {
        let mut metrics = SafetyReliabilityMetrics::new();
        let result = metrics
            .evaluate_risk_assessment(&[0.8, 0.5], &[0.3, 0.6], &[true, false, true], 8, 10, 3)
            .expect("evaluation should succeed");

        // mean(0.8*0.3, 0.5*0.6) = mean(0.24, 0.30) = 0.27
        assert!(
            (result.overall_risk_score - 0.27).abs() < 1e-9,
            "got {}",
            result.overall_risk_score
        );
        assert_eq!(result.safety_integrity_level, 3);
        assert!((result.hazard_coverage - 0.8).abs() < 1e-9);
        assert!((result.mitigation_effectiveness - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn redundancy_computes_real_coverage() {
        let mut metrics = SafetyReliabilityMetrics::new();
        let result = metrics
            .evaluate_redundancy(
                &[true, true, false, true],
                &[true, true, true, false, true],
                &[Duration::from_millis(500), Duration::from_millis(1500)],
                3,
            )
            .expect("evaluation should succeed");

        assert!((result.graceful_degradation - 0.75).abs() < 1e-9);
        assert!((result.fault_detection_coverage - 0.8).abs() < 1e-9);
        assert_eq!(result.recovery_time, Duration::from_millis(1000));
        assert_eq!(result.redundancy_level, 3);
    }

    #[test]
    fn maintenance_computes_real_metrics_not_hardcoded() {
        let mut metrics = SafetyReliabilityMetrics::new();
        let result = metrics
            .evaluate_maintenance(
                &[true, false, true, false],
                &[true, false, false, false],
                &[true, true, false],
                &[Duration::from_secs(60), Duration::from_secs(100)],
                &[Duration::from_secs(80), Duration::from_secs(100)],
                &[100.0, 50.0],
                &[80.0, 60.0],
            )
            .expect("evaluation should succeed");

        assert!(
            (result.predictive_accuracy - 0.75).abs() < 1e-9,
            "got {}",
            result.predictive_accuracy
        );
        assert!((result.diagnostic_coverage - 2.0 / 3.0).abs() < 1e-9);
        // ratios: 60/80=0.75, 100/100=1.0 -> mean 0.875
        assert!(
            (result.maintenance_efficiency - 0.875).abs() < 1e-9,
            "got {}",
            result.maintenance_efficiency
        );
        // MAPE = mean(|100-80|/80, |50-60|/60) = mean(0.25, 0.16667) = 0.208333
        let expected_life_accuracy = 1.0 - (0.25 + 1.0 / 6.0) / 2.0;
        assert!(
            (result.life_prediction_accuracy - expected_life_accuracy).abs() < 1e-6,
            "got {}",
            result.life_prediction_accuracy
        );
    }

    #[test]
    fn maintenance_rejects_non_positive_actual_life() {
        let mut metrics = SafetyReliabilityMetrics::new();
        assert!(metrics
            .evaluate_maintenance(
                &[true],
                &[true],
                &[true],
                &[Duration::from_secs(1)],
                &[Duration::from_secs(1)],
                &[10.0],
                &[0.0],
            )
            .is_err());
    }
}
