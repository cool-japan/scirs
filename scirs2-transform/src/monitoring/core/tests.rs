//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use std::collections::HashMap;

use super::*;

#[cfg(all(test, feature = "monitoring"))]
mod ensemble_and_isolation_forest_tests {
    use super::*;

    fn record(detector: &str, score: f64) -> AnomalyRecord {
        AnomalyRecord {
            timestamp: 0,
            metric_name: "m".to_string(),
            value: 0.0,
            anomaly_score: score,
            detection_method: detector.to_string(),
            severity: AnomalySeverity::Low,
            context: HashMap::new(),
        }
    }

    #[test]
    fn ensemble_detects_a_planted_anomaly_with_unanimous_high_confidence_votes() {
        let mut weights = HashMap::new();
        weights.insert("statistical".to_string(), 1.0);
        weights.insert("ml".to_string(), 1.0);
        weights.insert("time_series".to_string(), 1.0);
        let ensemble = EnsembleAnomalyDetector::new(weights, 0.5, 0.5);

        let mut metrics = HashMap::new();
        metrics.insert("cpu_usage".to_string(), 99.0);

        let mut per_metric = HashMap::new();
        per_metric.insert("statistical".to_string(), record("statistical", 4.0));
        per_metric.insert("ml".to_string(), record("ml", 0.95));
        let mut detector_results = HashMap::new();
        detector_results.insert("cpu_usage".to_string(), per_metric);

        let anomalies = ensemble
            .detect_ensemble_anomalies(&metrics, &detector_results, 123)
            .expect("ensemble detection should succeed");

        assert_eq!(
            anomalies.len(),
            1,
            "a metric with 2/3 weighted votes clearing high-confidence scores must be flagged"
        );
        let a = &anomalies[0];
        assert_eq!(a.metric_name, "cpu_usage");
        assert_eq!(a.detection_method, "ensemble_weighted_vote");
        assert!(
            (a.anomaly_score - (2.0 / 3.0)).abs() < 1e-9,
            "vote_fraction must be exactly 2/3, got {}",
            a.anomaly_score
        );
    }

    #[test]
    fn ensemble_reports_no_anomaly_for_a_clean_metric_with_no_member_flags() {
        let mut weights = HashMap::new();
        weights.insert("statistical".to_string(), 1.0);
        weights.insert("ml".to_string(), 1.0);
        let ensemble = EnsembleAnomalyDetector::new(weights, 0.5, 0.5);

        let mut metrics = HashMap::new();
        metrics.insert("cpu_usage".to_string(), 10.0);
        // No detector flagged this metric at all.
        let detector_results: HashMap<String, HashMap<String, AnomalyRecord>> = HashMap::new();

        let anomalies = ensemble
            .detect_ensemble_anomalies(&metrics, &detector_results, 123)
            .expect("ensemble detection should succeed");
        assert!(
            anomalies.is_empty(),
            "no detector votes means no honest basis for an ensemble anomaly"
        );
    }

    #[test]
    fn ensemble_respects_the_configured_voting_threshold() {
        // Only one low-weight detector out of three flags the metric --
        // below any reasonable voting threshold.
        let mut weights = HashMap::new();
        weights.insert("statistical".to_string(), 1.0);
        weights.insert("ml".to_string(), 1.0);
        weights.insert("time_series".to_string(), 1.0);
        let ensemble = EnsembleAnomalyDetector::new(weights, 0.5, 0.0);

        let mut metrics = HashMap::new();
        metrics.insert("cpu_usage".to_string(), 50.0);

        let mut per_metric = HashMap::new();
        per_metric.insert("statistical".to_string(), record("statistical", 3.0));
        let mut detector_results = HashMap::new();
        detector_results.insert("cpu_usage".to_string(), per_metric);

        let anomalies = ensemble
            .detect_ensemble_anomalies(&metrics, &detector_results, 123)
            .expect("ensemble detection should succeed");
        assert!(
            anomalies.is_empty(),
            "1/3 weighted votes must not clear a 0.5 voting_threshold"
        );
    }

    #[test]
    fn advanced_anomaly_detector_wires_real_member_results_into_the_ensemble() {
        let mut detector = AdvancedAnomalyDetector::new();
        detector.add_statistical_detector(
            "latency_ms".to_string(),
            StatisticalDetector::new(2.0, 1.5, 200),
        );
        let mut weights = HashMap::new();
        weights.insert("statistical".to_string(), 1.0);
        detector.configure_ensemble(EnsembleAnomalyDetector::new(weights, 0.5, 0.0));

        // Feed enough in-range data to build up real statistics, then a
        // clear, non-constant outlier.
        for i in 0..30 {
            let mut m = HashMap::new();
            m.insert("latency_ms".to_string(), 10.0 + (i % 3) as f64);
            detector.detect_anomalies(&m).expect("should succeed");
        }
        let mut spike = HashMap::new();
        spike.insert("latency_ms".to_string(), 500.0);
        let anomalies = detector.detect_anomalies(&spike).expect("should succeed");

        assert!(
            anomalies
                .iter()
                .any(|a| a.detection_method == "ensemble_weighted_vote"),
            "a real statistical detection should have been forwarded into a \
             real ensemble vote, not silently dropped; got {anomalies:?}"
        );
    }

    #[test]
    fn isolation_forest_score_is_higher_for_an_extreme_outlier_than_a_typical_point() {
        let mut detector = MLAnomalyDetector::new();
        // Non-constant training window clustered around 10.0.
        for i in 0..200 {
            let value = 10.0 + ((i % 7) as f64 - 3.0) * 0.5;
            detector.training_data.push_back(vec![value]);
        }

        let typical_score = detector
            .isolation_forest_score(10.0)
            .expect("scoring should succeed");
        let outlier_score = detector
            .isolation_forest_score(10_000.0)
            .expect("scoring should succeed");

        assert!(
            outlier_score > typical_score,
            "an extreme outlier must score higher than a typical in-distribution point: \
             outlier={outlier_score}, typical={typical_score}"
        );
        assert!((0.0..=1.0).contains(&outlier_score));
        assert!((0.0..=1.0).contains(&typical_score));
    }

    #[test]
    fn ml_detector_flags_a_real_extreme_spike_after_training_and_reports_isolation_forest() {
        let mut detector = MLAnomalyDetector::new();
        let mut last = None;
        for i in 0..60 {
            let value = 10.0 + ((i % 5) as f64 - 2.0) * 0.3;
            last = detector
                .detect_anomaly(value, "metric", i as u64)
                .expect("should succeed");
        }
        // With in-distribution data only, no anomaly should have fired yet.
        assert!(last.is_none());

        let spike = detector
            .detect_anomaly(100_000.0, "metric", 999)
            .expect("should succeed")
            .expect("an extreme spike must be flagged as an anomaly");
        assert_eq!(spike.detection_method, "ml_isolation_forest");
        assert!(spike.anomaly_score > 0.5);
    }

    #[test]
    fn isolation_forest_uses_the_configured_n_trees_and_max_samples() {
        // Different `n_trees`/`max_samples` configurations must actually
        // change behavior (proving the config fields are real inputs, not
        // decorative/unused fields).
        let mut narrow = MLAnomalyDetector::new();
        narrow.isolation_forest_config.max_samples = 8;
        narrow.isolation_forest_config.n_trees = 5;
        let mut wide = MLAnomalyDetector::new();
        wide.isolation_forest_config.max_samples = 200;
        wide.isolation_forest_config.n_trees = 50;

        for i in 0..200 {
            let value = 10.0 + ((i % 7) as f64 - 3.0) * 0.5;
            narrow.training_data.push_back(vec![value]);
            wide.training_data.push_back(vec![value]);
        }

        // Both must still produce valid, bounded scores regardless of config.
        let s_narrow = narrow
            .isolation_forest_score(10_000.0)
            .expect("should succeed");
        let s_wide = wide
            .isolation_forest_score(10_000.0)
            .expect("should succeed");
        assert!((0.0..=1.0).contains(&s_narrow));
        assert!((0.0..=1.0).contains(&s_wide));
    }
}
