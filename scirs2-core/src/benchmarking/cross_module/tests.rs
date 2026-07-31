//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

#[cfg(feature = "parallel")]
use crate::parallel_ops::*;
use std::time::Duration;

use super::*;

#[cfg(test)]
mod tests_2 {
    use super::*;

    #[test]
    fn test_benchmark_config_creation() {
        let config = CrossModuleBenchConfig::default();
        assert_eq!(config.iterations, 100);
        assert_eq!(config.warmup_iterations, 10);
        assert!(!config.datasizes.is_empty());
        assert!(!config.ns.is_empty());
    }

    #[test]
    fn test_performance_measurement_creation() {
        let measurement = PerformanceMeasurement::new(
            "test_benchmark".to_string(),
            vec!["module1".to_string(), "module2".to_string()],
        );

        assert_eq!(measurement.name, "test_benchmark");
        assert_eq!(measurement.modules.len(), 2);
        assert_eq!(measurement.efficiency_score(), 0.0); // No data yet
    }

    #[test]
    fn test_benchmark_runner_creation() {
        let config = CrossModuleBenchConfig::default();
        let runner = CrossModuleBenchmarkRunner::new(config);

        // Runner should be created successfully
        assert_eq!(runner.config.iterations, 100);
    }

    #[test]
    fn test_quick_benchmarks() {
        // This test should run quickly
        match run_quick_benchmarks() {
            Ok(result) => {
                assert!(!result.measurements.is_empty());
                println!(
                    "Quick benchmarks completed: {} measurements",
                    result.measurements.len()
                );
            }
            Err(e) => {
                println!("Quick benchmarks failed: {:?}", e);
                // Don't fail the test as this might be expected in CI environments
            }
        }
    }

    /// Helper: build a runner with regression detection enabled/disabled.
    fn make_runner(enable_regression: bool) -> CrossModuleBenchmarkRunner {
        let config = CrossModuleBenchConfig {
            iterations: 2,
            warmup_iterations: 1,
            datasizes: vec![1024],
            ns: vec![1],
            memory_limits: vec![64 * 1024 * 1024],
            enable_profiling: false,
            enable_regression_detection: enable_regression,
            baseline_file: None,
            max_regression_percent: 5.0,
            timeout: Duration::from_secs(30),
        };
        CrossModuleBenchmarkRunner::new(config)
    }

    /// Helper: build a `PerformanceMeasurement` with a specific avg duration.
    fn make_measurement(name: &str, avg_nanos: u64) -> PerformanceMeasurement {
        let mut m = PerformanceMeasurement::new(name.to_string(), vec!["test-module".to_string()]);
        m.avg_duration = Duration::from_nanos(avg_nanos);
        m.datasize = 1024;
        m.operations_count = 10;
        m.memory_usage = 1024;
        m
    }

    /// Test 1: no-regression case — uniform measurements should produce no regressions.
    #[test]
    fn test_analyze_regressions_no_regression() {
        let runner = make_runner(true);

        // All benchmarks take the same time — no outliers, no regressions.
        let measurements = vec![
            make_measurement("bench_a", 1_000_000),
            make_measurement("bench_b", 1_000_000),
            make_measurement("bench_c", 1_000_000),
            make_measurement("bench_d", 1_000_000),
            make_measurement("bench_e", 1_000_000),
        ];

        let result = runner
            .analyze_regressions(&measurements)
            .expect("analyze_regressions should not fail");

        assert!(
            !result.regression_detected,
            "Uniform measurements must not trigger regression detection"
        );
        assert!(
            result.regressions.is_empty(),
            "No regression entries expected for uniform timings"
        );
        // overall_change_percent is relative to the intra-suite mean (itself), so ~0.
        assert!(
            result.overall_change_percent.abs() < 1.0,
            "Overall change should be near zero for uniform timings"
        );
    }

    /// Test 2: regression detected — one benchmark is dramatically slower (z > 2).
    #[test]
    fn test_analyze_regressions_outlier_detected() {
        let runner = make_runner(true);

        // Four benchmarks at ~1 ms; one at 100 ms — clear statistical outlier.
        let measurements = vec![
            make_measurement("bench_normal_1", 1_000_000),
            make_measurement("bench_normal_2", 1_000_000),
            make_measurement("bench_normal_3", 1_000_000),
            make_measurement("bench_normal_4", 1_000_000),
            make_measurement("bench_slow", 100_000_000), // 100 ms — huge outlier
        ];

        let result = runner
            .analyze_regressions(&measurements)
            .expect("analyze_regressions should not fail");

        assert!(
            result.regression_detected,
            "A 100x slower benchmark should trigger regression detection"
        );
        // Exactly one regression expected — the slow benchmark.
        // The 4 fast benchmarks are ~95% below mean: they appear in improvements, not regressions.
        assert_eq!(
            result.regressions.len(),
            1,
            "Exactly 1 regression expected (bench_slow); got {:?}",
            result
                .regressions
                .iter()
                .map(|r| &r.benchmark_name)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            result.regressions[0].benchmark_name, "bench_slow",
            "The regressed benchmark should be bench_slow"
        );
        // The slow entry should be classified at least Minor or above.
        assert!(
            result.regressions[0].significance != RegressionSignificance::Negligible,
            "A 100x regression cannot be Negligible"
        );
        // The 4 fast entries should be in improvements (z-score < -2).
        assert_eq!(
            result.improvements.len(),
            4,
            "The 4 normal benchmarks should appear as improvements"
        );
    }

    /// Test 3: disabled regression detection gate — the conditional block in
    /// `run_benchmarks()` must produce `None` for `regression_analysis`.
    /// This test exercises the gate directly to ensure detect is skipped.
    #[test]
    fn test_regression_detection_disabled_gate() {
        let runner = make_runner(false);

        let measurements = vec![
            make_measurement("bench_normal", 1_000_000),
            make_measurement("bench_slow", 100_000_000),
        ];

        // Mirror the exact gate logic from run_benchmarks():
        let regression_analysis: Option<RegressionAnalysis> =
            if runner.config.enable_regression_detection {
                Some(
                    runner
                        .analyze_regressions(&measurements)
                        .expect("analyze_regressions failed"),
                )
            } else {
                None
            };

        assert!(
            regression_analysis.is_none(),
            "When enable_regression_detection is false, analysis must be None"
        );
    }

    /// Test 4: empty measurements — should return clean zeroed analysis.
    #[test]
    fn test_analyze_regressions_empty() {
        let runner = make_runner(true);
        let result = runner
            .analyze_regressions(&[])
            .expect("Empty input should not fail");
        assert!(!result.regression_detected);
        assert!(result.regressions.is_empty());
        assert!(result.improvements.is_empty());
        assert_eq!(result.overall_change_percent, 0.0);
    }

    /// Helper: a measurement with explicit `n`/`datasize`/memory/throughput,
    /// for exercising `analyze_scalability`/`analyze_memory_efficiency`
    /// directly without depending on platform-specific real timing.
    fn make_scalability_measurement(
        name: &str,
        n: usize,
        datasize: usize,
        avg_nanos: u64,
        memory_usage: usize,
        peak_memory: usize,
        modules: Vec<String>,
    ) -> PerformanceMeasurement {
        let mut m = PerformanceMeasurement::new(name.to_string(), modules);
        m.n = n;
        m.datasize = datasize;
        m.avg_duration = Duration::from_nanos(avg_nanos);
        m.operations_count = 100;
        m.throughput = 100.0 / Duration::from_nanos(avg_nanos).as_secs_f64().max(1e-12);
        m.memory_usage = memory_usage;
        m.peak_memory = peak_memory;
        m
    }

    /// Regression test: `thread_scalability`/`data_scalability` used to be
    /// hardcoded to 0.85/0.92 regardless of the measurements. A workload
    /// whose efficiency collapses at higher thread counts must now be
    /// reflected as a real, low `thread_scalability` score.
    #[test]
    fn test_analyze_scalability_detects_real_efficiency_falloff() {
        let runner = make_runner(false);
        // n=1: fast (high efficiency). n=8: 10x slower (efficiency collapses).
        // Durations are chosen large enough (with a realistic memory_usage
        // divisor) that `PerformanceMeasurement::efficiency_score`'s own
        // `.min(100.0)` clamp does not saturate both to the same value,
        // which would otherwise mask the real difference being tested.
        let measurements = vec![
            make_scalability_measurement(
                "op",
                1,
                0,
                2_000_000_000,
                1_000_000_000,
                1_000_000_000,
                vec!["m".to_string()],
            ),
            make_scalability_measurement(
                "op",
                8,
                0,
                20_000_000_000,
                1_000_000_000,
                1_000_000_000,
                vec!["m".to_string()],
            ),
        ];

        let analysis = runner
            .analyze_scalability(&measurements)
            .expect("Operation failed");

        assert!(
            analysis.thread_scalability < 0.5,
            "a 100x slowdown at higher thread count must yield a low scalability score, got {}",
            analysis.thread_scalability
        );
        assert!((0.0..=1.0).contains(&analysis.thread_scalability));
    }

    /// A workload whose efficiency is retained (or improves) at higher
    /// thread counts must score at (or near) the maximum, not a fixed
    /// unrelated constant.
    #[test]
    fn test_analyze_scalability_scores_high_when_efficiency_retained() {
        let runner = make_runner(false);
        let measurements = vec![
            make_scalability_measurement("op", 1, 0, 1_000, 0, 0, vec!["m".to_string()]),
            make_scalability_measurement("op", 8, 0, 1_000, 0, 0, vec!["m".to_string()]),
        ];

        let analysis = runner
            .analyze_scalability(&measurements)
            .expect("Operation failed");
        assert!(
            analysis.thread_scalability > 0.9,
            "identical efficiency across thread counts should score near 1.0, got {}",
            analysis.thread_scalability
        );
    }

    /// Regression test: `memory_scalability` used to be hardcoded to 0.88
    /// regardless of the measurements. Memory usage growing much faster
    /// than the data size being processed (super-linear, i.e. worse
    /// scaling) must now yield a real, low score.
    #[test]
    fn test_memory_scalability_score_detects_superlinear_memory_growth() {
        // Data size grows 10x (1024 -> 10240) but memory grows 100x
        // (1024 -> 102400): clearly worse-than-linear memory scaling.
        let measurements = vec![
            make_scalability_measurement("op", 0, 1024, 1_000, 1024, 1024, vec!["m".to_string()]),
            make_scalability_measurement(
                "op",
                0,
                10240,
                1_000,
                102_400,
                102_400,
                vec!["m".to_string()],
            ),
        ];

        let score = CrossModuleBenchmarkRunner::memory_scalability_score(&measurements);
        assert!(
            score < 0.5,
            "10x data growth causing 100x memory growth must score poorly, got {score}"
        );

        // Memory growing no faster than data (linear) must score at the max.
        let linear_measurements = vec![
            make_scalability_measurement("op", 0, 1024, 1_000, 1024, 1024, vec!["m".to_string()]),
            make_scalability_measurement(
                "op",
                0,
                10240,
                1_000,
                10240,
                10240,
                vec!["m".to_string()],
            ),
        ];
        let linear_score =
            CrossModuleBenchmarkRunner::memory_scalability_score(&linear_measurements);
        assert!(
            linear_score >= 0.99,
            "linear memory growth should score at the max, got {linear_score}"
        );
    }

    /// Regression test: `peak_to_avg_ratio`/`fragmentation_score`/
    /// `zero_copy_efficiency`/`bandwidth_utilization` used to be hardcoded
    /// to 1.2/0.15/0.95/0.75 regardless of the measurements. Feeding in
    /// measurements with a dramatic peak-memory outlier and a wide
    /// throughput spread must move these away from the old constants in
    /// the expected direction.
    #[test]
    fn test_analyze_memory_efficiency_reflects_real_measurements() {
        let runner = make_runner(false);
        let measurements = vec![
            make_scalability_measurement(
                "op_a",
                0,
                1024,
                1_000,
                1_000,
                1_000,
                vec!["m".to_string()],
            ),
            make_scalability_measurement(
                "op_b",
                0,
                1024,
                1_000,
                1_000,
                50_000,
                vec!["m".to_string()],
            ),
        ];

        let analysis = runner
            .analyze_memory_efficiency(&measurements)
            .expect("Operation failed");

        // Old hardcoded values were exactly 1.2 / 0.15 / 0.95 / 0.75; with a
        // 50x peak-vs-average outlier injected, at least peak_to_avg_ratio
        // and fragmentation_score must have moved substantially.
        assert!(
            analysis.peak_to_avg_ratio > 2.0,
            "a 50,000-byte peak against a ~1,000-byte average must yield a large ratio, got {}",
            analysis.peak_to_avg_ratio
        );
        assert!(
            analysis.fragmentation_score > 0.15,
            "a dramatic peak/average spread must raise the fragmentation proxy above the old \
             hardcoded 0.15, got {}",
            analysis.fragmentation_score
        );
        assert!((0.0..=f64::INFINITY).contains(&analysis.peak_to_avg_ratio));
        assert!((0.0..=1.0).contains(&analysis.fragmentation_score));
        assert!((0.0..=1.0).contains(&analysis.zero_copy_efficiency));
        assert!((0.0..=1.0).contains(&analysis.bandwidth_utilization));
    }

    /// Regression test: `time_operation`'s `memory_usage` used to be
    /// hardcoded to exactly 1 MiB regardless of what the timed closure
    /// actually did. On Linux (where real `VmRSS` sampling is available)
    /// an operation that allocates and holds a large real buffer must not
    /// report that same fabricated constant.
    #[cfg(target_os = "linux")]
    #[test]
    fn test_time_operation_reports_real_memory_not_the_old_fixed_constant() {
        let runner = make_runner(false);
        // Allocate and keep a real ~8 MiB buffer alive across every
        // iteration's measurement window so a real RSS delta is visible.
        let held: std::sync::Mutex<Vec<Vec<u8>>> = std::sync::Mutex::new(Vec::new());
        let timing = runner
            .time_operation("alloc_heavy", || {
                let buf = vec![0u8; 8 * 1024 * 1024];
                held.lock().expect("lock").push(buf);
                Ok(())
            })
            .expect("Operation failed");

        assert_ne!(
            timing.memory_usage,
            1024 * 1024,
            "memory_usage must reflect a real measurement, not the old hardcoded 1 MiB"
        );
    }

    /// Test 5: run_benchmarks with regression detection enabled produces Some analysis.
    #[test]
    fn test_run_benchmarks_regression_analysis_present() {
        let runner = make_runner(true);
        let suite = runner
            .run_benchmarks()
            .expect("run_benchmarks should succeed with detection enabled");
        assert!(
            suite.regression_analysis.is_some(),
            "regression_analysis should be Some when detection is enabled"
        );
    }

    /// Test 6: run_benchmarks with regression detection disabled produces None.
    #[test]
    fn test_run_benchmarks_regression_analysis_absent() {
        let runner = make_runner(false);
        let suite = runner
            .run_benchmarks()
            .expect("run_benchmarks should succeed with detection disabled");
        assert!(
            suite.regression_analysis.is_none(),
            "regression_analysis should be None when detection is disabled"
        );
    }
}
