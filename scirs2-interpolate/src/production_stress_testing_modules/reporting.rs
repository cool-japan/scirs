//! Test reporting and analysis for production stress testing
//!
//! This module handles the generation of comprehensive stress test reports,
//! analysis of results, and production readiness assessments.

use super::types::*;
use crate::traits::InterpolationFloat;
use std::fmt;
use std::time::Duration;

impl<T: InterpolationFloat + std::panic::RefUnwindSafe> ProductionStressTester<T> {
    /// Generate comprehensive stress test report
    pub fn generate_stress_test_report(&self) -> StressTestReport {
        let summary = self.calculate_summary_statistics();
        let critical_issues = self.extract_critical_issues();
        let performance_analysis = self.analyze_performance();
        let production_readiness = self.assess_production_readiness(&summary, &critical_issues);
        let production_recommendations = self.generate_production_recommendations(
            &summary,
            &critical_issues,
            &performance_analysis,
        );

        StressTestReport {
            results: self.results.clone(),
            config: self.config.clone(),
            production_readiness,
            summary,
            critical_issues,
            performance_analysis,
            production_recommendations,
        }
    }

    /// Calculate summary statistics from test results
    fn calculate_summary_statistics(&self) -> StressTestSummary {
        let total_tests = self.results.len();
        let tests_passed = self
            .results
            .iter()
            .filter(|r| r.status == TestStatus::Passed)
            .count();
        let tests_failed = self
            .results
            .iter()
            .filter(|r| matches!(r.status, TestStatus::Failed | TestStatus::Error))
            .count();
        let tests_with_warnings = self
            .results
            .iter()
            .filter(|r| r.status == TestStatus::PassedWithWarnings)
            .count();

        let total_execution_time: Duration = self.results.iter().map(|r| r.execution_time).sum();
        let average_execution_time = if total_tests > 0 {
            total_execution_time / (total_tests as u32)
        } else {
            Duration::from_millis(0)
        };

        // Calculate memory efficiency score (0-100)
        let memory_efficiency_score = self.calculate_memory_efficiency_score();

        // Calculate overall performance degradation
        let overall_degradation_factor = self.calculate_overall_degradation_factor();

        StressTestSummary {
            total_tests,
            tests_passed,
            tests_failed,
            tests_with_warnings,
            average_execution_time,
            total_execution_time,
            memory_efficiency_score,
            overall_degradation_factor,
        }
    }

    /// Calculate memory efficiency score
    fn calculate_memory_efficiency_score(&self) -> f64 {
        let memory_usages: Vec<f64> = self
            .results
            .iter()
            .map(|r| r.memory_usage.peak_usage as f64)
            .filter(|&usage| usage > 0.0)
            .collect();

        if memory_usages.is_empty() {
            return 80.0; // Default score
        }

        let average_usage = memory_usages.iter().sum::<f64>() / memory_usages.len() as f64;
        let expected_usage = self.estimate_expected_memory_usage();

        if expected_usage > 0.0 {
            let efficiency = expected_usage / average_usage;
            (efficiency * 100.0).min(100.0).max(0.0)
        } else {
            80.0
        }
    }

    /// Estimate expected memory usage
    fn estimate_expected_memory_usage(&self) -> f64 {
        // Simplified estimation based on typical test sizes
        let typical_test_size = 10_000;
        let bytes_per_element = std::mem::size_of::<T>();
        (typical_test_size * bytes_per_element * 3) as f64 // 3x for overhead
    }

    /// Calculate overall performance degradation factor
    fn calculate_overall_degradation_factor(&self) -> f64 {
        let degradation_factors: Vec<f64> = self
            .results
            .iter()
            .filter_map(|r| r.performance.degradation_factor)
            .collect();

        if degradation_factors.is_empty() {
            1.0
        } else {
            degradation_factors.iter().sum::<f64>() / degradation_factors.len() as f64
        }
    }

    /// Extract critical issues from test results
    fn extract_critical_issues(&self) -> Vec<StressTestIssue> {
        self.results
            .iter()
            .flat_map(|r| &r.issues)
            .filter(|issue| {
                matches!(
                    issue.severity,
                    IssueSeverity::Critical | IssueSeverity::High
                )
            })
            .cloned()
            .collect()
    }

    /// Analyze performance across all tests
    fn analyze_performance(&self) -> PerformanceAnalysis {
        let mut bottlenecks = Vec::new();
        let mut improvements = Vec::new();

        // Identify performance bottlenecks
        for result in &self.results {
            if let Some(degradation) = result.performance.degradation_factor {
                if degradation > 5.0 {
                    bottlenecks.push(format!(
                        "{}: {:.1}x degradation",
                        result.test_name, degradation
                    ));
                }
            }

            if result.execution_time > Duration::from_secs(10) {
                bottlenecks.push(format!(
                    "{}: Long execution time ({:.2}s)",
                    result.test_name,
                    result.execution_time.as_secs_f64()
                ));
            }
        }

        // Generate improvement suggestions
        if bottlenecks.is_empty() {
            improvements.push("Performance is within acceptable limits".to_string());
        } else {
            improvements.push("Consider optimizing algorithms for large datasets".to_string());
            improvements.push("Implement progressive loading for very large data".to_string());
            improvements.push("Add performance monitoring and alerting".to_string());
        }

        // Estimate production performance
        let production_performance_estimate = self.estimate_production_performance();

        // Assess scalability
        let scalability_assessment = self.assess_scalability();

        PerformanceAnalysis {
            bottlenecks,
            improvements,
            production_performance_estimate,
            scalability_assessment,
        }
    }

    /// Estimate production performance
    fn estimate_production_performance(&self) -> Option<Duration> {
        let typical_results: Vec<&StressTestResult> = self
            .results
            .iter()
            .filter(|r| r.status == TestStatus::Passed && !r.test_name.contains("extreme"))
            .collect();

        if typical_results.is_empty() {
            return None;
        }

        let average_time: Duration = typical_results
            .iter()
            .map(|r| r.execution_time)
            .sum::<Duration>()
            / (typical_results.len() as u32);

        // Add 20% buffer for production overhead
        Some(Duration::from_nanos(
            (average_time.as_nanos() as f64 * 1.2) as u64,
        ))
    }

    /// Assess scalability
    fn assess_scalability(&self) -> ScalabilityAssessment {
        let large_data_results: Vec<&StressTestResult> = self
            .results
            .iter()
            .filter(|r| r.test_name.contains("extreme") || r.test_name.contains("stress"))
            .collect();

        let can_handle_10x_load = large_data_results.iter().all(|r| {
            matches!(
                r.status,
                TestStatus::Passed | TestStatus::PassedWithWarnings
            )
        });

        let can_handle_100x_load = large_data_results
            .iter()
            .filter(|r| r.test_name.contains("extreme"))
            .all(|r| r.status == TestStatus::Passed);

        let max_scale_factor = if can_handle_100x_load {
            100.0
        } else if can_handle_10x_load {
            10.0
        } else {
            1.0
        };

        let mut limiting_factors = Vec::new();
        if !can_handle_10x_load {
            limiting_factors.push("Memory constraints with large datasets".to_string());
        }
        if large_data_results
            .iter()
            .any(|r| r.performance.degradation_factor.unwrap_or(1.0) > 10.0)
        {
            limiting_factors.push("Performance degradation with scale".to_string());
        }

        ScalabilityAssessment {
            can_handle_10x_load,
            can_handle_100x_load,
            max_scale_factor,
            limiting_factors,
        }
    }

    /// Assess overall production readiness
    fn assess_production_readiness(
        &self,
        summary: &StressTestSummary,
        critical_issues: &[StressTestIssue],
    ) -> ProductionReadiness {
        // Check for blocking issues
        if critical_issues
            .iter()
            .any(|i| i.production_impact == ProductionImpact::Blocking)
        {
            return ProductionReadiness::NotReady;
        }

        // Check for critical severity issues
        if critical_issues
            .iter()
            .any(|i| i.severity == IssueSeverity::Critical)
        {
            return ProductionReadiness::NeedsBugFixes;
        }

        // Check test failure rate
        let failure_rate = summary.tests_failed as f64 / summary.total_tests as f64;
        if failure_rate > 0.1 {
            return ProductionReadiness::NeedsBugFixes;
        }

        // Check performance degradation
        if summary.overall_degradation_factor > 10.0 {
            return ProductionReadiness::NeedsPerformanceTuning;
        }

        // Check for high severity issues
        if critical_issues
            .iter()
            .any(|i| i.severity == IssueSeverity::High)
        {
            return ProductionReadiness::ReadyWithMonitoring;
        }

        ProductionReadiness::Ready
    }

    /// Generate production deployment recommendations
    fn generate_production_recommendations(
        &self,
        summary: &StressTestSummary,
        critical_issues: &[StressTestIssue],
        performance_analysis: &PerformanceAnalysis,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        // General recommendations based on test results
        if summary.tests_failed > 0 {
            recommendations
                .push("Address all test failures before production deployment".to_string());
        }

        if summary.tests_with_warnings > 0 {
            recommendations.push("Review and address warnings to improve robustness".to_string());
        }

        // Critical issue recommendations
        for issue in critical_issues {
            if let Some(fix) = &issue.suggested_fix {
                recommendations.push(format!("CRITICAL: {}", fix));
            }
        }

        // Performance recommendations
        if summary.overall_degradation_factor > 5.0 {
            recommendations.push("Optimize performance for production workloads".to_string());
        }

        if summary.memory_efficiency_score < 70.0 {
            recommendations.push("Improve memory efficiency and monitor usage".to_string());
        }

        // Scalability recommendations
        if !performance_analysis
            .scalability_assessment
            .can_handle_10x_load
        {
            recommendations.push("Test and optimize for expected production scale".to_string());
        }

        // Monitoring recommendations
        recommendations.push("Implement comprehensive monitoring and alerting".to_string());
        recommendations.push("Set up performance benchmarks and regression testing".to_string());
        recommendations.push("Document known limitations and edge cases".to_string());

        if recommendations.is_empty() {
            recommendations.push("System appears ready for production deployment".to_string());
        }

        recommendations
    }
}

impl fmt::Display for StressTestReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Production Stress Test Report ===")?;
        writeln!(f)?;

        // Overall assessment
        writeln!(f, "Production Readiness: {:?}", self.production_readiness)?;
        writeln!(f)?;

        // Summary statistics
        writeln!(f, "Test Summary:")?;
        writeln!(f, "  Total Tests: {}", self.summary.total_tests)?;
        writeln!(f, "  Passed: {}", self.summary.tests_passed)?;
        writeln!(f, "  Failed: {}", self.summary.tests_failed)?;
        writeln!(f, "  Warnings: {}", self.summary.tests_with_warnings)?;
        writeln!(
            f,
            "  Success Rate: {:.1}%",
            (self.summary.tests_passed as f64 / self.summary.total_tests as f64) * 100.0
        )?;
        writeln!(f)?;

        // Performance metrics
        writeln!(f, "Performance Metrics:")?;
        writeln!(
            f,
            "  Average Execution Time: {:.3}s",
            self.summary.average_execution_time.as_secs_f64()
        )?;
        writeln!(
            f,
            "  Total Execution Time: {:.3}s",
            self.summary.total_execution_time.as_secs_f64()
        )?;
        writeln!(
            f,
            "  Memory Efficiency Score: {:.1}/100",
            self.summary.memory_efficiency_score
        )?;
        writeln!(
            f,
            "  Overall Degradation Factor: {:.1}x",
            self.summary.overall_degradation_factor
        )?;
        writeln!(f)?;

        // Critical issues
        if !self.critical_issues.is_empty() {
            writeln!(f, "Critical Issues ({}):", self.critical_issues.len())?;
            for issue in &self.critical_issues {
                writeln!(f, "  • {:?}: {}", issue.severity, issue.description)?;
                if let Some(fix) = &issue.suggested_fix {
                    writeln!(f, "    Suggested fix: {}", fix)?;
                }
            }
            writeln!(f)?;
        }

        // Performance analysis
        writeln!(f, "Performance Analysis:")?;
        if self.performance_analysis.bottlenecks.is_empty() {
            writeln!(f, "  No significant performance bottlenecks detected")?;
        } else {
            writeln!(f, "  Bottlenecks:")?;
            for bottleneck in &self.performance_analysis.bottlenecks {
                writeln!(f, "    • {}", bottleneck)?;
            }
        }

        if let Some(est_time) = self.performance_analysis.production_performance_estimate {
            writeln!(
                f,
                "  Estimated Production Performance: {:.3}s",
                est_time.as_secs_f64()
            )?;
        }
        writeln!(f)?;

        // Scalability assessment
        let scalability = &self.performance_analysis.scalability_assessment;
        writeln!(f, "Scalability Assessment:")?;
        writeln!(
            f,
            "  Can handle 10x load: {}",
            scalability.can_handle_10x_load
        )?;
        writeln!(
            f,
            "  Can handle 100x load: {}",
            scalability.can_handle_100x_load
        )?;
        writeln!(
            f,
            "  Max scale factor: {:.1}x",
            scalability.max_scale_factor
        )?;
        if !scalability.limiting_factors.is_empty() {
            writeln!(f, "  Limiting factors:")?;
            for factor in &scalability.limiting_factors {
                writeln!(f, "    • {}", factor)?;
            }
        }
        writeln!(f)?;

        // Production recommendations
        writeln!(f, "Production Recommendations:")?;
        for (i, recommendation) in self.production_recommendations.iter().enumerate() {
            writeln!(f, "  {}. {}", i + 1, recommendation)?;
        }
        writeln!(f)?;

        // Test configuration
        writeln!(f, "Test Configuration:")?;
        writeln!(f, "  Max Data Size: {}", self.config.max_data_size)?;
        writeln!(f, "  Stress Iterations: {}", self.config.stress_iterations)?;
        writeln!(f, "  Test Timeout: {}s", self.config.test_timeout)?;
        writeln!(
            f,
            "  Memory Limit: {:?}",
            self.config.memory_limit.map(|l| format!("{} bytes", l))
        )?;
        writeln!(f, "  Extreme Cases: {}", self.config.test_extreme_cases)?;
        writeln!(
            f,
            "  Max Performance Degradation: {:.1}x",
            self.config.max_performance_degradation
        )?;

        Ok(())
    }
}

impl fmt::Display for ProductionReadiness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProductionReadiness::Ready => write!(f, "Ready for Production"),
            ProductionReadiness::ReadyWithMonitoring => write!(f, "Ready with Enhanced Monitoring"),
            ProductionReadiness::NeedsPerformanceTuning => write!(f, "Needs Performance Tuning"),
            ProductionReadiness::NeedsBugFixes => write!(f, "Needs Bug Fixes"),
            ProductionReadiness::NotReady => write!(f, "Not Ready for Production"),
        }
    }
}

impl fmt::Display for StressTestCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StressTestCategory::ExtremeDataSize => write!(f, "Extreme Data Size"),
            StressTestCategory::PathologicalData => write!(f, "Pathological Data"),
            StressTestCategory::NumericalEdgeCases => write!(f, "Numerical Edge Cases"),
            StressTestCategory::MemoryPressure => write!(f, "Memory Pressure"),
            StressTestCategory::ErrorHandling => write!(f, "Error Handling"),
            StressTestCategory::PerformanceStress => write!(f, "Performance Stress"),
            StressTestCategory::ConcurrentAccess => write!(f, "Concurrent Access"),
            StressTestCategory::NumericalStability => write!(f, "Numerical Stability"),
            StressTestCategory::ResourceExhaustion => write!(f, "Resource Exhaustion"),
        }
    }
}

impl fmt::Display for TestStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TestStatus::Passed => write!(f, "PASSED"),
            TestStatus::PassedWithWarnings => write!(f, "PASSED (with warnings)"),
            TestStatus::Failed => write!(f, "FAILED"),
            TestStatus::TimedOut => write!(f, "TIMED OUT"),
            TestStatus::Skipped => write!(f, "SKIPPED"),
            TestStatus::Error => write!(f, "ERROR"),
        }
    }
}

impl fmt::Display for IssueSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IssueSeverity::Critical => write!(f, "CRITICAL"),
            IssueSeverity::High => write!(f, "HIGH"),
            IssueSeverity::Medium => write!(f, "MEDIUM"),
            IssueSeverity::Low => write!(f, "LOW"),
        }
    }
}

impl fmt::Display for ProductionImpact {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProductionImpact::Blocking => write!(f, "BLOCKING"),
            ProductionImpact::Major => write!(f, "MAJOR"),
            ProductionImpact::Minor => write!(f, "MINOR"),
            ProductionImpact::None => write!(f, "NONE"),
        }
    }
}
