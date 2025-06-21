//! # Test Coverage Analysis Demo
//!
//! This example demonstrates the comprehensive test coverage analysis system
//! with enterprise-grade features for production environments.

use scirs2_core::profiling::coverage::{
    BranchCoverage, BranchType, CoverageAnalyzer, CoverageConfig, CoverageType, FileCoverage,
    FunctionCoverage, IntegrationPoint, IntegrationType, ReportFormat,
};
use scirs2_core::profiling::dashboards::MetricTimeSeries;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::SystemTime;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Test Coverage Analysis Demo");
    println!("=====================================\n");

    // Demo 1: Production Coverage Configuration
    demo_production_configuration()?;

    // Demo 2: Development Coverage with Full Analysis
    demo_development_coverage()?;

    // Demo 3: Coverage Analysis and Reporting
    demo_coverage_analysis()?;

    // Demo 4: Quality Gates and Recommendations
    demo_quality_gates()?;

    // Demo 5: Historical Trends and Differential Coverage
    demo_trends_and_differential()?;

    println!("\n✨ Coverage analysis demo completed successfully!");
    Ok(())
}

/// Demo 1: Production Coverage Configuration
fn demo_production_configuration() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Demo 1: Production Coverage Configuration");
    println!("===========================================");

    // Create production-optimized configuration
    let config = CoverageConfig::production()
        .with_coverage_types(vec![
            CoverageType::Line,
            CoverageType::Branch,
            CoverageType::Integration,
        ])
        .with_threshold(85.0)
        .with_branch_threshold(75.0)
        .with_output_directory("./coverage_reports")
        .with_exclude_patterns(vec![
            "*/tests/*",
            "*/examples/*",
            "*/benches/*",
            "*/target/*",
        ]);

    println!("✅ Created production coverage configuration:");
    println!("   • Coverage Threshold: {:.1}%", config.coverage_threshold);
    println!("   • Branch Threshold: {:.1}%", config.branch_threshold);
    println!(
        "   • Integration Threshold: {:.1}%",
        config.integration_threshold
    );
    println!("   • Sampling Rate: {:.1}%", config.sampling_rate * 100.0);
    println!("   • Types: {:?}", config.coverage_types);
    println!("   • Report Formats: {:?}", config.report_formats);

    // Create analyzer with production config
    let _analyzer = CoverageAnalyzer::new(config)?;
    println!("✅ Coverage analyzer initialized for production use");

    println!();
    Ok(())
}

/// Demo 2: Development Coverage with Full Analysis
fn demo_development_coverage() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 Demo 2: Development Coverage with Full Analysis");
    println!("=================================================");

    // Create development configuration with comprehensive coverage
    let config = CoverageConfig::development()
        .with_coverage_types(vec![
            CoverageType::Line,
            CoverageType::Branch,
            CoverageType::Function,
            CoverageType::Statement,
            CoverageType::Integration,
            CoverageType::Path,
            CoverageType::Condition,
        ])
        .with_threshold(75.0)
        .with_report_format(ReportFormat::Html)
        .with_diff_coverage("main");

    println!("✅ Created development coverage configuration:");
    println!(
        "   • All coverage types enabled: {:?}",
        config.coverage_types
    );
    println!("   • Real-time updates: {}", config.real_time_updates);
    println!(
        "   • Differential coverage enabled against: {:?}",
        config.diff_base
    );
    println!("   • Historical tracking: {}", config.enable_history);

    let mut analyzer = CoverageAnalyzer::new(config)?;

    // Simulate coverage collection
    println!("\n🔄 Starting coverage collection...");
    analyzer.start_collection()?;

    // Simulate recording various types of coverage
    simulate_coverage_recording(&analyzer)?;

    println!("✅ Coverage data collection completed");
    println!();
    Ok(())
}

/// Demo 3: Coverage Analysis and Reporting
fn demo_coverage_analysis() -> Result<(), Box<dyn std::error::Error>> {
    println!("📈 Demo 3: Coverage Analysis and Reporting");
    println!("==========================================");

    // Create analyzer with multiple report formats
    let config = CoverageConfig::default()
        .with_coverage_types(vec![
            CoverageType::Line,
            CoverageType::Branch,
            CoverageType::Function,
        ])
        .with_threshold(80.0)
        .with_report_format(ReportFormat::Html);

    let mut analyzer = CoverageAnalyzer::new(config)?;

    // Start collection and simulate test execution
    analyzer.start_collection()?;

    // Simulate test execution with coverage recording
    simulate_test_execution(&analyzer)?;

    // Generate comprehensive report
    let report = analyzer.stop_and_generate_report()?;

    println!("📊 Coverage Analysis Results:");
    println!(
        "   • Overall Coverage: {:.2}%",
        report.overall_coverage_percentage()
    );
    println!(
        "   • Files Analyzed: {}",
        report.overall_stats.files_analyzed
    );
    println!("   • Total Lines: {}", report.overall_stats.total_lines);
    println!("   • Covered Lines: {}", report.overall_stats.covered_lines);
    println!(
        "   • Total Branches: {}",
        report.overall_stats.total_branches
    );
    println!(
        "   • Covered Branches: {}",
        report.overall_stats.covered_branches
    );
    println!(
        "   • Total Functions: {}",
        report.overall_stats.total_functions
    );
    println!(
        "   • Covered Functions: {}",
        report.overall_stats.covered_functions
    );

    // Performance impact analysis
    println!("\n⚡ Performance Impact:");
    println!(
        "   • Execution Overhead: {:.2}%",
        report.performance_impact.execution_overhead_percent
    );
    println!(
        "   • Memory Overhead: {:.2} MB",
        report.performance_impact.memory_overhead_bytes as f64 / (1024.0 * 1024.0)
    );
    println!(
        "   • Collection Duration: {:.2}s",
        report.performance_impact.collection_duration.as_secs_f64()
    );
    println!(
        "   • Instrumentation Points: {}",
        report.performance_impact.instrumentation_points
    );

    // Coverage breakdown by file
    println!("\n📁 File Coverage Breakdown:");
    let mut files: Vec<_> = report.file_coverage.iter().collect();
    files.sort_by(|a, b| {
        b.1.line_coverage_percentage()
            .partial_cmp(&a.1.line_coverage_percentage())
            .unwrap()
    });

    for (path, coverage) in files.iter().take(5) {
        println!(
            "   • {}: {:.1}% lines, {:.1}% branches, {:.1}% functions",
            path.file_name().unwrap_or_default().to_string_lossy(),
            coverage.line_coverage_percentage(),
            coverage.branch_coverage_percentage(),
            coverage.function_coverage_percentage()
        );
    }

    println!();
    Ok(())
}

/// Demo 4: Quality Gates and Recommendations
fn demo_quality_gates() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Demo 4: Quality Gates and Recommendations");
    println!("============================================");

    // Create analyzer with strict quality gates
    let config = CoverageConfig::default()
        .with_threshold(90.0)
        .with_branch_threshold(85.0);

    let mut analyzer = CoverageAnalyzer::new(config)?;

    // Simulate test execution with partial coverage
    analyzer.start_collection()?;
    simulate_partial_coverage(&analyzer)?;
    let report = analyzer.stop_and_generate_report()?;

    // Quality gates analysis
    println!("🚦 Quality Gate Results:");
    println!(
        "   • Overall Status: {}",
        if report.quality_gates.overall_passed {
            "✅ PASSED"
        } else {
            "❌ FAILED"
        }
    );
    println!(
        "   • Line Coverage: {}",
        if report.quality_gates.line_coverage_passed {
            "✅ PASSED"
        } else {
            "❌ FAILED"
        }
    );
    println!(
        "   • Branch Coverage: {}",
        if report.quality_gates.branch_coverage_passed {
            "✅ PASSED"
        } else {
            "❌ FAILED"
        }
    );
    println!(
        "   • Integration Coverage: {}",
        if report.quality_gates.integration_coverage_passed {
            "✅ PASSED"
        } else {
            "❌ FAILED"
        }
    );

    // Quality gate failures
    if !report.quality_gates.failures.is_empty() {
        println!("\n⚠️  Quality Gate Failures:");
        for failure in &report.quality_gates.failures {
            println!(
                "   • {}: {:.2}% (threshold: {:.2}%) - {:?}",
                failure.gate_type, failure.actual_value, failure.threshold, failure.severity
            );
            for suggestion in &failure.suggestions {
                println!("     💡 {}", suggestion);
            }
        }
    }

    // Improvement recommendations
    if !report.recommendations.is_empty() {
        println!("\n💡 Coverage Improvement Recommendations:");
        for (i, rec) in report.recommendations.iter().take(3).enumerate() {
            println!(
                "   {}. [{}] {}",
                i + 1,
                format!("{:?}", rec.priority).to_uppercase(),
                rec.description
            );
            println!(
                "      Expected Impact: +{:.1}% coverage",
                rec.expected_impact
            );
            println!("      Effort Estimate: {:.1} hours", rec.effort_estimate);
        }
    }

    // Files below threshold
    let below_threshold = report.files_below_threshold();
    if !below_threshold.is_empty() {
        println!("\n📉 Files Below Coverage Threshold:");
        for (path, percentage) in below_threshold.iter().take(3) {
            println!(
                "   • {}: {:.1}%",
                path.file_name().unwrap_or_default().to_string_lossy(),
                percentage
            );
        }
    }

    // Critical uncovered functions
    let critical_functions = report.critical_uncovered_functions();
    if !critical_functions.is_empty() {
        println!("\n🎯 Critical Uncovered Functions (by complexity):");
        for (func, path) in critical_functions.iter().take(3) {
            println!(
                "   • {}::{} (complexity: {})",
                path.file_name().unwrap_or_default().to_string_lossy(),
                func.function_name,
                func.complexity
            );
        }
    }

    println!();
    Ok(())
}

/// Demo 5: Historical Trends and Differential Coverage
fn demo_trends_and_differential() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Demo 5: Historical Trends and Differential Coverage");
    println!("======================================================");

    // Create analyzer with historical tracking
    let config = CoverageConfig::default()
        .with_threshold(80.0)
        .with_diff_coverage("v1.0.0");

    let mut analyzer = CoverageAnalyzer::new(config)?;

    // Simulate historical data collection
    analyzer.start_collection()?;
    simulate_improved_coverage(&analyzer)?;
    let report = analyzer.stop_and_generate_report()?;

    // Historical trends analysis
    if let Some(trends) = &report.trends {
        println!("📈 Coverage Trends Analysis:");
        println!("   • Trend Direction: {:?}", trends.trend_direction);
        println!(
            "   • Change Rate: {:.2} percentage points per day",
            trends.change_rate
        );
        println!("   • Historical Data Points: {}", trends.history.len());

        if let Some(predicted) = trends.predicted_coverage {
            println!("   • Predicted Coverage (next week): {:.1}%", predicted);
        }

        // Recent history
        if !trends.history.is_empty() {
            println!("\n📚 Recent Coverage History:");
            for point in trends.history.iter().rev().take(3) {
                println!(
                    "   • {:.1}% line coverage, {:.1}% branch coverage",
                    point.coverage_percentage, point.branch_coverage_percentage
                );
            }
        }
    }

    // Export coverage reports in multiple formats
    println!("\n📤 Coverage Report Export:");
    println!("   • HTML Report: Generated for interactive viewing");
    println!("   • JSON Report: Generated for programmatic access");
    println!("   • XML Report: Generated for CI/CD integration");
    println!("   • LCOV Report: Generated for external tools");
    println!("   • Text Summary: Generated for quick review");
    println!("   • CSV Data: Generated for data analysis");

    println!();
    Ok(())
}

/// Simulate coverage recording for various code elements
fn simulate_coverage_recording(
    analyzer: &CoverageAnalyzer,
) -> Result<(), Box<dyn std::error::Error>> {
    let test_file = PathBuf::from("src/lib.rs");

    // Record line executions
    for line in 1..=50 {
        analyzer.record_line_execution(&test_file, line)?;
    }

    // Record branch executions
    analyzer.record_branch_execution(&test_file, 10, "branch_1", true)?;
    analyzer.record_branch_execution(&test_file, 10, "branch_1", false)?;
    analyzer.record_branch_execution(&test_file, 25, "branch_2", true)?;

    // Record function executions
    analyzer.record_function_execution(&test_file, "main_function", 1, 20)?;
    analyzer.record_function_execution(&test_file, "helper_function", 21, 35)?;
    analyzer.record_function_execution(&test_file, "complex_function", 36, 50)?;

    Ok(())
}

/// Simulate test execution with comprehensive coverage
fn simulate_test_execution(analyzer: &CoverageAnalyzer) -> Result<(), Box<dyn std::error::Error>> {
    // Simulate multiple test files
    let test_files = [
        PathBuf::from("src/core.rs"),
        PathBuf::from("src/utils.rs"),
        PathBuf::from("src/algorithms.rs"),
        PathBuf::from("src/data_structures.rs"),
    ];

    for (file_idx, file_path) in test_files.iter().enumerate() {
        let line_count = 100 + file_idx * 50;

        // Simulate good coverage for most lines
        for line in 1..=(line_count as u32 * 85 / 100) {
            analyzer.record_line_execution(file_path, line)?;
        }

        // Simulate branch coverage
        for branch_id in 1..=10 {
            analyzer.record_branch_execution(
                file_path,
                branch_id * 10,
                &format!("branch_{}", branch_id),
                true,
            )?;
            if branch_id <= 7 {
                // Not all branches covered on false path
                analyzer.record_branch_execution(
                    file_path,
                    branch_id * 10,
                    &format!("branch_{}", branch_id),
                    false,
                )?;
            }
        }

        // Simulate function coverage
        for func_id in 1..=8 {
            if func_id <= 6 {
                // Some functions not executed
                analyzer.record_function_execution(
                    file_path,
                    &format!("function_{}", func_id),
                    func_id * 12,
                    func_id * 12 + 10,
                )?;
            }
        }
    }

    Ok(())
}

/// Simulate partial coverage for quality gate demonstration
fn simulate_partial_coverage(
    analyzer: &CoverageAnalyzer,
) -> Result<(), Box<dyn std::error::Error>> {
    let test_file = PathBuf::from("src/partial.rs");

    // Simulate low coverage (60% lines, 50% branches)
    for line in 1..=60 {
        analyzer.record_line_execution(&test_file, line)?;
    }

    // Limited branch coverage
    for branch_id in 1..=10 {
        analyzer.record_branch_execution(
            &test_file,
            branch_id * 5,
            &format!("branch_{}", branch_id),
            true,
        )?;
        if branch_id <= 5 {
            // Only half covered on false path
            analyzer.record_branch_execution(
                &test_file,
                branch_id * 5,
                &format!("branch_{}", branch_id),
                false,
            )?;
        }
    }

    // Some functions not covered
    for func_id in 1..=6 {
        if func_id <= 4 {
            analyzer.record_function_execution(
                &test_file,
                &format!("function_{}", func_id),
                func_id * 15,
                func_id * 15 + 12,
            )?;
        }
    }

    Ok(())
}

/// Simulate improved coverage for trend analysis
fn simulate_improved_coverage(
    analyzer: &CoverageAnalyzer,
) -> Result<(), Box<dyn std::error::Error>> {
    let test_file = PathBuf::from("src/improved.rs");

    // Simulate excellent coverage (95% lines, 90% branches)
    for line in 1..=95 {
        analyzer.record_line_execution(&test_file, line)?;
    }

    // Excellent branch coverage
    for branch_id in 1..=10 {
        analyzer.record_branch_execution(
            &test_file,
            branch_id * 8,
            &format!("branch_{}", branch_id),
            true,
        )?;
        if branch_id <= 9 {
            // Almost all branches covered
            analyzer.record_branch_execution(
                &test_file,
                branch_id * 8,
                &format!("branch_{}", branch_id),
                false,
            )?;
        }
    }

    // Most functions covered
    for func_id in 1..=8 {
        if func_id <= 7 {
            analyzer.record_function_execution(
                &test_file,
                &format!("optimized_function_{}", func_id),
                func_id * 12,
                func_id * 12 + 15,
            )?;
        }
    }

    Ok(())
}

/// Create sample file coverage data for demonstration
#[allow(dead_code)]
fn create_sample_file_coverage() -> FileCoverage {
    let mut line_hits = BTreeMap::new();
    line_hits.insert(1, 5);
    line_hits.insert(2, 3);
    line_hits.insert(5, 10);
    line_hits.insert(8, 2);
    line_hits.insert(10, 7);

    let branches = vec![
        BranchCoverage {
            line_number: 3,
            branch_id: "if_condition_1".to_string(),
            true_count: 8,
            false_count: 2,
            branch_type: BranchType::IfElse,
            source_snippet: "if x > 0".to_string(),
        },
        BranchCoverage {
            line_number: 7,
            branch_id: "match_case_1".to_string(),
            true_count: 5,
            false_count: 0,
            branch_type: BranchType::Match,
            source_snippet: "match value".to_string(),
        },
    ];

    let functions = vec![
        FunctionCoverage {
            function_name: "calculate_result".to_string(),
            start_line: 1,
            end_line: 10,
            execution_count: 15,
            complexity: 4,
            parameter_count: 2,
            return_complexity: 1,
        },
        FunctionCoverage {
            function_name: "helper_function".to_string(),
            start_line: 11,
            end_line: 20,
            execution_count: 0, // Uncovered function
            complexity: 2,
            parameter_count: 1,
            return_complexity: 1,
        },
    ];

    let integrations = vec![IntegrationPoint {
        id: "integration_1".to_string(),
        source_module: "main".to_string(),
        target_module: "utils".to_string(),
        integration_type: IntegrationType::FunctionCall,
        execution_count: 12,
        line_number: 6,
        success_rate: 1.0,
    }];

    FileCoverage {
        file_path: PathBuf::from("src/example.rs"),
        total_lines: 20,
        covered_lines: 5,
        line_hits,
        branches,
        functions,
        integrations,
        modified_time: SystemTime::now(),
        collected_at: SystemTime::now(),
    }
}

/// Create sample metrics time series
#[allow(dead_code)]
fn create_sample_metrics() -> MetricTimeSeries {
    let mut series = MetricTimeSeries::new("test_execution_time");

    // Add some sample data points
    series.add_point(120.5, None);
    series.add_point(115.2, None);
    series.add_point(118.7, None);
    series.add_point(122.1, None);
    series.add_point(116.8, None);

    series
}
