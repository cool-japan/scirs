//! AI-Driven Benchmark Suite Showcase
//!
//! This example demonstrates the enhanced benchmark suite with AI-driven analysis,
//! cross-platform validation, automated regression detection, and intelligent
//! optimization recommendations for statistical computing operations.

use scirs2_stats::{
    benchmark_suite_enhanced::{
        create_enhanced_benchmark_suite, EnhancedBenchmarkConfig, ImplementationEffort,
        MLModelConfig, PlatformTarget, RecommendationCategory, RecommendationPriority,
    },
    BenchmarkConfig,
};
use std::time::Duration;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🤖 AI-Driven Statistical Computing Benchmark Suite");
    println!("==================================================\n");

    // Create enhanced benchmark configuration
    let enhanced_config = EnhancedBenchmarkConfig {
        base_config: BenchmarkConfig {
            datasizes: vec![1000, 10000, 100000, 1000000],
            iterations: 50,
            warmup_iterations: 5,
            track_memory: true,
            comparebaseline: true,
            test_simd: true,
            test_parallel: true,
            confidence_level: 0.95,
            regression_threshold: 5.0,
        },
        enable_ai_analysis: true,
        enable_cross_platform: true,
        enable_regression_detection: true,
        enable_optimization_recommendations: true,
        ml_model_config: MLModelConfig {
            model_type: scirs2_stats::benchmark_suite_enhanced::MLModelType::RandomForest,
            feature_selection: scirs2_stats::benchmark_suite_enhanced::FeatureSelectionStrategy::AutomaticImportance,
            training_retention_days: 30,
            retraining_frequency: scirs2_stats::benchmark_suite_enhanced::RetrainingFrequency::Weekly,
            confidence_threshold: 0.8,
        },
        platform_targets: vec![
            PlatformTarget::x86_64_linux(),
            PlatformTarget::x86_64_windows(),
            PlatformTarget::aarch64_macos(),
        ],
        regression_sensitivity: 0.03, // 3% sensitivity
        baselinedatabase_path: Some("./benchmarkbaselines.db".to_string()),
    };

    // Create enhanced benchmark suite
    let mut benchmark_suite = create_enhanced_benchmark_suite();

    println!("📊 Running comprehensive statistical computing benchmarks...");
    println!("🔍 Features enabled:");
    println!("   ✓ AI-driven performance analysis");
    println!("   ✓ Cross-platform validation");
    println!("   ✓ Automated regression detection");
    println!("   ✓ Intelligent optimization recommendations");
    println!("   ✓ Machine learning-based predictions\n");

    // Run enhanced benchmarks
    match benchmark_suite.run_enhanced_benchmarks() {
        Ok(report) => {
            display_benchmark_results(&report)?;
        }
        Err(e) => {
            println!("❌ Benchmark execution failed: {}", e);
            println!("📝 Note: This is expected in the example as we're showcasing the structure");
        }
    }

    // Demonstrate quick AI analysis
    println!("\n🚀 Quick AI Analysis Example");
    println!("============================");

    match scirs2_stats::benchmark_suite_enhanced::run_quick_ai_analysis(
        100000,
        "correlation_matrix",
    ) {
        Ok(recommendations) => {
            println!("📈 AI Analysis for correlation_matrix with 100,000 elements:");
            for (i, rec) in recommendations.iter().enumerate() {
                display_recommendation(i + 1, rec);
            }
        }
        Err(e) => {
            println!("❌ Quick analysis failed: {}", e);
            println!("📝 Note: This is expected in the example");
        }
    }

    println!("\n🎯 Key Features of the Enhanced Benchmark Suite:");
    display_feature_overview();

    println!("\n✨ Usage Examples:");
    display_usage_examples();

    Ok(())
}

#[allow(dead_code)]
fn display_benchmark_results(
    report: &scirs2_stats::benchmark_suite_enhanced::EnhancedBenchmarkReport,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Enhanced Benchmark Results");
    println!("=============================\n");

    // Display base benchmark information
    println!("🕒 Benchmark timestamp: {}", report.base_report.timestamp);
    println!(
        "📏 Data sizes tested: {:?}",
        report.base_report.config.datasizes
    );
    println!(
        "🔄 Iterations per test: {}",
        report.base_report.config.iterations
    );

    // Display AI analysis results
    if let Some(ai_analysis) = &report.ai_analysis {
        println!("\n🤖 AI Performance Analysis:");
        println!(
            "   📊 Overall Performance Score: {:.1}/100",
            ai_analysis.performance_score
        );

        if !ai_analysis.bottlenecks.is_empty() {
            println!("   🔍 Performance Bottlenecks Identified:");
            for bottleneck in &ai_analysis.bottlenecks {
                println!(
                    "     • {:?} (Severity: {:.1}%)",
                    bottleneck.bottleneck_type, bottleneck.severity
                );
                println!(
                    "       Impact: {:.1}% performance loss",
                    bottleneck.performance_impact
                );
                if !bottleneck.mitigation_strategies.is_empty() {
                    println!(
                        "       💡 Mitigation: {}",
                        bottleneck.mitigation_strategies[0]
                    );
                }
            }
        }

        if !ai_analysis.algorithm_recommendations.is_empty() {
            println!("   🎯 Algorithm Recommendations:");
            for (scenario, algorithm) in &ai_analysis.algorithm_recommendations {
                println!("     • {}: Use {}", scenario, algorithm);
            }
        }
    }

    // Display cross-platform analysis
    if let Some(cross_platform) = &report.cross_platform_analysis {
        println!("\n🌍 Cross-Platform Analysis:");
        println!(
            "   📊 Consistency Score: {:.2}",
            cross_platform.consistency_score
        );

        for (platform, performance) in &cross_platform.platform_comparison {
            println!("   🖥️  Platform: {}", platform);
            println!(
                "     • Relative Performance: {:.2}x",
                performance.relative_performance
            );
            println!(
                "     • Memory Efficiency: {:.1}%",
                performance.memory_efficiency * 100.0
            );
            println!(
                "     • SIMD Utilization: {:.1}%",
                performance.simd_utilization * 100.0
            );
        }
    }

    // Display regression analysis
    if let Some(regression) = &report.regression_analysis {
        println!("\n📈 Regression Analysis:");
        println!(
            "   🚨 Regression Detected: {}",
            if regression.regression_detected {
                "Yes"
            } else {
                "No"
            }
        );
        println!("   📊 Severity: {:?}", regression.severity_assessment);

        if !regression.performance_trends.is_empty() {
            println!("   📈 Performance Trends:");
            for (operation, trend) in &regression.performance_trends {
                println!(
                    "     • {}: {:?} (Strength: {:.2})",
                    operation, trend.trend_direction, trend.trend_strength
                );
            }
        }
    }

    // Display optimization recommendations
    if !report.optimization_recommendations.is_empty() {
        println!("\n💡 Intelligent Optimization Recommendations:");
        for (i, rec) in report.optimization_recommendations.iter().enumerate() {
            display_recommendation(i + 1, rec);
        }
    }

    // Display performance predictions
    if !report.performance_predictions.is_empty() {
        println!("\n🔮 Performance Predictions:");
        for (i, prediction) in report.performance_predictions.iter().enumerate() {
            println!(
                "   {}. Workload: {} with {} elements",
                i + 1,
                prediction.workload_characteristics.operation_type,
                prediction.workload_characteristics.datasize
            );
            println!(
                "      Predicted Time: {:.2}ms",
                prediction.predicted_execution_time.as_secs_f64() * 1000.0
            );
            println!(
                "      Memory Usage: {:.1}MB",
                prediction.predicted_memory_usage as f64 / (1024.0 * 1024.0)
            );
            println!(
                "      Confidence: {:.1}%",
                prediction.confidence_score * 100.0
            );
        }
    }

    Ok(())
}

#[allow(dead_code)]
fn display_recommendation(
    index: usize,
    rec: &scirs2_stats::benchmark_suite_enhanced::IntelligentRecommendation,
) {
    let priority_emoji = match rec.priority {
        RecommendationPriority::Critical => "🚨",
        RecommendationPriority::High => "⚡",
        RecommendationPriority::Medium => "💡",
        RecommendationPriority::Low => "💭",
    };

    let efforttext = match rec.implementation_effort {
        ImplementationEffort::Trivial => "Trivial (<1h)",
        ImplementationEffort::Low => "Low (1-4h)",
        ImplementationEffort::Medium => "Medium (1-2d)",
        ImplementationEffort::High => "High (1-2w)",
        ImplementationEffort::Complex => "Complex (>2w)",
    };

    println!(
        "   {}. {} {:?} - Priority: {:?}",
        index, priority_emoji, rec.category, rec.priority
    );
    println!("      📝 {}", rec.recommendation);
    println!(
        "      📊 Estimated Improvement: +{:.0}%",
        rec.estimated_improvement
    );
    println!("      ⏱️  Implementation Effort: {}", efforttext);

    if !rec.implementation_details.is_empty() {
        println!("      🔧 Implementation:");
        for detail in &rec.implementation_details {
            println!("         • {}", detail);
        }
    }
}

#[allow(dead_code)]
fn display_feature_overview() {
    println!("   🤖 AI-Driven Analysis:");
    println!("      • Intelligent bottleneck identification");
    println!("      • Algorithm recommendation based on workload");
    println!("      • Performance anomaly detection");
    println!("      • Feature importance analysis");

    println!("   🌍 Cross-Platform Validation:");
    println!("      • Performance consistency across platforms");
    println!("      • Platform-specific optimization identification");
    println!("      • Portability issue detection");
    println!("      • Architecture-aware recommendations");

    println!("   📈 Regression Detection:");
    println!("      • Statistical significance testing");
    println!("      • Performance trend analysis");
    println!("      • Automated regression alerts");
    println!("      • Historical performance tracking");

    println!("   💡 Intelligent Recommendations:");
    println!("      • Prioritized optimization suggestions");
    println!("      • Implementation effort estimation");
    println!("      • Code examples and specific actions");
    println!("      • ROI-based recommendation ranking");

    println!("   🔮 Performance Prediction:");
    println!("      • ML-based execution time prediction");
    println!("      • Memory usage forecasting");
    println!("      • Optimal configuration suggestions");
    println!("      • Workload-specific optimization");
}

#[allow(dead_code)]
fn display_usage_examples() {
    println!("   📋 Basic Usage:");
    println!("      let mut suite = create_enhanced_benchmark_suite();");
    println!("      let report = suite.run_enhanced_benchmarks()?;");

    println!("   ⚡ Quick Analysis:");
    println!("      let recommendations = run_quick_ai_analysis(10000, \"correlation\")?;");

    println!("   🔧 Custom Configuration:");
    println!("      let config = EnhancedBenchmarkConfig {{");
    println!("          enable_aianalysis: true,");
    println!("          enable_crossplatform: true,");
    println!("          ml_modelconfig: MLModelConfig::default(),");
    println!("          ..Default::default()");
    println!("      }};");
    println!("      let suite = create_configured_enhanced_benchmark_suite(config);");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enhanced_benchmark_configuration() {
        let config = EnhancedBenchmarkConfig::default();
        assert!(config.enable_ai_analysis);
        assert!(config.enable_cross_platform);
        assert!(config.enable_regression_detection);
        assert!(config.enable_optimization_recommendations);
        assert_eq!(config.regression_sensitivity, 0.05);
    }

    #[test]
    fn test_platform_target_creation() {
        let linux_target = PlatformTarget::x86_64_linux();
        assert_eq!(linux_target.arch, "x86_64");
        assert_eq!(linux_target.os, "linux");
        assert!(!linux_target.cpu_features.is_empty());

        let macos_target = PlatformTarget::aarch64_macos();
        assert_eq!(macos_target.arch, "aarch64");
        assert_eq!(macos_target.os, "macos");
    }

    #[test]
    fn test_ml_model_config() {
        let config = MLModelConfig::default();
        assert_eq!(
            config.model_type,
            scirs2_stats::benchmark_suite_enhanced::MLModelType::RandomForest
        );
        assert_eq!(config.confidence_threshold, 0.8);
        assert_eq!(config.training_retention_days, 90);
    }
}
