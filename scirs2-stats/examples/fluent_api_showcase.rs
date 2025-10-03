//! Enhanced Fluent API Showcase
//!
//! This example demonstrates the enhanced fluent API with method chaining,
//! intelligent optimization, and streamlined statistical analysis workflows.

use scirs2_core::ndarray::Array1;
use scirs2_stats::{
    api_standardization_enhanced::{
        quick_correlation, quick_descriptive, stats, AutoOptimizationLevel, CorrelationMethod,
        CorrelationType, FluentStatsConfig, MemoryStrategy, ResultFormat,
    },
    NullHandling,
};
use statrs::statistics::Statistics;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌟 Enhanced Fluent API for Statistical Computing");
    println!("===============================================\n");

    // Demonstrate basic fluent API usage
    demonstrate_basic_fluent_api()?;

    // Demonstrate method chaining
    demonstrate_method_chaining()?;

    // Demonstrate intelligent optimization
    demonstrate_intelligent_optimization()?;

    // Demonstrate quick analysis functions
    demonstrate_quick_analysis()?;

    // Demonstrate advanced configuration
    demonstrate_advanced_configuration()?;

    println!("\n✨ Key Benefits of the Enhanced Fluent API:");
    display_api_benefits();

    Ok(())
}

#[allow(dead_code)]
fn demonstrate_basic_fluent_api() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Basic Fluent API Usage");
    println!("========================\n");

    // Create sample data
    let data = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    println!("📊 Sample data: {:?}", data);

    // Basic fluent API chain
    let mut chain = stats::<f64>()
        .parallel(true)
        .simd(true)
        .confidence(0.95)
        .descriptive()
        .mean()
        .variance(1)
        .std_dev(1)
        .and();

    println!("🔗 Created fluent chain with:");
    println!("   ✓ Parallel processing enabled");
    println!("   ✓ SIMD optimizations enabled");
    println!("   ✓ 95% confidence level");
    println!("   ✓ Descriptive statistics: mean, variance, std_dev");

    // Execute the chain (placeholder - would execute with real data)
    match chain.execute() {
        Ok(results) => {
            println!("✅ Chain executed successfully");
            println!("📈 Operations completed: {}", results.iter().count());
        }
        Err(e) => {
            println!("❌ Chain execution failed: {}", e);
            println!("📝 Note: This is expected in the example");
        }
    }

    Ok(())
}

#[allow(dead_code)]
fn demonstrate_method_chaining() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔗 Advanced Method Chaining");
    println!("===========================\n");

    // Complex analysis chain
    let mut complex_chain = stats::<f64>()
        .optimization(AutoOptimizationLevel::Intelligent)
        .format(ResultFormat::Comprehensive)
        .memory_limit(100 * 1024 * 1024) // 100MB limit
        .null_handling(NullHandling::Exclude)
        .descriptive()
        .all_basic() // Add all basic descriptive statistics
        .and()
        .correlation()
        .method(CorrelationMethod::Pearson)
        .matrix()
        .and()
        .test()
        .t_test_one_sample(5.0)
        .and()
        .regression()
        .linear()
        .and();

    println!("🔗 Complex analysis chain created with:");
    println!("   🤖 Intelligent optimization");
    println!("   📊 Comprehensive result format");
    println!("   💾 100MB memory limit");
    println!("   🚫 Null value exclusion");
    println!("   📈 All basic descriptive statistics");
    println!("   🔗 Pearson correlation matrix");
    println!("   🧪 One-sample t-test (μ=5.0)");
    println!("   📐 Linear regression");

    // Execute the chain
    match complex_chain.execute() {
        Ok(results) => {
            println!("✅ Complex chain executed successfully");
            println!("📊 Total operations: {}", results.iter().count());

            // Display hypothetical results
            for name_result in results.iter() {
                println!("   ✓ {:?}", name_result);
            }
        }
        Err(e) => {
            println!("❌ Chain execution failed: {}", e);
            println!("📝 Note: This is expected in the example");
        }
    }

    Ok(())
}

#[allow(dead_code)]
fn demonstrate_intelligent_optimization() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🤖 Intelligent Optimization Features");
    println!("===================================\n");

    // Configure intelligent optimization
    let config = FluentStatsConfig {
        enable_fluent_api: true,
        enable_result_caching: true,
        enable_streaming: true,
        auto_optimization_level: AutoOptimizationLevel::Intelligent,
        result_format: ResultFormat::Comprehensive,
        enable_performance_monitoring: true,
        memory_strategy: MemoryStrategy::Adaptive,
        ..Default::default()
    };

    let mut intelligent_stats =
        scirs2_stats::api_standardization_enhanced::stats_with::<f64>(config)
            .descriptive()
            .mean()
            .variance(1)
            .and();

    println!("🤖 Intelligent optimization features:");
    println!("   🔄 Result caching enabled");
    println!("   📊 Streaming operations for large datasets");
    println!("   🎯 ML-based operation optimization");
    println!("   📈 Real-time performance monitoring");
    println!("   🧠 Adaptive memory management");

    // Execute with intelligent optimization
    match intelligent_stats.execute() {
        Ok(_results) => {
            println!("✅ Intelligent optimization completed");
            println!("🎯 Operations automatically optimized for performance");
        }
        Err(e) => {
            println!("❌ Optimization failed: {}", e);
            println!("📝 Note: This is expected in the example");
        }
    }

    Ok(())
}

#[allow(dead_code)]
fn demonstrate_quick_analysis() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n⚡ Quick Analysis Functions");
    println!("===========================\n");

    println!("🚀 Quick descriptive analysis:");
    let _quick_desc = quick_descriptive::<f64>()
        .mean()
        .variance(1)
        .skewness()
        .kurtosis()
        .and();

    println!("   ✓ Quick descriptive statistics chain created");

    println!("\n🔗 Quick correlation analysis:");
    let _quick_corr = quick_correlation::<f64>().pearson().matrix().and();

    println!("   ✓ Quick Pearson correlation matrix analysis created");

    println!("\n💡 Benefits of quick functions:");
    println!("   🚀 Instant setup with sensible defaults");
    println!("   🎯 Focused on specific analysis types");
    println!("   🔧 Easily customizable");
    println!("   ⚡ Optimized for common workflows");

    Ok(())
}

#[allow(dead_code)]
fn demonstrate_advanced_configuration() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔧 Advanced Configuration Options");
    println!("================================\n");

    // Showcase different optimization levels
    println!("🎚️  Optimization Levels:");

    let basic_config = FluentStatsConfig {
        auto_optimization_level: AutoOptimizationLevel::Basic,
        ..Default::default()
    };
    println!("   📊 Basic: SIMD + Parallel optimizations");

    let intelligent_config = FluentStatsConfig {
        auto_optimization_level: AutoOptimizationLevel::Intelligent,
        ..Default::default()
    };
    println!("   🤖 Intelligent: ML-based optimization selection");

    let aggressive_config = FluentStatsConfig {
        auto_optimization_level: AutoOptimizationLevel::Aggressive,
        ..Default::default()
    };
    println!("   🚀 Aggressive: Maximum optimization (operation fusion)");

    // Showcase result formats
    println!("\n📊 Result Formats:");
    println!("   📋 Minimal: Just the result value");
    println!("   📈 Standard: Result with basic metadata");
    println!("   📚 Comprehensive: Full metadata and diagnostics");

    // Showcase memory strategies
    println!("\n💾 Memory Strategies:");
    println!("   🔒 Conservative: Minimize memory usage");
    println!("   ⚖️  Balanced: Balance memory and performance");
    println!("   🚀 Performance: Optimize for performance");
    println!("   🧠 Adaptive: Adapt based on system resources");

    // Create examples with different configurations
    let _conservative_chain =
        scirs2_stats::api_standardization_enhanced::stats_with::<f64>(FluentStatsConfig {
            memory_strategy: MemoryStrategy::Conservative,
            result_format: ResultFormat::Minimal,
            ..Default::default()
        });

    let _performance_chain =
        scirs2_stats::api_standardization_enhanced::stats_with::<f64>(FluentStatsConfig {
            memory_strategy: MemoryStrategy::Performance,
            result_format: ResultFormat::Comprehensive,
            auto_optimization_level: AutoOptimizationLevel::Aggressive,
            ..Default::default()
        });

    println!("\n✅ Different configuration examples created successfully");

    Ok(())
}

#[allow(dead_code)]
fn display_api_benefits() {
    println!("   🔗 **Method Chaining**: Fluent, readable statistical workflows");
    println!("   🤖 **Intelligent Optimization**: ML-based automatic optimization");
    println!("   ⚡ **Quick Functions**: Instant analysis with sensible defaults");
    println!("   🔧 **Flexible Configuration**: Extensive customization options");
    println!("   📊 **Comprehensive Results**: Rich metadata and diagnostics");
    println!("   🚀 **Performance**: SIMD, parallel, and memory optimizations");
    println!("   🔄 **Caching**: Automatic result caching for repeated operations");
    println!("   📈 **Streaming**: Support for large dataset processing");
    println!("   🎯 **Type Safety**: Compile-time guarantees for statistical operations");
    println!("   🔌 **Extensible**: Easy to add new operations and optimizations");

    println!("\n📚 Example Usage Patterns:");
    println!("   // Quick analysis");
    println!("   let results = quick_descriptive().mean().variance(1).execute()?;");
    println!();
    println!("   // Complex chained analysis");
    println!("   let results = stats()");
    println!("       .parallel(true).simd(true).confidence(0.99)");
    println!("       .descriptive().all_basic().and()");
    println!("       .correlation().pearson().matrix().and()");
    println!("       .test().t_test_independent().and()");
    println!("       .execute()?;");
    println!();
    println!("   // Custom configuration");
    println!("   let config = FluentStatsConfig {{");
    println!("       auto_optimizationlevel: AutoOptimizationLevel::Intelligent,");
    println!("       memorystrategy: MemoryStrategy::Adaptive,");
    println!("       resultformat: ResultFormat::Comprehensive,");
    println!("       ..Default::default()");
    println!("   }};");
    println!("   let results = stats_with(config).descriptive().mean().execute()?;");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fluent_api_creation() {
        let _stats = stats::<f64>();
        assert!(true); // Compilation test
    }

    #[test]
    fn test_quick_functions() {
        let _desc = quick_descriptive::<f64>();
        let _corr = quick_correlation::<f64>();
        assert!(true); // Compilation test
    }

    #[test]
    fn test_method_chaining() {
        let _chain = stats::<f64>()
            .parallel(true)
            .simd(true)
            .confidence(0.95)
            .descriptive()
            .mean()
            .variance(1)
            .and();

        assert!(true); // Compilation test
    }

    #[test]
    fn test_configuration_options() {
        let config = FluentStatsConfig {
            auto_optimization_level: AutoOptimizationLevel::Intelligent,
            memory_strategy: MemoryStrategy::Adaptive,
            result_format: ResultFormat::Comprehensive,
            ..Default::default()
        };

        let _stats = scirs2_stats::api_standardization_enhanced::stats_with::<f64>(config);
        assert!(true); // Compilation test
    }
}
