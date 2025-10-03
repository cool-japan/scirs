//! Advanced Mode Showcase
//!
//! This example demonstrates the advanced-sophisticated enhancements added to scirs2-datasets,
//! including advanced analytics, GPU optimization, and adaptive streaming processing.

use scirs2_datasets::{
    // Adaptive streaming
    create_adaptive_engine_with_config,
    // Core functionality
    make_classification,
    quick_quality_assessment,
    AdaptiveStreamConfig,
    AdvancedDatasetAnalyzer,
    AdvancedGpuOptimizer,
    ChunkMetadata,
    DataCharacteristics,
    Dataset,
    GpuBackend,
    GpuConfig,
    GpuContext,
    StatisticalMoments,
    StreamChunk,
    TrendDirection,
    TrendIndicators,
};
use statrs::statistics::Statistics;
use std::time::Instant;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 SciRS2-Datasets Advanced Mode Showcase");
    println!("===========================================\n");

    // Create a sample dataset for demonstration
    let dataset = create_sampledataset()?;
    println!(
        "📊 Created sample dataset: {} samples, {} features",
        dataset.n_samples(),
        dataset.n_features()
    );

    // Demonstrate advanced analytics
    demonstrate_advanced_analytics(&dataset)?;

    // Demonstrate advanced-GPU optimization
    demonstrate_advanced_gpu_optimization()?;

    // Demonstrate adaptive streaming
    demonstrate_adaptive_streaming(&dataset)?;

    println!("\n✅ Advanced mode demonstration completed successfully!");
    Ok(())
}

/// Create a sample dataset for demonstration
#[allow(dead_code)]
fn create_sampledataset() -> Result<Dataset, Box<dyn std::error::Error>> {
    println!("🔧 Generating sample classification dataset...");

    let dataset = make_classification(
        1000,     // n_samples
        10,       // n_features
        3,        // n_classes
        2,        // n_clusters_per_class
        5,        // n_informative
        Some(42), // random_state
    )?;

    Ok(dataset)
}

/// Demonstrate advanced analytics capabilities
#[allow(dead_code)]
fn demonstrate_advanced_analytics(dataset: &Dataset) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🧠 Advanced Analytics Demonstration");
    println!("==========================================");

    // Quick quality assessment
    println!("📈 Running quick quality assessment...");
    let quick_quality = quick_quality_assessment(dataset)?;
    println!("   Quality Score: {quick_quality:.3}");

    // Comprehensive advanced-analysis
    println!("🔬 Running comprehensive advanced-analysis...");
    let start_time = Instant::now();

    let analyzer = AdvancedDatasetAnalyzer::new()
        .with_gpu(true)
        .with_advanced_precision(true)
        .with_significance_threshold(0.01);

    let metrics = analyzer.analyze_dataset_quality(dataset)?;
    let analysis_time = start_time.elapsed();

    println!("   Analysis completed in: {analysis_time:?}");
    println!("   Complexity Score: {:.3}", metrics.complexity_score);
    println!("   Entropy: {:.3}", metrics.entropy);
    println!("   Outlier Score: {:.3}", metrics.outlier_score);
    println!("   ML Quality Score: {:.3}", metrics.ml_quality_score);

    // Display normality assessment
    println!("   Normality Assessment:");
    println!(
        "     Overall Normality: {:.3}",
        metrics.normality_assessment.overall_normality
    );
    println!(
        "     Shapiro-Wilk (avg): {:.3}",
        metrics.normality_assessment.shapiro_wilk_scores.mean()
    );

    // Display correlation insights
    println!("   Correlation Insights:");
    println!(
        "     Feature Importance (top 3): {:?}",
        metrics
            .correlation_insights
            .feature_importance
            .iter()
            .take(3)
            .map(|&x| format!("{x:.3}"))
            .collect::<Vec<_>>()
    );

    Ok(())
}

/// Demonstrate advanced-GPU optimization capabilities
#[allow(dead_code)]
fn demonstrate_advanced_gpu_optimization() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n⚡ Advanced-GPU Optimization Demonstration");
    println!("=====================================");

    // Create GPU context (falls back to CPU if no GPU available)
    println!("🔧 Initializing GPU context...");
    let gpu_config = GpuConfig {
        backend: GpuBackend::Cpu,
        threads_per_block: 1, // CPU backend only supports 1 thread per block
        ..Default::default()
    };
    let gpu_context = GpuContext::new(gpu_config)?; // Using CPU backend for demo
    println!("   Backend: {:?}", gpu_context.backend());

    // Create advanced-GPU optimizer
    let optimizer = AdvancedGpuOptimizer::new()
        .with_adaptive_kernels(true)
        .with_memory_prefetch(true)
        .with_multi_gpu(false) // Single GPU for demo
        .with_auto_tuning(true);

    // Generate advanced-optimized matrix
    println!("🔥 Generating advanced-optimized matrix...");
    let start_time = Instant::now();
    let matrix = optimizer.generate_advanced_optimized_matrix(
        &gpu_context,
        500,      // rows
        200,      // cols
        "normal", // distribution
    )?;
    let generation_time = start_time.elapsed();

    println!(
        "   Generated {}x{} matrix in: {:?}",
        matrix.nrows(),
        matrix.ncols(),
        generation_time
    );
    let matrix_mean = matrix.clone().mean();
    let matrix_std = matrix.var(1.0).sqrt();
    println!(
        "   Matrix stats: mean={:.3}, std={:.3}",
        matrix_mean, matrix_std
    );

    // Benchmark performance
    println!("📊 Running performance benchmarks...");
    let datashapes = vec![(100, 50), (500, 200), (1000, 500)];
    let benchmark_results =
        optimizer.benchmark_performance(&gpu_context, "matrix_generation", &datashapes)?;

    println!("   Benchmark Results:");
    println!(
        "     Best Speedup: {:.2}x",
        benchmark_results.best_speedup()
    );
    println!(
        "     Average Speedup: {:.2}x",
        benchmark_results.average_speedup()
    );
    println!(
        "     Total Memory Usage: {:.1} MB",
        benchmark_results.total_memory_usage()
    );

    Ok(())
}

/// Demonstrate adaptive streaming capabilities
#[allow(dead_code)]
fn demonstrate_adaptive_streaming(dataset: &Dataset) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🌊 Adaptive Streaming Demonstration");
    println!("===================================");

    // Configure streaming engine
    let config = AdaptiveStreamConfig::default();

    println!("🔧 Initializing adaptive streaming engine...");
    let mut engine = create_adaptive_engine_with_config(config);

    // Simulate streaming data
    println!("📡 Simulating data stream...");
    let data = &dataset.data;
    let chunksize = 20;
    let num_chunks = (data.nrows() / chunksize).min(10); // Limit for demo

    let mut total_processed = 0;
    let start_time = Instant::now();

    for i in 0..num_chunks {
        let start_row = i * chunksize;
        let end_row = (start_row + chunksize).min(data.nrows());

        // Create chunk from dataset slice
        let chunkdata = data
            .slice(scirs2_core::ndarray::s![start_row..end_row, ..])
            .to_owned();

        let chunk = StreamChunk {
            data: chunkdata,
            timestamp: Instant::now(),
            metadata: ChunkMetadata {
                source_id: format!("demo_source_{i}"),
                sequence_number: i as u64,
                characteristics: DataCharacteristics {
                    moments: StatisticalMoments {
                        mean: 0.0,
                        variance: 1.0,
                        skewness: 0.0,
                        kurtosis: 0.0,
                    },
                    entropy: 1.0,
                    trend: TrendIndicators {
                        linear_slope: 0.1,
                        trend_strength: 0.5,
                        direction: TrendDirection::Increasing,
                        seasonality: 0.2,
                    },
                    anomaly_score: 0.1,
                },
            },
            quality_score: 0.9,
        };

        // Process chunk
        let results = engine.process_stream(chunk)?;
        total_processed += results.len();

        if !results.is_empty() {
            println!(
                "   Processed batch {}: {} datasets generated",
                i + 1,
                results.len()
            );
        }
    }

    let streaming_time = start_time.elapsed();

    println!("   Streaming completed in: {streaming_time:?}");
    println!("   Total datasets processed: {total_processed}");

    // Get performance metrics
    println!("📈 Getting performance metrics...");
    let perf_metrics = engine.get_performance_metrics()?;
    println!("   Processing Latency: {:?}", perf_metrics.latency);
    println!("   Throughput: {:.1} chunks/sec", perf_metrics.throughput);
    println!(
        "   Memory Efficiency: {:.1}%",
        perf_metrics.memory_efficiency * 100.0
    );

    // Get quality metrics
    let quality_metrics = engine.get_quality_metrics()?;
    println!("   Quality Metrics:");
    println!(
        "     Integrity: {:.1}%",
        quality_metrics.integrity_score * 100.0
    );
    println!(
        "     Completeness: {:.1}%",
        quality_metrics.completeness_score * 100.0
    );
    println!(
        "     Overall Quality: {:.1}%",
        quality_metrics.overall_score * 100.0
    );

    // Get buffer statistics
    let buffer_stats = engine.get_buffer_statistics()?;
    println!("   Buffer Statistics:");
    println!("     Utilization: {:.1}%", buffer_stats.utilization * 100.0);
    println!("     Memory Usage: {} bytes", buffer_stats.memory_usage);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_dataset_creation() {
        let result = create_sampledataset();
        assert!(result.is_ok());
        let dataset = result.unwrap();
        assert_eq!(dataset.n_samples(), 1000);
        assert_eq!(dataset.n_features(), 10);
    }

    #[test]
    fn test_advanced_analytics_integration() {
        let dataset = create_sampledataset().unwrap();
        let result = demonstrate_advanced_analytics(&dataset);
        assert!(result.is_ok());
    }

    #[test]
    fn test_gpu_optimization_integration() {
        let result = demonstrate_advanced_gpu_optimization();
        match &result {
            Ok(_) => println!("GPU optimization test passed"),
            Err(e) => println!("GPU optimization test failed with error: {}", e),
        }
        assert!(result.is_ok());
    }

    #[test]
    fn test_adaptive_streaming_integration() {
        let dataset = create_sampledataset().unwrap();
        let result = demonstrate_adaptive_streaming(&dataset);
        assert!(result.is_ok());
    }
}
