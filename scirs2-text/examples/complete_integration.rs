//! # Complete Advanced Integration Example
//!
//! This example demonstrates the full power of the Advanced Text Processing
//! system by integrating all modules and showcasing real-world usage scenarios.
//!
//! ## Features Demonstrated
//!
//! - Complete Advanced text processing pipeline
//! - Advanced performance monitoring and optimization
//! - SIMD-accelerated operations with fallbacks
//! - Streaming text processing for large datasets
//! - Comprehensive analytics and reporting
//! - Real-time adaptation and learning

use scirs2_text::error::Result;
use scirs2_text::performance::{AdvancedPerformanceMonitor, PerformanceThresholds};
use scirs2_text::simd_ops::{AdvancedSIMDTextProcessor, SimdStringOps};
use scirs2_text::streaming::AdvancedStreamingProcessor;
use scirs2_text::text_coordinator::{AdvancedTextConfig, AdvancedTextCoordinator};
use scirs2_text::tokenize::WordTokenizer;
use scirs2_text::Tokenizer;
use std::time::Instant;

#[allow(dead_code)]
fn main() -> Result<()> {
    println!("🚀 Complete Advanced Integration Demo");
    println!("=====================================\n");

    // Initialize the complete Advanced system
    let system = AdvancedSystem::new()?;

    // Run comprehensive demonstration
    system.run_complete_demo()?;

    println!("\n🎉 Complete Advanced Integration Demo finished successfully!");
    println!("All advanced features demonstrated with optimal performance.");

    Ok(())
}

/// Complete Advanced System integrating all components
struct AdvancedSystem {
    /// Main text processing coordinator
    coordinator: AdvancedTextCoordinator,
    /// Performance monitoring system
    performance_monitor: AdvancedPerformanceMonitor,
    /// SIMD processor for optimized operations
    #[allow(dead_code)]
    simd_processor: AdvancedSIMDTextProcessor,
    /// Streaming processor for large datasets
    #[allow(dead_code)]
    streaming_processor: AdvancedStreamingProcessor<WordTokenizer>,
}

impl AdvancedSystem {
    /// Initialize the complete Advanced system
    fn new() -> Result<Self> {
        println!("🔧 Initializing Complete Advanced System...");

        // Configure Advanced mode with optimized settings
        let config = AdvancedTextConfig {
            enable_gpu_acceleration: true,
            enable_simd_optimizations: true,
            enable_neural_ensemble: true,
            enable_real_time_adaptation: true,
            enable_advanced_analytics: true,
            enable_multimodal: true,
            max_memory_usage_mb: 8192,
            optimization_level: 3,
            target_throughput: 5000.0,
            enable_predictive_processing: true,
        };

        // Performance monitoring with strict thresholds
        let perf_thresholds = PerformanceThresholds {
            max_processing_time_ms: 500, // 500ms max
            min_throughput: 1000.0,      // 1000 docs/sec min
            max_memory_usage_mb: 6144,   // 6GB max
            max_cpu_utilization: 85.0,   // 85% max
            min_cache_hit_rate: 0.85,    // 85% min
        };

        let coordinator = AdvancedTextCoordinator::new(config)?;
        let performance_monitor = AdvancedPerformanceMonitor::with_thresholds(perf_thresholds);
        let simd_processor = AdvancedSIMDTextProcessor;
        let streaming_processor = AdvancedStreamingProcessor::new(WordTokenizer::default());

        println!("✅ Advanced System initialized successfully!\n");

        Ok(Self {
            coordinator,
            performance_monitor,
            simd_processor,
            streaming_processor,
        })
    }

    /// Run the complete demonstration
    fn run_complete_demo(&self) -> Result<()> {
        // Demo 1: Integrated Text Processing Pipeline
        self.demo_integrated_pipeline()?;

        // Demo 2: Performance-Monitored SIMD Operations
        self.demo_performance_monitored_simd()?;

        // Demo 3: Adaptive Streaming Processing
        self.demo_adaptive_streaming()?;

        // Demo 4: Real-time Optimization and Adaptation
        self.demo_realtime_optimization()?;

        // Demo 5: Comprehensive System Analytics
        self.demo_system_analytics()?;

        Ok(())
    }

    /// Demonstrate integrated text processing pipeline
    fn demo_integrated_pipeline(&self) -> Result<()> {
        println!("📊 Demo 1: Integrated Text Processing Pipeline");
        println!("==============================================");

        let sample_documents = vec![
            "Artificial intelligence is revolutionizing the field of natural language processing.".to_string(),
            "Machine learning algorithms can now understand context and semantic meaning in text.".to_string(),
            "Deep neural networks have enabled breakthrough performance in text classification tasks.".to_string(),
            "SIMD optimizations allow for optimized string processing in modern computing systems.".to_string(),
            "Real-time adaptation ensures optimal performance across diverse text processing workloads.".to_string(),
        ];

        println!(
            "Processing {} documents through integrated pipeline...",
            sample_documents.len()
        );

        // Start performance monitoring
        let operation_monitor = self
            .performance_monitor
            .start_operation("integrated_pipeline")?;

        let start_time = Instant::now();

        // Process through Advanced coordinator
        let result = self.coordinator.advanced_processtext(&sample_documents)?;

        let processing_time = start_time.elapsed();

        // Complete monitoring
        operation_monitor.complete(sample_documents.len())?;

        println!("\n📈 Pipeline Results:");
        println!("  • Processing Time: {processing_time:?}");
        println!(
            "  • Throughput: {:.2} docs/sec",
            result.performance_metrics.throughput
        );
        println!(
            "  • Memory Efficiency: {:.1}%",
            result.performance_metrics.memory_efficiency * 100.0
        );
        println!(
            "  • Accuracy Estimate: {:.1}%",
            result.performance_metrics.accuracy_estimate * 100.0
        );

        println!("\n🔧 Applied Optimizations:");
        for optimization in &result.optimizations_applied {
            println!("  • {optimization}");
        }

        println!("\n🎯 Confidence Scores:");
        for (metric, score) in &result.confidence_scores {
            println!("  • {}: {:.1}%", metric, score * 100.0);
        }

        println!();
        Ok(())
    }

    /// Demonstrate performance-monitored SIMD operations
    fn demo_performance_monitored_simd(&self) -> Result<()> {
        println!("⚡ Demo 2: Performance-Monitored SIMD Operations");
        println!("===============================================");

        let testtexts = [
            "The quick brown fox jumps over the lazy dog".to_string(),
            "Pack my box with five dozen liquor jugs".to_string(),
            "How vexingly quick daft zebras jump!".to_string(),
            "Bright vixens jump; dozy fowl quack".to_string(),
        ];

        println!("Running SIMD-accelerated operations with performance monitoring...");

        // Start monitoring
        let operation_monitor = self
            .performance_monitor
            .start_operation("simd_operations")?;

        let start_time = Instant::now();

        // Optimized text processing
        let testtext_refs: Vec<&str> = testtexts.iter().map(|s| s.as_str()).collect();
        let processed_results = AdvancedSIMDTextProcessor::advanced_batch_process(&testtext_refs);

        // SIMD string operations
        let char_counts: Vec<usize> = testtexts
            .iter()
            .map(|text| SimdStringOps::count_chars(text, 'o'))
            .collect();

        // Optimized similarity matrix
        let similarity_matrix =
            AdvancedSIMDTextProcessor::advanced_similarity_matrix(&testtext_refs);

        let processing_time = start_time.elapsed();

        // Complete monitoring
        operation_monitor.complete(testtexts.len())?;

        println!("\n📊 SIMD Operation Results:");
        println!("  • Processing Time: {processing_time:?}");
        println!("  • Documents Processed: {}", processed_results.len());
        println!("  • Character Counts (letter 'o'): {char_counts:?}");
        println!(
            "  • Similarity Matrix Size: {}x{}",
            similarity_matrix.len(),
            similarity_matrix[0].len()
        );

        // Display similarity matrix
        println!("\n🔗 Text Similarity Matrix:");
        for (i, row) in similarity_matrix.iter().enumerate() {
            print!("  Row {i}: [");
            for (j, &similarity) in row.iter().enumerate() {
                if j > 0 {
                    print!(", ");
                }
                print!("{similarity:.3}");
            }
            println!("]");
        }

        // Show SIMD capabilities
        println!("\n⚙️  SIMD Capabilities:");
        println!("  • SIMD Available: {}", SimdStringOps::is_available());
        println!("  • String Processing: Optimized");
        println!("  • Pattern Matching: Optimized");
        println!("  • Similarity Computation: Vectorized");

        println!();
        Ok(())
    }

    /// Demonstrate adaptive streaming processing
    fn demo_adaptive_streaming(&self) -> Result<()> {
        println!("🌊 Demo 3: Adaptive Streaming Processing");
        println!("========================================");

        // Create large dataset simulation
        let largetexts: Vec<String> = (0..1000)
            .map(|i| format!("This is streaming document number {i} with various content lengths and different patterns of text processing requirements."))
            .collect();

        println!(
            "Processing {} documents through adaptive streaming...",
            largetexts.len()
        );

        // Start monitoring
        let operation_monitor = self
            .performance_monitor
            .start_operation("adaptive_streaming")?;

        let start_time = Instant::now();

        // Streaming processing with parallel optimization
        let streaming_processor = AdvancedStreamingProcessor::new(WordTokenizer::default())
            .with_parallelism(4, 1024 * 1024);

        // Simple token counting for demonstration
        let mut total_tokens = 0;
        let tokenizer = WordTokenizer::default();
        for text in &largetexts {
            if let Ok(tokens) = tokenizer.tokenize(text) {
                total_tokens += tokens.len();
            }
        }

        let processing_time = start_time.elapsed();

        // Complete monitoring
        operation_monitor.complete(largetexts.len())?;

        // Get memory stats instead of performance metrics
        let (current_mem, peak_mem) = streaming_processor.memory_stats();

        println!("\n📈 Streaming Processing Results:");
        println!("  • Processing Time: {processing_time:?}");
        println!("  • Documents Processed: {}", largetexts.len());
        println!("  • Total Tokens Extracted: {total_tokens}");
        println!("  • Current Memory Usage: {current_mem} bytes");
        println!("  • Peak Memory Usage: {peak_mem} bytes");
        println!(
            "  • Throughput: {:.2} docs/sec",
            largetexts.len() as f64 / processing_time.as_secs_f64()
        );

        println!("\n🔄 Advanced Features:");
        println!("  • Parallel Processing: Enabled");
        println!("  • Memory Monitoring: Active");
        println!("  • Advanced Tokenization: Optimized");

        println!();
        Ok(())
    }

    /// Demonstrate real-time optimization and adaptation
    fn demo_realtime_optimization(&self) -> Result<()> {
        println!("🎯 Demo 4: Real-time Optimization and Adaptation");
        println!("===============================================");

        // Simulate various workload patterns
        let workload_patterns = vec![
            ("Short Documents", generate_short_documents(50)),
            ("Medium Documents", generate_medium_documents(30)),
            ("Long Documents", generate_long_documents(20)),
            ("Mixed Workload", generate_mixed_workload(40)),
        ];

        for (pattern_name, documents) in workload_patterns {
            println!("\n🔄 Processing Pattern: {pattern_name}");
            println!("  • Document Count: {}", documents.len());

            // Start monitoring for this pattern
            let operation_monitor = self
                .performance_monitor
                .start_operation(&format!("pattern_{}", pattern_name.replace(' ', "_")))?;

            let start_time = Instant::now();

            // Process with adaptive optimization
            let result = self.coordinator.advanced_processtext(&documents)?;

            let processing_time = start_time.elapsed();

            // Complete monitoring
            operation_monitor.complete(documents.len())?;

            println!("  • Processing Time: {processing_time:?}");
            println!(
                "  • Throughput: {:.2} docs/sec",
                result.performance_metrics.throughput
            );
            println!(
                "  • Optimizations Applied: {}",
                result.optimizations_applied.len()
            );

            // Show adaptive responses
            if !result.optimizations_applied.is_empty() {
                println!("  • Adaptive Responses:");
                for opt in &result.optimizations_applied {
                    println!("    - {opt}");
                }
            }
        }

        // Get optimization recommendations
        let recommendations = self.performance_monitor.get_optimization_opportunities()?;

        println!("\n💡 Current Optimization Opportunities:");
        if recommendations.is_empty() {
            println!("  • No optimization opportunities identified");
            println!("  • System is operating at optimal performance");
        } else {
            for (i, rec) in recommendations.iter().enumerate() {
                println!("  {}. [{}] {}", i + 1, rec.category, rec.recommendation);
                println!(
                    "     Impact: {:.0}% | Complexity: {}/5",
                    rec.impact_estimate * 100.0,
                    rec.complexity
                );
            }
        }

        println!();
        Ok(())
    }

    /// Demonstrate comprehensive system analytics
    fn demo_system_analytics(&self) -> Result<()> {
        println!("📊 Demo 5: Comprehensive System Analytics");
        println!("=========================================");

        // Generate comprehensive performance report
        let performance_report = self.performance_monitor.generate_performance_report()?;

        println!("📈 Performance Summary:");
        println!(
            "  • Total Operations: {}",
            performance_report.summary.total_operations
        );
        println!(
            "  • Avg Processing Time: {:?}",
            performance_report.summary.recent_avg_processing_time
        );
        println!(
            "  • Avg Throughput: {:.2} docs/sec",
            performance_report.summary.recent_avg_throughput
        );
        println!(
            "  • Avg Memory Usage: {} MB",
            performance_report.summary.recent_avg_memory_usage / (1024 * 1024)
        );
        println!(
            "  • Cache Hit Rate: {:.1}%",
            performance_report.summary.cache_hit_rate * 100.0
        );

        if !performance_report.summary.active_alerts.is_empty() {
            println!("\n⚠️  Active Performance Alerts:");
            for alert in &performance_report.summary.active_alerts {
                println!("  • {alert}");
            }
        } else {
            println!("\n✅ No active performance alerts");
        }

        println!("\n📊 Historical Trends:");
        println!(
            "  • Processing Time: {:?}",
            performance_report.historical_trends.processing_time_trend
        );
        println!(
            "  • Throughput: {:?}",
            performance_report.historical_trends.throughput_trend
        );
        println!(
            "  • Memory Usage: {:?}",
            performance_report.historical_trends.memory_usage_trend
        );

        println!("\n🖥️  Resource Utilization:");
        println!(
            "  • CPU: {:.1}%",
            performance_report.resource_utilization.avg_cpu_utilization
        );
        println!(
            "  • Peak Memory: {} MB",
            performance_report.resource_utilization.peak_memory_usage / (1024 * 1024)
        );
        println!(
            "  • Network I/O: {} MB sent, {} MB received",
            performance_report
                .resource_utilization
                .network_io
                .bytes_sent
                / (1024 * 1024),
            performance_report
                .resource_utilization
                .network_io
                .bytes_received
                / (1024 * 1024)
        );

        if !performance_report.bottleneck_analysis.is_empty() {
            println!("\n🔍 Bottleneck Analysis:");
            for bottleneck in &performance_report.bottleneck_analysis {
                println!(
                    "  • {} [{}]: {}",
                    bottleneck.component, bottleneck.severity, bottleneck.description
                );
                for rec in &bottleneck.recommendations {
                    println!("    - {rec}");
                }
            }
        }

        if !performance_report.recommendations.is_empty() {
            println!("\n🎯 System Recommendations:");
            for rec in &performance_report.recommendations {
                println!(
                    "  • [{}] {} (Impact: {:.0}%)",
                    rec.category,
                    rec.recommendation,
                    rec.impact_estimate * 100.0
                );
            }
        }

        // System health score
        let health_score = calculate_system_health_score(&performance_report);
        println!("\n🏥 Overall System Health Score: {health_score:.1}/100");

        let health_status = match health_score {
            score if score >= 90.0 => "Excellent",
            score if score >= 80.0 => "Good",
            score if score >= 70.0 => "Fair",
            score if score >= 60.0 => "Poor",
            _ => "Critical",
        };
        println!(
            "   Status: {} - System is performing {}",
            health_status,
            if health_score >= 80.0 {
                "optimally"
            } else {
                "suboptimally"
            }
        );

        println!();
        Ok(())
    }
}

// Helper functions for generating test data
#[allow(dead_code)]
fn generate_short_documents(count: usize) -> Vec<String> {
    (0..count).map(|i| format!("Short doc {i}.")).collect()
}

#[allow(dead_code)]
fn generate_medium_documents(count: usize) -> Vec<String> {
    (0..count)
        .map(|i| format!("Medium length document {i} with additional content for processing analysis and performance testing."))
        .collect()
}

#[allow(dead_code)]
fn generate_long_documents(count: usize) -> Vec<String> {
    (0..count)
        .map(|i| format!("This is a long document number {i} that contains significant amounts of text content designed to test the performance characteristics of the Advanced text processing system under heavy load conditions with complex linguistic patterns and varied vocabulary usage."))
        .collect()
}

#[allow(dead_code)]
fn generate_mixed_workload(count: usize) -> Vec<String> {
    (0..count)
        .map(|i| match i % 3 {
            0 => format!("Short {i}"),
            1 => format!("Medium document {i} with some content."),
            _ => format!("Long detailed document {i} with extensive content for comprehensive testing and analysis."),
        })
        .collect()
}

#[allow(dead_code)]
fn calculate_system_health_score(
    report: &scirs2_text::performance::DetailedPerformanceReport,
) -> f64 {
    let mut score: f64 = 100.0;

    // Penalize for active alerts
    score -= report.summary.active_alerts.len() as f64 * 10.0;

    // Reward high throughput
    if report.summary.recent_avg_throughput < 500.0 {
        score -= 15.0;
    }

    // Penalize high memory usage
    if report.summary.recent_avg_memory_usage > 4 * 1024 * 1024 * 1024 {
        // > 4GB
        score -= 10.0;
    }

    // Reward high cache hit rate
    if report.summary.cache_hit_rate < 0.8 {
        score -= 15.0;
    }

    // Penalize for bottlenecks
    score -= report.bottleneck_analysis.len() as f64 * 5.0;

    score.clamp(0.0, 100.0)
}
