//! Performance Report Generation for scirs2-spatial
//!
//! This module provides comprehensive performance reporting tools that generate
//! charts, summaries, and actionable insights from benchmark results.

use plotters::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Write};

/// Performance data point for analysis
#[derive(Debug, Clone)]
pub struct PerformanceDataPoint {
    pub test_name: String,
    pub data_size: usize,
    pub dimensions: usize,
    pub duration_ms: f64,
    pub throughput_ops_per_sec: f64,
    pub memory_mb: f64,
    pub speedup: f64,
    pub metric_type: String,
}

/// Performance report generator
pub struct PerformanceReportGenerator {
    data_points: Vec<PerformanceDataPoint>,
    output_dir: String,
}

impl PerformanceReportGenerator {
    pub fn new(output_dir: &str) -> io::Result<Self> {
        std::fs::create_dir_all(output_dir)?;
        Ok(Self {
            data_points: Vec::new(),
            output_dir: output_dir.to_string(),
        })
    }

    /// Add a performance data point
    pub fn add_data_point(&mut self, data_point: PerformanceDataPoint) {
        self.data_points.push(data_point);
    }

    /// Generate comprehensive performance report
    pub fn generate_report(&self) -> io::Result<()> {
        println!("Generating comprehensive performance report...");

        // Generate various charts and analyses
        self.generate_speedup_charts()?;
        self.generate_scaling_analysis()?;
        self.generate_memory_analysis()?;
        self.generate_metric_comparison()?;
        self.generate_summary_report()?;
        self.generate_recommendations()?;

        println!("Performance report generated in: {}", self.output_dir);
        Ok(())
    }

    /// Generate speedup comparison charts
    fn generate_speedup_charts(&self) -> io::Result<()> {
        let chart_path = format!("{}/speedup_comparison.png", self.output_dir);
        let root = BitMapBackend::new(&chart_path, (800, 600)).into_drawing_area();
        root.fill(&WHITE)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("Chart error: {}", e)))?;

        let mut chart = ChartBuilder::on(&root)
            .caption("SIMD vs Scalar Speedup Comparison", ("sans-serif", 40))
            .margin(10)
            .x_label_area_size(50)
            .y_label_area_size(60)
            .build_cartesian_2d(0..20000usize, 0.0..10.0f64)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("Chart error: {}", e)))?;

        chart
            .configure_mesh()
            .x_desc("Data Size")
            .y_desc("Speedup (x)")
            .draw()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("Chart error: {}", e)))?;

        // Extract SIMD speedup data
        let simd_data: Vec<(usize, f64)> = self
            .data_points
            .iter()
            .filter(|dp| dp.test_name.contains("simd") && dp.speedup > 1.0)
            .map(|dp| (dp.data_size, dp.speedup))
            .collect();

        if !simd_data.is_empty() {
            chart
                .draw_series(LineSeries::new(simd_data.iter().cloned(), &BLUE))
                .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("Chart error: {}", e)))?
                .label("SIMD Speedup")
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], BLUE));
        }

        // Extract parallel speedup data
        let parallel_data: Vec<(usize, f64)> = self
            .data_points
            .iter()
            .filter(|dp| dp.test_name.contains("parallel") && dp.speedup > 1.0)
            .map(|dp| (dp.data_size, dp.speedup))
            .collect();

        if !parallel_data.is_empty() {
            chart
                .draw_series(LineSeries::new(parallel_data.iter().cloned(), &RED))
                .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("Chart error: {}", e)))?
                .label("Parallel Speedup")
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], RED));
        }

        chart
            .configure_series_labels()
            .draw()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("Chart error: {}", e)))?;
        root.present()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("Chart error: {}", e)))?;

        Ok(())
    }

    /// Generate scaling analysis charts
    fn generate_scaling_analysis(&self) -> io::Result<()> {
        let chart_path = format!("{}/scaling_analysis.png", self.output_dir);
        let root = BitMapBackend::new(&chart_path, (800, 600)).into_drawing_area();
        root.fill(&WHITE)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("Chart error: {}", e)))?;

        let max_size = self
            .data_points
            .iter()
            .map(|dp| dp.data_size)
            .max()
            .unwrap_or(1000);
        let max_time = self
            .data_points
            .iter()
            .map(|dp| dp.duration_ms)
            .fold(0.0f64, f64::max);

        let mut chart = ChartBuilder::on(&root)
            .caption("Performance Scaling Analysis", ("sans-serif", 40))
            .margin(10)
            .x_label_area_size(50)
            .y_label_area_size(60)
            .build_cartesian_2d(0..max_size, 0.0..max_time)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("Chart error: {}", e)))?;

        chart
            .configure_mesh()
            .x_desc("Data Size")
            .y_desc("Time (ms)")
            .draw()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("Chart error: {}", e)))?;

        // Group data by algorithm type
        let mut algorithm_data: HashMap<String, Vec<(usize, f64)>> = HashMap::new();

        for dp in &self.data_points {
            let algorithm = if dp.test_name.contains("simd") {
                "SIMD".to_string()
            } else if dp.test_name.contains("parallel") {
                "Parallel".to_string()
            } else if dp.test_name.contains("scalar") {
                "Scalar".to_string()
            } else {
                "Other".to_string()
            };

            algorithm_data
                .entry(algorithm)
                .or_default()
                .push((dp.data_size, dp.duration_ms));
        }

        let colors = [&BLUE, &RED, &GREEN, &MAGENTA];
        for (i, (algorithm, data)) in algorithm_data.iter().enumerate() {
            if !data.is_empty() {
                let color = colors[i % colors.len()];
                chart
                    .draw_series(LineSeries::new(data.iter().cloned(), color))
                    .map_err(|e| {
                        io::Error::new(io::ErrorKind::Other, format!("Chart error: {}", e))
                    })?
                    .label(algorithm)
                    .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], color));
            }
        }

        chart
            .configure_series_labels()
            .draw()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("Chart error: {}", e)))?;
        root.present()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("Chart error: {}", e)))?;

        Ok(())
    }

    /// Generate memory analysis charts
    fn generate_memory_analysis(&self) -> io::Result<()> {
        let chart_path = format!("{}/memory_analysis.png", self.output_dir);
        let root = BitMapBackend::new(&chart_path, (800, 600)).into_drawing_area();
        root.fill(&WHITE)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("Chart error: {}", e)))?;

        let max_size = self
            .data_points
            .iter()
            .map(|dp| dp.data_size)
            .max()
            .unwrap_or(1000);
        let max_memory = self
            .data_points
            .iter()
            .map(|dp| dp.memory_mb)
            .fold(0.0f64, f64::max);

        let mut chart = ChartBuilder::on(&root)
            .caption("Memory Usage Analysis", ("sans-serif", 40))
            .margin(10)
            .x_label_area_size(50)
            .y_label_area_size(60)
            .build_cartesian_2d(0..max_size, 0.0..max_memory)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("Chart error: {}", e)))?;

        chart
            .configure_mesh()
            .x_desc("Data Size")
            .y_desc("Memory Usage (MB)")
            .draw()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("Chart error: {}", e)))?;

        // Plot memory usage data
        let memory_data: Vec<(usize, f64)> = self
            .data_points
            .iter()
            .map(|dp| (dp.data_size, dp.memory_mb))
            .collect();

        if !memory_data.is_empty() {
            chart
                .draw_series(LineSeries::new(memory_data.iter().cloned(), &BLUE))
                .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("Chart error: {}", e)))?
                .label("Memory Usage")
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], BLUE));

            // Add theoretical O(n²) scaling for comparison
            let theoretical_data: Vec<(usize, f64)> = (0..=max_size)
                .step_by(max_size / 10)
                .map(|size| {
                    let theoretical_memory = (size * size) as f64 * 8.0 / (1024.0 * 1024.0);
                    (size, theoretical_memory.min(max_memory))
                })
                .collect();

            chart
                .draw_series(LineSeries::new(theoretical_data.iter().cloned(), &RED))
                .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("Chart error: {}", e)))?
                .label("Theoretical O(n²)")
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], RED));
        }

        chart
            .configure_series_labels()
            .draw()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("Chart error: {}", e)))?;
        root.present()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("Chart error: {}", e)))?;

        Ok(())
    }

    /// Generate metric comparison charts
    fn generate_metric_comparison(&self) -> io::Result<()> {
        let chart_path = format!("{}/metric_comparison.png", self.output_dir);
        let root = BitMapBackend::new(&chart_path, (800, 600)).into_drawing_area();
        root.fill(&WHITE)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("Chart error: {}", e)))?;

        // Group data by metric type
        let mut metric_groups: HashMap<String, Vec<f64>> = HashMap::new();

        for dp in &self.data_points {
            if !dp.metric_type.is_empty() {
                metric_groups
                    .entry(dp.metric_type.clone())
                    .or_default()
                    .push(dp.throughput_ops_per_sec);
            }
        }

        if metric_groups.is_empty() {
            return Ok(());
        }

        let metrics: Vec<String> = metric_groups.keys().cloned().collect();
        let max_throughput = metric_groups
            .values()
            .flat_map(|v| v.iter())
            .fold(0.0f64, |a, &b| a.max(b));

        let mut chart = ChartBuilder::on(&root)
            .caption("Distance Metric Performance Comparison", ("sans-serif", 40))
            .margin(10)
            .x_label_area_size(80)
            .y_label_area_size(60)
            .build_cartesian_2d(0..metrics.len(), 0.0..max_throughput)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("Chart error: {}", e)))?;

        chart
            .configure_mesh()
            .x_desc("Distance Metric")
            .y_desc("Throughput (ops/sec)")
            .x_label_formatter(&|x| metrics.get(*x).cloned().unwrap_or_default())
            .draw()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("Chart error: {}", e)))?;

        // Create bar chart for each metric
        for (i, metric) in metrics.iter().enumerate() {
            if let Some(throughputs) = metric_groups.get(metric) {
                let avg_throughput = throughputs.iter().sum::<f64>() / throughputs.len() as f64;

                chart
                    .draw_series(std::iter::once(Rectangle::new(
                        [(i, 0.0), (i, avg_throughput)],
                        BLUE.filled(),
                    )))
                    .map_err(|e| {
                        io::Error::new(io::ErrorKind::Other, format!("Chart error: {}", e))
                    })?;
            }
        }

        root.present()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("Chart error: {}", e)))?;

        Ok(())
    }

    /// Generate summary report
    fn generate_summary_report(&self) -> io::Result<()> {
        let report_path = format!("{}/performance_summary.txt", self.output_dir);
        let mut file = File::create(report_path)?;

        writeln!(file, "SCIRS2-SPATIAL PERFORMANCE SUMMARY REPORT")?;
        writeln!(file, "==========================================")?;
        writeln!(file)?;

        // System information
        writeln!(file, "System Information:")?;
        #[cfg(target_arch = "x86_64")]
        {
            writeln!(file, "  Architecture: x86_64")?;
            writeln!(file, "  SSE2: {}", is_x86_feature_detected!("sse2"))?;
            writeln!(file, "  AVX: {}", is_x86_feature_detected!("avx"))?;
            writeln!(file, "  AVX2: {}", is_x86_feature_detected!("avx2"))?;
            writeln!(file, "  AVX-512F: {}", is_x86_feature_detected!("avx512f"))?;
        }

        #[cfg(target_arch = "aarch64")]
        {
            writeln!(file, "  Architecture: aarch64")?;
            writeln!(
                file,
                "  NEON: {}",
                std::arch::is_aarch64_feature_detected!("neon")
            )?;
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            writeln!(file, "  Architecture: Other (scalar fallbacks)")?;
        }

        writeln!(file)?;

        // Performance statistics
        writeln!(file, "Performance Statistics:")?;

        let simd_speedups: Vec<f64> = self
            .data_points
            .iter()
            .filter(|dp| dp.test_name.contains("simd") && dp.speedup > 1.0)
            .map(|dp| dp.speedup)
            .collect();

        if !simd_speedups.is_empty() {
            let avg_simd = simd_speedups.iter().sum::<f64>() / simd_speedups.len() as f64;
            let max_simd = simd_speedups.iter().fold(0.0f64, |a, &b| a.max(b));
            writeln!(file, "  SIMD Average Speedup: {:.2}x", avg_simd)?;
            writeln!(file, "  SIMD Maximum Speedup: {:.2}x", max_simd)?;
        }

        let parallel_speedups: Vec<f64> = self
            .data_points
            .iter()
            .filter(|dp| dp.test_name.contains("parallel") && dp.speedup > 1.0)
            .map(|dp| dp.speedup)
            .collect();

        if !parallel_speedups.is_empty() {
            let avg_parallel =
                parallel_speedups.iter().sum::<f64>() / parallel_speedups.len() as f64;
            let max_parallel = parallel_speedups.iter().fold(0.0f64, |a, &b| a.max(b));
            writeln!(file, "  Parallel Average Speedup: {:.2}x", avg_parallel)?;
            writeln!(file, "  Parallel Maximum Speedup: {:.2}x", max_parallel)?;
        }

        // Memory statistics
        let total_memory: f64 = self.data_points.iter().map(|dp| dp.memory_mb).sum();
        let avg_memory = total_memory / self.data_points.len() as f64;
        let max_memory = self
            .data_points
            .iter()
            .map(|dp| dp.memory_mb)
            .fold(0.0f64, f64::max);

        writeln!(file, "  Average Memory Usage: {:.2} MB", avg_memory)?;
        writeln!(file, "  Peak Memory Usage: {:.2} MB", max_memory)?;

        // Throughput statistics
        let max_throughput = self
            .data_points
            .iter()
            .map(|dp| dp.throughput_ops_per_sec)
            .fold(0.0f64, f64::max);

        writeln!(file, "  Peak Throughput: {:.0} ops/sec", max_throughput)?;

        writeln!(file)?;

        // Data size analysis
        writeln!(file, "Data Size Analysis:")?;
        let data_sizes: Vec<usize> = self.data_points.iter().map(|dp| dp.data_size).collect();
        let min_size = data_sizes.iter().min().unwrap_or(&0);
        let max_size = data_sizes.iter().max().unwrap_or(&0);
        writeln!(
            file,
            "  Tested Size Range: {} - {} points",
            min_size, max_size
        )?;

        // Find optimal size ranges
        let small_data_perf: Vec<&PerformanceDataPoint> = self
            .data_points
            .iter()
            .filter(|dp| dp.data_size <= 1000)
            .collect();

        let large_data_perf: Vec<&PerformanceDataPoint> = self
            .data_points
            .iter()
            .filter(|dp| dp.data_size >= 10000)
            .collect();

        if !small_data_perf.is_empty() {
            let avg_small_speedup = small_data_perf.iter().map(|dp| dp.speedup).sum::<f64>()
                / small_data_perf.len() as f64;
            writeln!(
                file,
                "  Small Data (<= 1,000): Average speedup {:.2}x",
                avg_small_speedup
            )?;
        }

        if !large_data_perf.is_empty() {
            let avg_large_speedup = large_data_perf.iter().map(|dp| dp.speedup).sum::<f64>()
                / large_data_perf.len() as f64;
            writeln!(
                file,
                "  Large Data (>= 10,000): Average speedup {:.2}x",
                avg_large_speedup
            )?;
        }

        writeln!(file)?;

        Ok(())
    }

    /// Generate recommendations based on performance data
    fn generate_recommendations(&self) -> io::Result<()> {
        let recommendations_path = format!("{}/recommendations.txt", self.output_dir);
        let mut file = File::create(recommendations_path)?;

        writeln!(file, "PERFORMANCE OPTIMIZATION RECOMMENDATIONS")?;
        writeln!(file, "========================================")?;
        writeln!(file)?;

        // Analyze SIMD effectiveness
        let simd_effective = self
            .data_points
            .iter()
            .filter(|dp| dp.test_name.contains("simd"))
            .any(|dp| dp.speedup > 1.5);

        if simd_effective {
            writeln!(file, "✓ SIMD Optimizations:")?;
            writeln!(
                file,
                "  • SIMD implementations show significant performance gains"
            )?;
            writeln!(
                file,
                "  • Recommended for distance calculations with large datasets"
            )?;
            writeln!(
                file,
                "  • Best performance with vector dimensions that are multiples of SIMD width"
            )?;
        } else {
            writeln!(file, "⚠ SIMD Optimizations:")?;
            writeln!(file, "  • SIMD speedup is limited on this architecture")?;
            writeln!(
                file,
                "  • Consider scalar implementations for small datasets"
            )?;
        }
        writeln!(file)?;

        // Analyze parallel effectiveness
        let parallel_effective = self
            .data_points
            .iter()
            .filter(|dp| dp.test_name.contains("parallel"))
            .any(|dp| dp.speedup > 2.0);

        if parallel_effective {
            writeln!(file, "✓ Parallel Processing:")?;
            writeln!(file, "  • Parallel implementations provide good speedup")?;
            writeln!(file, "  • Recommended for datasets with > 1,000 points")?;
            writeln!(
                file,
                "  • Optimal thread count appears to be {}",
                num_cpus::get()
            )?;
        } else {
            writeln!(file, "⚠ Parallel Processing:")?;
            writeln!(
                file,
                "  • Parallel overhead may outweigh benefits for small datasets"
            )?;
            writeln!(
                file,
                "  • Consider sequential processing for datasets < 500 points"
            )?;
        }
        writeln!(file)?;

        // Memory recommendations
        let high_memory = self.data_points.iter().any(|dp| dp.memory_mb > 100.0);

        writeln!(file, "Memory Usage Guidelines:")?;
        if high_memory {
            writeln!(file, "  ⚠ High memory usage detected for large datasets")?;
            writeln!(
                file,
                "  • Consider chunked processing for datasets > 10,000 points"
            )?;
            writeln!(
                file,
                "  • Use condensed distance matrices to reduce memory by 50%"
            )?;
            writeln!(file, "  • Monitor memory usage in production environments")?;
        } else {
            writeln!(
                file,
                "  ✓ Memory usage is reasonable for tested dataset sizes"
            )?;
            writeln!(
                file,
                "  • Current implementations should work well within available memory"
            )?;
        }
        writeln!(file)?;

        // Algorithm selection recommendations
        writeln!(file, "Algorithm Selection Guide:")?;
        writeln!(
            file,
            "  • For point queries: Use KDTree (low-D) or BallTree (high-D)"
        )?;
        writeln!(
            file,
            "  • For distance matrices: Use parallel_pdist with appropriate metric"
        )?;
        writeln!(
            file,
            "  • For nearest neighbors: Use simd_knn_search for best performance"
        )?;
        writeln!(
            file,
            "  • For large datasets: Consider approximate algorithms (LSH, sampling)"
        )?;
        writeln!(file)?;

        // Data size specific recommendations
        writeln!(file, "Data Size Specific Recommendations:")?;
        writeln!(file, "  Small (< 1,000 points):")?;
        writeln!(file, "    • Standard algorithms perform well")?;
        writeln!(file, "    • SIMD benefits may be minimal")?;
        writeln!(file, "    • Memory usage typically < 10 MB")?;
        writeln!(file)?;
        writeln!(file, "  Medium (1,000 - 10,000 points):")?;
        writeln!(file, "    • SIMD and parallel processing recommended")?;
        writeln!(file, "    • Consider condensed distance matrices")?;
        writeln!(file, "    • Memory usage 10-100 MB range")?;
        writeln!(file)?;
        writeln!(file, "  Large (> 10,000 points):")?;
        writeln!(file, "    • Chunked processing recommended")?;
        writeln!(file, "    • Use spatial data structures for queries")?;
        writeln!(file, "    • Consider approximate algorithms")?;
        writeln!(file, "    • Monitor memory usage carefully")?;
        writeln!(file)?;

        Ok(())
    }

    /// Export data to CSV for further analysis
    pub fn export_to_csv(&self) -> io::Result<()> {
        let csv_path = format!("{}/benchmark_data.csv", self.output_dir);
        let mut file = File::create(csv_path)?;

        writeln!(file, "test_name,data_size,dimensions,duration_ms,throughput_ops_per_sec,memory_mb,speedup,metric_type")?;

        for dp in &self.data_points {
            writeln!(
                file,
                "{},{},{},{:.3},{:.2},{:.2},{:.3},{}",
                dp.test_name,
                dp.data_size,
                dp.dimensions,
                dp.duration_ms,
                dp.throughput_ops_per_sec,
                dp.memory_mb,
                dp.speedup,
                dp.metric_type
            )?;
        }

        Ok(())
    }
}

/// Create sample performance data for testing
pub fn create_sample_data() -> Vec<PerformanceDataPoint> {
    vec![
        PerformanceDataPoint {
            test_name: "scalar_euclidean".to_string(),
            data_size: 1000,
            dimensions: 5,
            duration_ms: 100.0,
            throughput_ops_per_sec: 10000.0,
            memory_mb: 5.0,
            speedup: 1.0,
            metric_type: "euclidean".to_string(),
        },
        PerformanceDataPoint {
            test_name: "simd_euclidean".to_string(),
            data_size: 1000,
            dimensions: 5,
            duration_ms: 40.0,
            throughput_ops_per_sec: 25000.0,
            memory_mb: 5.0,
            speedup: 2.5,
            metric_type: "euclidean".to_string(),
        },
        PerformanceDataPoint {
            test_name: "parallel_pdist".to_string(),
            data_size: 5000,
            dimensions: 5,
            duration_ms: 500.0,
            throughput_ops_per_sec: 50000.0,
            memory_mb: 95.0,
            speedup: 4.0,
            metric_type: "euclidean".to_string(),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_performance_report_generator() {
        let temp_dir = env::temp_dir().join("scirs2_spatial_test_reports");
        let generator_result = PerformanceReportGenerator::new(temp_dir.to_str().unwrap());

        assert!(generator_result.is_ok());

        let mut generator = generator_result.unwrap();

        // Add sample data
        for dp in create_sample_data() {
            generator.add_data_point(dp);
        }

        // Generate report (should not panic)
        let result = generator.generate_report();
        assert!(result.is_ok());

        // Export CSV
        let csv_result = generator.export_to_csv();
        assert!(csv_result.is_ok());

        // Clean up
        let _ = std::fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_sample_data_creation() {
        let data = create_sample_data();
        assert!(!data.is_empty());
        assert!(data.iter().any(|dp| dp.speedup > 1.0));
    }
}
