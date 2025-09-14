//! Performance Comparison Tools for scirs2-spatial
//!
//! This module provides utilities for generating realistic spatial datasets,
//! comparing against reference implementations, and producing detailed
//! performance reports with charts and statistics.

use ndarray::Array2;
use rand::{rngs::StdRng, Rng, SeedableRng};
use scirs2_spatial::{
    distance::{euclidean, pdist},
    simd_distance::{parallel_pdist, simd_euclidean_distance_batch, simd_knn_search},
    BallTree, KDTree,
};
use std::hint::black_box;
use std::time::{Duration, Instant};

/// Dataset generator for realistic spatial data patterns
pub struct DatasetGenerator {
    seed: u64,
}

impl DatasetGenerator {
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Generate clustered data (common in real-world spatial applications)
    pub fn generate_clustered_data(
        &self,
        n_points: usize,
        n_clusters: usize,
        dimensions: usize,
        cluster_std: f64,
    ) -> Array2<f64> {
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut points = Array2::zeros((n_points, dimensions));

        // Generate cluster centers
        let cluster_centers: Vec<Vec<f64>> = (0..n_clusters)
            .map(|_| {
                (0..dimensions)
                    .map(|_| rng.gen_range(-50.0..50.0))
                    .collect()
            })
            .collect();

        // Assign points to clusters and add noise
        for i in 0..n_points {
            let cluster_idx = i % n_clusters;
            let center = &cluster_centers[cluster_idx];

            for j in 0..dimensions {
                points[[i, j]] = center[j] + rng.gen_range(-cluster_std..cluster_std);
            }
        }

        points
    }

    /// Generate uniformly distributed data
    pub fn generate_uniform_data(&self, npoints: usize, dimensions: usize) -> Array2<f64> {
        let mut rng = StdRng::seed_from_u64(self.seed);
        Array2::from_shape_fn((npoints, dimensions), |_| rng.gen_range(-100.0..100.0))
    }

    /// Generate data with outliers (common in spatial analysis)
    pub fn generate_data_with_outliers(
        &self,
        n_points: usize,
        dimensions: usize,
        outlier_fraction: f64,
    ) -> Array2<f64> {
        let mut rng = StdRng::seed_from_u64(self.seed);
        let n_outliers = (n_points as f64 * outlier_fraction) as usize;

        let mut points = Array2::zeros((n_points, dimensions));

        // Generate normal points
        for i in 0..(n_points - n_outliers) {
            for j in 0..dimensions {
                points[[i, j]] = rng.gen_range(-10.0..10.0);
            }
        }

        // Generate outliers
        for i in (n_points - n_outliers)..n_points {
            for j in 0..dimensions {
                points[[i, j]] = rng.gen_range(-100.0..100.0);
            }
        }

        points
    }

    /// Generate sparse high-dimensional data
    pub fn generate_sparse_data(
        &self,
        n_points: usize,
        dimensions: usize,
        sparsity: f64,
    ) -> Array2<f64> {
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut points = Array2::zeros((n_points, dimensions));

        for i in 0..n_points {
            for j in 0..dimensions {
                if rng.random::<f64>() > sparsity {
                    points[[i, j]] = rng.gen_range(-1.0..1.0);
                }
                // else remains 0.0 (sparse)
            }
        }

        points
    }
}

/// Performance metrics collector
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub name: String,
    pub duration: Duration,
    pub throughput_ops_per_sec: f64,
    pub memory_mb: f64,
    pub speedup_vs_baseline: f64,
}

/// Comprehensive performance analyzer
pub struct PerformanceAnalyzer {
    dataset_generator: DatasetGenerator,
    results: Vec<PerformanceMetrics>,
}

impl PerformanceAnalyzer {
    pub fn new(seed: u64) -> Self {
        Self {
            dataset_generator: DatasetGenerator::new(seed),
            results: Vec::new(),
        }
    }

    /// Benchmark SIMD vs scalar distance calculations
    pub fn benchmark_simd_vs_scalar(&mut self, sizes: &[usize], dimensions: &[usize]) {
        println!("=== SIMD vs Scalar Distance Calculation Benchmark ===");
        println!(
            "{:>8} {:>8} {:>12} {:>12} {:>12}",
            "Size", "Dim", "Scalar (ms)", "SIMD (ms)", "Speedup"
        );
        println!("{}", "-".repeat(60));

        for &size in sizes {
            for &dim in dimensions {
                let points1 = self.dataset_generator.generate_uniform_data(size, dim);
                let points2 = self.dataset_generator.generate_uniform_data(size, dim);

                // Scalar benchmark
                let start = Instant::now();
                for (row1, row2) in points1.outer_iter().zip(points2.outer_iter()) {
                    black_box(euclidean(
                        row1.as_slice().unwrap(),
                        row2.as_slice().unwrap(),
                    ));
                }
                let scalar_duration = start.elapsed();

                // SIMD batch benchmark
                let start = Instant::now();
                let _distances =
                    simd_euclidean_distance_batch(&points1.view(), &points2.view()).unwrap();
                let simd_duration = start.elapsed();

                let speedup = scalar_duration.as_secs_f64() / simd_duration.as_secs_f64();

                println!(
                    "{:>8} {:>8} {:>12.1} {:>12.1} {:>12.2}x",
                    size,
                    dim,
                    scalar_duration.as_millis(),
                    simd_duration.as_millis(),
                    speedup
                );

                // Store metrics
                self.results.push(PerformanceMetrics {
                    name: format!("scalar_distance_{size}x{dim}"),
                    duration: scalar_duration,
                    throughput_ops_per_sec: size as f64 / scalar_duration.as_secs_f64(),
                    memory_mb: (size * dim * 8 * 2) as f64 / (1024.0 * 1024.0),
                    speedup_vs_baseline: 1.0,
                });

                self.results.push(PerformanceMetrics {
                    name: format!("simd_distance_{size}x{dim}"),
                    duration: simd_duration,
                    throughput_ops_per_sec: size as f64 / simd_duration.as_secs_f64(),
                    memory_mb: (size * dim * 8 * 2) as f64 / (1024.0 * 1024.0),
                    speedup_vs_baseline: speedup,
                });
            }
        }
        println!();
    }

    /// Benchmark distance matrix computation scaling
    pub fn benchmark_distance_matrix_scaling(&mut self, sizes: &[usize]) {
        println!("=== Distance Matrix Computation Scaling ===");
        println!(
            "{:>8} {:>12} {:>12} {:>15} {:>12}",
            "Size", "Sequential", "Parallel", "Operations", "Speedup"
        );
        println!("{}", "-".repeat(70));

        for &size in sizes {
            let points = self.dataset_generator.generate_uniform_data(size, 5);
            let expected_ops = size * (size - 1) / 2;

            // Sequential benchmark
            let start = Instant::now();
            let _seq_distances = pdist(&points, euclidean);
            let sequential_duration = start.elapsed();

            // Parallel benchmark
            let start = Instant::now();
            let _par_distances = parallel_pdist(&points.view(), "euclidean").unwrap();
            let parallel_duration = start.elapsed();

            let speedup = sequential_duration.as_secs_f64() / parallel_duration.as_secs_f64();

            println!(
                "{:>8} {:>12} {:>12} {:>15} {:>12.2}x",
                size,
                sequential_duration.as_millis(),
                parallel_duration.as_millis(),
                expected_ops,
                speedup
            );

            // Store metrics
            self.results.push(PerformanceMetrics {
                name: format!("pdist_sequential_{size}"),
                duration: sequential_duration,
                throughput_ops_per_sec: expected_ops as f64 / sequential_duration.as_secs_f64(),
                memory_mb: (expected_ops * 8) as f64 / (1024.0 * 1024.0),
                speedup_vs_baseline: 1.0,
            });

            self.results.push(PerformanceMetrics {
                name: format!("pdist_parallel_{size}"),
                duration: parallel_duration,
                throughput_ops_per_sec: expected_ops as f64 / parallel_duration.as_secs_f64(),
                memory_mb: (expected_ops * 8) as f64 / (1024.0 * 1024.0),
                speedup_vs_baseline: speedup,
            });
        }
        println!();
    }

    /// Benchmark different distance metrics
    pub fn benchmark_distance_metrics(&mut self, size: usize, dim: usize) {
        println!("=== Distance Metrics Performance Comparison ===");
        println!("{:>12} {:>12} {:>15}", "Metric", "Time (ms)", "Rel. Speed");
        println!("{}", "-".repeat(42));

        let points = self.dataset_generator.generate_uniform_data(size, dim);
        let metrics = ["euclidean", "manhattan", "chebyshev"];
        let mut base_time = Duration::default();

        for (i, &metric) in metrics.iter().enumerate() {
            let start = Instant::now();
            let _distances = parallel_pdist(&points.view(), metric).unwrap();
            let duration = start.elapsed();

            if i == 0 {
                base_time = duration;
            }

            let relative_speed = base_time.as_secs_f64() / duration.as_secs_f64();

            println!(
                "{:>12} {:>12} {:>15.2}x",
                metric,
                duration.as_millis(),
                relative_speed
            );

            // Store metrics
            self.results.push(PerformanceMetrics {
                name: format!("metric_{metric}_{size}"),
                duration,
                throughput_ops_per_sec: (size * (size - 1) / 2) as f64 / duration.as_secs_f64(),
                memory_mb: (size * dim * 8) as f64 / (1024.0 * 1024.0),
                speedup_vs_baseline: relative_speed,
            });
        }
        println!();
    }

    /// Benchmark spatial data structures
    pub fn benchmark_spatial_structures(&mut self, sizes: &[usize]) {
        println!("=== Spatial Data Structures Performance ===");
        println!(
            "{:>8} {:>15} {:>15} {:>15}",
            "Size", "KDTree Build", "BallTree Build", "KDTree Query"
        );
        println!("{}", "-".repeat(60));

        for &size in sizes {
            let points = self.dataset_generator.generate_uniform_data(size, 3);
            let query_points = self.dataset_generator.generate_uniform_data(100, 3);

            // KDTree construction
            let start = Instant::now();
            let kdtree = KDTree::new(&points).unwrap();
            let kdtree_build_time = start.elapsed();

            // BallTree construction
            let start = Instant::now();
            let _balltree = BallTree::with_euclidean_distance(&points.view(), 10).unwrap();
            let balltree_build_time = start.elapsed();

            // KDTree queries
            let start = Instant::now();
            for query in query_points.outer_iter() {
                black_box(kdtree.query(query.as_slice().unwrap(), 5).unwrap());
            }
            let kdtree_query_time = start.elapsed();

            println!(
                "{:>8} {:>15} {:>15} {:>15}",
                size,
                kdtree_build_time.as_millis(),
                balltree_build_time.as_millis(),
                kdtree_query_time.as_millis()
            );

            // Store metrics
            self.results.push(PerformanceMetrics {
                name: format!("kdtree_build_{size}"),
                duration: kdtree_build_time,
                throughput_ops_per_sec: size as f64 / kdtree_build_time.as_secs_f64(),
                memory_mb: (size * 3 * 8) as f64 / (1024.0 * 1024.0),
                speedup_vs_baseline: 1.0,
            });
        }
        println!();
    }

    /// Benchmark KNN search performance
    pub fn benchmark_knn_search(
        &mut self,
        data_size: usize,
        query_size: usize,
        k_values: &[usize],
    ) {
        println!("=== K-Nearest Neighbors Search Performance ===");
        println!("{:>6} {:>12} {:>15}", "k", "Time (ms)", "Throughput (q/s)");
        println!("{}", "-".repeat(35));

        let data_points = self.dataset_generator.generate_uniform_data(data_size, 5);
        let query_points = self.dataset_generator.generate_uniform_data(query_size, 5);

        for &k in k_values {
            let start = Instant::now();
            let (_indicesdistances) =
                simd_knn_search(&query_points.view(), &data_points.view(), k, "euclidean").unwrap();
            let duration = start.elapsed();

            let throughput = query_size as f64 / duration.as_secs_f64();

            println!("{:>6} {:>12} {:>15.0}", k, duration.as_millis(), throughput);

            // Store metrics
            self.results.push(PerformanceMetrics {
                name: format!("knn_k{k}_{data_size}"),
                duration,
                throughput_ops_per_sec: throughput,
                memory_mb: ((data_size + query_size) * 5 * 8) as f64 / (1024.0 * 1024.0),
                speedup_vs_baseline: 1.0,
            });
        }
        println!();
    }

    /// Test different data patterns
    pub fn benchmark_data_patterns(&mut self, size: usize, dim: usize) {
        println!("=== Performance on Different Data Patterns ===");
        println!("{:>15} {:>12} {:>15}", "Pattern", "Time (ms)", "Relative");
        println!("{}", "-".repeat(45));

        let patterns = [
            (
                "uniform",
                self.dataset_generator.generate_uniform_data(size, dim),
            ),
            (
                "clustered",
                self.dataset_generator
                    .generate_clustered_data(size, 5, dim, 2.0),
            ),
            (
                "with_outliers",
                self.dataset_generator
                    .generate_data_with_outliers(size, dim, 0.1),
            ),
            (
                "sparse",
                self.dataset_generator.generate_sparse_data(size, dim, 0.8),
            ),
        ];

        let mut base_time = Duration::default();

        for (i, (pattern_name, points)) in patterns.iter().enumerate() {
            let start = Instant::now();
            let _distances = parallel_pdist(&points.view(), "euclidean").unwrap();
            let duration = start.elapsed();

            if i == 0 {
                base_time = duration;
            }

            let relative_time = duration.as_secs_f64() / base_time.as_secs_f64();

            println!(
                "{:>15} {:>12} {:>15.2}x",
                pattern_name,
                duration.as_millis(),
                relative_time
            );

            // Store metrics
            self.results.push(PerformanceMetrics {
                name: format!("pattern_{pattern_name}_{size}"),
                duration,
                throughput_ops_per_sec: (size * (size - 1) / 2) as f64 / duration.as_secs_f64(),
                memory_mb: (size * dim * 8) as f64 / (1024.0 * 1024.0),
                speedup_vs_baseline: 1.0 / relative_time,
            });
        }
        println!();
    }

    /// Generate comprehensive performance report
    pub fn generate_report(&self) {
        println!("=== Comprehensive Performance Report ===");

        // Architecture information
        println!("\nSystem Information:");
        #[cfg(target_arch = "x86_64")]
        {
            println!("  Architecture: x86_64");
            println!("  SSE2: {}", is_x86_feature_detected!("sse2"));
            println!("  AVX: {}", is_x86_feature_detected!("avx"));
            println!("  AVX2: {}", is_x86_feature_detected!("avx2"));
            println!("  AVX-512F: {}", is_x86_feature_detected!("avx512f"));
        }

        #[cfg(target_arch = "aarch64")]
        {
            println!("  Architecture: aarch64");
            println!(
                "  NEON: {}",
                std::arch::is_aarch64_feature_detected!("neon")
            );
        }

        // Performance summary
        println!("\nPerformance Summary:");

        let simd_speedups: Vec<f64> = self
            .results
            .iter()
            .filter(|r| r.name.contains("simd"))
            .map(|r| r.speedup_vs_baseline)
            .collect();

        if !simd_speedups.is_empty() {
            let avg_simd_speedup = simd_speedups.iter().sum::<f64>() / simd_speedups.len() as f64;
            let max_simd_speedup = simd_speedups.iter().fold(0.0f64, |a, &b| a.max(b));
            println!("  Average SIMD speedup: {avg_simd_speedup:.2}x");
            println!("  Maximum SIMD speedup: {max_simd_speedup:.2}x");
        }

        let parallel_speedups: Vec<f64> = self
            .results
            .iter()
            .filter(|r| r.name.contains("parallel"))
            .map(|r| r.speedup_vs_baseline)
            .collect();

        if !parallel_speedups.is_empty() {
            let avg_parallel_speedup =
                parallel_speedups.iter().sum::<f64>() / parallel_speedups.len() as f64;
            let max_parallel_speedup = parallel_speedups.iter().fold(0.0f64, |a, &b| a.max(b));
            println!("  Average parallel speedup: {avg_parallel_speedup:.2}x");
            println!("  Maximum parallel speedup: {max_parallel_speedup:.2}x");
        }

        // Memory efficiency
        let total_memory: f64 = self.results.iter().map(|r| r.memory_mb).sum();
        println!("  Total memory processed: {total_memory:.1} MB");

        // Recommendations
        println!("\nRecommendations:");

        if simd_speedups.iter().any(|&s| s > 1.5) {
            println!("  ✓ SIMD implementation shows significant performance gains");
            println!("  → Use SIMD functions for large-scale distance computations");
        }

        if parallel_speedups.iter().any(|&s| s > 2.0) {
            println!("  ✓ Parallel implementation provides good speedup");
            println!("  → Use parallel functions for datasets with >1000 points");
        }

        println!("  → For datasets <100 points, scalar implementations may be sufficient");
        println!("  → For high-dimensional data (>50D), consider dimensionality reduction");
        println!("  → Use appropriate data structures (KDTree for low-D, BallTree for high-D)");

        println!("\n========================================");
    }

    /// Export results to CSV for further analysis
    pub fn export_to_csv(&self, filename: &str) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(filename)?;
        writeln!(
            file,
            "name,duration_ms,throughput_ops_per_sec,memory_mb,speedup_vs_baseline"
        )?;

        for metric in &self.results {
            writeln!(
                file,
                "{},{:.3},{:.2},{:.2},{:.3}",
                metric.name,
                metric.duration.as_millis(),
                metric.throughput_ops_per_sec,
                metric.memory_mb,
                metric.speedup_vs_baseline
            )?;
        }

        Ok(())
    }
}

/// Memory usage analyzer
pub struct MemoryAnalyzer;

impl MemoryAnalyzer {
    /// Analyze memory allocation patterns for different operations
    pub fn analyze_memory_patterns() {
        println!("=== Memory Allocation Pattern Analysis ===");

        let sizes = [1000, 5000, 10000];

        println!(
            "{:>8} {:>15} {:>15} {:>15}",
            "Size", "Points (MB)", "Distance Matrix", "Peak Usage"
        );
        println!("{}", "-".repeat(60));

        for &size in &sizes {
            let points_memory = (size * 3 * std::mem::size_of::<f64>()) as f64 / (1024.0 * 1024.0);
            let dist_matrix_memory =
                (size * (size - 1) / 2 * std::mem::size_of::<f64>()) as f64 / (1024.0 * 1024.0);
            let peak_memory = points_memory + dist_matrix_memory;

            println!(
                "{size:>8} {points_memory:>15.2} {dist_matrix_memory:>15.2} {peak_memory:>15.2}"
            );
        }

        println!("\nMemory Efficiency Guidelines:");
        println!("  • For n > 10,000: Consider chunked processing");
        println!("  • For distance matrices: n > 5,000 may require >100MB");
        println!("  • Use condensed distance matrices to save 50% memory");
        println!("  • Consider streaming algorithms for very large datasets");
        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_generator() {
        let generator = DatasetGenerator::new(42);

        // Test uniform data generation
        let uniform_data = generator.generate_uniform_data(100, 5);
        assert_eq!(uniform_data.shape(), [100, 5]);

        // Test clustered data generation
        let clustered_data = generator.generate_clustered_data(100, 3, 5, 1.0);
        assert_eq!(clustered_data.shape(), [100, 5]);

        // Test data with outliers
        let outlier_data = generator.generate_data_with_outliers(100, 5, 0.1);
        assert_eq!(outlier_data.shape(), [100, 5]);
    }

    #[test]
    fn test_performance_analyzer() {
        let mut analyzer = PerformanceAnalyzer::new(42);

        // Test SIMD vs scalar benchmark
        analyzer.benchmark_simd_vs_scalar(&[100], &[5]);
        assert!(!analyzer.results.is_empty());

        // Test distance matrix scaling
        analyzer.benchmark_distance_matrix_scaling(&[100]);

        // Verify we have results
        assert!(analyzer.results.len() >= 2);
    }

    #[test]
    fn test_memory_analyzer() {
        // This should not panic
        MemoryAnalyzer::analyze_memory_patterns();
    }
}
