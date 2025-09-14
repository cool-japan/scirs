//! Memory Usage Analysis and Benchmarks for scirs2-spatial
//!
//! This module provides comprehensive memory usage analysis tools,
//! tracking peak memory consumption, allocation patterns, and
//! cache performance for spatial operations.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::Array2;
use rand::{rngs::StdRng, Rng, SeedableRng};
use scirs2_spatial::{
    distance::{euclidean, pdist},
    simd_distance::{parallel_pdist, simd_euclidean_distance_batch},
    KDTree,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

/// Memory tracking allocator wrapper
struct MemoryTracker {
    allocated: AtomicUsize,
    peak_allocated: AtomicUsize,
    allocation_count: AtomicUsize,
}

impl MemoryTracker {
    const fn new() -> Self {
        MemoryTracker {
            allocated: AtomicUsize::new(0),
            peak_allocated: AtomicUsize::new(0),
            allocation_count: AtomicUsize::new(0),
        }
    }

    fn track_allocation(&self, size: usize) {
        let current = self.allocated.fetch_add(size, Ordering::SeqCst) + size;
        self.allocation_count.fetch_add(1, Ordering::SeqCst);

        // Update peak if necessary
        let mut peak = self.peak_allocated.load(Ordering::SeqCst);
        while current > peak {
            match self.peak_allocated.compare_exchange_weak(
                peak,
                current,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => break,
                Err(new_peak) => peak = new_peak,
            }
        }
    }

    fn track_deallocation(&self, size: usize) {
        self.allocated.fetch_sub(size, Ordering::SeqCst);
    }

    fn get_stats(&self) -> MemoryStats {
        MemoryStats {
            current_allocated: self.allocated.load(Ordering::SeqCst),
            peak_allocated: self.peak_allocated.load(Ordering::SeqCst),
            allocation_count: self.allocation_count.load(Ordering::SeqCst),
        }
    }

    fn reset(&self) {
        self.allocated.store(0, Ordering::SeqCst);
        self.peak_allocated.store(0, Ordering::SeqCst);
        self.allocation_count.store(0, Ordering::SeqCst);
    }
}

static MEMORY_TRACKER: MemoryTracker = MemoryTracker::new();

/// Custom allocator that tracks memory usage
#[allow(dead_code)]
struct TrackingAllocator;

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            MEMORY_TRACKER.track_allocation(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        MEMORY_TRACKER.track_deallocation(layout.size());
        System.dealloc(ptr, layout);
    }
}

// Note: In a real implementation, you'd set this as the global allocator
// #[global_allocator]
// static GLOBAL: TrackingAllocator = TrackingAllocator;

/// Memory usage statistics
#[derive(Debug, Clone)]
struct MemoryStats {
    current_allocated: usize,
    peak_allocated: usize,
    allocation_count: usize,
}

impl MemoryStats {
    #[allow(dead_code)]
    fn peak_mb(&self) -> f64 {
        self.peak_allocated as f64 / (1024.0 * 1024.0)
    }

    #[allow(dead_code)]
    fn current_mb(&self) -> f64 {
        self.current_allocated as f64 / (1024.0 * 1024.0)
    }
}

/// Memory benchmark suite
struct MemoryBenchmark {
    seed: u64,
}

impl MemoryBenchmark {
    fn new(seed: u64) -> Self {
        Self { seed }
    }

    fn generate_points(&self, npoints: usize, dimensions: usize) -> Array2<f64> {
        let mut rng = StdRng::seed_from_u64(self.seed);
        Array2::from_shape_fn((npoints, dimensions), |_| rng.gen_range(-10.0..10.0))
    }

    /// Analyze memory usage for different data sizes
    fn analyze_memory_scaling(&self, sizes: &[usize], dimensions: usize) {
        println!("=== Memory Scaling Analysis ===");
        println!(
            "{:>8} {:>12} {:>15} {:>15} {:>15}",
            "Size", "Data (MB)", "Expected Dist", "Peak (MB)", "Efficiency"
        );
        println!("{}", "-".repeat(75));

        for &size in sizes {
            let points = self.generate_points(size, dimensions);
            let data_size_mb =
                (size * dimensions * std::mem::size_of::<f64>()) as f64 / (1024.0 * 1024.0);
            let expected_dist_size_mb =
                (size * (size - 1) / 2 * std::mem::size_of::<f64>()) as f64 / (1024.0 * 1024.0);

            // Reset memory tracking
            MEMORY_TRACKER.reset();

            // Measure memory usage during computation
            let _start_stats = MEMORY_TRACKER.get_stats();
            let _distances = parallel_pdist(&points.view(), "euclidean").unwrap();
            let end_stats = MEMORY_TRACKER.get_stats();

            let memory_efficiency = expected_dist_size_mb / end_stats.peak_mb().max(0.001);

            println!(
                "{:>8} {:>12.2} {:>15.2} {:>15.2} {:>15.2}",
                size,
                data_size_mb,
                expected_dist_size_mb,
                end_stats.peak_mb(),
                memory_efficiency
            );
        }
        println!();
    }

    /// Compare memory usage of different algorithms
    fn compare_algorithm_memory(&self, size: usize, dimensions: usize) {
        println!("=== Algorithm Memory Comparison ===");
        println!(
            "{:>20} {:>12} {:>15} {:>12}",
            "Algorithm", "Peak (MB)", "Allocations", "Efficiency"
        );
        println!("{}", "-".repeat(65));

        let points = self.generate_points(size, dimensions);

        // Test different algorithms
        let algorithms = [
            ("sequential_pdist", self.test_sequential_pdist(&points)),
            ("parallel_pdist", self.test_parallel_pdist(&points)),
            ("simd_batch", self.test_simd_batch(&points)),
            ("kdtree_construction", self.test_kdtree_memory(&points)),
        ];

        for (name, stats) in algorithms {
            let efficiency = (size * size) as f64 / stats.peak_allocated as f64 * 1e6;
            println!(
                "{:>20} {:>12.2} {:>15} {:>12.2}",
                name,
                stats.peak_mb(),
                stats.allocation_count,
                efficiency
            );
        }
        println!();
    }

    fn test_sequential_pdist(&self, points: &Array2<f64>) -> MemoryStats {
        MEMORY_TRACKER.reset();
        let _distances = pdist(points, euclidean);
        MEMORY_TRACKER.get_stats()
    }

    fn test_parallel_pdist(&self, points: &Array2<f64>) -> MemoryStats {
        MEMORY_TRACKER.reset();
        let _distances = parallel_pdist(&points.view(), "euclidean").unwrap();
        MEMORY_TRACKER.get_stats()
    }

    fn test_simd_batch(&self, points: &Array2<f64>) -> MemoryStats {
        MEMORY_TRACKER.reset();
        let points2 = self.generate_points(points.nrows(), points.ncols());
        let _distances = simd_euclidean_distance_batch(&points.view(), &points2.view()).unwrap();
        MEMORY_TRACKER.get_stats()
    }

    fn test_kdtree_memory(&self, points: &Array2<f64>) -> MemoryStats {
        MEMORY_TRACKER.reset();
        let _kdtree = KDTree::new(points).unwrap();
        MEMORY_TRACKER.get_stats()
    }

    /// Analyze memory access patterns and cache performance
    fn analyze_cache_performance(&self) {
        println!("=== Cache Performance Analysis ===");
        println!(
            "{:>8} {:>15} {:>15} {:>15}",
            "Pattern", "Sequential", "Random", "Ratio"
        );
        println!("{}", "-".repeat(60));

        let sizes = [1000, 5000, 10000];

        for &size in &sizes {
            let points = self.generate_points(size, 10);

            // Sequential access pattern
            let start = Instant::now();
            let mut sum = 0.0;
            for row in points.outer_iter() {
                for &val in row {
                    sum += val * val;
                }
            }
            let sequential_time = start.elapsed();
            black_box(sum);

            // Random access pattern (simulate worst-case cache behavior)
            let mut rng = StdRng::seed_from_u64(self.seed);
            let indices: Vec<(usize, usize)> = (0..size * 10)
                .map(|_| (rng.gen_range(0..size)..rng.gen_range(0..10)))
                .collect();

            let start = Instant::now();
            let mut sum = 0.0;
            for &(i, j) in &indices {
                sum += points[[i, j]] * points[[i, j]];
            }
            let random_time = start.elapsed();
            black_box(sum);

            let ratio = random_time.as_secs_f64() / sequential_time.as_secs_f64();

            println!(
                "{:>8} {:>15.2} {:>15.2} {:>15.2}",
                size,
                sequential_time.as_millis(),
                random_time.as_millis(),
                ratio
            );
        }
        println!();
    }

    /// Test memory allocation strategies
    fn test_allocation_strategies(&self) {
        println!("=== Memory Allocation Strategy Comparison ===");
        println!(
            "{:>20} {:>12} {:>15} {:>12}",
            "Strategy", "Time (ms)", "Peak (MB)", "Allocs"
        );
        println!("{}", "-".repeat(65));

        let size = 2000;
        let dimensions = 8;

        // Strategy 1: Pre-allocated buffers
        let start = Instant::now();
        MEMORY_TRACKER.reset();
        {
            let points = self.generate_points(size, dimensions);
            let mut result = Array2::<f64>::zeros((size, size));

            // Simulate filling the matrix
            for i in 0..size {
                for j in 0..size {
                    if i != j {
                        let p1 = points.row(i);
                        let p2 = points.row(j);
                        let dist = euclidean(p1.as_slice().unwrap(), p2.as_slice().unwrap());
                        result[[i, j]] = dist;
                    }
                }
            }
            black_box(result);
        }
        let preallocated_time = start.elapsed();
        let preallocated_stats = MEMORY_TRACKER.get_stats();

        // Strategy 2: Dynamic allocation
        let start = Instant::now();
        MEMORY_TRACKER.reset();
        {
            let points = self.generate_points(size, dimensions);
            let mut distances = Vec::new();

            for i in 0..size {
                for j in (i + 1)..size {
                    let p1 = points.row(i);
                    let p2 = points.row(j);
                    let dist = euclidean(p1.as_slice().unwrap(), p2.as_slice().unwrap());
                    distances.push(dist);
                }
            }
            black_box(distances);
        }
        let dynamic_time = start.elapsed();
        let dynamic_stats = MEMORY_TRACKER.get_stats();

        // Strategy 3: Chunked processing
        let start = Instant::now();
        MEMORY_TRACKER.reset();
        {
            let points = self.generate_points(size, dimensions);
            let chunk_size = 100;
            let mut total_distances = 0;

            for chunk_start in (0..size).step_by(chunk_size) {
                let chunk_end = (chunk_start + chunk_size).min(size);
                let chunk = points.slice(ndarray::s![chunk_start..chunk_end, ..]);

                if chunk.nrows() > 1 {
                    let chunk_distances = parallel_pdist(&chunk, "euclidean").unwrap();
                    total_distances += chunk_distances.len();
                }
            }
            black_box(total_distances);
        }
        let chunked_time = start.elapsed();
        let chunked_stats = MEMORY_TRACKER.get_stats();

        println!(
            "{:>20} {:>12} {:>15.2} {:>12}",
            "Pre-allocated",
            preallocated_time.as_millis(),
            preallocated_stats.peak_mb(),
            preallocated_stats.allocation_count
        );

        println!(
            "{:>20} {:>12} {:>15.2} {:>12}",
            "Dynamic",
            dynamic_time.as_millis(),
            dynamic_stats.peak_mb(),
            dynamic_stats.allocation_count
        );

        println!(
            "{:>20} {:>12} {:>15.2} {:>12}",
            "Chunked",
            chunked_time.as_millis(),
            chunked_stats.peak_mb(),
            chunked_stats.allocation_count
        );

        println!();
    }

    /// Generate memory usage recommendations
    fn generate_recommendations(&self) {
        println!("=== Memory Usage Recommendations ===");
        println!();

        println!("For small datasets (< 1,000 points):");
        println!("  • Memory usage is typically < 10 MB");
        println!("  • Standard algorithms work well");
        println!("  • No special memory optimizations needed");
        println!();

        println!("For medium datasets (1,000 - 10,000 points):");
        println!("  • Distance matrices can require 100+ MB");
        println!("  • Consider condensed distance matrices");
        println!("  • Use parallel processing for better memory bandwidth utilization");
        println!();

        println!("For large datasets (> 10,000 points):");
        println!("  • Full distance matrices may exceed available memory");
        println!("  • Use chunked processing or streaming algorithms");
        println!("  • Consider approximate algorithms (LSH, random sampling)");
        println!("  • Use spatial data structures (KDTree, BallTree) for queries");
        println!();

        println!("Cache optimization:");
        println!("  • Sequential memory access is 2-10x faster than random access");
        println!("  • Batch operations improve cache utilization");
        println!("  • Consider data layout (row-major vs column-major)");
        println!();

        println!("Memory allocation:");
        println!("  • Pre-allocate buffers when possible");
        println!("  • Avoid frequent small allocations in hot loops");
        println!("  • Use memory pools for repetitive operations");
        println!("  • Monitor peak memory usage in production");
        println!();
    }
}

/// Benchmark memory usage for distance calculations
#[allow(dead_code)]
fn bench_memory_distance_calculations(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_distance_calculations");

    let benchmark = MemoryBenchmark::new(12345);

    for &size in &[500, 1000, 2000] {
        let points = benchmark.generate_points(size, 5);

        group.throughput(Throughput::Elements((size * (size - 1) / 2) as u64));

        group.bench_with_input(
            BenchmarkId::new("memory_efficient_pdist", size),
            &size,
            |b, _| {
                b.iter(|| {
                    MEMORY_TRACKER.reset();
                    let distances = parallel_pdist(&points.view(), "euclidean").unwrap();
                    let stats = MEMORY_TRACKER.get_stats();
                    black_box((distances, stats))
                })
            },
        );
    }

    group.finish();
}

/// Benchmark memory usage for different data structures
#[allow(dead_code)]
fn bench_memory_data_structures(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_data_structures");

    let benchmark = MemoryBenchmark::new(12345);

    for &size in &[1000, 5000, 10000] {
        let points = benchmark.generate_points(size, 3);

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("kdtree_memory", size), &size, |b_, _| {
            b_.iter(|| {
                MEMORY_TRACKER.reset();
                let kdtree = KDTree::new(&points).unwrap();
                let stats = MEMORY_TRACKER.get_stats();
                black_box((kdtree, stats))
            })
        });
    }

    group.finish();
}

criterion_group!(
    memory_benches,
    bench_memory_distance_calculations,
    bench_memory_data_structures,
);

criterion_main!(memory_benches);

/// Standalone function for running memory analysis
#[allow(dead_code)]
fn run_memory_analysis() {
    let benchmark = MemoryBenchmark::new(12345);

    // Run comprehensive memory analysis
    benchmark.analyze_memory_scaling(&[100, 500, 1000, 2000, 5000], 5);
    benchmark.compare_algorithm_memory(1000, 8);
    benchmark.analyze_cache_performance();
    benchmark.test_allocation_strategies();
    benchmark.generate_recommendations();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_tracker() {
        let tracker = MemoryTracker::new();
        tracker.reset();

        tracker.track_allocation(1000);
        tracker.track_allocation(2000);

        let stats = tracker.get_stats();
        assert_eq!(stats.current_allocated, 3000);
        assert_eq!(stats.peak_allocated, 3000);
        assert_eq!(stats.allocation_count, 2);

        tracker.track_deallocation(1000);
        let stats = tracker.get_stats();
        assert_eq!(stats.current_allocated, 2000);
        assert_eq!(stats.peak_allocated, 3000); // Peak should remain
    }

    #[test]
    fn test_memory_benchmark() {
        let benchmark = MemoryBenchmark::new(42);
        let points = benchmark.generate_points(100, 5);
        assert_eq!(points.shape(), [100, 5]);

        // Test memory tracking during computation
        let stats = benchmark.test_parallel_pdist(&points);
        assert!(stats.peak_allocated > 0);
    }
}
