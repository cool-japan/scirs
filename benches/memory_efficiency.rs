use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::{Array1, Array2, ArrayView2};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use scirs2_core::memory::{BufferPool, MemoryMetrics, MemorySnapshot};
use scirs2_core::memory_efficient::{chunk_wise_operation, ChunkProcessor};
use scirs2_linalg::{basic, decomposition, solve};
use std::sync::Arc;
use std::time::Instant;

const SEED: u64 = 42;
const MEMORY_TEST_SIZES: &[usize] = &[100, 500, 1000, 2000];

/// Memory usage tracker for benchmarks
struct MemoryTracker {
    metrics: MemoryMetrics,
    initial_snapshot: MemorySnapshot,
}

impl MemoryTracker {
    fn new() -> Self {
        let metrics = MemoryMetrics::new();
        let initial_snapshot = metrics.snapshot("initial");
        Self {
            metrics,
            initial_snapshot,
        }
    }

    fn measure<F, R>(&self, operation_name: &str, f: F) -> (R, MemorySnapshot)
    where
        F: FnOnce() -> R,
    {
        let result = f();
        let snapshot = self.metrics.snapshot(operation_name);
        (result, snapshot)
    }

    fn memory_used(&self, snapshot: &MemorySnapshot) -> f64 {
        (snapshot.current_usage.peak_bytes as f64
            - self.initial_snapshot.current_usage.peak_bytes as f64)
            / 1024.0
            / 1024.0
    }
}

/// Generate test data with specific memory characteristics
fn generate_memory_test_data(size: usize) -> Array2<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    Array2::random_using((size, size), Uniform::new(-1.0, 1.0), &mut rng)
}

/// Benchmark buffer pool efficiency
fn bench_buffer_pool_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("buffer_pool_efficiency");

    for &size in &[1024, 4096, 16384] {
        let pool = BufferPool::new(size, 10);

        group.bench_with_input(
            BenchmarkId::new("buffer_allocation", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let buffer = pool.get_buffer();
                    black_box(&buffer);
                    // Buffer is automatically returned to pool when dropped
                })
            },
        );

        group.bench_with_input(BenchmarkId::new("buffer_reuse", size), &size, |b, _| {
            b.iter(|| {
                // Test rapid allocation/deallocation
                for _ in 0..10 {
                    let buffer = pool.get_buffer();
                    black_box(&buffer);
                }
            })
        });
    }

    group.finish();
}

/// Benchmark chunked matrix operations
fn bench_chunked_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunked_operations");

    for &size in MEMORY_TEST_SIZES {
        let matrix = generate_memory_test_data(size);

        // Chunked matrix multiplication
        group.bench_with_input(BenchmarkId::new("chunked_matmul", size), &size, |b, _| {
            let chunk_size = (size / 4).max(64);
            b.iter(|| {
                let result =
                    chunk_wise_operation(&matrix.view(), chunk_size, |chunk: ArrayView2<f64>| {
                        chunk.dot(&chunk.t())
                    });
                black_box(result)
            })
        });

        // Compare with regular matrix multiplication
        group.bench_with_input(BenchmarkId::new("regular_matmul", size), &size, |b, _| {
            b.iter(|| {
                let result = matrix.dot(&matrix.t());
                black_box(result)
            })
        });

        // Chunked determinant computation
        if size <= 1000 {
            group.bench_with_input(BenchmarkId::new("chunked_det", size), &size, |b, _| {
                let chunk_size = (size / 2).max(50);
                b.iter(|| {
                    let result = chunk_wise_operation(
                        &matrix.view(),
                        chunk_size,
                        |chunk: ArrayView2<f64>| basic::det(&chunk).unwrap_or(0.0),
                    );
                    black_box(result)
                })
            });
        }
    }

    group.finish();
}

/// Benchmark memory usage patterns
fn bench_memory_usage_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage_patterns");

    for &size in &[500, 1000, 1500] {
        let matrix = generate_memory_test_data(size);
        let tracker = MemoryTracker::new();

        // In-place operations vs. copying operations
        group.bench_with_input(
            BenchmarkId::new("in_place_transpose", size),
            &size,
            |b, _| {
                b.iter_custom(|iters| {
                    let start = Instant::now();

                    for _ in 0..iters {
                        let mut matrix_copy = matrix.clone();
                        matrix_copy.swap_axes(0, 1); // In-place transpose
                        black_box(&matrix_copy);
                    }

                    start.elapsed()
                })
            },
        );

        group.bench_with_input(BenchmarkId::new("copy_transpose", size), &size, |b, _| {
            b.iter(|| {
                let result = matrix.t().to_owned(); // Creates a copy
                black_box(result)
            })
        });

        // Memory-efficient linear solve
        if size <= 1000 {
            let rhs = Array1::ones(size);

            group.bench_with_input(
                BenchmarkId::new("memory_efficient_solve", size),
                &size,
                |b, _| {
                    b.iter_custom(|iters| {
                        let (result, snapshot) = tracker.measure("solve", || {
                            let start = Instant::now();

                            for _ in 0..iters {
                                let _solution = solve::solve(&matrix.view(), &rhs.view());
                            }

                            start.elapsed()
                        });

                        // Track memory usage
                        let memory_used = tracker.memory_used(&snapshot);
                        if memory_used > 100.0 {
                            // More than 100MB
                            println!(
                                "High memory usage detected: {:.2} MB for size {}",
                                memory_used, size
                            );
                        }

                        result
                    })
                },
            );
        }
    }

    group.finish();
}

/// Benchmark memory allocation patterns
fn bench_allocation_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("allocation_patterns");

    // Pre-allocated vs dynamic allocation
    for &size in &[100, 500, 1000] {
        group.bench_with_input(
            BenchmarkId::new("preallocated_arrays", size),
            &size,
            |b, _| {
                // Pre-allocate arrays
                let mut matrices = Vec::with_capacity(10);
                for _ in 0..10 {
                    matrices.push(Array2::<f64>::zeros((size, size)));
                }

                b.iter(|| {
                    for matrix in &mut matrices {
                        matrix.fill(1.0);
                        black_box(matrix);
                    }
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("dynamic_allocation", size),
            &size,
            |b, _| {
                b.iter(|| {
                    for _ in 0..10 {
                        let matrix = Array2::<f64>::ones((size, size));
                        black_box(matrix);
                    }
                })
            },
        );
    }

    group.finish();
}

/// Benchmark memory fragmentation resistance
fn bench_fragmentation_resistance(c: &mut Criterion) {
    let mut group = c.benchmark_group("fragmentation_resistance");

    // Test with varying allocation sizes to simulate fragmentation
    let sizes = vec![64, 128, 256, 512, 1024];

    group.bench_function("mixed_size_allocations", |b| {
        b.iter(|| {
            let mut arrays = Vec::new();

            // Allocate arrays of different sizes
            for &size in &sizes {
                for _ in 0..5 {
                    arrays.push(Array2::<f64>::zeros((size, size)));
                }
            }

            // Perform operations on arrays
            for array in &arrays {
                let _sum: f64 = array.sum();
                black_box(_sum);
            }

            // Arrays are automatically dropped, testing deallocation
            arrays.clear();
        })
    });

    group.finish();
}

/// Benchmark large matrix operations with memory constraints
fn bench_large_matrix_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_matrix_operations");
    group.sample_size(10); // Fewer samples for large operations

    // Test operations that might cause memory pressure
    for &size in &[1000, 1500, 2000] {
        let matrix = generate_memory_test_data(size);
        let tracker = MemoryTracker::new();

        group.throughput(Throughput::Elements((size * size) as u64));

        if size <= 1500 {
            // Limit to avoid excessive memory usage
            group.bench_with_input(
                BenchmarkId::new("large_determinant", size),
                &size,
                |b, _| {
                    b.iter_custom(|iters| {
                        let (result, snapshot) = tracker.measure("large_det", || {
                            let start = Instant::now();

                            for _ in 0..iters {
                                let _det = basic::det(&matrix.view());
                                black_box(_det);
                            }

                            start.elapsed()
                        });

                        // Log memory usage for analysis
                        let memory_used = tracker.memory_used(&snapshot);
                        println!("Determinant {}: {:.2} MB peak memory", size, memory_used);

                        result
                    })
                },
            );
        }

        // Matrix multiplication with memory tracking
        group.bench_with_input(BenchmarkId::new("large_matmul", size), &size, |b, _| {
            b.iter_custom(|iters| {
                let (result, snapshot) = tracker.measure("large_matmul", || {
                    let start = Instant::now();

                    for _ in 0..iters {
                        let _product = matrix.dot(&matrix);
                        black_box(_product);
                    }

                    start.elapsed()
                });

                let memory_used = tracker.memory_used(&snapshot);
                println!(
                    "Matrix multiply {}: {:.2} MB peak memory",
                    size, memory_used
                );

                result
            })
        });
    }

    group.finish();
}

/// Benchmark zero-copy operations
fn bench_zero_copy_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("zero_copy_operations");

    for &size in &[500, 1000, 1500] {
        let large_matrix = generate_memory_test_data(size);

        // Zero-copy slicing vs copying
        group.bench_with_input(BenchmarkId::new("zero_copy_slice", size), &size, |b, _| {
            b.iter(|| {
                let half_size = size / 2;
                let slice = large_matrix.slice(ndarray::s![0..half_size, 0..half_size]);
                let _norm = slice.map(|&x| x * x).sum();
                black_box(_norm);
            })
        });

        group.bench_with_input(BenchmarkId::new("copy_submatrix", size), &size, |b, _| {
            b.iter(|| {
                let half_size = size / 2;
                let submatrix = large_matrix
                    .slice(ndarray::s![0..half_size, 0..half_size])
                    .to_owned();
                let _norm = submatrix.map(|&x| x * x).sum();
                black_box(_norm);
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_buffer_pool_efficiency,
    bench_chunked_operations,
    bench_memory_usage_patterns,
    bench_allocation_patterns,
    bench_fragmentation_resistance,
    bench_large_matrix_operations,
    bench_zero_copy_operations
);

criterion_main!(benches);
