//! Benchmarks for pattern recognition system
//!
//! This benchmark suite tests the performance of pattern detection algorithms
//! under various access patterns and data sizes.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use scirs2_core::memory_efficient::{ComplexPattern, PatternRecognitionConfig, PatternRecognizer};
use std::hint::black_box;

/// Simulate row-major access pattern
#[allow(dead_code)]
fn row_major_pattern(rows: usize, cols: usize) -> Vec<usize> {
    (0..rows * cols).collect()
}

/// Simulate column-major access pattern
#[allow(dead_code)]
fn column_major_pattern(rows: usize, cols: usize) -> Vec<usize> {
    let mut pattern = Vec::with_capacity(rows * cols);
    for j in 0..cols {
        for i in 0..rows {
            pattern.push(i * cols + j);
        }
    }
    pattern
}

/// Simulate zigzag access pattern
#[allow(dead_code)]
fn zigzag_pattern(rows: usize, cols: usize) -> Vec<usize> {
    let mut pattern = Vec::with_capacity(rows * cols);
    for row in 0..rows {
        if row % 2 == 0 {
            for col in 0..cols {
                pattern.push(row * cols + col);
            }
        } else {
            for col in (0..cols).rev() {
                pattern.push(row * cols + col);
            }
        }
    }
    pattern
}

/// Simulate diagonal access pattern
#[allow(dead_code)]
fn diagonal_pattern(rows: usize, cols: usize) -> Vec<usize> {
    let mut pattern = Vec::new();
    let min_dim = rows.min(cols);
    for i in 0..min_dim {
        pattern.push(i * cols + i);
    }
    pattern
}

/// Simulate block access pattern
#[allow(dead_code)]
fn block_pattern(rows: usize, cols: usize, blocksize: usize) -> Vec<usize> {
    let mut pattern = Vec::new();

    // Access in blocks
    for block_row in (0..rows).step_by(block_size) {
        for block_col in (0..cols).step_by(block_size) {
            // Access all elements within a block
            for i in 0..block_size.min(rows - block_row) {
                for j in 0..block_size.min(cols - block_col) {
                    pattern.push((block_row + i) * cols + (block_col + j));
                }
            }
        }
    }
    pattern
}

/// Simulate random access pattern
#[allow(dead_code)]
fn random_pattern(rows: usize, cols: usize, count: usize) -> Vec<usize> {
    use rand::{rngs::StdRng, Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(42);
    let max_idx = rows * cols;

    (0..count).map(|_| rng.gen_range(0..max_idx)).collect()
}

/// Simulate stencil access pattern (5-point stencil)
#[allow(dead_code)]
fn stencil_pattern(rows: usize, cols: usize) -> Vec<usize> {
    let mut pattern = Vec::new();

    // Access interior points with their 5-point stencil
    for i in 1..rows - 1 {
        for j in 1..cols - 1 {
            let center = i * cols + j;
            pattern.push(center);
            pattern.push(center - cols); // North
            pattern.push(center + 1); // East
            pattern.push(center + cols); // South
            pattern.push(center - 1); // West
        }
    }
    pattern
}

/// Benchmark pattern detection speed
#[allow(dead_code)]
fn bench_pattern_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group(pattern_detection);

    // Test different matrix sizes
    let sizes = vec![(32, 32), (64, 64), (128, 128), (256, 256)];

    for (rows, cols) in sizes {
        let size_id = format!("{}x{}", rows, cols);

        // Row-major pattern detection
        group.bench_with_input(
            BenchmarkId::new("row_major", &size_id),
            &(rows, cols),
            |b, &(rows, cols)| {
                let pattern = generate_row_major_pattern(rows, cols);
                b.iter(|| {
                    let mut recognizer =
                        PatternRecognizer::new(PatternRecognitionConfig::default());
                    recognizer.set_dimensions(vec![rows, cols]);

                    for &idx in &pattern {
                        recognizer.record_access(black_box(idx));
                    }

                    black_box(recognizer.get_best_pattern());
                });
            },
        );

        // Column-major pattern detection
        group.bench_with_input(
            BenchmarkId::new("column_major", &size_id),
            &(rows, cols),
            |b, &(rows, cols)| {
                let pattern = generate_column_major_pattern(rows, cols);
                b.iter(|| {
                    let mut recognizer =
                        PatternRecognizer::new(PatternRecognitionConfig::default());
                    recognizer.set_dimensions(vec![rows, cols]);

                    for &idx in &pattern {
                        recognizer.record_access(black_box(idx));
                    }

                    black_box(recognizer.get_best_pattern());
                });
            },
        );

        // Zigzag pattern detection
        group.bench_with_input(
            BenchmarkId::new("zigzag", &size_id),
            &(rows, cols),
            |b, &(rows, cols)| {
                let pattern = generate_zigzag_pattern(rows, cols);
                b.iter(|| {
                    let mut recognizer =
                        PatternRecognizer::new(PatternRecognitionConfig::default());
                    recognizer.set_dimensions(vec![rows, cols]);

                    for &idx in &pattern {
                        recognizer.record_access(black_box(idx));
                    }

                    black_box(recognizer.get_best_pattern());
                });
            },
        );

        // Block pattern detection
        group.bench_with_input(
            BenchmarkId::new("block_4x4", &size_id),
            &(rows, cols),
            |b, &(rows, cols)| {
                let pattern = generate_block_pattern(rows, cols, 4);
                b.iter(|| {
                    let mut recognizer =
                        PatternRecognizer::new(PatternRecognitionConfig::default());
                    recognizer.set_dimensions(vec![rows, cols]);

                    for &idx in &pattern {
                        recognizer.record_access(black_box(idx));
                    }

                    black_box(recognizer.get_best_pattern());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark recognition accuracy with mixed patterns
#[allow(dead_code)]
fn bench_mixed_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group(mixed_patterns);

    group.bench_function("transition_detection", |b| {
        let rows = 64;
        let cols = 64;

        b.iter(|| {
            let mut recognizer = PatternRecognizer::new(PatternRecognitionConfig::default());
            recognizer.set_dimensions(vec![rows, cols]);

            // Start with row-major
            for i in 0..1000 {
                recognizer.record_access(black_box(i));
            }

            // Transition to column-major
            for j in 0..32 {
                for i in 0..32 {
                    recognizer.record_access(black_box(i * cols + j));
                }
            }

            // Transition to diagonal
            for i in 0..32 {
                recognizer.record_access(black_box(i * cols + i));
            }

            black_box(recognizer.get_patterns());
        });
    });

    group.finish();
}

/// Benchmark memory overhead of pattern recognition
#[allow(dead_code)]
fn bench_memory_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group(memory_overhead);

    // Test with different history sizes
    let history_sizes = vec![100, 500, 1000, 5000];

    for history_size in history_sizes {
        group.bench_with_input(
            BenchmarkId::new("history", history_size),
            &history_size,
            |b, &history_size| {
                b.iter(|| {
                    let mut recognizer =
                        PatternRecognizer::new(PatternRecognitionConfig::default());
                    recognizer.set_dimensions(vec![1000, 1000]);

                    // Record many accesses to test memory usage
                    for i in 0..history_size {
                        recognizer.record_access(black_box(i % 100000));
                    }

                    black_box(recognizer.get_patterns().len());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark pattern-specific optimizations
#[allow(dead_code)]
fn bench_pattern_optimizations(c: &mut Criterion) {
    let mut group = c.benchmark_group(pattern_optimizations);

    // Test with different configurations
    let configs = vec![
        (
            "all_enabled",
            PatternRecognitionConfig {
                min_history_size: 20,
                pattern_expiry: std::time::Duration::from_secs(60),
                detect_diagonal: true,
                detect_block: true,
                detect_stencil: true,
                detect_sparse: true,
                use_machine_learning: false,
            },
        ),
        (
            "basic_only",
            PatternRecognitionConfig {
                min_history_size: 20,
                pattern_expiry: std::time::Duration::from_secs(60),
                detect_diagonal: false,
                detect_block: false,
                detect_stencil: false,
                detect_sparse: false,
                use_machine_learning: false,
            },
        ),
        (
            "ml_enabled",
            PatternRecognitionConfig {
                min_history_size: 20,
                pattern_expiry: std::time::Duration::from_secs(60),
                detect_diagonal: true,
                detect_block: true,
                detect_stencil: true,
                detect_sparse: true,
                use_machine_learning: true,
            },
        ),
    ];

    for (name, config) in configs {
        group.bench_with_input(
            BenchmarkId::new("config", name),
            &config,
            |b, config: &PatternRecognitionConfig| {
                let pattern = generate_stencil_pattern(64, 64);
                b.iter(|| {
                    let mut recognizer = PatternRecognizer::new(config.clone());
                    recognizer.set_dimensions(vec![64, 64]);

                    for &idx in &pattern {
                        recognizer.record_access(black_box(idx));
                    }

                    black_box(recognizer.get_patterns());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark real-world scenarios
#[allow(dead_code)]
fn bench_real_world_scenarios(c: &mut Criterion) {
    let mut group = c.benchmark_group(real_world);

    // Matrix multiplication access pattern
    group.bench_function("matrix_multiply", |b| {
        let n = 64;
        b.iter(|| {
            let mut recognizer = PatternRecognizer::new(PatternRecognitionConfig::default());
            recognizer.set_dimensions(vec![n, n]);

            // Simulate matrix multiplication access pattern
            for i in 0..n {
                for j in 0..n {
                    for k in 0..n {
                        // Access A[i,k]
                        recognizer.record_access(black_box(i * n + k));
                        // Access B[k,j]
                        recognizer.record_access(black_box(k * n + j));
                        // Access C[i,j]
                        recognizer.record_access(black_box(i * n + j));
                    }
                }
            }

            black_box(recognizer.get_patterns());
        });
    });

    // Convolution access pattern
    group.bench_function("convolution", |b| {
        let rows = 64;
        let cols = 64;
        let kernel_size = 3;

        b.iter(|| {
            let mut recognizer = PatternRecognizer::new(PatternRecognitionConfig::default());
            recognizer.set_dimensions(vec![rows, cols]);

            // Simulate 2D convolution access pattern
            for i in kernel_size / 2..rows - kernel_size / 2 {
                for j in kernel_size / 2..cols - kernel_size / 2 {
                    // Access kernel neighborhood
                    for ki in 0..kernel_size {
                        for kj in 0..kernel_size {
                            let idx =
                                (i - kernel_size / 2 + ki) * cols + (j - kernel_size / 2 + kj);
                            recognizer.record_access(black_box(idx));
                        }
                    }
                }
            }

            black_box(recognizer.get_patterns());
        });
    });

    // Sparse matrix access pattern
    group.bench_function("sparsematrix", |b| {
        let rows = 1000;
        let cols = 1000;
        let nnz = 5000; // 0.5% density

        b.iter(|| {
            let mut recognizer = PatternRecognizer::new(PatternRecognitionConfig::default());
            recognizer.set_dimensions(vec![rows, cols]);

            // Simulate sparse matrix access
            let pattern = generate_random_pattern(rows, cols, nnz);
            for &idx in &pattern {
                recognizer.record_access(black_box(idx));
            }

            let patterns = recognizer.get_patterns();
            // Should detect sparse pattern
            assert!(patterns
                .iter()
                .any(|p| matches!(p.pattern_type, ComplexPattern::Sparse { .. })));
            black_box(patterns);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_pattern_detection,
    bench_mixed_patterns,
    bench_memory_overhead,
    bench_pattern_optimizations,
    bench_real_world_scenarios
);
criterion_main!(benches);
