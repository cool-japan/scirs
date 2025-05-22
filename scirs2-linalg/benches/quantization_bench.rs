use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array1, Array2, ArrayView2};
use scirs2_linalg::quantization::{
    calibration::{calibrate_matrix, CalibrationConfig, CalibrationMethod},
    quantize_matrix, quantize_vector,
    simd::{simd_quantized_matmul, simd_quantized_matvec},
    QuantizationMethod,
};

// Helper functions to generate test data
fn create_random_array2_f32(rows: usize, cols: usize) -> Array2<f32> {
    Array2::from_shape_fn((rows, cols), |(i, j)| ((i * cols + j) % 100) as f32 / 100.0)
}

fn create_random_array1_f32(size: usize) -> Array1<f32> {
    Array1::from_iter((0..size).map(|i| (i % 100) as f32 / 100.0))
}

// Regular matrix multiplication for comparison
fn regular_matmul_f32(a: &ArrayView2<f32>, b: &ArrayView2<f32>) -> Array2<f32> {
    a.dot(b)
}

// Benchmark quantization and dequantization operations
fn bench_quantization(c: &mut Criterion) {
    let mut group = c.benchmark_group("Quantization");
    let sizes = [100, 500, 1000];
    let bit_widths = [4, 8];

    for &size in &sizes {
        let matrix = create_random_array2_f32(size, size);

        for &bits in &bit_widths {
            // Create benchmark ID using both size and bit width
            let id_string = format!("{}x{}_{}bit", size, size, bits);

            // Benchmark quantization with int8 method
            group.bench_with_input(
                BenchmarkId::new("Quantize_Int8", &id_string),
                &size,
                |b, _| {
                    b.iter(|| {
                        black_box(quantize_matrix(
                            &black_box(matrix.view()),
                            bits,
                            QuantizationMethod::Symmetric,
                        ))
                    })
                },
            );

            // Benchmark quantization with per-channel method
            group.bench_with_input(
                BenchmarkId::new("Quantize_PerChannel", &id_string),
                &size,
                |b, _| {
                    b.iter(|| {
                        black_box(quantize_matrix(
                            &black_box(matrix.view()),
                            bits,
                            QuantizationMethod::PerChannelSymmetric,
                        ))
                    })
                },
            );
        }
    }

    group.finish();
}

// Benchmark quantized matrix operations
fn bench_quantized_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("QuantizedOps");
    let sizes = [100, 500];
    let bit_widths = [4, 8];

    for &size in &sizes {
        let matrix_a = create_random_array2_f32(size, size);
        let matrix_b = create_random_array2_f32(size, size);
        let vector = create_random_array1_f32(size);

        for &bits in &bit_widths {
            // Create benchmark ID using both size and bit width
            let id_string = format!("{}x{}_{}bit", size, size, bits);

            // Quantize matrices and vector
            let (qa, qa_params) =
                quantize_matrix(&matrix_a.view(), bits, QuantizationMethod::Symmetric);
            let (qb, qb_params) =
                quantize_matrix(&matrix_b.view(), bits, QuantizationMethod::Symmetric);
            // We don't need qv since we're directly using vector.view() in the benchmark
            let _ = quantize_vector(&vector.view(), bits, QuantizationMethod::Symmetric);

            // Benchmark regular vs. quantized matrix multiplication
            group.bench_with_input(
                BenchmarkId::new("RegularMatMul", &id_string),
                &size,
                |b, _| {
                    b.iter(|| {
                        black_box(regular_matmul_f32(
                            &black_box(matrix_a.view()),
                            &black_box(matrix_b.view()),
                        ))
                    })
                },
            );

            // Clone qa and qb to avoid ownership issues
            let qa_clone = qa.clone();
            let qa_params_clone = qa_params.clone();
            let qb_clone = qb.clone();
            let qb_params_clone = qb_params.clone();

            group.bench_with_input(
                BenchmarkId::new("QuantizedMatMul", &id_string),
                &size,
                |b, _| {
                    b.iter(|| {
                        black_box(
                            simd_quantized_matmul(
                                &qa_clone,
                                &qa_params_clone,
                                &qb_clone,
                                &qb_params_clone,
                            )
                            .unwrap(),
                        )
                    })
                },
            );

            // Benchmark quantized matrix-vector multiplication
            // Create another clone for the matrix-vector multiplication benchmark
            let qa_clone2 = qa.clone();
            let qa_params_clone2 = qa_params.clone();

            group.bench_with_input(
                BenchmarkId::new("QuantizedMatVec", &id_string),
                &size,
                |b, _| {
                    b.iter(|| {
                        black_box(
                            simd_quantized_matvec(&qa_clone2, &qa_params_clone2, &vector.view())
                                .unwrap(),
                        )
                    })
                },
            );
        }
    }

    group.finish();
}

// Benchmark calibration methods
fn bench_calibration(c: &mut Criterion) {
    let mut group = c.benchmark_group("Calibration");
    let sizes = [100, 500];

    for &size in &sizes {
        let matrix = create_random_array2_f32(size, size);

        // Create different calibration configurations
        let minmax_config = CalibrationConfig {
            method: CalibrationMethod::MinMax,
            symmetric: true,
            ..Default::default()
        };

        let percentile_config = CalibrationConfig {
            method: CalibrationMethod::PercentileCalibration,
            percentile: 0.99,
            symmetric: true,
            ..Default::default()
        };

        let ema_config = CalibrationConfig {
            method: CalibrationMethod::ExponentialMovingAverage,
            ema_factor: 0.1,
            max_iterations: 5,
            convergence_threshold: 1e-4,
            symmetric: true,
            ..Default::default()
        };

        // Clone the configs to avoid move issues
        let minmax_config_clone = minmax_config.clone();
        let percentile_config_clone = percentile_config.clone();
        let ema_config_clone = ema_config.clone();

        // Benchmark different calibration methods
        group.bench_with_input(BenchmarkId::new("MinMax", size), &size, |b, _| {
            b.iter(|| black_box(calibrate_matrix(&matrix.view(), 8, &minmax_config_clone).unwrap()))
        });

        group.bench_with_input(BenchmarkId::new("Percentile", size), &size, |b, _| {
            b.iter(|| {
                black_box(calibrate_matrix(&matrix.view(), 8, &percentile_config_clone).unwrap())
            })
        });

        group.bench_with_input(BenchmarkId::new("EMA", size), &size, |b, _| {
            b.iter(|| black_box(calibrate_matrix(&matrix.view(), 8, &ema_config_clone).unwrap()))
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_quantization,
    bench_quantized_ops,
    bench_calibration
);
criterion_main!(benches);
