//! Benchmark for quantization calibration methods
//!
//! This example benchmarks different calibration methods in terms of
//! accuracy, performance, and memory savings.

use scirs2_core::essentials::{Normal, Uniform};
use scirs2_core::ndarray::Array2;
use scirs2_core::random::prelude::*;
use scirs2_core::random::{Cauchy, Distribution, LogNormal};
use scirs2_linalg::quantization::calibration::{
    calibrate_matrix, CalibrationConfig, CalibrationMethod,
};
use scirs2_linalg::quantization::{dequantize_matrix, quantize_matrix, QuantizationMethod};
use std::time::Instant;

// Benchmark configuration
const MATRIX_SIZE: usize = 100;
const NUM_ITERATIONS: usize = 10;
const BITS: u8 = 8;

#[allow(dead_code)]
fn main() {
    println!("Quantization Calibration Benchmark");
    println!("==================================\n");

    // Generate test matrices with different distributions
    println!("Generating test matrices...");
    let uniform_data = generate_uniform_data(MATRIX_SIZE);
    let normal_data = generate_normal_data(MATRIX_SIZE);
    let lognormal_data = generate_lognormal_data(MATRIX_SIZE);
    let bimodal_data = generate_bimodal_data(MATRIX_SIZE);
    let mixed_data = generate_mixed_scale_data(MATRIX_SIZE);
    let heavy_tailed_data = generate_heavy_tailed_data(MATRIX_SIZE);

    // Define distributions to test
    let distributions = [
        ("Uniform", &uniform_data),
        ("Normal", &normal_data),
        ("LogNormal", &lognormal_data),
        ("Bimodal", &bimodal_data),
        ("Mixed Scale", &mixed_data),
        ("Heavy Tailed", &heavy_tailed_data),
    ];

    // Define calibration methods to test
    let methods = [
        (CalibrationMethod::MinMax, "Min-Max"),
        (CalibrationMethod::MovingAverageMinMax, "Moving Avg Min-Max"),
        (CalibrationMethod::PercentileCalibration, "Percentile"),
        (CalibrationMethod::EntropyCalibration, "Entropy"),
        (CalibrationMethod::MSEOptimization, "MSE Optimization"),
        (CalibrationMethod::ExponentialMovingAverage, "EMA"),
    ];

    // Run benchmark for symmetric quantization
    println!("\nBenchmarking symmetric quantization (8-bit):");
    benchmark_methods(&distributions, &methods, true);

    // Run benchmark for asymmetric quantization
    println!("\nBenchmarking asymmetric quantization (8-bit):");
    benchmark_methods(&distributions, &methods, false);

    // Run bit-width benchmark
    println!("\nBenchmarking different bit widths using entropy calibration:");
    benchmark_bit_widths(&distributions);

    // Run hardware-friendly benchmark (Int4, UInt4, etc.)
    println!("\nBenchmarking hardware-friendly quantization formats:");
    benchmark_hardware_friendly(&distributions);
}

/// Generate a matrix with uniform distribution
#[allow(dead_code)]
fn generate_uniform_data(size: usize) -> Array2<f32> {
    let mut rng = thread_rng();
    let uniform = Uniform::new(-1.0, 1.0).unwrap();

    let mut data = Array2::zeros((size, size));
    for i in 0..size {
        for j in 0..size {
            data[[i, j]] = uniform.sample(&mut rng);
        }
    }

    data
}

/// Generate a matrix with normal distribution
#[allow(dead_code)]
fn generate_normal_data(size: usize) -> Array2<f32> {
    let mut rng = thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();

    let mut data = Array2::zeros((size, size));
    for i in 0..size {
        for j in 0..size {
            data[[i, j]] = normal.sample(&mut rng);
        }
    }

    data
}

/// Generate a matrix with log-normal distribution
#[allow(dead_code)]
fn generate_lognormal_data(size: usize) -> Array2<f32> {
    let mut rng = thread_rng();
    let lognormal = LogNormal::new(0.0, 1.0).unwrap();

    let mut data = Array2::zeros((size, size));
    for i in 0..size {
        for j in 0..size {
            data[[i, j]] = lognormal.sample(&mut rng);
        }
    }

    data
}

/// Generate a matrix with bimodal distribution
#[allow(dead_code)]
fn generate_bimodal_data(size: usize) -> Array2<f32> {
    let mut rng = thread_rng();
    let normal1 = Normal::new(-2.0, 0.5).unwrap();
    let normal2 = Normal::new(2.0, 0.5).unwrap();

    let mut data = Array2::zeros((size, size));
    for i in 0..size {
        for j in 0..size {
            // 50% chance of coming from each distribution
            if rng.random::<bool>() {
                data[[i, j]] = normal1.sample(&mut rng);
            } else {
                data[[i, j]] = normal2.sample(&mut rng);
            }
        }
    }

    data
}

/// Generate a matrix with mixed scales in different columns
#[allow(dead_code)]
fn generate_mixed_scale_data(size: usize) -> Array2<f32> {
    let mut rng = thread_rng();

    let mut data = Array2::zeros((size, size));

    // Divide the matrix into regions with different scales
    let regionsize = size / 4;

    // Region 1: small values around 0.1
    for i in 0..regionsize {
        for j in 0..size {
            data[[i, j]] = 0.1 + 0.05 * rng.random::<f32>();
        }
    }

    // Region 2: medium values around 1.0
    for i in regionsize..(2 * regionsize) {
        for j in 0..size {
            data[[i, j]] = 1.0 + 0.5 * rng.random::<f32>();
        }
    }

    // Region 3: large values around 10.0
    for i in (2 * regionsize)..(3 * regionsize) {
        for j in 0..size {
            data[[i, j]] = 10.0 + 5.0 * rng.random::<f32>();
        }
    }

    // Region 4: very large values around 100.0
    for i in (3 * regionsize)..size {
        for j in 0..size {
            data[[i, j]] = 100.0 + 50.0 * rng.random::<f32>();
        }
    }

    data
}

/// Generate a matrix with heavy-tailed distribution (Cauchy)
#[allow(dead_code)]
fn generate_heavy_tailed_data(size: usize) -> Array2<f32> {
    let mut rng = thread_rng();
    let cauchy = Cauchy::new(0.0, 1.0).unwrap();

    let mut data = Array2::zeros((size, size));
    for i in 0..size {
        for j in 0..size {
            // The Cauchy distribution can generate extreme outliers
            // We'll clamp the values to avoid numerical issues
            let val: f32 = cauchy.sample(&mut rng);
            data[[i, j]] = val.clamp(-100.0, 100.0);
        }
    }

    data
}

/// Benchmark different calibration methods on various distributions
#[allow(dead_code)]
fn benchmark_methods(
    distributions: &[(&str, &Array2<f32>)],
    methods: &[(CalibrationMethod, &str)],
    symmetric: bool,
) {
    println!(
        "{:^15} | {:^20} | {:^15} | {:^15} | {:^15}",
        "Distribution", "Method", "MSE", "Time (ms)", "Size Reduction"
    );
    println!(
        "{:-^15} | {:-^20} | {:-^15} | {:-^15} | {:-^15}",
        "", "", "", "", ""
    );

    for &(dist_name, data) in distributions {
        for &(method, method_name) in methods {
            // Configure calibration
            let config = CalibrationConfig {
                method,
                symmetric,
                ..Default::default()
            };

            // Measure calibration time
            let start = Instant::now();
            let mut _q_params = None;
            let mut total_mse = 0.0;

            for _ in 0..NUM_ITERATIONS {
                let params = calibrate_matrix(&data.view(), BITS, &config).unwrap();
                _q_params = Some(params.clone());

                // Quantize and dequantize
                let (quantized, _) = quantize_matrix(&data.view(), BITS, params.method);
                let dequantized = dequantize_matrix(&quantized, &params);

                // Calculate MSE
                let diff = data - &dequantized;
                let mse = diff.mapv(|x| x * x).sum() / data.len() as f32;
                total_mse += mse;
            }

            let elapsed = start.elapsed();
            let avg_time = elapsed.as_millis() as f32 / NUM_ITERATIONS as f32;
            let avg_mse = total_mse / NUM_ITERATIONS as f32;

            // Calculate memory savings
            let fp32size = 32;
            let size_reduction = (1.0 - (BITS as f32 / fp32size as f32)) * 100.0;

            println!(
                "{:^15} | {:^20} | {:^15.6} | {:^15.2} | {:^15.1}%",
                dist_name, method_name, avg_mse, avg_time, size_reduction
            );
        }

        // Add separator between distributions
        println!(
            "{:-^15} | {:-^20} | {:-^15} | {:-^15} | {:-^15}",
            "", "", "", "", ""
        );
    }
}

/// Benchmark different bit widths using entropy calibration
#[allow(dead_code)]
fn benchmark_bit_widths(distributions: &[(&str, &Array2<f32>)]) {
    let bit_widths = [4, 8, 16];

    println!(
        "{:^15} | {:^10} | {:^15} | {:^15} | {:^15}",
        "Distribution", "Bits", "MSE", "Rel Error (%)", "Size Reduction"
    );
    println!(
        "{:-^15} | {:-^10} | {:-^15} | {:-^15} | {:-^15}",
        "", "", "", "", ""
    );

    for &(dist_name, data) in distributions {
        for &bits in &bit_widths {
            // Configure calibration using entropy method (generally robust)
            let config = CalibrationConfig {
                method: CalibrationMethod::EntropyCalibration,
                symmetric: true,
                ..Default::default()
            };

            // Calibrate parameters
            let params = calibrate_matrix(&data.view(), bits, &config).unwrap();

            // Quantize and dequantize
            let (quantized, _) = quantize_matrix(&data.view(), bits, params.method);
            let dequantized = dequantize_matrix(&quantized, &params);

            // Calculate MSE
            let diff = data - &dequantized;
            let mse = diff.mapv(|x| x * x).sum() / data.len() as f32;

            // Calculate relative error
            let diff_abs = (data - &dequantized).mapv(|x| x.abs());
            let rel_error = diff_abs.sum() / data.mapv(|x| x.abs()).sum() * 100.0;

            // Calculate memory savings
            let fp32size = 32;
            let size_reduction = (1.0 - (bits as f32 / fp32size as f32)) * 100.0;

            println!(
                "{:^15} | {:^10} | {:^15.6} | {:^15.6} | {:^15.1}%",
                dist_name, bits, mse, rel_error, size_reduction
            );
        }

        // Add separator between _distributions
        println!(
            "{:-^15} | {:-^10} | {:-^15} | {:-^15} | {:-^15}",
            "", "", "", "", ""
        );
    }
}

/// Benchmark hardware-friendly quantization formats
#[allow(dead_code)]
fn benchmark_hardware_friendly(distributions: &[(&str, &Array2<f32>)]) {
    // Define hardware-friendly formats to test
    let formats = [
        (8, QuantizationMethod::Symmetric, "Int8 Symmetric"),
        (8, QuantizationMethod::Affine, "Int8 Affine"),
        (4, QuantizationMethod::Int4, "Int4"),
        (4, QuantizationMethod::UInt4, "UInt4"),
        (16, QuantizationMethod::Float16, "Float16"),
        (16, QuantizationMethod::BFloat16, "BFloat16"),
    ];

    println!(
        "{:^15} | {:^20} | {:^15} | {:^15} | {:^15}",
        "Distribution", "Format", "MSE", "Rel Error (%)", "Size Reduction"
    );
    println!(
        "{:-^15} | {:-^20} | {:-^15} | {:-^15} | {:-^15}",
        "", "", "", "", ""
    );

    for &(dist_name, data) in distributions {
        for &(bits, method, format_name) in &formats {
            // Measure calibration time
            let start = Instant::now();

            // Quantize using the specific format
            let (quantized, params) = quantize_matrix(&data.view(), bits, method);
            let _elapsed = start.elapsed();

            // Dequantize for error measurement
            let dequantized = dequantize_matrix(&quantized, &params);

            // Calculate MSE
            let diff = data - &dequantized;
            let mse = diff.mapv(|x| x * x).sum() / data.len() as f32;

            // Calculate relative error
            let diff_abs = (data - &dequantized).mapv(|x| x.abs());
            let rel_error = diff_abs.sum() / data.mapv(|x| x.abs()).sum() * 100.0;

            // Calculate memory savings
            let fp32size = 32;
            let size_reduction = (1.0 - (bits as f32 / fp32size as f32)) * 100.0;

            println!(
                "{:^15} | {:^20} | {:^15.6} | {:^15.6} | {:^15.1}%",
                dist_name, format_name, mse, rel_error, size_reduction
            );
        }

        // Add separator between _distributions
        println!(
            "{:-^15} | {:-^20} | {:-^15} | {:-^15} | {:-^15}",
            "", "", "", "", ""
        );
    }
}
