//! Example demonstrating SIMD-accelerated quantized matrix operations
//!
//! This example shows how to use the SIMD-accelerated quantized matrix operations
//! for improved performance while maintaining reasonable accuracy.

use ndarray::{Array1, Array2};
use scirs2_linalg::quantization::simd::{simd_quantized_matmul, simd_quantized_matvec};
use scirs2_linalg::quantization::{quantize_matrix, quantize_vector, QuantizationMethod};
use std::time::Instant;

#[allow(dead_code)]
fn main() {
    println!("SIMD-Accelerated Quantized Matrix Operations Example");
    println!("===================================================\n");

    // Create test matrices
    let m = 100;
    let k = 100;
    let n = 100;

    // Generate random matrices for testing
    println!("Generating random matrices...");
    let a = generate_randommatrix(m, k);
    let b = generate_randommatrix(k, n);
    let v = generate_random_vector(k);

    println!("\nPerforming matrix-matrix multiplication tests...");

    // Non-quantized reference implementation
    let start = Instant::now();
    let c_ref = a.dot(&b);
    let ref_time = start.elapsed();
    println!("Reference matmul time: {:?}", ref_time);

    // Test with 8-bit symmetric quantization
    let (a_q, a_params) = quantize_matrix(&a.view(), 8, QuantizationMethod::Symmetric);
    let (b_q, b_params) = quantize_matrix(&b.view(), 8, QuantizationMethod::Symmetric);

    let start = Instant::now();
    let c_q_simd = simd_quantized_matmul(&a_q, &a_params, &b_q, &b_params).unwrap();
    let simd_time = start.elapsed();
    println!("SIMD quantized matmul time: {:?}", simd_time);

    // Calculate error
    let max_error = (&c_ref - &c_q_simd)
        .mapv(|x| x.abs())
        .fold(0.0_f32, |acc, &x| acc.max(x));
    let rel_error = max_error / c_ref.fold(0.0_f32, |acc, &x| acc.max(x.abs()));
    println!("Max absolute error: {}", max_error);
    println!("Relative error: {}", rel_error);
    println!(
        "Speedup: {:.2}x",
        ref_time.as_secs_f64() / simd_time.as_secs_f64()
    );

    println!("\nPerforming matrix-vector multiplication tests...");

    // Non-quantized reference implementation
    let start = Instant::now();
    let r_ref = a.dot(&v);
    let ref_time = start.elapsed();
    println!("Reference matvec time: {:?}", ref_time);

    // Test with 8-bit symmetric quantization for matrix and vector
    let _v_q_v_params = quantize_vector(&v.view(), 8, QuantizationMethod::Symmetric);

    let start = Instant::now();
    let r_q_simd = simd_quantized_matvec(&a_q, &a_params, &v.view()).unwrap();
    let simd_time = start.elapsed();
    println!("SIMD quantized matvec time: {:?}", simd_time);

    // Calculate error
    let max_error = (&r_ref - &r_q_simd)
        .mapv(|x| x.abs())
        .fold(0.0_f32, |acc, &x| acc.max(x));
    let rel_error = max_error / r_ref.fold(0.0_f32, |acc, &x| acc.max(x.abs()));
    println!("Max absolute error: {}", max_error);
    println!("Relative error: {}", rel_error);
    println!(
        "Speedup: {:.2}x",
        ref_time.as_secs_f64() / simd_time.as_secs_f64()
    );

    // Benchmark with different quantization methods
    println!("\nBenchmarking different quantization methods for matrix multiplication:");
    benchmark_quantization_methods(&a, &b);
}

#[allow(dead_code)]
fn generate_randommatrix(rows: usize, cols: usize) -> Array2<f32> {
    let mut matrix = Array2::zeros((rows, cols));

    for i in 0..rows {
        for j in 0..cols {
            matrix[[i, j]] = rand::random::<f32>() * 2.0 - 1.0; // Values between -1 and 1
        }
    }

    matrix
}

#[allow(dead_code)]
fn generate_random_vector(length: usize) -> Array1<f32> {
    let mut vector = Array1::zeros(length);

    for i in 0..length {
        vector[i] = rand::random::<f32>() * 2.0 - 1.0; // Values between -1 and 1
    }

    vector
}

#[allow(dead_code)]
fn benchmark_quantization_methods(a: &Array2<f32>, b: &Array2<f32>) {
    // Define quantization methods to test
    let methods = [
        QuantizationMethod::Symmetric,
        QuantizationMethod::Affine,
        QuantizationMethod::Int4,
        QuantizationMethod::UInt4,
        QuantizationMethod::PerChannelSymmetric,
    ];

    // Define bit widths to test
    let bit_widths = [8, 4]; // Only test 8-bit and 4-bit

    println!(
        "{:^15} | {:^10} | {:^15} | {:^15} | {:^10}",
        "Method", "Bits", "Time (ms)", "Rel. Error", "Speedup"
    );
    println!(
        "{:-^15} | {:-^10} | {:-^15} | {:-^15} | {:-^10}",
        "", "", "", "", ""
    );

    // Non-quantized reference implementation
    let start = Instant::now();
    let c_ref = a.dot(b);
    let ref_time = start.elapsed();
    println!(
        "{:^15} | {:^10} | {:^15.3} | {:^15} | {:^10}",
        "Reference",
        "32",
        ref_time.as_millis() as f64,
        0.0,
        1.0
    );

    let max_abs_val = c_ref.fold(0.0_f32, |acc, &x| acc.max(x.abs()));

    for &method in &methods {
        for &bits in &bit_widths {
            // Skip incompatible combinations
            if bits != 4
                && (method == QuantizationMethod::Int4 || method == QuantizationMethod::UInt4)
            {
                continue;
            }
            if bits != 8 && method == QuantizationMethod::PerChannelSymmetric {
                continue;
            }

            // Quantize matrices
            let (a_q, a_params) = quantize_matrix(&a.view(), bits, method);
            let (b_q, b_params) = quantize_matrix(&b.view(), bits, method);

            // Benchmark SIMD quantized matmul
            let start = Instant::now();
            let c_q = simd_quantized_matmul(&a_q, &a_params, &b_q, &b_params).unwrap();
            let q_time = start.elapsed();

            // Calculate error
            let abs_error = (&c_ref - &c_q)
                .mapv(|x| x.abs())
                .fold(0.0_f32, |acc, &x| acc.max(x));
            let rel_error = abs_error / max_abs_val;

            // Calculate speedup
            let speedup = ref_time.as_secs_f64() / q_time.as_secs_f64();

            println!(
                "{:^15} | {:^10} | {:^15.3} | {:^15.6} | {:^10.2}",
                format!("{:?}", method),
                bits,
                q_time.as_millis() as f64,
                rel_error,
                speedup
            );
        }
    }
}
