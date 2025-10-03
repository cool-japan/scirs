//! Tests for Float16 and BFloat16 quantization
//!
//! Tests for floating-point quantization methods that use reduced precision
//! float formats instead of integer quantization.

use crate::quantization::{
    dequantize_matrix, quantize_matrix, quantize_vector, quantized_dot, quantized_matmul,
    quantized_matvec, QuantizationMethod, QuantizedDataType,
};
use scirs2_core::ndarray::{array, Array2};

#[test]
fn test_float16_quantization() {
    let a = array![[1.0_f32, 2.5, 3.7], [4.2, 5.0, 6.1]];

    // Quantize to float16
    let (quantized, params) = quantize_matrix(&a.view(), 16, QuantizationMethod::Float16);
    let a_dequantized = dequantize_matrix(&quantized, &params);

    // For float16, we should have minimal error
    let max_diff = (&a - &a_dequantized)
        .mapv(|x| x.abs())
        .fold(0.0_f32, |acc, &b| acc.max(b));
    println!("Max error (Float16): {}", max_diff);

    // Verify data type
    assert_eq!(quantized.data_type, QuantizedDataType::Float16);

    // Float16 should be accurate to ~3 decimal places for small values
    assert!(max_diff < 0.01, "Max error too large: {}", max_diff);
}

#[test]
fn test_bfloat16_quantization() {
    let a = array![[1.0_f32, 2.5, 3.7], [4.2, 5.0, 6.1]];

    // Quantize to bfloat16
    let (quantized, params) = quantize_matrix(&a.view(), 16, QuantizationMethod::BFloat16);
    let a_dequantized = dequantize_matrix(&quantized, &params);

    // BFloat16 has less precision but same exponent range as f32
    let max_diff = (&a - &a_dequantized)
        .mapv(|x| x.abs())
        .fold(0.0_f32, |acc, &b| acc.max(b));
    println!("Max error (BFloat16): {}", max_diff);

    // Verify data type
    assert_eq!(quantized.data_type, QuantizedDataType::BFloat16);

    // BFloat16 has less precision than Float16
    assert!(max_diff < 0.1, "Max error too large: {}", max_diff);
}

#[test]
fn test_storage_comparison() {
    let rows = 100;
    let cols = 100;

    // Create a large matrix
    let mut data = Vec::with_capacity(rows * cols);
    for i in 0..rows * cols {
        data.push((i % 1000) as f32 / 100.0); // Various values
    }

    let matrix = Array2::from_shape_vec((rows, cols), data).unwrap();

    // Test different quantization methods
    let (int8matrix, _) = quantize_matrix(&matrix.view(), 8, QuantizationMethod::Symmetric);
    let (int4matrix, _) = quantize_matrix(&matrix.view(), 4, QuantizationMethod::Int4);
    let (f16matrix, _) = quantize_matrix(&matrix.view(), 16, QuantizationMethod::Float16);
    let (bf16matrix, _) = quantize_matrix(&matrix.view(), 16, QuantizationMethod::BFloat16);

    // Calculate storage sizes (in bytes)
    let originalsize = matrix.len() * std::mem::size_of::<f32>();

    // Check actual memory footprint ratios
    println!("Original f32 size: {} bytes", originalsize);
    println!("Int8 storage: {} bytes", int8matrix.data.len());
    println!("Int4 storage: {} bytes", int4matrix.data.len());
    println!("Float16 storage: {} bytes", f16matrix.data.len() * 2); // f16 is 2 bytes each
    println!("BFloat16 storage: {} bytes", bf16matrix.data.len() * 2); // bf16 is 2 bytes each

    // Verify expected ratios
    assert!(int8matrix.data.len() * 4 <= originalsize); // 8-bit should be 25% of original (32-bit) size
    assert!(int4matrix.data.len() * 8 <= originalsize); // 4-bit should be 12.5% of original size
    assert!(f16matrix.data.len() * 2 <= originalsize); // 16-bit should be 50% of original size
    assert!(bf16matrix.data.len() * 2 <= originalsize); // 16-bit should be 50% of original size
}

#[test]
fn test_storage_efficiency_int4() {
    let rows = 100;
    let cols = 100;

    // Create a large matrix
    let mut data = Vec::with_capacity(rows * cols);
    for i in 0..rows * cols {
        data.push((i % 15) as f32 - 7.0); // Values between -7 and 7
    }

    let matrix = Array2::from_shape_vec((rows, cols), data).unwrap();

    // Quantize with 8-bit
    let (quantized8, _) = quantize_matrix(&matrix.view(), 8, QuantizationMethod::Symmetric);

    // Quantize with 4-bit
    let (quantized4, _) = quantize_matrix(&matrix.view(), 4, QuantizationMethod::Int4);

    // Check that the 4-bit version uses approximately half the memory
    println!("8-bit storage: {} bytes", quantized8.data.len());
    println!("4-bit storage: {} bytes", quantized4.data.len());

    // Should be close to 50% of the original size (allowing for some overhead)
    assert!(quantized4.data.len() as f32 <= 0.6 * quantized8.data.len() as f32);
}

#[test]
fn test_float16matrix_operations() {
    let a = array![[1.0_f32, 2.0], [3.0, 4.0]];
    let b = array![[5.0_f32, 6.0], [7.0, 8.0]];
    let x = array![5.0_f32, 6.0];

    // Quantize to float16
    let (a_q, a_params) = quantize_matrix(&a.view(), 16, QuantizationMethod::Float16);
    let (b_q, b_params) = quantize_matrix(&b.view(), 16, QuantizationMethod::Float16);
    let (x_q, x_params) = quantize_vector(&x.view(), 16, QuantizationMethod::Float16);

    // Test matrix multiplication
    let c_q = quantized_matmul(&a_q, &a_params, &b_q, &b_params).unwrap();
    let c = a.dot(&b);

    // Test matrix-vector multiplication
    let y_q = quantized_matvec(&a_q, &a_params, &x_q, &x_params).unwrap();
    let y = a.dot(&x);

    // Check errors
    let matmul_rel_error = (&c - &c_q).mapv(|x| x.abs()).sum() / c.sum();
    let matvec_rel_error = (&y - &y_q).mapv(|x| x.abs()).sum() / y.sum();

    println!("Float16 matmul relative error: {}", matmul_rel_error);
    println!("Float16 matvec relative error: {}", matvec_rel_error);

    // Float16 should give very accurate results for these simple operations
    assert!(matmul_rel_error < 0.001);
    assert!(matvec_rel_error < 0.001);
}

#[test]
fn test_bfloat16_vector_operations() {
    let a = array![1.0_f32, 2.0, 3.0, 4.0];
    let b = array![5.0_f32, 6.0, 7.0, 8.0];

    // Quantize vectors with bfloat16
    let (a_q, a_params) = quantize_vector(&a.view(), 16, QuantizationMethod::BFloat16);
    let (b_q, b_params) = quantize_vector(&b.view(), 16, QuantizationMethod::BFloat16);

    // Perform quantized dot product
    let dot_q = quantized_dot(&a_q, &a_params, &b_q, &b_params).unwrap();

    // Regular dot product for comparison
    let dot = a.dot(&b);

    // Check that the relative error is small for bfloat16
    let rel_error = (dot - dot_q).abs() / dot;
    println!("BFloat16 dot product relative error: {}", rel_error);

    // BFloat16 should have slightly more error than Float16 but still very good
    assert!(rel_error < 0.001);
}

#[test]
fn test_mixed_precision_operations() {
    let a = array![[1.0_f32, 2.0], [3.0, 4.0]];
    let b = array![[5.0_f32, 6.0], [7.0, 8.0]];
    let x = array![5.0_f32, 6.0];

    // Quantize with different float precisions
    let (a_f16, a_f16_params) = quantize_matrix(&a.view(), 16, QuantizationMethod::Float16);
    let (b_bf16, b_bf16_params) = quantize_matrix(&b.view(), 16, QuantizationMethod::BFloat16);
    let (x_bf16, x_bf16_params) = quantize_vector(&x.view(), 16, QuantizationMethod::BFloat16);

    // Test mixed precision operations (Float16 x BFloat16)
    let c_mixed = quantized_matmul(&a_f16, &a_f16_params, &b_bf16, &b_bf16_params).unwrap();
    let y_mixed = quantized_matvec(&a_f16, &a_f16_params, &x_bf16, &x_bf16_params).unwrap();

    // Regular operations for comparison
    let c = a.dot(&b);
    let y = a.dot(&x);

    // Check errors for mixed precision
    let matmul_rel_error = (&c - &c_mixed).mapv(|x| x.abs()).sum() / c.sum();
    let matvec_rel_error = (&y - &y_mixed).mapv(|x| x.abs()).sum() / y.sum();

    println!(
        "Mixed precision matmul relative error: {}",
        matmul_rel_error
    );
    println!(
        "Mixed precision matvec relative error: {}",
        matvec_rel_error
    );

    // Mixed float precisions should have low error
    assert!(matmul_rel_error < 0.001);
    assert!(matvec_rel_error < 0.001);
}
