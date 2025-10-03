//! Tests for quantized matrix and vector operations
//!
//! Tests for quantized_matmul, quantized_matvec, and quantized_dot operations.

use crate::quantization::{
    quantize_matrix, quantize_vector, quantized_dot, quantized_matmul, quantized_matvec,
    QuantizationMethod,
};
use scirs2_core::ndarray::array;

#[test]
fn test_quantized_matmul() {
    let a = array![[1.0_f32, 2.0], [3.0, 4.0]];
    let b = array![[5.0_f32, 6.0], [7.0, 8.0]];

    // Quantize matrices
    let (a_q, a_params) = quantize_matrix(&a.view(), 8, QuantizationMethod::Symmetric);
    let (b_q, b_params) = quantize_matrix(&b.view(), 8, QuantizationMethod::Symmetric);

    // Perform quantized matrix multiplication
    let c_q = quantized_matmul(&a_q, &a_params, &b_q, &b_params).unwrap();

    // Regular matrix multiplication for comparison
    let c = a.dot(&b);

    // Check that the relative error is small
    let rel_error = (&c - &c_q).mapv(|x| x.abs()).sum() / c.sum();
    assert!(rel_error < 0.1);
}

#[test]
fn test_quantized_matmul_int4() {
    let a = array![[1.0_f32, 2.0], [3.0, 4.0]];
    let b = array![[5.0_f32, 6.0], [7.0, 8.0]];

    // Quantize matrices with Int4
    let (a_q, a_params) = quantize_matrix(&a.view(), 4, QuantizationMethod::Int4);
    let (b_q, b_params) = quantize_matrix(&b.view(), 4, QuantizationMethod::Int4);

    // Perform quantized matrix multiplication
    let c_q = quantized_matmul(&a_q, &a_params, &b_q, &b_params).unwrap();

    // Regular matrix multiplication for comparison
    let c = a.dot(&b);

    // Check that the relative error is acceptable (higher for 4-bit)
    let rel_error = (&c - &c_q).mapv(|x| x.abs()).sum() / c.sum();
    println!("Int4 matmul relative error: {}", rel_error);
    assert!(rel_error < 0.2);
}

#[test]
fn test_quantized_matvec() {
    let a = array![[1.0_f32, 2.0], [3.0, 4.0]];
    let x = array![5.0_f32, 6.0];

    // Quantize matrix and vector
    let (a_q, a_params) = quantize_matrix(&a.view(), 8, QuantizationMethod::Symmetric);
    let (x_q, x_params) = quantize_vector(&x.view(), 8, QuantizationMethod::Symmetric);

    // Perform quantized matrix-vector multiplication
    let y_q = quantized_matvec(&a_q, &a_params, &x_q, &x_params).unwrap();

    // Regular matrix-vector multiplication for comparison
    let y = a.dot(&x);

    // Check that the relative error is small
    let rel_error = (&y - &y_q).mapv(|x| x.abs()).sum() / y.sum();
    assert!(rel_error < 0.1);
}

#[test]
fn test_quantized_matvec_uint4() {
    let a = array![[1.0_f32, 2.0], [3.0, 4.0]];
    let x = array![5.0_f32, 6.0];

    // Quantize matrix and vector with UInt4
    let (a_q, a_params) = quantize_matrix(&a.view(), 4, QuantizationMethod::UInt4);
    let (x_q, x_params) = quantize_vector(&x.view(), 4, QuantizationMethod::UInt4);

    // Perform quantized matrix-vector multiplication
    let y_q = quantized_matvec(&a_q, &a_params, &x_q, &x_params).unwrap();

    // Regular matrix-vector multiplication for comparison
    let y = a.dot(&x);

    // Check that the relative error is acceptable (higher for 4-bit)
    let rel_error = (&y - &y_q).mapv(|x| x.abs()).sum() / y.sum();
    println!("UInt4 matvec relative error: {}", rel_error);
    assert!(rel_error < 1.0);
}

#[test]
fn test_quantized_dot() {
    let a = array![1.0_f32, 2.0, 3.0, 4.0];
    let b = array![5.0_f32, 6.0, 7.0, 8.0];

    // Quantize vectors
    let (a_q, a_params) = quantize_vector(&a.view(), 8, QuantizationMethod::Symmetric);
    let (b_q, b_params) = quantize_vector(&b.view(), 8, QuantizationMethod::Symmetric);

    // Perform quantized dot product
    let dot_q = quantized_dot(&a_q, &a_params, &b_q, &b_params).unwrap();

    // Regular dot product for comparison
    let dot = a.dot(&b);

    // Check that the relative error is small
    let rel_error = (dot - dot_q).abs() / dot;
    assert!(rel_error < 0.1);
}

#[test]
fn test_quantized_dot_mixed() {
    let a = array![1.0_f32, 2.0, 3.0, 4.0];
    let b = array![5.0_f32, 6.0, 7.0, 8.0];

    // Quantize vectors with different methods
    let (a_q, a_params) = quantize_vector(&a.view(), 4, QuantizationMethod::Int4);
    let (b_q, b_params) = quantize_vector(&b.view(), 4, QuantizationMethod::UInt4);

    // Perform quantized dot product with mixed quantization
    let dot_q = quantized_dot(&a_q, &a_params, &b_q, &b_params).unwrap();

    // Regular dot product for comparison
    let dot = a.dot(&b);

    // Check that the relative error is acceptable (higher for mixed 4-bit)
    let rel_error = (dot - dot_q).abs() / dot;
    println!("Mixed Int4/UInt4 dot relative error: {}", rel_error);

    // Mixed precision can have higher error rates, especially with such small vectors
    assert!(rel_error < 0.8);
}
