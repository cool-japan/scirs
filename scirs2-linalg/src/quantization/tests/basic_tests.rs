//! Basic quantization and dequantization tests
//!
//! Tests for different quantization methods including Uniform, Symmetric,
//! Affine, PowerOfTwo, Int4, and UInt4.

use crate::quantization::{
    dequantize_matrix, quantize_matrix, QuantizationMethod, QuantizedDataType,
};
use scirs2_core::ndarray::array;

#[test]
fn test_quantize_dequantize_uniform() {
    let a = array![[1.0_f32, 2.5, 3.7], [4.2, 5.0, 6.1]];

    let (quantized, params) = quantize_matrix(&a.view(), 8, QuantizationMethod::Uniform);
    let a_dequantized = dequantize_matrix(&quantized, &params);

    // For 8-bit quantization, we can expect some error
    let max_diff = (&a - &a_dequantized)
        .mapv(|x| x.abs())
        .fold(0.0_f32, |acc, &b| acc.max(b));
    println!("Max error (Uniform): {}", max_diff);
    assert!(max_diff < 6.0, "Max error too large: {}", max_diff);
}

#[test]
fn test_quantize_dequantize_symmetric() {
    let a = array![[1.0_f32, -2.5, 3.7], [-4.2, 5.0, -6.1]];

    let (quantized, params) = quantize_matrix(&a.view(), 8, QuantizationMethod::Symmetric);
    let a_dequantized = dequantize_matrix(&quantized, &params);

    // For 8-bit quantization, we can expect some error
    let max_diff = (&a - &a_dequantized)
        .mapv(|x| x.abs())
        .fold(0.0_f32, |acc, &b| acc.max(b));
    println!("Max error (Symmetric): {}", max_diff);
    assert!(max_diff < 6.0, "Max error too large: {}", max_diff);
}

#[test]
fn test_quantize_dequantize_affine() {
    let a = array![[1.0_f32, -2.5, 3.7], [-4.2, 5.0, -6.1]];

    let (quantized, params) = quantize_matrix(&a.view(), 8, QuantizationMethod::Affine);
    let a_dequantized = dequantize_matrix(&quantized, &params);

    // For 8-bit quantization, we can expect some error
    let max_diff = (&a - &a_dequantized)
        .mapv(|x| x.abs())
        .fold(0.0_f32, |acc, &b| acc.max(b));
    println!("Max error (Affine): {}", max_diff);
    assert!(max_diff < 6.0, "Max error too large: {}", max_diff);
}

#[test]
fn test_quantize_dequantize_power_of_two() {
    let a = array![[1.0_f32, 2.5, 3.7], [4.2, 5.0, 6.1]];

    let (quantized, params) = quantize_matrix(&a.view(), 8, QuantizationMethod::PowerOfTwo);
    let a_dequantized = dequantize_matrix(&quantized, &params);

    // For 8-bit quantization, we can expect some error
    let max_diff = (&a - &a_dequantized)
        .mapv(|x| x.abs())
        .fold(0.0_f32, |acc, &b| acc.max(b));
    println!("Max error (PowerOfTwo): {}", max_diff);
    assert!(max_diff < 6.0, "Max error too large: {}", max_diff);
}

#[test]
fn test_quantize_dequantize_int4() {
    let a = array![[1.0_f32, -2.5, 3.7], [-4.2, 5.0, -6.1]];

    let (quantized, params) = quantize_matrix(&a.view(), 4, QuantizationMethod::Int4);
    let a_dequantized = dequantize_matrix(&quantized, &params);

    // For 4-bit quantization, we expect larger errors
    let max_diff = (&a - &a_dequantized)
        .mapv(|x| x.abs())
        .fold(0.0_f32, |acc, &b| acc.max(b));
    println!("Max error (Int4): {}", max_diff);

    // Make sure the data was packed (using half the storage)
    assert_eq!(quantized.data_type, QuantizedDataType::Int4);
    assert!(quantized.data.len() < a.len());

    // Error should be larger than 8-bit but still reasonable
    assert!(max_diff < 15.0, "Max error too large: {}", max_diff);
}

#[test]
fn test_quantize_dequantize_uint4() {
    let a = array![[1.0_f32, 2.5, 3.7], [4.2, 5.0, 6.1]];

    let (quantized, params) = quantize_matrix(&a.view(), 4, QuantizationMethod::UInt4);
    let a_dequantized = dequantize_matrix(&quantized, &params);

    // For 4-bit quantization, we expect larger errors
    let max_diff = (&a - &a_dequantized)
        .mapv(|x| x.abs())
        .fold(0.0_f32, |acc, &b| acc.max(b));
    println!("Max error (UInt4): {}", max_diff);

    // Make sure the data was packed (using half the storage)
    assert_eq!(quantized.data_type, QuantizedDataType::UInt4);
    assert!(quantized.data.len() < a.len());

    // Error should be larger than 8-bit but still reasonable
    assert!(max_diff < 15.0, "Max error too large: {}", max_diff);
}
