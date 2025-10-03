//! Tests for fake quantization
//!
//! Fake quantization simulates quantization effects without actually storing in reduced precision.
//! This is useful for quantization-aware training (QAT).

use crate::quantization::{fake_quantize, fake_quantize_vector, QuantizationMethod};
use scirs2_core::ndarray::array;

#[test]
fn test_fake_quantize() {
    let a = array![[1.0_f32, 2.5, 3.7], [4.2, 5.0, 6.1]];

    let a_fake_q = fake_quantize(&a.view(), 8, QuantizationMethod::Uniform);

    // For 8-bit quantization, we can expect some error
    let max_diff = (&a - &a_fake_q)
        .mapv(|x| x.abs())
        .fold(0.0_f32, |acc, &b| acc.max(b));
    println!("Max error (Fake Quantize): {}", max_diff);
    assert!(max_diff < 6.0, "Max error too large: {}", max_diff);

    // Check that values are different due to quantization
    assert!(a != a_fake_q);
}

#[test]
fn test_fake_quantize_int4() {
    let a = array![[1.0_f32, 2.5, 3.7], [4.2, 5.0, 6.1]];

    let a_fake_q = fake_quantize(&a.view(), 4, QuantizationMethod::Int4);

    // For 4-bit quantization, we expect larger errors
    let max_diff = (&a - &a_fake_q)
        .mapv(|x| x.abs())
        .fold(0.0_f32, |acc, &b| acc.max(b));
    println!("Max error (Fake Quantize Int4): {}", max_diff);

    // Error should be larger than 8-bit but still reasonable
    assert!(max_diff < 10.0, "Max error too large: {}", max_diff);

    // Check that values are different due to quantization
    assert!(a != a_fake_q);
}

#[test]
fn test_fake_quantize_vector() {
    let a = array![1.0_f32, 2.5, 3.7, 4.2, 5.0, 6.1];

    let a_fake_q = fake_quantize_vector(&a.view(), 8, QuantizationMethod::Uniform);

    // For 8-bit quantization, we can expect some error
    let max_diff = (&a - &a_fake_q)
        .mapv(|x| x.abs())
        .fold(0.0_f32, |acc, &b| acc.max(b));
    println!("Max error (Fake Quantize Vector): {}", max_diff);
    assert!(max_diff < 6.0, "Max error too large: {}", max_diff);

    // Check that values are different due to quantization
    assert!(a != a_fake_q);
}

#[test]
fn test_fake_quantize_vector_uint4() {
    let a = array![1.0_f32, 2.5, 3.7, 4.2, 5.0, 6.1];

    let a_fake_q = fake_quantize_vector(&a.view(), 4, QuantizationMethod::UInt4);

    // For 4-bit quantization, we expect larger errors
    let max_diff = (&a - &a_fake_q)
        .mapv(|x| x.abs())
        .fold(0.0_f32, |acc, &b| acc.max(b));
    println!("Max error (Fake Quantize Vector UInt4): {}", max_diff);

    // Error should be larger than 8-bit but still reasonable
    assert!(max_diff < 10.0, "Max error too large: {}", max_diff);

    // Check that values are different due to quantization
    assert!(a != a_fake_q);
}
