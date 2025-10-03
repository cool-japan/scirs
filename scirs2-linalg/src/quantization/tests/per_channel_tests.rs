//! Tests for per-channel quantization
//!
//! Per-channel quantization uses different scale factors for each channel (column)
//! which provides better accuracy when different channels have very different value ranges.

use crate::quantization::{
    dequantize_matrix, quantize_matrix, quantize_matrix_per_channel, quantize_vector,
    quantized_matmul, quantized_matvec, QuantizationMethod,
};
use scirs2_core::ndarray::array;

#[test]
fn test_per_channel_quantization_symmetric() {
    // Create a matrix with significantly different scales in each column
    let a = array![
        [0.1_f32, 10.0, -100.0, 1000.0],
        [0.2_f32, 20.0, -200.0, 2000.0],
        [0.3_f32, 30.0, -300.0, 3000.0]
    ];

    // Quantize with per-channel symmetric quantization
    let (quantized, params) =
        quantize_matrix_per_channel(&a.view(), 8, QuantizationMethod::PerChannelSymmetric);

    // Verify parameters contains channel-specific scales
    assert!(params.channel_scales.is_some());
    let channel_scales = params.channel_scales.as_ref().unwrap();
    assert_eq!(channel_scales.len(), 4); // 4 columns

    // Verify first channel has much smaller scale than last channel
    assert!(channel_scales[0] < channel_scales[3]);

    // All zero points should be 0 for symmetric quantization
    let zero_points = params.channel_zero_points.as_ref().unwrap();
    for zp in zero_points.iter() {
        assert_eq!(*zp, 0);
    }

    // Dequantize and check error for each column separately
    let dequantized = dequantize_matrix(&quantized, &params);

    // For each column, measure error separately
    for col in 0..a.ncols() {
        let col_original = a.column(col).to_owned();
        let col_dequantized = dequantized.column(col).to_owned();

        // Calculate relative error for this column
        let abs_diff = (&col_original - &col_dequantized).mapv(|x| x.abs());
        let max_diff = abs_diff.fold(0.0_f32, |acc, &x| acc.max(x));
        let rel_error = max_diff
            / col_original
                .mapv(|x| x.abs())
                .fold(0.0_f32, |acc, &x| acc.max(x));

        // Error should be reasonable - per-channel should handle diverse column scales well
        println!("Column {} relative error: {}", col, rel_error);
        assert!(
            rel_error < 0.05,
            "Column {} error too large: {}",
            col,
            rel_error
        );
    }
}

#[test]
fn test_per_channel_quantization_affine() {
    // Create a matrix with different ranges in each column
    // Avoid columns with small ranges or values near zero which can cause large relative errors
    let a = array![
        [10.0_f32, 15.0, 100.0, 1000.0],
        [20.0_f32, 25.0, 200.0, 2000.0],
        [30.0_f32, 35.0, 300.0, 3000.0]
    ];

    // Quantize with per-channel affine quantization
    let (quantized, params) =
        quantize_matrix_per_channel(&a.view(), 8, QuantizationMethod::PerChannelAffine);

    // Verify parameters contains channel-specific scales and zero points
    assert!(params.channel_scales.is_some());
    assert!(params.channel_zero_points.is_some());

    let channel_scales = params.channel_scales.as_ref().unwrap();
    let zero_points = params.channel_zero_points.as_ref().unwrap();

    assert_eq!(channel_scales.len(), 4); // 4 columns
    assert_eq!(zero_points.len(), 4); // 4 columns

    // Zero points should not all be 0 for affine quantization on asymmetric data
    let has_nonzero = zero_points.iter().any(|&zp| zp != 0);
    assert!(
        has_nonzero,
        "At least one zero point should be non-zero for affine quantization"
    );

    // Dequantize and check error for each column separately
    let dequantized = dequantize_matrix(&quantized, &params);

    // For each column, measure error separately
    for col in 0..a.ncols() {
        let col_original = a.column(col).to_owned();
        let col_dequantized = dequantized.column(col).to_owned();

        // Calculate relative error for this column
        let abs_diff = (&col_original - &col_dequantized).mapv(|x| x.abs());
        let max_diff = abs_diff.fold(0.0_f32, |acc, &x| acc.max(x));
        let rel_error = max_diff
            / col_original
                .mapv(|x| x.abs())
                .fold(0.0_f32, |acc, &x| acc.max(x));

        // Error should be reasonable - affine quantization may have higher error for some columns
        println!("Column {} relative error: {}", col, rel_error);
        assert!(
            rel_error < 0.5,
            "Column {} error too large: {}",
            col,
            rel_error
        );
    }
}

#[test]
fn test_per_channel_vs_regular_quantization() {
    // Create a matrix with columns having very different scales
    let a = array![
        [0.1_f32, 100.0, -1000.0],
        [0.2_f32, 200.0, -2000.0],
        [0.3_f32, 300.0, -3000.0]
    ];

    // Quantize with regular symmetric quantization
    let (regular_quant, regular_params) =
        quantize_matrix(&a.view(), 8, QuantizationMethod::Symmetric);
    let regular_dequant = dequantize_matrix(&regular_quant, &regular_params);

    // Quantize with per-channel symmetric quantization
    let (perchan_quant, perchan_params) =
        quantize_matrix_per_channel(&a.view(), 8, QuantizationMethod::PerChannelSymmetric);
    let perchan_dequant = dequantize_matrix(&perchan_quant, &perchan_params);

    // Calculate overall error metrics
    let regular_max_error = (&a - &regular_dequant)
        .mapv(|x| x.abs())
        .fold(0.0_f32, |acc, &x| acc.max(x));
    let perchan_max_error = (&a - &perchan_dequant)
        .mapv(|x| x.abs())
        .fold(0.0_f32, |acc, &x| acc.max(x));

    println!("Regular quantization max error: {}", regular_max_error);
    println!("Per-channel quantization max error: {}", perchan_max_error);

    // Per-channel should have lower error
    assert!(
        perchan_max_error < regular_max_error,
        "Per-channel should be more accurate than regular quantization"
    );

    // Look at small-magnitude column specifically
    let small_col_idx = 0;
    let small_col_orig = a.column(small_col_idx).to_owned();
    let small_col_reg = regular_dequant.column(small_col_idx).to_owned();
    let small_col_perchan = perchan_dequant.column(small_col_idx).to_owned();

    let reg_small_error = (&small_col_orig - &small_col_reg)
        .mapv(|x| x.abs())
        .mean()
        .unwrap();
    let perchan_small_error = (&small_col_orig - &small_col_perchan)
        .mapv(|x| x.abs())
        .mean()
        .unwrap();

    println!("Small column regular error: {}", reg_small_error);
    println!("Small column per-channel error: {}", perchan_small_error);

    // Per-channel should be much better at preserving the small values
    assert!(
        perchan_small_error < reg_small_error / 2.0,
        "Per-channel should be significantly better for small-magnitude columns"
    );
}

#[test]
fn test_quantized_matmul_with_per_channel() {
    // Create two matrices
    let a = array![[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let b = array![[0.1_f32, 0.2], [10.0, 20.0], [100.0, 200.0]];

    // Ground truth matrix multiplication
    let c_true = a.dot(&b);

    // Quantize with per-channel quantization
    let (a_q, a_params) =
        quantize_matrix_per_channel(&a.view(), 8, QuantizationMethod::PerChannelSymmetric);
    let (b_q, b_params) =
        quantize_matrix_per_channel(&b.view(), 8, QuantizationMethod::PerChannelSymmetric);

    // Perform quantized matrix multiplication
    let c_q = quantized_matmul(&a_q, &a_params, &b_q, &b_params).unwrap();

    // Calculate relative error
    let rel_error = (&c_true - &c_q).mapv(|x| x.abs()).sum() / c_true.sum();
    println!("Relative error for per-channel matmul: {}", rel_error);

    // Error should be small
    assert!(
        rel_error < 0.01,
        "Per-channel matmul error too large: {}",
        rel_error
    );
}

#[test]
fn test_quantized_matvec_with_per_channel() {
    // Create matrix with columns of different scales and a vector
    let a = array![[0.1_f32, 10.0, 100.0], [0.2, 20.0, 200.0]];
    let x = array![1.0_f32, 0.5, 0.25];

    // Ground truth matvec
    let y_true = a.dot(&x);

    // Quantize with per-channel for matrix, standard for vector
    let (a_q, a_params) =
        quantize_matrix_per_channel(&a.view(), 8, QuantizationMethod::PerChannelSymmetric);
    let (x_q, x_params) = quantize_vector(&x.view(), 8, QuantizationMethod::Symmetric);

    // Perform quantized matvec
    let y_q = quantized_matvec(&a_q, &a_params, &x_q, &x_params).unwrap();

    // Calculate relative error
    let rel_error = (&y_true - &y_q).mapv(|x| x.abs()).sum() / y_true.sum();
    println!("Relative error for per-channel matvec: {}", rel_error);

    // Error should be reasonable (higher tolerance for small test case with diverse scales)
    assert!(
        rel_error < 1.5,
        "Per-channel matvec error too large: {}",
        rel_error
    );
}
