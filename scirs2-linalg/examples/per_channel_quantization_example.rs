//! Example demonstrating per-channel quantization for machine learning applications
//!
//! Per-channel quantization applies different quantization parameters to each channel
//! (column) of a matrix, which significantly improves accuracy when different channels
//! have very different value distributions. This is particularly important for weights
//! in neural networks.

use scirs2_core::ndarray::array;
use scirs2_linalg::quantization::{
    dequantize_matrix, quantize_matrix, quantize_matrix_per_channel, QuantizationMethod,
};

#[allow(dead_code)]
fn main() {
    println!("Per-Channel Quantization Example");
    println!("===============================\n");

    // Create a simple 2-layer neural network weight matrix
    // Each column has a very different scale - this is common in ML weights
    // - Column 0: small values around 0.1
    // - Column 1: medium values around 10.0
    // - Column 2: large values around 100.0
    let weights = array![
        [0.1, 10.0, 100.0],
        [0.2, 20.0, 200.0],
        [0.3, 30.0, 300.0],
        [0.15, 15.0, 150.0]
    ];

    println!("Original weights matrix:");
    println!("{:?}\n", weights);

    // Standard quantization (same scale for all values)
    let (std_quantized, std_params) =
        quantize_matrix(&weights.view(), 8, QuantizationMethod::Symmetric);

    println!("Standard symmetric quantization parameters:");
    println!("  Scale: {}", std_params.scale);
    println!("  Zero point: {}", std_params.zero_point);
    println!(
        "  Min/Max: [{}, {}]\n",
        std_params.min_val, std_params.max_val
    );

    // Per-channel quantization (different scale for each column)
    let (perchan_quantized, perchan_params) =
        quantize_matrix_per_channel(&weights.view(), 8, QuantizationMethod::PerChannelSymmetric);

    println!("Per-channel symmetric quantization parameters:");
    println!(
        "  Global min/max: [{}, {}]",
        perchan_params.min_val, perchan_params.max_val
    );

    // Print per-channel scales
    if let Some(scales) = &perchan_params.channel_scales {
        println!("  Per-channel scales:");
        for (i, &scale) in scales.iter().enumerate() {
            println!("    Column {}: {}", i, scale);
        }
    }
    println!();

    // Dequantize both versions
    let std_dequantized = dequantize_matrix(&std_quantized, &std_params);
    let perchan_dequantized = dequantize_matrix(&perchan_quantized, &perchan_params);

    // Calculate errors by column
    println!("Error comparison by column (absolute error):");
    println!(
        "{:^10} | {:^20} | {:^20}",
        "Column", "Standard Error", "Per-Channel Error"
    );
    println!("{:-^10} | {:-^20} | {:-^20}", "", "", "");

    for col in 0..weights.ncols() {
        let orig_col = weights.column(col);
        let std_col = std_dequantized.column(col);
        let perchan_col = perchan_dequantized.column(col);

        // Calculate max absolute error
        let std_max_err = (&orig_col - &std_col)
            .mapv(|x| x.abs())
            .fold(0.0_f32, |acc, &x| acc.max(x));
        let perchan_max_err = (&orig_col - &perchan_col)
            .mapv(|x| x.abs())
            .fold(0.0_f32, |acc, &x| acc.max(x));

        println!(
            "{:^10} | {:^20.6} | {:^20.6}",
            col, std_max_err, perchan_max_err
        );
    }
    println!();

    // Calculate overall error
    let std_total_err = (&weights - &std_dequantized).mapv(|x| x.abs()).sum();
    let perchan_total_err = (&weights - &perchan_dequantized).mapv(|x| x.abs()).sum();

    println!("Total absolute error:");
    println!("  Standard quantization: {:.6}", std_total_err);
    println!("  Per-channel quantization: {:.6}", perchan_total_err);
    println!("  Improvement: {:.2}x\n", std_total_err / perchan_total_err);

    // Simulate an activation vector and perform a forward pass
    let activation = array![1.0, 0.5, 0.25, 0.75];

    // True result using original weights
    let true_output = activation.dot(&weights);
    println!("True forward pass output:");
    println!("{:?}\n", true_output);

    // Forward pass with standard quantization
    let std_output = activation.dot(&std_dequantized);
    println!("Standard quantization output:");
    println!("{:?}", std_output);
    println!(
        "Absolute error: {:?}\n",
        (&true_output - &std_output).mapv(|x| x.abs())
    );

    // Forward pass with per-channel quantization
    let perchan_output = activation.dot(&perchan_dequantized);
    println!("Per-channel quantization output:");
    println!("{:?}", perchan_output);
    println!(
        "Absolute error: {:?}\n",
        (&true_output - &perchan_output).mapv(|x| x.abs())
    );

    // Demonstrate with asymmetric data (not centered around zero)
    println!("\nAsymmetric Data Example");
    println!("----------------------\n");

    let asymmetric_weights = array![
        [100.0, 110.0, 500.0],
        [105.0, 120.0, 600.0],
        [115.0, 130.0, 700.0],
        [125.0, 140.0, 800.0]
    ];

    // Compare standard affine with per-channel affine
    let (std_asym_quantized, std_asym_params) =
        quantize_matrix(&asymmetric_weights.view(), 8, QuantizationMethod::Affine);

    let (perchan_asym_quantized, perchan_asym_params) = quantize_matrix_per_channel(
        &asymmetric_weights.view(),
        8,
        QuantizationMethod::PerChannelAffine,
    );

    println!("Standard affine quantization parameters:");
    println!("  Scale: {}", std_asym_params.scale);
    println!("  Zero point: {}", std_asym_params.zero_point);

    println!("\nPer-channel affine quantization parameters:");
    if let Some(scales) = &perchan_asym_params.channel_scales {
        if let Some(zero_points) = &perchan_asym_params.channel_zero_points {
            for i in 0..asymmetric_weights.ncols() {
                println!(
                    "  Column {}: scale = {}, zero_point = {}",
                    i, scales[i], zero_points[i]
                );
            }
        }
    }

    // Dequantize both versions
    let std_asym_dequantized = dequantize_matrix(&std_asym_quantized, &std_asym_params);
    let perchan_asym_dequantized = dequantize_matrix(&perchan_asym_quantized, &perchan_asym_params);

    // Calculate overall error
    let std_asym_total_err = (&asymmetric_weights - &std_asym_dequantized)
        .mapv(|x| x.abs())
        .sum();
    let perchan_asym_total_err = (&asymmetric_weights - &perchan_asym_dequantized)
        .mapv(|x| x.abs())
        .sum();

    println!("\nTotal absolute error for asymmetric data:");
    println!("  Standard affine quantization: {:.6}", std_asym_total_err);
    println!(
        "  Per-channel affine quantization: {:.6}",
        perchan_asym_total_err
    );
    println!(
        "  Improvement: {:.2}x",
        std_asym_total_err / perchan_asym_total_err
    );
}
