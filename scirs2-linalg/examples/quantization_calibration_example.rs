//! Example demonstrating quantization calibration utilities
//!
//! This example shows how different calibration methods can be used
//! to select optimal quantization parameters for various data distributions.

use scirs2_core::ndarray::Array2;
use scirs2_core::random::{rng, Rng};
use scirs2_core::random::{Distribution, Normal, Uniform};
use scirs2_linalg::quantization::calibration::{
    calibrate_matrix, CalibrationConfig, CalibrationMethod,
};
use scirs2_linalg::quantization::{dequantize_matrix, quantize_matrix};

#[allow(dead_code)]
fn main() {
    println!("Quantization Calibration Example");
    println!("================================\n");

    // Create a synthetic dataset with different distributions
    println!("Creating synthetic data with multiple distributions...");
    let uniformdata = create_uniformdata();
    let normaldata = create_normaldata();
    let bimodaldata = create_bimodaldata();
    let mixeddata = create_mixed_scaledata();

    // Compare different calibration methods
    println!("\nComparing calibration methods on uniform distribution:");
    compare_calibration_methods(&uniformdata, 8);

    println!("\nComparing calibration methods on normal distribution:");
    compare_calibration_methods(&normaldata, 8);

    println!("\nComparing calibration methods on bimodal distribution:");
    compare_calibration_methods(&bimodaldata, 8);

    // Compare per-channel vs standard quantization
    println!("\nComparing per-channel quantization on mixed scale data:");
    compare_per_channel_quantization(&mixeddata, 8);

    // Compare different bit-width quantization
    println!("\nComparing different bit-widths using entropy calibration:");
    compare_bit_widths(&normaldata);
}

/// Create a matrix with uniform distribution
#[allow(dead_code)]
fn create_uniformdata() -> Array2<f32> {
    let mut rng = scirs2_core::random::rng();
    let uniform = Uniform::new(-1.0, 1.0).unwrap();

    let mut data = Array2::zeros((10, 10));
    for i in 0..10 {
        for j in 0..10 {
            data[[i, j]] = uniform.sample(&mut rng);
        }
    }

    data
}

/// Create a matrix with normal distribution
#[allow(dead_code)]
fn create_normaldata() -> Array2<f32> {
    let mut rng = scirs2_core::random::rng();
    let normal = Normal::new(0.0, 1.0).unwrap();

    let mut data = Array2::zeros((10, 10));
    for i in 0..10 {
        for j in 0..10 {
            data[[i, j]] = normal.sample(&mut rng);
        }
    }

    data
}

/// Create a matrix with bimodal distribution
#[allow(dead_code)]
fn create_bimodaldata() -> Array2<f32> {
    let mut rng = scirs2_core::random::rng();
    let normal1 = Normal::new(-2.0, 0.5).unwrap();
    let normal2 = Normal::new(2.0, 0.5).unwrap();

    let mut data = Array2::zeros((10, 10));
    for i in 0..10 {
        for j in 0..10 {
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

/// Create a matrix with mixed scales in different columns
#[allow(dead_code)]
fn create_mixed_scaledata() -> Array2<f32> {
    let mut rng = scirs2_core::random::rng();

    let mut data = Array2::zeros((10, 3));

    // Column 0: small values around 0.1
    for i in 0..10 {
        data[[i, 0]] = 0.1 + 0.05 * rng.random::<f32>();
    }

    // Column 1: medium values around 10.0
    for i in 0..10 {
        data[[i, 1]] = 10.0 + 5.0 * rng.random::<f32>();
    }

    // Column 2: large values around 100.0
    for i in 0..10 {
        data[[i, 2]] = 100.0 + 50.0 * rng.random::<f32>();
    }

    data
}

/// Compare different calibration methods on the same data
#[allow(dead_code)]
fn compare_calibration_methods(data: &Array2<f32>, bits: u8) {
    let methods = [
        CalibrationMethod::MinMax,
        CalibrationMethod::MovingAverageMinMax,
        CalibrationMethod::PercentileCalibration,
        CalibrationMethod::EntropyCalibration,
        CalibrationMethod::MSEOptimization,
    ];

    println!(
        "{:^20} | {:^15} | {:^15} | {:^15} | {:^10}",
        "Method", "Scale", "Min", "Max", "MSE"
    );
    println!(
        "{:-^20} | {:-^15} | {:-^15} | {:-^15} | {:-^10}",
        "", "", "", "", ""
    );

    for &method in &methods {
        // Configure calibration
        let config = CalibrationConfig {
            method,
            symmetric: true,
            percentile: 0.995,
            num_bins: 100,
            windowsize: 3,
            per_channel: false,
            ema_factor: 0.9,
            convergence_threshold: 0.01,
            max_iterations: 100,
        };

        // Calibrate parameters
        let params = calibrate_matrix(&data.view(), bits, &config).unwrap();

        // Quantize and dequantize
        let (quantized, _) = quantize_matrix(&data.view(), bits, params.method);
        let dequantized = dequantize_matrix(&quantized, &params);

        // Calculate MSE
        let mse = (data - &dequantized).mapv(|x| x * x).sum() / data.len() as f32;

        // Print results
        println!(
            "{:^20} | {:^15.6} | {:^15.6} | {:^15.6} | {:^10.6}",
            format!("{:?}", method),
            params.scale,
            params.min_val,
            params.max_val,
            mse
        );
    }
}

/// Compare per-channel vs standard quantization
#[allow(dead_code)]
fn compare_per_channel_quantization(data: &Array2<f32>, bits: u8) {
    println!("Standard Symmetric Quantization:");
    let config_std = CalibrationConfig {
        method: CalibrationMethod::MinMax,
        symmetric: true,
        per_channel: false,
        ..Default::default()
    };

    let params_std = calibrate_matrix(&data.view(), bits, &config_std).unwrap();
    let (quantized_std_, _) = quantize_matrix(&data.view(), bits, params_std.method);
    let dequantized_std = dequantize_matrix(&quantized_std_, &params_std);

    let mse_std = (data - &dequantized_std).mapv(|x| x * x).sum() / data.len() as f32;

    println!("  Global scale: {}", params_std.scale);
    println!("  MSE: {}\n", mse_std);

    println!("Per-Channel Symmetric Quantization:");
    let config_pc = CalibrationConfig {
        method: CalibrationMethod::MinMax,
        symmetric: true,
        per_channel: true,
        ..Default::default()
    };

    let params_pc = calibrate_matrix(&data.view(), bits, &config_pc).unwrap();

    // Print per-channel scales
    if let Some(scales) = &params_pc.channel_scales {
        for (i, &scale) in scales.iter().enumerate() {
            println!("  Channel {} scale: {}", i, scale);
        }
    }

    let (quantized_pc_, _) = quantize_matrix(&data.view(), bits, params_pc.method);
    let dequantized_pc = dequantize_matrix(&quantized_pc_, &params_pc);

    let mse_pc = (data - &dequantized_pc).mapv(|x| x * x).sum() / data.len() as f32;

    println!("  MSE: {}", mse_pc);
    println!("  Improvement: {:.2}x", mse_std / mse_pc);

    // Compare error by column
    let (_, cols) = data.dim();
    println!("\nError comparison by column (MSE):");
    println!(
        "{:^10} | {:^15} | {:^15}",
        "Column", "Standard", "Per-Channel"
    );
    println!("{:-^10} | {:-^15} | {:-^15}", "", "", "");

    for j in 0..cols {
        let coldata = data.column(j);
        let col_std = dequantized_std.column(j);
        let col_pc = dequantized_pc.column(j);

        let col_mse_std = (&coldata - &col_std).mapv(|x| x * x).sum() / coldata.len() as f32;

        let col_mse_pc = (&coldata - &col_pc).mapv(|x| x * x).sum() / coldata.len() as f32;

        println!("{:^10} | {:^15.6} | {:^15.6}", j, col_mse_std, col_mse_pc);
    }
}

/// Compare different bit-widths using entropy calibration
#[allow(dead_code)]
fn compare_bit_widths(data: &Array2<f32>) {
    let bits = [4, 8, 16];

    println!(
        "{:^10} | {:^15} | {:^15} | {:^10}",
        "Bits", "Scale", "MSE", "Rel Error"
    );
    println!("{:-^10} | {:-^15} | {:-^15} | {:-^10}", "", "", "", "");

    for &bit in &bits {
        // Configure calibration
        let config = CalibrationConfig {
            method: CalibrationMethod::EntropyCalibration,
            symmetric: true,
            per_channel: false,
            ..Default::default()
        };

        // Calibrate parameters
        let params = calibrate_matrix(&data.view(), bit, &config).unwrap();

        // Quantize and dequantize
        let (quantized, _) = quantize_matrix(&data.view(), bit, params.method);
        let dequantized = dequantize_matrix(&quantized, &params);

        // Calculate MSE
        let mse = (data - &dequantized).mapv(|x| x * x).sum() / data.len() as f32;

        // Calculate relative error
        let rel_error =
            (data - &dequantized).mapv(|x| x.abs()).sum() / data.mapv(|x| x.abs()).sum();

        // Print results
        println!(
            "{:^10} | {:^15.6} | {:^15.6} | {:^10.6}",
            bit, params.scale, mse, rel_error
        );
    }
}
