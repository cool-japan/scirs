//! Example demonstrating quantization for machine learning models
//!
//! This example shows how calibration methods can be applied to
//! typical machine learning model operations like matrix multiplication.

use ndarray::Array2;
use rand::{rng, Rng};
use rand_distr::{Distribution, Normal};
use scirs2_linalg::quantization::calibration::{
    calibrate_matrix, CalibrationConfig, CalibrationMethod,
};
use scirs2_linalg::quantization::{dequantize_matrix, quantize_matrix, quantized_matmul};

#[allow(dead_code)]
fn main() {
    println!("Quantization for Machine Learning Example");
    println!("=========================================\n");

    // Create synthetic weight and activation matrices
    println!("Creating synthetic model weights and activations...");
    let weights = create_model_weights(64, 32);
    let activations = create_activations(10, 64);

    // Compare matmul accuracy with different calibration methods
    println!("\nComparing matrix multiplication accuracy with different calibration methods:");
    compare_matmul_accuracy(&weights, &activations, 8);

    // Compare performance across different bit widths
    println!("\nComparing quantization bit widths for matrix multiplication:");
    compare_bit_widths_matmul(&weights, &activations);

    // Demonstrate mixed precision operations
    println!("\nDemonstrating mixed precision quantization:");
    demonstrate_mixed_precision(&weights, &activations);
}

/// Create synthetic model weights with normal distribution
/// (typical for trained neural networks)
#[allow(dead_code)]
fn create_model_weights(inputsize: usize, outputsize: usize) -> Array2<f32> {
    let mut rng = rand::rng();
    let normal = Normal::new(0.0, 0.1).unwrap(); // Small standard deviation typical for weights

    let mut weights = Array2::zeros((outputsize, inputsize));
    for i in 0..outputsize {
        for j in 0..inputsize {
            weights[[i, j]] = normal.sample(&mut rng);
        }
    }

    weights
}

/// Create synthetic activations with various distributions
#[allow(dead_code)]
fn create_activations(_batchsize: usize, featuresize: usize) -> Array2<f32> {
    let mut rng = rand::rng();

    // Create activations with ReLU-like distribution (many zeros, positive values)
    let mut activations = Array2::zeros((_batchsize, featuresize));
    for i in 0.._batchsize {
        for j in 0..featuresize {
            let val = Normal::new(0.0, 1.0).unwrap().sample(&mut rng);
            activations[[i, j]] = if val > 0.0 { val } else { 0.0 };
        }
    }

    // Add some larger activation values to simulate feature importance
    for _ in 0..5 {
        let i = rng.random_range(0.._batchsize);
        let j = rng.random_range(0..featuresize);
        activations[[i, j]] = rng.random_range(2.0..5.0);
    }

    activations
}

/// Compare matmul accuracy with different calibration methods
#[allow(dead_code)]
fn compare_matmul_accuracy(weights: &Array2<f32>, activations: &Array2<f32>, bits: u8) {
    // Calculate reference result with full precision
    let reference_result = activations.dot(&weights.t());

    // Define calibration methods to compare
    let methods = [
        CalibrationMethod::MinMax,
        CalibrationMethod::PercentileCalibration,
        CalibrationMethod::EntropyCalibration,
        CalibrationMethod::MSEOptimization,
    ];

    println!(
        "{:^25} | {:^12} | {:^12} | {:^15} | {:^15}",
        "Method", "Weight MSE", "Act MSE", "MatMul MSE", "Rel Error (%)"
    );
    println!(
        "{:-^25} | {:-^12} | {:-^12} | {:-^15} | {:-^15}",
        "", "", "", "", ""
    );

    for &method in &methods {
        // Configure calibration for _weights
        let config_weights = CalibrationConfig {
            method,
            symmetric: true,  // Weights often work better with symmetric quantization
            percentile: 0.99, // Exclude extreme outliers for percentile method
            num_bins: 256,
            ..Default::default()
        };

        // Configure calibration for activations
        let config_activations = CalibrationConfig {
            method,
            symmetric: false, // Activations often work better with asymmetric quantization (e.g., ReLU outputs)
            percentile: 0.99,
            num_bins: 256,
            ..Default::default()
        };

        // Calibrate and quantize _weights
        let weights_params = calibrate_matrix(&weights.view(), bits, &config_weights).unwrap();
        let (quantized_weights, _) = quantize_matrix(&weights.view(), bits, weights_params.method);
        let dequantized_weights = dequantize_matrix(&quantized_weights, &weights_params);

        // Calculate _weights quantization error
        let weights_mse =
            (weights - &dequantized_weights).mapv(|x| x * x).sum() / weights.len() as f32;

        // Calibrate and quantize activations
        let activations_params =
            calibrate_matrix(&activations.view(), bits, &config_activations).unwrap();
        let (quantized_activations, _) =
            quantize_matrix(&activations.view(), bits, activations_params.method);
        let dequantized_activations =
            dequantize_matrix(&quantized_activations, &activations_params);

        // Calculate activations quantization error
        let activations_mse = (activations - &dequantized_activations)
            .mapv(|x| x * x)
            .sum()
            / activations.len() as f32;

        // Perform quantized matrix multiplication
        let quantized_result = match quantized_matmul(
            &quantized_weights,
            &weights_params,
            &quantized_activations,
            &activations_params,
        ) {
            Ok(result) => result,
            Err(e) => {
                println!("Error in quantized matmul: {:?}", e);
                // Fallback to dequantized matmul
                dequantized_activations.dot(&dequantized_weights.t())
            }
        };

        // Calculate matrix multiplication error
        let matmul_mse = (&reference_result - &quantized_result)
            .mapv(|x| x * x)
            .sum()
            / reference_result.len() as f32;

        // Calculate relative error as percentage
        let rel_error = (&reference_result - &quantized_result)
            .mapv(|x| x.abs())
            .sum()
            / reference_result.mapv(|x| x.abs()).sum()
            * 100.0;

        // Print results
        println!(
            "{:^25} | {:^12.6} | {:^12.6} | {:^15.6} | {:^15.6}",
            format!("{:?}", method),
            weights_mse,
            activations_mse,
            matmul_mse,
            rel_error
        );
    }
}

/// Compare different bit-widths for matrix multiplication
#[allow(dead_code)]
fn compare_bit_widths_matmul(weights: &Array2<f32>, activations: &Array2<f32>) {
    // Calculate reference result with full precision
    let reference_result = activations.dot(&weights.t());

    // Define bit widths to compare
    let bit_widths = [4, 8, 16];

    println!(
        "{:^10} | {:^15} | {:^15} | {:^15}",
        "Bits", "MatMul MSE", "Rel Error (%)", "Memory Savings (%)"
    );
    println!("{:-^10} | {:-^15} | {:-^15} | {:-^15}", "", "", "", "");

    for &bits in &bit_widths {
        // Use entropy calibration as it often provides good balance
        let config = CalibrationConfig {
            method: CalibrationMethod::EntropyCalibration,
            symmetric: true,
            num_bins: 256,
            ..Default::default()
        };

        // Calibrate and quantize _weights
        let weights_params = calibrate_matrix(&weights.view(), bits, &config).unwrap();
        let (quantized_weights, _) = quantize_matrix(&weights.view(), bits, weights_params.method);

        // Calibrate and quantize activations with asymmetric quantization
        let config_act = CalibrationConfig {
            method: CalibrationMethod::EntropyCalibration,
            symmetric: false,
            num_bins: 256,
            ..Default::default()
        };
        let activations_params = calibrate_matrix(&activations.view(), bits, &config_act).unwrap();
        let (quantized_activations, _) =
            quantize_matrix(&activations.view(), bits, activations_params.method);

        // Perform quantized matrix multiplication
        let quantized_result = match quantized_matmul(
            &quantized_weights,
            &weights_params,
            &quantized_activations,
            &activations_params,
        ) {
            Ok(result) => result,
            Err(_) => {
                // Fallback to dequantized matmul
                let dequantized_weights = dequantize_matrix(&quantized_weights, &weights_params);
                let dequantized_activations =
                    dequantize_matrix(&quantized_activations, &activations_params);
                dequantized_activations.dot(&dequantized_weights.t())
            }
        };

        // Calculate matrix multiplication error
        let matmul_mse = (&reference_result - &quantized_result)
            .mapv(|x| x * x)
            .sum()
            / reference_result.len() as f32;

        // Calculate relative error as percentage
        let rel_error = (&reference_result - &quantized_result)
            .mapv(|x| x.abs())
            .sum()
            / reference_result.mapv(|x| x.abs()).sum()
            * 100.0;

        // Calculate memory savings
        let fp32size = 32;
        let memory_savings = (1.0 - (bits as f32 / fp32size as f32)) * 100.0;

        // Print results
        println!(
            "{:^10} | {:^15.6} | {:^15.6} | {:^15.1}",
            bits, matmul_mse, rel_error, memory_savings
        );
    }
}

/// Demonstrate mixed precision quantization (different bit widths for weights and activations)
#[allow(dead_code)]
fn demonstrate_mixed_precision(weights: &Array2<f32>, activations: &Array2<f32>) {
    // Calculate reference result with full precision
    let reference_result = activations.dot(&weights.t());

    // Define mixed precision configurations to test
    let configs = [
        (8, 8, "Standard (8-bit weights, 8-bit activations)"),
        (4, 8, "Mixed (4-bit weights, 8-bit activations)"),
        (8, 4, "Mixed (8-bit weights, 4-bit activations)"),
        (4, 16, "Mixed (4-bit weights, 16-bit activations)"),
    ];

    println!(
        "{:^40} | {:^15} | {:^15} | {:^15}",
        "Configuration", "MatMul MSE", "Rel Error (%)", "Memory Savings (%)"
    );
    println!("{:-^40} | {:-^15} | {:-^15} | {:-^15}", "", "", "", "");

    for &(weight_bits, act_bits, desc) in &configs {
        // Configure _weights with entropy calibration
        let weights_config = CalibrationConfig {
            method: CalibrationMethod::EntropyCalibration,
            symmetric: true,
            num_bins: 256,
            ..Default::default()
        };

        // Configure activations with percentile calibration (often better for activations)
        let activations_config = CalibrationConfig {
            method: CalibrationMethod::PercentileCalibration,
            symmetric: false,
            percentile: 0.995,
            ..Default::default()
        };

        // Calibrate and quantize _weights
        let weights_params =
            calibrate_matrix(&weights.view(), weight_bits, &weights_config).unwrap();
        let (quantized_weights, _) =
            quantize_matrix(&weights.view(), weight_bits, weights_params.method);

        // Calibrate and quantize activations
        let activations_params =
            calibrate_matrix(&activations.view(), act_bits, &activations_config).unwrap();
        let (quantized_activations, _) =
            quantize_matrix(&activations.view(), act_bits, activations_params.method);

        // For simplicity, dequantize and perform standard matmul since mixed precision is for demonstration
        let dequantized_weights = dequantize_matrix(&quantized_weights, &weights_params);
        let dequantized_activations =
            dequantize_matrix(&quantized_activations, &activations_params);
        let mixed_result = dequantized_activations.dot(&dequantized_weights.t());

        // Calculate matrix multiplication error
        let matmul_mse = (&reference_result - &mixed_result).mapv(|x| x * x).sum()
            / reference_result.len() as f32;

        // Calculate relative error as percentage
        let rel_error = (&reference_result - &mixed_result).mapv(|x| x.abs()).sum()
            / reference_result.mapv(|x| x.abs()).sum()
            * 100.0;

        // Calculate memory savings (weighted average based on typical model composition)
        // Assuming _weights are 75% of model size, activations 25%
        let fp32size = 32;
        let weight_savings = 1.0 - (weight_bits as f32 / fp32size as f32);
        let act_savings = 1.0 - (act_bits as f32 / fp32size as f32);
        let memory_savings = (weight_savings * 0.75 + act_savings * 0.25) * 100.0;

        // Print results
        println!(
            "{:^40} | {:^15.6} | {:^15.6} | {:^15.1}",
            desc, matmul_mse, rel_error, memory_savings
        );
    }
}
