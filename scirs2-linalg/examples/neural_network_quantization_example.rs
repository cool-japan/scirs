//! Example demonstrating neural network quantization using helper utilities
//!
//! This example shows how to use the neural network-specific helper functions
//! for quantizing different components of a neural network.

use ndarray::{Array2, ArrayView2};
use rand::rng;
use rand_distr::{Distribution, Normal};
use scirs2_linalg::prelude::*;
use scirs2_linalg::quantization::{
    dequantize_matrix, quantize_matrix, quantized_matmul, QuantizationParams, QuantizedMatrix,
};
use scirs2_linalg::{
    calibrate_matrix, get_activation_calibration_config, get_weight_calibration_config,
};

// Simple struct to represent a neural network layer
struct SimpleLayer {
    weights: Array2<f32>,
    biases: Array2<f32>,
    name: String,
}

fn main() {
    println!("Neural Network Quantization Example");
    println!("===================================\n");

    // Create a simple example network (2 layers)
    println!("Creating example neural network...");
    let network = create_example_network();

    // Show original network behavior
    println!("\nRunning inference with full precision...");
    let test_input = create_test_input(5, 32);
    let full_precision_output = run_network_full_precision(&network, &test_input);

    // Quantize network
    println!("\nQuantizing network...");
    let (quantized_network, quantization_params) = quantize_network(&network, 8);

    // Run inference with quantized network
    println!("\nRunning inference with quantized network...");
    let quantized_output = run_network_quantized(
        &network,
        &quantized_network,
        &quantization_params,
        &test_input,
    );

    // Compare results
    println!("\nComparing results...");
    compare_outputs(&full_precision_output, &quantized_output);

    // Demonstrate per-layer bit width selection
    println!("\nDemonstrating mixed precision quantization (per-layer bit width selection)...");
    mixed_precision_quantization(&network, &test_input);
}

/// Create a simple example network with 2 layers
fn create_example_network() -> Vec<SimpleLayer> {
    let mut rng = rng();

    // Create first layer (32 input features -> 64 hidden features)
    // Use Kaiming/He initialization for weights (assuming ReLU activation)
    let std_dev1 = (2.0 / 32.0).sqrt();
    let mut weights1 = Array2::zeros((64, 32));
    for i in 0..weights1.dim().0 {
        for j in 0..weights1.dim().1 {
            weights1[[i, j]] = Normal::new(0.0, std_dev1).unwrap().sample(&mut rng);
        }
    }
    let biases1 = Array2::zeros((64, 1));

    // Create second layer (64 hidden features -> 10 output features)
    let std_dev2 = (2.0 / 64.0).sqrt();
    let mut weights2 = Array2::zeros((10, 64));
    for i in 0..weights2.dim().0 {
        for j in 0..weights2.dim().1 {
            weights2[[i, j]] = Normal::new(0.0, std_dev2).unwrap().sample(&mut rng);
        }
    }
    let mut biases2 = Array2::zeros((10, 1));
    // Initialize biases with small values
    for i in 0..biases2.dim().0 {
        biases2[[i, 0]] = 0.01 * Normal::new(0.0, 1.0).unwrap().sample(&mut rng);
    }

    // Return the network
    vec![
        SimpleLayer {
            weights: weights1,
            biases: biases1,
            name: "hidden_layer".to_string(),
        },
        SimpleLayer {
            weights: weights2,
            biases: biases2,
            name: "output_layer".to_string(),
        },
    ]
}

/// Create test input data
fn create_test_input(batch_size: usize, input_size: usize) -> Array2<f32> {
    let mut rng = rng();
    let mut input = Array2::zeros((batch_size, input_size));

    for i in 0..batch_size {
        for j in 0..input_size {
            input[[i, j]] = Normal::new(0.0, 1.0).unwrap().sample(&mut rng);
        }
    }

    input
}

/// ReLU activation function
fn relu(x: &ArrayView2<f32>) -> Array2<f32> {
    x.mapv(|v| if v > 0.0 { v } else { 0.0 })
}

/// Run inference with full precision network
fn run_network_full_precision(network: &[SimpleLayer], input: &Array2<f32>) -> Array2<f32> {
    // First layer
    let layer1 = &network[0];
    let hidden = input.dot(&layer1.weights.t());
    let hidden_bias = &hidden + &layer1.biases.t(); // Add biases
    let hidden_activated = relu(&hidden_bias.view());

    // Second layer
    let layer2 = &network[1];
    let output = hidden_activated.dot(&layer2.weights.t());
    let output_bias = &output + &layer2.biases.t(); // Add biases

    output_bias
}

type QuantizedLayerPair = (QuantizedMatrix, QuantizedMatrix);
type QuantizationParamsPair = (QuantizationParams, QuantizationParams);

/// Quantize a neural network
fn quantize_network(
    network: &[SimpleLayer],
    bits: u8,
) -> (Vec<QuantizedLayerPair>, Vec<QuantizationParamsPair>) {
    let mut quantized_layers = Vec::new();
    let mut quantization_params = Vec::new();

    for (i, layer) in network.iter().enumerate() {
        println!("Quantizing layer {}: {}", i, layer.name);

        // Use helper functions to get recommended calibration configs
        let weights_config = get_weight_calibration_config(bits, false);
        let bias_config = get_weight_calibration_config(bits, true); // More aggressive for biases

        // Quantize weights
        println!("  Weights shape: {:?}", layer.weights.dim());
        let weights_params =
            calibrate_matrix(&layer.weights.view(), bits, &weights_config).unwrap();
        let (quantized_weights, _) =
            quantize_matrix(&layer.weights.view(), bits, weights_params.method);

        // Quantize biases
        println!("  Biases shape: {:?}", layer.biases.dim());
        let bias_params = calibrate_matrix(&layer.biases.view(), bits, &bias_config).unwrap();
        let (quantized_biases, _) = quantize_matrix(&layer.biases.view(), bits, bias_params.method);

        // Save quantized weights and biases
        quantized_layers.push((quantized_weights, quantized_biases));
        quantization_params.push((weights_params.clone(), bias_params));

        // Print quantization statistics
        println!("  Weight scale: {:.6}", weights_params.scale);
        if let Some(channel_scales) = &weights_params.channel_scales {
            let min_scale = channel_scales.iter().fold(f32::MAX, |a, &b| a.min(b));
            let max_scale = channel_scales.iter().fold(f32::MIN, |a, &b| a.max(b));
            println!(
                "  Per-channel scales: min={:.6}, max={:.6}",
                min_scale, max_scale
            );
        }
    }

    // Return the quantized layers and parameters
    (quantized_layers, quantization_params)
}

/// Run inference with quantized network
fn run_network_quantized(
    network: &[SimpleLayer],
    quantized_network: &[(QuantizedMatrix, QuantizedMatrix)],
    quantization_params: &[(QuantizationParams, QuantizationParams)],
    input: &Array2<f32>,
) -> Array2<f32> {
    // First layer
    let _layer = &network[0];
    let (q_weights, q_biases) = &quantized_network[0];
    let (weights_params, bias_params) = &quantization_params[0];

    // For activations, we need to quantize the input
    let activation_config = get_activation_calibration_config(8, false, false);
    let act_params = calibrate_matrix(&input.view(), 8, &activation_config).unwrap();
    let (q_input, _) = quantize_matrix(&input.view(), 8, act_params.method);

    // Perform quantized matrix multiplication for first layer
    let hidden = match quantized_matmul(q_weights, weights_params, &q_input, &act_params) {
        Ok(result) => result,
        Err(e) => {
            println!("Error in quantized matmul: {:?}", e);
            // Fallback to dequantized computation
            let dq_weights = dequantize_matrix(q_weights, weights_params);
            let dq_input = dequantize_matrix(&q_input, &act_params);
            dq_input.dot(&dq_weights.t())
        }
    };

    // Add biases (dequantize biases)
    let dq_biases = dequantize_matrix(q_biases, bias_params);
    let hidden_bias = &hidden + &dq_biases.t();

    // Apply ReLU activation
    let hidden_activated = relu(&hidden_bias.view());

    // For second layer, quantize the activations from first layer
    let hidden_config = get_activation_calibration_config(8, true, false); // ReLU output is non-negative
    let hidden_params = calibrate_matrix(&hidden_activated.view(), 8, &hidden_config).unwrap();
    let (q_hidden, _) = quantize_matrix(&hidden_activated.view(), 8, hidden_params.method);

    // Second layer
    let (q_weights2, q_biases2) = &quantized_network[1];
    let (weights_params2, bias_params2) = &quantization_params[1];

    // Perform quantized matrix multiplication for second layer
    let output = match quantized_matmul(q_weights2, weights_params2, &q_hidden, &hidden_params) {
        Ok(result) => result,
        Err(e) => {
            println!("Error in quantized matmul: {:?}", e);
            // Fallback to dequantized computation
            let dq_weights = dequantize_matrix(q_weights2, weights_params2);
            let dq_hidden = dequantize_matrix(&q_hidden, &hidden_params);
            dq_hidden.dot(&dq_weights.t())
        }
    };

    // Add biases (dequantize biases)
    let dq_biases2 = dequantize_matrix(q_biases2, bias_params2);
    let output_bias = &output + &dq_biases2.t();

    output_bias
}

/// Compare outputs from full precision and quantized networks
fn compare_outputs(full_precision: &Array2<f32>, quantized: &Array2<f32>) {
    // Calculate MSE
    let mse = (full_precision - quantized).mapv(|x| x * x).sum() / full_precision.len() as f32;

    // Calculate max absolute error
    let max_error = (full_precision - quantized)
        .mapv(|x| x.abs())
        .fold(0.0f32, |a, &b| a.max(b));

    // Calculate relative error
    let rel_error = (full_precision - quantized).mapv(|x| x.abs()).sum()
        / full_precision.mapv(|x| x.abs()).sum()
        * 100.0;

    println!("Mean Squared Error: {:.6}", mse);
    println!("Maximum Absolute Error: {:.6}", max_error);
    println!("Relative Error: {:.6}%", rel_error);

    // For classification tasks, check if top predictions match
    let batch_size = full_precision.dim().0;
    let mut top1_matches = 0;

    for i in 0..batch_size {
        let mut fp_max_idx = 0;
        let mut fp_max_val = full_precision[[i, 0]];
        let mut q_max_idx = 0;
        let mut q_max_val = quantized[[i, 0]];

        for j in 1..full_precision.dim().1 {
            if full_precision[[i, j]] > fp_max_val {
                fp_max_val = full_precision[[i, j]];
                fp_max_idx = j;
            }
            if quantized[[i, j]] > q_max_val {
                q_max_val = quantized[[i, j]];
                q_max_idx = j;
            }
        }

        if fp_max_idx == q_max_idx {
            top1_matches += 1;
        }
    }

    println!(
        "Top-1 Accuracy Match: {}/{} ({:.1}%)",
        top1_matches,
        batch_size,
        (top1_matches as f32 / batch_size as f32) * 100.0
    );
}

/// Demonstrate mixed precision quantization with different bit widths per layer
fn mixed_precision_quantization(network: &[SimpleLayer], input: &Array2<f32>) {
    // Define quantization bit widths for each layer
    // First layer: weights=8-bit, activations=8-bit
    // Second layer: weights=4-bit, activations=8-bit
    let layer_configs = [
        (8, 8, "First layer (8-bit weights, 8-bit activations)"),
        (4, 8, "Second layer (4-bit weights, 8-bit activations)"),
    ];

    println!("Layer configuration:");
    for (i, &(_w_bits, _a_bits, desc)) in layer_configs.iter().enumerate() {
        println!("  Layer {}: {}", i, desc);
    }

    // Run full precision network first for reference
    let full_precision_output = run_network_full_precision(network, input);

    // Quantize and run network with mixed precision
    let mut quantized_layers = Vec::new();
    let mut quantization_params = Vec::new();

    // Quantize each layer with its specific bit width
    for (i, (layer, &(w_bits, _, _))) in network.iter().zip(layer_configs.iter()).enumerate() {
        let weights_config = get_weight_calibration_config(w_bits, false);
        let bias_config = get_weight_calibration_config(w_bits, true);

        // Quantize weights
        let weights_params =
            calibrate_matrix(&layer.weights.view(), w_bits, &weights_config).unwrap();
        let (quantized_weights, _) =
            quantize_matrix(&layer.weights.view(), w_bits, weights_params.method);

        // Quantize biases (typically keep biases at higher precision)
        let bias_params = calibrate_matrix(&layer.biases.view(), 8, &bias_config).unwrap();
        let (quantized_biases, _) = quantize_matrix(&layer.biases.view(), 8, bias_params.method);

        quantized_layers.push((quantized_weights, quantized_biases));
        quantization_params.push((weights_params.clone(), bias_params));

        println!(
            "Layer {} quantized with weights at {}-bit, params scale: {:.6}",
            i, w_bits, weights_params.scale
        );
    }

    // First layer forward pass
    let layer0 = &network[0];
    let (q_weights0, q_biases0) = &quantized_layers[0];
    let (weights_params0, bias_params0) = &quantization_params[0];
    let (_, a_bits0, _) = layer_configs[0];

    // Quantize input
    let activation_config = get_activation_calibration_config(a_bits0, false, false);
    let act_params = calibrate_matrix(&input.view(), a_bits0, &activation_config).unwrap();
    let (q_input, _) = quantize_matrix(&input.view(), a_bits0, act_params.method);

    // First layer quantized matmul
    let hidden = match quantized_matmul(q_weights0, weights_params0, &q_input, &act_params) {
        Ok(result) => result,
        Err(_) => {
            // Fallback to dequantized computation
            let dq_weights = dequantize_matrix(q_weights0, weights_params0);
            let dq_input = dequantize_matrix(&q_input, &act_params);
            dq_input.dot(&dq_weights.t())
        }
    };

    // Add biases and apply ReLU
    let dq_biases0 = dequantize_matrix(q_biases0, bias_params0);
    let hidden_bias = &hidden + &dq_biases0.t();
    let hidden_activated = relu(&hidden_bias.view());

    // Quantize activations for second layer
    let (_, a_bits1, _) = layer_configs[1];
    let hidden_config = get_activation_calibration_config(a_bits1, true, false);
    let hidden_params =
        calibrate_matrix(&hidden_activated.view(), a_bits1, &hidden_config).unwrap();
    let (q_hidden, _) = quantize_matrix(&hidden_activated.view(), a_bits1, hidden_params.method);

    // Second layer forward pass
    let (q_weights1, q_biases1) = &quantized_layers[1];
    let (weights_params1, bias_params1) = &quantization_params[1];

    // Second layer quantized matmul
    let output = match quantized_matmul(q_weights1, weights_params1, &q_hidden, &hidden_params) {
        Ok(result) => result,
        Err(_) => {
            // Fallback to dequantized computation
            let dq_weights = dequantize_matrix(q_weights1, weights_params1);
            let dq_hidden = dequantize_matrix(&q_hidden, &hidden_params);
            dq_hidden.dot(&dq_weights.t())
        }
    };

    // Add biases
    let dq_biases1 = dequantize_matrix(q_biases1, bias_params1);
    let output_bias = &output + &dq_biases1.t();

    // Compare with full precision
    println!("\nMixed precision quantization results:");
    compare_outputs(&full_precision_output, &output_bias);

    // Calculate memory footprint reduction
    let fp32_weight_size = (layer0.weights.len()
        + layer0.biases.len()
        + network[1].weights.len()
        + network[1].biases.len())
        * 4; // 4 bytes per f32

    let mixed_weight_size = (layer0.weights.len() * layer_configs[0].0 as usize / 8) + // First layer weights
        (layer0.biases.len() * 8 / 8) + // First layer biases (8-bit)
        (network[1].weights.len() * layer_configs[1].0 as usize / 8) + // Second layer weights
        (network[1].biases.len() * 8 / 8); // Second layer biases (8-bit)

    let memory_reduction = (1.0 - (mixed_weight_size as f32 / fp32_weight_size as f32)) * 100.0;

    println!("\nMemory footprint:");
    println!("  Full precision: {} bytes", fp32_weight_size);
    println!("  Mixed precision: {} bytes", mixed_weight_size);
    println!("  Reduction: {:.1}%", memory_reduction);
}
