//! Example demonstrating dynamic quantization calibration
//!
//! This example shows how to use exponential moving average (EMA) calibration
//! for adapting to changing data distributions.

use ndarray::Array2;
use rand::{rng, Rng};
use rand_distr::{Distribution, Normal};
use scirs2_linalg::quantization::calibration::{
    calibrate_matrix, CalibrationConfig, CalibrationMethod,
};
use scirs2_linalg::quantization::{dequantize_matrix, quantize_matrix};

#[allow(dead_code)]
fn main() {
    println!("Dynamic Quantization Calibration Example");
    println!("=======================================\n");

    // Create a sequence of data distributions with drift
    println!("Creating sequence of data with distribution drift...");
    let datasequence = create_drifting_data_sequence(10, 0.3);

    // Demonstrate static vs dynamic calibration
    println!("\nComparing static vs dynamic calibration:");
    compare_static_vs_dynamic_calibration(&datasequence, 8);

    // Demonstrate effect of EMA factor
    println!("\nComparing different EMA factors:");
    compare_ema_factors(&datasequence, 8);

    // Real-world example: streaming data scenario
    println!("\nSimulating streaming data scenario:");
    simulate_streaming_data();
}

/// Create a sequence of data matrices with drifting distribution
#[allow(dead_code)]
fn create_drifting_data_sequence(_num_matrices: usize, driftfactor: f32) -> Vec<Array2<f32>> {
    let mut rng = rand::rng();
    let mut result = Vec::with_capacity(_num_matrices);

    // Start with a base distribution
    let mut mean = 0.0;
    let mut std_dev = 1.0;

    for i in 0.._num_matrices {
        // Create matrix with current distribution
        let mut matrix = Array2::zeros((10, 10));
        let normal = Normal::new(mean, std_dev).unwrap();

        for r in 0..10 {
            for c in 0..10 {
                matrix[[r, c]] = normal.sample(&mut rng);
            }
        }

        // Add some outliers occasionally
        if i % 3 == 0 {
            let r = rng.random_range(0..10);
            let c = rng.random_range(0..10);
            matrix[[r, c]] = if rng.random_bool(0.5) {
                mean + std_dev * 5.0
            } else {
                mean - std_dev * 5.0
            };
        }

        result.push(matrix);

        // Drift the distribution parameters
        mean += driftfactor * rng.random_range(-1.0..1.0);
        std_dev = (std_dev + driftfactor * rng.random_range(-0.1..0.3)).clamp(0.5f32, 3.0f32);
    }

    result
}

/// Compare static (one-time) vs dynamic (EMA) calibration
#[allow(dead_code)]
fn compare_static_vs_dynamic_calibration(datasequence: &[Array2<f32>], bits: u8) {
    println!(
        "{:^10} | {:^15} | {:^15} | {:^15}",
        "Batch", "Static MSE", "Dynamic MSE", "Improvement (%)"
    );
    println!("{:-^10} | {:-^15} | {:-^15} | {:-^15}", "", "", "", "");

    // One-time calibration using first batch
    let static_config = CalibrationConfig {
        method: CalibrationMethod::MinMax,
        symmetric: true,
        ..Default::default()
    };
    let static_params = calibrate_matrix(&datasequence[0].view(), bits, &static_config).unwrap();

    // Dynamic calibration using EMA
    let dynamic_config = CalibrationConfig {
        method: CalibrationMethod::ExponentialMovingAverage,
        symmetric: true,
        ema_factor: 0.2, // Moderate adaptivity
        ..Default::default()
    };
    let mut dynamic_params =
        calibrate_matrix(&datasequence[0].view(), bits, &dynamic_config).unwrap();

    let mut total_static_mse = 0.0;
    let mut total_dynamic_mse = 0.0;

    // Process each data batch
    for (i, data) in datasequence.iter().enumerate() {
        // Static calibration always uses the same parameters
        let (static_quantized_, _) = quantize_matrix(&data.view(), bits, static_params.method);
        let static_dequantized = dequantize_matrix(&static_quantized_, &static_params);
        let static_mse = (data - &static_dequantized).mapv(|x| x * x).sum() / data.len() as f32;

        // Update dynamic calibration for each batch
        if i > 0 {
            // In a real application, we would adjust parameters based on recent data
            dynamic_params = calibrate_matrix(&data.view(), bits, &dynamic_config).unwrap();
        }

        let (dynamic_quantized_, _) = quantize_matrix(&data.view(), bits, dynamic_params.method);
        let dynamic_dequantized = dequantize_matrix(&dynamic_quantized_, &dynamic_params);
        let dynamic_mse = (data - &dynamic_dequantized).mapv(|x| x * x).sum() / data.len() as f32;

        // Calculate improvement
        let improvement = if static_mse > 0.0 {
            ((static_mse - dynamic_mse) / static_mse) * 100.0
        } else {
            0.0
        };

        println!(
            "{:^10} | {:^15.6} | {:^15.6} | {:^15.2}",
            i, static_mse, dynamic_mse, improvement
        );

        total_static_mse += static_mse;
        total_dynamic_mse += dynamic_mse;
    }

    // Print overall summary
    let avg_static_mse = total_static_mse / datasequence.len() as f32;
    let avg_dynamic_mse = total_dynamic_mse / datasequence.len() as f32;
    let overall_improvement = ((avg_static_mse - avg_dynamic_mse) / avg_static_mse) * 100.0;

    println!("\nOverall Results:");
    println!("Average Static MSE: {:.6}", avg_static_mse);
    println!("Average Dynamic MSE: {:.6}", avg_dynamic_mse);
    println!("Overall Improvement: {:.2}%", overall_improvement);
}

/// Compare different EMA factors for dynamic calibration
#[allow(dead_code)]
fn compare_ema_factors(datasequence: &[Array2<f32>], bits: u8) {
    let ema_factors = [0.05, 0.1, 0.3, 0.5, 0.9];

    println!(
        "{:^10} | {:^15} | {:^15} | {:^15} | {:^15} | {:^15}",
        "Batch", "EMA=0.05", "EMA=0.1", "EMA=0.3", "EMA=0.5", "EMA=0.9"
    );
    println!(
        "{:-^10} | {:-^15} | {:-^15} | {:-^15} | {:-^15} | {:-^15}",
        "", "", "", "", "", ""
    );

    // Setup calibration configs with different EMA factors
    let mut configs = Vec::new();
    let mut params_list = Vec::new();

    for &factor in &ema_factors {
        let config = CalibrationConfig {
            method: CalibrationMethod::ExponentialMovingAverage,
            symmetric: true,
            ema_factor: factor,
            ..Default::default()
        };
        configs.push(config);

        // Initialize with first batch
        let params =
            calibrate_matrix(&datasequence[0].view(), bits, configs.last().unwrap()).unwrap();
        params_list.push(params);
    }

    // Process each data batch
    for (i, data) in datasequence.iter().enumerate().skip(1) {
        let mut mse_values = Vec::new();

        // Test each EMA factor
        for j in 0..ema_factors.len() {
            // Update calibration parameters
            params_list[j] = calibrate_matrix(&data.view(), bits, &configs[j]).unwrap();

            // Quantize and measure error
            let (quantized, _) = quantize_matrix(&data.view(), bits, params_list[j].method);
            let dequantized = dequantize_matrix(&quantized, &params_list[j]);
            let mse = (data - &dequantized).mapv(|x| x * x).sum() / data.len() as f32;

            mse_values.push(mse);
        }

        // Print results for this batch
        print!("{:^10} |", i);
        for mse in &mse_values {
            print!(" {:^15.6} |", mse);
        }
        println!();
    }

    println!("\nAnalysis of EMA factors:");
    println!("- Lower factors (0.05-0.1) are more stable but adapt slower to distribution changes");
    println!("- Higher factors (0.5-0.9) adapt quickly but can be unstable if data varies a lot");
    println!("- Mid-range factors (0.2-0.3) often provide a good balance for most applications");
}

/// Simulate a streaming data scenario with dynamic calibration
#[allow(dead_code)]
fn simulate_streaming_data() {
    let bits = 8;
    let mut rng = rand::rng();

    // Setup for a streaming scenario where data distribution changes over time
    println!("Simulating a streaming sensor with changing data distribution...");

    // Initial calibration
    let mut drift = 0.0;
    let mut amplitude = 1.0;
    let initial_data = generate_sensor_batch(50, drift, amplitude, &mut rng);

    // Configure dynamic calibration
    let dynamic_config = CalibrationConfig {
        method: CalibrationMethod::ExponentialMovingAverage,
        symmetric: false,  // Sensor data often has asymmetric distribution
        ema_factor: 0.1,   // Moderate adaptation rate
        max_iterations: 5, // Limit iterations for efficiency
        ..Default::default()
    };

    // Initial calibration
    let mut params = calibrate_matrix(&initial_data.view(), bits, &dynamic_config).unwrap();

    println!("\nStreaming data simulation:");
    println!(
        "{:^6} | {:^12} | {:^12} | {:^12} | {:^12} | {:^10}",
        "Time", "Min Value", "Max Value", "Scale", "Zero Point", "Quant. Error"
    );
    println!(
        "{:-^6} | {:-^12} | {:-^12} | {:-^12} | {:-^12} | {:-^10}",
        "", "", "", "", "", ""
    );

    // Simulate streaming data over time
    for t in 0..10 {
        // Update data distribution parameters (simulating real-world drift)
        drift += rng.random_range(-0.2..0.2);

        // Every few time steps..introduce a significant change
        if t % 3 == 0 {
            amplitude *= rng.random_range(0.8..1.3);
        }

        // Generate new data batch
        let data = generate_sensor_batch(50, drift, amplitude, &mut rng);

        // Use current parameters to quantize
        let (quantized, _) = quantize_matrix(&data.view(), bits, params.method);
        let dequantized = dequantize_matrix(&quantized, &params);

        // Calculate quantization error
        let error = (&data - &dequantized).mapv(|x| x.abs()).mean().unwrap();

        // Print current state
        println!(
            "{:^6} | {:^12.4} | {:^12.4} | {:^12.6} | {:^12} | {:^10.6}",
            t,
            data.iter().fold(f32::MAX, |a, &b| a.min(b)),
            data.iter().fold(f32::MIN, |a, &b| a.max(b)),
            params.scale,
            params.zero_point,
            error
        );

        // Update calibration parameters for next batch using newly observed data
        params = calibrate_matrix(&data.view(), bits, &dynamic_config).unwrap();
    }

    println!("\nObservations:");
    println!("1. The calibration parameters adapt to the changing data distribution");
    println!("2. Scale and zero point adjust to maintain quantization accuracy");
    println!("3. Quantization error remains relatively stable despite distribution changes");
}

/// Generate a batch of simulated sensor data with specified drift and amplitude
#[allow(dead_code)]
fn generate_sensor_batch(
    size: usize,
    drift: f32,
    amplitude: f32,
    rng: &mut impl Rng,
) -> Array2<f32> {
    let mut data = Array2::zeros((size, 1));

    // Basic normal distribution for sensor readings
    let normal = Normal::new(drift, amplitude).unwrap();

    // Fill data array
    for i in 0..size {
        data[[i, 0]] = normal.sample(rng);
    }

    // Add occasional outliers (sensor glitches)
    let num_outliers = (size as f32 * 0.05) as usize; // 5% outliers
    for _ in 0..num_outliers {
        let idx = rng.random_range(0..size);
        let outlier_factor = rng.random_range(3.0..5.0);
        data[[idx, 0]] = if rng.random_bool(0.5) {
            drift + amplitude * outlier_factor
        } else {
            drift - amplitude * outlier_factor
        };
    }

    data
}
