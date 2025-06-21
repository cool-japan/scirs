//! Example of adaptive filtering using LMS, NLMS, and RLS algorithms
//!
//! This example demonstrates adaptive filter applications including:
//! - System identification
//! - Noise cancellation
//! - Echo cancellation
//! - Equalization

use scirs2_signal::adaptive::{LmsFilter, NlmsFilter, RlsFilter};
use std::f64::consts::PI;

fn main() {
    println!("Adaptive Filter Example");
    println!("======================\n");

    // Example 1: System Identification with LMS
    println!("1. System Identification with LMS Filter");
    println!("---------------------------------------");

    // Define an unknown system to identify: h = [0.5, -0.3, 0.2, 0.1]
    let unknown_system = vec![0.5, -0.3, 0.2, 0.1];
    println!("Unknown system coefficients: {:?}", unknown_system);

    // Create LMS filter
    let mut lms = LmsFilter::new(4, 0.01, 0.0).unwrap();
    println!("LMS filter: 4 taps, step size = 0.01");

    // Generate input signal (white noise-like)
    let num_samples = 500;
    let mut inputs = Vec::with_capacity(num_samples);
    let mut desired = Vec::with_capacity(num_samples);

    for i in 0..num_samples {
        // Input signal: mixture of sinusoids
        let input = 0.5 * (2.0 * PI * i as f64 * 0.01).sin()
            + 0.3 * (2.0 * PI * i as f64 * 0.02).cos()
            + 0.2 * (2.0 * PI * i as f64 * 0.03).sin();
        inputs.push(input);

        // Generate desired output by filtering input through unknown system
        let mut output = 0.0;
        for (j, &coeff) in unknown_system.iter().enumerate() {
            if i >= j {
                output += coeff * inputs[i - j];
            }
        }
        desired.push(output);
    }

    // Run adaptation
    let (_outputs, errors, _mse) = lms.adapt_batch(&inputs, &desired).unwrap();

    println!("Final LMS weights: {:?}", lms.weights());
    println!("Target weights:    {:?}", unknown_system);

    // Calculate identification error
    let final_error: f64 = lms
        .weights()
        .iter()
        .zip(unknown_system.iter())
        .map(|(&w, &t)| (w - t).powi(2))
        .sum::<f64>()
        .sqrt();
    println!("Weight estimation error (RMSE): {:.6}", final_error);

    // Calculate final MSE
    let final_mse: f64 = errors.iter().rev().take(50).map(|&e| e * e).sum::<f64>() / 50.0;
    println!("Final MSE (last 50 samples): {:.6}", final_mse);

    // Example 2: Noise Cancellation with NLMS
    println!("\n\n2. Noise Cancellation with NLMS Filter");
    println!("-------------------------------------");

    // Create desired signal (speech-like)
    let signal_length = 300;
    let mut clean_signal = Vec::with_capacity(signal_length);
    let mut noise_signal = Vec::with_capacity(signal_length);
    let mut noisy_signal = Vec::with_capacity(signal_length);

    for i in 0..signal_length {
        // Clean signal: mixture of speech-like frequencies
        let clean = 0.8 * (2.0 * PI * i as f64 * 0.05).sin()
            + 0.4 * (2.0 * PI * i as f64 * 0.12).sin()
            + 0.2 * (2.0 * PI * i as f64 * 0.08).cos();
        clean_signal.push(clean);

        // Noise: correlated noise that we want to cancel
        let noise =
            0.5 * (2.0 * PI * i as f64 * 0.25).sin() + 0.3 * (2.0 * PI * i as f64 * 0.18).cos();
        noise_signal.push(noise);

        // Noisy signal = clean signal + noise
        noisy_signal.push(clean + noise);
    }

    // Create NLMS filter for noise cancellation
    let mut nlms = NlmsFilter::new(8, 0.5, 1e-6).unwrap();
    println!("NLMS filter: 8 taps, step size = 0.5");

    // Adaptive noise cancellation
    // Input: noise reference, Desired: noisy signal, Output: estimate of noise
    let mut clean_estimate = Vec::with_capacity(signal_length);
    let mut noise_estimate = Vec::with_capacity(signal_length);

    for i in 0..signal_length {
        let (noise_est, _error, _mse) = nlms.adapt(noise_signal[i], noisy_signal[i]).unwrap();
        noise_estimate.push(noise_est);

        // Clean estimate = noisy signal - noise estimate
        clean_estimate.push(noisy_signal[i] - noise_est);
    }

    // Calculate noise reduction performance
    let original_snr = calculate_snr(&clean_signal, &noise_signal);
    let enhanced_snr = calculate_snr(
        &clean_signal,
        &get_noise_from_clean(&clean_estimate, &clean_signal),
    );

    println!("Original SNR: {:.2} dB", original_snr);
    println!("Enhanced SNR: {:.2} dB", enhanced_snr);
    println!("SNR improvement: {:.2} dB", enhanced_snr - original_snr);

    // Example 3: RLS vs LMS Convergence Comparison
    println!("\n\n3. RLS vs LMS Convergence Comparison");
    println!("-----------------------------------");

    // Target system for comparison
    let target = vec![0.8, -0.6, 0.4];
    println!("Target system: {:?}", target);

    // Create both filters
    let mut lms_comp = LmsFilter::new(3, 0.05, 0.0).unwrap();
    let mut rls_comp = RlsFilter::new(3, 0.99, 100.0).unwrap();

    let convergence_samples = 100;
    let mut lms_weights_history = Vec::new();
    let mut rls_weights_history = Vec::new();
    let mut lms_errors = Vec::new();
    let mut rls_errors = Vec::new();

    for i in 0..convergence_samples {
        // Input signal
        let input = (i as f64 * 0.1).sin() + 0.5 * (i as f64 * 0.05).cos();

        // Desired output from target system
        let mut desired = 0.0;
        if i >= 2 {
            desired = target[0] * input
                + target[1] * ((i - 1) as f64 * 0.1).sin()
                + target[2] * ((i - 2) as f64 * 0.1).sin();
        }

        // Adapt both filters
        let (_out_lms, err_lms, _) = lms_comp.adapt(input, desired).unwrap();
        let (_out_rls, err_rls, _) = rls_comp.adapt(input, desired).unwrap();

        lms_weights_history.push(lms_comp.weights().to_vec());
        rls_weights_history.push(rls_comp.weights().to_vec());
        lms_errors.push(err_lms.abs());
        rls_errors.push(err_rls.abs());
    }

    println!("Final LMS weights: {:?}", lms_comp.weights());
    println!("Final RLS weights: {:?}", rls_comp.weights());

    // Calculate final convergence error for both
    let lms_final_error: f64 = lms_comp
        .weights()
        .iter()
        .zip(target.iter())
        .map(|(&w, &t)| (w - t).powi(2))
        .sum::<f64>()
        .sqrt();

    let rls_final_error: f64 = rls_comp
        .weights()
        .iter()
        .zip(target.iter())
        .map(|(&w, &t)| (w - t).powi(2))
        .sum::<f64>()
        .sqrt();

    println!("LMS final weight error: {:.6}", lms_final_error);
    println!("RLS final weight error: {:.6}", rls_final_error);

    // Show convergence speed (samples to reach 90% of final performance)
    let lms_90_percent = find_convergence_point(&lms_errors, 0.1);
    let rls_90_percent = find_convergence_point(&rls_errors, 0.1);

    println!("LMS convergence time (90%): {} samples", lms_90_percent);
    println!("RLS convergence time (90%): {} samples", rls_90_percent);

    // Example 4: Echo Cancellation Simulation
    println!("\n\n4. Echo Cancellation Simulation");
    println!("------------------------------");

    // Simulate echo path: h_echo = [0.0, 0.0, 0.3, -0.1, 0.05]
    let echo_path = vec![0.0, 0.0, 0.3, -0.1, 0.05];
    let _echo_delay = 2; // 2-sample delay
    println!("Echo path coefficients: {:?}", echo_path);

    // Create adaptive echo canceler
    let mut echo_canceler = LmsFilter::new(8, 0.01, 0.0).unwrap();

    let num_echo_samples = 200;
    let mut far_end_signal = Vec::with_capacity(num_echo_samples);
    let mut echo_signal = Vec::with_capacity(num_echo_samples);
    let mut near_end_signal = Vec::with_capacity(num_echo_samples);
    let mut microphone_signal = Vec::with_capacity(num_echo_samples);

    // Generate signals
    for i in 0..num_echo_samples {
        // Far-end signal (speaker output)
        let far_end =
            0.7 * (2.0 * PI * i as f64 * 0.03).sin() + 0.3 * (2.0 * PI * i as f64 * 0.07).cos();
        far_end_signal.push(far_end);

        // Echo signal (delayed and filtered far-end)
        let mut echo = 0.0;
        for (j, &coeff) in echo_path.iter().enumerate() {
            if i >= j {
                echo += coeff * far_end_signal[i - j];
            }
        }
        echo_signal.push(echo);

        // Near-end signal (local speaker)
        let near_end = if i > 50 && i < 150 {
            0.5 * (2.0 * PI * i as f64 * 0.02).sin()
        } else {
            0.0
        };
        near_end_signal.push(near_end);

        // Microphone picks up near-end + echo
        microphone_signal.push(near_end + echo);
    }

    // Perform echo cancellation
    let mut echo_canceled = Vec::with_capacity(num_echo_samples);

    for i in 0..num_echo_samples {
        // Adapt filter using far-end as input and microphone as desired
        let (echo_estimate, _error, _mse) = echo_canceler
            .adapt(far_end_signal[i], microphone_signal[i])
            .unwrap();

        // Cancel echo: output = microphone - echo_estimate
        echo_canceled.push(microphone_signal[i] - echo_estimate);
    }

    // Calculate echo cancellation performance
    let echo_power: f64 =
        echo_signal.iter().map(|&x| x * x).sum::<f64>() / echo_signal.len() as f64;
    let residual_power: f64 = echo_canceled
        .iter()
        .skip(50)
        .take(50)
        .map(|&x| x * x)
        .sum::<f64>()
        / 50.0;
    let echo_cancellation_db = 10.0 * (echo_power / residual_power).log10();

    println!("Echo cancellation achieved: {:.1} dB", echo_cancellation_db);
    println!("Final echo canceler weights: {:?}", echo_canceler.weights());

    // Example 5: Performance Characteristics
    println!("\n\n5. Adaptive Filter Performance Summary");
    println!("------------------------------------");

    println!("Algorithm Comparison:");
    println!("┌─────────┬─────────────┬─────────────┬──────────────┬─────────────┐");
    println!("│Algorithm│ Convergence │ Computational│ Tracking     │ Stability   │");
    println!("│         │ Speed       │ Complexity   │ Capability   │             │");
    println!("├─────────┼─────────────┼─────────────┼──────────────┼─────────────┤");
    println!("│ LMS     │ Slow        │ O(N)        │ Good         │ High        │");
    println!("│ NLMS    │ Medium      │ O(N)        │ Better       │ High        │");
    println!("│ RLS     │ Fast        │ O(N²)       │ Excellent    │ Medium      │");
    println!("└─────────┴─────────────┴─────────────┴──────────────┴─────────────┘");

    println!("\nApplication Guidelines:");
    println!("- LMS: General purpose, robust, low complexity");
    println!("- NLMS: Better for varying signal power levels");
    println!("- RLS: Fast convergence, non-stationary signals, higher complexity");

    println!("\nTypical Applications:");
    println!("- System identification: Channel estimation, inverse modeling");
    println!("- Noise cancellation: Active noise control, hearing aids");
    println!("- Echo cancellation: Telephony, video conferencing");
    println!("- Equalization: Digital communications, audio processing");
    println!("- Prediction: Time series forecasting, compression");
}

/// Calculate Signal-to-Noise Ratio in dB
fn calculate_snr(signal: &[f64], noise: &[f64]) -> f64 {
    let signal_power: f64 = signal.iter().map(|&x| x * x).sum::<f64>() / signal.len() as f64;
    let noise_power: f64 = noise.iter().map(|&x| x * x).sum::<f64>() / noise.len() as f64;

    if noise_power > 1e-10 {
        10.0 * (signal_power / noise_power).log10()
    } else {
        100.0 // Very high SNR
    }
}

/// Extract noise from difference between clean estimate and true clean signal
fn get_noise_from_clean(clean_estimate: &[f64], true_clean: &[f64]) -> Vec<f64> {
    clean_estimate
        .iter()
        .zip(true_clean.iter())
        .map(|(&est, &true_val)| est - true_val)
        .collect()
}

/// Find convergence point (sample number where error drops below threshold)
fn find_convergence_point(errors: &[f64], threshold_ratio: f64) -> usize {
    if errors.is_empty() {
        return 0;
    }

    let final_error = errors[errors.len() - 1];
    let threshold = final_error + threshold_ratio * errors[0];

    for (i, &error) in errors.iter().enumerate() {
        if error <= threshold {
            return i;
        }
    }

    errors.len() - 1
}
