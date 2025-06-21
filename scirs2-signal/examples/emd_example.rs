//! Example demonstrating Empirical Mode Decomposition (EMD) for signal analysis
//!
//! This example shows how to use EMD and EEMD to decompose a signal into
//! Intrinsic Mode Functions (IMFs), as well as computing the Hilbert-Huang
//! spectrum for time-frequency analysis.

use scirs2_signal::emd::{eemd, emd, hilbert_huang_spectrum, EmdConfig};
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Empirical Mode Decomposition (EMD) Example");
    println!("===========================================");

    // Parameters for the test signal
    let sample_rate = 1000.0; // 1 kHz
    let duration = 2.0; // 2 seconds
    let n_samples = (sample_rate * duration) as usize;

    println!("Generating test signals:");
    println!("- Sample rate: {:.1} Hz", sample_rate);
    println!("- Duration: {:.1} seconds", duration);
    println!("- Samples: {}", n_samples);

    // Generate time vector
    let time: Vec<f64> = (0..n_samples).map(|i| i as f64 / sample_rate).collect();

    // Generate the first test signal: sum of three sinusoids of different frequencies
    println!("\nTest Signal 1: Sum of sinusoids with frequencies 5 Hz, 20 Hz, and 60 Hz");

    let signal1: Vec<f64> = time
        .iter()
        .map(|&t| {
            (2.0 * PI * 5.0 * t).sin() +           // 5 Hz component
        0.5 * (2.0 * PI * 20.0 * t).sin() +    // 20 Hz component
        0.2 * (2.0 * PI * 60.0 * t).sin() // 60 Hz component
        })
        .collect();

    // Configure EMD
    let config = EmdConfig {
        max_imfs: 5,             // Maximum number of IMFs to extract
        sift_threshold: 0.05,    // Stopping criterion for sifting process
        max_sift_iterations: 50, // Maximum sifting iterations per IMF
        ..Default::default()
    };

    println!("\nApplying EMD with parameters:");
    println!("- Max IMFs: {}", config.max_imfs);
    println!("- Sift threshold: {:.3}", config.sift_threshold);
    println!("- Max sift iterations: {}", config.max_sift_iterations);

    // Apply EMD
    let result1 = emd(&signal1, &config)?;

    // Print results
    println!("\nEMD Results for Signal 1:");
    println!("- Extracted {} IMFs", result1.imfs.shape()[0]);

    for i in 0..result1.imfs.shape()[0] {
        let energy = result1.energies[i];
        let iterations = result1.iterations[i];
        println!(
            "  IMF {}: Energy = {:.6}, Iterations = {}",
            i + 1,
            energy,
            iterations
        );
    }

    // Compute Hilbert-Huang spectrum for time-frequency analysis
    let num_freqs = 100; // Number of frequency bins
    println!(
        "\nComputing Hilbert-Huang spectrum with {} frequency bins",
        num_freqs
    );

    let (times, freqs, hhs) = hilbert_huang_spectrum(&result1, sample_rate, num_freqs)?;

    // Print frequency range
    println!(
        "- Frequency range: {:.2} Hz to {:.2} Hz",
        freqs[0],
        freqs[freqs.len() - 1]
    );

    // Find the times and frequencies with maximum energy
    let mut max_energy = 0.0;
    let mut max_time_idx = 0;
    let mut max_freq_idx = 0;

    for i in 0..freqs.len() {
        for j in 0..times.len() {
            if hhs[[i, j]] > max_energy {
                max_energy = hhs[[i, j]];
                max_freq_idx = i;
                max_time_idx = j;
            }
        }
    }

    println!(
        "- Maximum energy at: Time = {:.3} s, Frequency = {:.2} Hz, Energy = {:.6}",
        times[max_time_idx], freqs[max_freq_idx], max_energy
    );

    // Generate the second test signal: chirp (frequency-modulated signal)
    println!("\nTest Signal 2: Chirp signal with frequency increasing from 10 Hz to 100 Hz");

    let signal2: Vec<f64> = time
        .iter()
        .map(|&t| {
            // Linear chirp from 10 Hz to 100 Hz
            let _instantaneous_freq = 10.0 + 45.0 * t;
            let phase = 2.0 * PI * (10.0 * t + 22.5 * t * t);
            phase.sin()
        })
        .collect();

    // Apply EEMD (Ensemble EMD) for better handling of mode mixing
    let ensemble_size = 10; // Number of ensemble trials (use 50-100 in practice)
    let noise_std = 0.1; // Standard deviation of added white noise

    println!("\nApplying EEMD with parameters:");
    println!("- Ensemble size: {}", ensemble_size);
    println!("- Noise standard deviation: {:.2}", noise_std);

    let result2 = eemd(&signal2, &config, ensemble_size, noise_std)?;

    // Print results
    println!("\nEEMD Results for Signal 2:");
    println!("- Extracted {} IMFs", result2.imfs.shape()[0]);

    for i in 0..result2.imfs.shape()[0] {
        let energy = result2.energies[i];
        println!("  IMF {}: Energy = {:.6}", i + 1, energy);
    }

    // Compute Hilbert-Huang spectrum for time-frequency analysis of the chirp
    println!("\nComputing Hilbert-Huang spectrum for the chirp signal");

    let (times2, freqs2, hhs2) = hilbert_huang_spectrum(&result2, sample_rate, num_freqs)?;

    // Check frequency tracking
    println!("Time-frequency analysis of the chirp signal:");

    // Print frequency with maximum energy at different time points
    let time_points = [0.2, 0.5, 1.0, 1.5];

    for &t in &time_points {
        // Find nearest time index
        let t_idx = (t * sample_rate) as usize;
        if t_idx < times2.len() {
            // Find frequency with maximum energy at this time
            let mut max_energy = 0.0;
            let mut max_freq_idx = 0;

            for i in 0..freqs2.len() {
                if hhs2[[i, t_idx]] > max_energy {
                    max_energy = hhs2[[i, t_idx]];
                    max_freq_idx = i;
                }
            }

            // Calculate expected frequency from chirp formula
            let expected_freq = 10.0 + 45.0 * t;

            println!(
                "  Time = {:.1} s: Detected f = {:.1} Hz, Expected f = {:.1} Hz",
                t, freqs2[max_freq_idx], expected_freq
            );
        }
    }

    println!(
        "\nNote: Visualization of IMFs and HHS would be done with external plotting libraries."
    );

    Ok(())
}
