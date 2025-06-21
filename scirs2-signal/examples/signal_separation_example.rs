//! Example demonstrating signal separation techniques
//!
//! This example shows how to use multi-band and harmonic/percussive
//! separation methods for signal analysis and processing.

use ndarray::Array1;
use scirs2_signal::separation::{
    harmonic_percussive_separation, multiband_separation, HarmonicPercussiveConfig, MultibandConfig,
};
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Signal Separation Examples");
    println!("==========================\n");

    // Create a test signal with multiple frequency components
    let sample_rate = 1000.0;
    let duration = 2.0;
    let t: Vec<f64> = (0..(sample_rate * duration) as usize)
        .map(|i| i as f64 / sample_rate)
        .collect();

    // Create signal with multiple frequency components and noise
    let signal: Vec<f64> = t
        .iter()
        .enumerate()
        .map(|(i, &t)| {
            let low_freq = (2.0 * PI * 30.0 * t).sin(); // Low frequency component
            let mid_freq = (2.0 * PI * 120.0 * t).sin(); // Mid frequency component
            let high_freq = (2.0 * PI * 300.0 * t).sin(); // High frequency component

            // Add some transient events (percussive-like)
            let transient = if (i % 200) < 5 { 0.8 } else { 0.0 };

            low_freq + 0.7 * mid_freq + 0.5 * high_freq + transient
        })
        .collect();

    let signal_array = Array1::from(signal);

    println!("Original signal stats:");
    println!("  Length: {} samples", signal_array.len());
    println!("  RMS: {:.4}", calculate_rms(&signal_array));
    println!(
        "  Peak: {:.4}",
        signal_array.iter().fold(0.0f64, |a, &b| a.max(b.abs()))
    );
    println!();

    // Example 1: Multi-band separation
    println!("1. Multi-band Signal Separation");
    println!("================================");

    // Define frequency bands (normalized to Nyquist = 500 Hz)
    // Bands: 0-80Hz, 80-200Hz, 200-400Hz, 400-500Hz
    let cutoff_freqs = vec![0.16, 0.4, 0.8]; // 80Hz, 200Hz, 400Hz normalized

    let config = MultibandConfig::default();
    let bands = multiband_separation(&signal_array, &cutoff_freqs, sample_rate, Some(config))?;

    println!("Created {} frequency bands:", bands.len());
    for (i, band) in bands.iter().enumerate() {
        let rms = calculate_rms(band);
        let peak = band.iter().fold(0.0f64, |a, &b| a.max(b.abs()));

        let freq_range = match i {
            0 => "0-80 Hz".to_string(),
            1 => "80-200 Hz".to_string(),
            2 => "200-400 Hz".to_string(),
            3 => "400-500 Hz".to_string(),
            _ => format!("Band {}", i),
        };

        println!(
            "  Band {} ({}): RMS = {:.4}, Peak = {:.4}",
            i + 1,
            freq_range,
            rms,
            peak
        );
    }
    println!();

    // Example 2: Harmonic/percussive separation
    println!("2. Harmonic/Percussive Separation");
    println!("==================================");

    let hp_config = HarmonicPercussiveConfig {
        separation_power: 2.0,
        ..Default::default()
    };

    let (harmonic, percussive) =
        harmonic_percussive_separation(&signal_array, sample_rate, Some(hp_config))?;

    println!("Separation results:");
    println!("  Original signal:");
    println!(
        "    RMS = {:.4}, Peak = {:.4}",
        calculate_rms(&signal_array),
        signal_array.iter().fold(0.0f64, |a, &b| a.max(b.abs()))
    );

    println!("  Harmonic component:");
    println!(
        "    RMS = {:.4}, Peak = {:.4}",
        calculate_rms(&harmonic),
        harmonic.iter().fold(0.0f64, |a, &b| a.max(b.abs()))
    );

    println!("  Percussive component:");
    println!(
        "    RMS = {:.4}, Peak = {:.4}",
        calculate_rms(&percussive),
        percussive.iter().fold(0.0f64, |a, &b| a.max(b.abs()))
    );
    println!();

    // Example 3: Advanced multi-band configuration
    println!("3. Advanced Multi-band Configuration");
    println!("====================================");

    let advanced_config = MultibandConfig {
        filter_order: 8, // Higher order for steeper rolloff
        overlap: 0.05,   // Less overlap between bands
        filter_type: scirs2_signal::filter::FilterType::Lowpass, // Will be overridden per band
    };

    // More detailed frequency separation for audio analysis
    let audio_cutoffs = vec![0.1, 0.2, 0.4, 0.6]; // 50, 100, 200, 300 Hz
    let audio_bands = multiband_separation(
        &signal_array,
        &audio_cutoffs,
        sample_rate,
        Some(advanced_config),
    )?;

    println!("Advanced separation into {} bands:", audio_bands.len());
    let band_names = [
        "Bass (0-50Hz)",
        "Low-mid (50-100Hz)",
        "Mid (100-200Hz)",
        "High-mid (200-300Hz)",
        "Treble (300-500Hz)",
    ];

    for (i, band) in audio_bands.iter().enumerate() {
        let energy = band.iter().map(|&x| x * x).sum::<f64>();
        let name = band_names.get(i).unwrap_or(&"Unknown");
        println!("  {}: Energy = {:.2}", name, energy);
    }
    println!();

    // Example 4: Signal reconstruction validation
    println!("4. Signal Reconstruction Validation");
    println!("===================================");

    // Test that harmonic + percussive ≈ original (for our simple method)
    let reconstructed: Vec<f64> = harmonic
        .iter()
        .zip(percussive.iter())
        .map(|(&h, &p)| (h + p) * 0.5) // Simple combination
        .collect();

    let original_energy: f64 = signal_array.iter().map(|&x| x * x).sum();
    let reconstructed_energy: f64 = reconstructed.iter().map(|&x| x * x).sum();

    println!("Energy comparison:");
    println!("  Original energy: {:.2}", original_energy);
    println!("  Reconstructed energy: {:.2}", reconstructed_energy);
    println!(
        "  Reconstruction ratio: {:.3}",
        reconstructed_energy / original_energy
    );
    println!();

    println!("Signal separation examples completed successfully!");

    Ok(())
}

/// Calculate RMS (Root Mean Square) of a signal
fn calculate_rms(signal: &Array1<f64>) -> f64 {
    let mean_square: f64 = signal.iter().map(|&x| x * x).sum::<f64>() / signal.len() as f64;
    mean_square.sqrt()
}
