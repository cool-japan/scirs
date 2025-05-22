//! Example of spectral descriptors using the utilities module
//!
//! This example demonstrates how to use the spectral utilities to analyze
//! the characteristics of a signal's frequency content.

use scirs2_signal::spectral::periodogram;
use scirs2_signal::utilities::spectral::*;
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Spectral Descriptors Example");
    println!("----------------------------");

    // Generate a test signal: sum of two sinusoids with different frequencies
    let fs = 1000.0; // Sample rate in Hz
    let duration = 1.0; // Duration in seconds
    let n_samples = (fs * duration) as usize;

    let f1 = 50.0; // First component: 50 Hz
    let f2 = 200.0; // Second component: 200 Hz
    let a1 = 1.0; // Amplitude of first component
    let a2 = 0.5; // Amplitude of second component

    let mut signal = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let t = i as f64 / fs;
        let value = a1 * (2.0 * PI * f1 * t).sin() + a2 * (2.0 * PI * f2 * t).sin();
        signal.push(value);
    }

    // Compute power spectral density using periodogram
    let (psd, freqs) = periodogram(&signal, Some(fs), Some("hann"), Some(n_samples), None, None)?;

    println!("Signal information:");
    println!("  - Sample rate: {} Hz", fs);
    println!("  - Duration: {} seconds", duration);
    println!("  - Components: {} Hz and {} Hz", f1, f2);
    println!();

    // Calculate basic spectral descriptors
    let centroid = spectral_centroid(&psd, &freqs)?;
    let spread = spectral_spread(&psd, &freqs, None)?;
    let skewness = spectral_skewness(&psd, &freqs, None, None)?;
    let kurtosis = spectral_kurtosis(&psd, &freqs, None, None)?;
    let flatness = spectral_flatness(&psd)?;
    let rolloff_85 = spectral_rolloff(&psd, &freqs, 0.85)?;

    // Calculate additional spectral descriptors
    let crest = spectral_crest(&psd)?;
    let slope = spectral_slope(&psd, &freqs)?;
    let decrease = spectral_decrease(&psd, &freqs)?;
    let bandwidth = spectral_bandwidth(&psd, &freqs, -3.0)?;
    let contrast = spectral_contrast(&psd, &freqs, 4)?;
    let (dominant_freq, dominant_mag) = dominant_frequency(&psd, &freqs)?;
    let top_peaks = dominant_frequencies(&psd, &freqs, 3, 10.0)?;

    // Additional spectral representations
    let _esd = energy_spectral_density(&psd, fs)?;
    let _normalized = normalized_psd(&psd)?;

    println!("Basic spectral descriptors:");
    println!("  - Centroid: {:.2} Hz", centroid);
    println!("  - Spread: {:.2} Hz", spread);
    println!("  - Skewness: {:.4}", skewness);
    println!("  - Kurtosis: {:.4}", kurtosis);
    println!("  - Flatness: {:.4}", flatness);
    println!("  - 85% Rolloff: {:.2} Hz", rolloff_85);

    println!("\nAdditional spectral descriptors:");
    println!("  - Crest factor: {:.4}", crest);
    println!("  - Spectral slope: {:.4e}", slope);
    println!("  - Spectral decrease: {:.4}", decrease);
    println!("  - -3dB Bandwidth: {:.2} Hz", bandwidth);
    println!(
        "  - Dominant frequency: {:.2} Hz (magnitude: {:.4})",
        dominant_freq, dominant_mag
    );

    println!("\nTop frequency peaks:");
    for (i, (freq, mag)) in top_peaks.iter().enumerate() {
        println!("  {}. {:.2} Hz (magnitude: {:.4})", i + 1, freq, mag);
    }

    println!("\nSpectral contrast by band:");
    for (i, c) in contrast.iter().enumerate() {
        println!("  Band {}: {:.4}", i + 1, c);
    }
    println!();

    // Generate a second signal with noise to compare
    let mut signal2 = Vec::with_capacity(n_samples);
    let mut rng = rand::rng();
    for i in 0..n_samples {
        let t = i as f64 / fs;
        let noise = rand::Rng::random_range(&mut rng, -0.5..0.5);
        let value = a1 * (2.0 * PI * f1 * t).sin() + a2 * (2.0 * PI * f2 * t).sin() + noise;
        signal2.push(value);
    }

    // Compute power spectral density for the second signal
    let (psd2, _) = periodogram(
        &signal2,
        Some(fs),
        Some("hann"),
        Some(n_samples),
        None,
        None,
    )?;

    // Calculate spectral flux between the two signals
    let flux_l1 = spectral_flux(&psd, &psd2, "l1")?;
    let flux_l2 = spectral_flux(&psd, &psd2, "l2")?;

    println!("Spectral flux between clean and noisy signals:");
    println!("  - L1 norm: {:.6}", flux_l1);
    println!("  - L2 norm: {:.6}", flux_l2);

    // Show how spectral flatness changes with increasing noise
    println!();
    println!("Effect of noise on spectral flatness:");
    for noise_level in [0.0, 0.5, 1.0, 2.0, 5.0] {
        let mut noisy_signal = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let t = i as f64 / fs;
            let noise = rand::Rng::random_range(&mut rng, -noise_level..noise_level);
            let value = a1 * (2.0 * PI * f1 * t).sin() + a2 * (2.0 * PI * f2 * t).sin() + noise;
            noisy_signal.push(value);
        }

        let (noisy_psd, _) = periodogram(
            &noisy_signal,
            Some(fs),
            Some("hann"),
            Some(n_samples),
            None,
            None,
        )?;
        let noisy_flatness = spectral_flatness(&noisy_psd)?;

        println!(
            "  - Noise level {:.1}: Flatness = {:.4}",
            noise_level, noisy_flatness
        );
    }

    Ok(())
}
