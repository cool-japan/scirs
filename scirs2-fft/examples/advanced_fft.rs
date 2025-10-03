//! Advanced FFT examples
//!
//! This example demonstrates more advanced usage of the FFT module
//! including the Hilbert transform, fractional FFT, and non-uniform FFT.

use scirs2_core::Complex64;
use scirs2_fft::nufft::InterpolationType;
use scirs2_fft::{fft, frft, frft_complex, hilbert, nufft, rfft};
use std::f64::consts::PI;
#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Advanced FFT Examples");
    println!("===================\n");

    // Create a test signal
    let fs = 1000.0; // 1 kHz sampling rate
    let duration = 1.0; // 1 second
    let n_samples = (fs * duration) as usize;
    let t: Vec<f64> = (0..n_samples).map(|i| i as f64 / fs).collect();

    // Create a multi-component signal
    let signal: Vec<f64> = t
        .iter()
        .map(|&ti| {
            // 100 Hz sine wave + 250 Hz cosine wave
            (2.0 * PI * 100.0 * ti).sin() + 0.5 * (2.0 * PI * 250.0 * ti).cos()
        })
        .collect();

    println!("Created a test signal with {} samples", signal.len());
    println!("Sampling rate: {fs} Hz");
    println!("Signal duration: {duration} seconds");

    // Example 1: Hilbert Transform
    println!("\n1. Hilbert Transform");

    // Apply Hilbert transform to get the analytic signal
    let analytic_signal = hilbert(&signal)?;

    // Compute the envelope (instantaneous amplitude)
    let envelope: Vec<f64> = analytic_signal.iter().map(|c| c.norm()).collect();

    // Compute the instantaneous phase
    let phase: Vec<f64> = analytic_signal.iter().map(|c| c.im.atan2(c.re)).collect();

    // Compute the instantaneous frequency
    let inst_freq: Vec<f64> = phase
        .windows(2)
        .map(|window| {
            let mut diff = window[1] - window[0];
            // Handle phase wrapping
            if diff > PI {
                diff -= 2.0 * PI;
            } else if diff < -PI {
                diff += 2.0 * PI;
            }
            // Convert to Hz: diff / (2*pi*dt)
            diff * fs / (2.0 * PI)
        })
        .collect();

    println!("  Computed analytic signal using Hilbert transform");
    println!("  Signal envelope statistics:");
    println!(
        "    Min: {:.4}",
        envelope.iter().fold(f64::INFINITY, |a, &b| a.min(b))
    );
    println!(
        "    Max: {:.4}",
        envelope.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
    );
    println!(
        "    Mean: {:.4}",
        envelope.iter().sum::<f64>() / envelope.len() as f64
    );

    println!("  Instantaneous frequency statistics:");
    println!(
        "    Min: {:.2} Hz",
        inst_freq.iter().fold(f64::INFINITY, |a, &b| a.min(b))
    );
    println!(
        "    Max: {:.2} Hz",
        inst_freq.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
    );
    println!(
        "    Mean: {:.2} Hz",
        inst_freq.iter().sum::<f64>() / inst_freq.len() as f64
    );

    // Example 2: Fractional Fourier Transform
    println!("\n2. Fractional Fourier Transform");

    // Create a simple signal for FrFT
    let chirp: Vec<f64> = t
        .iter()
        .map(|&ti| {
            // Linear chirp signal (frequency increases linearly with time)
            let _inst_freq = 50.0 + 200.0 * ti; // 50 to 250 Hz
            let phase = 2.0 * PI * (50.0 * ti + 100.0 * ti.powi(2));
            phase.sin()
        })
        .collect();

    // Apply standard FFT
    let fft_result = fft(&chirp, None)?;

    // Apply Fractional Fourier Transform with different orders
    let frft_orders = vec![0.25, 0.5, 0.75, 1.0];

    println!("  Computing Fractional Fourier Transform with different orders:");
    for &order in &frft_orders {
        let frft_result = frft(&chirp, order, None)?;

        // Calculate energy of the transform
        let energy = frft_result.iter().map(|c| c.norm_sqr()).sum::<f64>();

        println!("    Order = {order:.2} (α = {order:.2}π): Energy = {energy:.2e}");

        // For order = 0.5, the FrFT is halfway between time and frequency domain
        if (order - 0.5).abs() < 1e-6 {
            println!("      (Order 0.5 represents the signal in the time-frequency domain)");
        }

        // For order = 1.0, the FrFT should be equivalent to the FFT
        if (order - 1.0).abs() < 1e-6 {
            // Check equivalence with regular FFT
            let max_diff = frft_result
                .iter()
                .zip(fft_result.iter())
                .map(|(a, b)| (a - b).norm())
                .fold(0.0, f64::max);
            println!("      Max difference from regular FFT: {max_diff:.2e}");
        }
    }

    // Example 3: FrFT on complex input
    println!("\n3. Fractional Fourier Transform on Complex Input");

    // Create a complex signal
    let complex_signal: Vec<Complex64> = t
        .iter()
        .map(|&ti| {
            let real = (2.0 * PI * 100.0 * ti).sin();
            let imag = (2.0 * PI * 100.0 * ti).cos();
            Complex64::new(real, imag)
        })
        .collect();

    // Apply FrFT with order = 0.5 (halfway between time and frequency domain)
    let complex_frft = frft_complex(&complex_signal, 0.5, None)?;

    println!("  Computed FrFT on complex signal with order = 0.5");
    println!(
        "  Transform energy: {:.2e}",
        complex_frft.iter().map(|c| c.norm_sqr()).sum::<f64>()
    );

    // Example 4: Non-Uniform FFT
    println!("\n4. Non-Uniform FFT (Type 1)");

    // Create non-uniform sample points
    let n_nonuniform = 200;
    let mut rng = scirs2_core::random::rng();

    // Generate random sampling points between 0 and 2π
    let x: Vec<f64> = (0..n_nonuniform)
        .map(|_| 2.0 * PI * scirs2_core::random::Rng::random_range(&mut rng, 0.0..1.0))
        .collect();

    // Generate a signal at these non-uniform points: sum of two sine waves
    let signal_nonuniform: Vec<Complex64> = x
        .iter()
        .map(|&xi| {
            let val = (3.0 * xi).sin() + 0.5 * (7.0 * xi).sin();
            Complex64::new(val, 0.0)
        })
        .collect();

    // Compute Type 1 NUFFT (non-uniform samples to uniform frequencies)
    let n_modes = 32; // Number of frequency modes
    let epsilon = 1e-6; // Accuracy parameter
    let nufft_result = nufft::nufft_type1(
        &x,
        &signal_nonuniform,
        n_modes,
        InterpolationType::Gaussian,
        epsilon,
    )?;

    println!(
        "  Computed Type 1 NUFFT from {n_nonuniform} non-uniform samples to {n_modes} frequency modes"
    );

    // Find the dominant frequencies
    let mut freq_magnitudes: Vec<(i32, f64)> = nufft_result
        .iter()
        .enumerate()
        .map(|(i, &val)| {
            let freq = if i <= n_modes / 2 {
                i as i32
            } else {
                (i as i32) - (n_modes as i32)
            };
            (freq, val.norm())
        })
        .collect();

    // Sort by magnitude
    freq_magnitudes.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("  Top 3 frequency components:");
    for (i, (freq, magnitude)) in freq_magnitudes.iter().take(3).enumerate() {
        println!(
            "    {}: k = {} with magnitude {:.4}",
            i + 1,
            freq,
            magnitude
        );
    }

    // Example 5: Real FFT
    println!("\n5. Real FFT Optimization");

    // Create a real-valued signal
    let real_signal: Vec<f64> = t
        .iter()
        .map(|&ti| (2.0 * PI * 100.0 * ti).sin() + 0.5 * (2.0 * PI * 200.0 * ti).sin())
        .collect();

    // Compare standard FFT and RFFT
    let start_time = std::time::Instant::now();
    let fft_complex = fft(&real_signal, None)?;
    let fft_duration = start_time.elapsed();

    let start_time = std::time::Instant::now();
    let rfft_result = rfft(&real_signal, None)?;
    let rfft_duration = start_time.elapsed();

    println!(
        "  Standard FFT: {} points in {:?}",
        fft_complex.len(),
        fft_duration
    );
    println!(
        "  Real FFT: {} points in {:?}",
        rfft_result.len(),
        rfft_duration
    );
    println!(
        "  RFFT is approximately {:.1}x faster",
        fft_duration.as_nanos() as f64 / rfft_duration.as_nanos() as f64
    );
    println!(
        "  RFFT uses {:.1}% of the memory compared to standard FFT",
        100.0 * rfft_result.len() as f64 / fft_complex.len() as f64
    );

    println!("\nAdvanced FFT examples completed successfully!");
    Ok(())
}
