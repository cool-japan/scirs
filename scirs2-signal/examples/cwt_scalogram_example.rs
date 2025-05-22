//! Example demonstrating the use of CWT and scalogram for time-frequency analysis
//!
//! This example shows how to compute and visualize scalograms using the
//! continuous wavelet transform (CWT) functionality in scirs2-signal.

use scirs2_signal::wavelets::{cwt_magnitude, morlet, paul, scale_to_frequency, scalogram};
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Continuous Wavelet Transform (CWT) and Scalogram Example");
    println!("======================================================");

    // Generate a chirp signal (increasing frequency)
    let n = 1024;
    let dt = 0.01;
    let fs = 1.0 / dt;

    println!("Generating a chirp signal with {n} points (fs = {fs} Hz)");

    let time: Vec<f64> = (0..n).map(|i| i as f64 * dt).collect();
    let signal: Vec<f64> = time
        .iter()
        .map(|&t| {
            let freq = 2.0 + 10.0 * t; // Linear chirp from 2 Hz to 12 Hz
            (2.0 * PI * freq * t).sin()
        })
        .collect();

    // Define scales for the wavelet transform (logarithmically spaced)
    let num_scales = 32;
    let scales: Vec<f64> = (0..num_scales)
        .map(|i| 2.0_f64.powf(i as f64 / 8.0))
        .collect();

    println!("Computing scalogram with {num_scales} scales");

    // Choose a wavelet (Morlet with w0=6.0)
    // Use a closure to create a wavelet function that takes points and scale
    let wavelet = |points, scale| morlet(points, 6.0, scale);

    // Convert scales to approximate frequency values
    // For Morlet wavelets with w0=6.0, the central frequency is approximately 0.85/(2*PI)
    let central_freq = 0.85 / (2.0 * std::f64::consts::PI);
    let freqs = scale_to_frequency(&scales, central_freq, dt)?;

    // Compute the scalogram (normalized)
    let scalo = scalogram(&signal, wavelet, &scales, Some(true))?;

    // Print some stats and the frequency range
    println!(
        "Frequency range: {:.2} Hz to {:.2} Hz",
        freqs[freqs.len() - 1],
        freqs[0]
    );

    // Show a simple ASCII visualization of the scalogram
    // (limited to 20 time points and 10 frequency points for readability)
    println!("\nScalogram Visualization (rudimentary ASCII):");
    println!("(Darker characters indicate higher energy)");
    println!("Time →");

    // Subsample for visualization
    let time_step = n / 20;
    let freq_step = num_scales / 10;

    // Define ASCII intensity characters (from low to high)
    let chars = [' ', '.', ':', ';', 'o', 'O', '8', '#', '@'];

    for i in (0..num_scales).step_by(freq_step).rev() {
        print!("{:6.1} Hz |", freqs[i]);
        for j in (0..n).step_by(time_step) {
            let val = scalo[i][j];
            let idx = (val * (chars.len() - 1) as f64).round() as usize;
            let idx = idx.min(chars.len() - 1); // Clamp to maximum index
            print!("{}", chars[idx]);
        }
        println!();
    }

    println!(
        "\nTime (s)  |{}",
        (0..20)
            .map(|i| format!("{:4.1}", i as f64 * dt * time_step as f64))
            .collect::<Vec<_>>()
            .join("")
    );

    // Compare with a different wavelet
    println!("\nComparing with Paul wavelet (m=4)");
    let paul_wavelet = |points, scale| paul(points, 4, scale);
    let paul_scalo = cwt_magnitude(&signal, paul_wavelet, &scales, Some(true))?;

    println!("\nPaul Wavelet Scalogram:");
    for i in (0..num_scales).step_by(freq_step).rev() {
        print!("{:6.1} Hz |", freqs[i]);
        for j in (0..n).step_by(time_step) {
            let val = paul_scalo[i][j];
            let idx = (val * (chars.len() - 1) as f64).round() as usize;
            let idx = idx.min(chars.len() - 1); // Clamp to maximum index
            print!("{}", chars[idx]);
        }
        println!();
    }

    println!(
        "\nTime (s)  |{}",
        (0..20)
            .map(|i| format!("{:4.1}", i as f64 * dt * time_step as f64))
            .collect::<Vec<_>>()
            .join("")
    );

    println!("\nNote: For proper visualization, consider plotting the scalogram as a 2D heatmap");
    println!(
        "using an external plotting library like plotters, matplotlib (via PyO3), or gnuplot."
    );

    Ok(())
}
