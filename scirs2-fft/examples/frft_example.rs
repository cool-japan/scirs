//! Example of using Fractional Fourier Transform

use num_complex::Complex64;
use scirs2_fft::{frft, frft_complex};
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Example 1: Real-valued signal
    println!("Example 1: Real-valued signal");
    let n = 64;
    let signal: Vec<f64> = (0..n)
        .map(|i| (2.0 * PI * 5.0 * i as f64 / n as f64).sin())
        .collect();

    // Compute FrFT with different orders
    for &alpha in &[0.0, 0.5, 1.0, 1.5, 2.0] {
        let result = frft(&signal, alpha, None)?;
        let energy: f64 = result.iter().map(|c| c.norm_sqr()).sum();
        println!("FrFT order {}: energy = {:.6}", alpha, energy);
    }

    // Example 2: Complex-valued signal
    println!("\nExample 2: Complex-valued signal");
    let complex_signal: Vec<Complex64> = (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            Complex64::new((2.0 * PI * 3.0 * t).cos(), (2.0 * PI * 7.0 * t).sin())
        })
        .collect();

    // Compute FrFT for complex signal
    let result_complex = frft_complex(&complex_signal, 0.5, None)?;
    let complex_energy: f64 = result_complex.iter().map(|c| c.norm_sqr()).sum();
    println!("FrFT of complex signal: energy = {:.6}", complex_energy);

    // Example 3: Demonstrate additivity property
    println!("\nExample 3: Additivity property");
    let alpha1 = 0.3;
    let alpha2 = 0.4;

    // Method 1: Direct computation with combined alpha
    let result1 = frft(&signal, alpha1 + alpha2, None)?;

    // Method 2: Sequential application
    // Convert to complex for consistency
    let signal_complex: Vec<Complex64> = signal.iter().map(|&x| Complex64::new(x, 0.0)).collect();
    let temp = frft_complex(&signal_complex, alpha2, None)?;
    let result2 = frft_complex(&temp, alpha1, None)?;

    // Compare energies (they should be similar)
    let energy1: f64 = result1.iter().map(|c| c.norm_sqr()).sum();
    let energy2: f64 = result2.iter().map(|c| c.norm_sqr()).sum();
    println!("Energy from direct computation: {:.6}", energy1);
    println!("Energy from sequential application: {:.6}", energy2);
    println!(
        "Relative difference: {:.2e}",
        ((energy1 - energy2) / energy1).abs()
    );

    // Example 4: Special cases verification
    println!("\nExample 4: Special cases");

    // α = 0 should be identity
    let identity_result = frft(&signal, 0.0, None)?;
    let identity_match = signal
        .iter()
        .zip(identity_result.iter())
        .all(|(&s, c)| (s - c.re).abs() < 1e-10 && c.im.abs() < 1e-10);
    println!(
        "α = 0 (identity): {}",
        if identity_match { "✓" } else { "✗" }
    );

    // α = 2 should be time reversal
    let reversal_result = frft(&signal, 2.0, None)?;
    let reversal_match = signal.iter().enumerate().all(|(i, &s)| {
        let rev_idx = n - 1 - i;
        (s - reversal_result[rev_idx].re).abs() < 1e-10
    });
    println!(
        "α = 2 (time reversal): {}",
        if reversal_match { "✓" } else { "✗" }
    );

    Ok(())
}
