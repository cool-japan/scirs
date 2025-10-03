//! Example demonstrating the Hartley transform

use scirs2_core::ndarray::{array, Array1};
use scirs2_fft::hartley::{dht, dht2, fht, idht};
use std::f64::consts::PI;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Hartley Transform Example ===");
    println!();

    // Example 1: Basic Hartley transform
    println!("1. Basic Hartley Transform:");
    let signal = array![1.0, 2.0, 3.0, 4.0];
    println!("   Input signal: {signal:?}");

    let hartley = dht(&signal)?;
    println!("   Hartley transform: {hartley:?}");

    let recovered = idht(&hartley)?;
    println!("   Recovered signal: {recovered:?}");

    let error: f64 = (&signal - &recovered).mapv(|x| x.abs()).sum();
    println!("   Recovery error: {error:.2e}");
    println!();

    // Example 2: Hartley transform of a sinusoidal signal
    println!("2. Hartley Transform of Sinusoid:");
    let n = 64;
    let mut sin_signal = Array1::<f64>::zeros(n);
    let freq = 5.0; // 5 cycles over n samples

    for i in 0..n {
        let t = i as f64 / n as f64;
        sin_signal[i] = (2.0 * PI * freq * t).sin();
    }

    let hartley_sin = fht(&sin_signal)?;

    // Find peaks in the spectrum
    let mut peaks = Vec::new();
    for i in 1..n - 1 {
        if hartley_sin[i].abs() > hartley_sin[i - 1].abs()
            && hartley_sin[i].abs() > hartley_sin[i + 1].abs()
            && hartley_sin[i].abs() > 0.1
        {
            peaks.push((i, hartley_sin[i]));
        }
    }

    println!("   Signal frequency: {freq} cycles");
    println!("   Detected peaks (bin, magnitude):");
    for (bin, mag) in peaks {
        println!("      Bin {bin}: {mag:.3}");
    }
    println!();

    // Example 3: 2D Hartley transform
    println!("3. 2D Hartley Transform:");
    let image = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    println!("   Input image:");
    for row in image.outer_iter() {
        println!("      {row:?}");
    }

    let hartley_2d = dht2(&image, None)?;
    println!("   2D Hartley transform:");
    for row in hartley_2d.outer_iter() {
        println!("      {row:?}");
    }
    println!();

    // Example 4: Comparing Hartley with FFT
    println!("4. Hartley vs FFT Relationship:");
    let test_signal = array![1.0, 0.0, -1.0, 0.0];
    println!("   Test signal: {test_signal:?}");

    // Compute Hartley transform
    let hartley_result = dht(&test_signal)?;
    println!("   Hartley transform: {hartley_result:?}");

    // Compute FFT and derive Hartley from it
    use scirs2_core::Complex64;
    use scirs2_fft::fft::fft;

    let mut complex_signal = Array1::<Complex64>::zeros(test_signal.len());
    for (i, &val) in test_signal.iter().enumerate() {
        complex_signal[i] = Complex64::new(val, 0.0);
    }

    let fft_result = fft(&complex_signal.to_vec(), None)?;
    let mut hartley_from_fft = Array1::<f64>::zeros(test_signal.len());
    for i in 0..test_signal.len() {
        hartley_from_fft[i] = fft_result[i].re - fft_result[i].im;
    }

    println!("   Hartley from FFT: {hartley_from_fft:?}");

    let diff: f64 = (&hartley_result - &hartley_from_fft)
        .mapv(|x| x.abs())
        .sum();
    println!("   Difference: {diff:.2e}");
    println!();

    // Example 5: Properties of Hartley transform
    println!("5. Hartley Transform Properties:");
    println!("   - Real-valued transform of real signals");
    println!("   - Self-inverse (with scaling)");
    println!("   - Linear relationship with FFT");
    println!("   - Useful for convolution without complex arithmetic");

    Ok(())
}
