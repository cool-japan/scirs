//! Advanced FFT Mode Showcase
//!
//! This example demonstrates basic FFT operations as a placeholder
//! for advanced FFT optimization capabilities.

use scirs2_core::ndarray::Array1;
use scirs2_core::Complex64;
use scirs2_fft::fft;
use std::f64::consts::PI;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Advanced FFT Mode Showcase");
    println!("=====================================");

    // Create a test signal
    let signal = create_test_signal(1024);

    // Perform basic FFT
    let result = fft(&signal, None)?;

    println!("FFT completed successfully on {} samples", signal.len());
    println!("Result has {} frequency components", result.len());

    Ok(())
}

/// Create a test signal with sine components
#[allow(dead_code)]
fn create_test_signal(length: usize) -> Vec<Complex64> {
    (0..length)
        .map(|i| {
            let t = i as f64 / length as f64;
            let value = (2.0 * PI * 10.0 * t).sin() + 0.5 * (2.0 * PI * 25.0 * t).sin();
            Complex64::new(value, 0.0)
        })
        .collect()
}
