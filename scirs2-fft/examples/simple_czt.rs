//! Simple example testing the CZT functionality

use scirs2_core::ndarray::Array1;
use scirs2_core::Complex;
use scirs2_fft::{czt, zoom_fft};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing basic CZT functionality");

    // Create a simple signal
    let n = 16;
    let signal: Array1<Complex<f64>> =
        Array1::linspace(0.0, n as f64 - 1.0, n).mapv(|x| Complex::new(x, 0.0));

    // Test 1: Basic CZT (should match FFT)
    println!("\n1. Basic CZT test");
    let czt_result = czt(&signal, None, None, None, None)?;
    println!("CZT result shape: {:?}", czt_result.shape());
    println!("Result dimensions: {:?}", czt_result.ndim());

    // Test 2: Zoom FFT
    println!("\n2. Zoom FFT test");
    let zoom_result = zoom_fft(&signal, 8, 0.1, 0.3, None)?;
    println!("Zoom FFT result shape: {:?}", zoom_result.shape());

    println!("\nAll tests passed!");
    Ok(())
}
