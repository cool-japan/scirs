//! Simple example demonstrating automatic padding

use scirs2_core::ndarray::Array1;
use scirs2_core::Complex;
use scirs2_fft::{auto_pad_complex, AutoPadConfig, PaddingMode};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing automatic padding");

    // Create a small signal
    let signal = Array1::from_vec(vec![
        Complex::new(1.0, 0.0),
        Complex::new(2.0, 0.0),
        Complex::new(3.0, 0.0),
    ]);

    println!("Original signal: {signal:?}");
    println!("Length: {}", signal.len());

    // Test zero padding
    let config = AutoPadConfig::new(PaddingMode::Zero);
    let padded = auto_pad_complex(&signal, &config)?;
    println!("\nZero padded: {padded:?}");
    println!("Padded length: {}", padded.len());

    // Test power of 2 padding
    let config_pow2 = AutoPadConfig::new(PaddingMode::Zero).with_power_of_2();
    let padded_pow2 = auto_pad_complex(&signal, &config_pow2)?;
    println!("\nPower of 2 padded length: {}", padded_pow2.len());

    // Test edge padding
    let config_edge = AutoPadConfig::new(PaddingMode::Edge);
    let padded_edge = auto_pad_complex(&signal, &config_edge)?;
    println!("\nEdge padded: {padded_edge:?}");

    Ok(())
}
