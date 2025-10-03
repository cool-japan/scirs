//! Example demonstrating automatic padding strategies for FFT
//!
//! This example shows how to use automatic padding to improve
//! FFT performance by ensuring optimal input sizes.

use scirs2_core::ndarray::Array1;
use scirs2_core::Complex;
use scirs2_fft::{auto_pad_complex, fft, remove_padding_1d, AutoPadConfig, PaddingMode};
use std::time::Instant;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Automatic Padding Example");
    println!("========================\n");

    // Create a signal with non-optimal length (prime number)
    let n = 97; // Prime number - worst case for FFT
    let signal: Array1<Complex<f64>> = Array1::linspace(0.0, 1.0, n)
        .mapv(|x| Complex::new((2.0 * std::f64::consts::PI * 10.0 * x).sin(), 0.0));

    println!("Original signal length: {n} (prime number)");

    // Benchmark without padding
    println!("\n1. FFT without padding:");
    let start = Instant::now();
    let _result_no_pad = fft(&signal.to_vec(), None)?;
    let time_no_pad = start.elapsed();
    println!("   Time: {time_no_pad:?}");

    // Test different padding strategies
    let padding_configs = vec![
        ("Zero padding", AutoPadConfig::new(PaddingMode::Zero)),
        (
            "Power of 2",
            AutoPadConfig::new(PaddingMode::Zero).with_power_of_2(),
        ),
        ("Edge padding", AutoPadConfig::new(PaddingMode::Edge)),
        ("Linear ramp", AutoPadConfig::new(PaddingMode::LinearRamp)),
        (
            "Centered",
            AutoPadConfig::new(PaddingMode::Zero).with_center(),
        ),
    ];

    for (name, config) in padding_configs {
        println!("\n2. FFT with {name}:");

        // Apply automatic padding
        let padded = auto_pad_complex(&signal, &config)?;
        println!("   Padded length: {}", padded.len());

        // Perform FFT on padded signal
        let start = Instant::now();
        let result_padded = fft(&padded.to_vec(), None)?;
        let time_padded = start.elapsed();
        println!("   Time: {time_padded:?}");

        // Calculate speedup
        let speedup = time_no_pad.as_secs_f64() / time_padded.as_secs_f64();
        println!("   Speedup: {speedup:.2}x");

        // Remove padding from result if needed
        let result_unpadded = Array1::from_vec(result_padded);
        let result_trimmed = remove_padding_1d(&result_unpadded, n, &config);
        println!("   Trimmed result length: {}", result_trimmed.len());
    }

    println!("\n3. Demonstrating padding modes visually:");

    // Small signal to show padding behavior
    let small_signal = Array1::from_vec(vec![
        Complex::new(1.0, 0.0),
        Complex::new(2.0, 0.0),
        Complex::new(3.0, 0.0),
        Complex::new(2.0, 0.0),
        Complex::new(1.0, 0.0),
    ]);

    let visual_configs = vec![
        ("Zero", PaddingMode::Zero),
        ("Constant(0.5)", PaddingMode::Constant(0.5)),
        ("Edge", PaddingMode::Edge),
        ("Wrap", PaddingMode::Wrap),
        ("LinearRamp", PaddingMode::LinearRamp),
    ];

    for (name, mode) in visual_configs {
        let config = AutoPadConfig::new(mode).with_min_pad(5);
        let padded = auto_pad_complex(&small_signal, &config)?;

        print!("   {name}: [");
        for (i, val) in padded.iter().enumerate() {
            if i > 0 {
                print!(", ");
            }
            print!("{:.1}", val.re);
        }
        println!("]");
    }

    println!("\n4. Maximum padding constraint:");
    let config_max = AutoPadConfig::new(PaddingMode::Zero).with_max_pad(20);

    let padded_constrained = auto_pad_complex(&signal, &config_max)?;
    println!("   Original length: {}", signal.len());
    println!("   Padded length (max 20): {}", padded_constrained.len());

    Ok(())
}
