//! Example of minimum phase filter conversion
//!
//! This example demonstrates how to convert filters to minimum phase equivalents,
//! which preserve the magnitude response while ensuring all zeros are in stable locations.

use num_traits::Zero;
use scirs2_signal::filter::{group_delay, minimum_phase};
use std::f64::consts::PI;

fn main() {
    println!("Minimum Phase Filter Conversion Example");
    println!("======================================\n");

    // Example 1: Basic minimum phase conversion
    println!("1. Basic Minimum Phase Conversion");
    println!("---------------------------------");

    // Create a filter with zeros outside the unit circle
    // H(z) = (z - 2)(z - 0.5) = z^2 - 2.5z + 1
    let original_b = vec![1.0, -2.5, 1.0];
    println!("Original filter coefficients: {:?}", original_b);
    println!("Zeros at z = 2.0 (outside unit circle) and z = 0.5 (inside unit circle)");

    // Convert to minimum phase
    let min_phase_b = minimum_phase(&original_b, true).unwrap();
    println!("Minimum phase coefficients: {:?}", min_phase_b);

    // Verify the conversion by checking magnitude responses
    let frequencies = vec![0.0, PI / 4.0, PI / 2.0, 3.0 * PI / 4.0, PI];
    println!("\nMagnitude response comparison:");
    println!("Frequency (rad) | Original |H(f)| | Min Phase |H(f)|");
    println!("----------------|----------------|------------------");

    for &freq in &frequencies {
        let h_orig = evaluate_filter_response(&original_b, freq);
        let h_min = evaluate_filter_response(&min_phase_b, freq);

        println!(
            "{:14.3} | {:13.6} | {:15.6}",
            freq,
            h_orig.norm(),
            h_min.norm()
        );
    }

    // Example 2: Already minimum phase filter
    println!("\n\n2. Already Minimum Phase Filter");
    println!("------------------------------");

    // Create a filter that's already minimum phase
    // H(z) = (z + 0.5)(z + 0.3) = z^2 + 0.8z + 0.15
    let already_min_b = vec![1.0, 0.8, 0.15];
    println!("Original coefficients: {:?}", already_min_b);
    println!("Zeros at z = -0.5 and z = -0.3 (both inside unit circle)");

    let still_min_b = minimum_phase(&already_min_b, true).unwrap();
    println!("After conversion: {:?}", still_min_b);

    // Should be nearly identical
    let difference: f64 = already_min_b
        .iter()
        .zip(still_min_b.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    println!("Total coefficient difference: {:.2e}", difference);

    // Example 3: Continuous-time filter
    println!("\n\n3. Continuous-Time Filter Conversion");
    println!("-----------------------------------");

    // Create a continuous-time filter with zeros in the right half-plane
    // H(s) = (s - 1)(s + 2) = s^2 + s - 2
    let ct_b = vec![1.0, 1.0, -2.0];
    println!("Continuous-time filter: {:?}", ct_b);
    println!("Zeros at s = 1 (unstable) and s = -2 (stable)");

    let ct_min_b = minimum_phase(&ct_b, false).unwrap();
    println!("Minimum phase version: {:?}", ct_min_b);

    // Example 4: Group delay comparison
    println!("\n\n4. Group Delay Comparison");
    println!("------------------------");

    let b_non_min = vec![1.0, -2.0, 1.0]; // (z-1)^2
    let b_min = minimum_phase(&b_non_min, true).unwrap();
    let a = vec![1.0]; // No denominator (FIR filter)

    let freqs = vec![0.1, 0.5, 1.0, 2.0, 3.0];

    println!("Computing group delay (may take a moment for numerical differentiation)...");

    let gd_orig = group_delay(&b_non_min, &a, &freqs).unwrap();
    let gd_min = group_delay(&b_min, &a, &freqs).unwrap();

    println!("Frequency (rad/s) | Original GD | Min Phase GD");
    println!("------------------|-------------|-------------");

    for ((freq, orig_gd), min_gd) in freqs.iter().zip(gd_orig.iter()).zip(gd_min.iter()) {
        println!("{:16.1} | {:10.3} | {:11.3}", freq, orig_gd, min_gd);
    }

    // Example 5: High-order filter
    println!("\n\n5. Higher-Order Filter Example");
    println!("-----------------------------");

    // Create a 4th order filter with multiple zeros outside unit circle
    // This is a more challenging case for the root-finding algorithm
    let high_order_b = vec![1.0, -5.0, 8.0, -4.0]; // Approximate (z-1)(z-2)(z-2)
    println!("4th order filter coefficients: {:?}", high_order_b);

    match minimum_phase(&high_order_b, true) {
        Ok(ho_min_b) => {
            println!("Minimum phase coefficients: {:?}", ho_min_b);

            // Check if the conversion preserved the overall magnitude
            let orig_dc = evaluate_filter_response(&high_order_b, 0.0).norm();
            let min_dc = evaluate_filter_response(&ho_min_b, 0.0).norm();
            println!("DC gain preservation: {:.6} -> {:.6}", orig_dc, min_dc);
        }
        Err(e) => {
            println!("Higher-order conversion failed: {:?}", e);
            println!("This demonstrates the limitations of the simplified root-finding algorithm");
        }
    }

    // Example 6: Practical considerations
    println!("\n\n6. Practical Considerations");
    println!("--------------------------");
    println!("Minimum phase filters have several important properties:");
    println!("- Preserve magnitude response exactly");
    println!("- Have the smallest possible group delay");
    println!("- Are causal and stable (for proper design)");
    println!("- Useful in filter design and system identification");
    println!("\nApplications:");
    println!("- Audio processing (phase linearization)");
    println!("- Control systems (stability guarantees)");
    println!("- Signal reconstruction");
    println!("- Equalization and inverse filtering");

    // Example 7: Error handling
    println!("\n\n7. Error Handling");
    println!("----------------");

    // Test with empty coefficients
    match minimum_phase(&[], true) {
        Ok(_) => println!("Unexpected success with empty coefficients"),
        Err(e) => println!("Expected error with empty coefficients: {:?}", e),
    }

    // Test with all-zero coefficients
    let zero_b = vec![0.0, 0.0, 0.0];
    match minimum_phase(&zero_b, true) {
        Ok(result) => println!("All-zero filter result: {:?}", result),
        Err(e) => println!("Error with all-zero coefficients: {:?}", e),
    }

    println!("\n\nMinimum phase conversion is a fundamental operation in signal processing,");
    println!(
        "providing a way to ensure filter stability while preserving magnitude characteristics."
    );
}

/// Helper function to evaluate filter response at a frequency
fn evaluate_filter_response(b: &[f64], w: f64) -> num_complex::Complex64 {
    use num_complex::Complex64;

    let z = Complex64::new(w.cos(), w.sin());
    let mut response = Complex64::zero();

    for (i, &coeff) in b.iter().enumerate() {
        let power = b.len() - 1 - i;
        response += Complex64::new(coeff, 0.0) * z.powi(power as i32);
    }

    response
}
