//! Example demonstrating the Fast Hankel Transform (FHT)
//!
//! This example shows how to use the Fast Hankel Transform for various applications.

use scirs2_fft::{fht, fht_sample_points, fhtoffset, ifht};
// use std::f64::consts::PI;  // Unused import

/// J0 Bessel function approximation for testing
#[allow(dead_code)]
fn bessel_j0(x: f64) -> f64 {
    if x.abs() < 1e-6 {
        1.0
    } else {
        (x.sin() / x) * (1.0 - x * x / 6.0 + x.powi(4) / 120.0)
    }
}

fn main() {
    println!("Fast Hankel Transform Example");
    println!("============================");
    println!();

    // Example 1: Basic FHT with order 0 (J0 Bessel function)
    example_basic_fht();

    // Example 2: FHT with different orders
    example_different_orders();

    // Example 3: Using bias for power-law behavior
    example_biased_transform();

    // Example 4: Optimal offset calculation
    example_optimal_offset();
}

fn example_basic_fht() {
    println!("Example 1: Basic FHT with order 0");
    println!("---------------------------------");

    let n = 128;
    let dln = 0.05;
    let mu = 0.0; // Order 0 (J0 Bessel function)

    // Generate logarithmically spaced sample points
    let r = fht_sample_points(n, dln, 0.0);

    // Create a test function: Gaussian
    let sigma = 1.0;
    let f: Vec<f64> = r
        .iter()
        .map(|&ri| (-ri * ri / (2.0 * sigma * sigma)).exp())
        .collect();

    // Compute FHT
    let f_transform = fht(&f, dln, mu, None, None).unwrap();

    // The Hankel transform of a Gaussian is also a Gaussian
    // with reciprocal width
    println!("Input: Gaussian with σ = {}", sigma);
    println!("Transform: Should be Gaussian with σ' ≈ {}", 1.0 / sigma);

    // Find the peak of the transform
    let max_idx = f_transform
        .iter()
        .position(|&x| x == f_transform.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)))
        .unwrap();
    println!("Peak at index: {}", max_idx);

    // Test inverse transform
    let f_recovered = ifht(&f_transform, dln, mu, None, None).unwrap();

    // Check recovery error
    let error: f64 = f
        .iter()
        .zip(f_recovered.iter())
        .map(|(&a, &b)| (a - b).abs())
        .sum::<f64>()
        / n as f64;

    println!("Average recovery error: {:.2e}", error);
    println!();
}

fn example_different_orders() {
    println!("Example 2: FHT with different orders");
    println!("-----------------------------------");

    let n = 64;
    let dln = 0.1;

    // Test different Bessel function orders
    let orders = vec![0.0, 0.5, 1.0, 2.0];

    for mu in orders {
        println!("Testing order μ = {}", mu);

        // Create a simple test signal
        let r = fht_sample_points(n, dln, 0.0);
        let f: Vec<f64> = r.iter().map(|&ri| (-ri).exp()).collect();

        // Compute FHT
        let f_transform = fht(&f, dln, mu, None, None).unwrap();

        // Check that transform succeeded
        let has_nan = f_transform.iter().any(|x| x.is_nan());
        println!("  Transform successful: {}", !has_nan);

        // Simple check: transform should be non-zero
        let norm: f64 = f_transform.iter().map(|x| x * x).sum::<f64>().sqrt();
        println!("  Transform norm: {:.3e}", norm);
    }
    println!();
}

fn example_biased_transform() {
    println!("Example 3: Biased transform for power laws");
    println!("-----------------------------------------");

    let n = 128;
    let dln = 0.05;
    let mu = 0.0;

    // Create a power-law signal: r^(-alpha)
    let alpha = 1.5;
    let r = fht_sample_points(n, dln, 0.0);
    let f: Vec<f64> = r.iter().map(|&ri| ri.powf(-alpha)).collect();

    // Transform without bias
    let f_unbiased = fht(&f, dln, mu, None, None).unwrap();

    // Transform with bias matching the power law
    let f_biased = fht(&f, dln, mu, None, Some(alpha)).unwrap();

    // The biased transform should handle the power law better
    let norm_unbiased: f64 = f_unbiased.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_biased: f64 = f_biased.iter().map(|x| x * x).sum::<f64>().sqrt();

    println!("Power law: r^(-{})", alpha);
    println!("Unbiased transform norm: {:.3e}", norm_unbiased);
    println!("Biased transform norm: {:.3e}", norm_biased);
    println!(
        "Ratio (biased/unbiased): {:.3}",
        norm_biased / norm_unbiased
    );
    println!();
}

fn example_optimal_offset() {
    println!("Example 4: Optimal offset calculation");
    println!("------------------------------------");

    let dln = 0.1;
    let mu = 0.5;

    // Calculate optimal offset for different bias values
    let bias_values = vec![0.0, 0.5, 1.0, 2.0];

    for bias in bias_values {
        let offset = fhtoffset(dln, mu, None, Some(bias)).unwrap();
        println!("Bias = {}, Optimal offset = {}", bias, offset);
    }

    println!();

    // Show how offset affects the transform
    let n = 64;
    let offsets = vec![0.0, 0.1, 0.2, 0.5];

    // Create test signal
    let r = fht_sample_points(n, dln, 0.0);
    let f: Vec<f64> = r.iter().map(|&ri| (-ri * ri).exp()).collect();

    println!("Effect of offset on transform:");
    for offset in offsets {
        let f_transform = fht(&f, dln, mu, Some(offset), None).unwrap();
        let norm: f64 = f_transform.iter().map(|x| x * x).sum::<f64>().sqrt();
        println!("  Offset = {}, Transform norm = {:.3e}", offset, norm);
    }
    println!();
}

// Additional example: Application to radial functions
#[allow(dead_code)]
fn example_radial_transform() {
    println!("Example 5: Radial function transform");
    println!("-----------------------------------");

    // For a radially symmetric 2D function f(r), the 2D Fourier transform
    // is related to the 0-order Hankel transform

    let n = 128;
    let dln = 0.05;
    let mu = 0.0; // Order 0 for radial symmetry

    // Create a radial function: Mexican hat wavelet
    let r = fht_sample_points(n, dln, -2.0); // Start from smaller r
    let f: Vec<f64> = r
        .iter()
        .map(|&ri| {
            let x = ri * ri;
            (1.0 - x) * (-x / 2.0).exp()
        })
        .collect();

    // Compute the Hankel transform
    let f_transform = fht(&f, dln, mu, None, None).unwrap();

    // The transform represents the radial profile of the 2D Fourier transform
    println!("Mexican hat wavelet in real space");
    println!("Hankel transform gives radial frequency profile");

    // Find characteristic frequency
    let k = fht_sample_points(n, dln, -2.0);
    let max_idx = f_transform
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();

    println!("Peak frequency at k ≈ {:.3}", k[max_idx]);
    println!();
}
