//! Kolmogorov-Smirnov test examples
//!
//! This example demonstrates the usage of the Kolmogorov-Smirnov two-sample test,
//! which is used to test if two samples come from the same distribution.

use scirs2_core::ndarray::array;
use scirs2_stats::{distributions, ks_2samp};

#[allow(dead_code)]
fn main() {
    println!("Kolmogorov-Smirnov Test Examples");
    println!("================================\n");

    // Example 1: Similar distributions
    println!("Example 1: Samples from similar distributions");
    let sample1 = array![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    let sample2 = array![0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05];

    let (stat, p_value) = ks_2samp(&sample1.view(), &sample2.view(), "two-sided").unwrap();
    println!("KS test statistic: {:.6}", stat);
    println!("p-value: {:.6}", p_value);
    println!("Null hypothesis (samples come from the same distribution):");
    if p_value < 0.05 {
        println!("  Rejected (p < 0.05)");
    } else {
        println!("  Not rejected (p >= 0.05)");
    }
    println!();

    // Example 2: Different distributions
    println!("Example 2: Samples from different distributions");
    // Generate samples from very different distributions
    let sample1 = array![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    let sample2 = array![5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0];

    let (stat, p_value) = ks_2samp(&sample1.view(), &sample2.view(), "two-sided").unwrap();
    println!("KS test statistic: {:.6}", stat);
    println!("p-value: {:.6}", p_value);
    println!("Null hypothesis (samples come from the same distribution):");
    if p_value < 0.05 {
        println!("  Rejected (p < 0.05)");
    } else {
        println!("  Not rejected (p >= 0.05)");
    }
    println!();

    // Example 3: One-sided test (less)
    println!("Example 3: One-sided test (less)");
    // First sample tends to be less than the second
    let sample1 = array![1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9];
    let sample2 = array![1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4];

    let (stat, p_value) = ks_2samp(&sample1.view(), &sample2.view(), "less").unwrap();
    println!("KS test statistic (less): {:.6}", stat);
    println!("p-value: {:.6}", p_value);
    println!("Null hypothesis (CDF of sample1 ≥ CDF of sample2):");
    if p_value < 0.05 {
        println!("  Rejected in favor of alternative (CDF of sample1 < CDF of sample2)");
    } else {
        println!("  Not rejected (p >= 0.05)");
    }
    println!();

    // Example 4: One-sided test (greater)
    println!("Example 4: One-sided test (greater)");
    // First sample tends to be greater than the second (switch the samples from example 3)
    let sample1 = array![1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4];
    let sample2 = array![1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9];

    let (stat, p_value) = ks_2samp(&sample1.view(), &sample2.view(), "greater").unwrap();
    println!("KS test statistic (greater): {:.6}", stat);
    println!("p-value: {:.6}", p_value);
    println!("Null hypothesis (CDF of sample1 ≤ CDF of sample2):");
    if p_value < 0.05 {
        println!("  Rejected in favor of alternative (CDF of sample1 > CDF of sample2)");
    } else {
        println!("  Not rejected (p >= 0.05)");
    }
    println!();

    // Example 5: Using Random Number Generators
    println!("Example 5: Comparing samples from different random distributions");

    // Generate samples
    // Create a normal distribution sample
    let normal_dist = distributions::norm(0.0f64, 1.0).unwrap();
    let normal_sample = normal_dist.rvs(20).unwrap();

    // Create a uniform distribution sample
    let uniform_dist = distributions::uniform(0.0f64, 1.0).unwrap();
    let uniform_sample = uniform_dist.rvs(20).unwrap();

    // Run the test
    let (stat, p_value) =
        ks_2samp(&normal_sample.view(), &uniform_sample.view(), "two-sided").unwrap();
    println!("KS test statistic: {:.6}", stat);
    println!("p-value: {:.6}", p_value);
    println!("Null hypothesis (normal and uniform samples come from the same distribution):");
    if p_value < 0.05 {
        println!("  Rejected (p < 0.05)");
    } else {
        println!("  Not rejected (p >= 0.05)");
    }
}
