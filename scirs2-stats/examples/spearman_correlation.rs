use scirs2_core::ndarray::array;
use scirs2_stats::{spearman_r, spearmanr};

#[allow(dead_code)]
fn main() {
    // Create two datasets with a monotonic (but not linear) relationship
    let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = array![1.0, 4.0, 9.0, 16.0, 25.0]; // y = x²

    // Calculate Spearman correlation coefficient (without p-value)
    let rho = spearman_r(&x.view(), &y.view()).unwrap();
    println!("Spearman correlation coefficient: {}", rho);
    // Perfect monotonic relationship (rho should be 1.0)
    assert!((rho - 1.0f64).abs() < 1e-10f64);

    // Calculate Spearman correlation coefficient with p-value
    let (rho, p_value) = spearmanr(&x.view(), &y.view(), "two-sided").unwrap();
    println!("Spearman correlation coefficient: {}", rho);
    println!("Two-sided p-value: {}", p_value);
    // Perfect monotonic relationship (rho should be 1.0)
    assert!((rho - 1.0f64).abs() < 1e-10f64);

    // A p-value should be reported, though with a perfect correlation
    // and a small sample, it may still be statistically significant
    println!(
        "Is correlation significant at alpha=0.05? {}",
        p_value < 0.05
    );

    // Try with a non-perfect correlation
    let y2 = array![1.0, 3.8, 8.7, 17.2, 23.5]; // Slightly noisy data
    let (rho2, p_value2) = spearmanr(&x.view(), &y2.view(), "two-sided").unwrap();
    println!("\nSpearman correlation with noisy data: {}", rho2);
    println!("Two-sided p-value: {}", p_value2);

    // Try with a negative correlation (decreasing relationship)
    let y3 = array![25.0, 16.0, 9.0, 4.0, 1.0]; // y = (6-x)²
    let (rho3, p_value3) = spearmanr(&x.view(), &y3.view(), "two-sided").unwrap();
    println!(
        "\nSpearman correlation with negative relationship: {}",
        rho3
    );
    println!("Two-sided p-value: {}", p_value3);
    // Perfect negative monotonic relationship (rho should be -1.0)
    assert!((rho3 - (-1.0f64)).abs() < 1e-10f64);

    // Try with one-sided tests
    let (_, p_greater) = spearmanr(&x.view(), &y.view(), "greater").unwrap();
    let (_, p_less) = spearmanr(&x.view(), &y.view(), "less").unwrap();
    println!("\nOne-sided tests for positive correlation:");
    println!("P-value (greater): {}", p_greater); // Should be small
    println!("P-value (less): {}", p_less); // Should be 1.0

    // Try with no correlation
    let y4 = array![10.0, 5.0, 15.0, 8.0, 12.0]; // Random values
    let (rho4, p_value4) = spearmanr(&x.view(), &y4.view(), "two-sided").unwrap();
    println!("\nSpearman correlation with uncorrelated data: {}", rho4);
    println!("Two-sided p-value: {}", p_value4);
    // Probably not significant given small sample size
}
