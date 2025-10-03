use scirs2_core::ndarray::array;
use scirs2_stats::{pearson_r, pearsonr};

#[allow(dead_code)]
fn main() {
    println!("Pearson Correlation Coefficient Examples");
    println!("=======================================\n");

    // Example 1: Perfect positive correlation
    let x1 = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let y1 = array![2.0, 4.0, 6.0, 8.0, 10.0]; // y = 2x

    // Basic Pearson correlation (coefficient only)
    let r1: f64 = pearson_r(&x1.view(), &y1.view()).unwrap();

    // Pearson correlation with p-value
    let (r1_with_p, p1) = pearsonr(&x1.view(), &y1.view(), "two-sided").unwrap();

    println!("Example 1: Perfect positive correlation (y = 2x)");
    println!("Pearson r: {:.6}", r1);
    println!("With p-value: r = {:.6}, p = {:.6}", r1_with_p, p1);
    println!(
        "Interpretation: {} correlation, {} significant at α=0.05",
        if r1.abs() > 0.8 {
            "Strong"
        } else if r1.abs() > 0.5 {
            "Moderate"
        } else {
            "Weak"
        },
        if p1 < 0.05 { "statistically" } else { "not" }
    );
    println!();

    // Example 2: Perfect negative correlation
    let x2 = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let y2 = array![10.0, 8.0, 6.0, 4.0, 2.0]; // y = -2x + 12

    let r2: f64 = pearson_r(&x2.view(), &y2.view()).unwrap();
    let (r2_with_p, p2) = pearsonr(&x2.view(), &y2.view(), "two-sided").unwrap();

    println!("Example 2: Perfect negative correlation (y = -2x + 12)");
    println!("Pearson r: {:.6}", r2);
    println!("With p-value: r = {:.6}, p = {:.6}", r2_with_p, p2);
    println!(
        "Interpretation: {} negative correlation, {} significant at α=0.05",
        if r2.abs() > 0.8 {
            "Strong"
        } else if r2.abs() > 0.5 {
            "Moderate"
        } else {
            "Weak"
        },
        if p2 < 0.05 { "statistically" } else { "not" }
    );
    println!();

    // Example 3: No correlation (random data)
    let x3 = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let y3 = array![5.2, 2.1, 8.3, 3.7, 6.9]; // Random values

    let r3: f64 = pearson_r(&x3.view(), &y3.view()).unwrap();
    let (r3_with_p, p3) = pearsonr(&x3.view(), &y3.view(), "two-sided").unwrap();

    println!("Example 3: No linear correlation (random data)");
    println!("Pearson r: {:.6}", r3);
    println!("With p-value: r = {:.6}, p = {:.6}", r3_with_p, p3);
    println!(
        "Interpretation: {} correlation, {} significant at α=0.05",
        if r3.abs() > 0.8 {
            "Strong"
        } else if r3.abs() > 0.5 {
            "Moderate"
        } else {
            "Weak"
        },
        if p3 < 0.05 { "statistically" } else { "not" }
    );
    println!();

    // Example 4: One-sided hypothesis testing
    let x4 = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let y4 = array![1.2, 1.9, 3.2, 4.1, 4.8, 5.9, 7.2, 8.1, 9.3, 9.9]; // Positive correlation

    // Two-sided test (correlation != 0)
    let (r4_two, p4_two) = pearsonr(&x4.view(), &y4.view(), "two-sided").unwrap();

    // One-sided test (correlation > 0)
    let (r4_greater, p4_greater) = pearsonr(&x4.view(), &y4.view(), "greater").unwrap();

    // One-sided test (correlation < 0)
    let (r4_less, p4_less) = pearsonr(&x4.view(), &y4.view(), "less").unwrap();

    println!("Example 4: Hypothesis testing variants");
    println!("Data with positive correlation:");
    println!(
        "Two-sided test (r ≠ 0): r = {:.6}, p = {:.6}",
        r4_two, p4_two
    );
    println!(
        "One-sided test (r > 0): r = {:.6}, p = {:.6}",
        r4_greater, p4_greater
    );
    println!(
        "One-sided test (r < 0): r = {:.6}, p = {:.6}",
        r4_less, p4_less
    );
    println!("Note: For positive correlation, p-value for 'greater' is smaller than 'two-sided',");
    println!("      while p-value for 'less' is 1.0 (since data contradicts this hypothesis)");
    println!();

    // Example 5: Small sample size
    let x5 = array![1.0, 2.0];
    let y5 = array![2.0, 4.0];

    let (r5, p5) = pearsonr(&x5.view(), &y5.view(), "two-sided").unwrap();

    println!("Example 5: Small sample size (n=2)");
    println!("Pearson r: {:.6}", r5);
    println!("With p-value: r = {:.6}, p = {:.6}", r5, p5);
    println!("Note: For n=2, the p-value is always 1.0 regardless of the correlation value");
}
