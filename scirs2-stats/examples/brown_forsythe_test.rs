use scirs2_core::ndarray::array;
use scirs2_stats::{brown_forsythe, levene};

#[allow(dead_code)]
fn main() {
    // Example data from SciPy documentation
    let a = array![8.88, 9.12, 9.04, 8.98, 9.00, 9.08, 9.01, 8.85, 9.06, 8.99];
    let b = array![8.88, 8.95, 9.29, 9.44, 9.15, 9.58, 8.36, 9.18, 8.67, 9.05];
    let c = array![8.95, 9.12, 8.95, 8.85, 9.03, 8.84, 9.07, 8.98, 8.86, 8.98];

    println!("Brown-Forsythe Test for Homogeneity of Variance");
    println!("===============================================\n");

    // Print sample variances
    println!("Sample variances:");
    println!("Group A: {:.6}", variance(&a));
    println!("Group B: {:.6}", variance(&b));
    println!("Group C: {:.6}", variance(&c));
    println!();

    // Test using Brown-Forsythe test (Levene's with median)
    let samples = vec![a.view(), b.view(), c.view()];
    let (stat, p_value) = brown_forsythe(&samples).unwrap();

    println!("Brown-Forsythe test results:");
    println!("Test statistic: {:.6}", stat);
    println!("P-value: {:.6}", p_value);
    println!(
        "Interpretation at α=0.05: {}",
        if p_value < 0.05 {
            "Reject null hypothesis - variances are different"
        } else {
            "Fail to reject null hypothesis - no evidence variances are different"
        }
    );
    println!();

    // Compare Brown-Forsythe test (median-based) with different center options of Levene's test
    println!("Comparison with Levene's test using different centers:");

    // 1. Brown-Forsythe (Levene with median)
    let (stat_bf, p_bf) = brown_forsythe(&samples).unwrap();

    // 2. Levene with mean (original Levene's test)
    let (stat_mean, p_mean) = levene(&samples, "mean", 0.05).unwrap();

    // 3. Levene with trimmed mean
    let (stat_trim, p_trim) = levene(&samples, "trimmed", 0.1).unwrap();

    println!(
        "Brown-Forsythe (median): stat = {:.6}, p = {:.6}",
        stat_bf, p_bf
    );
    println!("Levene (mean): stat = {:.6}, p = {:.6}", stat_mean, p_mean);
    println!(
        "Levene (trimmed): stat = {:.6}, p = {:.6}",
        stat_trim, p_trim
    );
    println!();

    // Example with data containing outliers
    println!("Example with outliers (demonstrating Brown-Forsythe's robustness):");
    let d = array![1.0, 2.0, 3.0, 4.0, 5.0]; // normal data
    let e = array![1.0, 2.0, 3.0, 4.0, 30.0]; // data with outlier

    println!("Sample variances:");
    println!("Group D: {:.6}", variance(&d));
    println!("Group E: {:.6}", variance(&e));

    let samples2 = vec![d.view(), e.view()];

    // Compare Brown-Forsythe (median-based) with original Levene's test (mean-based)
    let (stat_bf, p_bf) = brown_forsythe(&samples2).unwrap();
    let (stat_mean, p_mean) = levene(&samples2, "mean", 0.05).unwrap();

    println!(
        "Brown-Forsythe (less affected by outliers): stat = {:.6}, p = {:.6}",
        stat_bf, p_bf
    );
    println!(
        "Levene's mean-based test: stat = {:.6}, p = {:.6}",
        stat_mean, p_mean
    );
    println!();

    println!("Note: The Brown-Forsythe test is generally more robust against non-normality");
    println!("and outliers compared to Bartlett's test and mean-based Levene's test.");
}

// Simple variance calculation for the example
#[allow(dead_code)]
fn variance(data: &scirs2_core::ndarray::Array1<f64>) -> f64 {
    let n = data.len() as f64;
    let mean = data.sum() / n;
    let sum_squared_diff = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>();
    sum_squared_diff / (n - 1.0) // Sample variance (unbiased)
}
