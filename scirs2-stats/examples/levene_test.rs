use scirs2_core::ndarray::array;
use scirs2_stats::levene;

#[allow(dead_code)]
fn main() {
    // Example data from SciPy documentation
    let a = array![8.88, 9.12, 9.04, 8.98, 9.00, 9.08, 9.01, 8.85, 9.06, 8.99];
    let b = array![8.88, 8.95, 9.29, 9.44, 9.15, 9.58, 8.36, 9.18, 8.67, 9.05];
    let c = array![8.95, 9.12, 8.95, 8.85, 9.03, 8.84, 9.07, 8.98, 8.86, 8.98];

    println!("Levene's Test for Homogeneity of Variance");
    println!("=========================================\n");

    // Print sample variances
    println!("Sample variances:");
    println!("Group A: {:.6}", variance(&a));
    println!("Group B: {:.6}", variance(&b));
    println!("Group C: {:.6}", variance(&c));
    println!();

    // Test using different center options

    // 1. Using median (default, recommended for skewed distributions)
    let samples = vec![a.view(), b.view(), c.view()];
    let (stat, p_value) = levene(&samples, "median", 0.05).unwrap();

    println!("Levene's test with center='median':");
    println!("Test statistic (W): {:.6}", stat);
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

    // 2. Using mean (original Levene's test, good for symmetric distributions)
    let (stat, p_value) = levene(&samples, "mean", 0.05).unwrap();

    println!("Levene's test with center='mean':");
    println!("Test statistic (W): {:.6}", stat);
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

    // 3. Using trimmed mean (good for heavy-tailed distributions)
    let (stat, p_value) = levene(&samples, "trimmed", 0.1).unwrap();

    println!("Levene's test with center='trimmed' (10% trim):");
    println!("Test statistic (W): {:.6}", stat);
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

    // Example with clearly different variances
    let d = array![1.0, 1.1, 1.2, 0.9, 1.0]; // low variance
    let e = array![1.0, 3.0, 5.0, 7.0, 9.0]; // high variance

    println!("Example with clearly different variances:");
    println!("Group D variance: {:.6}", variance(&d));
    println!("Group E variance: {:.6}", variance(&e));

    let samples2 = vec![d.view(), e.view()];
    let (stat, p_value) = levene(&samples2, "median", 0.05).unwrap();

    println!("Levene's test with center='median':");
    println!("Test statistic (W): {:.6}", stat);
    println!("P-value: {:.6}", p_value);
    println!(
        "Interpretation at α=0.05: {}",
        if p_value < 0.05 {
            "Reject null hypothesis - variances are different"
        } else {
            "Fail to reject null hypothesis - no evidence variances are different"
        }
    );
}

// Simple variance calculation for the example
#[allow(dead_code)]
fn variance(data: &scirs2_core::ndarray::Array1<f64>) -> f64 {
    let n = data.len() as f64;
    let mean = data.sum() / n;
    let sum_squared_diff = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>();
    sum_squared_diff / (n - 1.0) // Sample variance (unbiased)
}
