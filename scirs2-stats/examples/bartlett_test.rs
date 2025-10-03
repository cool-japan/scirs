use scirs2_core::ndarray::array;
use scirs2_stats::bartlett;

#[allow(dead_code)]
fn main() {
    // Example data from SciPy documentation
    let a = array![8.88, 9.12, 9.04, 8.98, 9.00, 9.08, 9.01, 8.85, 9.06, 8.99];
    let b = array![8.88, 8.95, 9.29, 9.44, 9.15, 9.58, 8.36, 9.18, 8.67, 9.05];
    let c = array![8.95, 9.12, 8.95, 8.85, 9.03, 8.84, 9.07, 8.98, 8.86, 8.98];

    println!("Bartlett's Test for Homogeneity of Variance");
    println!("===========================================\n");

    // Print sample variances
    println!("Sample variances:");
    println!("Group A: {:.6}", variance(&a));
    println!("Group B: {:.6}", variance(&b));
    println!("Group C: {:.6}", variance(&c));
    println!();

    // Test whether the groups have equal variances
    let samples = vec![a.view(), b.view(), c.view()];
    let (stat, p_value) = bartlett(&samples).unwrap();

    println!("Bartlett's test results:");
    println!("Test statistic: {:.6}", stat);
    println!("P-value: {:.10}", p_value);
    println!(
        "Interpretation at α=0.05: {}",
        if p_value < 0.05 {
            "Reject null hypothesis - variances are different"
        } else {
            "Fail to reject null hypothesis - no evidence variances are different"
        }
    );
    println!();

    // Compare with Levene's test:
    println!("Note: Bartlett's test is more powerful than Levene's test when");
    println!("the assumption of normality is satisfied, but it is more sensitive");
    println!("to departures from normality.");
    println!();

    // Example with similar variances
    println!("Example with more similar variances:");
    let d = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let e = array![0.8, 1.9, 3.2, 3.9, 5.1];

    println!("Sample variances:");
    println!("Group D: {:.6}", variance(&d));
    println!("Group E: {:.6}", variance(&e));

    let samples2 = vec![d.view(), e.view()];
    let (stat, p_value) = bartlett(&samples2).unwrap();

    println!("Bartlett's test results:");
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

    // Example with clearly different variances
    println!("Example with clearly different variances:");
    let f = array![1.0, 1.1, 1.2, 0.9, 1.0]; // low variance
    let g = array![1.0, 3.0, 5.0, 7.0, 9.0]; // high variance

    println!("Sample variances:");
    println!("Group F: {:.6}", variance(&f));
    println!("Group G: {:.6}", variance(&g));

    let samples3 = vec![f.view(), g.view()];
    let (stat, p_value) = bartlett(&samples3).unwrap();

    println!("Bartlett's test results:");
    println!("Test statistic: {:.6}", stat);
    println!("P-value: {:.10}", p_value);
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
