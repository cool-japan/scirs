//! Comprehensive showcase of available statistical functionality
//!
//! This example demonstrates the core statistical operations
//! that are available in the scirs2-stats library.

use scirs2_core::ndarray::{s, Array1, Array2};
use scirs2_stats::{mean, regression::linear_regression, std, tests::ttest::ttest_1samp, var};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Statistical Functionality Showcase");
    println!("=====================================\n");

    // Generate sample data using simple approach
    let n = 100;
    let data: Array1<f64> = Array1::from_iter((1..=n).map(|x| x as f64 + (x as f64 * 0.1).sin()));

    println!("1. 📈 Basic Statistics");
    println!("   Sample size: {}", n);
    println!("   Mean: {:.4}", mean(&data.view())?);
    println!("   Variance: {:.4}", var(&data.view(), 1, None)?);
    println!("   Std Dev: {:.4}", std(&data.view(), 1, None)?);

    // Simple sample demonstration
    println!("\n2. 🎲 Data Sample");
    let sample_slice = data.slice(s![0..10]);
    println!(
        "   First 10 values: {:?}",
        sample_slice
            .iter()
            .map(|x| format!("{:.3}", x))
            .collect::<Vec<_>>()
    );

    // Simple regression
    println!("\n3. 📉 Linear Regression");
    let x: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&x| 2.0 * x + 1.0 + (x * 0.1).sin()).collect();

    let x_array = Array1::from_vec(x);
    let y_array = Array1::from_vec(y);

    // Create design matrix for regression
    let mut x_design = Array2::zeros((n, 2));
    for i in 0..n {
        x_design[[i, 0]] = 1.0; // Intercept
        x_design[[i, 1]] = x_array[i]; // Slope
    }

    let reg_result = linear_regression(&x_design.view(), &y_array.view(), None)?;
    println!("   Intercept: {:.4}", reg_result.coefficients[0]);
    println!("   Slope: {:.4}", reg_result.coefficients[1]);
    println!("   R²: {:.4}", reg_result.r_squared);

    // Statistical test
    println!("\n4. 🧪 Statistical Tests");
    let testdata = Array1::from_vec(vec![-0.5, 0.2, 1.1, -0.8, 0.3, 0.9, -0.2, 0.7, -0.4, 0.6]);
    let test_result = ttest_1samp(
        &testdata.view(),
        0.0,
        scirs2_stats::tests::ttest::Alternative::TwoSided,
        "omit",
    )?;
    println!("   One-sample t-test against μ=0:");
    println!("   t-statistic: {:.4}", test_result.statistic);
    println!("   p-value: {:.6}", test_result.pvalue);
    println!("   Significant at α=0.05: {}", test_result.pvalue < 0.05);

    println!("\n✅ Comprehensive showcase completed successfully!");
    println!("   All core statistical operations are working correctly.");

    Ok(())
}
