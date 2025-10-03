//! SciRS2 Statistics Showcase
//!
//! This example demonstrates the comprehensive statistical functionality available
//! in scirs2-stats, including distributions, descriptive statistics, tests, and correlations.

use scirs2_core::ndarray::array;
use scirs2_stats::distributions::{ChiSquare, Normal, Poisson};
use scirs2_stats::tests::ttest::Alternative;
use scirs2_stats::traits::{ContinuousDistribution, DiscreteDistribution, Distribution};
use scirs2_stats::{mean, pearson_r, std, ttest_1samp};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 SciRS2 Statistics Showcase");
    println!("=============================\n");

    // Demonstrate distributions
    demonstrate_distributions()?;

    // Demonstrate descriptive statistics
    demonstrate_descriptive_stats()?;

    // Demonstrate statistical tests
    demonstrate_statistical_tests()?;

    println!("\n✅ SciRS2 statistics showcase completed successfully!");

    Ok(())
}

/// Demonstrate statistical distributions
#[allow(dead_code)]
fn demonstrate_distributions() -> Result<(), Box<dyn std::error::Error>> {
    println!("📈 Statistical Distributions");
    println!("----------------------------");

    // Normal distribution
    let normal = Normal::new(0.0, 1.0)?;
    println!("🔔 Normal Distribution (μ=0, σ=1):");
    println!("  Mean: {:.3}", normal.mean());
    println!("  Variance: {:.3}", normal.var());
    println!("  PDF(0.0): {:.6}", normal.pdf(0.0));
    println!("  CDF(1.96): {:.6}", normal.cdf(1.96)); // ~0.975

    // Chi-square distribution
    let chi2 = ChiSquare::new(2.0, 0.0, 1.0)?;
    println!("\n🎯 Chi-Square Distribution (df=2):");
    println!("  Mean: {:.3}", chi2.mean());
    println!("  Variance: {:.3}", chi2.var());
    println!("  PDF(1.0): {:.6}", chi2.pdf(1.0));
    println!("  CDF(2.0): {:.6}", chi2.cdf(2.0));

    // Poisson distribution
    let poisson = Poisson::new(3.0, 0.0)?;
    println!("\n⚡ Poisson Distribution (λ=3):");
    println!("  Mean: {:.3}", poisson.mean());
    println!("  Variance: {:.3}", poisson.var());
    println!("  PMF(2): {:.6}", poisson.pmf(2.0));
    println!("  PMF(3): {:.6}", poisson.pmf(3.0));

    // Generate some samples
    println!("\n🎲 Random Samples:");
    let normal_samples = normal.rvs(5)?;
    let poisson_samples = poisson.rvs(5)?;
    println!("  Normal samples: {:?}", normal_samples.to_vec());
    println!("  Poisson samples: {:?}", poisson_samples.to_vec());

    Ok(())
}

/// Demonstrate descriptive statistics
#[allow(dead_code)]
fn demonstrate_descriptive_stats() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📊 Descriptive Statistics");
    println!("-------------------------");

    // Create sample data
    let data = array![
        1.2, 2.5, 3.1, 2.8, 4.0, 3.5, 2.9, 3.8, 4.2, 2.4, 3.7, 2.6, 3.3, 4.1, 2.7, 3.9, 2.3, 3.4,
        4.3, 2.8
    ];

    println!(
        "🔢 Sample Data (n={}): {:?}",
        data.len(),
        &data.to_vec()[..5]
    );
    println!("   ... (showing first 5 values)");

    // Calculate basic statistics
    let mean_val = mean(&data.view())?;
    let std_val = std(&data.view(), 1, None)?; // ddof=1 for sample standard deviation

    println!("\n📈 Basic Statistics:");
    println!("  Mean: {:.3}", mean_val);
    println!("  Standard Deviation: {:.3}", std_val);
    println!(
        "  Min: {:.3}",
        data.iter().fold(f64::INFINITY, |a, &b| a.min(b))
    );
    println!(
        "  Max: {:.3}",
        data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
    );

    // Demonstrate correlation with another variable
    let x = array![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        17.0, 18.0, 19.0, 20.0
    ];
    let y = &data;

    let correlation = pearson_r(&x.view(), &y.view())?;
    println!("\n🔗 Correlation Analysis:");
    println!("  Pearson correlation coefficient: {:.4}", correlation);

    Ok(())
}

/// Demonstrate statistical tests
#[allow(dead_code)]
fn demonstrate_statistical_tests() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🧪 Statistical Tests");
    println!("--------------------");

    // Create sample data for testing
    let sampledata = array![
        5.1, 4.9, 6.2, 5.7, 5.3, 5.0, 5.8, 5.4, 5.2, 5.6, 4.8, 5.9, 5.1, 5.3, 5.5, 5.2, 5.4, 5.0,
        5.1, 4.9
    ];

    println!(
        "🔬 Sample Data (n={}): Testing if mean = 5.0",
        sampledata.len()
    );

    // One-sample t-test
    let ttest_result = ttest_1samp(&sampledata.view(), 5.0, Alternative::TwoSided, "propagate")?;

    println!("\n📊 One-Sample t-Test Results:");
    println!("  Null hypothesis: μ = 5.0");
    println!("  Alternative: μ ≠ 5.0 (two-sided)");
    println!("  t-statistic: {:.4}", ttest_result.statistic);
    println!("  p-value: {:.6}", ttest_result.pvalue);

    let significance_level = 0.05;
    let is_significant = ttest_result.pvalue < significance_level;
    println!(
        "  Significant at α = {}: {}",
        significance_level,
        if is_significant { "Yes ✅" } else { "No ❌" }
    );

    if is_significant {
        println!("  Conclusion: Reject H₀ - the mean is significantly different from 5.0");
    } else {
        println!("  Conclusion: Fail to reject H₀ - insufficient evidence that mean ≠ 5.0");
    }

    // Show confidence interval information
    let sample_mean = mean(&sampledata.view())?;
    let sample_std = std(&sampledata.view(), 1, None)?;
    let n = sampledata.len() as f64;
    let se = sample_std / n.sqrt();

    println!("\n📈 Descriptive Statistics:");
    println!("  Sample mean: {:.4}", sample_mean);
    println!("  Sample std dev: {:.4}", sample_std);
    println!("  Standard error: {:.4}", se);
    println!(
        "  95% CI estimate: ({:.4}, {:.4})",
        sample_mean - 1.96 * se,
        sample_mean + 1.96 * se
    );

    Ok(())
}
