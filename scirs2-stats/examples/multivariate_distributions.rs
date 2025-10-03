//! Example of using multivariate distributions in scirs2-stats

use scirs2_core::ndarray::{array, Array2, Axis};
use scirs2_stats::distributions::multivariate;

#[allow(dead_code)]
fn main() {
    println!("Multivariate Distributions Example");
    println!("==================================");

    // ---- Multivariate Normal Distribution ----

    // Define mean vector and covariance matrix
    let mean = array![0.0, 0.0];
    let cov = array![[1.0, 0.5], [0.5, 2.0]];

    // Create the multivariate normal distribution
    let mvn = multivariate::multivariate_normal(mean.clone(), cov.clone())
        .expect("Failed to create multivariate normal distribution");

    // Evaluate PDF at various points
    println!("\nMultivariate Normal Distribution:");
    println!("--------------------------------");
    println!("PDF at [0, 0]: {:.6}", mvn.pdf(&array![0.0, 0.0]));
    println!("PDF at [1, 1]: {:.6}", mvn.pdf(&array![1.0, 1.0]));
    println!("logPDF at [0, 0]: {:.6}", mvn.logpdf(&array![0.0, 0.0]));

    // Generate random samples
    let n_samples = 1000;
    let samples = mvn.rvs(n_samples).expect("Failed to generate samples");
    let sample_mean = samples.mean_axis(Axis(0)).unwrap();

    println!("\nNormal distribution sample statistics (1000 samples):");
    println!("Mean: [{:.4}, {:.4}]", sample_mean[0], sample_mean[1]);

    // Calculate sample covariance
    let centered = samples.clone() - &sample_mean;
    let sample_cov = centered.t().dot(&centered) / (n_samples as f64 - 1.0);
    println!(
        "Covariance:\n[{:.4}, {:.4}]\n[{:.4}, {:.4}]",
        sample_cov[[0, 0]],
        sample_cov[[0, 1]],
        sample_cov[[1, 0]],
        sample_cov[[1, 1]]
    );

    // ---- Multivariate Student's t-Distribution ----

    println!("\nMultivariate Student's t-Distribution (df=5):");
    println!("--------------------------------------------");

    // Create the multivariate t-distribution with 5 degrees of freedom
    let mvt = multivariate::multivariate_t(mean.clone(), cov.clone(), 5.0)
        .expect("Failed to create multivariate t-distribution");

    println!("PDF at [0, 0]: {:.6}", mvt.pdf(&array![0.0, 0.0]));
    println!("PDF at [1, 1]: {:.6}", mvt.pdf(&array![1.0, 1.0]));
    println!("logPDF at [0, 0]: {:.6}", mvt.logpdf(&array![0.0, 0.0]));

    // Generate samples from t-distribution
    let t_samples = mvt.rvs(n_samples).expect("Failed to generate t samples");
    let t_sample_mean = t_samples.mean_axis(Axis(0)).unwrap();

    println!("\nT-distribution sample statistics (1000 samples):");
    println!("Mean: [{:.4}, {:.4}]", t_sample_mean[0], t_sample_mean[1]);

    // Calculate variance ratio to show heavier tails
    let normal_var = calculate_variance(&samples, 0);
    let t_var = calculate_variance(&t_samples, 0);

    println!("\nVariance comparison (shows heavier tails of t-distribution):");
    println!(
        "Variance ratio (t/normal) for dimension 0: {:.4}",
        t_var / normal_var
    );

    // Compare extreme values
    let normal_max = find_max_abs(&samples, 0);
    let t_max = find_max_abs(&t_samples, 0);

    println!("\nExtreme value comparison:");
    println!("Max absolute value in normal samples: {:.4}", normal_max);
    println!("Max absolute value in t samples: {:.4}", t_max);
    println!("Ratio (t/normal): {:.4}", t_max / normal_max);
}

/// Calculate variance of a specific dimension in a matrix of samples
#[allow(dead_code)]
fn calculate_variance(samples: &Array2<f64>, dim: usize) -> f64 {
    let n = samples.shape()[0];
    let mean = samples.mean_axis(Axis(0)).unwrap()[dim];
    let mut sum_sq = 0.0;

    for i in 0..n {
        let diff = samples[[i, dim]] - mean;
        sum_sq += diff * diff;
    }

    sum_sq / (n as f64 - 1.0)
}

/// Find maximum absolute value in a specific dimension
#[allow(dead_code)]
fn find_max_abs(samples: &Array2<f64>, dim: usize) -> f64 {
    let n = samples.shape()[0];
    let mut max_abs = 0.0;

    for i in 0..n {
        let abs_val = samples[[i, dim]].abs();
        if abs_val > max_abs {
            max_abs = abs_val;
        }
    }

    max_abs
}
