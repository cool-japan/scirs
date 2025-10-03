//! Example of using multivariate Student's t-distribution in scirs2-stats

use scirs2_core::ndarray::array;
use scirs2_stats::distributions::multivariate;

#[allow(dead_code)]
fn main() {
    // Create a multivariate Student's t-distribution
    println!("Multivariate Student's t-Distribution Example");
    println!("===========================================");

    // Define mean vector, scale matrix, and degrees of freedom
    let mean = array![0.0, 0.0];
    let scale = array![[1.0, 0.5], [0.5, 2.0]];
    let df = 5.0;

    // Create the multivariate t-distribution
    let mvt = multivariate::multivariate_t(mean.clone(), scale.clone(), df)
        .expect("Failed to create multivariate t-distribution");

    // Evaluate PDF at various points
    println!("\nProbability Density Function (PDF):");
    println!("----------------------------------");
    println!("PDF at [0, 0]: {}", mvt.pdf(&array![0.0, 0.0]));
    println!("PDF at [1, 1]: {}", mvt.pdf(&array![1.0, 1.0]));
    println!("PDF at [-1, 2]: {}", mvt.pdf(&array![-1.0, 2.0]));

    // Evaluate log-PDF
    println!("\nLog-Probability Density Function (logPDF):");
    println!("------------------------------------------");
    println!("logPDF at [0, 0]: {}", mvt.logpdf(&array![0.0, 0.0]));
    println!("logPDF at [1, 1]: {}", mvt.logpdf(&array![1.0, 1.0]));

    // Compare with multivariate normal distribution
    println!("\nComparison with Multivariate Normal Distribution:");
    println!("-----------------------------------------------");
    let mvn = multivariate::multivariate_normal(mean.clone(), scale.clone())
        .expect("Failed to create multivariate normal distribution");

    println!("Normal PDF at [2, 2]: {}", mvn.pdf(&array![2.0, 2.0]));
    println!("t-dist PDF at [2, 2]: {}", mvt.pdf(&array![2.0, 2.0]));
    // The t-distribution should have heavier tails
    println!("Normal PDF at [5, 5]: {}", mvn.pdf(&array![5.0, 5.0]));
    println!("t-dist PDF at [5, 5]: {}", mvt.pdf(&array![5.0, 5.0]));

    // Generate random samples
    println!("\nRandom Sampling:");
    println!("---------------");

    // Generate a single sample
    let single_sample = mvt.rvs_single().expect("Failed to generate sample");
    println!("Single sample: {}", single_sample);

    // Generate multiple samples
    let n_samples = 10;
    let samples = mvt.rvs(n_samples).expect("Failed to generate samples");
    println!(
        "\nMultiple samples ({} rows, {} columns):",
        samples.shape()[0],
        samples.shape()[1]
    );
    for i in 0..n_samples.min(5) {
        let row = samples.slice(scirs2_core::ndarray::s![i, ..]);
        println!("Sample {}: {}", i, row);
    }
    if n_samples > 5 {
        println!("... (and {} more samples)", n_samples - 5);
    }

    // Compute sample statistics
    let sample_mean = samples.mean_axis(scirs2_core::ndarray::Axis(0)).unwrap();
    println!(
        "\nSample mean: [{:.3}, {:.3}] (Expected: [0, 0])",
        sample_mean[0], sample_mean[1]
    );

    // Example with different parameters
    println!("\n\nExample with different parameters:");
    println!("----------------------------------");
    let mean2 = array![5.0, -2.0, 3.0];
    let scale2 = array![[2.0, 0.3, 0.1], [0.3, 1.0, 0.4], [0.1, 0.4, 1.5]];
    let df2 = 3.0; // Lower df means heavier tails

    let mvt2 = multivariate::multivariate_t(mean2, scale2, df2)
        .expect("Failed to create 3D multivariate t-distribution");

    println!(
        "Created 3D multivariate t-distribution with df = {}",
        mvt2.df()
    );
    println!("Dimension: {}", mvt2.dim());

    // Generate a single 3D sample
    let sample3d = mvt2.rvs_single().expect("Failed to generate 3D sample");
    println!("\nSingle 3D sample: {}", sample3d);
}
