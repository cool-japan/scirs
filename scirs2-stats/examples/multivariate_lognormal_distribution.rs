use scirs2_core::ndarray::{array, Array1, Array2};
use scirs2_stats::distributions::multivariate::multivariate_lognormal::MultivariateLognormal;
use statrs::statistics::Statistics;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Multivariate Lognormal Distribution Example");
    println!("------------------------------------------");

    // Create a 2D multivariate lognormal distribution
    // Using small covariance values to keep the distribution reasonable
    let mu = array![0.0, 0.0]; // Mean of underlying normal, gives median = [1.0, 1.0]
    let sigma = array![[0.25, 0.1], [0.1, 0.25]]; // Covariance of underlying normal

    let mvln = MultivariateLognormal::new(mu.clone(), sigma.clone())?;

    println!("Created Multivariate Lognormal distribution:");
    println!("  Normal mean vector (μ): {:?}", mu);
    println!("  Normal covariance matrix (Σ): {:?}", sigma);
    println!();

    // Calculate distribution properties
    println!("Distribution properties:");
    let mean = mvln.mean();
    println!("  Mean: {:?}", mean);
    let median = mvln.median();
    println!("  Median: {:?}", median);
    let mode = mvln.mode();
    println!("  Mode: {:?}", mode);
    let cov = mvln.cov();
    println!("  Covariance matrix:");
    for i in 0..2 {
        println!("    {:?}", cov.row(i));
    }
    println!();

    // Calculate PDF at various points
    println!("PDF values at different points:");

    // At median point [1.0, 1.0]
    let x1 = array![1.0, 1.0];
    let pdf1 = mvln.pdf(&x1);
    println!("  PDF at {:?} (median): {:.6}", x1, pdf1);

    // At a point with both values greater than median
    let x2 = array![2.0, 2.0];
    let pdf2 = mvln.pdf(&x2);
    println!("  PDF at {:?}: {:.6}", x2, pdf2);

    // At a point with both values less than median
    let x3 = array![0.5, 0.5];
    let pdf3 = mvln.pdf(&x3);
    println!("  PDF at {:?}: {:.6}", x3, pdf3);

    // At a point with one value 0 (PDF should be 0)
    let x4 = array![0.0, 1.0];
    let pdf4 = mvln.pdf(&x4);
    println!("  PDF at {:?}: {:.6}", x4, pdf4);
    println!();

    // Generate random samples
    println!("Generating random samples:");
    let n_samples = 1000;
    let samples = mvln.rvs(n_samples)?;

    // Display first few samples
    for i in 0..5 {
        println!("  Sample {}: {:?}", i + 1, samples.row(i));
    }
    println!("  ...");
    println!();

    // Calculate sample statistics
    let sample_mean = calculate_mean(&samples, n_samples);
    let sample_median = calculate_median(&samples, n_samples);
    let sample_cov = calculate_covariance(&samples, &sample_mean, n_samples);

    println!("Sample statistics from {} samples:", n_samples);
    println!("  Sample mean: {:?}", sample_mean);
    println!("  Theoretical mean: {:?}", mean);
    println!("  Sample median: {:?}", sample_median);
    println!("  Theoretical median: {:?}", median);
    println!("  Sample covariance matrix:");
    for i in 0..2 {
        println!("    {:?}", sample_cov.row(i));
    }
    println!();

    // Illustrate the effect of correlation
    println!("Effect of correlation on the distribution:");
    let correlations = vec![0.0, 0.4, 0.8];

    for &corr in &correlations {
        // Create sigma matrix with given correlation
        let var: f64 = 0.25; // Variance for both dimensions
        let cov = corr * f64::sqrt(var * var); // Covariance from correlation
        let sigma_corr = array![[var, cov], [cov, var]];

        let mvln_corr = MultivariateLognormal::new(mu.clone(), sigma_corr)?;
        let cov_matrix = mvln_corr.cov();

        // Calculate correlation from covariance
        let corr_computed = cov_matrix[[0, 1]] / (cov_matrix[[0, 0]] * cov_matrix[[1, 1]]).sqrt();

        println!(
            "  Normal correlation: {:.1}, Lognormal correlation: {:.6}",
            corr, corr_computed
        );
    }
    println!();

    // Applications section
    println!("Applications of the Multivariate Lognormal distribution:");
    println!("1. Modeling asset returns in finance");
    println!("2. Modeling correlated positive-valued random variables");
    println!("3. Modeling size distributions in natural phenomena");
    println!("4. Modeling rainfall and hydrological data");
    println!("5. Income and wealth distribution modeling");

    Ok(())
}

// Helper function to calculate mean of samples
#[allow(dead_code)]
fn calculate_mean(samples: &Array2<f64>, nsamples: usize) -> Array1<f64> {
    let mut mean = Array1::zeros(samples.shape()[1]);

    for i in 0..nsamples {
        for j in 0..samples.shape()[1] {
            mean[j] += samples[[i, j]];
        }
    }

    mean / nsamples as f64
}

// Helper function to calculate median of samples
#[allow(dead_code)]
fn calculate_median(samples: &Array2<f64>, nsamples: usize) -> Array1<f64> {
    let dim = samples.shape()[1];
    let mut median = Array1::zeros(dim);

    for j in 0..dim {
        let mut values: Vec<f64> = samples.column(j).to_vec();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        if nsamples.is_multiple_of(2) {
            // Even number of samples, take average of middle two
            median[j] = (values[nsamples / 2 - 1] + values[nsamples / 2]) / 2.0;
        } else {
            // Odd number of samples, take middle value
            median[j] = values[nsamples / 2];
        }
    }

    median
}

// Helper function to calculate covariance matrix of samples
#[allow(dead_code)]
fn calculate_covariance(
    samples: &Array2<f64>,
    mean: &Array1<f64>,
    n_samples: usize,
) -> Array2<f64> {
    let dim = samples.shape()[1];
    let mut cov = Array2::zeros((dim, dim));

    for i in 0..n_samples {
        for j in 0..dim {
            for k in 0..dim {
                cov[[j, k]] += (samples[[i, j]] - mean[j]) * (samples[[i, k]] - mean[k]);
            }
        }
    }

    cov / (n_samples as f64 - 1.0)
}
