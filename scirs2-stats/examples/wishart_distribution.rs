use scirs2_core::ndarray::{array, s, Array2};
use scirs2_stats::distributions::multivariate;
use statrs::statistics::Statistics;

#[allow(dead_code)]
fn main() {
    println!("Wishart Distribution Example");
    println!("===========================");

    // --- Example 1: Simple Wishart with Identity Scale Matrix ---
    println!("\nSimple Wishart (Σ = I, df = 5):");
    println!("----------------------------------");

    // Create a Wishart distribution with identity scale matrix
    let scale_identity = array![[1.0, 0.0], [0.0, 1.0]];
    let df = 5.0;
    let wishart_identity = multivariate::wishart(scale_identity.clone(), df).unwrap();

    // Print distribution properties
    println!("Dimension: {}", wishart_identity.dim);
    println!("Degrees of freedom: {}", wishart_identity.df);

    // Calculate mean
    let mean = wishart_identity.mean();
    println!("\nMean of the distribution:");
    println!("[{:.2}, {:.2}]", mean[[0, 0]], mean[[0, 1]]);
    println!("[{:.2}, {:.2}]", mean[[1, 0]], mean[[1, 1]]);

    // Calculate mode (if exists)
    if let Some(mode) = wishart_identity.mode() {
        println!("\nMode of the distribution:");
        println!("[{:.2}, {:.2}]", mode[[0, 0]], mode[[0, 1]]);
        println!("[{:.2}, {:.2}]", mode[[1, 0]], mode[[1, 1]]);
    } else {
        println!("\nMode does not exist for these parameters");
    }

    // --- Example 2: Wishart with Correlated Scale Matrix ---
    println!("\nCorrelated Wishart (df = 8):");
    println!("------------------------------");

    // Create a Wishart distribution with correlated scale matrix
    let scale_correlated = array![[2.0, 0.8], [0.8, 1.5]];
    let df_corr = 8.0;
    let wishart_correlated = multivariate::wishart(scale_correlated.clone(), df_corr).unwrap();

    println!("Scale matrix:");
    println!(
        "[{:.2}, {:.2}]",
        scale_correlated[[0, 0]],
        scale_correlated[[0, 1]]
    );
    println!(
        "[{:.2}, {:.2}]",
        scale_correlated[[1, 0]],
        scale_correlated[[1, 1]]
    );

    // Evaluate PDF at different points
    let points = [
        array![[8.0, 1.6], [1.6, 6.0]],   // Close to the expected mean
        array![[20.0, 4.0], [4.0, 15.0]], // Larger values
        array![[4.0, 0.4], [0.4, 3.0]],   // Smaller values
    ];

    println!("\nPDF evaluations:");
    for (i, point) in points.iter().enumerate() {
        println!("Point {}:", i + 1);
        println!("[{:.1}, {:.1}]", point[[0, 0]], point[[0, 1]]);
        println!("[{:.1}, {:.1}]", point[[1, 0]], point[[1, 1]]);
        println!("PDF: {:.10e}", wishart_correlated.pdf(point));
        println!("Log PDF: {:.6}", wishart_correlated.logpdf(point));
        println!();
    }

    // --- Example 3: Random Sampling and Statistics ---
    println!("\nRandom Sampling and Statistics:");
    println!("--------------------------------");

    // Generate samples
    let n_samples = 1000;
    let samples = wishart_correlated.rvs(n_samples).unwrap();

    println!("Generated {} samples from Wishart distribution", n_samples);

    // Print a few samples
    println!("\nFirst sample:");
    print_matrix(&samples[0]);

    // Check sample properties
    let sample_mean = compute_mean(&samples);

    println!("\nSample mean (should be close to df * scale):");
    print_matrix(&sample_mean);

    println!("\nExpected mean (df * scale):");
    let expected_mean = scale_correlated.mapv(|x| x * df_corr);
    print_matrix(&expected_mean);

    // --- Example 4: Bayesian Covariance Estimation ---
    println!("\nBayesian Covariance Estimation Example:");
    println!("--------------------------------------");
    println!("In Bayesian statistics, the Wishart distribution is often used");
    println!("as a prior for covariance matrices. Here's a simple example:");

    // Prior parameters
    let prior_df = 3.0;
    let prior_scale = array![[1.0, 0.0], [0.0, 1.0]];
    println!("\nPrior: Wishart with df = {}, scale = I", prior_df);

    // Simulated data (normally distributed with true covariance [[2.0, 0.8], [0.8, 1.5]])
    println!("\nGenerating simulated data...");
    let true_cov = array![[2.0, 0.8], [0.8, 1.5]];
    let true_mean = array![0.0, 0.0];
    let mvn = multivariate::multivariate_normal(true_mean, true_cov.clone()).unwrap();

    let data_samples = mvn.rvs(50).unwrap();

    // Compute scatter matrix from data
    let mut scatter_matrix = Array2::<f64>::zeros((2, 2));
    for i in 0..data_samples.shape()[0] {
        let sample = data_samples.slice(s![i, ..]);
        for j in 0..2 {
            for k in 0..2 {
                scatter_matrix[[j, k]] += sample[j] * sample[k];
            }
        }
    }

    println!("\nScatter matrix from data:");
    print_matrix(&scatter_matrix);

    // Posterior parameters
    let posterior_df = prior_df + data_samples.shape()[0] as f64;
    let posterior_scale = prior_scale + scatter_matrix;

    println!(
        "\nPosterior: Wishart with df = {}, scale = prior_scale + scatter",
        posterior_df
    );
    println!("\nPosterior scale matrix:");
    print_matrix(&posterior_scale);

    // Generate samples from posterior
    let posterior = multivariate::wishart(posterior_scale, posterior_df).unwrap();
    let posterior_samples = posterior.rvs(1000).unwrap();

    // Compute mean of posterior samples (estimator of the covariance)
    let posterior_mean = compute_mean(&posterior_samples);
    println!("\nPosterior mean (estimator of the covariance):");
    print_matrix(&posterior_mean);

    println!("\nTrue covariance matrix:");
    print_matrix(&true_cov);
}

#[allow(dead_code)]
fn print_matrix(matrix: &Array2<f64>) {
    for i in 0..matrix.shape()[0] {
        print!("[");
        for j in 0..matrix.shape()[1] {
            print!("{:.4}", matrix[[i, j]]);
            if j < matrix.shape()[1] - 1 {
                print!(", ");
            }
        }
        println!("]");
    }
}

#[allow(dead_code)]
fn compute_mean(samples: &[Array2<f64>]) -> Array2<f64> {
    let n_samples = samples.len();
    if n_samples == 0 {
        return Array2::<f64>::zeros((0, 0));
    }

    let shape = samples[0].shape();
    let mut mean = Array2::<f64>::zeros((shape[0], shape[1]));

    for sample in samples {
        mean += sample;
    }

    mean /= n_samples as f64;
    mean
}
