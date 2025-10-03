use scirs2_core::ndarray::{array, Array2};
use scirs2_stats::distributions::multivariate::inverse_wishart::InverseWishart;
use statrs::statistics::Statistics;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Inverse Wishart Distribution Example");
    println!("-----------------------------------");

    // Create a 2x2 scale matrix (must be positive definite)
    let scale = array![[2.0, 0.5], [0.5, 1.0]];

    // Create an InverseWishart distribution with 5 degrees of freedom
    let df = 5.0;
    let inv_wishart = InverseWishart::new(scale.clone(), df)?;

    println!("Created InverseWishart distribution:");
    println!("  Scale matrix: {:?}", scale);
    println!("  Degrees of freedom: {}", df);
    println!();

    // Calculate distribution properties
    println!("Distribution properties:");
    if let Ok(mean) = inv_wishart.mean() {
        println!("  Mean: {:?}", mean);
    } else {
        println!("  Mean: Undefined (df must be > dim + 1)");
    }
    println!("  Mode: {:?}", inv_wishart.mode());
    println!();

    // Calculate PDF at a few points
    println!("PDF values at different matrices:");

    // A matrix close to the expected mean
    let x1 = array![[1.0, 0.25], [0.25, 0.5]];
    println!("  PDF at {:?}: {}", x1, inv_wishart.pdf(&x1));

    // Identity matrix
    let x2 = array![[1.0, 0.0], [0.0, 1.0]];
    println!("  PDF at {:?}: {}", x2, inv_wishart.pdf(&x2));

    // A matrix further from mean
    let x3 = array![[0.2, 0.05], [0.05, 0.15]];
    println!("  PDF at {:?}: {}", x3, inv_wishart.pdf(&x3));
    println!();

    // Generate random samples
    println!("Generating random samples:");
    let n_samples = 5;
    let samples = inv_wishart.rvs(n_samples)?;

    for (i, sample) in samples.iter().enumerate() {
        println!("  Sample {}: {:?}", i + 1, sample);
    }
    println!();

    // Calculate sample statistics
    println!("Sample statistics:");
    let mut sample_mean = Array2::<f64>::zeros((2, 2));
    for sample in &samples {
        sample_mean += sample;
    }
    sample_mean /= n_samples as f64;

    println!(
        "  Sample mean (from {} samples): {:?}",
        n_samples, sample_mean
    );
    if let Ok(true_mean) = inv_wishart.mean() {
        println!("  True mean: {:?}", true_mean);
    }
    println!();

    // Demonstrate Bayesian application
    println!("Bayesian application example:");
    println!("  The InverseWishart distribution is commonly used as a conjugate");
    println!("  prior for covariance matrices in Bayesian inference.");
    println!();
    println!("  In a multivariate normal model with unknown covariance matrix Σ,");
    println!("  if we set a prior Σ ~ InverseWishart(Ψ, ν), then after");
    println!("  observing data, the posterior is also an InverseWishart.");
    println!();
    println!("  For example, with a scatter matrix S, n observations, and prior");
    println!("  parameters (Ψ, ν), the posterior is:");
    println!("  Σ | data ~ InverseWishart(Ψ + S, ν + n)");

    // Example of posterior calculation
    let data_scatter = array![[0.8, 0.3], [0.3, 1.2]]; // Example scatter matrix
    let n_observations = 10;

    // Calculate posterior parameters
    let posterior_scale = scale.clone() + data_scatter.clone();
    let posterior_df = df + n_observations as f64;

    println!();
    println!("  Prior:     InverseWishart({:?}, {})", scale, df);
    println!(
        "  Data:      Scatter = {:?}, n = {}",
        data_scatter, n_observations
    );
    println!(
        "  Posterior: InverseWishart({:?}, {})",
        posterior_scale, posterior_df
    );

    // Create posterior distribution
    let posterior = InverseWishart::new(posterior_scale.clone(), posterior_df)?;

    // Generate samples from posterior
    println!();
    println!("  Posterior sample:");
    let posterior_sample = posterior.rvs_single()?;
    println!("    {:?}", posterior_sample);

    Ok(())
}
