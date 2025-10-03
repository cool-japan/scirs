use scirs2_core::ndarray::{array, Array1};
use scirs2_stats::distributions::multivariate;

#[allow(dead_code)]
fn main() {
    println!("Dirichlet Distribution Example");
    println!("=============================");

    // --- Example 1: Uniform Dirichlet Distribution ---
    println!("\nUniform Dirichlet (α = [1, 1, 1]):");
    println!("-----------------------------------");

    // Create a uniform Dirichlet distribution (flat distribution on the simplex)
    let alpha_uniform = array![1.0, 1.0, 1.0];
    let uniform_dirichlet = multivariate::dirichlet(&alpha_uniform).unwrap();

    println!("For uniform Dirichlet, PDF is constant across the simplex.");

    // Evaluate PDF at different points on the simplex
    let points = [
        array![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        array![0.5, 0.25, 0.25],
        array![0.1, 0.3, 0.6],
    ];

    for (i, point) in points.iter().enumerate() {
        println!(
            "Point {}: {:?}  -  PDF: {:.6}",
            i + 1,
            point,
            uniform_dirichlet.pdf(point)
        );
    }

    // --- Example 2: Concentrated Dirichlet Distribution ---
    println!("\nConcentrated Dirichlet (α = [10, 10, 10]):");
    println!("-----------------------------------------");

    // Create a concentrated Dirichlet (favors the center of the simplex)
    let alpha_concentrated = array![10.0, 10.0, 10.0];
    let concentrated_dirichlet = multivariate::dirichlet(&alpha_concentrated).unwrap();

    println!("For concentrated Dirichlet, PDF is highest at the center.");

    for (i, point) in points.iter().enumerate() {
        println!(
            "Point {}: {:?}  -  PDF: {:.6}",
            i + 1,
            point,
            concentrated_dirichlet.pdf(point)
        );
    }

    // --- Example 3: Asymmetric Dirichlet Distribution ---
    println!("\nAsymmetric Dirichlet (α = [5, 2, 1]):");
    println!("-----------------------------------");

    // Create an asymmetric Dirichlet (favors the first component)
    let alpha_asymmetric = array![5.0, 2.0, 1.0];
    let asymmetric_dirichlet = multivariate::dirichlet(&alpha_asymmetric).unwrap();

    println!("For asymmetric Dirichlet, PDF is highest where x₁ is large.");

    let asymmetric_points = [
        array![0.6, 0.3, 0.1], // Favored by this distribution
        array![0.1, 0.6, 0.3], // Less favored
        array![0.1, 0.3, 0.6], // Least favored
    ];

    for (i, point) in asymmetric_points.iter().enumerate() {
        println!(
            "Point {}: {:?}  -  PDF: {:.6}",
            i + 1,
            point,
            asymmetric_dirichlet.pdf(point)
        );
    }

    // --- Example 4: Random Sampling ---
    println!("\nRandom Sampling from Dirichlet:");
    println!("------------------------------");

    let n_samples = 10;
    let samples = asymmetric_dirichlet.rvs(n_samples).unwrap();

    println!(
        "Generated {} samples from asymmetric Dirichlet(α = [5, 2, 1]):",
        n_samples
    );
    for (i, sample) in samples.iter().enumerate() {
        println!(
            "Sample {}: [{:.4}, {:.4}, {:.4}]",
            i + 1,
            sample[0],
            sample[1],
            sample[2]
        );
    }

    // --- Example 5: Mean of Dirichlet Distribution ---
    println!("\nEmpirical Mean vs. Theoretical Mean:");
    println!("----------------------------------");

    // Generate more samples to estimate mean
    let n_samples = 10000;
    let samples = asymmetric_dirichlet.rvs(n_samples).unwrap();

    // Calculate empirical mean
    let mut empirical_mean = Array1::zeros(3);
    for sample in &samples {
        empirical_mean += sample;
    }
    empirical_mean /= n_samples as f64;

    // Theoretical mean: E[X_i] = α_i / sum(α)
    let alpha_sum = alpha_asymmetric.sum();
    let theoretical_mean = alpha_asymmetric.mapv(|a| a / alpha_sum);

    println!(
        "Empirical mean:    [{:.4}, {:.4}, {:.4}]",
        empirical_mean[0], empirical_mean[1], empirical_mean[2]
    );
    println!(
        "Theoretical mean:  [{:.4}, {:.4}, {:.4}]",
        theoretical_mean[0], theoretical_mean[1], theoretical_mean[2]
    );

    // --- Example 6: Dirichlet distributions in practice ---
    println!("\nApplications of Dirichlet distributions:");
    println!("-------------------------------------");
    println!("1. Bayesian inference for categorical distributions");
    println!("2. Topic modeling (e.g., Latent Dirichlet Allocation)");
    println!("3. Modeling compositional data (e.g., mineral compositions)");
    println!("4. Mixture proportions in mixture models");
    println!("5. Color distributions in image analysis");
}
