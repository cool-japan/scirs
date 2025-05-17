use ndarray::{Array1, ArrayView1};
use rand::prelude::*;
use rand_distr::{Distribution, Normal};
use scirs2_integrate::monte_carlo::{importance_sampling, monte_carlo, MonteCarloOptions};
use std::f64::consts::PI;

fn main() {
    println!("Monte Carlo Integration Examples\n");

    // Example 1: Basic 1D integration
    println!("Example 1: Integrate x^2 from 0 to 1 using basic Monte Carlo");
    println!("Exact value: {}", 1.0 / 3.0);

    // Configure Monte Carlo integration with reproducible results
    let options = MonteCarloOptions {
        n_samples: 100_000,
        seed: Some(42), // For reproducibility
        ..Default::default()
    };

    let result = monte_carlo(|x| x[0] * x[0], &[(0.0, 1.0)], Some(options)).unwrap();

    println!("Monte Carlo result: {:.8}", result.value);
    println!("Standard error: {:.8}", result.std_error);
    println!(
        "Absolute error: {:.8}",
        (result.value - 1.0_f64 / 3.0).abs()
    );
    println!("Number of evaluations: {}", result.n_evals);
    println!();

    // Example 2: Higher-dimensional integration
    println!("Example 2: Integrate x^2 + y^2 + z^2 over unit cube [0,1]^3");
    println!("Exact value: {}", 1.0); // 1/3 + 1/3 + 1/3 = 1

    let options = MonteCarloOptions {
        n_samples: 100_000,
        seed: Some(42),
        ..Default::default()
    };

    let result = monte_carlo(
        |x| x[0] * x[0] + x[1] * x[1] + x[2] * x[2],
        &[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
        Some(options),
    )
    .unwrap();

    println!("Monte Carlo result: {:.8}", result.value);
    println!("Standard error: {:.8}", result.std_error);
    println!("Absolute error: {:.8}", (result.value - 1.0_f64).abs());
    println!("Number of evaluations: {}", result.n_evals);
    println!();

    // Example 3: Comparing with and without antithetic sampling
    println!("Example 3: Variance reduction with antithetic sampling");
    println!("Integrating e^(-x^2) from 0 to 1");

    // Approximate reference value
    let reference: f64 = 0.7468241328124271;
    println!("Reference value: {:.15}", reference);

    // Without antithetic sampling
    let options_standard = MonteCarloOptions {
        n_samples: 100_000,
        seed: Some(42),
        use_antithetic: false,
        ..Default::default()
    };

    let result_standard = monte_carlo(
        |x| {
            let val: f64 = -x[0] * x[0];
            val.exp()
        },
        &[(0.0, 1.0)],
        Some(options_standard),
    )
    .unwrap();

    // With antithetic sampling
    let options_antithetic = MonteCarloOptions {
        n_samples: 100_000, // Same total number of samples
        seed: Some(42),     // Same seed for fair comparison
        use_antithetic: true,
        ..Default::default()
    };

    let result_antithetic = monte_carlo(
        |x| {
            let val: f64 = -x[0] * x[0];
            val.exp()
        },
        &[(0.0, 1.0)],
        Some(options_antithetic),
    )
    .unwrap();

    println!("Without antithetic sampling:");
    println!("  Result: {:.15}", result_standard.value);
    println!("  Standard error: {:.10}", result_standard.std_error);
    println!(
        "  Absolute error: {:.10}",
        (result_standard.value as f64 - reference).abs()
    );

    println!("With antithetic sampling:");
    println!("  Result: {:.15}", result_antithetic.value);
    println!("  Standard error: {:.10}", result_antithetic.std_error);
    println!(
        "  Absolute error: {:.10}",
        (result_antithetic.value as f64 - reference).abs()
    );
    println!();

    // Example 4: Importance sampling for an integral with a peak
    println!("Example 4: Importance sampling for integral with a peak");
    println!("Integrating 1/(0.1 + (x-0.5)^2) from 0 to 1");

    // This function has a peak at x=0.5
    let peak_function = |x: ArrayView1<f64>| 1.0 / (0.1 + (x[0] - 0.5).powi(2));

    // Approximate reference value
    let reference = 3.139525976351711;
    println!("Reference value: {:.15}", reference);

    // Standard Monte Carlo
    let options_standard = MonteCarloOptions {
        n_samples: 100_000,
        seed: Some(42),
        ..Default::default()
    };

    let result_standard =
        monte_carlo(peak_function, &[(0.0, 1.0)], Some(options_standard)).unwrap();

    // Importance sampling using a normal distribution centered at the peak
    // Create a sampler function that generates samples from a normal distribution
    let normal_sampler = |rng: &mut StdRng, dims: usize| {
        let mut point = Array1::zeros(dims);
        let normal = Normal::new(0.5, 0.2).unwrap(); // centered at peak x=0.5

        for i in 0..dims {
            // Sample and clamp to [0, 1]
            let mut x: f64 = normal.sample(rng);
            x = x.clamp(0.0, 1.0);
            point[i] = x;
        }
        point
    };

    // PDF of the truncated normal distribution (simplified approximation)
    let normal_pdf = |x: ArrayView1<f64>| {
        let mut pdf = 1.0;
        for &xi in x.iter() {
            let z = (xi - 0.5) / 0.2;
            // Prevent numerical underflow by setting a minimum
            let density = (-0.5 * z * z).exp() / (0.2 * (2.0 * PI).sqrt());
            pdf *= density.max(1e-10); // Prevent zero density
        }
        pdf
    };

    let options_importance = MonteCarloOptions {
        n_samples: 100_000,
        seed: Some(42),
        ..Default::default()
    };

    let result_importance = importance_sampling(
        peak_function,
        normal_pdf,
        normal_sampler,
        &[(0.0, 1.0)],
        Some(options_importance),
    )
    .unwrap();

    println!("Standard Monte Carlo:");
    println!("  Result: {:.15}", result_standard.value);
    println!("  Standard error: {:.10}", result_standard.std_error);
    println!(
        "  Absolute error: {:.10}",
        (result_standard.value as f64 - reference).abs()
    );

    println!("With importance sampling:");
    println!("  Result: {:.15}", result_importance.value);
    println!("  Standard error: {:.10}", result_importance.std_error);
    println!(
        "  Absolute error: {:.10}",
        (result_importance.value as f64 - reference).abs()
    );
    println!();

    // Example 5: High-dimensional integration
    println!("Example 5: High-dimensional integration (5D hypersphere volume)");

    // The volume of a 5D hypersphere with radius 1 is pi^(5/2) / (5/2 * Gamma(5/2))
    // This simplifies to pi^(5/2) / (15/4), which is approximately 5.2638
    let exact_volume: f64 = PI.powf(2.5) * 4.0 / 15.0;
    println!("Exact volume of 5D unit hypersphere: {:.15}", exact_volume);

    // Estimate the volume using Monte Carlo integration over [-1,1]^5
    // We'll use the indicator function: 1 if inside the sphere, 0 if outside
    let options = MonteCarloOptions {
        n_samples: 1_000_000, // More samples for higher dimensions
        seed: Some(42),
        ..Default::default()
    };

    let result = monte_carlo(
        |x| {
            // Calculate squared distance from origin
            let r_squared = x.iter().map(|&xi| xi * xi).sum::<f64>();
            // 1 if inside unit sphere, 0 if outside
            if r_squared <= 1.0 {
                1.0
            } else {
                0.0
            }
        },
        &[
            (-1.0, 1.0),
            (-1.0, 1.0),
            (-1.0, 1.0),
            (-1.0, 1.0),
            (-1.0, 1.0),
        ],
        Some(options),
    )
    .unwrap();

    // The Monte Carlo integration gives us the ratio of points inside the sphere
    // To get the actual volume, we multiply by the volume of the 5D hypercube (2^5=32)
    let estimated_volume = result.value * 32.0;

    println!(
        "Monte Carlo result (ratio in hypercube): {:.8}",
        result.value
    );
    println!("Estimated hypersphere volume: {:.8}", estimated_volume);
    println!("Standard error (scaled): {:.8}", result.std_error * 32.0);
    println!(
        "Absolute error: {:.8}",
        (estimated_volume as f64 - exact_volume).abs()
    );
    println!("Number of evaluations: {}", result.n_evals);
}
