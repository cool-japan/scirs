//! Example of separable least squares optimization
//!
//! This demonstrates the variable projection algorithm for problems
//! where the model is linear in some parameters and nonlinear in others.

use ndarray::{array, Array1, Array2};
use scirs2_optimize::least_squares::{separable_least_squares, LinearSolver, SeparableOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Separable Least Squares Example");
    println!("==============================");
    println!();

    // Example 1: Exponential decay with offset
    // Model: y = α₁ * exp(-β * t) + α₂
    // Linear parameters: α = [α₁, α₂]
    // Nonlinear parameter: β

    fn exponential_basis(t: &[f64], beta: &[f64]) -> Array2<f64> {
        let n = t.len();
        let mut phi = Array2::zeros((n, 2));

        for i in 0..n {
            phi[[i, 0]] = (-beta[0] * t[i]).exp(); // exp(-β*t)
            phi[[i, 1]] = 1.0; // constant term
        }
        phi
    }

    fn exponential_jacobian(t: &[f64], beta: &[f64]) -> Array2<f64> {
        let n = t.len();
        let mut dphi_dbeta = Array2::zeros((n * 2, 1));

        for i in 0..n {
            // d/dβ(exp(-β*t)) = -t * exp(-β*t)
            dphi_dbeta[[i, 0]] = -t[i] * (-beta[0] * t[i]).exp();
            // d/dβ(1) = 0
            dphi_dbeta[[n + i, 0]] = 0.0;
        }
        dphi_dbeta
    }

    // Generate synthetic data
    let t_data = array![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
    let true_alpha1 = 3.0;
    let true_alpha2 = 0.5;
    let true_beta = 0.8;

    let mut y_data = Array1::zeros(t_data.len());
    for i in 0..t_data.len() {
        y_data[i] = true_alpha1 * ((-true_beta * t_data[i]) as f64).exp() + true_alpha2;
        // Add small noise
        y_data[i] += 0.02 * (2.0 * i as f64 / 8.0 - 1.0);
    }

    println!("Example 1: Exponential decay with offset");
    println!("Model: y = α₁ * exp(-β * t) + α₂");
    println!(
        "True parameters: α₁ = {:.1}, α₂ = {:.1}, β = {:.1}",
        true_alpha1, true_alpha2, true_beta
    );
    println!();

    // Initial guess for nonlinear parameter
    let beta0 = array![0.5];

    let result = separable_least_squares(
        exponential_basis,
        exponential_jacobian,
        &t_data,
        &y_data,
        &beta0,
        None,
    )?;

    println!("Estimated parameters:");
    println!("  α₁ = {:.3}", result.linear_params[0]);
    println!("  α₂ = {:.3}", result.linear_params[1]);
    println!("  β = {:.3}", result.result.x[0]);
    println!("  Cost: {:.6}", result.result.fun);
    println!("  Iterations: {}", result.result.nit);
    println!();

    // Example 2: Sum of Gaussians
    // Model: y = Σ αᵢ * exp(-(t - μᵢ)² / (2σ²))
    // Linear parameters: α (amplitudes)
    // Nonlinear parameters: μ (centers), σ (width)

    fn gaussian_basis(t: &[f64], params: &[f64]) -> Array2<f64> {
        let n = t.len();
        let n_gaussians = (params.len() - 1) / 2;
        let mut phi = Array2::zeros((n, n_gaussians));

        let sigma = params[params.len() - 1];
        let sigma_sq = sigma * sigma;

        for i in 0..n {
            for j in 0..n_gaussians {
                let mu_j = params[j * 2];
                let diff = t[i] - mu_j;
                phi[[i, j]] = (-diff * diff / (2.0 * sigma_sq)).exp();
            }
        }
        phi
    }

    fn gaussian_jacobian(t: &[f64], params: &[f64]) -> Array2<f64> {
        let n = t.len();
        let n_gaussians = (params.len() - 1) / 2;
        let n_params = params.len();
        let mut dphi = Array2::zeros((n * n_gaussians, n_params));

        let sigma = params[params.len() - 1];
        let sigma_sq = sigma * sigma;
        let sigma_cu = sigma_sq * sigma;

        for i in 0..n {
            for j in 0..n_gaussians {
                let mu_j = params[j * 2];
                let diff = t[i] - mu_j;
                let gauss = (-diff * diff / (2.0 * sigma_sq)).exp();

                // Derivative w.r.t. μⱼ
                dphi[[j * n + i, j * 2]] = gauss * diff / sigma_sq;

                // Derivative w.r.t. σ
                dphi[[j * n + i, n_params - 1]] = gauss * diff * diff / sigma_cu;
            }
        }
        dphi
    }

    println!("Example 2: Sum of Gaussians");
    println!("Model: y = Σ αᵢ * exp(-(t - μᵢ)² / (2σ²))");

    // Generate data from two Gaussians
    let t_gauss = Array1::linspace(0.0, 10.0, 50);
    let true_alphas = array![2.0, 1.5];
    let true_mus = array![3.0, 7.0];
    let true_sigma = 0.8;

    let mut y_gauss = Array1::zeros(t_gauss.len());
    for i in 0..t_gauss.len() {
        for j in 0..2 {
            let diff = t_gauss[i] - true_mus[j];
            y_gauss[i] +=
                true_alphas[j] * ((-diff * diff / (2.0 * true_sigma * true_sigma)) as f64).exp();
        }
        // Add small noise
        y_gauss[i] += 0.05 * (i as f64 / 25.0 - 1.0);
    }

    println!("True parameters:");
    println!("  α = {:?}", true_alphas);
    println!("  μ = {:?}", true_mus);
    println!("  σ = {:.1}", true_sigma);
    println!();

    // Initial guess: μ values slightly off, σ close
    let params0 = array![2.5, 0.0, 6.5, 0.0, 1.0]; // [μ₁, 0, μ₂, 0, σ]

    let mut options = SeparableOptions::default();
    options.max_iter = 100;
    options.linear_solver = LinearSolver::QR;

    let result2 = separable_least_squares(
        gaussian_basis,
        gaussian_jacobian,
        &t_gauss,
        &y_gauss,
        &params0,
        Some(options),
    )?;

    println!("Estimated parameters:");
    println!("  α = {:?}", result2.linear_params);
    println!(
        "  μ = [{:.3}, {:.3}]",
        result2.result.x[0], result2.result.x[2]
    );
    println!("  σ = {:.3}", result2.result.x[4]);
    println!("  Cost: {:.6}", result2.result.fun);
    println!("  Iterations: {}", result2.result.nit);
    println!();

    // Example 3: Comparison with standard least squares
    println!("Example 3: Benefits of separable approach");
    println!("---------------------------------------");

    // For the exponential model, compare convergence speed
    let beta_values = vec![0.2, 0.5, 1.0, 1.5, 2.0];

    println!("Starting from different initial guesses:");
    println!("β₀     | Iterations | Final cost");
    println!("-------|------------|------------");

    for beta_init in beta_values {
        let result = separable_least_squares(
            exponential_basis,
            exponential_jacobian,
            &t_data,
            &y_data,
            &array![beta_init],
            None,
        )?;

        println!(
            "{:.1}    | {:10} | {:.8}",
            beta_init, result.result.nit, result.result.fun
        );
    }

    println!();
    println!("Key advantages of separable least squares:");
    println!("1. Reduces dimension of nonlinear optimization");
    println!("2. Often faster convergence");
    println!("3. Better conditioning");
    println!("4. Linear parameters determined optimally at each step");

    Ok(())
}
