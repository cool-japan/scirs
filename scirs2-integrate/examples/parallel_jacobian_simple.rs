//! Simple example demonstrating parallel Jacobian computation
//!
//! This example shows a minimal implementation of parallel Jacobian computation
//! without depending on other parts of the codebase. It compares the performance
//! of serial vs parallel Jacobian computation for a large ODE system.

use ndarray::{Array1, Array2, ArrayView1};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;
use std::time::Instant;

/// Compute Jacobian matrix using standard (serial) finite differences
fn finite_difference_jacobian<F, Func>(
    f: &Func,
    t: F,
    y: &Array1<F>,
    f_current: &Array1<F>,
    perturbation_scale: F,
) -> Array2<F>
where
    F: Float + FromPrimitive + Debug,
    Func: Fn(F, ArrayView1<F>) -> Array1<F>,
{
    let n_dim = y.len();
    let mut jacobian = Array2::<F>::zeros((n_dim, n_dim));

    // Calculate base perturbation size
    let eps_base = F::from_f64(1e-8).unwrap() * perturbation_scale;

    // Compute columns serially
    for j in 0..n_dim {
        // Scale perturbation by variable magnitude
        let eps = eps_base * (F::one() + y[j].abs()).max(F::one());

        // Perturb the j-th component
        let mut y_perturbed = y.clone();
        y_perturbed[j] = y_perturbed[j] + eps;

        // Evaluate function at perturbed point
        let f_perturbed = f(t, y_perturbed.view());

        // Calculate the j-th column using finite differences
        for i in 0..n_dim {
            jacobian[[i, j]] = (f_perturbed[i] - f_current[i]) / eps;
        }
    }

    jacobian
}

/// Compute Jacobian matrix using parallel finite differences (if parallel_jacobian feature is enabled)
fn parallel_finite_difference_jacobian<F, Func>(
    f: &Func,
    t: F,
    y: &Array1<F>,
    f_current: &Array1<F>,
    perturbation_scale: F,
) -> Array2<F>
where
    F: Float + FromPrimitive + Debug + Send + Sync,
    Func: Fn(F, ArrayView1<F>) -> Array1<F> + Sync,
{
    #[allow(unused_variables)]
    let n_dim = y.len();

    #[cfg(feature = "parallel_jacobian")]
    {
        use rayon::prelude::*;

        // Calculate base perturbation size
        let eps_base = F::from_f64(1e-8).unwrap() * perturbation_scale;

        // Compute columns in parallel using rayon
        let columns: Vec<(usize, Array1<F>)> = (0..n_dim)
            .into_par_iter()
            .map(|j| {
                // Scale perturbation by variable magnitude
                let eps = eps_base * (F::one() + y[j].abs()).max(F::one());

                // Perturb the j-th component
                let mut y_perturbed = y.clone();
                y_perturbed[j] = y_perturbed[j] + eps;

                // Evaluate function at perturbed point
                let f_perturbed = f(t, y_perturbed.view());

                // Calculate the j-th column using finite differences
                let mut column = Array1::<F>::zeros(n_dim);
                for i in 0..n_dim {
                    column[i] = (f_perturbed[i] - f_current[i]) / eps;
                }

                (j, column)
            })
            .collect();

        // Assemble the Jacobian from columns
        let mut jacobian = Array2::<F>::zeros((n_dim, n_dim));
        for (j, column) in columns {
            for i in 0..n_dim {
                jacobian[[i, j]] = column[i];
            }
        }
        jacobian
    }

    #[cfg(not(feature = "parallel_jacobian"))]
    {
        // Fall back to serial implementation
        finite_difference_jacobian(f, t, y, f_current, perturbation_scale)
    }
}

// A large ODE system for demonstration
fn large_ode_system(_t: f64, y: ArrayView1<f64>) -> Array1<f64> {
    let n = y.len();
    let mut result = Array1::zeros(n);

    // Create a system with some nonlinear terms
    for i in 0..n {
        // Diagonal term
        result[i] += -0.1 * y[i] * y[i] * y[i];

        // Coupling with neighbors
        let prev = if i > 0 { i - 1 } else { n - 1 };
        let next = if i < n - 1 { i + 1 } else { 0 };

        result[i] += 0.05 * (y[prev] - 2.0 * y[i] + y[next]);
    }

    result
}

fn main() {
    println!("Simple Parallel Jacobian Example");
    println!("================================");

    // Check if parallel_jacobian feature is enabled
    println!(
        "Parallel Jacobian Feature: {}",
        if cfg!(feature = "parallel_jacobian") {
            "Enabled"
        } else {
            "Disabled"
        }
    );

    // System size for performance testing
    let system_sizes = [10, 100, 500];

    for &n in &system_sizes {
        println!("\nTesting with system size: {}", n);

        // Create a random initial state
        let mut y = Array1::zeros(n);
        for i in 0..n {
            y[i] = (i as f64).sin();
        }

        let t = 0.0;
        let f_y = large_ode_system(t, y.view());

        // Serial computation
        let serial_start = Instant::now();
        let _serial_jac = finite_difference_jacobian(&large_ode_system, t, &y, &f_y, 1.0);
        let serial_time = serial_start.elapsed();
        println!("  Serial computation: {:?}", serial_time);

        // Parallel computation (uses serial if feature is disabled)
        let parallel_start = Instant::now();
        let _parallel_jac =
            parallel_finite_difference_jacobian(&large_ode_system, t, &y, &f_y, 1.0);
        let parallel_time = parallel_start.elapsed();
        println!("  Parallel computation: {:?}", parallel_time);

        // Calculate speedup
        let speedup = serial_time.as_secs_f64() / parallel_time.as_secs_f64();
        println!("  Speedup: {:.2}x", speedup);
    }

    println!("\nNote: For best performance, compile with --release flag");
    println!("and enable the parallel_jacobian feature.");
}
