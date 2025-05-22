use ndarray::{array, Array1, ArrayView1};
use scirs2_optimize::prelude::*;
use scirs2_optimize::unconstrained::Method;
use scirs2_sparse::{csr_array::CsrArray, sparray::SparseArray};
use std::time::Instant;

/// Simple example to demonstrate sparse numerical differentiation
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Sparse Numerical Differentiation Examples");
    println!("=========================================\n");

    // Example 1: Simple Jacobian computation
    simple_jacobian_example()?;

    // Example 2: Sparse vs Dense Jacobian performance
    sparse_vs_dense_jacobian()?;

    // Example 3: Sparse Hessian computation with different methods
    sparse_hessian_methods_comparison()?;

    // Example 4: Large sparse optimization problem
    large_sparse_optimization()?;

    Ok(())
}

/// Simple example of sparse Jacobian computation
fn simple_jacobian_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("\nExample 1: Simple Jacobian Computation");
    println!("-------------------------------------");

    // Define a function with a sparse Jacobian (diagonal + some off-diagonal elements)
    fn test_function(x: &ArrayView1<f64>) -> Array1<f64> {
        let n = x.len();
        let mut result = Array1::zeros(n);

        for i in 0..n {
            // Diagonal element
            result[i] += 2.0 * x[i];

            // Some off-diagonal elements
            if i > 0 {
                result[i] += 0.5 * x[i - 1]; // Lower diagonal
            }
            if i < n - 1 {
                result[i] += 0.3 * x[i + 1]; // Upper diagonal
            }
        }

        result
    }

    // Create a test point
    let n = 5;
    let x = Array1::from_vec((0..n).map(|i| i as f64).collect());

    println!("Computing Jacobian of a function with sparse structure");
    println!("at point x = {:?}", x);

    // Create a sparsity pattern (tridiagonal)
    let mut row_indices = Vec::new();
    let mut col_indices = Vec::new();
    let mut data = Vec::new();

    for i in 0..n {
        // Diagonal
        row_indices.push(i);
        col_indices.push(i);
        data.push(1.0);

        // Lower diagonal
        if i > 0 {
            row_indices.push(i);
            col_indices.push(i - 1);
            data.push(1.0);
        }

        // Upper diagonal
        if i < n - 1 {
            row_indices.push(i);
            col_indices.push(i + 1);
            data.push(1.0);
        }
    }

    // Create sparsity pattern
    let sparsity = CsrArray::from_triplets(&row_indices, &col_indices, &data, (n, n), false)?;

    // Compute Jacobian with different methods
    let methods = ["2-point", "3-point", "cs"];

    for method in methods {
        let mut options = SparseFiniteDiffOptions::default();
        options.method = method.to_string();

        let start = Instant::now();
        let jac = sparse_jacobian(
            test_function,
            &x.view(),
            None,
            Some(&sparsity),
            Some(options),
        )?;
        let duration = start.elapsed();

        println!("\nJacobian using {} method:", method);
        let dense_jac = jac.to_array();
        println!("{:?}", dense_jac);
        println!("Computation time: {:?}", duration);
    }

    Ok(())
}

/// Compare performance of sparse vs dense Jacobian computation
fn sparse_vs_dense_jacobian() -> Result<(), Box<dyn std::error::Error>> {
    println!("\nExample 2: Sparse vs Dense Jacobian Performance");
    println!("---------------------------------------------");

    // Define a function with a sparse Jacobian (tridiagonal)
    fn tridiagonal_function(x: &ArrayView1<f64>) -> Array1<f64> {
        let n = x.len();
        let mut result = Array1::zeros(n);

        for i in 0..n {
            // Diagonal
            result[i] += 2.0 * x[i];

            // Off-diagonal elements
            if i > 0 {
                result[i] += x[i - 1]; // Lower diagonal
            }
            if i < n - 1 {
                result[i] += x[i + 1]; // Upper diagonal
            }
        }

        result
    }

    // Test with different problem sizes
    let sizes = [10, 50, 100, 500];

    println!("Comparing sparse vs dense Jacobian computation for different problem sizes");
    println!("Size\tDense (ms)\tSparse (ms)\tSpeedup");
    println!("----\t----------\t-----------\t-------");

    for &n in &sizes {
        // Create a test point
        let x = Array1::from_vec((0..n).map(|i| i as f64).collect());

        // Create a sparsity pattern (tridiagonal)
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();
        let mut data = Vec::new();

        for i in 0..n {
            // Diagonal
            row_indices.push(i);
            col_indices.push(i);
            data.push(1.0);

            // Lower diagonal
            if i > 0 {
                row_indices.push(i);
                col_indices.push(i - 1);
                data.push(1.0);
            }

            // Upper diagonal
            if i < n - 1 {
                row_indices.push(i);
                col_indices.push(i + 1);
                data.push(1.0);
            }
        }

        // Create sparsity pattern
        let sparsity = CsrArray::from_triplets(&row_indices, &col_indices, &data, (n, n), false)?;

        // Compute dense Jacobian (no sparsity pattern)
        let start = Instant::now();
        let _ = sparse_jacobian(tridiagonal_function, &x.view(), None, None, None)?;
        let dense_time = start.elapsed().as_millis();

        // Compute sparse Jacobian (with sparsity pattern)
        let start = Instant::now();
        let _ = sparse_jacobian(tridiagonal_function, &x.view(), None, Some(&sparsity), None)?;
        let sparse_time = start.elapsed().as_millis();

        // Compute speedup
        let speedup = if sparse_time > 0 {
            dense_time as f64 / sparse_time as f64
        } else {
            f64::INFINITY
        };

        println!(
            "{}\t{}\t\t{}\t\t{:.2}x",
            n, dense_time, sparse_time, speedup
        );
    }

    Ok(())
}

/// Compare different methods for sparse Hessian computation
fn sparse_hessian_methods_comparison() -> Result<(), Box<dyn std::error::Error>> {
    println!("\nExample 3: Sparse Hessian Methods Comparison");
    println!("-------------------------------------------");

    // Define a function with a sparse Hessian (Rosenbrock function)
    fn rosenbrock(x: &ArrayView1<f64>) -> f64 {
        let n = x.len();
        let mut sum = 0.0;

        for i in 0..n - 1 {
            sum += 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2);
        }

        sum
    }

    // Create a test point
    let x = array![1.2, 1.2, 1.2, 1.2, 1.2];
    let n = x.len();

    println!("Computing Hessian of Rosenbrock function");
    println!("at point x = {:?}", x);

    // Create a sparsity pattern (banded structure of Rosenbrock)
    let mut row_indices = Vec::new();
    let mut col_indices = Vec::new();
    let mut data = Vec::new();

    for i in 0..n {
        // Diagonal element
        row_indices.push(i);
        col_indices.push(i);
        data.push(1.0);

        // Off-diagonal elements
        if i < n - 1 {
            // Each variable interacts with the next one
            row_indices.push(i);
            col_indices.push(i + 1);
            data.push(1.0);

            row_indices.push(i + 1);
            col_indices.push(i);
            data.push(1.0);
        }
    }

    // Create sparsity pattern
    let sparsity = CsrArray::from_triplets(&row_indices, &col_indices, &data, (n, n), false)?;

    // Compute Hessian with different methods
    let methods = ["2-point", "3-point", "cs"];

    for method in methods {
        let mut options = SparseFiniteDiffOptions::default();
        options.method = method.to_string();

        let start = Instant::now();
        let hess = sparse_hessian(
            rosenbrock,
            None::<fn(&ArrayView1<f64>) -> Array1<f64>>,
            &x.view(),
            None,
            None,
            Some(&sparsity),
            Some(options),
        )?;
        let duration = start.elapsed();

        println!("\nHessian using {} method:", method);
        // Just print diagonal and first off-diagonal to save space
        let dense_hess = hess.to_array();
        println!(
            "Diagonal elements: {:?}",
            (0..n).map(|i| dense_hess[[i, i]]).collect::<Vec<_>>()
        );
        println!("Computation time: {:?}", duration);

        // Verify positive definiteness (should be positive definite away from minimum)
        let mut is_pd = true;
        for i in 0..n {
            if dense_hess[[i, i]] <= 0.0 {
                is_pd = false;
                break;
            }
        }
        println!("Is positive definite: {}", is_pd);
    }

    Ok(())
}

/// Example of using sparse Jacobian/Hessian in a large optimization problem
fn large_sparse_optimization() -> Result<(), Box<dyn std::error::Error>> {
    println!("\nExample 4: Large Sparse Optimization Problem");
    println!("-------------------------------------------");

    // Diffusion PDE discretized on a 1D grid
    // -d²u/dx² + u = f(x)
    // Using finite difference on a grid gives a tridiagonal system
    fn diffusion_residual(u: &ArrayView1<f64>) -> Array1<f64> {
        let n = u.len();
        let h = 1.0 / (n as f64 + 1.0); // Grid spacing
        let h2 = h * h;

        let mut residual = Array1::zeros(n);

        for i in 0..n {
            // Source term: f(x) = sin(π*x)
            let x = (i + 1) as f64 * h;
            let source = (std::f64::consts::PI * x).sin();

            // -d²u/dx² using central difference
            let d2u = if i == 0 {
                (2.0 * u[i] - u[i + 1]) / h2
            } else if i == n - 1 {
                (2.0 * u[i] - u[i - 1]) / h2
            } else {
                (2.0 * u[i] - u[i - 1] - u[i + 1]) / h2
            };

            // Residual: -d²u/dx² + u - f
            residual[i] = -d2u + u[i] - source;
        }

        residual
    }

    // Function to minimize (sum of squared residuals)
    fn objective(u: &ArrayView1<f64>) -> f64 {
        let residual = diffusion_residual(u);
        residual.iter().map(|&r| r * r).sum()
    }

    // Grid size
    let n = 100;
    println!("Solving discretized diffusion equation on {} point grid", n);

    // Initial guess (all zeros)
    let initial_guess = Array1::zeros(n);

    // Create a sparsity pattern for the Jacobian (tridiagonal)
    let mut row_indices = Vec::new();
    let mut col_indices = Vec::new();
    let mut data = Vec::new();

    for i in 0..n {
        // Diagonal
        row_indices.push(i);
        col_indices.push(i);
        data.push(1.0);

        // Lower diagonal
        if i > 0 {
            row_indices.push(i);
            col_indices.push(i - 1);
            data.push(1.0);
        }

        // Upper diagonal
        if i < n - 1 {
            row_indices.push(i);
            col_indices.push(i + 1);
            data.push(1.0);
        }
    }

    // Create Jacobian sparsity pattern
    let jac_sparsity = CsrArray::from_triplets(&row_indices, &col_indices, &data, (n, n), false)?;

    // Create Hessian sparsity pattern (also tridiagonal-like)
    let hess_sparsity = CsrArray::from_triplets(&row_indices, &col_indices, &data, (n, n), false)?;

    // Set up two optimization methods to compare
    let methods = [
        ("without sparsity", None, None),
        (
            "with sparsity",
            Some(jac_sparsity.clone()),
            Some(hess_sparsity.clone()),
        ),
    ];

    for (method_name, _jac_pattern, _hess_pattern) in methods {
        println!("\nOptimizing {} pattern:", method_name);

        // Create sparse finite difference options
        let _sparse_diff_options = SparseFiniteDiffOptions {
            method: "3-point".to_string(),
            parallel: Some(ParallelOptions::default()),
            ..Default::default()
        };

        // Create custom options
        let mut options = Options::default();
        options.gtol = 1e-6;
        options.xtol = 1e-6;
        options.max_iter = 100;

        // Set up numerical differentiation
        options.finite_diff = true; // Use finite differences for gradient calculation
        options.eps = 1e-6; // Set finite difference step size

        // Optimize
        let start = Instant::now();
        // Convert array to vec for minimize function
        let initial_vec: Vec<f64> = initial_guess.to_vec();

        // Our objective function directly works with ArrayView1<f64>
        let result = minimize(objective, &initial_vec, Method::BFGS, Some(options))?;
        let duration = start.elapsed();

        println!("Optimization result:");
        println!("  Converged: {}", result.success);
        println!("  Iterations: {}", result.iterations);
        println!("  Function value: {:.6e}", result.fun);
        println!("  Time: {:?}", duration);
    }

    Ok(())
}
