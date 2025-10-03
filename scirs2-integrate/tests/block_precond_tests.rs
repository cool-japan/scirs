use scirs2_core::ndarray::{Array1, Array2};
use scirs2_integrate::dae::{create_block_ilu_preconditioner, create_block_jacobi_preconditioner};
use std::time::Instant;

// Test helper: Create a test matrix with block structure
#[allow(dead_code)]
fn create_test_matrix(
    n_x: usize,
    n_y: usize,
) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
    // Create the blocks for a semi-explicit DAE system
    let mut f_x = Array2::zeros((n_x, n_x));
    let mut f_y = Array2::zeros((n_x, n_y));
    let mut g_x = Array2::zeros((n_y, n_x));
    let mut g_y = Array2::zeros((n_y, n_y));

    // Fill the blocks with some representative values
    // f_x: tridiagonal with dominant diagonal
    for i in 0..n_x {
        f_x[[i, i]] = 4.0;
        if i > 0 {
            f_x[[i, i - 1]] = -1.0;
        }
        if i < n_x - 1 {
            f_x[[i, i + 1]] = -1.0;
        }
    }

    // f_y: dense block
    for i in 0..n_x {
        for j in 0..n_y {
            f_y[[i, j]] = 0.5 / (1.0 + (i as f64 - j as f64).abs());
        }
    }

    // g_x: dense block
    for i in 0..n_y {
        for j in 0..n_x {
            g_x[[i, j]] = 0.7 / (1.0 + (i as f64 - j as f64).abs());
        }
    }

    // g_y: dominant diagonal
    for i in 0..n_y {
        g_y[[i, i]] = 3.0;
        if i > 0 {
            g_y[[i, i - 1]] = -0.5;
        }
        if i < n_y - 1 {
            g_y[[i, i + 1]] = -0.5;
        }
    }

    (f_x, f_y, g_x, g_y)
}

// Test helper: Create a right-hand side vector
#[allow(dead_code)]
fn create_rhs(n_x: usize, ny: usize) -> Array1<f64> {
    let mut rhs = Array1::zeros(n_x + ny);

    // Fill with representative values
    for i in 0..n_x {
        rhs[i] = (i as f64 + 1.0).sin();
    }
    for i in 0..ny {
        rhs[n_x + i] = (i as f64 + 1.0).cos();
    }

    rhs
}

// Test helper: Apply the full Jacobian matrix to a vector
#[allow(dead_code)]
fn matrix_vector_product(
    f_x: &Array2<f64>,
    f_y: &Array2<f64>,
    g_x: &Array2<f64>,
    g_y: &Array2<f64>,
    h: f64,
    beta: f64,
    v: &Array1<f64>,
) -> Array1<f64> {
    let n_x = f_x.shape()[0];
    let n_y = g_y.shape()[0];
    let n_total = n_x + n_y;

    let mut result = Array1::zeros(n_total);

    // Extract the _x and _y components of the input vector
    let v_x = v.slice(scirs2_core::ndarray::s![0..n_x]).to_owned();
    let v_y = v.slice(scirs2_core::ndarray::s![n_x..]).to_owned();

    // Apply the Jacobian blocks
    // (I - h * β * ∂f/∂_x) * v_x
    for i in 0..n_x {
        result[i] = v_x[i]; // Identity part
        for j in 0..n_x {
            result[i] -= h * beta * f_x[[i, j]] * v_x[j];
        }
    }

    // (-h * β * ∂f/∂_y) * v_y
    for i in 0..n_x {
        for j in 0..n_y {
            result[i] -= h * beta * f_y[[i, j]] * v_y[j];
        }
    }

    // (∂g/∂_x) * v_x
    for i in 0..n_y {
        for j in 0..n_x {
            result[n_x + i] += g_x[[i, j]] * v_x[j];
        }
    }

    // (∂g/∂_y) * v_y
    for i in 0..n_y {
        for j in 0..n_y {
            result[n_x + i] += g_y[[i, j]] * v_y[j];
        }
    }

    result
}

// Test helper: Simple GMRES solver to check preconditioner effectiveness
type PreconditionerFn = Box<dyn Fn(&Array1<f64>) -> Array1<f64>>;

#[allow(dead_code)]
fn simple_gmres<F>(
    matvec: F,
    b: &Array1<f64>,
    preconditioner: Option<PreconditionerFn>,
    tol: f64,
    max_iter: usize,
) -> (Array1<f64>, usize)
where
    F: Fn(&Array1<f64>) -> Array1<f64>,
{
    let n = b.len();
    let mut x = Array1::<f64>::zeros(n);

    // Apply preconditioner to b if provided
    let b_precond = match &preconditioner {
        Some(precond) => precond(b),
        None => b.clone(),
    };

    // Compute initial residual: r = P⁻¹(b - Ax)
    let r0 = if x.iter().all(|&v| v == 0.0) {
        // If x is zero, r = P⁻¹b
        b_precond.clone()
    } else {
        // Otherwise, r = P⁻¹(b - Ax)
        let ax = matvec(&x);
        let residual = b - &ax;
        match &preconditioner {
            Some(precond) => precond(&residual),
            None => residual,
        }
    };

    let r0_norm = r0.iter().fold(0.0, |acc, &v| acc + v * v).sqrt();

    // Initial check for convergence or zero RHS
    if r0_norm <= tol {
        return (x, 0); // Already converged
    }

    // Allocate Krylov subspace
    let mut v = Vec::with_capacity(max_iter + 1);
    v.push(r0 / r0_norm);

    // Allocate Hessenberg matrix
    let mut h = vec![vec![0.0; max_iter]; max_iter + 1];

    // Allocate solution update
    let mut y = vec![0.0; max_iter];

    // Allocate rotation coefficients
    let mut cs = vec![0.0; max_iter];
    let mut sn = vec![0.0; max_iter];

    // Allocate rhs vector for least squares
    let mut g = vec![0.0; max_iter + 1];
    g[0] = r0_norm;

    #[allow(unused_assignments)]
    let mut residual_norm = 0.0;
    let mut iters = 0;

    // Main GMRES iteration
    for j in 0..max_iter {
        // Apply matrix and preconditioner
        let mut w = matvec(&v[j]);
        if let Some(precond) = &preconditioner {
            w = precond(&w);
        }

        // Modified Gram-Schmidt orthogonalization
        for i in 0..=j {
            h[i][j] = v[i].iter().zip(w.iter()).map(|(&a, &b)| a * b).sum();
            for k in 0..n {
                w[k] -= h[i][j] * v[i][k];
            }
        }

        // Compute the norm of the new vector
        let w_norm = w.iter().fold(0.0, |acc, &val| acc + val * val).sqrt();
        h[j + 1][j] = w_norm;

        // Check for breakdown
        if w_norm < 1e-14 {
            // Early convergence
            iters = j + 1;
            break;
        }

        // Add new orthonormal vector to Krylov subspace
        v.push(w / w_norm);

        // Apply Givens rotations to Hessenberg matrix
        for i in 0..j {
            let temp = h[i][j];
            h[i][j] = cs[i] * temp + sn[i] * h[i + 1][j];
            h[i + 1][j] = -sn[i] * temp + cs[i] * h[i + 1][j];
        }

        // Compute new Givens rotation
        let beta = (h[j][j] * h[j][j] + h[j + 1][j] * h[j + 1][j]).sqrt();
        if beta < 1e-14 {
            cs[j] = 1.0;
            sn[j] = 0.0;
        } else {
            cs[j] = h[j][j] / beta;
            sn[j] = h[j + 1][j] / beta;
        }

        // Apply rotation to last column
        h[j][j] = cs[j] * h[j][j] + sn[j] * h[j + 1][j];
        h[j + 1][j] = 0.0;

        // Apply rotation to rhs vector
        let temp = g[j];
        g[j] = cs[j] * temp;
        g[j + 1] = -sn[j] * temp;

        // Get residual norm
        residual_norm = g[j + 1].abs();

        // Check for convergence
        if residual_norm <= tol {
            iters = j + 1;
            break;
        }

        iters = j + 1;
    }

    // Solve triangular system to get solution update
    for j in (0..iters).rev() {
        let mut sum = g[j];
        #[allow(clippy::needless_range_loop)]
        for k in j + 1..iters {
            sum -= h[j][k] * y[k];
        }
        y[j] = sum / h[j][j];
    }

    // Apply solution update
    for j in 0..iters {
        for i in 0..n {
            x[i] += y[j] * v[j][i];
        }
    }

    (x, iters)
}

#[test]
#[allow(dead_code)]
fn test_block_ilu_preconditioner() {
    // Small problem to verify correctness
    let n_x = 5;
    let n_y = 3;
    let n_total = n_x + n_y;

    // Create the test matrix blocks
    let (f_x, f_y, g_x, g_y) = create_test_matrix(n_x, n_y);

    // Create the right-hand side
    let b = create_rhs(n_x, n_y);

    // Step size and BDF coefficient
    let h = 0.1;
    let beta = 0.5;

    // Define the matrix-vector product function
    let matvec = |v: &Array1<f64>| matrix_vector_product(&f_x, &f_y, &g_x, &g_y, h, beta, v);

    // Create the block ILU preconditioner
    let precond = create_block_ilu_preconditioner(&f_x, &f_y, &g_x, &g_y, h, beta);
    let precond_boxed = Box::new(precond);

    // Reference solution: solve without preconditioning
    let (x_ref, iter_no_precond) = simple_gmres(matvec, &b, None, 1e-10, 100);

    // Solve with block ILU preconditioning
    let (x_ilu, iter_ilu) = simple_gmres(matvec, &b, Some(precond_boxed), 1e-10, 100);

    // Check that solutions match
    for i in 0..n_total {
        assert!(
            (x_ref[i] - x_ilu[i]).abs() < 1e-10,
            "Solutions differ at index {}: {:?} vs {:?}",
            i,
            x_ref[i],
            x_ilu[i]
        );
    }

    // Verify that preconditioner reduces iteration count
    assert!(
        iter_ilu < iter_no_precond,
        "Block ILU preconditioner didn't reduce iteration count: {iter_ilu} vs {iter_no_precond}"
    );

    println!(
        "Block ILU preconditioner test succeeded: reduced iterations from {iter_no_precond} to {iter_ilu}"
    );
}

#[test]
#[allow(dead_code)]
fn test_block_jacobi_preconditioner() {
    // Create a larger test case
    let n_x = 10;
    let n_y = 6;
    let n_total = n_x + n_y;

    // Create the test matrix blocks
    let (f_x, f_y, g_x, g_y) = create_test_matrix(n_x, n_y);

    // Create the right-hand side
    let b = create_rhs(n_x, n_y);

    // Step size and BDF coefficient
    let h = 0.1;
    let beta = 0.5;

    // Define the matrix-vector product function
    let matvec = |v: &Array1<f64>| matrix_vector_product(&f_x, &f_y, &g_x, &g_y, h, beta, v);

    // Create the full Jacobian matrix for the Block Jacobi preconditioner
    let mut jacobian = Array2::zeros((n_total, n_total));

    // Fill the blocks
    // Top-left: I - h * β * ∂f/∂x
    for i in 0..n_x {
        for j in 0..n_x {
            if i == j {
                jacobian[[i, j]] = 1.0;
            }
            jacobian[[i, j]] -= h * beta * f_x[[i, j]];
        }
    }

    // Top-right: -h * β * ∂f/∂y
    for i in 0..n_x {
        for j in 0..n_y {
            jacobian[[i, n_x + j]] = -h * beta * f_y[[i, j]];
        }
    }

    // Bottom-left: ∂g/∂x
    for i in 0..n_y {
        for j in 0..n_x {
            jacobian[[n_x + i, j]] = g_x[[i, j]];
        }
    }

    // Bottom-right: ∂g/∂y
    for i in 0..n_y {
        for j in 0..n_y {
            jacobian[[n_x + i, n_x + j]] = g_y[[i, j]];
        }
    }

    // Create the block Jacobi preconditioner with block size 2
    let block_size = 2;
    let precond = create_block_jacobi_preconditioner(&jacobian, block_size);
    let precond_boxed = Box::new(precond);

    // Reference solution: solve without preconditioning
    let (x_ref, iter_no_precond) = simple_gmres(matvec, &b, None, 1e-10, 100);

    // Solve with block Jacobi preconditioning
    let (x_jacobi, iter_jacobi) = simple_gmres(matvec, &b, Some(precond_boxed), 1e-10, 100);

    // Check that solutions match
    for i in 0..n_total {
        assert!(
            (x_ref[i] - x_jacobi[i]).abs() < 1e-10,
            "Solutions differ at index {}: {:?} vs {:?}",
            i,
            x_ref[i],
            x_jacobi[i]
        );
    }

    // Verify that preconditioner reduces iteration count
    assert!(
        iter_jacobi < iter_no_precond,
        "Block Jacobi preconditioner didn't reduce iteration count: {iter_jacobi} vs {iter_no_precond}"
    );

    println!(
        "Block Jacobi preconditioner test succeeded: reduced iterations from {iter_no_precond} to {iter_jacobi}"
    );
}

#[test]
#[allow(dead_code)]
fn test_preconditioner_performance() {
    // Create a large test case for performance comparison
    let n_x = 50;
    let n_y = 20;

    // Create the test matrix blocks
    let (f_x, f_y, g_x, g_y) = create_test_matrix(n_x, n_y);

    // Create the right-hand side
    let b = create_rhs(n_x, n_y);

    // Step size and BDF coefficient
    let h = 0.1;
    let beta = 0.5;

    // Define the matrix-vector product function
    let matvec = |v: &Array1<f64>| matrix_vector_product(&f_x, &f_y, &g_x, &g_y, h, beta, v);

    // Timing: Without preconditioning
    let start = Instant::now();
    let (_x_ref, iter_no_precond) = simple_gmres(matvec, &b, None, 1e-8, 1000);
    let duration_no_precond = start.elapsed();

    // Create the block ILU preconditioner
    let precond = create_block_ilu_preconditioner(&f_x, &f_y, &g_x, &g_y, h, beta);
    let precond_boxed = Box::new(precond);

    // Timing: With block ILU preconditioning
    let start = Instant::now();
    let (_x_ilu, iter_ilu) = simple_gmres(matvec, &b, Some(precond_boxed), 1e-8, 1000);
    let duration_ilu = start.elapsed();

    // Print performance comparison
    println!("Performance comparison for problem size n_x={n_x}, n_y={n_y}:");
    println!("  No preconditioning:  {duration_no_precond:?}, iterations: {iter_no_precond}");
    println!("  Block ILU:           {duration_ilu:?}, iterations: {iter_ilu}");
    println!(
        "  Block ILU speedup:   {:.2}x",
        duration_no_precond.as_secs_f64() / duration_ilu.as_secs_f64()
    );

    // Verify that the block ILU preconditioner reduces the iteration count
    assert!(
        iter_ilu < iter_no_precond,
        "Block ILU preconditioner didn't reduce iteration count for large problem"
    );
}
