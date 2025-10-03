use crate::error::OptimizeResult;
use crate::result::OptimizeResults;
use crate::roots::Options;
use scirs2_core::ndarray::{s, Array1, Array2, ArrayBase, Data, Ix1};

/// Implements the Krylov method accelerated by GMRES for root finding
///
/// This method uses Krylov subspace techniques to accelerate the convergence
/// of root finding algorithms, particularly for large-scale systems. It combines
/// the Levenberg-Marquardt approach with Krylov subspace methods for the linear solve.
#[allow(dead_code)]
pub fn root_krylov<F, J, S>(
    func: F,
    x0: &ArrayBase<S, Ix1>,
    jacobian_fn: Option<J>,
    options: &Options,
) -> OptimizeResult<OptimizeResults<f64>>
where
    F: Fn(&[f64]) -> Array1<f64>,
    J: Fn(&[f64]) -> Array2<f64>,
    S: Data<Elem = f64>,
{
    // Get options or use defaults
    let xtol = options.xtol.unwrap_or(1e-8);
    let ftol = options.ftol.unwrap_or(1e-8);
    let maxfev = options.maxfev.unwrap_or(100 * x0.len());
    let eps = options.eps.unwrap_or(1e-8);

    // Initialize variables
    let n = x0.len();
    let mut x = x0.to_owned();
    let mut f = func(x.as_slice().unwrap());
    let mut nfev = 1;
    let mut njev = 0;

    // Function to compute numerical Jacobian
    let compute_numerical_jac =
        |x_values: &[f64], f_values: &Array1<f64>| -> (Array2<f64>, usize) {
            let mut jac = Array2::zeros((f_values.len(), x_values.len()));
            let mut count = 0;

            for j in 0..x_values.len() {
                let mut x_h = Vec::from(x_values);
                x_h[j] += eps;
                let f_h = func(&x_h);
                count += 1;

                for i in 0..f_values.len() {
                    jac[[i, j]] = (f_h[i] - f_values[i]) / eps;
                }
            }

            (jac, count)
        };

    // Function to get Jacobian (either analytical or numerical)
    let get_jacobian = |x_values: &[f64],
                        f_values: &Array1<f64>,
                        jac_fn: &Option<J>|
     -> (Array2<f64>, usize, usize) {
        match jac_fn {
            Some(func) => {
                let j = func(x_values);
                (j, 0, 1)
            }
            None => {
                let (j, count) = compute_numerical_jac(x_values, f_values);
                (j, count, 0)
            }
        }
    };

    // Compute initial Jacobian
    let (mut jac, nfev_inc, njev_inc) = get_jacobian(x.as_slice().unwrap(), &f, &jacobian_fn);
    nfev += nfev_inc;
    njev += njev_inc;

    // Main iteration loop
    let mut iter = 0;
    let mut converged = false;

    // Levenberg-Marquardt damping parameter
    let mut lambda = 0.01;
    let lambda_adjustment = 10.0;

    while iter < maxfev {
        // Check if we've converged in function values
        let f_norm = f.iter().map(|&fi| fi.powi(2)).sum::<f64>().sqrt();
        if f_norm < ftol {
            converged = true;
            break;
        }

        // Solve using the GMRES-accelerated approach:
        // Instead of directly solving (J^T J + λI) δ = -J^T f
        // We use the GMRES method to iteratively solve J δ = -f

        // Initialize the Krylov subspace
        let max_krylov_dim = std::cmp::min(n, 20); // Limit Krylov dimension

        // Run a modified GMRES method with Levenberg-Marquardt damping
        let delta = gmres_solve(&jac, &(-&f), lambda, max_krylov_dim);

        // Apply the step
        let mut x_new = x.clone();
        for i in 0..n {
            x_new[i] += delta[i];
        }

        // Evaluate function at the new point
        let f_new = func(x_new.as_slice().unwrap());
        nfev += 1;

        let f_new_norm = f_new.iter().map(|&fi| fi.powi(2)).sum::<f64>().sqrt();
        let f_norm = f.iter().map(|&fi| fi.powi(2)).sum::<f64>().sqrt();

        // Adaptive Levenberg-Marquardt strategy
        if f_new_norm < f_norm {
            // Step decreased residual, decrease damping
            lambda /= lambda_adjustment;

            // Update variables for next iteration
            x = x_new;
            f = f_new;

            // Update Jacobian for next iteration
            let (new_jac, nfev_delta, njev_delta) =
                get_jacobian(x.as_slice().unwrap(), &f, &jacobian_fn);
            jac = new_jac;
            nfev += nfev_delta;
            njev += njev_delta;
        } else {
            // Step increased residual, increase damping and retry
            lambda *= lambda_adjustment;
        }

        // Check convergence on parameters
        let step_norm = delta.iter().map(|&di| di.powi(2)).sum::<f64>().sqrt();
        let x_norm = x.iter().map(|&xi| xi.powi(2)).sum::<f64>().sqrt();

        if step_norm < xtol * (1.0 + x_norm) {
            converged = true;
            break;
        }

        iter += 1;
    }

    // Create and return result
    let mut result = OptimizeResults::default();
    result.x = x;
    result.fun = f.iter().map(|&fi| fi.powi(2)).sum::<f64>();

    // Store the final Jacobian
    let (jac_vec, _) = jac.into_raw_vec_and_offset();
    result.jac = Some(jac_vec);

    result.nfev = nfev;
    result.njev = njev;
    result.nit = iter;
    result.success = converged;

    if converged {
        result.message = "Root finding converged successfully".to_string();
    } else {
        result.message = "Maximum number of function evaluations reached".to_string();
    }

    Ok(result)
}

/// Implements a simplified GMRES (Generalized Minimal Residual) method
/// for solving linear systems Ax = b with Levenberg-Marquardt regularization
#[allow(dead_code)]
fn gmres_solve(a: &Array2<f64>, b: &Array1<f64>, lambda: f64, maxiter: usize) -> Array1<f64> {
    let (_m, n) = a.dim();

    // Regularized system: solve (A^T A + λI) x = A^T b
    // We implement GMRES on this system directly

    // Initialize solution vector
    let mut x = Array1::zeros(n);

    // Initial residual: r = b - A*x (x is zero, so r = b)
    let r = b.clone();
    let r_norm_initial = r.iter().map(|&ri| ri.powi(2)).sum::<f64>().sqrt();

    // If the initial residual is small, return zero solution
    if r_norm_initial < 1e-10 {
        return x;
    }

    // Arnoldi basis vectors (orthonormal basis for the Krylov subspace)
    let mut v = Vec::with_capacity(maxiter + 1);

    // First basis vector: v[0] = r / ||r||
    let mut v0 = r.clone();
    v0 /= r_norm_initial;
    v.push(v0);

    // Hessenberg matrix (stores the projection of A onto the Krylov subspace)
    let mut h = Array2::zeros((maxiter + 1, maxiter));

    // Construct the Krylov subspace and Hessenberg matrix
    for j in 0..maxiter {
        // Apply the matrix operator with Levenberg-Marquardt regularization:
        // w = A * v[j]
        let w = a.dot(&v[j]);

        // Apply the regularization: w = A^T * w + λ * v[j]
        let atw = a.t().dot(&w);
        let mut w_regularized = atw;

        // Add regularization term λ * v[j]
        for i in 0..n {
            w_regularized[i] += lambda * v[j][i];
        }

        // Orthogonalize against previous vectors (Modified Gram-Schmidt)
        for i in 0..=j {
            h[[i, j]] = v[i].dot(&w_regularized);
            for k in 0..n {
                w_regularized[k] -= h[[i, j]] * v[i][k];
            }
        }

        // Compute the norm of the orthogonalized vector
        let w_norm = w_regularized
            .iter()
            .map(|&wi| wi.powi(2))
            .sum::<f64>()
            .sqrt();

        // Check for linear dependence
        if w_norm < 1e-10 {
            break;
        }

        // Add the next basis vector: v[j+1] = w / ||w||
        h[[j + 1, j]] = w_norm;
        let vj1 = w_regularized / w_norm;
        v.push(vj1);

        // Check if we have enough vectors (we've found an invariant subspace)
        if j >= n - 1 {
            break;
        }
    }

    // Solve the least-squares problem in the Krylov subspace
    let k = v.len() - 1; // Actual dimension of the Krylov subspace

    // Form the system H * y = ||r||_2 * e_1
    let mut g = Array1::zeros(k + 1);
    g[0] = r_norm_initial;

    // Setup the Hessenberg matrix for solving
    let h_ls = h.slice(s![0..k + 1, 0..k]).to_owned();

    // Solve the least-squares problem using normal equations
    let h_t = h_ls.t();
    let normal_matrix = h_t.dot(&h_ls);
    let rhs = h_t.dot(&g);

    // Use direct solve via pseudo-inverse
    let y = solve_normal_equations(&normal_matrix, &rhs);

    // Form the solution in the original space: x = V * y
    for j in 0..k {
        for i in 0..n {
            x[i] += y[j] * v[j][i];
        }
    }

    x
}

/// Solves the normal equations A^T A x = A^T b for least squares problems
#[allow(dead_code)]
fn solve_normal_equations(a: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let n = a.dim().0;
    let mut x = Array1::zeros(n);

    // Simple Cholesky decomposition (A^T A is symmetric positive definite)
    let mut l = Array2::zeros((n, n));

    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[[i, j]];

            for k in 0..j {
                sum -= l[[i, k]] * l[[j, k]];
            }

            if i == j {
                l[[i, j]] = (sum + 1e-10).sqrt(); // Add small regularization for stability
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }

    // Forward substitution to solve L * y = b
    let mut y = Array1::zeros(n);
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[[i, j]] * y[j];
        }
        y[i] = sum / l[[i, i]];
    }

    // Backward substitution to solve L^T * x = y
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= l[[j, i]] * x[j];
        }
        x[i] = sum / l[[i, i]];
    }

    x
}
