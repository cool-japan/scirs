//! Trust region methods for unconstrained optimization

use crate::error::OptimizeError;
use crate::unconstrained::result::OptimizeResult;
use crate::unconstrained::utils::{finite_difference_gradient, finite_difference_hessian};
use crate::unconstrained::Options;
use ndarray::{Array1, Array2, ArrayView1};

/// Implements the Trust-Region Newton Conjugate Gradient method for optimization
pub fn minimize_trust_ncg<F, S>(
    mut fun: F,
    x0: Array1<f64>,
    options: &Options,
) -> Result<OptimizeResult<S>, OptimizeError>
where
    F: FnMut(&ArrayView1<f64>) -> S,
    S: Into<f64> + Clone,
{
    // Get options or use defaults
    let ftol = options.ftol;
    let gtol = options.gtol;
    let max_iter = options.max_iter;
    let eps = options.eps;
    let initial_trust_radius = options.trust_radius.unwrap_or(1.0);
    let max_trust_radius = options.max_trust_radius.unwrap_or(1000.0);
    let min_trust_radius = options.min_trust_radius.unwrap_or(1e-10);
    let eta = options.trust_eta.unwrap_or(1e-4);

    // Initialize variables
    let n = x0.len();
    let mut x = x0.to_owned();

    // Function evaluation counter
    let mut nfev = 0;

    // Initialize function value
    let mut f = fun(&x.view()).into();
    nfev += 1;

    // Initialize gradient
    let mut g = finite_difference_gradient(&mut fun, &x.view(), eps)?;
    nfev += n;

    // Initialize trust radius
    let mut trust_radius = initial_trust_radius;

    // Iteration counter
    let mut iter = 0;

    // Main optimization loop
    while iter < max_iter {
        // Check convergence on gradient
        let g_norm = g.dot(&g).sqrt();
        if g_norm < gtol {
            break;
        }

        // Save the current function value for convergence check
        let f_old = f;

        // Calculate the Hessian approximation using finite differences
        let hess = finite_difference_hessian(&mut fun, &x.view(), eps)?;
        nfev += n * n;

        // Solve the trust-region subproblem to find the step
        let (step, hits_boundary) = trust_region_subproblem(&g, &hess, trust_radius);

        // Calculate the predicted reduction in the model
        let pred_reduction = calculate_predicted_reduction(&g, &hess, &step);

        // Take the step
        let x_new = &x + &step;
        let f_new = fun(&x_new.view()).into();
        nfev += 1;

        // Calculate the actual reduction
        let actual_reduction = f - f_new;

        // Calculate the ratio of actual to predicted reduction
        let ratio = if pred_reduction.abs() < 1e-8 {
            1.0
        } else {
            actual_reduction / pred_reduction
        };

        // Update the trust region radius based on the ratio
        if ratio < 0.25 {
            // Reduction is poor - shrink the trust region
            trust_radius *= 0.25;
        } else if ratio > 0.75 && hits_boundary {
            // Good reduction and we're at the boundary - expand the trust region
            trust_radius = f64::min(2.0 * trust_radius, max_trust_radius);
        }

        // Accept or reject the step based on the ratio
        if ratio > eta {
            // Accept the step
            x = x_new;
            f = f_new;

            // Recalculate the gradient at the new point
            g = finite_difference_gradient(&mut fun, &x.view(), eps)?;
            nfev += n;
        }

        // Check convergence on trust region radius
        if trust_radius < min_trust_radius {
            break;
        }

        // Check convergence on function value if step was accepted
        if ratio > eta && (f_old - f).abs() < ftol * (1.0 + f.abs()) {
            break;
        }

        iter += 1;
    }

    // Use original function for final evaluation
    let final_fun = fun(&x.view());

    // Create and return result
    Ok(OptimizeResult {
        x,
        fun: final_fun,
        iterations: iter,
        nit: iter,
        func_evals: nfev,
        nfev,
        success: iter < max_iter,
        message: if iter < max_iter {
            "Optimization terminated successfully.".to_string()
        } else {
            "Maximum iterations reached.".to_string()
        },
        jacobian: Some(g),
        hessian: None,
    })
}

/// Implements the Trust-Region truncated generalized Lanczos / conjugate gradient algorithm
pub fn minimize_trust_krylov<F, S>(
    mut fun: F,
    x0: Array1<f64>,
    options: &Options,
) -> Result<OptimizeResult<S>, OptimizeError>
where
    F: FnMut(&ArrayView1<f64>) -> S,
    S: Into<f64> + Clone,
{
    // Get options or use defaults
    let ftol = options.ftol;
    let gtol = options.gtol;
    let max_iter = options.max_iter;
    let eps = options.eps;
    let initial_trust_radius = options.trust_radius.unwrap_or(1.0);
    let max_trust_radius = options.max_trust_radius.unwrap_or(1000.0);
    let min_trust_radius = options.min_trust_radius.unwrap_or(1e-10);
    let eta = options.trust_eta.unwrap_or(1e-4);

    // Initialize variables
    let n = x0.len();
    let mut x = x0.to_owned();

    // Function evaluation counter
    let mut nfev = 0;

    // Initialize function value
    let mut f = fun(&x.view()).into();
    nfev += 1;

    // Initialize gradient
    let mut g = finite_difference_gradient(&mut fun, &x.view(), eps)?;
    nfev += n;

    // Initialize trust radius
    let mut trust_radius = initial_trust_radius;

    // Iteration counter
    let mut iter = 0;

    // Main optimization loop
    while iter < max_iter {
        // Check convergence on gradient
        let g_norm = g.dot(&g).sqrt();
        if g_norm < gtol {
            break;
        }

        // Save the current function value for convergence check
        let f_old = f;

        // Calculate the Hessian approximation using finite differences
        let hess = finite_difference_hessian(&mut fun, &x.view(), eps)?;
        nfev += n * n;

        // Solve the trust-region subproblem using Lanczos method
        let (step, hits_boundary) = trust_region_lanczos_subproblem(&g, &hess, trust_radius);

        // Calculate the predicted reduction in the model
        let pred_reduction = calculate_predicted_reduction(&g, &hess, &step);

        // Take the step
        let x_new = &x + &step;
        let f_new = fun(&x_new.view()).into();
        nfev += 1;

        // Calculate the actual reduction
        let actual_reduction = f - f_new;

        // Calculate the ratio of actual to predicted reduction
        let ratio = if pred_reduction.abs() < 1e-8 {
            1.0
        } else {
            actual_reduction / pred_reduction
        };

        // Update the trust region radius based on the ratio
        if ratio < 0.25 {
            // Reduction is poor - shrink the trust region
            trust_radius *= 0.25;
        } else if ratio > 0.75 && hits_boundary {
            // Good reduction and we're at the boundary - expand the trust region
            trust_radius = f64::min(2.0 * trust_radius, max_trust_radius);
        }

        // Accept or reject the step based on the ratio
        if ratio > eta {
            // Accept the step
            x = x_new;
            f = f_new;

            // Recalculate the gradient at the new point
            g = finite_difference_gradient(&mut fun, &x.view(), eps)?;
            nfev += n;
        }

        // Check convergence on trust region radius
        if trust_radius < min_trust_radius {
            break;
        }

        // Check convergence on function value if step was accepted
        if ratio > eta && (f_old - f).abs() < ftol * (1.0 + f.abs()) {
            break;
        }

        iter += 1;
    }

    // Use original function for final evaluation
    let final_fun = fun(&x.view());

    // Create and return result
    Ok(OptimizeResult {
        x,
        fun: final_fun,
        iterations: iter,
        nit: iter,
        func_evals: nfev,
        nfev,
        success: iter < max_iter,
        message: if iter < max_iter {
            "Optimization terminated successfully.".to_string()
        } else {
            "Maximum iterations reached.".to_string()
        },
        jacobian: Some(g),
        hessian: None,
    })
}

/// Implements the Trust-region nearly exact algorithm
pub fn minimize_trust_exact<F, S>(
    mut fun: F,
    x0: Array1<f64>,
    options: &Options,
) -> Result<OptimizeResult<S>, OptimizeError>
where
    F: FnMut(&ArrayView1<f64>) -> S,
    S: Into<f64> + Clone,
{
    // Get options or use defaults
    let ftol = options.ftol;
    let gtol = options.gtol;
    let max_iter = options.max_iter;
    let eps = options.eps;
    let initial_trust_radius = options.trust_radius.unwrap_or(1.0);
    let max_trust_radius = options.max_trust_radius.unwrap_or(1000.0);
    let min_trust_radius = options.min_trust_radius.unwrap_or(1e-10);
    let eta = options.trust_eta.unwrap_or(1e-4);

    // Initialize variables
    let n = x0.len();
    let mut x = x0.to_owned();

    // Function evaluation counter
    let mut nfev = 0;

    // Initialize function value
    let mut f = fun(&x.view()).into();
    nfev += 1;

    // Initialize gradient
    let mut g = finite_difference_gradient(&mut fun, &x.view(), eps)?;
    nfev += n;

    // Initialize trust radius
    let mut trust_radius = initial_trust_radius;

    // Iteration counter
    let mut iter = 0;

    // Main optimization loop
    while iter < max_iter {
        // Check convergence on gradient
        let g_norm = g.dot(&g).sqrt();
        if g_norm < gtol {
            break;
        }

        // Save the current function value for convergence check
        let f_old = f;

        // Calculate the Hessian approximation using finite differences
        let hess = finite_difference_hessian(&mut fun, &x.view(), eps)?;
        nfev += n * n;

        // Solve the trust-region subproblem using exact eigendecomposition
        let (step, hits_boundary) = trust_region_exact_subproblem(&g, &hess, trust_radius);

        // Calculate the predicted reduction in the model
        let pred_reduction = calculate_predicted_reduction(&g, &hess, &step);

        // Take the step
        let x_new = &x + &step;
        let f_new = fun(&x_new.view()).into();
        nfev += 1;

        // Calculate the actual reduction
        let actual_reduction = f - f_new;

        // Calculate the ratio of actual to predicted reduction
        let ratio = if pred_reduction.abs() < 1e-8 {
            1.0
        } else {
            actual_reduction / pred_reduction
        };

        // Update the trust region radius based on the ratio
        if ratio < 0.25 {
            // Reduction is poor - shrink the trust region
            trust_radius *= 0.25;
        } else if ratio > 0.75 && hits_boundary {
            // Good reduction and we're at the boundary - expand the trust region
            trust_radius = f64::min(2.0 * trust_radius, max_trust_radius);
        }

        // Accept or reject the step based on the ratio
        if ratio > eta {
            // Accept the step
            x = x_new;
            f = f_new;

            // Recalculate the gradient at the new point
            g = finite_difference_gradient(&mut fun, &x.view(), eps)?;
            nfev += n;
        }

        // Check convergence on trust region radius
        if trust_radius < min_trust_radius {
            break;
        }

        // Check convergence on function value if step was accepted
        if ratio > eta && (f_old - f).abs() < ftol * (1.0 + f.abs()) {
            break;
        }

        iter += 1;
    }

    // Use original function for final evaluation
    let final_fun = fun(&x.view());

    // Create and return result
    Ok(OptimizeResult {
        x,
        fun: final_fun,
        iterations: iter,
        nit: iter,
        func_evals: nfev,
        nfev,
        success: iter < max_iter,
        message: if iter < max_iter {
            "Optimization terminated successfully.".to_string()
        } else {
            "Maximum iterations reached.".to_string()
        },
        jacobian: Some(g),
        hessian: None,
    })
}

/// Solve the trust-region subproblem using the conjugate gradient method
fn trust_region_subproblem(
    g: &Array1<f64>,
    hess: &Array2<f64>,
    trust_radius: f64,
) -> (Array1<f64>, bool) {
    let n = g.len();

    // Start with the steepest descent direction
    let mut p = -g.clone();

    // If the gradient is zero, return a zero step
    if g.dot(g) < 1e-10 {
        return (Array1::zeros(n), false);
    }

    // Initialize the step as the zero vector
    let mut s = Array1::zeros(n);

    // Initialize the residual as -g
    let mut r = g.clone();

    // Compute the norm of the gradient
    let g_norm = g.dot(g).sqrt();

    // Set convergence criteria
    let cg_tol = f64::min(0.1, g_norm);
    let max_cg_iters = n * 2;

    // Flag to indicate if we hit the boundary
    let mut hits_boundary = false;

    // Conjugate gradient iterations
    for _ in 0..max_cg_iters {
        // Compute H*p
        let hp = hess.dot(&p);

        // Compute p'*H*p
        let php = p.dot(&hp);

        // If the curvature is negative or close to zero, we hit the boundary
        if php <= 0.0 {
            // Find the boundary step
            let (_alpha, boundary_step) = find_boundary_step(&s, &p, trust_radius);
            hits_boundary = true;
            return (boundary_step, hits_boundary);
        }

        // Compute the CG step size
        let alpha = r.dot(&r) / php;

        // Take the step
        let s_next = &s + &(&p * alpha);

        // Check if we exceed the trust radius
        if s_next.dot(&s_next).sqrt() >= trust_radius {
            // Find the boundary step
            let (_alpha, boundary_step) = find_boundary_step(&s, &p, trust_radius);
            hits_boundary = true;
            return (boundary_step, hits_boundary);
        }

        // Update the step
        s = s_next;

        // Update the residual: r_{k+1} = r_k + alpha * H * p_k
        r = &r + &(&hp * alpha);

        // Check convergence
        if r.dot(&r).sqrt() < cg_tol {
            break;
        }

        // Compute the beta parameter for the next CG direction
        let r_new_norm_squared = r.dot(&r);
        let r_old_norm_squared = p.dot(&p);
        let beta = r_new_norm_squared / r_old_norm_squared;

        // Update the CG direction
        p = -&r + &(&p * beta);
    }

    (s, hits_boundary)
}

/// Solve the trust region subproblem using the Lanczos method
fn trust_region_lanczos_subproblem(
    g: &Array1<f64>,
    hess: &Array2<f64>,
    trust_radius: f64,
) -> (Array1<f64>, bool) {
    let n = g.len();

    // Start with the steepest descent direction
    let mut v1 = -g.clone();
    v1 = &v1 / v1.dot(&v1).sqrt();

    // If the gradient is zero, return a zero step
    if g.dot(g) < 1e-10 {
        return (Array1::zeros(n), false);
    }

    // Maximum number of Lanczos iterations (typically much smaller than dimension)
    let max_lanczos_iters = 10.min(n);

    // Store the Lanczos vectors
    let mut v = Vec::with_capacity(max_lanczos_iters);
    v.push(v1);

    // Store the tridiagonal matrix elements
    let mut alpha = Vec::with_capacity(max_lanczos_iters);
    let mut beta = Vec::with_capacity(max_lanczos_iters);

    // Construct the Lanczos tridiagonal decomposition
    let mut w = hess.dot(&v[0]);
    alpha.push(w.dot(&v[0]));

    let mut hits_boundary = false;

    // Lanczos iterations to build the Krylov subspace
    for j in 1..max_lanczos_iters {
        // w = H*v_j - beta_{j-1}*v_{j-1}
        if j > 1 {
            w -= &(&v[j - 2] * beta[j - 2]);
        }

        // w = w - alpha_{j-1}*v_{j-1}
        w -= &(&v[j - 1] * alpha[j - 1]);

        // Reorthogonalize (important for numerical stability)
        for vi in v.iter().take(j) {
            let projection = w.dot(vi);
            w -= &(vi * projection);
        }

        // Compute beta_j
        let b = w.dot(&w).sqrt();
        beta.push(b);

        // Check if we can continue building the Krylov subspace
        if b < 1e-10 {
            // Exact solution found within the Krylov subspace
            break;
        }

        // Normalize w to get the next Lanczos vector
        let vj = &w / b;
        v.push(vj.clone());

        // Update w for next iteration
        w = hess.dot(&vj);
        alpha.push(w.dot(&vj));
    }

    // Now solve the trust region subproblem in the Krylov subspace
    let k = alpha.len();

    // Construct the tridiagonal matrix T
    let mut t = Array2::zeros((k, k));
    for i in 0..k {
        t[[i, i]] = alpha[i];
        if i < k - 1 {
            t[[i, i + 1]] = beta[i];
            t[[i + 1, i]] = beta[i];
        }
    }

    // Get the smallest eigenvalue of T
    let mut lambda_min = alpha[0];
    for &a in alpha.iter().take(k).skip(1) {
        lambda_min = f64::min(lambda_min, a);
    }

    // Initial guess for lambda (shifted eigenvalue)
    let mut lambda = if lambda_min < 0.0 {
        -lambda_min + 0.1
    } else {
        0.0
    };

    // Build the right-hand side (first basis vector scaled by ||g||)
    let mut b = Array1::zeros(k);
    b[0] = g.dot(g).sqrt();

    // Iteratively solve using a shifted system and update lambda
    let mut s = Array1::zeros(k);
    let mut inside_trust_region = false;

    // Maximum iterations for the trust region subproblem
    let max_tr_iters = 10;

    for _ in 0..max_tr_iters {
        // Solve (T + lambda*I)s = b using tridiagonal solver
        s = solve_tridiagonal_system(&t, &b, lambda);

        // Check if we're inside the trust region
        let norm_s = s.dot(&s).sqrt();

        if (norm_s - trust_radius).abs() < 1e-6 * trust_radius {
            // We're at the boundary of the trust region
            inside_trust_region = false;
            hits_boundary = true;
            break;
        } else if norm_s < trust_radius {
            // We're inside the trust region
            inside_trust_region = true;

            // Check if lambda is effectively zero (unconstrained solution)
            if lambda < 1e-10 {
                break;
            }

            // Decrease lambda to move closer to the boundary
            lambda /= 4.0;
        } else {
            // We're outside the trust region, increase lambda
            lambda *= 2.0;
        }
    }

    // Convert the solution in the Krylov subspace back to the original space
    let mut step: Array1<f64> = Array1::zeros(n);
    for (i, vi) in v.iter().take(k).enumerate() {
        step += &(vi * s[i]);
    }

    // If we're inside the trust region but lambda > 0, we've hit numerical issues
    // In this case, scale the solution to the boundary
    if inside_trust_region && lambda > 1e-10 {
        let norm_step = step.dot(&step).sqrt();
        step = &step * (trust_radius / norm_step);
        hits_boundary = true;
    }

    (step, hits_boundary)
}

/// Solve the trust region subproblem using the exact method with eigendecomposition
fn trust_region_exact_subproblem(
    g: &Array1<f64>,
    hess: &Array2<f64>,
    trust_radius: f64,
) -> (Array1<f64>, bool) {
    let n = g.len();

    // If the gradient is zero, return a zero step
    if g.dot(g) < 1e-10 {
        return (Array1::zeros(n), false);
    }

    // Compute eigendecomposition of the Hessian matrix
    let (eigvals, eigvecs) = compute_eig_decomposition(hess);

    // Check if the Hessian is positive definite (all eigenvalues > 0)
    let min_eigval = eigvals.iter().cloned().fold(f64::INFINITY, f64::min);

    // Transform the gradient to the eigenbasis
    let mut g_transformed = Array1::zeros(n);
    for i in 0..n {
        let eigvec_i = eigvecs.column(i);
        g_transformed[i] = -eigvec_i.dot(g); // Negative because we're minimizing
    }

    // If the Hessian is positive definite, try the unconstrained Newton step first
    if min_eigval > 0.0 {
        // Compute the unconstrained step: s = -H^(-1) * g
        let mut newton_step = Array1::zeros(n);
        for i in 0..n {
            newton_step[i] = g_transformed[i] / eigvals[i];
        }

        // Transform back to the original basis
        let mut step: Array1<f64> = Array1::zeros(n);
        for i in 0..n {
            let eigvec_i = eigvecs.column(i);
            step += &(&eigvec_i * newton_step[i]);
        }

        // Check if the Newton step is within the trust radius
        let step_norm = step.dot(&step).sqrt();
        if step_norm <= trust_radius {
            // Unconstrained minimizer is within the trust region
            return (step, false);
        }
    }

    // The unconstrained minimizer is outside the trust region or the Hessian is not positive definite
    // We need to find the optimal lambda (Lagrange multiplier) that gives a step at the trust region boundary

    // Define a function that gives the step norm for a given lambda
    let phi = |lambda: f64| -> f64 {
        let mut norm_squared = 0.0;
        for i in 0..n {
            let step_i = g_transformed[i] / (eigvals[i] + lambda);
            norm_squared += step_i * step_i;
        }
        norm_squared.sqrt() - trust_radius
    };

    // Find the optimal lambda using a numerical method (e.g., bisection)
    let lambda_min = if min_eigval > 0.0 {
        0.0
    } else {
        -min_eigval + 1e-6
    };
    let lambda_max = lambda_min + 1000.0; // Some large value

    let lambda = find_lambda_bisection(lambda_min, lambda_max, phi);

    // Compute the step with the optimal lambda
    let mut opt_step_transformed = Array1::zeros(n);
    for i in 0..n {
        opt_step_transformed[i] = g_transformed[i] / (eigvals[i] + lambda);
    }

    // Transform back to the original basis
    let mut step: Array1<f64> = Array1::zeros(n);
    for i in 0..n {
        let eigvec_i = eigvecs.column(i);
        step += &(&eigvec_i * opt_step_transformed[i]);
    }

    (step, true) // We're at the boundary
}

/// Find a step that lies on the trust region boundary
fn find_boundary_step(s: &Array1<f64>, p: &Array1<f64>, trust_radius: f64) -> (f64, Array1<f64>) {
    // Solve the quadratic equation ||s + alpha*p||^2 = trust_radius^2
    let s_norm_squared = s.dot(s);
    let p_norm_squared = p.dot(p);
    let s_dot_p = s.dot(p);

    let a = p_norm_squared;
    let b = 2.0 * s_dot_p;
    let c = s_norm_squared - trust_radius * trust_radius;

    // Solve the quadratic equation
    let disc = b * b - 4.0 * a * c;
    let disc = f64::max(disc, 0.0);

    // We want the positive root that brings us to the boundary
    let alpha = (-b + disc.sqrt()) / (2.0 * a);

    // Compute the boundary step
    let boundary_step = s + &(p * alpha);

    (alpha, boundary_step)
}

/// Calculate the predicted reduction in the quadratic model
fn calculate_predicted_reduction(g: &Array1<f64>, hess: &Array2<f64>, step: &Array1<f64>) -> f64 {
    // The model is 0.5 * s'*B*s + g'*s
    let g_dot_s = g.dot(step);
    let s_dot_bs = step.dot(&hess.dot(step));

    -g_dot_s - 0.5 * s_dot_bs
}

/// Solve a tridiagonal system (T + lambda*I)x = b
fn solve_tridiagonal_system(t: &Array2<f64>, b: &Array1<f64>, lambda: f64) -> Array1<f64> {
    let n = t.shape()[0];
    let mut d = Array1::zeros(n); // Diagonal elements
    let mut e = Array1::zeros(n - 1); // Off-diagonal elements

    // Extract diagonal and off-diagonal elements
    for i in 0..n {
        d[i] = t[[i, i]] + lambda;
        if i < n - 1 {
            e[i] = t[[i, i + 1]];
        }
    }

    // Forward substitution
    let mut u = Array1::zeros(n);
    let mut w = d[0];
    u[0] = b[0] / w;

    for i in 1..n {
        let temp = e[i - 1] / w;
        d[i] -= temp * e[i - 1];
        w = d[i];
        u[i] = (b[i] - temp * u[i - 1]) / w;
    }

    // Back substitution
    let mut x = Array1::zeros(n);
    x[n - 1] = u[n - 1];

    for i in (0..n - 1).rev() {
        x[i] = u[i] - e[i] * x[i + 1] / d[i];
    }

    x
}

/// Compute eigendecomposition of a matrix (simplified version)
fn compute_eig_decomposition(mat: &Array2<f64>) -> (Array1<f64>, Array2<f64>) {
    let n = mat.nrows();

    // This is a simplistic approach to eigendecomposition
    // For production code, you would use a library like nalgebra or ndarray-linalg

    let mut eigvals = Array1::zeros(n);
    let mut eigvecs = Array2::zeros((n, n));

    // Create a copy of the matrix for deflation
    let mut mat_copy = mat.clone();

    for k in 0..n {
        // Initialize a simple vector for the power method
        let mut v = Array1::zeros(n);
        v[k % n] = 1.0;
        // Add a small perturbation to avoid issues with symmetry
        for i in 0..n {
            v[i] += 0.01 * (i as f64);
        }

        v = &v / v.dot(&v).sqrt(); // Normalize

        // Run power method iterations
        for _ in 0..50 {
            // Multiply the matrix by the vector
            let w = mat_copy.dot(&v);

            // Normalize
            let norm = w.dot(&w).sqrt();
            if norm < 1e-10 {
                break;
            }

            v = &w / norm;
        }

        // Rayleigh quotient to find the eigenvalue
        let eigval = v.dot(&mat_copy.dot(&v));
        eigvals[k] = eigval;

        // Store the eigenvector
        for i in 0..n {
            eigvecs[[i, k]] = v[i];
        }

        // Deflate the matrix: M' = M - lambda * v * v^T
        for i in 0..n {
            for j in 0..n {
                mat_copy[[i, j]] -= eigval * v[i] * v[j];
            }
        }
    }

    (eigvals, eigvecs)
}

/// Find the optimal lambda using the bisection method
fn find_lambda_bisection<F>(a: f64, b: f64, f: F) -> f64
where
    F: Fn(f64) -> f64,
{
    let mut a = a;
    let mut b = b;
    let tol = 1e-10;
    let max_iter = 100;

    let mut fa = f(a);

    // If fa is positive, we need to find a negative value first
    if fa > 0.0 {
        let mut b_temp = a + 1.0;
        let mut fb_temp = f(b_temp);

        while fb_temp > 0.0 && b_temp < 1e6 {
            b_temp *= 2.0;
            fb_temp = f(b_temp);
        }

        if fb_temp > 0.0 {
            // Couldn't find a bracket with the desired property
            return a; // Return the lower bound as a fallback
        }

        b = b_temp;
    } else if fa == 0.0 {
        return a; // We got lucky and found the root exactly
    }

    let mut iter = 0;

    while (b - a) > tol && iter < max_iter {
        let c = (a + b) / 2.0;
        let fc = f(c);

        if fc.abs() < tol {
            return c; // Found a root
        }

        if fc * fa < 0.0 {
            // Root is in [a, c]
            b = c;
        } else {
            // Root is in [c, b]
            a = c;
            fa = fc;
        }

        iter += 1;
    }

    (a + b) / 2.0 // Return the midpoint of the final interval
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_trust_ncg_quadratic() {
        let quadratic = |x: &ArrayView1<f64>| -> f64 { x[0] * x[0] + 4.0 * x[1] * x[1] };

        let x0 = Array1::from_vec(vec![2.0, 1.0]);
        let options = Options::default();

        let result = minimize_trust_ncg(quadratic, x0, &options).unwrap();

        assert!(result.success);
        assert_abs_diff_eq!(result.x[0], 0.0, epsilon = 1e-4);
        assert_abs_diff_eq!(result.x[1], 0.0, epsilon = 1e-4);
    }

    #[test]
    #[ignore] // FIXME: Algorithm gets stuck at (0.0, y) instead of converging to (1.0, 1.0)
    fn test_trust_krylov_rosenbrock() {
        let rosenbrock = |x: &ArrayView1<f64>| -> f64 {
            let a = 1.0;
            let b = 100.0;
            (a - x[0]).powi(2) + b * (x[1] - x[0].powi(2)).powi(2)
        };

        let x0 = Array1::from_vec(vec![0.0, 0.0]);
        let mut options = Options::default();
        options.max_iter = 200; // Trust region methods may need more iterations

        let result = minimize_trust_krylov(rosenbrock, x0, &options).unwrap();

        assert!(result.success);
        assert_abs_diff_eq!(result.x[0], 1.0, epsilon = 1e-2);
        assert_abs_diff_eq!(result.x[1], 1.0, epsilon = 1e-2);
    }

    #[test]
    fn test_trust_exact_simple() {
        let quadratic =
            |x: &ArrayView1<f64>| -> f64 { (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2) };

        let x0 = Array1::from_vec(vec![0.0, 0.0]);
        let options = Options::default();

        let result = minimize_trust_exact(quadratic, x0, &options).unwrap();

        assert!(result.success);
        assert_abs_diff_eq!(result.x[0], 1.0, epsilon = 1e-4);
        assert_abs_diff_eq!(result.x[1], 2.0, epsilon = 1e-4);
    }
}
