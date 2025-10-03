use crate::error::OptimizeResult;
use crate::result::OptimizeResults;
use crate::roots::Options;
use scirs2_core::ndarray::{Array1, Array2, ArrayBase, Data, Ix1};

/// Implements Anderson mixing method for root finding
///
/// Anderson mixing is an acceleration technique that combines the current and previous
/// iterates to accelerate convergence. It can be viewed as a type of multisecant quasi-Newton
/// method that uses information from multiple previous iterations.
#[allow(dead_code)]
pub fn root_anderson<F, J, S>(
    func: F,
    x0: &ArrayBase<S, Ix1>,
    _jacobian_fn: Option<J>,
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

    // Anderson specific parameters
    let m = options.m_anderson.unwrap_or(5).min(x0.len()); // Number of previous iterations to use
    let beta = options.beta_anderson.unwrap_or(0.5); // Damping parameter

    // Initialize variables
    let n = x0.len();
    let mut x = x0.to_owned();
    let mut f = func(x.as_slice().unwrap());
    let mut nfev = 1;

    // Storage for previous residuals and step differences
    let mut residuals: Vec<Array1<f64>> = Vec::with_capacity(m + 1);
    let mut steps: Vec<Array1<f64>> = Vec::with_capacity(m + 1);

    // Main iteration loop
    let mut iter = 0;
    let mut converged = false;

    while iter < maxfev {
        // Check if we've converged in function values
        let f_norm = f.iter().map(|&fi| fi.powi(2)).sum::<f64>().sqrt();
        if f_norm < ftol {
            converged = true;
            break;
        }

        // Simple step (fixed-point iteration): x_{n+1} = x_n - f(x_n)
        let mut x_simple = x.clone();
        for i in 0..n {
            x_simple[i] = x[i] - f[i];
        }

        // Apply Anderson mixing if we have previous iterations
        let x_new = if iter >= 1 {
            // Add current residual to the history
            residuals.push(f.clone());

            // Use only m most recent iterations
            if residuals.len() > m {
                residuals.remove(0);
                steps.remove(0);
            }

            // Number of previous iterations to use (limited by available history)
            let mk = residuals.len() - 1;

            if mk >= 1 {
                // Construct the matrix of residual differences
                let mut df = Array2::zeros((n, mk));
                for j in 0..mk {
                    for i in 0..n {
                        df[[i, j]] = residuals[j + 1][i] - residuals[0][i];
                    }
                }

                // Solve the least squares problem: min ||residuals[0] - df * alpha||
                // To find coefficients alpha that best combine previous iterations
                let alpha = match solve_least_squares(&df, &(-&residuals[0])) {
                    Some(a) => a,
                    None => {
                        // If the least squares problem fails, use simple iteration
                        let mut x_next = x.clone();
                        for i in 0..n {
                            x_next[i] = (1.0 - beta) * x[i] + beta * x_simple[i];
                        }
                        x_next
                    }
                };

                // Apply the mixing to compute the next iterate
                let mut x_next = x.clone();

                // Start with weighted current point
                for i in 0..n {
                    x_next[i] = x_simple[i] * (1.0 - alpha.iter().sum::<f64>());
                }

                // Add weighted previous steps
                for j in 0..mk {
                    for i in 0..n {
                        x_next[i] += alpha[j] * steps[j][i];
                    }
                }

                // Apply damping
                for i in 0..n {
                    x_next[i] = (1.0 - beta) * x[i] + beta * x_next[i];
                }

                x_next
            } else {
                // Use simple iteration if we don't have enough history
                let mut x_next = x.clone();
                for i in 0..n {
                    x_next[i] = (1.0 - beta) * x[i] + beta * x_simple[i];
                }
                x_next
            }
        } else {
            // For the first step, just use damped fixed-point iteration
            let mut x_next = x.clone();
            for i in 0..n {
                x_next[i] = (1.0 - beta) * x[i] + beta * x_simple[i];
            }
            x_next
        };

        // Save step for next iteration
        steps.push(x_simple);

        // Evaluate function at the new point
        let f_new = func(x_new.as_slice().unwrap());
        nfev += 1;

        // Check convergence on parameters
        let step_norm = (0..n)
            .map(|i| (x_new[i] - x[i]).powi(2))
            .sum::<f64>()
            .sqrt();
        let x_norm = (0..n).map(|i| x[i].powi(2)).sum::<f64>().sqrt();

        if step_norm < xtol * (1.0 + x_norm) {
            converged = true;
            x = x_new;
            f = f_new;
            break;
        }

        // Update variables for next iteration
        x = x_new;
        f = f_new;

        iter += 1;
    }

    // Create and return result
    let mut result = OptimizeResults::default();
    result.x = x;
    result.fun = f.iter().map(|&fi| fi.powi(2)).sum::<f64>();
    result.nfev = nfev;
    result.nit = iter;
    result.success = converged;

    if converged {
        result.message = "Root finding converged successfully".to_string();
    } else {
        result.message = "Maximum number of function evaluations reached".to_string();
    }

    Ok(result)
}

/// Solves a least squares problem ||Ax - b|| using QR decomposition
#[allow(dead_code)]
pub fn solve_least_squares(a: &Array2<f64>, b: &Array1<f64>) -> Option<Array1<f64>> {
    let (m, n) = (a.nrows(), a.ncols());
    if m < n {
        return None; // Underdetermined system
    }

    // Simple QR decomposition using Householder reflections
    let mut q: Array2<f64> = Array2::eye(m);
    let mut r = a.clone();

    for k in 0..n {
        // Householder transformation to introduce zeros below the diagonal
        let mut x = Array1::zeros(m - k);
        for i in k..m {
            x[i - k] = r[[i, k]];
        }

        let x_norm = x.iter().map(|&xi| xi.powi(2)).sum::<f64>().sqrt();
        if x_norm < 1e-10 {
            continue; // Skip if column is already zeroed
        }

        // Compute Householder vector
        let mut v = x.clone();
        v[0] += if x[0] >= 0.0 { x_norm } else { -x_norm };

        let v_norm = v.iter().map(|&vi| vi.powi(2)).sum::<f64>().sqrt();
        if v_norm < 1e-10 {
            continue;
        }

        // Normalize v
        for i in 0..v.len() {
            v[i] /= v_norm;
        }

        // Apply Householder transformation to R
        for j in k..n {
            let mut dot_product = 0.0;
            for i in 0..v.len() {
                dot_product += v[i] * r[[i + k, j]];
            }

            let factor = 2.0 * dot_product;
            for i in 0..v.len() {
                r[[i + k, j]] -= factor * v[i];
            }
        }

        // Apply Householder transformation to Q
        for j in 0..m {
            let mut dot_product = 0.0;
            for i in 0..v.len() {
                dot_product += v[i] * q[[j, i + k]];
            }

            let factor = 2.0 * dot_product;
            for i in 0..v.len() {
                q[[j, i + k]] -= factor * v[i];
            }
        }
    }

    // Now we have R and Q (actually Q^T) such that A = QR
    // To solve least squares, we compute x = R^-1 Q^T b

    // Apply Q^T to b: y = Q^T b
    let mut y = Array1::zeros(m);
    for i in 0..m {
        for j in 0..m {
            y[i] += q[[i, j]] * b[j];
        }
    }

    // Back-substitution to solve Rx = y
    let mut x = Array1::zeros(n);

    for i in (0..n).rev() {
        let mut sum: f64 = y[i];
        for j in (i + 1)..n {
            sum -= r[[i, j]] * x[j];
        }

        if r[[i, i]].abs() < 1e-10 {
            return None; // Singular matrix
        }

        x[i] = sum / r[[i, i]];
    }

    Some(x)
}
