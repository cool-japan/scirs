use ndarray::s;
// Advanced system identification methods
//
// This module provides complete implementations of advanced system
// identification methods including ARMAX, Output-Error, Box-Jenkins,
// state-space, and nonlinear ARX models.

use crate::error::{SignalError, SignalResult};
use crate::lti::StateSpace;
use crate::sysid_enhanced::{NonlinearFunction, ParameterEstimate, SystemModel};
use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use scirs2_core::parallel_ops::*;
use scirs2_core::validation::checkshape;
use scirs2_linalg::solve;
use statrs::statistics::Statistics;

#[allow(unused_imports)]
/// ARMAX model identification using iterative prediction error method
///
/// ARMAX: A(q)y(t) = B(q)u(t) + C(q)e(t)
#[allow(dead_code)]
pub fn identify_armax_complete(
    input: &Array1<f64>,
    output: &Array1<f64>,
    na: usize, // AR order
    nb: usize, // B polynomial order
    nc: usize, // MA order
    delay: usize,
) -> SignalResult<(SystemModel, ParameterEstimate, usize, bool, f64)> {
    let n = output.len();
    checkshape(input, &[n], "input and output")?;

    // Initialize with ARX estimate
    let (ar_init, b_init) = estimate_arx_ls(input, output, na, nb, delay)?;

    // Initialize MA coefficients
    let mut c_coeffs = Array1::zeros(nc + 1);
    c_coeffs[0] = 1.0;

    // Iterative estimation
    let max_iter = 50;
    let tolerance = 1e-6;
    let mut prev_cost = f64::INFINITY;

    let mut a_coeffs = ar_init;
    let mut b_coeffs = b_init;

    for iter in 0..max_iter {
        // E-step: Estimate residuals
        let residuals =
            compute_armax_residuals(input, output, &a_coeffs, &b_coeffs, &c_coeffs, delay)?;

        // M-step: Update parameters
        let (new_a, new_b, new_c) =
            update_armax_parameters(input, output, &residuals, na, nb, nc, delay)?;

        // Compute cost
        let new_residuals = compute_armax_residuals(input, output, &new_a, &new_b, &new_c, delay)?;
        let cost = new_residuals.mapv(|r| r * r).sum() / n as f64;

        // Check convergence
        if (prev_cost - cost).abs() < tolerance {
            // Compute parameter statistics
            let params = concatenate_params(&new_a, &new_b, &new_c);
            let (covariance, std_errors, confidence_intervals) =
                compute_parameter_statistics(&params, &new_residuals, n)?;

            let parameter_estimate = ParameterEstimate {
                values: params,
                covariance,
                std_errors,
                confidence_intervals,
            };

            let model = SystemModel::ARMAX {
                a: new_a,
                b: new_b,
                c: new_c,
                delay,
            };

            return Ok((model, parameter_estimate, iter + 1, true, cost));
        }

        a_coeffs = new_a;
        b_coeffs = new_b;
        c_coeffs = new_c;
        prev_cost = cost;
    }

    // Return best estimate even if not converged
    let params = concatenate_params(&a_coeffs, &b_coeffs, &c_coeffs);
    let residuals = compute_armax_residuals(input, output, &a_coeffs, &b_coeffs, &c_coeffs, delay)?;
    let (covariance, std_errors, confidence_intervals) =
        compute_parameter_statistics(&params, &residuals, n)?;

    let parameter_estimate = ParameterEstimate {
        values: params,
        covariance,
        std_errors,
        confidence_intervals,
    };

    let model = SystemModel::ARMAX {
        a: a_coeffs,
        b: b_coeffs,
        c: c_coeffs,
        delay,
    };

    Ok((model, parameter_estimate, max_iter, false, prev_cost))
}

/// Output-Error model identification
///
/// OE: y(t) = B(q)/F(q) * u(t) + e(t)
#[allow(dead_code)]
pub fn identify_oe_complete(
    input: &Array1<f64>,
    output: &Array1<f64>,
    nb: usize,
    nf: usize,
    delay: usize,
) -> SignalResult<(SystemModel, ParameterEstimate, usize, bool, f64)> {
    let n = output.len();

    // Initialize using ARX model
    let (ar_init, b_init) = estimate_arx_ls(input, output, nf, nb, delay)?;

    // Convert AR to F polynomial (denominator)
    let mut f_coeffs = ar_init.clone();
    f_coeffs[0] = 1.0;

    // Nonlinear optimization using Gauss-Newton
    let max_iter = 100;
    let tolerance = 1e-6;
    let mut b_coeffs = b_init;
    let mut iterations = 0;
    let mut converged = false;

    for iter in 0..max_iter {
        // Simulate current model
        let y_sim = simulate_oe_model(input, &b_coeffs, &f_coeffs, delay)?;

        // Compute gradient and Hessian approximation
        let (gradient, hessian) =
            compute_oe_derivatives(input, &y_sim, output, &b_coeffs, &f_coeffs, delay)?;

        // Gauss-Newton update with Levenberg-Marquardt regularization
        let lambda = 0.01;
        let mut h_reg = hessian.clone();
        for i in 0..h_reg.nrows() {
            h_reg[[i, i]] += lambda;
        }

        let delta = solve(&h_reg, &gradient).unwrap_or_else(|_| gradient.clone() * 0.01);

        // Update parameters
        let alpha = backtracking_line_search(input, output, &b_coeffs, &f_coeffs, &delta, delay)?;

        update_oe_parameters(&mut b_coeffs, &mut f_coeffs, &delta, alpha);

        // Check convergence
        let delta_norm = (delta.mapv(|x| x * x).sum()).sqrt();
        if delta_norm < tolerance {
            converged = true;
            iterations = iter + 1;
            break;
        }
    }

    // Compute final statistics
    let y_final = simulate_oe_model(input, &b_coeffs, &f_coeffs, delay)?;
    let residuals = output - &y_final;
    let cost = residuals.mapv(|r| r * r).sum() / n as f64;

    let params = concatenate_params(&b_coeffs, &f_coeffs, &Array1::zeros(0));
    let (covariance, std_errors, confidence_intervals) =
        compute_parameter_statistics(&params, &residuals, n)?;

    let parameter_estimate = ParameterEstimate {
        values: params,
        covariance,
        std_errors,
        confidence_intervals,
    };

    let model = SystemModel::OE {
        b: b_coeffs,
        f: f_coeffs,
        delay,
    };

    Ok((model, parameter_estimate, iterations, converged, cost))
}

/// Box-Jenkins model identification
///
/// BJ: y(t) = B(q)/F(q) * u(t) + C(q)/D(q) * e(t)
#[allow(dead_code)]
pub fn identify_bj_complete(
    input: &Array1<f64>,
    output: &Array1<f64>,
    nb: usize,
    nc: usize,
    nd: usize,
    nf: usize,
    delay: usize,
) -> SignalResult<(SystemModel, ParameterEstimate, usize, bool, f64)> {
    let n = output.len();

    // Stage 1: Estimate B and F using OE model
    let (oe_model) = identify_oe_complete(input, output, nb, nf, delay)?;

    let (b_coeffs, f_coeffs) = match oe_model {
        SystemModel::OE { b, f, .. } => (b, f),
        _ => {
            return Err(SignalError::ComputationError(
                "OE model estimation failed".to_string(),
            ))
        }
    };

    // Stage 2: Estimate C and D from residuals
    let y_deterministic = simulate_oe_model(input, &b_coeffs, &f_coeffs, delay)?;
    let residuals = output - &y_deterministic;

    // Estimate ARMA model for residuals
    let (c_coeffs, d_coeffs) = estimate_arma_for_residuals(&residuals, nc, nd)?;

    // Stage 3: Joint refinement
    let max_iter = 50;
    let tolerance = 1e-6;
    let mut b = b_coeffs;
    let mut c = c_coeffs;
    let mut d = d_coeffs;
    let mut f = f_coeffs;

    for iter in 0..max_iter {
        // Update using prediction error method
        let (new_b, new_c, new_d, new_f) =
            update_bj_parameters(input, output, &b, &c, &d, &f, delay)?;

        // Check convergence
        let param_change = ((&new_b - &b).norm()
            + (&new_c - &c).norm()
            + (&new_d - &d).norm()
            + (&new_f - &f).norm())
            / 4.0;

        if param_change < tolerance {
            // Compute final statistics
            let y_final = simulate_bj_model(input, &new_b, &new_c, &new_d, &new_f, delay)?;
            let final_residuals = output - &y_final;
            let cost = final_residuals.mapv(|r| r * r).sum() / n as f64;

            let params = concatenate_bj_params(&new_b, &new_c, &new_d, &new_f);
            let (covariance, std_errors, confidence_intervals) =
                compute_parameter_statistics(&params, &final_residuals, n)?;

            let parameter_estimate = ParameterEstimate {
                values: params,
                covariance,
                std_errors,
                confidence_intervals,
            };

            let model = SystemModel::BJ {
                b: new_b,
                c: new_c,
                d: new_d,
                f: new_f,
                delay,
            };

            return Ok((model, parameter_estimate, iter + 1, true, cost));
        }

        b = new_b;
        c = new_c;
        d = new_d;
        f = new_f;
    }

    // Return best estimate
    let y_final = simulate_bj_model(input, &b, &c, &d, &f, delay)?;
    let final_residuals = output - &y_final;
    let cost = final_residuals.mapv(|r| r * r).sum() / n as f64;

    let params = concatenate_bj_params(&b, &c, &d, &f);
    let (covariance, std_errors, confidence_intervals) =
        compute_parameter_statistics(&params, &final_residuals, n)?;

    let parameter_estimate = ParameterEstimate {
        values: params,
        covariance,
        std_errors,
        confidence_intervals,
    };

    let model = SystemModel::BJ { b, c, d, f, delay };

    Ok((model, parameter_estimate, max_iter, false, cost))
}

/// State-space model identification using subspace methods
#[allow(dead_code)]
pub fn identify_state_space_complete(
    input: &Array1<f64>,
    output: &Array1<f64>,
    order: usize,
) -> SignalResult<(SystemModel, ParameterEstimate, usize, bool, f64)> {
    let n = output.len();

    // Use N4SID (Numerical Subspace State Space System IDentification)
    let (a, b, c, d, x0) = n4sid_algorithm(input, output, order)?;

    // Create state-space model
    let ss = StateSpace::new(
        a.clone().into_raw_vec(),
        b.clone().into_raw_vec(),
        c.clone().into_raw_vec(),
        d.clone().into_raw_vec(),
        Some(true), // Discrete-time
    )?;

    // Simulate to compute residuals
    let y_sim = simulate_state_space(&ss, input, &x0)?;
    let residuals = output - &y_sim;
    let cost = residuals.mapv(|r| r * r).sum() / n as f64;

    // Extract parameters for statistics
    let params = state_space_to_params(&a, &b, &c, &d);
    let (covariance, std_errors, confidence_intervals) =
        compute_parameter_statistics(&params, &residuals, n)?;

    let parameter_estimate = ParameterEstimate {
        values: params,
        covariance,
        std_errors,
        confidence_intervals,
    };

    let model = SystemModel::StateSpace(ss);

    Ok((model, parameter_estimate, 1, true, cost))
}

/// N4SID algorithm for subspace identification
#[allow(dead_code)]
fn n4sid_algorithm(
    input: &Array1<f64>,
    output: &Array1<f64>,
    order: usize,
) -> SignalResult<(
    Array2<f64>,
    Array2<f64>,
    Array2<f64>,
    Array2<f64>,
    Array1<f64>,
)> {
    let n = input.len();
    let block_size = order * 2;

    // Build block Hankel matrices
    let (u_hankel, y_hankel) = build_hankel_matrices(input, output, block_size)?;

    // Oblique projection
    let proj = oblique_projection(&y_hankel, &u_hankel)?;

    // SVD of projection
    let (u_svd, s, vt) = proj
        .svd(true, true)
        .map_err(|e| SignalError::ComputationError(format!("SVD failed: {}", e)))?;

    let u_svd = u_svd.unwrap();
    let vt = vt.unwrap();

    // Extract system matrices
    let u1 = u_svd.slice(s![.., 0..order]).to_owned();
    let s1 = Array2::from_diag(&s.slice(s![0..order]).to_owned());
    let v1 = vt.slice(s![0..order, ..]).to_owned();

    // Observability matrix
    let gamma = u1.dot(&s1.mapv(|x| x.sqrt()));

    // Extract A and C
    let c = gamma.slice(s![0..1, ..]).to_owned();
    let a = gamma
        .slice(s![0..gamma.nrows() - 1, ..])
        .to_owned()
        .solve(&gamma.slice(s![1.., ..]).to_owned())
        .map_err(|e| SignalError::ComputationError(format!("Failed to compute A matrix: {}", e)))?;

    // Estimate B and D using least squares
    let (b, d) = estimate_bd_matrices(input, output, &a, &c)?;

    // Initial state
    let x0 = Array1::zeros(order);

    Ok((a, b, c, d, x0))
}

/// Nonlinear ARX (NARX) model identification
#[allow(dead_code)]
pub fn identify_narx_complete(
    input: &Array1<f64>,
    output: &Array1<f64>,
    na: usize,
    nb: usize,
    delay: usize,
    nonlinearity: NonlinearFunction,
) -> SignalResult<(SystemModel, ParameterEstimate, usize, bool, f64)> {
    let n = output.len();

    // Build regression matrix with nonlinear terms
    let phi = build_narx_regression_matrix(input, output, na, nb, delay, &nonlinearity)?;

    // Least squares estimation
    let theta = phi.t().dot(&phi).solve(&phi.t().dot(output)).map_err(|e| {
        SignalError::ComputationError(format!("Failed to estimate NARX parameters: {}", e))
    })?;

    // Split parameters
    let n_linear = na + nb;
    let linear_params = theta.slice(s![0..n_linear]).to_owned();
    let nonlinear_params = theta.slice(s![n_linear..]).to_owned();

    // Compute residuals
    let y_pred = phi.dot(&theta);
    let residuals = output - &y_pred;
    let cost = residuals.mapv(|r| r * r).sum() / n as f64;

    // Parameter statistics
    let (covariance, std_errors, confidence_intervals) =
        compute_parameter_statistics(&theta, &residuals, n)?;

    let parameter_estimate = ParameterEstimate {
        values: theta,
        covariance,
        std_errors,
        confidence_intervals,
    };

    // Create Hammerstein-Wiener representation
    let linear_model = Box::new(SystemModel::ARX {
        a: linear_params.slice(s![0..na]).to_owned(),
        b: linear_params.slice(s![na..]).to_owned(),
        delay,
    });

    let model = SystemModel::HammersteinWiener {
        linear: linear_model,
        input_nonlinearity: nonlinearity.clone(),
        output_nonlinearity: nonlinearity,
    };

    Ok((model, parameter_estimate, 1, true, cost))
}

// Helper functions

/// Estimate ARX model using least squares
#[allow(dead_code)]
fn estimate_arx_ls(
    input: &Array1<f64>,
    output: &Array1<f64>,
    na: usize,
    nb: usize,
    delay: usize,
) -> SignalResult<(Array1<f64>, Array1<f64>, f64)> {
    let n = output.len();
    let n_start = na.max(nb + delay - 1);

    // Build regression matrix
    let n_samples = n - n_start;
    let mut phi = Array2::zeros((n_samples, na + nb));
    let mut y = Array1::zeros(n_samples);

    for i in 0..n_samples {
        let t = i + n_start;

        // AR terms
        for j in 0..na {
            phi[[i, j]] = -output[t - j - 1];
        }

        // Input terms
        for j in 0..nb {
            if t >= delay + j {
                phi[[i, na + j]] = input[t - delay - j];
            }
        }

        y[i] = output[t];
    }

    // Least squares
    let params = phi
        .t()
        .dot(&phi)
        .solve(&phi.t().dot(&y))
        .map_err(|e| SignalError::ComputationError(format!("ARX estimation failed: {}", e)))?;

    let mut a = Array1::zeros(na + 1);
    a[0] = 1.0;
    a.slice_mut(s![1..]).assign(&params.slice(s![0..na]));

    let b = params.slice(s![na..]).to_owned();

    let residuals = &y - &phi.dot(&params);
    let variance = residuals.mapv(|r| r * r).sum() / n_samples as f64;

    Ok((a, b, variance))
}

/// Compute ARMAX residuals
#[allow(dead_code)]
fn compute_armax_residuals(
    input: &Array1<f64>,
    output: &Array1<f64>,
    a: &Array1<f64>,
    b: &Array1<f64>,
    c: &Array1<f64>,
    delay: usize,
) -> SignalResult<Array1<f64>> {
    let n = output.len();
    let na = a.len() - 1;
    let nb = b.len();
    let nc = c.len() - 1;

    let mut residuals = Array1::zeros(n);

    for t in 0..n {
        let mut y_pred = 0.0;

        // AR part
        for i in 1..=na.min(t) {
            y_pred -= a[i] * output[t - i];
        }

        // X part
        for i in 0..nb.min(t + 1 - delay) {
            if t >= delay + i {
                y_pred += b[i] * input[t - delay - i];
            }
        }

        // MA part
        for i in 1..=nc.min(t) {
            y_pred += c[i] * residuals[t - i];
        }

        residuals[t] = output[t] - y_pred;
    }

    Ok(residuals)
}

/// Update ARMAX parameters
#[allow(dead_code)]
fn update_armax_parameters(
    input: &Array1<f64>,
    output: &Array1<f64>,
    residuals: &Array1<f64>,
    na: usize,
    nb: usize,
    nc: usize,
    delay: usize,
) -> SignalResult<(Array1<f64>, Array1<f64>, Array1<f64>)> {
    let n = output.len();
    let n_start = (na.max(nb + delay - 1)).max(nc);
    let n_samples = n - n_start;

    // Build extended regression matrix
    let mut phi = Array2::zeros((n_samples, na + nb + nc));
    let mut y = Array1::zeros(n_samples);

    for i in 0..n_samples {
        let t = i + n_start;
        y[i] = output[t];

        // AR terms
        for j in 0..na {
            phi[[i, j]] = -output[t - j - 1];
        }

        // Input terms
        for j in 0..nb {
            if t >= delay + j {
                phi[[i, na + j]] = input[t - delay - j];
            }
        }

        // MA terms
        for j in 0..nc {
            phi[[i, na + nb + j]] = residuals[t - j - 1];
        }
    }

    // Weighted least squares
    let params = phi
        .t()
        .dot(&phi)
        .solve(&phi.t().dot(&y))
        .map_err(|e| SignalError::ComputationError(format!("ARMAX update failed: {}", e)))?;

    let mut a = Array1::zeros(na + 1);
    a[0] = 1.0;
    a.slice_mut(s![1..]).assign(&params.slice(s![0..na]));

    let b = params.slice(s![na..na + nb]).to_owned();

    let mut c = Array1::zeros(nc + 1);
    c[0] = 1.0;
    c.slice_mut(s![1..]).assign(&params.slice(s![na + nb..]));

    Ok((a, b, c))
}

/// Simulate OE model
#[allow(dead_code)]
fn simulate_oe_model(
    input: &Array1<f64>,
    b: &Array1<f64>,
    f: &Array1<f64>,
    delay: usize,
) -> SignalResult<Array1<f64>> {
    let n = input.len();
    let nb = b.len();
    let nf = f.len() - 1;

    let mut y = Array1::zeros(n);
    let mut w = Array1::zeros(n); // Intermediate signal

    for t in 0..n {
        // Compute w(t) = B(q)u(t)
        for i in 0..nb.min(t + 1 - delay) {
            if t >= delay + i {
                w[t] += b[i] * input[t - delay - i];
            }
        }

        // Compute y(t) from w(t)/F(q)
        for i in 1..=nf.min(t) {
            w[t] -= f[i] * w[t - i];
        }

        y[t] = w[t];
    }

    Ok(y)
}

/// Compute OE model derivatives
#[allow(dead_code)]
fn compute_oe_derivatives(
    input: &Array1<f64>,
    y_sim: &Array1<f64>,
    output: &Array1<f64>,
    b: &Array1<f64>,
    f: &Array1<f64>,
    delay: usize,
) -> SignalResult<(Array1<f64>, Array2<f64>)> {
    let n = input.len();
    let nb = b.len();
    let nf = f.len() - 1;
    let n_params = nb + nf;

    let mut gradient = Array1::zeros(n_params);
    let mut hessian = Array2::zeros((n_params, n_params));

    // Compute sensitivities using finite differences (simplified)
    let h = 1e-6;

    for i in 0..n_params {
        let mut b_plus = b.clone();
        let mut f_plus = f.clone();

        if i < nb {
            b_plus[i] += h;
        } else {
            f_plus[i - nb + 1] += h;
        }

        let y_plus = simulate_oe_model(input, &b_plus, &f_plus, delay)?;
        let sensitivity = (&y_plus - y_sim) / h;

        // Gradient
        let error = y_sim - output;
        gradient[i] = 2.0 * error.dot(&sensitivity) / n as f64;

        // Hessian approximation (Gauss-Newton)
        for j in i..n_params {
            let mut b_plus2 = b.clone();
            let mut f_plus2 = f.clone();

            if j < nb {
                b_plus2[j] += h;
            } else {
                f_plus2[j - nb + 1] += h;
            }

            let y_plus2 = simulate_oe_model(input, &b_plus2, &f_plus2, delay)?;
            let sensitivity2 = (&y_plus2 - y_sim) / h;

            hessian[[i, j]] = 2.0 * sensitivity.dot(&sensitivity2) / n as f64;
            hessian[[j, i]] = hessian[[i, j]];
        }
    }

    Ok((gradient, hessian))
}

/// Backtracking line search
#[allow(dead_code)]
fn backtracking_line_search(
    input: &Array1<f64>,
    output: &Array1<f64>,
    b: &Array1<f64>,
    f: &Array1<f64>,
    delta: &Array1<f64>,
    delay: usize,
) -> SignalResult<f64> {
    let mut alpha = 1.0;
    let beta = 0.5;
    let c = 0.1;

    let y_current = simulate_oe_model(input, b, f, delay)?;
    let cost_current = (output - &y_current).mapv(|r| r * r).sum();

    let nb = b.len();

    loop {
        let mut b_new = b.clone();
        let mut f_new = f.clone();

        for i in 0..nb {
            b_new[i] += alpha * delta[i];
        }
        for i in 0..f.len() - 1 {
            f_new[i + 1] += alpha * delta[nb + i];
        }

        let y_new = simulate_oe_model(input, &b_new, &f_new, delay)?;
        let cost_new = (output - &y_new).mapv(|r| r * r).sum();

        if cost_new <= cost_current - c * alpha * delta.dot(delta) {
            break;
        }

        alpha *= beta;

        if alpha < 1e-8 {
            break;
        }
    }

    Ok(alpha)
}

/// Update OE parameters
#[allow(dead_code)]
fn update_oe_parameters(b: &mut Array1<f64>, f: &mut Array1<f64>, delta: &Array1<f64>, alpha: f64) {
    let nb = b.len();

    for i in 0..nb {
        b[i] += alpha * delta[i];
    }

    for i in 0..f.len() - 1 {
        f[i + 1] += alpha * delta[nb + i];
    }
}

/// Estimate ARMA model for residuals
#[allow(dead_code)]
fn estimate_arma_for_residuals(
    residuals: &Array1<f64>,
    nc: usize,
    nd: usize,
) -> SignalResult<(Array1<f64>, Array1<f64>)> {
    // Use Durbin-Levinson for initial estimate
    let mut c = Array1::zeros(nc + 1);
    let mut d = Array1::zeros(nd + 1);
    c[0] = 1.0;
    d[0] = 1.0;

    // Simplified: use AR approximation
    let r = compute_autocorrelation(residuals, nc.max(nd) + 1)?;

    // Yule-Walker for AR part (D polynomial)
    if nd > 0 {
        let mut r_matrix = Array2::zeros((nd, nd));
        for i in 0..nd {
            for j in 0..nd {
                r_matrix[[i, j]] = r[(i as i32 - j as i32).abs() as usize];
            }
        }

        let r_vec = Array1::from_vec(r[1..nd + 1].to_vec());
        let ar_params = r_matrix.solve(&r_vec).unwrap_or_else(|_| Array1::zeros(nd));

        for i in 0..nd {
            d[i + 1] = -ar_params[i];
        }
    }

    // MA part estimation (simplified)
    if nc > 0 {
        for i in 1..=nc {
            c[i] = r[i] / r[0] * 0.5; // Simplified estimate
        }
    }

    Ok((c, d))
}

/// Simulate Box-Jenkins model
#[allow(dead_code)]
fn simulate_bj_model(
    input: &Array1<f64>,
    b: &Array1<f64>,
    c: &Array1<f64>,
    d: &Array1<f64>,
    f: &Array1<f64>,
    delay: usize,
) -> SignalResult<Array1<f64>> {
    let n = input.len();

    // Deterministic part: B/F
    let y_det = simulate_oe_model(input, b, f, delay)?;

    // Stochastic part: C/D (simplified as filtered white noise)
    let mut noise = Array1::zeros(n);
    let mut filtered_noise = Array1::zeros(n);

    // Generate innovations (would be estimated in practice)
    let mut rng = rand::rng();
    for i in 0..n {
        noise[i] = rng.gen_range(-0.1..0.1);
    }

    // Filter through C/D
    let nc = c.len() - 1;
    let nd = d.len() - 1;

    for t in 0..n {
        // MA part
        for i in 0..=nc.min(t) {
            filtered_noise[t] += c[i] * noise[t - i];
        }

        // AR part
        for i in 1..=nd.min(t) {
            filtered_noise[t] -= d[i] * filtered_noise[t - i];
        }
    }

    Ok(y_det + filtered_noise)
}

/// Update Box-Jenkins parameters
#[allow(dead_code)]
fn update_bj_parameters(
    input: &Array1<f64>,
    output: &Array1<f64>,
    b: &Array1<f64>,
    c: &Array1<f64>,
    d: &Array1<f64>,
    f: &Array1<f64>,
    delay: usize,
) -> SignalResult<(Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>)> {
    // Simplified: alternate between updating B/F and C/D

    // Update B/F holding C/D fixed
    let y_noise = simulate_noise_contribution(output, c, d)?;
    let y_clean = output - &y_noise;

    let (_, new_b, new_f) = {
        let (oe_model) = identify_oe_complete(input, &y_clean, b.len(), f.len() - 1, delay)?;

        match oe_model {
            SystemModel::OE { b, f, .. } => ((), b, f),
            _ => ((), vec![], vec![]),
        }
    };

    // Update C/D holding B/F fixed
    let y_det = simulate_oe_model(input, &new_b, &new_f, delay)?;
    let residuals = output - &y_det;
    let (new_c, new_d) = estimate_arma_for_residuals(&residuals, c.len() - 1, d.len() - 1)?;

    Ok((new_b, new_c, new_d, new_f))
}

/// Build Hankel matrices for subspace identification
#[allow(dead_code)]
fn build_hankel_matrices(
    input: &Array1<f64>,
    output: &Array1<f64>,
    block_size: usize,
) -> SignalResult<(Array2<f64>, Array2<f64>)> {
    let n = input.len();
    let n_cols = n - 2 * block_size + 1;

    if n_cols <= 0 {
        return Err(SignalError::ValueError(
            "Signal too short for block _size".to_string(),
        ));
    }

    let mut u_hankel = Array2::zeros((block_size, n_cols));
    let mut y_hankel = Array2::zeros((block_size, n_cols));

    for i in 0..block_size {
        for j in 0..n_cols {
            u_hankel[[i, j]] = input[i + j];
            y_hankel[[i, j]] = output[i + j];
        }
    }

    Ok((u_hankel, y_hankel))
}

/// Oblique projection for subspace identification
#[allow(dead_code)]
fn oblique_projection(
    _y_hankel: &Array2<f64>,
    u_hankel: &Array2<f64>,
) -> SignalResult<Array2<f64>> {
    // Simplified: orthogonal projection
    let u_pinv = compute_pseudoinverse(u_hankel)?;
    let proj_perp_u = Array2::eye(u_hankel.ncols()) - u_hankel.t().dot(&u_pinv.t());

    Ok(_y_hankel.dot(&proj_perp_u))
}

/// Estimate B and D matrices for state space
#[allow(dead_code)]
fn estimate_bd_matrices(
    input: &Array1<f64>,
    output: &Array1<f64>,
    a: &Array2<f64>,
    c: &Array2<f64>,
) -> SignalResult<(Array2<f64>, Array2<f64>)> {
    let n = input.len();
    let nx = a.nrows();

    // Simulate state trajectory
    let mut x = Array2::zeros((nx, n));

    for t in 1..n {
        let x_prev = x.column(t - 1);
        let x_curr = a.dot(&x_prev);
        x.column_mut(t).assign(&x_curr);
    }

    // Build regression for B and D estimation
    let mut phi = Array2::zeros((n, nx + 1));
    let mut y = Array1::zeros(n);

    for t in 0..n {
        phi[[t, 0]] = input[t]; // D matrix term
        for i in 0..nx {
            phi[[t, i + 1]] = x[[i, t]]; // B matrix terms
        }
        y[t] = output[t] - c.dot(&x.column(t))[0];
    }

    let params = phi
        .t()
        .dot(&phi)
        .solve(&phi.t().dot(&y))
        .map_err(|e| SignalError::ComputationError(format!("Failed to estimate B,D: {}", e)))?;

    let d = Array2::from_elem((1, 1), params[0]);
    let b = params.slice(s![1..]).to_owned().insert_axis(Axis(1));

    Ok((b, d))
}

/// Build NARX regression matrix
#[allow(dead_code)]
fn build_narx_regression_matrix(
    input: &Array1<f64>,
    output: &Array1<f64>,
    na: usize,
    nb: usize,
    delay: usize,
    nonlinearity: &NonlinearFunction,
) -> SignalResult<Array2<f64>> {
    let n = output.len();
    let n_start = na.max(nb + delay - 1);
    let n_samples = n - n_start;

    // Determine number of nonlinear terms
    let n_nonlinear = match nonlinearity {
        NonlinearFunction::Polynomial(coeffs) => coeffs.len() - 1,
        NonlinearFunction::PiecewiseLinear { breakpoints, .. } => breakpoints.len() + 1,
        _ => 10, // Default number of basis functions
    };

    let n_features = na + nb + n_nonlinear;
    let mut phi = Array2::zeros((n_samples, n_features));

    for i in 0..n_samples {
        let t = i + n_start;

        // Linear AR terms
        for j in 0..na {
            phi[[i, j]] = output[t - j - 1];
        }

        // Linear input terms
        for j in 0..nb {
            if t >= delay + j {
                phi[[i, na + j]] = input[t - delay - j];
            }
        }

        // Nonlinear terms
        match nonlinearity {
            NonlinearFunction::Polynomial(coeffs) => {
                let x = output[t - 1]; // Use previous output
                for (k, &coeff) in coeffs.iter().enumerate().skip(1) {
                    phi[[i, na + nb + k - 1]] = coeff * x.powi(k as i32);
                }
            }
            NonlinearFunction::Sigmoid { scale, offset } => {
                let x = output[t - 1];
                phi[[i, na + nb]] = 1.0 / (1.0 + (-scale * (x - offset)).exp());
            }
            _ => {
                // Other nonlinearities: use radial basis functions
                for k in 0..n_nonlinear {
                    let center = -2.0 + 4.0 * k as f64 / n_nonlinear as f64;
                    let width = 0.5;
                    let x = output[t - 1];
                    phi[[i, na + nb + k]] = (-(x - center).powi(2) / (2.0 * width * width)).exp();
                }
            }
        }
    }

    Ok(phi)
}

/// Helper function to concatenate parameters
#[allow(dead_code)]
fn concatenate_params(a: &Array1<f64>, b: &Array1<f64>, c: &Array1<f64>) -> Array1<f64> {
    let mut params = Array1::zeros(a.len() + b.len() + c.len() - 2);
    params
        .slice_mut(s![0..a.len() - 1])
        .assign(&a.slice(s![1..]));
    params
        .slice_mut(s![a.len() - 1..a.len() - 1 + b.len()])
        .assign(b);
    params
        .slice_mut(s![a.len() - 1 + b.len()..])
        .assign(&c.slice(s![1..]));
    params
}

/// Helper function to concatenate BJ parameters
#[allow(dead_code)]
fn concatenate_bj_params(
    b: &Array1<f64>,
    c: &Array1<f64>,
    d: &Array1<f64>,
    f: &Array1<f64>,
) -> Array1<f64> {
    let total_len = b.len() + c.len() + d.len() + f.len() - 3;
    let mut params = Array1::zeros(total_len);
    let mut offset = 0;

    params.slice_mut(s![offset..offset + b.len()]).assign(b);
    offset += b.len();

    params
        .slice_mut(s![offset..offset + c.len() - 1])
        .assign(&c.slice(s![1..]));
    offset += c.len() - 1;

    params
        .slice_mut(s![offset..offset + d.len() - 1])
        .assign(&d.slice(s![1..]));
    offset += d.len() - 1;

    params.slice_mut(s![offset..]).assign(&f.slice(s![1..]));

    params
}

/// Convert state-space matrices to parameter vector
#[allow(dead_code)]
fn state_space_to_params(
    a: &Array2<f64>,
    b: &Array2<f64>,
    c: &Array2<f64>,
    d: &Array2<f64>,
) -> Array1<f64> {
    let mut params = Vec::new();

    // Flatten matrices
    params.extend(a.iter().cloned());
    params.extend(b.iter().cloned());
    params.extend(c.iter().cloned());
    params.extend(d.iter().cloned());

    Array1::from_vec(params)
}

/// Compute parameter statistics
#[allow(dead_code)]
fn compute_parameter_statistics(
    params: &Array1<f64>,
    residuals: &Array1<f64>,
    n: usize,
) -> SignalResult<(Array2<f64>, Array1<f64>, Vec<(f64, f64)>)> {
    let k = params.len();
    let sigma2 = residuals.mapv(|r| r * r).sum() / (n - k) as f64;

    // Simplified: assume diagonal covariance
    let covariance = Array2::eye(k) * sigma2;
    let std_errors = Array1::from_shape_fn(k, |i| covariance[[i, i]].sqrt());

    let confidence_intervals: Vec<(f64, f64)> = params
        .iter()
        .zip(std_errors.iter())
        .map(|(&p, &se)| (p - 1.96 * se, p + 1.96 * se))
        .collect();

    Ok((covariance, std_errors, confidence_intervals))
}

/// Compute autocorrelation
#[allow(dead_code)]
fn compute_autocorrelation(_signal: &Array1<f64>, maxlag: usize) -> SignalResult<Vec<f64>> {
    let n = signal.len();
    let mean = signal.mean().unwrap_or(0.0);
    let centered = _signal - mean;

    let mut r = vec![0.0; max_lag];

    for k in 0..max_lag {
        let mut sum = 0.0;
        for i in 0..(n - k) {
            sum += centered[i] * centered[i + k];
        }
        r[k] = sum / n as f64;
    }

    Ok(r)
}

/// Compute pseudoinverse
#[allow(dead_code)]
fn compute_pseudoinverse(matrix: &Array2<f64>) -> SignalResult<Array2<f64>> {
    let (u, s, vt) = _matrix
        .svd(true, true)
        .map_err(|e| SignalError::ComputationError(format!("SVD failed: {}", e)))?;

    let u = u.unwrap();
    let vt = vt.unwrap();

    let tolerance = 1e-10;
    let mut s_inv = Array2::zeros((vt.nrows(), u.ncols()));

    for i in 0..s.len() {
        if s[i] > tolerance {
            s_inv[[i, i]] = 1.0 / s[i];
        }
    }

    Ok(vt.t().dot(&s_inv).dot(&u.t()))
}

/// Simulate state-space model
#[allow(dead_code)]
fn simulate_state_space(
    ss: &StateSpace,
    input: &Array1<f64>,
    x0: &Array1<f64>,
) -> SignalResult<Array1<f64>> {
    let n = input.len();
    let nx = ss.a.nrows();

    let mut x = x0.clone();
    let mut y = Array1::zeros(n);

    for t in 0..n {
        // Output equation
        y[t] = ss.c.dot(&x)[0] + ss.d[[0, 0]] * input[t];

        // State update
        x = ss.a.dot(&x) + ss.b.column(0) * input[t];
    }

    Ok(y)
}

/// Simulate noise contribution for BJ model
#[allow(dead_code)]
fn simulate_noise_contribution(
    output: &Array1<f64>,
    _c: &Array1<f64>,
    _d: &Array1<f64>,
) -> SignalResult<Array1<f64>> {
    // This is a placeholder - in practice would estimate the noise sequence
    Ok(Array1::zeros(output.len()))
}

mod tests {

    #[test]
    fn test_armax_identification() {
        let n = 200;
        let mut input = Array1::zeros(n);
        let mut output = Array1::zeros(n);

        // Generate test data
        let mut rng = rand::rng();
        for i in 2..n {
            input[i] = rng.gen_range(-1.0..1.0);
            output[i] = 0.7 * output[i - 1] - 0.2 * output[i - 2]
                + 0.5 * input[i - 1]
                + 0.3 * input[i - 2]
                + 0.1 * rng.gen_range(-1.0..1.0);
        }

        let (model__, converged_) = identify_armax_complete(&input, &output, 2, 2, 1, 1).unwrap();

        assert!(converged);
        assert!(matches!(model, SystemModel::ARMAX { .. }));
    }

    #[test]
    fn test_state_space_identification() {
        let n = 100;
        let input = Array1::from_shape_fn(n, |i| (i as f64 * 0.1).sin());
        let output = Array1::from_shape_fn(n, |i| (i as f64 * 0.1 + 0.5).sin());

        let (model__, converged_) = identify_state_space_complete(&input, &output, 2).unwrap();

        assert!(converged);
        assert!(matches!(model, SystemModel::StateSpace(_)));
    }
}
