//! N4SID (Numerical algorithms for Subspace State Space System IDentification)
//!
//! Implements the classic N4SID algorithm for state-space identification:
//!
//! 1. Construct block Hankel matrices from input-output data
//! 2. Compute oblique projection using QR decomposition
//! 3. Determine state order via SVD and singular value gap
//! 4. Extract state sequence and estimate (A, B, C, D) matrices
//!
//! Reference: Van Overschee & De Moor, "N4SID: Subspace algorithms for the
//! identification of combined deterministic-stochastic systems", Automatica, 1994.

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{s, Array1, Array2, Axis};

use super::types::{compute_fit_percentage, N4sidConfig, SubspaceIdResult};

/// Estimate a state-space model using the N4SID subspace algorithm
///
/// # Arguments
/// * `y` - Output signal (single output)
/// * `u` - Input signal (single input)
/// * `config` - N4SID configuration
///
/// # Returns
/// * `SubspaceIdResult` with estimated (A, B, C, D) matrices
///
/// # Example
/// ```rust
/// use scirs2_signal::system_identification::{n4sid_estimate, N4sidConfig};
/// use scirs2_core::ndarray::Array1;
///
/// let n = 500;
/// let mut u = Array1::<f64>::zeros(n);
/// let mut y = Array1::<f64>::zeros(n);
/// for i in 0..n { u[i] = ((i as f64) * 0.1).sin(); }
/// for i in 1..n { y[i] = 0.8 * y[i - 1] + 0.5 * u[i - 1]; }
///
/// let config = N4sidConfig {
///     state_order: Some(1),
///     block_rows: 5,
///     sv_threshold: 0.01,
/// };
/// let result = n4sid_estimate(&y, &u, &config).expect("N4SID failed");
/// assert_eq!(result.state_order, 1);
/// ```
pub fn n4sid_estimate(
    y: &Array1<f64>,
    u: &Array1<f64>,
    config: &N4sidConfig,
) -> SignalResult<SubspaceIdResult> {
    let n = y.len();
    if n != u.len() {
        return Err(SignalError::DimensionMismatch(
            "Input and output must have the same length".into(),
        ));
    }

    let i = config.block_rows; // block row dimension

    if n < 2 * i + 1 {
        return Err(SignalError::ValueError(
            "Insufficient data for the specified block_rows parameter".into(),
        ));
    }

    let j = n - 2 * i + 1; // number of columns in Hankel matrix

    if j < 2 * i + 1 {
        return Err(SignalError::ValueError(
            "Insufficient data for the specified block_rows parameter".into(),
        ));
    }

    // Step 1: Construct block Hankel matrices
    // U_past (i x j), U_future (i x j), Y_past (i x j), Y_future (i x j)
    // For SISO: each "block row" is 1 row
    let mut u_past = Array2::<f64>::zeros((i, j));
    let mut u_future = Array2::<f64>::zeros((i, j));
    let mut y_past = Array2::<f64>::zeros((i, j));
    let mut y_future = Array2::<f64>::zeros((i, j));

    for col in 0..j {
        for row in 0..i {
            u_past[[row, col]] = u[col + row];
            u_future[[row, col]] = u[col + i + row];
            y_past[[row, col]] = y[col + row];
            y_future[[row, col]] = y[col + i + row];
        }
    }

    // Step 2: Build the combined data matrix and compute via QR
    // Stack: [U_future; Y_future; U_past; Y_past] -> (4i x j)
    let total_rows = 4 * i;
    let mut data_matrix = Array2::<f64>::zeros((total_rows, j));
    for row in 0..i {
        for col in 0..j {
            data_matrix[[row, col]] = u_future[[row, col]];
            data_matrix[[i + row, col]] = y_future[[row, col]];
            data_matrix[[2 * i + row, col]] = u_past[[row, col]];
            data_matrix[[3 * i + row, col]] = y_past[[row, col]];
        }
    }

    // Compute R = data_matrix * data_matrix^T / j (covariance-like)
    let data_t = data_matrix.t();

    // Step 3: Compute the oblique projection
    // The extended observability matrix is found from the SVD of a weighted
    // projection of Y_future onto the past data space, orthogonal to U_future.
    //
    // Simplified approach: compute the projection P = Y_f * [U_p; Y_p]^T * pinv([U_p; Y_p])
    // orthogonal to U_f

    // Build W_p = [U_past; Y_past] (2i x j)
    let mut w_past = Array2::<f64>::zeros((2 * i, j));
    for row in 0..i {
        for col in 0..j {
            w_past[[row, col]] = u_past[[row, col]];
            w_past[[i + row, col]] = y_past[[row, col]];
        }
    }

    // Oblique projection: first remove U_future effect
    // Compute projection of Y_future onto W_past, perpendicular to U_future
    // Using least squares approach

    // Regress Y_future on [U_future; W_past]
    let combined_rows = i + 2 * i;
    let mut combined = Array2::<f64>::zeros((combined_rows, j));
    for row in 0..i {
        for col in 0..j {
            combined[[row, col]] = u_future[[row, col]];
        }
    }
    for row in 0..(2 * i) {
        for col in 0..j {
            combined[[i + row, col]] = w_past[[row, col]];
        }
    }

    // Solve: Y_future = [Theta_u | Theta_wp] * [U_future; W_past]
    // Using transpose formulation: Y_f^T = X^T * Theta^T where X = combined
    // So Theta^T = pinv(X^T) * Y_f^T
    // Least squares in the (j x combined_rows) overdetermined form, solved from a
    // rank-truncated SVD. `combined` is rank-deficient for any input that is not
    // persistently exciting of order 2i, so the minimum-norm solution is what keeps
    // the oblique projection meaningful; see `pseudoinverse_product`.
    let theta_t = pseudoinverse_product(&combined.t().to_owned(), &y_future.t().to_owned())?;
    debug_assert_eq!(theta_t.dim(), (combined_rows, i));

    // The oblique projection is Theta_wp * W_past where Theta_wp = theta_t[i..,:]
    let theta_wp = theta_t.slice(s![i.., ..]).to_owned(); // (2i x i)
    let obl_proj = theta_wp.t().dot(&w_past); // (i x j)

    // Step 4: SVD of the oblique projection to get observability matrix
    let (u_svd, s_svd, vt_svd) = svd_thin(&obl_proj)?;

    let sv = Array1::from_vec(s_svd.clone());

    // Determine state order
    let state_order = match config.state_order {
        Some(order) => order,
        None => {
            // Automatic order selection: find the gap in singular values
            let mut order = 1;
            if s_svd.len() > 1 {
                let max_sv = s_svd[0].max(1e-30);
                for k in 1..s_svd.len() {
                    if s_svd[k] / max_sv < config.sv_threshold {
                        break;
                    }
                    order = k + 1;
                }
            }
            order.min(i - 1).max(1)
        }
    };

    if state_order > i.saturating_sub(1) {
        return Err(SignalError::ValueError(
            "State order exceeds maximum allowed by block_rows".into(),
        ));
    }

    // Step 5: Extract extended observability matrix O_i = U_n * S_n^{1/2}
    let obs_rows = u_svd.nrows();
    let mut obs_matrix = Array2::<f64>::zeros((obs_rows, state_order));
    for row_idx in 0..obs_rows {
        for col_idx in 0..state_order {
            if col_idx < u_svd.ncols() {
                obs_matrix[[row_idx, col_idx]] = u_svd[[row_idx, col_idx]] * s_svd[col_idx].sqrt();
            }
        }
    }

    // C = first block row of O_i (for SISO: first row)
    let c_mat = obs_matrix.slice(s![0..1, ..]).to_owned(); // (1 x state_order)

    // A = pinv(O_{i-1}) * O_{shifted} where O_{shifted} = O_i without last block row
    let a_mat = if state_order > 0 && obs_rows > 1 {
        let obs_up = obs_matrix.slice(s![0..obs_rows - 1, ..]).to_owned();
        let obs_down = obs_matrix.slice(s![1..obs_rows, ..]).to_owned();
        pseudoinverse_product(&obs_up, &obs_down)?
    } else {
        Array2::<f64>::zeros((state_order, state_order))
    };

    // Step 6: Estimate B and D from input-output equation
    // y(t) = C * x(t) + D * u(t)
    // x(t+1) = A * x(t) + B * u(t)
    // Use least squares with the estimated state sequence

    // Recover state sequence directly from SVD: X = S_n^{1/2} * V_n^T
    // This is numerically more stable than computing pinv(O) * obl_proj
    let vt_cols = vt_svd.ncols();
    let mut state_seq = Array2::<f64>::zeros((state_order, vt_cols));
    for k in 0..state_order {
        let scale = s_svd[k].sqrt();
        for col_idx in 0..vt_cols {
            state_seq[[k, col_idx]] = scale * vt_svd[[k, col_idx]];
        }
    }
    let j = vt_cols; // update j to match actual column count from SVD

    // Estimate B and D separately for better numerical stability
    // B: from x(t+1) - A*x(t) = B*u(t)
    // D: from y(t) - C*x(t) = D*u(t)
    let valid_cols = j.saturating_sub(1).min(n.saturating_sub(i + 1));
    if valid_cols < state_order + 2 {
        let b_mat = Array2::<f64>::zeros((state_order, 1));
        let d_mat = Array2::<f64>::zeros((1, 1));
        return Ok(SubspaceIdResult {
            a: a_mat,
            b: b_mat,
            c: c_mat,
            d: d_mat,
            state_order,
            singular_values: sv,
            fit_percentage: 0.0,
            noise_variance: f64::INFINITY,
        });
    }

    // Estimate B from: x(t+1) - A*x(t) = B*u(t)
    // Build residual r(t) = x(t+1) - A*x(t) and u(t) vectors
    let mut u_vec = Array1::<f64>::zeros(valid_cols);
    let mut x_residual = Array2::<f64>::zeros((state_order, valid_cols));
    for col_idx in 0..valid_cols {
        u_vec[col_idx] = u[col_idx + i];
        for k in 0..state_order {
            let mut ax = 0.0;
            for l in 0..state_order {
                ax += a_mat[[k, l]] * state_seq[[l, col_idx]];
            }
            x_residual[[k, col_idx]] = state_seq[[k, col_idx + 1]] - ax;
        }
    }

    // B = x_residual * u^T / (u * u^T)  (least squares for each row of B)
    let utu: f64 = u_vec.dot(&u_vec);
    let mut b_mat_est = Array2::<f64>::zeros((state_order, 1));
    if utu.abs() > 1e-30 {
        for k in 0..state_order {
            let row = x_residual.row(k);
            b_mat_est[[k, 0]] = row.dot(&u_vec) / utu;
        }
    }

    // Estimate D from: y(t) - C*x(t) = D*u(t)
    let mut y_residual = Array1::<f64>::zeros(valid_cols);
    for col_idx in 0..valid_cols {
        let mut cx = 0.0;
        for k in 0..state_order {
            cx += c_mat[[0, k]] * state_seq[[k, col_idx]];
        }
        y_residual[col_idx] = y[col_idx + i] - cx;
    }
    let mut d_mat_est = Array2::<f64>::zeros((1, 1));
    if utu.abs() > 1e-30 {
        d_mat_est[[0, 0]] = y_residual.dot(&u_vec) / utu;
    }

    // Use observability-based A (more reliable from SVD) for the final model
    let fit_pct = compute_simulation_fit(y, u, &a_mat, &b_mat_est, &c_mat, &d_mat_est);
    let noise_var = compute_simulation_noise_var(y, u, &a_mat, &b_mat_est, &c_mat, &d_mat_est);

    Ok(SubspaceIdResult {
        a: a_mat,
        b: b_mat_est,
        c: c_mat,
        d: d_mat_est,
        state_order,
        singular_values: sv,
        fit_percentage: fit_pct,
        noise_variance: noise_var,
    })
}

/// Thin SVD: returns (U, sigma, V^T) where U is (m x min(m,n)).
///
/// `full_matrices` must stay `false`. The oblique projection is a very wide
/// `i x j` matrix (8 x 485 for the first-order test), and asking for the full
/// decomposition materialises a `j x j` V — 485x485 here, which costs seconds
/// while every column past the first `min(i, j)` is discarded by the caller.
fn svd_thin(matrix: &Array2<f64>) -> SignalResult<(Array2<f64>, Vec<f64>, Array2<f64>)> {
    match scirs2_linalg::svd(&matrix.view(), false, None) {
        Ok((u, s_vec, vt)) => {
            let sigma: Vec<f64> = s_vec.iter().copied().collect();
            Ok((u, sigma, vt))
        }
        Err(_) => Err(SignalError::ComputationError(
            "SVD computation failed".into(),
        )),
    }
}

/// Minimum-norm least-squares solve `pinv(A) * B`, from a thin SVD with
/// relative rank truncation.
///
/// This deliberately avoids the normal equations (`solve(A^T A, A^T B)`). That
/// form squares the condition number, and the regression N4SID performs here is
/// genuinely rank-deficient whenever the input is not persistently exciting of
/// order `2i` — a single sinusoid, the most common smoke-test input, excites
/// only two directions no matter how long the record. `A^T A` is then singular,
/// and `solve` does not report that: LU with a near-zero pivot happily returns
/// an arbitrary huge-norm solution, so the `Err` fallback never fires and the
/// oblique projection silently becomes noise. Truncating at `tol` and taking the
/// minimum-norm solution keeps the projection well defined for such data.
fn pseudoinverse_product(a: &Array2<f64>, b: &Array2<f64>) -> SignalResult<Array2<f64>> {
    let (u, s, vt) = svd_thin(a)?;

    let rank_cap = s.len().min(u.ncols()).min(vt.nrows());
    let ncols_out = b.ncols();
    let nrows_out = vt.ncols();
    let mut result = Array2::<f64>::zeros((nrows_out, ncols_out));
    if rank_cap == 0 {
        return Ok(result);
    }

    // Relative truncation threshold, the usual `max(m, n) * eps * sigma_max`.
    let smax = s.iter().take(rank_cap).fold(0.0f64, |acc, &v| acc.max(v));
    let tol = smax * (a.nrows().max(a.ncols()) as f64) * f64::EPSILON;

    // result = V * diag(1/s) * (U^T * B), skipping the truncated directions.
    for k in 0..rank_cap {
        if s[k] <= tol {
            continue;
        }
        let inv_s = 1.0 / s[k];
        for col in 0..ncols_out {
            // (U^T B)[k, col]
            let ub: f64 = (0..a.nrows()).map(|row| u[[row, k]] * b[[row, col]]).sum();
            let scaled = ub * inv_s;
            for row in 0..nrows_out {
                result[[row, col]] += vt[[k, row]] * scaled;
            }
        }
    }

    Ok(result)
}

/// Simulate state-space model and compute fit percentage
fn compute_simulation_fit(
    y: &Array1<f64>,
    u: &Array1<f64>,
    a: &Array2<f64>,
    b: &Array2<f64>,
    c: &Array2<f64>,
    d: &Array2<f64>,
) -> f64 {
    let n = y.len().min(u.len());
    let nx = a.nrows();
    if nx == 0 || n < 2 {
        return 0.0;
    }

    let mut x = Array1::<f64>::zeros(nx);
    let mut y_hat = Array1::<f64>::zeros(n);

    for t in 0..n {
        // y_hat(t) = C * x(t) + D * u(t)
        let mut yh = 0.0;
        for k in 0..nx {
            yh += c[[0, k]] * x[k];
        }
        if d.nrows() > 0 && d.ncols() > 0 {
            yh += d[[0, 0]] * u[t];
        }
        y_hat[t] = yh;

        // x(t+1) = A * x(t) + B * u(t)
        let mut x_new = Array1::<f64>::zeros(nx);
        for k in 0..nx {
            let mut val = 0.0;
            for l in 0..nx {
                val += a[[k, l]] * x[l];
            }
            if b.ncols() > 0 {
                val += b[[k, 0]] * u[t];
            }
            x_new[k] = val;
        }
        x = x_new;
    }

    compute_fit_percentage(y, &y_hat)
}

/// Compute noise variance from simulation residuals
fn compute_simulation_noise_var(
    y: &Array1<f64>,
    u: &Array1<f64>,
    a: &Array2<f64>,
    b: &Array2<f64>,
    c: &Array2<f64>,
    d: &Array2<f64>,
) -> f64 {
    let n = y.len().min(u.len());
    let nx = a.nrows();
    if nx == 0 || n < 2 {
        return f64::INFINITY;
    }

    let mut x = Array1::<f64>::zeros(nx);
    let mut ss_res = 0.0;

    for t in 0..n {
        let mut yh = 0.0;
        for k in 0..nx {
            yh += c[[0, k]] * x[k];
        }
        if d.nrows() > 0 && d.ncols() > 0 {
            yh += d[[0, 0]] * u[t];
        }
        ss_res += (y[t] - yh).powi(2);

        let mut x_new = Array1::<f64>::zeros(nx);
        for k in 0..nx {
            let mut val = 0.0;
            for l in 0..nx {
                val += a[[k, l]] * x[l];
            }
            if b.ncols() > 0 {
                val += b[[k, 0]] * u[t];
            }
            x_new[k] = val;
        }
        x = x_new;
    }

    let n_params = nx * nx + nx + nx + 1; // A + B + C + D
    let dof = (n as f64 - n_params as f64).max(1.0);
    ss_res / dof
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_n4sid_first_order() {
        // True system: x[k+1] = 0.8*x[k] + 0.5*u[k], y[k] = x[k]
        let n = 500;
        let mut u = Array1::<f64>::zeros(n);
        let mut y = Array1::<f64>::zeros(n);

        for i in 0..n {
            u[i] = ((i as f64) * 0.1).sin();
        }
        let mut x = 0.0;
        for i in 0..n {
            y[i] = x;
            x = 0.8 * x + 0.5 * u[i];
        }

        let config = N4sidConfig {
            state_order: Some(1),
            block_rows: 8,
            sv_threshold: 0.01,
        };
        let result = n4sid_estimate(&y, &u, &config).expect("N4SID failed");

        assert_eq!(result.state_order, 1);
        assert_eq!(result.a.nrows(), 1);
        assert_eq!(result.b.nrows(), 1);
        assert_eq!(result.c.ncols(), 1);
        // The fit should be reasonable for a noise-free system
        assert!(
            result.fit_percentage > 50.0,
            "Fit = {}",
            result.fit_percentage
        );
    }

    #[test]
    fn test_n4sid_automatic_order() {
        // First-order system, let N4SID determine order
        let n = 500;
        let mut u = Array1::<f64>::zeros(n);
        let mut y = Array1::<f64>::zeros(n);

        for i in 0..n {
            u[i] = ((i as f64) * 0.08).sin() + 0.3 * ((i as f64) * 0.2).cos();
        }
        let mut x = 0.0;
        for i in 0..n {
            y[i] = x;
            x = 0.7 * x + 0.4 * u[i];
        }

        let config = N4sidConfig {
            state_order: None,
            block_rows: 10,
            sv_threshold: 0.05,
        };
        let result = n4sid_estimate(&y, &u, &config).expect("N4SID auto order failed");

        // Should pick a low order for a first-order system
        assert!(
            result.state_order >= 1 && result.state_order <= 5,
            "Order = {}",
            result.state_order
        );
        assert!(result.singular_values.len() > 0);
    }

    #[test]
    fn test_n4sid_dimension_mismatch() {
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let u = Array1::from_vec(vec![1.0, 2.0]);
        let config = N4sidConfig::default();
        assert!(n4sid_estimate(&y, &u, &config).is_err());
    }

    #[test]
    fn test_n4sid_insufficient_data() {
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let u = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let config = N4sidConfig {
            state_order: Some(1),
            block_rows: 10,
            sv_threshold: 0.01,
        };
        assert!(n4sid_estimate(&y, &u, &config).is_err());
    }
}
