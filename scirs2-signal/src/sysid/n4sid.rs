// N4SID (Numerical Algorithms for Subspace State Space System Identification)

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{s, Array1, Array2};
use scirs2_linalg::svd;

/// N4SID (Numerical Algorithms for Subspace State Space System Identification)
///
/// Implements the unweighted N4SID algorithm for SISO (single-input, single-output)
/// state-space model identification from input-output data.
///
/// The algorithm:
/// 1. Constructs block Hankel matrices from past/future input and output signals.
/// 2. Computes an oblique projection of the future outputs onto the past data
///    along the future input subspace via SVD-based pseudoinverse.
/// 3. Performs SVD of the oblique projection to extract the state sequence.
/// 4. Solves a least-squares problem for the system matrices A, B, C, D.
///
/// # Arguments
/// * `input` - Input signal (length N)
/// * `output` - Output signal (length N)
/// * `state_order` - Desired state-space model order n
/// * `past_horizon` - Number of block rows in the past Hankel matrix (i_p ≥ n)
/// * `future_horizon` - Number of block rows in the future Hankel matrix (i_f ≥ n)
///
/// # Returns
/// * State-space matrices `(A, B, C, D)` of sizes
///   `(n×n, n×1, 1×n, 1×1)` respectively.
///
/// # Errors
/// Returns `SignalError::ValueError` when the data length or horizon/order
/// parameters are inconsistent.
#[allow(clippy::type_complexity)]
pub fn n4sid_identification(
    input: &Array1<f64>,
    output: &Array1<f64>,
    state_order: usize,
    past_horizon: usize,
    future_horizon: usize,
) -> SignalResult<(Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>)> {
    let n_samples = input.len();

    if n_samples != output.len() {
        return Err(SignalError::ValueError(
            "input and output must have the same length".to_string(),
        ));
    }
    if state_order == 0 {
        return Err(SignalError::ValueError(
            "state_order must be at least 1".to_string(),
        ));
    }
    if past_horizon == 0 || future_horizon == 0 {
        return Err(SignalError::ValueError(
            "past_horizon and future_horizon must be at least 1".to_string(),
        ));
    }
    if past_horizon < state_order || future_horizon < state_order {
        return Err(SignalError::ValueError(
            "past_horizon and future_horizon must each be >= state_order".to_string(),
        ));
    }
    // Number of columns in Hankel matrices
    let total_rows = past_horizon + future_horizon;
    if n_samples < total_rows + 1 {
        return Err(SignalError::ValueError(
            "Insufficient data for N4SID identification".to_string(),
        ));
    }
    // j = number of data columns available (must exceed state_order)
    let j = n_samples - total_rows + 1;
    if j <= state_order {
        return Err(SignalError::ValueError(format!(
            "Too few data columns j={j} for state_order={state_order}; increase data length or decrease horizons/order",
        )));
    }

    // -------------------------------------------------------------------------
    // Step 1: Build block Hankel matrices (SISO, so 1 row per time shift)
    //
    // Each matrix has `horizon` rows and `j` columns.
    // Row k, col t: signal[k + t]  (0-indexed)
    //
    // U_p  (past input):    rows 0..past_horizon,            j cols
    // U_f  (future input):  rows past_horizon..total_rows,   j cols
    // Y_p  (past output):   rows 0..past_horizon,            j cols
    // Y_f  (future output): rows past_horizon..total_rows,   j cols
    // -------------------------------------------------------------------------
    let build_hankel = |signal: &Array1<f64>, row_start: usize, row_count: usize| {
        let mut h = Array2::<f64>::zeros((row_count, j));
        for r in 0..row_count {
            for c in 0..j {
                h[[r, c]] = signal[row_start + r + c];
            }
        }
        h
    };

    let u_p = build_hankel(input, 0, past_horizon);
    let u_f = build_hankel(input, past_horizon, future_horizon);
    let y_p = build_hankel(output, 0, past_horizon);
    let y_f = build_hankel(output, past_horizon, future_horizon);

    // -------------------------------------------------------------------------
    // Step 2: Build the "past state" matrix W_p = vstack(Y_p, U_p)
    //         and the full regressor M = vstack(U_f, W_p)
    //
    //   W_p: (2*past_horizon) × j
    //   M:   (future_horizon + 2*past_horizon) × j
    // -------------------------------------------------------------------------
    let w_p_rows = 2 * past_horizon;
    let mut w_p = Array2::<f64>::zeros((w_p_rows, j));
    for r in 0..past_horizon {
        for c in 0..j {
            w_p[[r, c]] = y_p[[r, c]];
        }
    }
    for r in 0..past_horizon {
        for c in 0..j {
            w_p[[past_horizon + r, c]] = u_p[[r, c]];
        }
    }

    let m_rows = future_horizon + w_p_rows;
    let mut m = Array2::<f64>::zeros((m_rows, j));
    for r in 0..future_horizon {
        for c in 0..j {
            m[[r, c]] = u_f[[r, c]];
        }
    }
    for r in 0..w_p_rows {
        for c in 0..j {
            m[[future_horizon + r, c]] = w_p[[r, c]];
        }
    }

    // -------------------------------------------------------------------------
    // Step 3: Oblique projection via SVD-based pseudoinverse
    //
    // O_i = Y_f * pinv(M) * W_p   (in row-space formulation)
    //
    // With row-major data (matrices are (rows × cols)):
    //   Y_f @ M^+  gives (future_horizon × j) @ (j × m_rows) = (future_horizon × m_rows)
    //   We keep only the columns corresponding to W_p (columns future_horizon..)
    //
    // Practical computation:
    //   1. pinv(M) = V * diag(1/s) * U^T   (economy SVD of M)
    //   2. coeff = Y_f @ V @ diag(1/s) @ U^T   → (future_horizon × m_rows)
    //   3. O_i = coeff[:, future_horizon:] @ W_p  → (future_horizon × j)
    //
    // Using economy SVD on M^T to avoid huge intermediate products:
    //   M is (m_rows × j); if m_rows > j use svd on M^T to get compact form.
    // -------------------------------------------------------------------------
    let svd_tol = 1e-10_f64;

    // M is (m_rows × j). Compute SVD: M = U_m * diag(s_m) * Vt_m
    let (u_m, s_m, vt_m) = svd(&m.view(), false, None)
        .map_err(|e| SignalError::ComputationError(format!("SVD of M failed: {e}")))?;

    // Compute pinv(M) = Vt_m^T * diag(1/s_m) * U_m^T
    // pinv(M) has shape (j × m_rows)
    // Instead of materializing pinv(M) fully, compute O_i directly:
    //   Y_f @ pinv(M) = Y_f @ Vt_m^T @ diag(1/s_m) @ U_m^T
    //                 = (Y_f @ Vt_m^T) * (1/s_m)^T  × U_m^T
    let rank_m = s_m.iter().filter(|&&sv| sv > svd_tol).count();
    if rank_m == 0 {
        return Err(SignalError::ComputationError(
            "Regressor matrix M has zero numerical rank; check data quality".to_string(),
        ));
    }

    // y_f_vt_t: (future_horizon × rank_m) = Y_f @ Vt_m^T[:rank_m, :]^T
    //         = Y_f @ Vt_m[:rank_m, :]^T   (vt_m is (k × j), take first rank_m rows)
    let vt_m_r = vt_m.slice(s![..rank_m, ..]).to_owned(); // (rank_m × j)
                                                          // Y_f @ vt_m_r^T → (future_horizon × rank_m)
    let y_f_v = y_f.dot(&vt_m_r.t());

    // Scale columns by 1/s_m
    let mut y_f_v_scaled = Array2::<f64>::zeros((future_horizon, rank_m));
    for col in 0..rank_m {
        let scale = 1.0 / s_m[col];
        for row in 0..future_horizon {
            y_f_v_scaled[[row, col]] = y_f_v[[row, col]] * scale;
        }
    }

    // proj_full = y_f_v_scaled @ U_m^T[:, :rank_m]^T … wait, more carefully:
    // U_m is (m_rows × k_u) from economy SVD; U_m^T is (k_u × m_rows)
    // y_f_v_scaled @ U_m^T → (future_horizon × m_rows)
    let u_m_r = u_m.slice(s![.., ..rank_m]).to_owned(); // (m_rows × rank_m)
    let proj_full = y_f_v_scaled.dot(&u_m_r.t()); // (future_horizon × m_rows)

    // Extract the W_p part: columns [future_horizon..m_rows]
    let proj_wp = proj_full.slice(s![.., future_horizon..]).to_owned(); // (future_horizon × w_p_rows)

    // O_i = proj_wp @ W_p → (future_horizon × j)
    let o_i = proj_wp.dot(&w_p); // (future_horizon × j)

    // -------------------------------------------------------------------------
    // Step 4: SVD of the oblique projection (unweighted: W1 = I, W2 = I)
    // -------------------------------------------------------------------------
    let (u_o, s_o, _vt_o) = svd(&o_i.view(), false, None).map_err(|e| {
        SignalError::ComputationError(format!("SVD of oblique projection failed: {e}"))
    })?;

    // -------------------------------------------------------------------------
    // Step 5: Truncate to model order n
    // -------------------------------------------------------------------------
    let n = state_order.min(s_o.len());
    if n == 0 {
        return Err(SignalError::ComputationError(
            "No non-zero singular values in oblique projection".to_string(),
        ));
    }

    let u1 = u_o.slice(s![.., ..n]).to_owned(); // (future_horizon × n)
    let s1 = s_o.slice(s![..n]).to_owned(); // (n,)

    // -------------------------------------------------------------------------
    // Step 6: State sequence estimate
    //   X = sqrt(S1) * U1^T * O_i
    //   shape: (n × j)
    // -------------------------------------------------------------------------
    let mut sqrt_s1 = Array1::<f64>::zeros(n);
    for i in 0..n {
        sqrt_s1[i] = s1[i].max(0.0).sqrt();
    }

    // Scale rows of U1^T by sqrt_s1 → (n × future_horizon)
    let u1_t = u1.t().to_owned(); // (n × future_horizon)
    let mut scaled_u1t = Array2::<f64>::zeros((n, future_horizon));
    for i in 0..n {
        for k in 0..future_horizon {
            scaled_u1t[[i, k]] = sqrt_s1[i] * u1_t[[i, k]];
        }
    }
    let x_hat = scaled_u1t.dot(&o_i); // (n × j)

    // -------------------------------------------------------------------------
    // Step 7: Least-squares identification of A, B, C, D
    //
    // Subspace regression (one-step prediction):
    //   [x_hat[:, 1..j] ; y_f[0, 0..j-1]]    =   [A B; C D]  @  [x_hat[:, 0..j-1]; u_f[0, 0..j-1]]
    //
    // Left-hand side  LHS: ((n+1) × (j-1))
    // Right-hand side RHS: ((n+1) × (j-1))
    // Solve:  LHS = ABCD @ RHS
    //         ABCD = LHS @ RHS^+  = LHS @ RHS^T @ (RHS @ RHS^T)^{-1}
    //
    // We use column-by-column lstsq via RHS^T @ x = LHS_col^T for each of
    // the (n+1) output rows, which is equivalent to solving
    //   RHS^T @ ABCD^T = LHS^T  in one shot using solve_multiple on normal eq.
    // -------------------------------------------------------------------------
    let jm1 = j - 1; // number of regression pairs
    if jm1 == 0 {
        return Err(SignalError::ComputationError(
            "No regression pairs available; increase data length".to_string(),
        ));
    }

    // Regression matrix: (n+1) × jm1 — stacked shifted state + input
    let rhs_rows = n + 1;
    let mut rhs = Array2::<f64>::zeros((rhs_rows, jm1));
    for i in 0..n {
        for c in 0..jm1 {
            rhs[[i, c]] = x_hat[[i, c]];
        }
    }
    // Last row: u_f[0, 0..jm1]  (first row of u_f = input at time past_horizon + c)
    for c in 0..jm1 {
        rhs[[n, c]] = u_f[[0, c]];
    }

    // Target matrix: (n+1) × jm1 — stacked next-state + current output
    let lhs_rows = n + 1;
    let mut lhs = Array2::<f64>::zeros((lhs_rows, jm1));
    for i in 0..n {
        for c in 0..jm1 {
            lhs[[i, c]] = x_hat[[i, c + 1]];
        }
    }
    // Last row: y_f[0, 0..jm1] (first row of y_f = output at time past_horizon + c)
    for c in 0..jm1 {
        lhs[[n, c]] = y_f[[0, c]];
    }

    // Solve via normal equations: ABCD = LHS @ RHS^T @ (RHS @ RHS^T)^{-1}
    //   <=> ABCD @ (RHS @ RHS^T) = LHS @ RHS^T
    //   <=> (RHS @ RHS^T)^T @ ABCD^T = (LHS @ RHS^T)^T
    // We compute this per output row using SVD-based LS on RHS^T (j-1 × n+1):
    let rhs_t = rhs.t().to_owned(); // (jm1 × rhs_rows)

    // SVD of rhs_t for pseudoinverse reuse across all output rows
    let (u_rhs, s_rhs, vt_rhs) = svd(&rhs_t.view(), false, None).map_err(|e| {
        SignalError::ComputationError(format!("SVD of regression matrix failed: {e}"))
    })?;

    let rank_rhs = s_rhs.iter().filter(|&&sv| sv > svd_tol).count();

    // pinv(rhs_t) = vt_rhs[:rank_rhs,:]^T @ diag(1/s_rhs[:rank_rhs]) @ u_rhs[:,:rank_rhs]^T
    // Result shape: (rhs_rows × jm1)
    let u_rhs_r = u_rhs.slice(s![.., ..rank_rhs]).to_owned(); // (jm1 × rank_rhs)
    let vt_rhs_r = vt_rhs.slice(s![..rank_rhs, ..]).to_owned(); // (rank_rhs × rhs_rows)

    // For each output row o in 0..lhs_rows, solve:
    //   x_sol[o, :] = pinv(rhs_t) @ lhs[o, :]
    //               = vt_rhs_r^T @ diag(1/s_rhs[:rank_rhs]) @ u_rhs_r^T @ lhs[o, :]
    let mut abcd = Array2::<f64>::zeros((lhs_rows, rhs_rows));
    for o in 0..lhs_rows {
        // lhs_col: (jm1,) — the target for this output row transposed
        let lhs_col = lhs.slice(s![o, ..]).to_owned(); // (jm1,)
                                                       // u_rhs_r^T @ lhs_col → (rank_rhs,)
        let ut_b = u_rhs_r.t().dot(&lhs_col); // (rank_rhs,)
                                              // scale by 1/s_rhs
        let mut scaled = Array1::<f64>::zeros(rank_rhs);
        for k in 0..rank_rhs {
            scaled[k] = ut_b[k] / s_rhs[k];
        }
        // vt_rhs_r^T @ scaled → (rhs_rows,)
        let sol = vt_rhs_r.t().dot(&scaled); // (rhs_rows,)
        for k in 0..rhs_rows {
            abcd[[o, k]] = sol[k];
        }
    }

    // -------------------------------------------------------------------------
    // Step 8: Extract A (n×n), B (n×1), C (1×n), D (1×1) from abcd (n+1 × n+1)
    // -------------------------------------------------------------------------
    let a_mat = abcd.slice(s![..n, ..n]).to_owned();
    let b_mat = abcd.slice(s![..n, n..n + 1]).to_owned();
    let c_mat = abcd.slice(s![n..n + 1, ..n]).to_owned();
    let d_mat = abcd.slice(s![n..n + 1, n..n + 1]).to_owned();

    Ok((a_mat, b_mat, c_mat, d_mat))
}
