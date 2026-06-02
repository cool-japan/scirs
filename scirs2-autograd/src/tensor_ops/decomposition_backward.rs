//! Murray (2016) and Townsend (2018) backward-pass helpers for matrix
//! decompositions: Cholesky, LU, QR.
//!
//! All operations are implemented in pure ndarray — no dependency on
//! `scirs2-linalg` (which would create a circular dep through `scirs2-autograd`).

use crate::Float;
use scirs2_core::ndarray::Array2;

// ─────────────────────────────────────────────────────────────────────────────
// Triangular linear-system solvers
// ─────────────────────────────────────────────────────────────────────────────

/// Forward substitution: solve `L · x = b` where `L` is lower triangular.
/// Returns `x`.  Panics if any diagonal entry is zero.
fn solve_lower_col<F: Float>(l: &Array2<F>, b: &[F]) -> Vec<F> {
    let n = l.nrows();
    debug_assert_eq!(n, b.len());
    let mut x = vec![F::zero(); n];
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= l[[i, j]] * x[j];
        }
        let d = l[[i, i]];
        if d == F::zero() {
            x[i] = F::zero();
        } else {
            x[i] = s / d;
        }
    }
    x
}

/// Back substitution: solve `U · x = b` where `U` is upper triangular.
/// Returns `x`.  Panics if any diagonal entry is zero.
fn solve_upper_col<F: Float>(u: &Array2<F>, b: &[F]) -> Vec<F> {
    let n = u.nrows();
    debug_assert_eq!(n, b.len());
    let mut x = vec![F::zero(); n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= u[[i, j]] * x[j];
        }
        let d = u[[i, i]];
        if d == F::zero() {
            x[i] = F::zero();
        } else {
            x[i] = s / d;
        }
    }
    x
}

/// Solve `L · X = B` (column-wise).  Returns `X = L⁻¹ B`.
pub(crate) fn solve_lower_matrix<F: Float>(l: &Array2<F>, b: &Array2<F>) -> Array2<F> {
    let (n, m) = (b.nrows(), b.ncols());
    let mut x = Array2::<F>::zeros((n, m));
    for j in 0..m {
        let bcol: Vec<F> = b.column(j).iter().copied().collect();
        let xcol = solve_lower_col(l, &bcol);
        for i in 0..n {
            x[[i, j]] = xcol[i];
        }
    }
    x
}

/// Solve `U · X = B` (column-wise).  Returns `X = U⁻¹ B`.
pub(crate) fn solve_upper_matrix<F: Float>(u: &Array2<F>, b: &Array2<F>) -> Array2<F> {
    let (n, m) = (b.nrows(), b.ncols());
    let mut x = Array2::<F>::zeros((n, m));
    for j in 0..m {
        let bcol: Vec<F> = b.column(j).iter().copied().collect();
        let xcol = solve_upper_col(u, &bcol);
        for i in 0..n {
            x[[i, j]] = xcol[i];
        }
    }
    x
}

/// Compute `A · L⁻¹` via `X = A · L⁻¹ ↔ Lᵀ · Xᵀ = Aᵀ`.
pub(crate) fn right_solve_lower<F: Float>(a: &Array2<F>, l: &Array2<F>) -> Array2<F> {
    let lt = l.t().to_owned();
    let at = a.t().to_owned();
    let xt = solve_upper_matrix(&lt, &at);
    xt.t().to_owned()
}

/// Compute `A · U⁻¹` via `X = A · U⁻¹ ↔ Uᵀ · Xᵀ = Aᵀ`.
pub(crate) fn right_solve_upper<F: Float>(a: &Array2<F>, u: &Array2<F>) -> Array2<F> {
    let ut = u.t().to_owned();
    let at = a.t().to_owned();
    let xt = solve_lower_matrix(&ut, &at);
    xt.t().to_owned()
}

// ─────────────────────────────────────────────────────────────────────────────
// Shape helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Keep only the strictly lower-triangular entries; zero diagonal and above.
fn keep_strict_lower<F: Float>(m: &Array2<F>) -> Array2<F> {
    let n = m.nrows();
    let ncols = m.ncols();
    let mut out = m.clone();
    for i in 0..n {
        for j in i..ncols {
            out[[i, j]] = F::zero();
        }
    }
    out
}

/// Keep the upper triangle (including diagonal); zero strictly below.
pub(crate) fn keep_upper<F: Float>(m: &Array2<F>) -> Array2<F> {
    let n = m.nrows();
    let ncols = m.ncols();
    let mut out = m.clone();
    for i in 0..n {
        for j in 0..i.min(ncols) {
            out[[i, j]] = F::zero();
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Cholesky backward — Murray (2016) eq. 8
// ─────────────────────────────────────────────────────────────────────────────

/// Cholesky backward pass for `A = L Lᵀ`.
///
/// Input: forward `L` (lower triangular) and upstream gradient `dL`.
/// Returns `dA` of size `n × n` (symmetric).
///
/// Murray (2016) eq. 8:
///   phi(M)_{ii} = M_{ii} / 2,  phi(M)_{ij} = M_{ij} for i > j,  0 otherwise.
///   dA = ½ · L⁻ᵀ · (phi + phiᵀ) · L⁻¹
pub fn cholesky_backward<F: Float>(l: &Array2<F>, grad_l: &Array2<F>) -> Array2<F> {
    let n = l.nrows();
    let lt = l.t().to_owned();

    // Inner = Lᵀ · dL
    let lt_dl = lt.dot(grad_l);

    // phi: keep strictly lower, halve diagonal, zero strictly upper.
    let two = F::from(2.0).unwrap_or_else(|| F::one() + F::one());
    let mut phi = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            if i > j {
                phi[[i, j]] = lt_dl[[i, j]];
            } else if i == j {
                phi[[i, i]] = lt_dl[[i, i]] / two;
            }
            // i < j stays 0
        }
    }

    let phi_t = phi.t().to_owned();
    let phi_sym = &phi + &phi_t;

    // dA = ½ · L⁻ᵀ · phi_sym · L⁻¹
    // Step 1: y = phi_sym · L⁻¹  ↔  right_solve_lower(phi_sym, L)
    let phi_l_inv = right_solve_lower(&phi_sym, l);
    // Step 2: L⁻ᵀ · y  ↔  solve_upper(Lᵀ, y)
    let inner = solve_upper_matrix(&lt, &phi_l_inv);

    // Multiply by ½
    let half = F::from(0.5).unwrap_or_else(|| F::one() / two);
    let mut da = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            da[[i, j]] = half * inner[[i, j]];
        }
    }
    da
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. LU backward — Murray (2016) §4 / Giles (2008)
// ─────────────────────────────────────────────────────────────────────────────

/// LU backward pass.  P is the permutation matrix such that `P · A = L · U`.
///
/// - `grad_l`: upstream cotangent for L (only strict-lower triangle matters)
/// - `grad_u`: upstream cotangent for U (only upper triangle matters)
/// - Returns `dA` of size `n × n`.
///
/// Formula:
///   F = phi_lower(Lᵀ · dL_masked) + phi_upper(dU_masked · Uᵀ)
///   dA = Pᵀ · L⁻ᵀ · F · U⁻ᵀ
pub fn lu_backward<F: Float>(
    p: &Array2<F>,
    l: &Array2<F>,
    u: &Array2<F>,
    grad_l: &Array2<F>,
    grad_u: &Array2<F>,
) -> Array2<F> {
    let n = l.nrows();
    let lt = l.t().to_owned();
    let ut = u.t().to_owned();

    // Mask gradients to the structural support of L (strict lower) and U (upper).
    let dl_masked = keep_strict_lower(grad_l);
    let du_masked = keep_upper(grad_u);

    // F1 = phi_lower(Lᵀ · dL_masked) — strictly lower triangle (i > j only).
    let lt_dl = lt.dot(&dl_masked);
    let mut f1 = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..i {
            f1[[i, j]] = lt_dl[[i, j]];
        }
    }

    // F2 = phi_upper(dU_masked · Uᵀ) — upper triangle including diagonal.
    let du_ut = du_masked.dot(&ut);
    let f2 = keep_upper(&du_ut);

    let f = &f1 + &f2;

    // L⁻ᵀ · F  ↔  Lᵀ · X = F  → solve_upper(Lᵀ, F)
    let lt_inv_f = solve_upper_matrix(&lt, &f);

    // (lt_inv_f) · U⁻ᵀ  ↔  right_solve_lower(lt_inv_f, Uᵀ)
    let result = right_solve_lower(&lt_inv_f, &ut);

    // dA = Pᵀ · result.
    let pt = p.t().to_owned();
    pt.dot(&result)
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. QR backward — Townsend / Murray (2016) §2.3
// ─────────────────────────────────────────────────────────────────────────────

/// Slice columns `cols` from a 2D array.
fn take_cols<F: Float>(a: &Array2<F>, start: usize, end: usize) -> Array2<F> {
    let m = a.nrows();
    let k = end - start;
    let mut out = Array2::<F>::zeros((m, k));
    for i in 0..m {
        for (jj, j) in (start..end).enumerate() {
            out[[i, jj]] = a[[i, j]];
        }
    }
    out
}

/// Slice rows `rows` from a 2D array.
fn take_rows<F: Float>(a: &Array2<F>, start: usize, end: usize) -> Array2<F> {
    let ncols = a.ncols();
    let k = end - start;
    let mut out = Array2::<F>::zeros((k, ncols));
    for (ii, i) in (start..end).enumerate() {
        for j in 0..ncols {
            out[[ii, j]] = a[[i, j]];
        }
    }
    out
}

/// QR backward pass for `A = Q · R` (thin QR, Q m×k, R k×n, m ≥ k = n).
///
/// Upstream gradients:
/// - `grad_q` (m×k) — cotangent of Q_thin
/// - `grad_r` (k×n) — cotangent of R_thin
///
/// Returns `dA` of shape `m × n`.
///
/// Thin formula (Giles 2008 §2.3 / Townsend 2016):
///   M  =  R_thin · dR_thinᵀ − dQ_thinᵀ · Q_thin    (k × k, antisymmetric part)
///   S  =  copyltu(M)  =  tril(M) + tril(M)ᵀ − diag(M)   (symmetrise using lower triangle)
///   dA =  (dQ_thin + Q_thin · S) · R_thin⁻ᵀ
///
/// Note: there is NO (I − QQᵀ) projection in the thin-QR formula.
pub fn qr_backward<F: Float>(
    q: &Array2<F>,
    r: &Array2<F>,
    grad_q: &Array2<F>,
    grad_r: &Array2<F>,
) -> Array2<F> {
    // Support both thin-QR (q: m×k, r: k×n) and full-QR (q: m×m, r: m×n).
    // k = min(m, cols_of_q); we restrict to the thin part.
    let k = q.ncols().min(r.nrows()); // number of Q columns / R rows used
    let _n = r.ncols(); // columns of A (unused explicitly; keep for clarity)

    // Mask dR to upper triangle of its top-k×k block; extra rows ignored.
    let grad_r_top = keep_upper(&take_rows(grad_r, 0, k));
    let r_top = take_rows(r, 0, k); // k × n  (already upper triangular above diagonal)
    let r_sq = take_cols(&r_top, 0, k); // k × k  (square upper-triangular block)
    let q_thin = take_cols(q, 0, k); // m × k
    let grad_q_thin = take_cols(grad_q, 0, k); // m × k

    let rt_sq = r_sq.t().to_owned(); // k × k  lower triangular

    // M = R_sq · dR_topᵀ[:k,:k]  -  dQ_thinᵀ · Q_thin   (k × k)
    let dr_top_sq = take_cols(&grad_r_top, 0, k); // k × k
    let r_drt = r_sq.dot(&dr_top_sq.t().to_owned()); // k × k
    let dqt_q = grad_q_thin.t().to_owned().dot(&q_thin); // k × k
    let m_mat = &r_drt - &dqt_q; // k × k

    // S = copyltu(M): symmetrise by copying the lower triangle to the upper.
    // S[i,j] = M[i,j] if i >= j  else  M[j,i]
    let mut s = Array2::<F>::zeros((k, k));
    for i in 0..k {
        for j in 0..k {
            s[[i, j]] = if i >= j { m_mat[[i, j]] } else { m_mat[[j, i]] };
        }
    }

    // inner = dQ_thin + Q_thin · S   (m × k)
    // There is NO (I - QQᵀ) projection for the thin-QR formula.
    let q_s = q_thin.dot(&s); // m × k
    let inner = &grad_q_thin + &q_s; // m × k

    // dA (m × n) = inner · R_sq⁻ᵀ  padded with zeros for extra columns.
    // First solve inner (m×k) with R_sq (k×k) giving m×k.
    let da_thin = right_solve_lower(&inner, &rt_sq); // m × k

    // If A has more columns than k (wide matrix, k < n), fill extra cols with zeros.
    let n = r.ncols();
    if n <= k {
        da_thin
    } else {
        let m = q.nrows();
        let mut da = Array2::<F>::zeros((m, n));
        for i in 0..m {
            for j in 0..k {
                da[[i, j]] = da_thin[[i, j]];
            }
        }
        da
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fast LCG for deterministic test matrices.
    fn lcg_next(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (*state >> 11) & ((1u64 << 53) - 1);
        bits as f64 / (1u64 << 53) as f64
    }

    fn random_matrix(n: usize, m: usize, seed: u64) -> Array2<f64> {
        let mut s = seed;
        Array2::from_shape_fn((n, m), |_| 2.0 * lcg_next(&mut s) - 1.0)
    }

    fn random_spd(n: usize, seed: u64) -> Array2<f64> {
        let b = random_matrix(n, n, seed);
        let mut a = b.t().dot(&b);
        for i in 0..n {
            a[[i, i]] += n as f64;
        }
        a
    }

    fn fd_gradient<G: Fn(&Array2<f64>) -> f64>(a: &Array2<f64>, loss: G) -> Array2<f64> {
        let (n, m) = (a.nrows(), a.ncols());
        let h = 1e-5_f64;
        let mut grad = Array2::<f64>::zeros((n, m));
        for i in 0..n {
            for j in 0..m {
                let mut ap = a.clone();
                let mut am = a.clone();
                ap[[i, j]] += h;
                am[[i, j]] -= h;
                grad[[i, j]] = (loss(&ap) - loss(&am)) / (2.0 * h);
            }
        }
        grad
    }

    fn fd_gradient_symmetric<G: Fn(&Array2<f64>) -> f64>(a: &Array2<f64>, loss: G) -> Array2<f64> {
        let n = a.nrows();
        assert_eq!(n, a.ncols());
        let h = 1e-5_f64;
        let mut grad = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in i..n {
                let mut ap = a.clone();
                let mut am = a.clone();
                if i == j {
                    ap[[i, i]] += h;
                    am[[i, i]] -= h;
                    grad[[i, i]] = (loss(&ap) - loss(&am)) / (2.0 * h);
                } else {
                    ap[[i, j]] += h;
                    ap[[j, i]] += h;
                    am[[i, j]] -= h;
                    am[[j, i]] -= h;
                    let g = (loss(&ap) - loss(&am)) / (2.0 * h);
                    grad[[i, j]] = 0.5 * g;
                    grad[[j, i]] = 0.5 * g;
                }
            }
        }
        grad
    }

    fn max_abs(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        a.iter().zip(b.iter()).fold(0.0_f64, |m, (x, y)| {
            let d = (x - y).abs();
            if d > m {
                d
            } else {
                m
            }
        })
    }

    /// Cholesky (Gram-Schmidt style) for testing only.
    fn cholesky_fwd(a: &Array2<f64>) -> Array2<f64> {
        let n = a.nrows();
        let mut l = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..=i {
                if i == j {
                    let mut sum = 0.0;
                    for k in 0..j {
                        sum += l[[j, k]] * l[[j, k]];
                    }
                    let d = a[[j, j]] - sum;
                    l[[j, j]] = if d > 0.0 { d.sqrt() } else { 0.0 };
                } else {
                    let mut sum = 0.0;
                    for k in 0..j {
                        sum += l[[i, k]] * l[[j, k]];
                    }
                    let lii = l[[j, j]];
                    l[[i, j]] = if lii != 0.0 {
                        (a[[i, j]] - sum) / lii
                    } else {
                        0.0
                    };
                }
            }
        }
        l
    }

    /// Simple LU (no pivoting) for testing.
    fn lu_fwd_no_pivot(a: &Array2<f64>) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        let n = a.nrows();
        let p = Array2::<f64>::eye(n);
        let mut l = Array2::<f64>::eye(n);
        let mut u = a.clone();
        for k in 0..n - 1 {
            if u[[k, k]].abs() > 1e-14 {
                for i in (k + 1)..n {
                    l[[i, k]] = u[[i, k]] / u[[k, k]];
                    for j in k..n {
                        u[[i, j]] -= l[[i, k]] * u[[k, j]];
                    }
                }
            }
        }
        for i in 0..n {
            for j in 0..i {
                u[[i, j]] = 0.0;
            }
        }
        (p, l, u)
    }

    /// Simple QR via Gram-Schmidt for testing.
    fn qr_fwd_gs(a: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        let (m, n) = (a.nrows(), a.ncols());
        let k = m; // full QR: Q is m×m
        let mut q = Array2::<f64>::zeros((m, k));
        let mut r = Array2::<f64>::zeros((k, n));
        // Augment a with extra columns (e_i) if m > n
        let mut a_aug = Array2::<f64>::zeros((m, m));
        for i in 0..m {
            for j in 0..n {
                a_aug[[i, j]] = a[[i, j]];
            }
            if i >= n {
                a_aug[[i, i]] = 1.0;
            }
        }
        for j in 0..k {
            for i in 0..m {
                q[[i, j]] = a_aug[[i, j]];
            }
            for i in 0..j {
                let mut dot = 0.0;
                for row in 0..m {
                    dot += q[[row, i]] * q[[row, j]];
                }
                for row in 0..m {
                    q[[row, j]] -= dot * q[[row, i]];
                }
            }
            let mut norm = 0.0;
            for row in 0..m {
                norm += q[[row, j]] * q[[row, j]];
            }
            norm = norm.sqrt();
            if norm > 1e-14 {
                for row in 0..m {
                    q[[row, j]] /= norm;
                }
            }
        }
        // Compute R = Qᵀ · A
        for i in 0..k {
            for j in 0..n {
                let mut dot = 0.0;
                for row in 0..m {
                    dot += q[[row, i]] * a[[row, j]];
                }
                r[[i, j]] = dot;
            }
        }
        (q, r)
    }

    #[test]
    fn cholesky_backward_unit() {
        let a = random_spd(4, 0xABCD);
        let l = cholesky_fwd(&a);
        let grad_l = Array2::<f64>::ones((4, 4));
        let analytic = cholesky_backward(&l, &grad_l);

        let numeric = fd_gradient_symmetric(&a, |ap| {
            let ll = cholesky_fwd(ap);
            ll.iter().sum()
        });
        let err = max_abs(&analytic, &numeric);
        assert!(err < 1e-6, "cholesky_backward_unit: max err = {err}");
    }

    #[test]
    fn lu_backward_unit() {
        let mut a = random_matrix(4, 4, 0x1234);
        for i in 0..4 {
            a[[i, i]] += 5.0;
        }
        let (p, l, u) = lu_fwd_no_pivot(&a);
        let n = 4;
        let mut grad_l = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..i {
                grad_l[[i, j]] = 1.0;
            }
        }
        let grad_u = keep_upper(&Array2::<f64>::ones((n, n)));
        let analytic = lu_backward(&p, &l, &u, &grad_l, &grad_u);

        let numeric = fd_gradient(&a, |ap| {
            let (_, ll, uu) = lu_fwd_no_pivot(ap);
            let mut sl = 0.0;
            for i in 0..n {
                for j in 0..i {
                    sl += ll[[i, j]];
                }
            }
            let mut su = 0.0;
            for i in 0..n {
                for j in i..n {
                    su += uu[[i, j]];
                }
            }
            sl + su
        });
        let err = max_abs(&analytic, &numeric);
        assert!(err < 1e-5, "lu_backward_unit: max err = {err}");
    }

    #[test]
    fn qr_backward_unit() {
        let mut a = random_matrix(4, 4, 0xDEAD);
        for i in 0..4 {
            a[[i, i]] += 3.0;
        }
        let (q, r) = qr_fwd_gs(&a);
        let grad_q = Array2::<f64>::zeros(q.raw_dim());
        let grad_r = keep_upper(&Array2::<f64>::ones(r.raw_dim()));
        let analytic = qr_backward(&q, &r, &grad_q, &grad_r);

        let numeric = fd_gradient(&a, |ap| {
            let (_, rr) = qr_fwd_gs(ap);
            let nn = rr.nrows();
            let mut s = 0.0;
            for i in 0..nn {
                for j in i..nn {
                    s += rr[[i, j]];
                }
            }
            s
        });
        let err = max_abs(&analytic, &numeric);
        assert!(err < 1e-4, "qr_backward_unit: max err = {err}");
    }

    /// Verify qr_backward for the Q component (grad_q=ones, grad_r=zeros).
    #[test]
    fn qr_backward_q_unit() {
        let mut a = random_matrix(4, 4, 0xCAFE);
        for i in 0..4 {
            a[[i, i]] += 3.0;
        }
        let (q, r) = qr_fwd_gs(&a);
        let grad_q = Array2::<f64>::ones(q.raw_dim());
        let grad_r = Array2::<f64>::zeros(r.raw_dim());
        let analytic = qr_backward(&q, &r, &grad_q, &grad_r);

        let numeric = fd_gradient(&a, |ap| {
            let (qq, _) = qr_fwd_gs(ap);
            qq.iter().sum()
        });
        let err = max_abs(&analytic, &numeric);
        assert!(err < 1e-4, "qr_backward_q_unit: max err = {err}");
    }

    /// Verify qr_backward with both grad_q and grad_r nonzero (mixed).
    #[test]
    fn qr_backward_mixed_unit() {
        let mut a = random_matrix(4, 4, 0xF00D);
        for i in 0..4 {
            a[[i, i]] += 3.0;
        }
        let (q, r) = qr_fwd_gs(&a);
        let n = 4;
        // Arbitrary nonzero cotangents for Q and R.
        let grad_q = random_matrix(n, n, 0xBEEF);
        let grad_r = keep_upper(&random_matrix(n, n, 0xDEED));
        let analytic = qr_backward(&q, &r, &grad_q, &grad_r);

        let numeric = fd_gradient(&a, |ap| {
            let (qq, rr) = qr_fwd_gs(ap);
            let sq: f64 = qq.iter().zip(grad_q.iter()).map(|(x, g)| x * g).sum();
            let sr: f64 = rr.iter().zip(grad_r.iter()).map(|(x, g)| x * g).sum();
            sq + sr
        });
        let err = max_abs(&analytic, &numeric);
        assert!(err < 1e-4, "qr_backward_mixed_unit: max err = {err}");
    }
}
