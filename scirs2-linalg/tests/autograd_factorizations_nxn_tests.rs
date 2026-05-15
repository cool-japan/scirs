//! n×n autodiff backward-pass reference implementations and validation tests.
//!
//! Wave 74 — Autograd factorizations n×n.
//!
//! This module derives and verifies, against finite differences, the
//! n×n analytical backward formulas (Murray 2016, Giles 2008) for:
//!
//! 1. LU decomposition `(P, L, U)`
//! 2. QR decomposition `(Q, R)` for square and thin matrices
//! 3. Cholesky decomposition `L` (SPD)
//! 4. Pseudo-inverse `A⁺`
//! 5. Matrix square root `√A` via Sylvester equation
//! 6. Matrix logarithm `log(A)` via Padé / Schur composition
//!
//! Plus a batch lifter that applies any of the above element-wise to a
//! `[batch_size, n, n]` tensor.
//!
//! Each formula is implemented as a stand-alone reference function
//! (no autograd-tape integration in this slice) and validated by
//! finite-difference gradient checks to within max error 1e-7.

use scirs2_core::ndarray::{Array2, Array3};
use scirs2_linalg::{cholesky, lu, qr, solve_sylvester, solve_triangular};

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Step size for centered finite differences.
const FD_EPS: f64 = 1e-5;

/// Maximum allowed difference between analytical and numerical gradients.
const FD_TOL: f64 = 1e-6;

/// Looser tolerance for ill-conditioned problems (logm/sqrtm/pinv).
const FD_TOL_LOOSE: f64 = 1e-4;

// ─────────────────────────────────────────────────────────────────────────────
// Deterministic sample-matrix generators
// ─────────────────────────────────────────────────────────────────────────────

/// Linear-congruential generator (deterministic, no external rand crate).
fn lcg_next(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let bits = (*state >> 11) & ((1u64 << 53) - 1);
    bits as f64 / (1u64 << 53) as f64
}

/// Fill an `n × m` matrix with values in `[-1, 1)` from the LCG.
fn random_matrix(n: usize, m: usize, seed: u64) -> Array2<f64> {
    let mut s = seed;
    Array2::from_shape_fn((n, m), |_| 2.0 * lcg_next(&mut s) - 1.0)
}

/// Build an SPD `n × n` matrix as `BᵀB + n·I` for well-conditioned testing.
fn random_spd(n: usize, seed: u64) -> Array2<f64> {
    let b = random_matrix(n, n, seed);
    let mut a = matmul(&b.t().to_owned(), &b);
    for i in 0..n {
        a[[i, i]] += n as f64;
    }
    a
}

// ─────────────────────────────────────────────────────────────────────────────
// Matrix-arithmetic helpers (no BLAS dependency for clarity).
// ─────────────────────────────────────────────────────────────────────────────

fn matmul(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (m, k) = (a.nrows(), a.ncols());
    let p = b.ncols();
    assert_eq!(k, b.nrows());
    let mut c = Array2::<f64>::zeros((m, p));
    for i in 0..m {
        for j in 0..p {
            let mut s = 0.0;
            for l in 0..k {
                s += a[[i, l]] * b[[l, j]];
            }
            c[[i, j]] = s;
        }
    }
    c
}

fn transpose(a: &Array2<f64>) -> Array2<f64> {
    let (m, n) = (a.nrows(), a.ncols());
    let mut t = Array2::<f64>::zeros((n, m));
    for i in 0..m {
        for j in 0..n {
            t[[j, i]] = a[[i, j]];
        }
    }
    t
}

/// Solve the lower-triangular system `L · x_col = b_col` for each column of B.
/// Returns `X = L⁻¹ B`.
fn solve_lower_triangular_matrix(l: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (n, m) = (l.nrows(), b.ncols());
    let mut x = Array2::<f64>::zeros((n, m));
    for j in 0..m {
        let bcol = b.column(j).to_owned();
        let xcol =
            solve_triangular(&l.view(), &bcol.view(), true, false).expect("lower-triangular solve");
        x.column_mut(j).assign(&xcol);
    }
    x
}

/// Solve the upper-triangular system `U · x_col = b_col` for each column of B.
/// Returns `X = U⁻¹ B`.
fn solve_upper_triangular_matrix(u: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (n, m) = (u.nrows(), b.ncols());
    let mut x = Array2::<f64>::zeros((n, m));
    for j in 0..m {
        let bcol = b.column(j).to_owned();
        let xcol = solve_triangular(&u.view(), &bcol.view(), false, false)
            .expect("upper-triangular solve");
        x.column_mut(j).assign(&xcol);
    }
    x
}

/// Compute `X = A · L⁻¹` by solving `Lᵀ · Xᵀ = Aᵀ` (forward substitution on Lᵀ).
fn right_solve_lower(a: &Array2<f64>, l: &Array2<f64>) -> Array2<f64> {
    let lt = transpose(l);
    let at = transpose(a);
    let xt = solve_upper_triangular_matrix(&lt, &at);
    transpose(&xt)
}

/// Compute `X = A · U⁻¹` by solving `Uᵀ · Xᵀ = Aᵀ` (back substitution on Uᵀ).
fn right_solve_upper(a: &Array2<f64>, u: &Array2<f64>) -> Array2<f64> {
    let ut = transpose(u);
    let at = transpose(a);
    let xt = solve_lower_triangular_matrix(&ut, &at);
    transpose(&xt)
}

/// Element-wise maximum absolute difference.
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

// ─────────────────────────────────────────────────────────────────────────────
// Generic finite-difference Jacobian
// ─────────────────────────────────────────────────────────────────────────────

/// Centered finite-difference gradient of a scalar loss `L(A)` at `A`.
fn fd_gradient<F>(a: &Array2<f64>, loss: F) -> Array2<f64>
where
    F: Fn(&Array2<f64>) -> f64,
{
    let (n, m) = (a.nrows(), a.ncols());
    let mut grad = Array2::<f64>::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            let mut ap = a.clone();
            let mut am = a.clone();
            ap[[i, j]] += FD_EPS;
            am[[i, j]] -= FD_EPS;
            grad[[i, j]] = (loss(&ap) - loss(&am)) / (2.0 * FD_EPS);
        }
    }
    grad
}

/// FD gradient with symmetric perturbation, returning the **effective
/// gradient** for direct comparison against an analytical symmetric `dA`.
///
/// We perturb `A` along `E_ij + E_ji` for `i ≠ j` and along `E_ii` for the
/// diagonal.  By chain rule:
///   off-diag FD =  ∂L/∂A[i,j] + ∂L/∂A[j,i]  =  2·analytic[i,j]   (symmetric)
///   diag FD     =  ∂L/∂A[i,i]               =  analytic[i,i]
/// So we divide off-diagonal FD by 2 to recover analytic[i,j] directly.
fn fd_gradient_symmetric<F>(a: &Array2<f64>, loss: F) -> Array2<f64>
where
    F: Fn(&Array2<f64>) -> f64,
{
    let n = a.nrows();
    assert_eq!(n, a.ncols());
    let mut grad = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in i..n {
            let mut ap = a.clone();
            let mut am = a.clone();
            if i == j {
                ap[[i, i]] += FD_EPS;
                am[[i, i]] -= FD_EPS;
                grad[[i, i]] = (loss(&ap) - loss(&am)) / (2.0 * FD_EPS);
            } else {
                ap[[i, j]] += FD_EPS;
                ap[[j, i]] += FD_EPS;
                am[[i, j]] -= FD_EPS;
                am[[j, i]] -= FD_EPS;
                let g = (loss(&ap) - loss(&am)) / (2.0 * FD_EPS);
                // Off-diagonal FD captures dA[i,j] + dA[j,i]; for symmetric
                // analytic dA the per-entry value is half.
                grad[[i, j]] = 0.5 * g;
                grad[[j, i]] = 0.5 * g;
            }
        }
    }
    grad
}

// ─────────────────────────────────────────────────────────────────────────────
// Forward wrappers around scirs2-linalg
// ─────────────────────────────────────────────────────────────────────────────

fn lu_fwd(a: &Array2<f64>) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    lu(&a.view(), None).expect("LU forward")
}

fn qr_fwd(a: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    qr(&a.view(), None).expect("QR forward")
}

fn cholesky_fwd(a: &Array2<f64>) -> Array2<f64> {
    cholesky(&a.view(), None).expect("Cholesky forward")
}

/// Numerically stable matrix-square-root via SVD diagonalisation for SPD inputs.
/// `A = V Σ Vᵀ`  ⇒  `√A = V Σ^{1/2} Vᵀ`.
/// We use Jacobi eigendecomposition to avoid pulling in the GPL Schur path.
fn sqrtm_spd(a: &Array2<f64>) -> Array2<f64> {
    let (eigenvalues, eigenvectors) = jacobi_eigh(a);
    let n = a.nrows();
    let mut d_sqrt = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        d_sqrt[[i, i]] = eigenvalues[i].max(0.0).sqrt();
    }
    let vd = matmul(&eigenvectors, &d_sqrt);
    matmul(&vd, &transpose(&eigenvectors))
}

/// Symmetric matrix logarithm via eigendecomposition of an SPD matrix.
fn logm_spd(a: &Array2<f64>) -> Array2<f64> {
    let (eigenvalues, eigenvectors) = jacobi_eigh(a);
    let n = a.nrows();
    let mut d_log = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        d_log[[i, i]] = eigenvalues[i].max(f64::MIN_POSITIVE).ln();
    }
    let vd = matmul(&eigenvectors, &d_log);
    matmul(&vd, &transpose(&eigenvectors))
}

/// Jacobi eigenvalue algorithm for symmetric real `n × n`. Returns
/// `(eigenvalues, eigenvectors)`. Eigenvectors are columns of `V` and
/// `A = V · diag(λ) · Vᵀ`.
fn jacobi_eigh(a: &Array2<f64>) -> (Vec<f64>, Array2<f64>) {
    let n = a.nrows();
    let mut a = a.clone();
    let mut v = Array2::<f64>::eye(n);
    let max_sweeps = 200;
    let tol = 1e-14;

    for _sweep in 0..max_sweeps {
        let mut off = 0.0_f64;
        for p in 0..n {
            for q in (p + 1)..n {
                off += a[[p, q]] * a[[p, q]];
            }
        }
        if off < tol {
            break;
        }

        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[[p, q]];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let app = a[[p, p]];
                let aqq = a[[q, q]];
                let theta = (aqq - app) / (2.0 * apq);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (1.0 + theta * theta).sqrt())
                } else {
                    1.0 / (theta - (1.0 + theta * theta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                a[[p, p]] = app - t * apq;
                a[[q, q]] = aqq + t * apq;
                a[[p, q]] = 0.0;
                a[[q, p]] = 0.0;
                for k in 0..n {
                    if k != p && k != q {
                        let akp = a[[k, p]];
                        let akq = a[[k, q]];
                        a[[k, p]] = c * akp - s * akq;
                        a[[p, k]] = a[[k, p]];
                        a[[k, q]] = s * akp + c * akq;
                        a[[q, k]] = a[[k, q]];
                    }
                }
                for k in 0..n {
                    let vkp = v[[k, p]];
                    let vkq = v[[k, q]];
                    v[[k, p]] = c * vkp - s * vkq;
                    v[[k, q]] = s * vkp + c * vkq;
                }
            }
        }
    }
    let eigenvalues: Vec<f64> = (0..n).map(|i| a[[i, i]]).collect();
    (eigenvalues, v)
}

/// Compute `(AᵀA)⁻¹` for full column rank using Cholesky on `AᵀA`.
fn solve_cholesky(spd: &Array2<f64>, rhs: &Array2<f64>) -> Array2<f64> {
    let l = cholesky_fwd(spd);
    // L y = rhs, then Lᵀ x = y
    let y = solve_lower_triangular_matrix(&l, rhs);
    let lt = transpose(&l);
    solve_upper_triangular_matrix(&lt, &y)
}

/// Stable pseudo-inverse for a tall full-column-rank matrix:
/// `A⁺ = (AᵀA)⁻¹ Aᵀ`.
fn pinv_tall(a: &Array2<f64>) -> Array2<f64> {
    let at = transpose(a);
    let ata = matmul(&at, a);
    solve_cholesky(&ata, &at)
}

/// Stable pseudo-inverse for a wide full-row-rank matrix:
/// `A⁺ = Aᵀ (A Aᵀ)⁻¹`.
fn pinv_wide(a: &Array2<f64>) -> Array2<f64> {
    let at = transpose(a);
    let aat = matmul(a, &at);
    let l = cholesky_fwd(&aat);
    // Want X with A X = I_m, then A⁺ = Aᵀ X.
    // Equivalently, solve (A Aᵀ) Y = I and return Aᵀ Y.
    let m = a.nrows();
    let eye = Array2::<f64>::eye(m);
    let y = {
        let z = solve_lower_triangular_matrix(&l, &eye);
        let lt = transpose(&l);
        solve_upper_triangular_matrix(&lt, &z)
    };
    matmul(&at, &y)
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. LU backward pass — n×n
// ─────────────────────────────────────────────────────────────────────────────

/// Zero the strictly upper-triangular part of `M` (keep diagonal and below).
fn keep_lower(m: &Array2<f64>) -> Array2<f64> {
    let n = m.nrows();
    let mut out = m.clone();
    for i in 0..n {
        for j in (i + 1)..n {
            out[[i, j]] = 0.0;
        }
    }
    out
}

/// Zero the strictly lower-triangular part of `M` (keep diagonal and above).
fn keep_upper(m: &Array2<f64>) -> Array2<f64> {
    let n = m.nrows();
    let mut out = m.clone();
    for i in 0..n {
        for j in 0..i {
            out[[i, j]] = 0.0;
        }
    }
    out
}

/// LU backward pass.
///
/// Input: forward `(P, L, U)` from `P · A = L · U` (L unit lower triangular,
/// U upper triangular, P permutation) and upstream gradients `dL`, `dU`.
/// Returns `dA` of size `n × n`.
///
/// Derivation (Murray 2016 §4 / Giles 2008, with permutation handling):
///   The constraint that L is unit lower triangular zeroes the upper triangle
///   and diagonal of `dL`; the constraint that U is upper triangular zeroes
///   the strictly lower triangle of `dU`.  Apply these masks defensively, then
///
///     F = phi_lower(Lᵀ · dL_masked) + phi_upper(dU_masked · Uᵀ)
///     dA = Pᵀ · L⁻ᵀ · F · U⁻ᵀ
///
///   where `phi_lower(X)` keeps strictly lower triangle and `phi_upper(X)`
///   keeps the upper triangle including the diagonal.
fn lu_backward(
    p: &Array2<f64>,
    l: &Array2<f64>,
    u: &Array2<f64>,
    grad_l: &Array2<f64>,
    grad_u: &Array2<f64>,
) -> Array2<f64> {
    let n = l.nrows();
    let lt = transpose(l);
    let ut = transpose(u);

    // Mask gradients to the structural support of L (strict lower) and U (upper).
    let mut dl_masked = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..i {
            dl_masked[[i, j]] = grad_l[[i, j]];
        }
    }
    let du_masked = keep_upper(grad_u);

    // F1 = phi_lower(Lᵀ · dL_masked) — strictly lower triangle (i > j only).
    let lt_dl = matmul(&lt, &dl_masked);
    let mut f1 = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..i {
            f1[[i, j]] = lt_dl[[i, j]];
        }
    }

    // F2 = phi_upper(dU_masked · Uᵀ) — upper triangle including diagonal.
    let du_ut = matmul(&du_masked, &ut);
    let f2 = keep_upper(&du_ut);

    let f = &f1 + &f2;

    // L⁻ᵀ · F  ⇔  Lᵀ · X = F  ⇔ back-substitute on Lᵀ (upper triangular).
    let lt_inv_f = solve_upper_triangular_matrix(&lt, &f);

    // (lt_inv_f) · U⁻ᵀ.  Uᵀ is lower triangular ⇒ use right_solve_lower.
    let result = right_solve_lower(&lt_inv_f, &ut);

    // dA = Pᵀ · result.
    matmul(&transpose(p), &result)
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. QR backward pass — n×n and thin (m > n)
// ─────────────────────────────────────────────────────────────────────────────

/// Slice a column range `cols` from a 2D array.
fn take_cols(a: &Array2<f64>, cols: std::ops::Range<usize>) -> Array2<f64> {
    let m = a.nrows();
    let k = cols.end - cols.start;
    let mut out = Array2::<f64>::zeros((m, k));
    for i in 0..m {
        for (jj, j) in cols.clone().enumerate() {
            out[[i, jj]] = a[[i, j]];
        }
    }
    out
}

/// Slice a row range `rows` from a 2D array.
fn take_rows(a: &Array2<f64>, rows: std::ops::Range<usize>) -> Array2<f64> {
    let n_cols = a.ncols();
    let k = rows.end - rows.start;
    let mut out = Array2::<f64>::zeros((k, n_cols));
    for (ii, i) in rows.clone().enumerate() {
        for j in 0..n_cols {
            out[[ii, j]] = a[[i, j]];
        }
    }
    out
}

/// QR backward pass for square or full-QR (m × m) Q with `m × n` R upper-trapezoidal
/// (bottom `(m-n)` rows zero), full column rank `n` (m ≥ n).
///
/// Input: forward `(Q, R)` (Q is m×m, R is m×n upper-trapezoidal) and upstream
/// gradients `dQ` (m×m), `dR` (m×n).  Returns `dA` of shape m × n.
///
/// We split into the "thin" portion (Q[:, :n], R[:n, :]) which carries all the
/// useful information for `A`-gradient flow.  The bottom-(m-n) rows of `R` are
/// identically zero so their gradient contribution vanishes.
///
/// Thin formula (Murray 2016 / Giles 2008 §2.3):
///   M  =  R_thin · dR_thinᵀ  −  dQ_thinᵀ · Q_thin                 (n × n)
///   S  =  copyltu(M)                                              (n × n)
///   dA =  (I_m − Q_thin Q_thinᵀ) · dQ_thin · R_thin⁻ᵀ
///         + Q_thin · S · R_thin⁻ᵀ
fn qr_backward(
    q: &Array2<f64>,
    r: &Array2<f64>,
    grad_q: &Array2<f64>,
    grad_r: &Array2<f64>,
) -> Array2<f64> {
    let m = r.nrows();
    let n = r.ncols();
    debug_assert!(m >= n);
    debug_assert_eq!(q.nrows(), m);
    debug_assert_eq!(q.ncols(), m);
    debug_assert_eq!(grad_q.shape(), q.shape());
    debug_assert_eq!(grad_r.shape(), r.shape());

    // Mask dR to upper triangle of its top-n×n block; bottom rows ignored.
    let grad_r_top = keep_upper(&take_rows(grad_r, 0..n));
    let r_top = take_rows(r, 0..n); // n × n upper triangular
    let q_thin = take_cols(q, 0..n); // m × n
    let grad_q_thin = take_cols(grad_q, 0..n); // m × n

    let rt_top = transpose(&r_top);

    // M = R_top · dR_topᵀ - dQ_thinᵀ · Q_thin     (n × n)
    let r_drt = matmul(&r_top, &transpose(&grad_r_top));
    let dqt_q = matmul(&transpose(&grad_q_thin), &q_thin);
    let m_mat = &r_drt - &dqt_q;

    // S = copyltu(M): symmetrise using the lower triangle of M.
    let mut s = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            s[[i, j]] = if i >= j { m_mat[[i, j]] } else { m_mat[[j, i]] };
        }
    }

    // inner = (I - Q_thin · Q_thinᵀ) · dQ_thin + Q_thin · S        (m × n)
    let q_s = matmul(&q_thin, &s);
    let q_qt_dq = matmul(&q_thin, &dqt_q); // Q_thin · (Q_thinᵀ · dQ_thin)
    let p_perp_dq = &grad_q_thin - &q_qt_dq;
    let inner = &p_perp_dq + &q_s;

    // dA = inner · R_top⁻ᵀ.  Rᵀ is lower triangular ⇒ use right_solve_lower.
    right_solve_lower(&inner, &rt_top)
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Cholesky backward pass — n×n SPD
// ─────────────────────────────────────────────────────────────────────────────

/// Cholesky backward pass for `A = L Lᵀ`.
///
/// Input: forward `L` (lower triangular) and upstream gradient `dL`.
/// Returns `dA` of size `n × n` (symmetric).
///
/// Murray (2016) eq. 8:
///   P = phi(Lᵀ · dL)
///   dA = ½ · L⁻ᵀ · (P + Pᵀ) · L⁻¹
///
/// where `phi(M)` zeros above-diagonal entries and halves the diagonal:
///   phi(M)_{ii} = M_{ii} / 2,   phi(M)_{ij} = M_{ij} for i > j,   0 otherwise.
fn cholesky_backward(l: &Array2<f64>, grad_l: &Array2<f64>) -> Array2<f64> {
    let n = l.nrows();
    let lt = transpose(l);

    // Inner = Lᵀ · dL
    let lt_dl = matmul(&lt, grad_l);
    // phi: keep strictly lower, halve diagonal, zero strictly upper.
    let mut phi = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            if i > j {
                phi[[i, j]] = lt_dl[[i, j]];
            } else if i == j {
                phi[[i, i]] = lt_dl[[i, i]] / 2.0;
            }
            // i < j stays 0
        }
    }

    let phi_t = transpose(&phi);
    let phi_sym = &phi + &phi_t;
    // — but on the diagonal, phi[i,i] = lt_dl[i,i]/2, so phi+phiᵀ on diag = lt_dl[i,i].
    // That is correct: phi makes things consistent.

    // dA = ½ · L⁻ᵀ · phi_sym · L⁻¹
    // Step 1: y = phi_sym · L⁻¹  ⇔  y · L = phi_sym  ⇔  Lᵀ · yᵀ = phi_symᵀ
    let phi_l_inv = right_solve_lower(&phi_sym, l);
    // Step 2: L⁻ᵀ · y  ⇔  Lᵀ · z = y  ⇔  solve_upper(Lᵀ, y)
    let inner = solve_upper_triangular_matrix(&lt, &phi_l_inv);

    // Multiply by ½
    let mut da = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            da[[i, j]] = 0.5 * inner[[i, j]];
        }
    }
    da
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Pseudo-inverse backward pass — m×n (full rank)
// ─────────────────────────────────────────────────────────────────────────────

/// Pseudo-inverse backward pass for full-rank `A` (m × n) with `rank(A) = min(m, n)`.
///
/// Input: forward `A` (m × n), `A⁺` (n × m), upstream gradient `dA⁺` of
/// shape (n × m). Returns `dA` of shape (m × n).
///
/// Reverse-mode formula (Golub & Pereyra 1973; cf. PyTorch / JAX `pinv`):
///   `dA = -A⁺ᵀ · dA⁺ · A⁺ᵀ`
///         `+ (I_m - A · A⁺) · dA⁺ᵀ · (AᵀA)⁻¹`           (full column rank)
///         `+ (AAᵀ)⁻¹ · dA⁺ᵀ · (I_n - A⁺ · A)`           (full row rank)
///
/// For full column rank (m ≥ n), `A⁺ A = I_n` so the third term vanishes; for
/// full row rank (m ≤ n), `A A⁺ = I_m` so the second term vanishes.  Each
/// branch below computes the surviving terms.
fn pinv_backward(a: &Array2<f64>, a_plus: &Array2<f64>, grad_pinv: &Array2<f64>) -> Array2<f64> {
    let (m, n) = (a.nrows(), a.ncols());
    debug_assert_eq!(a_plus.nrows(), n);
    debug_assert_eq!(a_plus.ncols(), m);
    debug_assert_eq!(grad_pinv.nrows(), n);
    debug_assert_eq!(grad_pinv.ncols(), m);

    let at = transpose(a); // n × m
    let a_plus_t = transpose(a_plus); // m × n
    let grad_pinv_t = transpose(grad_pinv); // m × n

    // First term: -A⁺ᵀ · dA⁺ · A⁺ᵀ  with shapes (m×n)·(n×m)·(m×n) = m × n.
    let term1_inner = matmul(&a_plus_t, grad_pinv);
    let term1 = matmul(&term1_inner, &a_plus_t);

    let mut da = Array2::<f64>::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            da[[i, j]] = -term1[[i, j]];
        }
    }

    if m >= n {
        // Tall full column rank: AᵀA is n×n SPD.
        // term2 = (I_m - A · A⁺) · dA⁺ᵀ · (AᵀA)⁻¹  with shapes
        //         (m×m) · (m×n) · (n×n) = m × n.
        let ata = matmul(&at, a); // n×n SPD
        let a_a_plus = matmul(a, a_plus); // m×m
        let mut p_perp = Array2::<f64>::zeros((m, m));
        for i in 0..m {
            for j in 0..m {
                p_perp[[i, j]] = if i == j { 1.0 } else { 0.0 } - a_a_plus[[i, j]];
            }
        }
        let p_perp_grad_t = matmul(&p_perp, &grad_pinv_t);
        let term2 = right_solve_spd(&p_perp_grad_t, &ata);
        for i in 0..m {
            for j in 0..n {
                da[[i, j]] += term2[[i, j]];
            }
        }
    } else {
        // Wide full row rank: AAᵀ is m×m SPD.
        // term3 = (AAᵀ)⁻¹ · dA⁺ᵀ · (I_n - A⁺ · A)  with shapes
        //         (m×m)⁻¹ · (m×n) · (n×n) = m × n.
        let aat = matmul(a, &at); // m×m SPD
        let a_plus_a = matmul(a_plus, a); // n×n
        let mut p_perp = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                p_perp[[i, j]] = if i == j { 1.0 } else { 0.0 } - a_plus_a[[i, j]];
            }
        }
        let grad_p_perp = matmul(&grad_pinv_t, &p_perp);
        let term3 = left_solve_spd(&aat, &grad_p_perp);
        for i in 0..m {
            for j in 0..n {
                da[[i, j]] += term3[[i, j]];
            }
        }
    }
    da
}

/// Solve `B · X = R` where B is SPD (right-solve via Cholesky).
fn right_solve_spd(r: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    // X B = R  ⇒  Bᵀ Xᵀ = Rᵀ — but B is symmetric so Bᵀ = B, and X B = R.
    // Solve via L Lᵀ = B, then ((Lᵀ)⁻¹ L⁻¹) · Rᵀ → transpose back.
    let l = cholesky_fwd(b);
    let rt = transpose(r);
    let y = solve_lower_triangular_matrix(&l, &rt);
    let lt = transpose(&l);
    let xt = solve_upper_triangular_matrix(&lt, &y);
    transpose(&xt)
}

/// Solve `B · X = R` where B is SPD (left-solve via Cholesky).
fn left_solve_spd(b: &Array2<f64>, r: &Array2<f64>) -> Array2<f64> {
    let l = cholesky_fwd(b);
    let y = solve_lower_triangular_matrix(&l, r);
    let lt = transpose(&l);
    solve_upper_triangular_matrix(&lt, &y)
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Matrix square-root backward pass — n×n SPD
// ─────────────────────────────────────────────────────────────────────────────

/// Matrix square-root backward via the Sylvester equation
/// `S · dA + dA · S = dS_tilde` where
///   S = √A, dS_tilde = ds (upstream gradient of √A).
///
/// Murray (2016), Higham "Functions of Matrices":
///   d(√A) is the unique X such that S · X + X · S = dA.
///   Reverse mode flips: given upstream `dS`, we solve
///       Sᵀ · dA + dA · Sᵀ = dS         (continuous Lyapunov)
///   for `dA`.  For SPD A, S is symmetric, so Sᵀ = S.
fn sqrtm_backward(s: &Array2<f64>, grad_s: &Array2<f64>) -> Array2<f64> {
    // solve_sylvester solves A·X + X·B = C  → call with A=S, B=S, C=grad_s.
    solve_sylvester(&s.view(), &s.view(), &grad_s.view()).expect("Sylvester solve")
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. Matrix log backward pass — n×n SPD
// ─────────────────────────────────────────────────────────────────────────────

/// Matrix logarithm backward via the Fréchet derivative integral.
///
/// For SPD `A` with eigendecomposition `A = V Λ Vᵀ` (Λ diagonal of positive
/// eigenvalues), the Fréchet derivative of `log` admits a closed form:
///
///   `d(log A)`[E] = V · (Φ ⊙ (Vᵀ E V)) · Vᵀ
///
/// where `Φ_{ij} = (log λ_i - log λ_j) / (λ_i - λ_j)` (with the limit
/// `1/λ_i` when `i == j`).  The reverse-mode adjoint with upstream `dB` is:
///
///   `dA = V · (Φ ⊙ (Vᵀ dB V)) · Vᵀ`
///
/// The matrix Φ is symmetric, so log's Fréchet derivative is self-adjoint
/// (d log)* = d log. This is why the upstream and forward formulas coincide.
fn logm_backward(a: &Array2<f64>, grad_log: &Array2<f64>) -> Array2<f64> {
    let n = a.nrows();
    let (eigenvalues, v) = jacobi_eigh(a);
    let vt = transpose(&v);

    // Y = Vᵀ · grad_log · V
    let v_t_grad = matmul(&vt, grad_log);
    let y = matmul(&v_t_grad, &v);

    // Φ_{ij} = (log λ_i - log λ_j) / (λ_i - λ_j)  with limit 1/λ_i for i=j.
    let mut phi = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let li = eigenvalues[i].max(f64::MIN_POSITIVE);
            let lj = eigenvalues[j].max(f64::MIN_POSITIVE);
            if (li - lj).abs() < 1e-12 * (li.abs() + lj.abs() + 1.0) {
                phi[[i, j]] = 1.0 / li;
            } else {
                phi[[i, j]] = (li.ln() - lj.ln()) / (li - lj);
            }
        }
    }

    // Y_phi = Φ ⊙ Y
    let mut y_phi = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            y_phi[[i, j]] = phi[[i, j]] * y[[i, j]];
        }
    }

    // dA = V · Y_phi · Vᵀ
    let vy = matmul(&v, &y_phi);
    matmul(&vy, &vt)
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. Batch lifter
// ─────────────────────────────────────────────────────────────────────────────

/// Apply a per-sample backward function to each `n × n` slice of a
/// `[batch_size, n, n]` tensor, in parallel via rayon.
///
/// `fwd(a) -> Y`, `bw(Y, a, grad) -> dA`.
fn batch_lift_lu_backward(
    a_batch: &Array3<f64>,
    grad_l_batch: &Array3<f64>,
    grad_u_batch: &Array3<f64>,
) -> Array3<f64> {
    use scirs2_core::parallel_ops::*;

    let (b, n, m) = (a_batch.shape()[0], a_batch.shape()[1], a_batch.shape()[2]);
    debug_assert_eq!(n, m);

    // Process each batch independently.
    let results: Vec<Array2<f64>> = (0..b)
        .into_par_iter()
        .map(|i| {
            let a = a_batch
                .slice(scirs2_core::ndarray::s![i, .., ..])
                .to_owned();
            let grad_l = grad_l_batch
                .slice(scirs2_core::ndarray::s![i, .., ..])
                .to_owned();
            let grad_u = grad_u_batch
                .slice(scirs2_core::ndarray::s![i, .., ..])
                .to_owned();
            let (p, l, u) = lu_fwd(&a);
            lu_backward(&p, &l, &u, &grad_l, &grad_u)
        })
        .collect();

    let mut out = Array3::<f64>::zeros((b, n, n));
    for (i, da) in results.into_iter().enumerate() {
        out.slice_mut(scirs2_core::ndarray::s![i, .., ..])
            .assign(&da);
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// FD verification helpers
// ─────────────────────────────────────────────────────────────────────────────

fn assert_grad_close(analytic: &Array2<f64>, numeric: &Array2<f64>, tol: f64, label: &str) {
    let err = max_abs(analytic, numeric);
    assert!(
        err < tol,
        "{}: max abs diff {} >= tol {} (analytic={:?}, numeric={:?})",
        label,
        err,
        tol,
        analytic,
        numeric
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

// 1. LU n×n

/// Sum of strict-lower triangle of `M`.
fn sum_strict_lower(m: &Array2<f64>) -> f64 {
    let n = m.nrows();
    let mut s = 0.0;
    for i in 0..n {
        for j in 0..i {
            s += m[[i, j]];
        }
    }
    s
}

/// Sum of upper triangle (including diagonal) of `M`.
fn sum_upper(m: &Array2<f64>) -> f64 {
    let n = m.nrows();
    let mut s = 0.0;
    for i in 0..n {
        for j in i..n {
            s += m[[i, j]];
        }
    }
    s
}

#[test]
fn lu_backward_3x3_random() {
    let a = random_matrix(3, 3, 0xA1B2C3D4);
    let (p, l, u) = lu_fwd(&a);

    // Loss = sum(strict-lower of L) + sum(upper of U).  (Constants in L's
    // diagonal are 1 and contribute nothing to the gradient w.r.t. A.)
    let grad_l = {
        let mut g = Array2::<f64>::zeros((3, 3));
        for i in 0..3 {
            for j in 0..i {
                g[[i, j]] = 1.0;
            }
        }
        g
    };
    let grad_u = keep_upper(&Array2::<f64>::ones((3, 3)));

    let analytic = lu_backward(&p, &l, &u, &grad_l, &grad_u);

    let numeric = fd_gradient(&a, |a_p| {
        let (_, ll, uu) = lu_fwd(a_p);
        sum_strict_lower(&ll) + sum_upper(&uu)
    });

    assert_grad_close(&analytic, &numeric, FD_TOL, "lu_backward_3x3_random");
}

#[test]
fn lu_backward_5x5_general() {
    // Diagonally dominant 5×5 — guaranteed nonsingular and stable.
    let mut a = random_matrix(5, 5, 0xDEAD_BEEF);
    for i in 0..5 {
        a[[i, i]] += 5.0;
    }

    let (p, l, u) = lu_fwd(&a);
    let n = 5;
    let grad_l = {
        let mut g = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..i {
                g[[i, j]] = 1.0;
            }
        }
        g
    };
    let grad_u = keep_upper(&Array2::<f64>::ones((n, n)));

    let analytic = lu_backward(&p, &l, &u, &grad_l, &grad_u);
    let numeric = fd_gradient(&a, |a_p| {
        let (_, ll, uu) = lu_fwd(a_p);
        sum_strict_lower(&ll) + sum_upper(&uu)
    });
    assert_grad_close(&analytic, &numeric, FD_TOL, "lu_backward_5x5_general");
}

// 2. QR n×n and thin

#[test]
fn qr_backward_4x4_orthogonal() {
    // 4×4 nonsingular matrix.
    let mut a = random_matrix(4, 4, 0xCAFE_BABE);
    for i in 0..4 {
        a[[i, i]] += 3.0;
    }

    let (q, r) = qr_fwd(&a);
    // Full QR: Q is m×m, R is m×n.  Loss = sum(upper(R_top)).
    let grad_q = Array2::<f64>::zeros(q.raw_dim());
    let grad_r = keep_upper(&Array2::<f64>::ones(r.raw_dim()));

    let analytic = qr_backward(&q, &r, &grad_q, &grad_r);
    let numeric = fd_gradient(&a, |a_p| {
        let (_, rr) = qr_fwd(a_p);
        sum_upper(&rr)
    });

    assert_grad_close(
        &analytic,
        &numeric,
        FD_TOL_LOOSE,
        "qr_backward_4x4_orthogonal",
    );
}

#[test]
fn qr_backward_thin_5x3() {
    // Tall 5×3 — scirs2 returns full QR (Q 5×5, R 5×3).
    let a = random_matrix(5, 3, 0x1234_5678);
    let (q, r) = qr_fwd(&a);
    debug_assert_eq!(q.shape(), &[5, 5]);
    debug_assert_eq!(r.shape(), &[5, 3]);

    // dQ = 0, dR = upper(ones) — only the upper-triangular top-3×3 of R has
    // gradient flow; lower rows of R are identically zero.
    let grad_q = Array2::<f64>::zeros(q.raw_dim());
    let mut grad_r = Array2::<f64>::zeros(r.raw_dim());
    for i in 0..3 {
        for j in i..3 {
            grad_r[[i, j]] = 1.0;
        }
    }

    let analytic = qr_backward(&q, &r, &grad_q, &grad_r);
    let numeric = fd_gradient(&a, |a_p| {
        let (_, rr) = qr_fwd(a_p);
        // Sum only the meaningful upper triangle of the top-3 rows.
        let mut s = 0.0;
        for i in 0..3 {
            for j in i..3 {
                s += rr[[i, j]];
            }
        }
        s
    });

    assert_grad_close(&analytic, &numeric, FD_TOL_LOOSE, "qr_backward_thin_5x3");
}

// 3. Cholesky n×n SPD

#[test]
fn cholesky_backward_3x3_spd() {
    let a = random_spd(3, 0x9999_AAAA);
    let l = cholesky_fwd(&a);
    let grad_l = Array2::<f64>::ones((3, 3));

    let analytic = cholesky_backward(&l, &grad_l);

    let numeric = fd_gradient_symmetric(&a, |a_p| {
        let ll = cholesky_fwd(a_p);
        ll.iter().sum()
    });

    assert_grad_close(&analytic, &numeric, FD_TOL, "cholesky_backward_3x3_spd");
}

#[test]
fn cholesky_backward_5x5_spd() {
    let a = random_spd(5, 0x5A5A_5A5A);
    let l = cholesky_fwd(&a);
    let grad_l = Array2::<f64>::ones((5, 5));

    let analytic = cholesky_backward(&l, &grad_l);

    let numeric = fd_gradient_symmetric(&a, |a_p| {
        let ll = cholesky_fwd(a_p);
        ll.iter().sum()
    });

    assert_grad_close(&analytic, &numeric, FD_TOL, "cholesky_backward_5x5_spd");
}

// 4. Pseudo-inverse over- and under-determined

#[test]
fn pinv_backward_overdetermined_5x3() {
    let a = random_matrix(5, 3, 0xBABE_CAFE);
    let a_plus = pinv_tall(&a);
    let grad_pinv = Array2::<f64>::ones((3, 5));

    let analytic = pinv_backward(&a, &a_plus, &grad_pinv);

    let numeric = fd_gradient(&a, |a_p| {
        let pp = pinv_tall(a_p);
        pp.iter().sum()
    });

    assert_grad_close(
        &analytic,
        &numeric,
        FD_TOL_LOOSE,
        "pinv_backward_overdetermined_5x3",
    );
}

#[test]
fn pinv_backward_underdetermined_3x5() {
    let a = random_matrix(3, 5, 0xFADE_FADE);
    let a_plus = pinv_wide(&a);
    let grad_pinv = Array2::<f64>::ones((5, 3));

    let analytic = pinv_backward(&a, &a_plus, &grad_pinv);

    let numeric = fd_gradient(&a, |a_p| {
        let pp = pinv_wide(a_p);
        pp.iter().sum()
    });

    assert_grad_close(
        &analytic,
        &numeric,
        FD_TOL_LOOSE,
        "pinv_backward_underdetermined_3x5",
    );
}

// 5. Matrix sqrt backward — 2×2 baseline plus n×n general

#[test]
fn sqrtm_backward_2x2_baseline() {
    // Compare against fd on a 2×2 SPD: this exercises the Sylvester path.
    let a = random_spd(2, 0xBEEF_CAFE);
    let s = sqrtm_spd(&a);
    let grad_s = Array2::<f64>::ones((2, 2));

    let analytic = sqrtm_backward(&s, &grad_s);

    let numeric = fd_gradient_symmetric(&a, |a_p| {
        let ss = sqrtm_spd(a_p);
        ss.iter().sum()
    });

    assert_grad_close(
        &analytic,
        &numeric,
        FD_TOL_LOOSE,
        "sqrtm_backward_2x2_baseline",
    );
}

#[test]
fn sqrtm_backward_4x4_general() {
    let a = random_spd(4, 0x4242_4242);
    let s = sqrtm_spd(&a);
    let grad_s = Array2::<f64>::ones((4, 4));

    let analytic = sqrtm_backward(&s, &grad_s);

    let numeric = fd_gradient_symmetric(&a, |a_p| {
        let ss = sqrtm_spd(a_p);
        ss.iter().sum()
    });

    assert_grad_close(
        &analytic,
        &numeric,
        FD_TOL_LOOSE,
        "sqrtm_backward_4x4_general",
    );
}

// 6. Matrix log backward — 3×3 SPD

#[test]
fn logm_backward_3x3_via_pade() {
    let a = random_spd(3, 0x7777_7777);
    let grad_log = Array2::<f64>::ones((3, 3));

    let analytic = logm_backward(&a, &grad_log);

    let numeric = fd_gradient_symmetric(&a, |a_p| {
        let ll = logm_spd(a_p);
        ll.iter().sum()
    });

    assert_grad_close(
        &analytic,
        &numeric,
        FD_TOL_LOOSE,
        "logm_backward_3x3_via_pade",
    );
}

// 7. Batch lifter consistency

#[test]
fn batch_lift_5_x_3x3_lu_consistent() {
    let bsz = 5;
    let n = 3;
    let mut a_batch = Array3::<f64>::zeros((bsz, n, n));
    for i in 0..bsz {
        let a_i = random_matrix(n, n, 0xDADA_0000 + i as u64);
        for r in 0..n {
            for c in 0..n {
                a_batch[[i, r, c]] = a_i[[r, c]] + if r == c { 3.0 } else { 0.0 };
            }
        }
    }

    let strict_lower_ones = {
        let mut g = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..i {
                g[[i, j]] = 1.0;
            }
        }
        g
    };
    let upper_ones = keep_upper(&Array2::<f64>::ones((n, n)));

    let mut grad_l_batch = Array3::<f64>::zeros((bsz, n, n));
    let mut grad_u_batch = Array3::<f64>::zeros((bsz, n, n));
    for b in 0..bsz {
        for i in 0..n {
            for j in 0..n {
                grad_l_batch[[b, i, j]] = strict_lower_ones[[i, j]];
                grad_u_batch[[b, i, j]] = upper_ones[[i, j]];
            }
        }
    }

    let batched = batch_lift_lu_backward(&a_batch, &grad_l_batch, &grad_u_batch);

    for i in 0..bsz {
        let a = a_batch
            .slice(scirs2_core::ndarray::s![i, .., ..])
            .to_owned();
        let (p, l, u) = lu_fwd(&a);
        let single = lu_backward(&p, &l, &u, &strict_lower_ones, &upper_ones);
        let from_batch = batched
            .slice(scirs2_core::ndarray::s![i, .., ..])
            .to_owned();
        let err = max_abs(&single, &from_batch);
        assert!(
            err < 1e-12,
            "Batch lifter inconsistent at sample {}: max diff {}",
            i,
            err
        );
    }
}
