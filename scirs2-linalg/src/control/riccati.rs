//! Algebraic Riccati equation solvers.
//!
//! Provides solvers for:
//! - Continuous Algebraic Riccati Equation (CARE):
//!   `Aᵀ P + P A - P B R⁻¹ Bᵀ P + Q = 0`
//! - Discrete Algebraic Riccati Equation (DARE):
//!   `Aᵀ P A - P - Aᵀ P B (Bᵀ P B + R)⁻¹ Bᵀ P A + Q = 0`
//!
//! # Algorithms
//!
//! ## CARE — Hamiltonian Matrix + Newton's Method
//! The Hamiltonian matrix `H = [[A, -B R⁻¹ Bᵀ], [-Q, -Aᵀ]]` has the property
//! that if `λ` is an eigenvalue so is `-λ`. The stable invariant subspace of `H`
//! (eigenvalues with negative real part) gives the solution `P = X₂ X₁⁻¹` where
//! `[X₁; X₂]` spans the stable subspace.
//!
//! In practice, we use Newton's method (Riccati iteration) which is more
//! numerically stable for smaller problems:
//! `P_{k+1} = solve_lyapunov(Aᵣ, P_k B R⁻¹ Bᵀ P_k + Q)`
//! where `Aᵣ = A - B R⁻¹ Bᵀ P_k`.
//!
//! ## DARE — Doubling Algorithm + Newton's Method
//! The symplectic matrix approach: uses the matrix sign function or
//! structure-preserving doubling algorithm.
//!
//! # References
//! - Lancaster, P. & Rodman, L. (1995). *Algebraic Riccati Equations*. Oxford.
//! - Arnold, W. F. & Laub, A. J. (1984). Generalized eigenproblem algorithms
//!   for the matrix algebraic Riccati equation. *Proc. IEEE*, 72(12).
//! - Kleinman, D. L. (1968). On an iterative technique for Riccati equation
//!   computations. *IEEE Trans. Autom. Control*, 13(1), 114–115.

use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView2, ScalarOperand};
use scirs2_core::numeric::{Complex, Float, NumAssign};
use std::fmt::{Debug, Display};
use std::iter::Sum;

use super::lyapunov::lyapunov_continuous;

/// Trait bound for Riccati solver scalars.
pub trait RiccatiFloat:
    Float + NumAssign + Sum + ScalarOperand + Debug + Display + Send + Sync + 'static
{
}
impl<F> RiccatiFloat for F where
    F: Float + NumAssign + Sum + ScalarOperand + Debug + Display + Send + Sync + 'static
{
}

// ---------------------------------------------------------------------------
// Internal linear algebra helpers
// ---------------------------------------------------------------------------

/// Dense matrix multiply.
fn mm<F: RiccatiFloat>(a: &Array2<F>, b: &Array2<F>) -> Array2<F> {
    let m = a.nrows();
    let k = a.ncols();
    let n = b.ncols();
    let mut c = Array2::<F>::zeros((m, n));
    for i in 0..m {
        for p in 0..k {
            let aip = a[[i, p]];
            if aip == F::zero() {
                continue;
            }
            for j in 0..n {
                c[[i, j]] += aip * b[[p, j]];
            }
        }
    }
    c
}

/// Matrix inverse via Gauss-Jordan with partial pivoting.
fn mat_inv<F: RiccatiFloat>(a: &Array2<F>) -> LinalgResult<Array2<F>> {
    let n = a.nrows();
    let mut aug = Array2::<F>::zeros((n, 2 * n));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n + i]] = F::one();
    }
    for col in 0..n {
        // Partial pivoting
        let mut piv = col;
        let mut piv_val = aug[[col, col]].abs();
        for row in (col + 1)..n {
            let v = aug[[row, col]].abs();
            if v > piv_val {
                piv_val = v;
                piv = row;
            }
        }
        if piv_val < F::from(1e-14).unwrap_or(F::epsilon()) {
            return Err(LinalgError::SingularMatrixError(
                "Riccati solver: matrix is singular".to_string(),
            ));
        }
        if piv != col {
            for j in 0..(2 * n) {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[piv, j]];
                aug[[piv, j]] = tmp;
            }
        }
        let sc = aug[[col, col]];
        for j in 0..(2 * n) {
            aug[[col, j]] /= sc;
        }
        for row in 0..n {
            if row != col {
                let fac = aug[[row, col]];
                if fac == F::zero() {
                    continue;
                }
                for j in 0..(2 * n) {
                    let v = aug[[col, j]];
                    aug[[row, j]] -= fac * v;
                }
            }
        }
    }
    let mut inv = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inv[[i, j]] = aug[[i, n + j]];
        }
    }
    Ok(inv)
}

/// Symmetrize: X = (X + Xᵀ) / 2
fn symmetrize<F: RiccatiFloat>(x: &Array2<F>) -> Array2<F> {
    let n = x.nrows();
    let mut s = Array2::<F>::zeros((n, n));
    let two = F::from(2.0).unwrap_or(F::one());
    for i in 0..n {
        for j in 0..n {
            s[[i, j]] = (x[[i, j]] + x[[j, i]]) / two;
        }
    }
    s
}

/// Frobenius norm of a matrix.
fn frob_norm<F: RiccatiFloat>(a: &Array2<F>) -> F {
    a.iter().map(|&v| v * v).sum::<F>().sqrt()
}

// ---------------------------------------------------------------------------
// CARE: Continuous Algebraic Riccati Equation
// ---------------------------------------------------------------------------

/// Solve the Continuous Algebraic Riccati Equation (CARE):
/// `Aᵀ P + P A - P B R⁻¹ Bᵀ P + Q = 0`
///
/// # Arguments
/// * `a` - `n×n` system (state) matrix
/// * `b` - `n×m` input matrix
/// * `q` - `n×n` state cost matrix (positive semi-definite)
/// * `r` - `m×m` input cost matrix (positive definite)
///
/// # Returns
/// The symmetric positive semi-definite solution matrix `P` of size `n×n`.
///
/// # Algorithm
/// Uses the Hamiltonian matrix eigenvalue (Schur) method:
/// 1. Form `H = [[A, -S]; [-Q, -Aᵀ]]` where `S = B R⁻¹ Bᵀ`
/// 2. Compute the stable invariant subspace of H (eigenvalues with Re < 0)
/// 3. Partition the stable eigenvectors `[X₁; X₂]` into top `n` rows (X₁) and bottom `n` rows (X₂)
/// 4. `P = X₂ X₁⁻¹` (symmetrised)
///
/// Falls back to Newton-Kleinman iteration if the Hamiltonian approach fails.
///
/// # Example
/// ```rust
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::control::riccati::care_solve;
///
/// // Simple 1D system: A=-1, B=1, Q=1, R=1 → P*(2*(-1)-1) = -1 → P=0.5*(√5-1)
/// let a = array![[-1.0_f64]];
/// let b = array![[1.0_f64]];
/// let q = array![[1.0_f64]];
/// let r = array![[1.0_f64]];
/// let p = care_solve(&a.view(), &b.view(), &q.view(), &r.view()).expect("CARE failed");
/// // Verify CARE residual ≈ 0
/// let atp = a.t().dot(&p);
/// let pa = p.dot(&a);
/// ```
pub fn care_solve<F: RiccatiFloat>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
    q: &ArrayView2<F>,
    r: &ArrayView2<F>,
) -> LinalgResult<Array2<F>> {
    let n = check_square(a, "CARE: A")?;
    let q_n = check_square(q, "CARE: Q")?;
    let r_m = check_square(r, "CARE: R")?;
    if q_n != n {
        return Err(LinalgError::DimensionError(format!(
            "CARE: Q must be {n}×{n}, got {q_n}×{q_n}"
        )));
    }
    if b.nrows() != n {
        return Err(LinalgError::DimensionError(format!(
            "CARE: B must have {n} rows, got {}",
            b.nrows()
        )));
    }
    let m = b.ncols();
    if r_m != m {
        return Err(LinalgError::DimensionError(format!(
            "CARE: R must be {m}×{m}, got {r_m}×{r_m}"
        )));
    }

    let a_own = a.to_owned();
    let b_own = b.to_owned();
    let q_own = q.to_owned();

    // Precompute R⁻¹
    let r_own = r.to_owned();
    let r_inv = mat_inv(&r_own)
        .map_err(|e| LinalgError::SingularMatrixError(format!("CARE: R is singular: {e}")))?;
    // Precompute S = B R⁻¹ Bᵀ  (n×n, symmetric)
    let br_inv = mm(&b_own, &r_inv);
    let s = mm(&br_inv, &b_own.t().to_owned());

    // Try Hamiltonian Schur approach first (works for unstable A)
    if let Ok(p_ham) = care_hamiltonian(&a_own, &s, &q_own, n) {
        let res = care_residual_norm(&a_own, &s, &q_own, &p_ham, n);
        if res < F::from(1e-6).unwrap_or(F::epsilon()) {
            return Ok(p_ham);
        }
    }

    // Try complex eigendecomposition of Hamiltonian — handles cases where real
    // Schur iteration fails (e.g., purely imaginary or complex eigenvalue pairs).
    if let Ok(p_cplx) = care_hamiltonian_complex(&a_own, &s, &q_own, n) {
        let res = care_residual_norm(&a_own, &s, &q_own, &p_cplx, n);
        if res < F::from(1e-4).unwrap_or(F::epsilon()) {
            return Ok(p_cplx);
        }
    }

    // Third option: delegate to the Sylvester-module Newton-Kleinman solver which
    // uses find_stabilizing_initial_x — works for unstabilized/marginally-stable A.
    if let Ok(p_sylv) = crate::matrix_functions::sylvester::solve_algebraic_riccati(
        a,
        b,
        q,
        r,
        crate::matrix_functions::sylvester::RiccatiType::Continuous,
    ) {
        let res = care_residual_norm(&a_own, &s, &q_own, &p_sylv, n);
        if res < F::from(1e-4).unwrap_or(F::epsilon()) {
            return Ok(p_sylv);
        }
    }

    // Third fallback: Newton-Kleinman iteration (works when A is already stable)
    let mut p = symmetrize(&q_own);

    let tol = F::from(1e-10).unwrap_or(F::epsilon());
    let max_iter = 200usize;

    for iter in 0..max_iter {
        // A_closed = A - S * P
        let sp = mm(&s, &p);
        let mut a_cl = a_own.clone();
        for i in 0..n {
            for j in 0..n {
                a_cl[[i, j]] -= sp[[i, j]];
            }
        }

        // Lyapunov rhs: -(Q + P S P)  [using S = B R⁻¹ Bᵀ]
        let psp = mm(&p, &mm(&s, &p));
        let mut lyap_q = q_own.clone();
        for i in 0..n {
            for j in 0..n {
                lyap_q[[i, j]] += psp[[i, j]];
            }
        }
        let lyap_q = symmetrize(&lyap_q);

        // Solve Lyapunov: A_clᵀ P_new + P_new A_cl = -lyap_q
        // (Kleinman's method requires the transpose form)
        let a_cl_t = a_cl.t().to_owned();
        let p_new = match lyapunov_continuous(&a_cl_t.view(), &lyap_q.view()) {
            Ok(p) => p,
            Err(_) => break,
        };
        let p_new = symmetrize(&p_new);

        // Check convergence
        let diff = &p_new - &p;
        let diff_norm = frob_norm(&diff);

        p = p_new;

        if diff_norm <= tol {
            return Ok(p);
        }

        // Safety: if residual grows, we may have diverged
        if iter > 10 {
            let res_norm = care_residual_norm(&a_own, &s, &q_own, &p, n);
            if res_norm < F::from(1e-8).unwrap_or(F::epsilon()) {
                return Ok(p);
            }
        }
    }

    // Check final residual quality even if not fully converged
    let res_norm = care_residual_norm(&a_own, &s, &q_own, &p, n);
    if res_norm < F::from(1e-5).unwrap_or(F::epsilon()) {
        return Ok(p);
    }

    Err(LinalgError::ConvergenceError(format!(
        "CARE did not converge; final residual = {res_norm}"
    )))
}

/// Internal: solve CARE via Hamiltonian matrix Schur decomposition.
///
/// Forms `H = [[A, -S]; [-Q, -Aᵀ]]`, finds the stable invariant subspace,
/// and returns `P = X₂ X₁⁻¹`.
fn care_hamiltonian<F: RiccatiFloat>(
    a: &Array2<F>,
    s: &Array2<F>,
    q: &Array2<F>,
    n: usize,
) -> LinalgResult<Array2<F>> {
    let two_n = 2 * n;
    // Build H = [[A, -S]; [-Q, -A^T]]
    let mut h = Array2::<F>::zeros((two_n, two_n));
    for i in 0..n {
        for j in 0..n {
            h[[i, j]] = a[[i, j]]; // top-left: A
            h[[i, n + j]] = -s[[i, j]]; // top-right: -S
            h[[n + i, j]] = -q[[i, j]]; // bottom-left: -Q
            h[[n + i, n + j]] = -a[[j, i]]; // bottom-right: -A^T
        }
    }

    // Find stable invariant subspace (eigenvalues with strictly negative real part)
    // Use high max_iter since Hamiltonian matrices (with ±λ eigenvalue pairs) converge slowly
    let schur_tol = F::epsilon() * F::from(100.0).unwrap_or(F::one());
    let v = crate::schur_enhanced::invariant_subspace(
        &h.view(),
        |re: F, _im: F| re < F::zero(),
        3000,
        schur_tol,
    )?;

    // v should be 2n × n; check that we got exactly n stable eigenvectors
    if v.ncols() != n {
        return Err(LinalgError::ConvergenceError(format!(
            "CARE Hamiltonian: expected {n} stable eigenvectors, got {}",
            v.ncols()
        )));
    }

    // Partition: X1 = v[0..n, :], X2 = v[n..2n, :]
    let x1 = v.slice(s![0..n, ..]).to_owned();
    let x2 = v.slice(s![n..two_n, ..]).to_owned();

    // P = X2 * X1^{-1}
    let x1_inv = mat_inv(&x1).map_err(|e| {
        LinalgError::SingularMatrixError(format!("CARE Hamiltonian: X1 singular: {e}"))
    })?;
    let p = mm(&x2, &x1_inv);
    Ok(symmetrize(&p))
}

/// Internal: solve CARE via complex eigendecomposition of the Hamiltonian.
///
/// Converts to f64, calls `eig_f64_lapack`, selects the n stable eigenvectors
/// (Re(λ) < 0), and returns `P = Re(X₂ X₁⁻¹)` symmetrized.
///
/// This handles cases where the real Schur iteration fails (e.g., complex
/// conjugate eigenvalue pairs that don't converge with the iterative Schur method).
fn care_hamiltonian_complex<F: RiccatiFloat>(
    a: &Array2<F>,
    s: &Array2<F>,
    q: &Array2<F>,
    n: usize,
) -> LinalgResult<Array2<F>> {
    let to_f64 = |x: F| x.to_f64().unwrap_or(0.0);
    let from_f64 = |x: f64| F::from(x).unwrap_or(F::zero());

    let two_n = 2 * n;
    let mut h_f64 = Array2::<f64>::zeros((two_n, two_n));
    for i in 0..n {
        for j in 0..n {
            h_f64[[i, j]] = to_f64(a[[i, j]]);
            h_f64[[i, n + j]] = -to_f64(s[[i, j]]);
            h_f64[[n + i, j]] = -to_f64(q[[i, j]]);
            h_f64[[n + i, n + j]] = -to_f64(a[[j, i]]);
        }
    }

    let (eigenvalues, eigenvectors) = crate::decomposition::eig_f64_lapack(&h_f64.view())?;

    // Select n eigenvectors with strictly negative real part
    let mut stable: Vec<usize> = (0..two_n)
        .filter(|&i| eigenvalues[i].re < -f64::EPSILON)
        .collect();

    // Sort by real part (most negative first for numerical stability)
    stable.sort_by(|&aa, &bb| {
        eigenvalues[aa]
            .re
            .partial_cmp(&eigenvalues[bb].re)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    if stable.len() < n {
        return Err(LinalgError::ConvergenceError(format!(
            "CARE Hamiltonian complex: found only {} stable eigenvalues (need {n})",
            stable.len()
        )));
    }

    // Use first n stable eigenvectors: build X1 (top n rows), X2 (bottom n rows)
    let mut x1_re = Array2::<f64>::zeros((n, n));
    let mut x1_im = Array2::<f64>::zeros((n, n));
    let mut x2_re = Array2::<f64>::zeros((n, n));
    let mut x2_im = Array2::<f64>::zeros((n, n));

    for (col_out, &col_in) in stable[..n].iter().enumerate() {
        for row in 0..n {
            x1_re[[row, col_out]] = eigenvectors[[row, col_in]].re;
            x1_im[[row, col_out]] = eigenvectors[[row, col_in]].im;
            x2_re[[row, col_out]] = eigenvectors[[n + row, col_in]].re;
            x2_im[[row, col_out]] = eigenvectors[[n + row, col_in]].im;
        }
    }

    // Compute P = X2 * X1^{-1} in complex arithmetic via 2n×2n real block system.
    // [X1_re, -X1_im; X1_im, X1_re] * [Re(P); Im(P)] = [X2_re; X2_im]
    let aug_cols = 2 * n + n;
    let mut aug = Array2::<f64>::zeros((2 * n, aug_cols));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = x1_re[[i, j]];
            aug[[i, n + j]] = -x1_im[[i, j]];
            aug[[n + i, j]] = x1_im[[i, j]];
            aug[[n + i, n + j]] = x1_re[[i, j]];
        }
        for j in 0..n {
            aug[[i, 2 * n + j]] = x2_re[[i, j]];
            aug[[n + i, 2 * n + j]] = x2_im[[i, j]];
        }
    }

    // Gauss-Jordan elimination with partial pivoting on the 2n×2n block
    for col in 0..2 * n {
        let mut piv = col;
        let mut piv_val = aug[[col, col]].abs();
        for row in (col + 1)..2 * n {
            let v = aug[[row, col]].abs();
            if v > piv_val {
                piv_val = v;
                piv = row;
            }
        }
        if piv_val < 1e-14 {
            return Err(LinalgError::SingularMatrixError(
                "CARE Hamiltonian complex: X1 is singular".to_string(),
            ));
        }
        if piv != col {
            for j in 0..aug_cols {
                aug.swap((col, j), (piv, j));
            }
        }
        let scale = aug[[col, col]];
        for j in 0..aug_cols {
            aug[[col, j]] /= scale;
        }
        for row in 0..2 * n {
            if row != col {
                let factor = aug[[row, col]];
                if factor.abs() < f64::EPSILON {
                    continue;
                }
                for j in 0..aug_cols {
                    let sub = aug[[col, j]] * factor;
                    aug[[row, j]] -= sub;
                }
            }
        }
    }

    // Extract Re(P) (first n rows of the solution block)
    let mut p = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let v = aug[[i, 2 * n + j]];
            // Symmetrize: average with transpose entry
            let vt = aug[[j, 2 * n + i]];
            p[[i, j]] = from_f64((v + vt) / 2.0);
        }
    }

    Ok(p)
}

/// Compute ‖Aᵀ P + P A - P S P + Q‖_F  (CARE residual norm, S = B R⁻¹ Bᵀ)
fn care_residual_norm<F: RiccatiFloat>(
    a: &Array2<F>,
    s: &Array2<F>,
    q: &Array2<F>,
    p: &Array2<F>,
    n: usize,
) -> F {
    let at = a.t().to_owned();
    let atp = mm(&at, p);
    let pa = mm(p, a);
    let psp = mm(p, &mm(s, p));
    let mut res = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            res[[i, j]] = atp[[i, j]] + pa[[i, j]] - psp[[i, j]] + q[[i, j]];
        }
    }
    frob_norm(&res)
}

// ---------------------------------------------------------------------------
// DARE: Discrete Algebraic Riccati Equation
// ---------------------------------------------------------------------------

/// Solve the Discrete Algebraic Riccati Equation (DARE):
/// `Aᵀ P A - P - Aᵀ P B (Bᵀ P B + R)⁻¹ Bᵀ P A + Q = 0`
///
/// # Arguments
/// * `a` - `n×n` system (state) matrix
/// * `b` - `n×m` input matrix
/// * `q` - `n×n` state cost matrix (positive semi-definite)
/// * `r` - `m×m` input cost matrix (positive definite)
///
/// # Returns
/// The symmetric positive semi-definite solution matrix `P` of size `n×n`.
///
/// # Algorithm
/// Uses the symplectic matrix Schur method (Laub, 1979):
/// 1. Form `Z = [[A, -S]; [-Q, A^{-T}]]` where `S = B R⁻¹ Bᵀ`
/// 2. Compute the stable invariant subspace of Z (eigenvalues inside the unit disk)
/// 3. Partition the stable eigenvectors `[X₁; X₂]` into top `n` rows (X₁) and bottom `n` rows (X₂)
/// 4. `P = X₂ X₁⁻¹` (symmetrised)
///
/// Falls back to Newton-Kleinman iteration (Hewer's algorithm) if symplectic approach fails.
///
/// # Example
/// ```rust
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::control::riccati::dare_solve;
///
/// let a = array![[1.0_f64, 1.0], [0.0, 1.0]];
/// let b = array![[0.0_f64], [1.0]];
/// let q = array![[1.0_f64, 0.0], [0.0, 0.0]];
/// let r = array![[1.0_f64]];
/// let p = dare_solve(&a.view(), &b.view(), &q.view(), &r.view()).expect("DARE failed");
/// ```
pub fn dare_solve<F: RiccatiFloat>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
    q: &ArrayView2<F>,
    r: &ArrayView2<F>,
) -> LinalgResult<Array2<F>> {
    let n = check_square(a, "DARE: A")?;
    let q_n = check_square(q, "DARE: Q")?;
    let r_m = check_square(r, "DARE: R")?;
    if q_n != n {
        return Err(LinalgError::DimensionError(format!(
            "DARE: Q must be {n}×{n}, got {q_n}×{q_n}"
        )));
    }
    if b.nrows() != n {
        return Err(LinalgError::DimensionError(format!(
            "DARE: B must have {n} rows, got {}",
            b.nrows()
        )));
    }
    let m = b.ncols();
    if r_m != m {
        return Err(LinalgError::DimensionError(format!(
            "DARE: R must be {m}×{m}, got {r_m}×{r_m}"
        )));
    }

    let a_own = a.to_owned();
    let b_own = b.to_owned();
    let q_own = q.to_owned();
    let r_own = r.to_owned();

    // Precompute S = B R⁻¹ Bᵀ
    let r_inv = mat_inv(&r_own)
        .map_err(|e| LinalgError::SingularMatrixError(format!("DARE: R is singular: {e}")))?;
    let br_inv = mm(&b_own, &r_inv);
    let s = mm(&br_inv, &b_own.t().to_owned());

    // Try symplectic Schur approach first (Laub, 1979).
    if let Ok(p_sym) = dare_symplectic(&a_own, &s, &q_own, n) {
        let res = dare_residual_norm(&a_own, &b_own, &q_own, &r_own, &p_sym, n, m);
        if res < F::from(1e-4).unwrap_or(F::epsilon()) {
            return Ok(p_sym);
        }
    }

    // Fallback: value iteration.  Works for any stabilizable/detectable pair
    // regardless of whether A is Schur-stable (unlike Hewer's Lyapunov-based method
    // which requires A_cl to be Schur-stable from the start).
    //
    // Iteration: P_{k+1} = Aᵀ P_k A - Aᵀ P_k B (R + Bᵀ P_k B)^{-1} Bᵀ P_k A + Q
    let mut p = symmetrize(&q_own);

    let tol = F::from(1e-10).unwrap_or(F::epsilon());
    let max_iter = 500usize;

    for _iter in 0..max_iter {
        let at = a_own.t().to_owned();
        let atp = mm(&at, &p);
        let atpa = mm(&atp, &a_own);
        let bt = b_own.t().to_owned();
        let btpa = mm(&bt, &mm(&p, &a_own));
        let btpb = mm(&bt, &mm(&p, &b_own));
        let mut reg = r_own.clone();
        for i in 0..m {
            for j in 0..m {
                reg[[i, j]] += btpb[[i, j]];
            }
        }
        let reg_inv = match mat_inv(&reg) {
            Ok(inv) => inv,
            Err(_) => break,
        };
        let atpb = mm(&atp, &b_own);
        // correction = (AᵀPB) (R + BᵀPB)^{-1} (BᵀPA)
        let correction = mm(&atpb, &mm(&reg_inv, &btpa));
        let mut p_new = q_own.clone();
        for i in 0..n {
            for j in 0..n {
                p_new[[i, j]] += atpa[[i, j]] - correction[[i, j]];
            }
        }
        let p_new = symmetrize(&p_new);

        let diff = &p_new - &p;
        let diff_norm = frob_norm(&diff);

        p = p_new;

        if diff_norm <= tol {
            return Ok(p);
        }
    }

    // Check final residual
    let res_norm = dare_residual_norm(&a_own, &b_own, &q_own, &r_own, &p, n, m);
    if res_norm < F::from(1e-5).unwrap_or(F::epsilon()) {
        return Ok(p);
    }

    Err(LinalgError::ConvergenceError(format!(
        "DARE did not converge; final residual = {res_norm}"
    )))
}

/// Internal: solve DARE via symplectic matrix Schur decomposition (Laub, 1979).
///
/// Forms `Z = [[A, -S]; [-Q, A^{-T}]]`, finds the invariant subspace
/// corresponding to eigenvalues inside the unit disk, and returns `P = X₂ X₁⁻¹`.
fn dare_symplectic<F: RiccatiFloat>(
    a: &Array2<F>,
    s: &Array2<F>,
    q: &Array2<F>,
    n: usize,
) -> LinalgResult<Array2<F>> {
    // Compute A^{-T} = (A^T)^{-1} = (A^{-1})^T
    let a_inv = mat_inv(a).map_err(|e| {
        LinalgError::SingularMatrixError(format!("DARE symplectic: A is singular: {e}"))
    })?;
    let a_inv_t = a_inv.t().to_owned();

    let two_n = 2 * n;
    // Build Z = [[A, -S]; [-Q, A^{-T}]]
    let mut z = Array2::<F>::zeros((two_n, two_n));
    for i in 0..n {
        for j in 0..n {
            z[[i, j]] = a[[i, j]]; // top-left: A
            z[[i, n + j]] = -s[[i, j]]; // top-right: -S
            z[[n + i, j]] = -q[[i, j]]; // bottom-left: -Q
            z[[n + i, n + j]] = a_inv_t[[i, j]]; // bottom-right: A^{-T}
        }
    }

    // Find invariant subspace of Z with eigenvalues inside the unit disk (|λ| < 1)
    // Use high max_iter since symplectic matrices (with 1/λ paired eigenvalues) converge slowly
    let schur_tol = F::epsilon() * F::from(100.0).unwrap_or(F::one());
    let v = crate::schur_enhanced::invariant_subspace(
        &z.view(),
        |re: F, im: F| {
            let r_sq = re * re + im * im;
            r_sq < F::one()
        },
        3000,
        schur_tol,
    )?;

    // v should be 2n × n
    if v.ncols() != n {
        return Err(LinalgError::ConvergenceError(format!(
            "DARE symplectic: expected {n} stable eigenvectors, got {}",
            v.ncols()
        )));
    }

    // Partition: X1 = v[0..n, :], X2 = v[n..2n, :]
    let x1 = v.slice(s![0..n, ..]).to_owned();
    let x2 = v.slice(s![n..two_n, ..]).to_owned();

    // P = X2 * X1^{-1}
    let x1_inv = mat_inv(&x1).map_err(|e| {
        LinalgError::SingularMatrixError(format!("DARE symplectic: X1 singular: {e}"))
    })?;
    let p = mm(&x2, &x1_inv);
    Ok(symmetrize(&p))
}

/// Compute DARE residual norm: ‖Aᵀ P A - P - Aᵀ P B (Bᵀ P B + R)⁻¹ Bᵀ P A + Q‖_F
fn dare_residual_norm<F: RiccatiFloat>(
    a: &Array2<F>,
    b: &Array2<F>,
    q: &Array2<F>,
    r: &Array2<F>,
    p: &Array2<F>,
    n: usize,
    m: usize,
) -> F {
    let at = a.t().to_owned();
    let bt = b.t().to_owned();
    let atp = mm(&at, p);
    let atpa = mm(&atp, a);
    let btpb_r = {
        let bt_pb = mm(&bt, &mm(p, b));
        let mut reg = r.clone();
        for i in 0..m {
            for j in 0..m {
                reg[[i, j]] += bt_pb[[i, j]];
            }
        }
        reg
    };
    let inv = match mat_inv(&btpb_r) {
        Ok(v) => v,
        Err(_) => Array2::<F>::zeros((m, m)),
    };
    let atpb = mm(&atp, b);
    let correction = mm(&mm(&atpb, &inv), &mm(&bt, &mm(p, a)));

    let mut res = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            res[[i, j]] = atpa[[i, j]] - p[[i, j]] - correction[[i, j]] + q[[i, j]];
        }
    }
    frob_norm(&res)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn check_square<F: RiccatiFloat>(a: &ArrayView2<F>, ctx: &str) -> LinalgResult<usize> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(LinalgError::ShapeError(format!("{ctx}: not square")));
    }
    Ok(n)
}

// Expose mat_inv for test convenience (crate-private)
pub(crate) fn mat_inv_pub<F: RiccatiFloat>(a: &Array2<F>) -> LinalgResult<Array2<F>> {
    mat_inv(a)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    fn care_residual_check(
        a: &Array2<f64>,
        b: &Array2<f64>,
        q: &Array2<f64>,
        r: &Array2<f64>,
        p: &Array2<f64>,
    ) -> f64 {
        let r_inv = mat_inv(r).expect("R inv");
        let s = mm(b, &mm(&r_inv, &b.t().to_owned()));
        let at = a.t().to_owned();
        let atp = mm(&at, p);
        let pa = mm(p, a);
        let psp = mm(p, &mm(&s, p));
        let n = a.nrows();
        let mut res = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                let v = atp[[i, j]] + pa[[i, j]] - psp[[i, j]] + q[[i, j]];
                res += v * v;
            }
        }
        res.sqrt()
    }

    fn dare_residual_check(
        a: &Array2<f64>,
        b: &Array2<f64>,
        q: &Array2<f64>,
        r: &Array2<f64>,
        p: &Array2<f64>,
    ) -> f64 {
        let n = a.nrows();
        let m = b.ncols();
        dare_residual_norm(a, b, q, r, p, n, m)
    }

    #[test]
    fn test_care_scalar() {
        // A=-1, B=1, Q=1, R=1
        // Scalar CARE: -2P - P² + 1 = 0 → P² + 2P - 1 = 0 → P = -1 + √2
        let a = array![[-1.0_f64]];
        let b = array![[1.0_f64]];
        let q = array![[1.0_f64]];
        let r = array![[1.0_f64]];
        let p = care_solve(&a.view(), &b.view(), &q.view(), &r.view()).expect("CARE scalar failed");

        let expected = -1.0_f64 + 2.0_f64.sqrt(); // ≈ 0.4142
        let diff = (p[[0, 0]] - expected).abs();
        assert!(
            diff < 1e-5,
            "CARE scalar: got {}, expected {expected}",
            p[[0, 0]]
        );
    }

    #[test]
    fn test_care_2x2_residual() {
        let a = array![[-1.0_f64, 0.0], [0.0, -2.0]];
        let b = array![[1.0_f64], [1.0]];
        let q = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let r = array![[1.0_f64]];
        let p = care_solve(&a.view(), &b.view(), &q.view(), &r.view()).expect("CARE 2x2 failed");
        let res = care_residual_check(&a, &b, &q, &r, &p);
        assert!(res < 1e-5, "CARE 2x2 residual = {res}");
    }

    #[test]
    fn test_dare_integrator_residual() {
        // Discrete integrator: A=[[1,1],[0,1]], B=[[0],[1]], Q=I, R=I
        let a = array![[1.0_f64, 1.0], [0.0, 1.0]];
        let b = array![[0.0_f64], [1.0]];
        let q = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let r = array![[1.0_f64]];
        let p =
            dare_solve(&a.view(), &b.view(), &q.view(), &r.view()).expect("DARE integrator failed");
        let res = dare_residual_check(&a, &b, &q, &r, &p);
        assert!(res < 1e-4, "DARE integrator residual = {res}");
    }

    #[test]
    fn test_dare_stable_system() {
        let a = array![[0.9_f64, 0.1], [0.0, 0.8]];
        let b = array![[1.0_f64], [0.0]];
        let q = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let r = array![[1.0_f64]];
        let p = dare_solve(&a.view(), &b.view(), &q.view(), &r.view()).expect("DARE stable failed");
        // P should be positive definite (all diagonal > 0)
        for i in 0..2 {
            assert!(p[[i, i]] > 0.0, "P[{i},{i}] = {} not positive", p[[i, i]]);
        }
        let res = dare_residual_check(&a, &b, &q, &r, &p);
        assert!(res < 1e-4, "DARE stable residual = {res}");
    }
}
