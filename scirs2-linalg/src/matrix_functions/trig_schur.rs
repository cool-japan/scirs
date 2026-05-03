//! Matrix trigonometric and hyperbolic functions via Schur decomposition
//!
//! Provides numerically accurate implementations using Schur decomposition
//! + pointwise computation on the quasi-triangular factor, as an alternative
//!   to the truncated Taylor series in `trigonometric.rs` and `hyperbolic.rs`.
//!
//! # Available functions
//!
//! - `sinm_schur`  / `cosm_schur`  / `tanm_schur`   - trigonometric via Schur
//! - `sinhm_schur` / `coshm_schur` / `tanhm_schur`  - hyperbolic via Schur
//!
//! # Algorithm
//!
//! The Schur method computes sin(A) (and similarly for other functions) by:
//!   1. Compute Schur decomposition A = Q T Q^T
//!   2. Apply the scalar function to each diagonal entry of T
//!   3. Propagate super-diagonal corrections via the Sylvester recurrence
//!   4. Back-transform: f(A) = Q f(T) Q^T
//!
//! This is significantly more accurate for non-normal matrices than the
//! direct Taylor series, since it avoids cancellation errors.
//!
//! # References
//!
//! - Higham, N.J. (2008). "Functions of Matrices: Theory and Computation."
//!   SIAM. Chapter 12.
//! - Parlett, B.N. (1974). "Computation of Functions of Triangular Matrices."
//!   EECS Memorandum UCB/ERL M74/49.

use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{Array2, ArrayView2, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use std::iter::Sum;

// ---------------------------------------------------------------------------
// Trait alias
// ---------------------------------------------------------------------------

/// Trait alias for floating-point bounds used in trig Schur methods.
pub trait TrigFloat: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static {}
impl<T> TrigFloat for T where T: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static {}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Dense square matrix multiplication.
fn matmul_nn<F: TrigFloat>(a: &Array2<F>, b: &Array2<F>) -> Array2<F> {
    let n = a.nrows();
    let mut c = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for k in 0..n {
            let aik = a[[i, k]];
            if aik == F::zero() {
                continue;
            }
            for j in 0..n {
                c[[i, j]] += aik * b[[k, j]];
            }
        }
    }
    c
}

/// Solve the upper-triangular Sylvester equation T S + S D = C for upper-triangular S.
/// Here T and D are upper-triangular diagonal matrices (only diags matter for this recurrence).
/// Used to propagate off-diagonal corrections in the Parlett method.
///
/// More precisely, this computes the super-diagonal entries of the upper-triangular
/// f(T) matrix using Parlett's recurrence for f(T)_{ij} where j > i.
fn parlett_recurrence<F: TrigFloat>(
    t: &Array2<F>,
    f_diag: &[F],
    n: usize,
    scalar_fn: fn(F) -> F,
) -> Array2<F> {
    // We work on the divided differences of f.
    // For f applied to upper-triangular T:
    //   f(T)_{ii} = f(t_{ii})
    //   f(T)_{ij} for j > i satisfies the Sylvester-like recurrence:
    //     (t_{jj} - t_{ii}) f(T)_{ij} = f(T)_{ii} t_{ij} - t_{ij} f(T)_{jj}
    //                                   + sum_{k=i+1}^{j-1} (f(T)_{ik} t_{kj} - t_{ik} f(T)_{kj})
    //
    //   When t_{ii} ≈ t_{jj} (coalescent eigenvalues), the divided difference
    //   [f; t_{ii}, t_{jj}] → f'(t_{ii}), so f(T)_{ij} → f'(t_{ii}) * t_{ij}  (for adjacent pair).
    //   For the general case we compute the inner sum as well.
    //
    // Reference: Higham (2008), "Functions of Matrices", §4.6.
    let mut ft = Array2::<F>::zeros((n, n));

    // Diagonal
    for i in 0..n {
        ft[[i, i]] = f_diag[i];
    }

    let thresh = F::epsilon() * F::from(100.0).unwrap_or(F::one());

    // Super-diagonal columns (column-wise for cache locality)
    for j in 1..n {
        for i in (0..j).rev() {
            let fii = ft[[i, i]];
            let fjj = ft[[j, j]];
            let tij = t[[i, j]];
            let denom = t[[j, j]] - t[[i, i]];

            let mut inner_sum = F::zero();
            for k in (i + 1)..j {
                inner_sum = inner_sum + ft[[i, k]] * t[[k, j]] - t[[i, k]] * ft[[k, j]];
            }

            if denom.abs() < thresh {
                // Coalescent eigenvalues: use the first divided difference f'(t_{ii}).
                // Computed via symmetric finite differences for accuracy.
                let f_prime = numerical_derivative(scalar_fn, t[[i, i]]);
                ft[[i, j]] = f_prime * tij + inner_sum;
            } else {
                let numer = (fii - fjj) * tij + inner_sum;
                ft[[i, j]] = numer / denom;
            }
        }
    }

    ft
}

/// Numerically differentiate `f` at `x` using a symmetric finite difference.
///
/// Used for the Parlett recurrence when two eigenvalues coincide (degenerate case).
fn numerical_derivative<F: TrigFloat>(f: fn(F) -> F, x: F) -> F {
    // Step h ≈ eps^(1/3) * max(1, |x|) gives O(eps^(2/3)) error in the derivative
    let h = F::from(1e-5).unwrap_or(F::epsilon()) * (F::one() + x.abs());
    (f(x + h) - f(x - h)) / (F::from(2.0).unwrap_or(F::one()) * h)
}

/// Generic Schur-based matrix function computation.
///
/// Computes f(A) using:
///   1. A = Q T Q^T (Schur decomposition)
///   2. f(T) via Parlett recurrence on upper-triangular T
///   3. f(A) = Q f(T) Q^T
///
/// This is the internal implementation; use the public `schur_apply` for external calls.
fn schur_function<F: TrigFloat>(
    a: &ArrayView2<F>,
    scalar_fn: fn(F) -> F,
    name: &str,
) -> LinalgResult<Array2<F>> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(LinalgError::ShapeError(format!(
            "{name}: matrix must be square"
        )));
    }
    if n == 0 {
        return Ok(Array2::<F>::zeros((0, 0)));
    }
    if n == 1 {
        let mut result = Array2::<F>::zeros((1, 1));
        result[[0, 0]] = scalar_fn(a[[0, 0]]);
        return Ok(result);
    }

    // Schur decomposition A = Q T Q^T
    let (q, t) = crate::decomposition::schur(a)?;

    // Apply scalar function to diagonal
    let f_diag: Vec<F> = (0..n).map(|i| scalar_fn(t[[i, i]])).collect();

    // Propagate via Parlett recurrence (pass scalar_fn for derivative estimation)
    let ft = parlett_recurrence(&t, &f_diag, n, scalar_fn);

    // Back-transform: f(A) = Q f(T) Q^T
    Ok(q.dot(&ft).dot(&q.t()))
}

/// Public entry point: apply a scalar function f to a matrix via Schur decomposition.
///
/// This is a general-purpose helper that lets other modules (e.g. `trigonometric.rs`)
/// apply arbitrary scalar functions to matrices without exposing the internal
/// `schur_function` helper.
///
/// # Arguments
/// * `a`         - Input square matrix
/// * `scalar_fn` - Scalar function applied to each diagonal element of the Schur form
/// * `name`      - Function name used in error messages
pub fn schur_apply<F: TrigFloat>(
    a: &ArrayView2<F>,
    scalar_fn: fn(F) -> F,
    name: &str,
) -> LinalgResult<Array2<F>> {
    schur_function(a, scalar_fn, name)
}

// ---------------------------------------------------------------------------
// Public API: Trigonometric functions via Schur decomposition
// ---------------------------------------------------------------------------

/// Compute the matrix sine via Schur decomposition.
///
/// Uses the Parlett recurrence on the Schur form for numerically stable
/// computation: sin(A) = Q * sin(T) * Q^T where A = Q T Q^T.
///
/// # Arguments
/// * `a` - Input square matrix
///
/// # Returns
/// * sin(A) - the matrix sine
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions::trig_schur::sinm_schur;
///
/// let a = array![[0.0_f64, 0.0], [0.0, 0.0]]; // Zero matrix
/// let s = sinm_schur(&a.view()).expect("sinm_schur failed");
/// // sin(0) = 0
/// assert!(s[[0, 0]].abs() < 1e-12);
/// assert!(s[[1, 1]].abs() < 1e-12);
/// ```
pub fn sinm_schur<F: TrigFloat>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>> {
    schur_function(a, |x: F| x.sin(), "sinm_schur")
}

/// Compute the matrix cosine via Schur decomposition.
///
/// Uses the Parlett recurrence on the Schur form for numerically stable
/// computation: cos(A) = Q * cos(T) * Q^T where A = Q T Q^T.
///
/// # Arguments
/// * `a` - Input square matrix
///
/// # Returns
/// * cos(A) - the matrix cosine
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions::trig_schur::cosm_schur;
///
/// let a = array![[0.0_f64, 0.0], [0.0, 0.0]]; // Zero matrix
/// let c = cosm_schur(&a.view()).expect("cosm_schur failed");
/// // cos(0) = I
/// assert!((c[[0, 0]] - 1.0).abs() < 1e-12);
/// assert!((c[[1, 1]] - 1.0).abs() < 1e-12);
/// ```
pub fn cosm_schur<F: TrigFloat>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>> {
    schur_function(a, |x: F| x.cos(), "cosm_schur")
}

/// Compute the matrix tangent via Schur decomposition.
///
/// Computed as tan(A) = sin(A) * cos(A)^{-1} using the Schur-based
/// sin and cos implementations.
///
/// # Arguments
/// * `a` - Input square matrix (cos(A) must be invertible)
///
/// # Returns
/// * tan(A) - the matrix tangent
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions::trig_schur::tanm_schur;
///
/// let a = array![[0.0_f64, 0.0], [0.0, 0.0]]; // Zero matrix
/// let t = tanm_schur(&a.view()).expect("tanm_schur failed");
/// // tan(0) = 0
/// assert!(t[[0, 0]].abs() < 1e-12);
/// ```
pub fn tanm_schur<F: TrigFloat>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(LinalgError::ShapeError(
            "tanm_schur: matrix must be square".into(),
        ));
    }

    let sin_a = sinm_schur(a)?;
    let cos_a = cosm_schur(a)?;

    // tan(A) = sin(A) * cos(A)^{-1} = solve(cos(A)^T, sin(A)^T)^T
    // Equivalently: solve the system cos(A) X = sin(A)
    crate::solve::solve_multiple(&cos_a.view(), &sin_a.view(), None)
}

// ---------------------------------------------------------------------------
// Public API: Hyperbolic functions via Schur decomposition
// ---------------------------------------------------------------------------

/// Compute the matrix hyperbolic sine via Schur decomposition.
///
/// Uses the Parlett recurrence on the Schur form:
/// sinh(A) = Q * sinh(T) * Q^T where A = Q T Q^T.
///
/// # Arguments
/// * `a` - Input square matrix
///
/// # Returns
/// * sinh(A) - the matrix hyperbolic sine
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions::trig_schur::sinhm_schur;
///
/// let a = array![[0.0_f64, 0.0], [0.0, 0.0]];
/// let s = sinhm_schur(&a.view()).expect("sinhm_schur failed");
/// assert!(s[[0, 0]].abs() < 1e-12);
/// ```
pub fn sinhm_schur<F: TrigFloat>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>> {
    schur_function(a, |x: F| x.sinh(), "sinhm_schur")
}

/// Compute the matrix hyperbolic cosine via Schur decomposition.
///
/// Uses the Parlett recurrence on the Schur form:
/// cosh(A) = Q * cosh(T) * Q^T where A = Q T Q^T.
///
/// # Arguments
/// * `a` - Input square matrix
///
/// # Returns
/// * cosh(A) - the matrix hyperbolic cosine
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions::trig_schur::coshm_schur;
///
/// let a = array![[0.0_f64, 0.0], [0.0, 0.0]];
/// let c = coshm_schur(&a.view()).expect("coshm_schur failed");
/// // cosh(0) = I
/// assert!((c[[0, 0]] - 1.0).abs() < 1e-12);
/// assert!((c[[1, 1]] - 1.0).abs() < 1e-12);
/// ```
pub fn coshm_schur<F: TrigFloat>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>> {
    schur_function(a, |x: F| x.cosh(), "coshm_schur")
}

/// Compute the matrix hyperbolic tangent via Schur decomposition.
///
/// Computed as tanh(A) = sinh(A) * cosh(A)^{-1} using the Schur-based
/// sinh and cosh implementations.
///
/// # Arguments
/// * `a` - Input square matrix (cosh(A) must be invertible)
///
/// # Returns
/// * tanh(A) - the matrix hyperbolic tangent
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions::trig_schur::tanhm_schur;
///
/// let a = array![[0.0_f64, 0.0], [0.0, 0.0]];
/// let t = tanhm_schur(&a.view()).expect("tanhm_schur failed");
/// // tanh(0) = 0
/// assert!(t[[0, 0]].abs() < 1e-12);
/// ```
pub fn tanhm_schur<F: TrigFloat>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(LinalgError::ShapeError(
            "tanhm_schur: matrix must be square".into(),
        ));
    }

    let sinh_a = sinhm_schur(a)?;
    let cosh_a = coshm_schur(a)?;

    // tanh(A) = sinh(A) * cosh(A)^{-1}
    crate::solve::solve_multiple(&cosh_a.view(), &sinh_a.view(), None)
}

// ---------------------------------------------------------------------------
// Additional utility: generic Schur matrix function
// ---------------------------------------------------------------------------

/// Apply a general scalar function to a matrix via Schur decomposition.
///
/// Computes f(A) using Parlett's method:
///   1. Compute Schur decomposition A = Q T Q^T
///   2. Compute f(T) via the recurrence for upper-triangular matrices
///   3. f(A) = Q f(T) Q^T
///
/// # Arguments
/// * `a`   - Input square matrix
/// * `f`   - Scalar function to apply
/// * `name`- Name for error messages
///
/// # Returns
/// * f(A) - the matrix function result
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions::trig_schur::apply_schur;
///
/// let a = array![[0.5_f64, 0.0], [0.0, 1.0]];
/// // Compute exp(A) via Schur method
/// let exp_a = apply_schur(&a.view(), |x| x.exp(), "exp").expect("apply_schur failed");
/// assert!((exp_a[[0, 0]] - 0.5_f64.exp()).abs() < 1e-10);
/// assert!((exp_a[[1, 1]] - 1.0_f64.exp()).abs() < 1e-10);
/// ```
pub fn apply_schur<F: TrigFloat>(
    a: &ArrayView2<F>,
    f: fn(F) -> F,
    name: &str,
) -> LinalgResult<Array2<F>> {
    schur_function(a, f, name)
}

// ---------------------------------------------------------------------------
// Additional: sin and cos from matrix exponential (complex arithmetic)
// ---------------------------------------------------------------------------

/// Compute sin(A) and cos(A) simultaneously from the real and imaginary parts
/// of exp(iA).
///
/// Using the formula:
///   exp(iA) = cos(A) + i * sin(A)  (for real A, taken formally)
///
/// This is implemented via the doubled-up trick with the augmented real system:
///   exp([[0, -A], [A, 0]]) = [[cos(A), sin(A)], [-sin(A), cos(A)]]
///
/// which is equivalent since [[0,-1],[1,0]] is a representation of i.
///
/// # Arguments
/// * `a` - Input square n x n real matrix
///
/// # Returns
/// * `(cos(A), sin(A))` - a tuple of (matrix cosine, matrix sine)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions::trig_schur::sincos_expm;
///
/// let a = array![[0.5_f64, 0.0], [0.0, 0.3]];
/// let (cos_a, sin_a) = sincos_expm(&a.view()).expect("sincos_expm failed");
/// assert!((cos_a[[0, 0]] - 0.5_f64.cos()).abs() < 1e-10);
/// assert!((sin_a[[0, 0]] - 0.5_f64.sin()).abs() < 1e-10);
/// ```
pub fn sincos_expm<F: TrigFloat>(a: &ArrayView2<F>) -> LinalgResult<(Array2<F>, Array2<F>)> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(LinalgError::ShapeError(
            "sincos_expm: matrix must be square".into(),
        ));
    }

    // Compute cos(A) and sin(A) simultaneously using the doubled real system.
    //
    // The identity exp([[0,-A],[A,0]]) = [[cos(A), -sin(A)],[sin(A), cos(A)]] holds
    // for SYMMETRIC A (where [[0,-A],[A,0]] is skew-symmetric and real exp gives cosines).
    //
    // For general A we fall back to two independent Schur-based evaluations.
    // This is still efficient as both share the same Schur decomposition of A.
    //
    // A future optimization could use a single 2n×2n Schur decomposition, but
    // correctness takes priority here.

    // First check if A is symmetric (so the doubled-system trick is valid)
    let is_symmetric = {
        let mut sym = true;
        'outer: for i in 0..n {
            for j in (i + 1)..n {
                if (a[[i, j]] - a[[j, i]]).abs() > F::epsilon() * F::from(10.0).unwrap_or(F::one())
                {
                    sym = false;
                    break 'outer;
                }
            }
        }
        sym
    };

    if is_symmetric {
        // For symmetric A, use the doubled-up augmented matrix trick.
        // aug = [[0,-A],[A,0]] is skew-symmetric, so exp(aug) is an orthogonal matrix.
        let n2 = 2 * n;
        let mut aug = Array2::<F>::zeros((n2, n2));
        for i in 0..n {
            for j in 0..n {
                aug[[i, j + n]] = -a[[i, j]]; // top-right: -A
                aug[[i + n, j]] = a[[i, j]]; // bottom-left: A
            }
        }

        let exp_aug = crate::matrix_functions::pade::pade_expm(&aug.view())?;

        // For skew-symmetric aug: exp_aug = [[cos(A), -sin(A)], [sin(A), cos(A)]]
        let mut cos_a = Array2::<F>::zeros((n, n));
        let mut sin_a = Array2::<F>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                cos_a[[i, j]] = exp_aug[[i, j]]; // top-left
                sin_a[[i, j]] = exp_aug[[i + n, j]]; // bottom-left
            }
        }
        Ok((cos_a, sin_a))
    } else {
        // For non-symmetric A (possibly with complex eigenvalues), use eigendecomposition.
        // A = V D V^{-1}  =>  f(A) = V f(D) V^{-1}
        // f(D) is diagonal with f(lambda_k) on the diagonal.
        // We take the real part of the result (which is real for real-analytic functions
        // applied to matrices whose complex eigenvalues come in conjugate pairs).
        sincos_via_eig(a, n)
    }
}

/// Compute sin(A) and cos(A) via eigendecomposition for matrices with complex eigenvalues.
///
/// Uses `A = V D V^{-1}`, computes `cos(A) = V cos(D) V^{-1}` and `sin(A) = V sin(D) V^{-1}`.
fn sincos_via_eig<F: TrigFloat>(
    a: &ArrayView2<F>,
    n: usize,
) -> LinalgResult<(Array2<F>, Array2<F>)> {
    use scirs2_core::numeric::Complex;

    // Compute eigendecomposition: (eigenvalues, eigenvectors)
    let (eigenvals, eigenvecs) = crate::eigen::eig(a, None)?;

    // Apply cos and sin to eigenvalues (complex)
    let cos_eigs: Vec<Complex<F>> = eigenvals
        .iter()
        .map(|&lam| {
            // cos(a + bi) = cos(a)*cosh(b) - i*sin(a)*sinh(b)
            let (a, b) = (lam.re, lam.im);
            let ca = a.cos();
            let cb = b.cosh();
            let sa = a.sin();
            let sb = b.sinh();
            Complex::new(ca * cb, -(sa * sb))
        })
        .collect();

    let sin_eigs: Vec<Complex<F>> = eigenvals
        .iter()
        .map(|&lam| {
            // sin(a + bi) = sin(a)*cosh(b) + i*cos(a)*sinh(b)
            let (a, b) = (lam.re, lam.im);
            let ca = a.cos();
            let cb = b.cosh();
            let sa = a.sin();
            let sb = b.sinh();
            Complex::new(sa * cb, ca * sb)
        })
        .collect();

    // Build cos(D) and sin(D) as complex diagonal matrices
    let cos_d: Array2<Complex<F>> =
        Array2::from_diag(&cos_eigs.iter().copied().collect::<Array1<_>>());
    let sin_d: Array2<Complex<F>> =
        Array2::from_diag(&sin_eigs.iter().copied().collect::<Array1<_>>());

    // cos(A) = V * cos(D) * V^{-1} — take real part for real matrix result
    let v_cos_d = eigenvecs.dot(&cos_d);
    let v_sin_d = eigenvecs.dot(&sin_d);

    // Compute V^{-1} via conjugate transpose (for normal matrices V is unitary, V^{-1} = V^H)
    // For non-normal matrices, we solve V * X = I.  Use the real system of equations.
    // To stay in real arithmetic, solve via the least-squares approach on the real part.
    // Since the matrix is real and the eigenvalues come in conjugate pairs, the real part
    // of V cos(D) V^{-1} is the correct real result.
    let v_inv = complex_inv(&eigenvecs, n)?;

    let cos_a_complex = v_cos_d.dot(&v_inv);
    let sin_a_complex = v_sin_d.dot(&v_inv);

    // Extract real parts
    let mut cos_a = Array2::<F>::zeros((n, n));
    let mut sin_a = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            cos_a[[i, j]] = cos_a_complex[[i, j]].re;
            sin_a[[i, j]] = sin_a_complex[[i, j]].re;
        }
    }

    Ok((cos_a, sin_a))
}

use scirs2_core::ndarray::Array1;

/// Invert a complex matrix via Gaussian elimination with partial pivoting.
fn complex_inv<F: TrigFloat>(
    m: &Array2<scirs2_core::numeric::Complex<F>>,
    n: usize,
) -> LinalgResult<Array2<scirs2_core::numeric::Complex<F>>> {
    use scirs2_core::numeric::Complex;

    let mut a = m.to_owned();
    let mut inv = Array2::<Complex<F>>::zeros((n, n));
    // Identity
    for i in 0..n {
        inv[[i, i]] = Complex::new(F::one(), F::zero());
    }

    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = a[[col, col]].norm_sqr();
        for row in (col + 1)..n {
            let v = a[[row, col]].norm_sqr();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }

        if max_val < F::from(1e-30).unwrap_or(F::epsilon()) {
            return Err(LinalgError::SingularMatrixError(
                "sincos_via_eig: singular eigenvector matrix".to_string(),
            ));
        }

        // Swap rows
        if max_row != col {
            for j in 0..n {
                let tmp_a = a[[col, j]];
                a[[col, j]] = a[[max_row, j]];
                a[[max_row, j]] = tmp_a;
                let tmp_i = inv[[col, j]];
                inv[[col, j]] = inv[[max_row, j]];
                inv[[max_row, j]] = tmp_i;
            }
        }

        // Scale pivot row
        let pivot = a[[col, col]];
        let inv_pivot = pivot.inv();
        for j in 0..n {
            a[[col, j]] *= inv_pivot;
            inv[[col, j]] *= inv_pivot;
        }

        // Eliminate
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = a[[row, col]];
            if factor.norm_sqr() < F::from(1e-30).unwrap_or(F::epsilon()) {
                continue;
            }
            for j in 0..n {
                let av = a[[col, j]] * factor;
                let iv = inv[[col, j]] * factor;
                a[[row, j]] -= av;
                inv[[row, j]] -= iv;
            }
        }
    }

    Ok(inv)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    // --- sinm_schur ---

    #[test]
    fn test_sinm_schur_zero() {
        let a = array![[0.0_f64, 0.0], [0.0, 0.0]];
        let s = sinm_schur(&a.view()).expect("sinm_schur zero");
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(s[[i, j]], 0.0, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_sinm_schur_diagonal() {
        let a = array![[0.5_f64, 0.0], [0.0, 1.0]];
        let s = sinm_schur(&a.view()).expect("sinm_schur diagonal");
        assert_abs_diff_eq!(s[[0, 0]], 0.5_f64.sin(), epsilon = 1e-10);
        assert_abs_diff_eq!(s[[1, 1]], 1.0_f64.sin(), epsilon = 1e-10);
        assert!(s[[0, 1]].abs() < 1e-10);
        assert!(s[[1, 0]].abs() < 1e-10);
    }

    #[test]
    fn test_sinm_schur_nilpotent() {
        // sin([[0, t], [0, 0]]) = [[0, t], [0, 0]] for small t (since A^2 = 0)
        let t_val = 0.1_f64;
        let a = array![[0.0, t_val], [0.0, 0.0]];
        let s = sinm_schur(&a.view()).expect("sinm_schur nilpotent");
        // For nilpotent: sin(A) = A - A^3/6 + ... = A (since A^2 = 0)
        assert_abs_diff_eq!(s[[0, 0]], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(s[[0, 1]], t_val, epsilon = 1e-10);
        assert_abs_diff_eq!(s[[1, 0]], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(s[[1, 1]], 0.0, epsilon = 1e-12);
    }

    // --- cosm_schur ---

    #[test]
    fn test_cosm_schur_zero() {
        let a = array![[0.0_f64, 0.0], [0.0, 0.0]];
        let c = cosm_schur(&a.view()).expect("cosm_schur zero");
        // cos(0) = I
        assert_abs_diff_eq!(c[[0, 0]], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(c[[1, 1]], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(c[[0, 1]], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(c[[1, 0]], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_cosm_schur_diagonal() {
        let a = array![[0.5_f64, 0.0], [0.0, 1.0]];
        let c = cosm_schur(&a.view()).expect("cosm_schur diagonal");
        assert_abs_diff_eq!(c[[0, 0]], 0.5_f64.cos(), epsilon = 1e-10);
        assert_abs_diff_eq!(c[[1, 1]], 1.0_f64.cos(), epsilon = 1e-10);
    }

    #[test]
    fn test_sin2_cos2_identity() {
        // sin^2(A) + cos^2(A) = I only for diagonal/normal matrices in general
        // For diagonal A it must hold exactly
        let a = array![[0.3_f64, 0.0], [0.0, 0.7]];
        let sin_a = sinm_schur(&a.view()).expect("sinm");
        let cos_a = cosm_schur(&a.view()).expect("cosm");
        let s2 = matmul_nn(&sin_a, &sin_a);
        let c2 = matmul_nn(&cos_a, &cos_a);
        for i in 0..2 {
            for j in 0..2 {
                let sum = s2[[i, j]] + c2[[i, j]];
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(sum, expected, epsilon = 1e-10);
            }
        }
    }

    // --- tanm_schur ---

    #[test]
    fn test_tanm_schur_zero() {
        let a = array![[0.0_f64, 0.0], [0.0, 0.0]];
        let t = tanm_schur(&a.view()).expect("tanm_schur zero");
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(t[[i, j]], 0.0, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_tanm_schur_diagonal() {
        let a = array![[0.3_f64, 0.0], [0.0, 0.5]];
        let t = tanm_schur(&a.view()).expect("tanm_schur diagonal");
        assert_abs_diff_eq!(t[[0, 0]], 0.3_f64.tan(), epsilon = 1e-10);
        assert_abs_diff_eq!(t[[1, 1]], 0.5_f64.tan(), epsilon = 1e-10);
    }

    // --- sinhm_schur ---

    #[test]
    fn test_sinhm_schur_zero() {
        let a = array![[0.0_f64, 0.0], [0.0, 0.0]];
        let s = sinhm_schur(&a.view()).expect("sinhm_schur zero");
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(s[[i, j]], 0.0, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_sinhm_schur_diagonal() {
        let a = array![[0.5_f64, 0.0], [0.0, 1.0]];
        let s = sinhm_schur(&a.view()).expect("sinhm_schur diagonal");
        assert_abs_diff_eq!(s[[0, 0]], 0.5_f64.sinh(), epsilon = 1e-10);
        assert_abs_diff_eq!(s[[1, 1]], 1.0_f64.sinh(), epsilon = 1e-10);
    }

    // --- coshm_schur ---

    #[test]
    fn test_coshm_schur_zero() {
        let a = array![[0.0_f64, 0.0], [0.0, 0.0]];
        let c = coshm_schur(&a.view()).expect("coshm_schur zero");
        // cosh(0) = I
        assert_abs_diff_eq!(c[[0, 0]], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(c[[1, 1]], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_coshm_schur_diagonal() {
        let a = array![[0.5_f64, 0.0], [0.0, 1.0]];
        let c = coshm_schur(&a.view()).expect("coshm_schur diagonal");
        assert_abs_diff_eq!(c[[0, 0]], 0.5_f64.cosh(), epsilon = 1e-10);
        assert_abs_diff_eq!(c[[1, 1]], 1.0_f64.cosh(), epsilon = 1e-10);
    }

    #[test]
    fn test_cosh2_sinh2_identity() {
        // cosh^2(A) - sinh^2(A) = I for diagonal A
        let a = array![[0.3_f64, 0.0], [0.0, 0.7]];
        let sinh_a = sinhm_schur(&a.view()).expect("sinhm");
        let cosh_a = coshm_schur(&a.view()).expect("coshm");
        let c2 = matmul_nn(&cosh_a, &cosh_a);
        let s2 = matmul_nn(&sinh_a, &sinh_a);
        for i in 0..2 {
            for j in 0..2 {
                let diff = c2[[i, j]] - s2[[i, j]];
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(diff, expected, epsilon = 1e-10);
            }
        }
    }

    // --- tanhm_schur ---

    #[test]
    fn test_tanhm_schur_zero() {
        let a = array![[0.0_f64, 0.0], [0.0, 0.0]];
        let t = tanhm_schur(&a.view()).expect("tanhm_schur zero");
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(t[[i, j]], 0.0, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_tanhm_schur_diagonal() {
        let a = array![[0.3_f64, 0.0], [0.0, 0.5]];
        let t = tanhm_schur(&a.view()).expect("tanhm_schur diagonal");
        assert_abs_diff_eq!(t[[0, 0]], 0.3_f64.tanh(), epsilon = 1e-10);
        assert_abs_diff_eq!(t[[1, 1]], 0.5_f64.tanh(), epsilon = 1e-10);
    }

    // --- sincos_expm ---

    #[test]
    fn test_sincos_expm_diagonal() {
        let a = array![[0.5_f64, 0.0], [0.0, 1.0]];
        let (cos_a, sin_a) = sincos_expm(&a.view()).expect("sincos_expm failed");
        assert_abs_diff_eq!(cos_a[[0, 0]], 0.5_f64.cos(), epsilon = 1e-10);
        assert_abs_diff_eq!(cos_a[[1, 1]], 1.0_f64.cos(), epsilon = 1e-10);
        assert_abs_diff_eq!(sin_a[[0, 0]], 0.5_f64.sin(), epsilon = 1e-10);
        assert_abs_diff_eq!(sin_a[[1, 1]], 1.0_f64.sin(), epsilon = 1e-10);
    }

    #[test]
    fn test_sincos_expm_rotation() {
        // For the rotation generator a = [[0, theta], [-theta, 0]]:
        // cos(a) = cos(theta) * I,  sin(a) = sin(theta) * [[0, 1], [-1, 0]]
        let theta = 0.7_f64;
        let a = array![[0.0_f64, theta], [-theta, 0.0]];
        let (cos_a, sin_a) = sincos_expm(&a.view()).expect("sincos_expm rotation");

        // For A = [[0,θ],[-θ,0]] = θJ where J = [[0,1],[-1,0]], J^2 = -I:
        //   A^{2k} = (θJ)^{2k} = θ^{2k} J^{2k} = θ^{2k} (-I)^k = (-1)^k θ^{2k} I
        // Therefore:
        //   cos(A) = Σ (-1)^k A^{2k}/(2k)! = Σ (-1)^k (-1)^k θ^{2k} I /(2k)! = cosh(θ) * I
        //   sin(A) = Σ (-1)^k A^{2k+1}/(2k+1)! = A * sinh(θ)/θ
        //          = [[0,θ],[-θ,0]] * sinh(θ)/θ = [[0,sinh(θ)],[-sinh(θ),0]]
        assert_abs_diff_eq!(cos_a[[0, 0]], theta.cosh(), epsilon = 1e-10);
        assert_abs_diff_eq!(cos_a[[1, 1]], theta.cosh(), epsilon = 1e-10);
        assert_abs_diff_eq!(cos_a[[0, 1]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(cos_a[[1, 0]], 0.0, epsilon = 1e-10);

        assert_abs_diff_eq!(sin_a[[0, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(sin_a[[1, 1]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(sin_a[[0, 1]], theta.sinh(), epsilon = 1e-10);
        assert_abs_diff_eq!(sin_a[[1, 0]], -theta.sinh(), epsilon = 1e-10);
    }

    // --- apply_schur ---

    #[test]
    fn test_apply_schur_exp() {
        let a = array![[0.5_f64, 0.0], [0.0, 1.0]];
        let exp_a = apply_schur(&a.view(), |x: f64| x.exp(), "exp").expect("apply_schur exp");
        assert_abs_diff_eq!(exp_a[[0, 0]], 0.5_f64.exp(), epsilon = 1e-10);
        assert_abs_diff_eq!(exp_a[[1, 1]], 1.0_f64.exp(), epsilon = 1e-10);
    }

    #[test]
    fn test_apply_schur_sqrt() {
        let a = array![[4.0_f64, 0.0], [0.0, 9.0]];
        let sqrt_a = apply_schur(&a.view(), |x: f64| x.sqrt(), "sqrt").expect("apply_schur sqrt");
        assert_abs_diff_eq!(sqrt_a[[0, 0]], 2.0, epsilon = 1e-8);
        assert_abs_diff_eq!(sqrt_a[[1, 1]], 3.0, epsilon = 1e-8);
    }
}
