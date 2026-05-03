//! Padé approximation for matrix functions
//!
//! Implements high-order Padé approximants for the matrix exponential,
//! including scaling-and-squaring, Fréchet derivative, and condition number.
//!
//! # References
//!
//! - Higham, N.J. (2005). "The Scaling and Squaring Method for the Matrix
//!   Exponential Revisited." SIAM Journal on Matrix Analysis and Applications.
//! - Al-Mohy, A.H. & Higham, N.J. (2009). "A New Scaling and Squaring
//!   Algorithm for the Matrix Exponential." SIAM Journal on Matrix Analysis.

use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{Array2, ArrayView2, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use std::iter::Sum;

// ---------------------------------------------------------------------------
// Trait alias
// ---------------------------------------------------------------------------

/// Trait alias for floating-point bounds used in Padé approximation.
pub trait PadeFloat: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static {}
impl<T> PadeFloat for T where T: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static {}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Dense matrix multiply: C = A * B (n x n)
fn matmul_nn<F: PadeFloat>(a: &Array2<F>, b: &Array2<F>) -> Array2<F> {
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

/// 1-norm of matrix (maximum absolute column sum).
fn one_norm<F: PadeFloat>(a: &Array2<F>) -> F {
    let n = a.ncols();
    let m = a.nrows();
    let mut max_col = F::zero();
    for j in 0..n {
        let mut col_sum = F::zero();
        for i in 0..m {
            col_sum += a[[i, j]].abs();
        }
        if col_sum > max_col {
            max_col = col_sum;
        }
    }
    max_col
}

/// Add identity * scalar to a matrix in-place.
fn add_identity_scaled<F: PadeFloat>(a: &mut Array2<F>, scale: F) {
    let n = a.nrows();
    for i in 0..n {
        a[[i, i]] += scale;
    }
}

// ---------------------------------------------------------------------------
// Padé coefficients
// ---------------------------------------------------------------------------

/// Return the Padé numerator coefficients p_{m,k} for order m.
///
/// The Padé approximant R_m(x) = p(x) / q(x) where
///   p(x) = sum_{k=0}^{m} c_k x^k
///   q(x) = sum_{k=0}^{m} (-1)^k c_k x^k
///
/// Coefficients c_k = (2m - k)! * m! / ((2m)! * k! * (m - k)!)
///
/// Orders supported: 3, 5, 7, 9, 13.
pub fn pade_coefficients(m: usize) -> Vec<f64> {
    match m {
        3 => vec![120.0, 60.0, 12.0, 1.0],
        5 => vec![30240.0, 15120.0, 3360.0, 420.0, 30.0, 1.0],
        7 => vec![
            17297280.0, 8648640.0, 1995840.0, 277200.0, 25200.0, 1512.0, 56.0, 1.0,
        ],
        9 => vec![
            17643225600.0,
            8821612800.0,
            2075673600.0,
            302702400.0,
            30270240.0,
            2162160.0,
            110880.0,
            3960.0,
            90.0,
            1.0,
        ],
        13 => vec![
            64764752532480000.0,
            32382376266240000.0,
            7771770303897600.0,
            1187353796428800.0,
            129060195264000.0,
            10559470521600.0,
            670442572800.0,
            33522128640.0,
            1323241920.0,
            40840800.0,
            960960.0,
            16380.0,
            182.0,
            1.0,
        ],
        _ => {
            // Fallback: compute factorial-based coefficients for general m
            let mut coeffs = vec![0.0f64; m + 1];
            // c_k = (2m-k)! * m! / ((2m)! * k! * (m-k)!)
            // Compute using iterative approach to avoid overflow
            let two_m_fact = factorial_f64(2 * m);
            let m_fact = factorial_f64(m);
            #[allow(clippy::needless_range_loop)]
            for k in 0..=m {
                let two_m_minus_k_fact = factorial_f64(2 * m - k);
                let k_fact = factorial_f64(k);
                let m_minus_k_fact = factorial_f64(m - k);
                coeffs[k] = two_m_minus_k_fact * m_fact / (two_m_fact * k_fact * m_minus_k_fact);
            }
            coeffs
        }
    }
}

fn factorial_f64(n: usize) -> f64 {
    if n == 0 {
        1.0
    } else {
        let mut result = 1.0f64;
        for i in 2..=n {
            result *= i as f64;
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Order-m Padé approximant for exp(A)
// ---------------------------------------------------------------------------

/// Compute the order-m Padé approximant for the matrix exponential.
///
/// Given a matrix A with ||A|| sufficiently small, computes R_m(A) ~ exp(A)
/// using the [m/m] Padé approximant:
///   R_m(A) = (D_m(A))^{-1} N_m(A)
/// where N_m(A) and D_m(A) are the numerator and denominator polynomials.
///
/// # Arguments
/// * `a` - Input square matrix (should have small 1-norm for good accuracy)
/// * `m` - Padé order (typically 3, 5, 7, 9, or 13)
///
/// # Returns
/// * The Padé approximant R_m(A) to exp(A)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions::pade::expm_pade;
///
/// let a = array![[0.1_f64, 0.0], [0.0, 0.2]];
/// let ea = expm_pade(&a.view(), 13).expect("Padé approximant failed");
/// // Should approximate exp(diag(0.1, 0.2))
/// assert!((ea[[0, 0]] - 0.1f64.exp()).abs() < 1e-12);
/// assert!((ea[[1, 1]] - 0.2f64.exp()).abs() < 1e-12);
/// ```
pub fn expm_pade<F: PadeFloat>(a: &ArrayView2<F>, m: usize) -> LinalgResult<Array2<F>> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(LinalgError::ShapeError(
            "expm_pade: matrix must be square".into(),
        ));
    }
    if n == 0 {
        return Ok(Array2::<F>::zeros((0, 0)));
    }

    let coeffs_f64 = pade_coefficients(m);
    // Convert to F
    let coeffs: Vec<F> = coeffs_f64
        .iter()
        .map(|&c| {
            F::from(c).unwrap_or_else(|| {
                // For very large coefficients, use the best approximation
                F::from(c as i64).unwrap_or(F::one())
            })
        })
        .collect();

    let a_owned = a.to_owned();

    // Build odd/even polynomial evaluations using Horner's method
    // For m=13 we use the algorithm from Higham (2005) with precomputed powers
    // For other m, use direct Horner evaluation

    let (n_pade, d_pade) = if m == 13 {
        pade_13_polynomials(&a_owned, &coeffs)?
    } else {
        pade_general_polynomials(&a_owned, &coeffs, m)?
    };

    // Solve D_m * X = N_m  for X = exp(A)
    crate::solve::solve_multiple(&d_pade.view(), &n_pade.view(), None)
}

/// Build N and D polynomials for the [13/13] Padé approximant.
///
/// Uses the Paterson-Stockmeyer algorithm to minimize matrix multiplications:
///   U = A(A^6(c13*A^6 + c11*A^4 + c9*A^2 + c7) + c5*A^4 + c3*A^2 + c1*I)
///   V = A^6(c12*A^6 + c10*A^4 + c8*A^2 + c6) + c4*A^4 + c2*A^2 + c0*I
///   N = U + V
///   D = V - U
fn pade_13_polynomials<F: PadeFloat>(
    a: &Array2<F>,
    c: &[F],
) -> LinalgResult<(Array2<F>, Array2<F>)> {
    let n = a.nrows();
    let eye = Array2::<F>::eye(n);

    // Precompute powers A^2, A^4, A^6
    let a2 = matmul_nn(a, a);
    let a4 = matmul_nn(&a2, &a2);
    let a6 = matmul_nn(&a2, &a4);

    // U_inner = c[13]*A^6 + c[11]*A^4 + c[9]*A^2 + c[7]*I
    let mut u_inner = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            u_inner[[i, j]] = c[13] * a6[[i, j]] + c[11] * a4[[i, j]] + c[9] * a2[[i, j]];
        }
    }
    add_identity_scaled(&mut u_inner, c[7]);

    // V_inner = c[12]*A^6 + c[10]*A^4 + c[8]*A^2 + c[6]*I
    let mut v_inner = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            v_inner[[i, j]] = c[12] * a6[[i, j]] + c[10] * a4[[i, j]] + c[8] * a2[[i, j]];
        }
    }
    add_identity_scaled(&mut v_inner, c[6]);

    // U_rest = A^6 * U_inner + c[5]*A^4 + c[3]*A^2 + c[1]*I
    let a6_u_inner = matmul_nn(&a6, &u_inner);
    let mut u_rest = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            u_rest[[i, j]] = a6_u_inner[[i, j]] + c[5] * a4[[i, j]] + c[3] * a2[[i, j]];
        }
    }
    add_identity_scaled(&mut u_rest, c[1]);

    // V_rest = A^6 * V_inner + c[4]*A^4 + c[2]*A^2 + c[0]*I
    let a6_v_inner = matmul_nn(&a6, &v_inner);
    let mut v_rest = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            v_rest[[i, j]] = a6_v_inner[[i, j]] + c[4] * a4[[i, j]] + c[2] * a2[[i, j]];
        }
    }
    add_identity_scaled(&mut v_rest, c[0]);

    // U = A * U_rest
    let u = matmul_nn(a, &u_rest);
    let v = v_rest;

    // N = U + V,  D = V - U
    let mut n_pade = Array2::<F>::zeros((n, n));
    let mut d_pade = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            n_pade[[i, j]] = v[[i, j]] + u[[i, j]];
            d_pade[[i, j]] = v[[i, j]] - u[[i, j]];
        }
    }

    // Suppress unused variable warning
    let _ = eye;

    Ok((n_pade, d_pade))
}

/// Build N and D polynomials for a general-order [m/m] Padé approximant.
///
/// Uses the standard decomposition:
///   p(A) = c_0 I + c_1 A + c_2 A^2 + ... + c_m A^m
///   q(A) = c_0 I - c_1 A + c_2 A^2 - ... + (-1)^m c_m A^m
///   N = p(A), D = q(A)
///
/// Evaluated with Horner's method.
fn pade_general_polynomials<F: PadeFloat>(
    a: &Array2<F>,
    c: &[F],
    m: usize,
) -> LinalgResult<(Array2<F>, Array2<F>)> {
    let n = a.nrows();

    // Build up powers of A
    let mut a_powers: Vec<Array2<F>> = Vec::with_capacity(m + 1);
    a_powers.push(Array2::<F>::eye(n)); // A^0 = I
    if m >= 1 {
        a_powers.push(a.to_owned()); // A^1
    }
    for k in 2..=m {
        let prev = a_powers[k - 1].clone();
        a_powers.push(matmul_nn(a, &prev));
    }

    let mut n_pade = Array2::<F>::zeros((n, n));
    let mut d_pade = Array2::<F>::zeros((n, n));

    for k in 0..=m {
        let sign = if k % 2 == 0 { F::one() } else { -F::one() };
        for i in 0..n {
            for j in 0..n {
                let term = c[k] * a_powers[k][[i, j]];
                n_pade[[i, j]] += term;
                d_pade[[i, j]] += sign * term;
            }
        }
    }

    Ok((n_pade, d_pade))
}

// ---------------------------------------------------------------------------
// Scaling and squaring: main entry point
// ---------------------------------------------------------------------------

/// Theta values (1-norm thresholds) for choosing Padé order in scaling-and-squaring.
///
/// These are taken from Higham (2005), Table 4.1.
const THETA: [f64; 5] = [
    1.495_585_217_958_292e-2, // m=3
    2.539_398_330_063_23e-1,  // m=5
    9.504_178_996_162_932e-1, // m=7
    2.097_847_961_257_068,    // m=9
    5.371_920_351_148_152,    // m=13
];

const PADE_ORDERS: [usize; 5] = [3, 5, 7, 9, 13];

/// Compute the matrix exponential via [13/13] Padé approximant with scaling-and-squaring.
///
/// This is the algorithm of Higham (2005) / Al-Mohy & Higham (2009):
///   1. Scale A: choose s such that ||A / 2^s||_1 <= theta_13
///   2. Compute R = R_{13}(A / 2^s) via Padé approximant
///   3. Unsquare: exp(A) = R^{2^s}
///
/// # Arguments
/// * `a` - Input square matrix
///
/// # Returns
/// * exp(A) to full double precision
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions::pade::pade_expm;
///
/// let a = array![[0.0_f64, 1.0], [-1.0, 0.0]]; // Rotation generator
/// let ea = pade_expm(&a.view()).expect("expm failed");
/// // exp([[0,1],[-1,0]]) = [[cos(1), sin(1)], [-sin(1), cos(1)]]
/// let cos1 = 1.0_f64.cos();
/// let sin1 = 1.0_f64.sin();
/// assert!((ea[[0, 0]] - cos1).abs() < 1e-12);
/// assert!((ea[[0, 1]] - sin1).abs() < 1e-12);
/// assert!((ea[[1, 0]] + sin1).abs() < 1e-12);
/// assert!((ea[[1, 1]] - cos1).abs() < 1e-12);
/// ```
pub fn pade_expm<F: PadeFloat>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(LinalgError::ShapeError(
            "pade_expm: matrix must be square".into(),
        ));
    }
    if n == 0 {
        return Ok(Array2::<F>::zeros((0, 0)));
    }

    // Special case: 1x1
    if n == 1 {
        let mut result = Array2::<F>::zeros((1, 1));
        result[[0, 0]] = a[[0, 0]].exp();
        return Ok(result);
    }

    let a_owned = a.to_owned();
    let norm = one_norm(&a_owned);
    let norm_f64 = norm
        .to_f64()
        .ok_or_else(|| LinalgError::ComputationError("Cannot convert norm to f64".into()))?;

    // Determine Padé order without scaling
    for (idx, &theta) in THETA.iter().enumerate() {
        if norm_f64 <= theta {
            let m = PADE_ORDERS[idx];
            return expm_pade(a, m);
        }
    }

    // Need scaling
    let theta_13 = THETA[4];
    // Choose s such that ||A / 2^s|| <= theta_13
    let s_f64 = (norm_f64 / theta_13).log2().ceil().max(0.0);
    let s = s_f64 as u32;
    let two_s = F::from(2.0_f64.powi(s as i32))
        .ok_or_else(|| LinalgError::ComputationError("Cannot convert 2^s".into()))?;

    // Scale A
    let a_scaled = a_owned.map(|&x| x / two_s);

    // Padé [13/13] approximant for the scaled matrix
    let mut result = expm_pade(&a_scaled.view(), 13)?;

    // Squaring phase: result = result^{2^s}
    for _ in 0..s {
        result = matmul_nn(&result.clone(), &result.clone());
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Fréchet derivative of the matrix exponential
// ---------------------------------------------------------------------------

/// Compute the matrix exponential and its Fréchet derivative.
///
/// The Fréchet derivative L(A, E) is the unique linear map such that:
///   exp(A + tE) = exp(A) + t * L(A, E) + O(t^2)
///
/// Computed via the augmented matrix method (Kenney & Laub, 1989):
///
///   exp([[A, E], [0, A]]) = [[exp(A), L(A,E)], [0, exp(A)]]
///
/// where the off-diagonal block of the 2n x 2n matrix exponential gives L(A, E).
///
/// # Arguments
/// * `a` - n x n input matrix
/// * `e` - n x n direction matrix (Fréchet direction)
///
/// # Returns
/// * `(exp(A), L(A, E))` - a tuple of (matrix exponential, Fréchet derivative)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions::pade::expm_frechet;
///
/// let a = array![[1.0_f64, 0.0], [0.0, 2.0]];
/// let e = array![[1.0_f64, 0.0], [0.0, 1.0]]; // Identity direction
/// let (ea, la) = expm_frechet(&a.view(), &e.view()).expect("frechet failed");
/// // For diagonal A: L(diag(a1,a2), diag(e1,e2)) = diag(e1*exp(a1), e2*exp(a2))
/// assert!((ea[[0, 0]] - 1.0_f64.exp()).abs() < 1e-12);
/// assert!((ea[[1, 1]] - 2.0_f64.exp()).abs() < 1e-12);
/// assert!((la[[0, 0]] - 1.0_f64.exp()).abs() < 1e-10);
/// assert!((la[[1, 1]] - 2.0_f64.exp()).abs() < 1e-10);
/// ```
pub fn expm_frechet<F: PadeFloat>(
    a: &ArrayView2<F>,
    e: &ArrayView2<F>,
) -> LinalgResult<(Array2<F>, Array2<F>)> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(LinalgError::ShapeError(
            "expm_frechet: A must be square".into(),
        ));
    }
    if e.nrows() != n || e.ncols() != n {
        return Err(LinalgError::ShapeError(
            "expm_frechet: E must have same shape as A".into(),
        ));
    }

    // Build 2n x 2n augmented matrix [[A, E], [0, A]]
    let n2 = 2 * n;
    let mut aug = Array2::<F>::zeros((n2, n2));

    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]]; // top-left block A
            aug[[i, j + n]] = e[[i, j]]; // top-right block E
            aug[[i + n, j + n]] = a[[i, j]]; // bottom-right block A
                                             // bottom-left block remains zero
        }
    }

    // Compute exp(aug) via pade_expm
    let exp_aug = pade_expm(&aug.view())?;

    // Extract exp(A) from top-left block and L(A,E) from top-right block
    let mut exp_a = Array2::<F>::zeros((n, n));
    let mut frechet = Array2::<F>::zeros((n, n));

    for i in 0..n {
        for j in 0..n {
            exp_a[[i, j]] = exp_aug[[i, j]];
            frechet[[i, j]] = exp_aug[[i, j + n]];
        }
    }

    Ok((exp_a, frechet))
}

// ---------------------------------------------------------------------------
// Condition number of the matrix exponential
// ---------------------------------------------------------------------------

/// Estimate the 1-norm condition number of the matrix exponential.
///
/// The condition number kappa(exp, A) is defined as:
///   kappa = ||L||_F / (||E||_F / ||exp(A)||_F)
///
/// where L is the Fréchet derivative and the maximum is over all unit E.
///
/// Estimated using a sequence of random (but deterministic) perturbation
/// directions E_k, computing the Fréchet derivative via the augmented matrix.
///
/// # Arguments
/// * `a` - Input square matrix
///
/// # Returns
/// * Estimated condition number (>= 1.0)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions::pade::expm_cond;
///
/// let a = array![[0.5_f64, 0.0], [0.0, 0.5]];
/// let kappa = expm_cond(&a.view()).expect("condition number failed");
/// assert!(kappa >= 1.0);
/// ```
pub fn expm_cond(a: &ArrayView2<f64>) -> LinalgResult<f64> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(LinalgError::ShapeError(
            "expm_cond: matrix must be square".into(),
        ));
    }

    let exp_a = pade_expm(a)?;
    let norm_exp_a = one_norm(&exp_a);
    if norm_exp_a < f64::EPSILON {
        return Ok(f64::INFINITY);
    }

    // Generate n^2 deterministic unit perturbation directions (standard basis)
    // and find the one that maximizes ||L(A, E)||_1 / ||E||_1
    let mut max_ratio: f64 = 0.0;

    // Use n unit vectors (columns of identity matrix) as E = e_i e_j^T
    // for a coverage of directions in O(n^2) Fréchet computations
    // We do a maximum of min(n^2, 25) directions for efficiency
    let n_dirs = (n * n).min(25);

    for dir in 0..n_dirs {
        let row = dir / n;
        let col = dir % n;
        let mut e_mat = Array2::<f64>::zeros((n, n));
        e_mat[[row.min(n - 1), col.min(n - 1)]] = 1.0;

        let (_, l_ae) = expm_frechet(a, &e_mat.view())?;
        let norm_l = one_norm(&l_ae);
        // e_mat has 1-norm = 1 already

        if norm_l > max_ratio {
            max_ratio = norm_l;
        }
    }

    // Condition number: max_ratio * ||A||_1 / ||exp(A)||_1
    let norm_a = one_norm(&a.to_owned());
    let cond = max_ratio * norm_a / norm_exp_a;

    Ok(cond.max(1.0))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_pade_coefficients_m3() {
        let c = pade_coefficients(3);
        assert_eq!(c.len(), 4);
        // Check leading coefficient is 120
        assert!((c[0] - 120.0).abs() < 1e-6);
        assert!((c[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_pade_coefficients_m13() {
        let c = pade_coefficients(13);
        assert_eq!(c.len(), 14);
        // Last coefficient should be 1
        assert!((c[13] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_expm_pade_identity() {
        let eye = array![[1.0, 0.0], [0.0, 1.0]];
        // exp(I) = diag(e, e)
        let result = expm_pade(&eye.view(), 13).expect("pade failed");
        let e_val = std::f64::consts::E;
        assert_abs_diff_eq!(result[[0, 0]], e_val, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[1, 1]], e_val, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[0, 1]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[1, 0]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_expm_pade_zero() {
        let z = array![[0.0, 0.0], [0.0, 0.0]];
        let result = expm_pade(&z.view(), 13).expect("pade failed");
        assert_abs_diff_eq!(result[[0, 0]], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(result[[1, 1]], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(result[[0, 1]], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(result[[1, 0]], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_pade_expm_rotation() {
        // exp([[0, 1], [-1, 0]]) = [[cos(1), sin(1)], [-sin(1), cos(1)]]
        let a = array![[0.0_f64, 1.0], [-1.0, 0.0]];
        let result = pade_expm(&a.view()).expect("pade_expm failed");
        let cos1 = 1.0_f64.cos();
        let sin1 = 1.0_f64.sin();
        assert_abs_diff_eq!(result[[0, 0]], cos1, epsilon = 1e-12);
        assert_abs_diff_eq!(result[[0, 1]], sin1, epsilon = 1e-12);
        assert_abs_diff_eq!(result[[1, 0]], -sin1, epsilon = 1e-12);
        assert_abs_diff_eq!(result[[1, 1]], cos1, epsilon = 1e-12);
    }

    #[test]
    fn test_pade_expm_large_norm() {
        // Test scaling-and-squaring with large norm matrix
        let a = array![[10.0_f64, 0.0], [0.0, -5.0]];
        let result = pade_expm(&a.view()).expect("pade_expm failed");
        assert_abs_diff_eq!(result[[0, 0]], 10.0_f64.exp(), epsilon = 1e-4);
        assert_abs_diff_eq!(result[[1, 1]], (-5.0_f64).exp(), epsilon = 1e-10);
        assert_abs_diff_eq!(result[[0, 1]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[1, 0]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pade_expm_nilpotent() {
        // exp([[0, 1], [0, 0]]) = [[1, 1], [0, 1]]
        let a = array![[0.0_f64, 1.0], [0.0, 0.0]];
        let result = pade_expm(&a.view()).expect("pade_expm failed");
        assert_abs_diff_eq!(result[[0, 0]], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(result[[0, 1]], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(result[[1, 0]], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(result[[1, 1]], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_expm_frechet_diagonal() {
        // For diagonal A: L(diag(a1,a2), diag(e1,e2)) = diag(e1*exp(a1), e2*exp(a2))
        let a = array![[1.0_f64, 0.0], [0.0, 2.0]];
        let e = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let (exp_a, l_ae) = expm_frechet(&a.view(), &e.view()).expect("frechet failed");

        assert_abs_diff_eq!(exp_a[[0, 0]], 1.0_f64.exp(), epsilon = 1e-10);
        assert_abs_diff_eq!(exp_a[[1, 1]], 2.0_f64.exp(), epsilon = 1e-10);
        assert_abs_diff_eq!(l_ae[[0, 0]], 1.0_f64.exp(), epsilon = 1e-8);
        assert_abs_diff_eq!(l_ae[[1, 1]], 2.0_f64.exp(), epsilon = 1e-8);
    }

    #[test]
    fn test_expm_frechet_linearity() {
        // L(A, alpha*E) = alpha * L(A, E)
        let a = array![[0.5_f64, 0.2], [0.1, 0.3]];
        let e = array![[0.1_f64, 0.0], [0.0, 0.2]];
        let alpha = 3.0_f64;

        let mut e_scaled = Array2::<f64>::zeros((2, 2));
        for i in 0..2 {
            for j in 0..2 {
                e_scaled[[i, j]] = alpha * e[[i, j]];
            }
        }

        let (_, l1) = expm_frechet(&a.view(), &e.view()).expect("frechet failed");
        let (_, l2) = expm_frechet(&a.view(), &e_scaled.view()).expect("frechet failed");

        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(l2[[i, j]], alpha * l1[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_expm_cond_positive() {
        let a = array![[0.5_f64, 0.0], [0.0, 0.5]];
        let kappa = expm_cond(&a.view()).expect("cond failed");
        assert!(kappa >= 1.0);
        assert!(kappa.is_finite());
    }

    #[test]
    fn test_expm_cond_identity() {
        let eye = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let kappa = expm_cond(&eye.view()).expect("cond failed");
        assert!(kappa >= 1.0);
        // Identity matrix exp is well-conditioned
        assert!(kappa < 1000.0);
    }

    #[test]
    fn test_expm_pade_all_orders() {
        // For small matrices all orders should give the same answer
        let a = array![[0.01_f64, 0.005], [-0.005, 0.02]];
        let expected = pade_expm(&a.view()).expect("expected failed");

        for &m in &[3usize, 5, 7, 9, 13] {
            let result = expm_pade(&a.view(), m).expect("pade order failed");
            for i in 0..2 {
                for j in 0..2 {
                    assert_abs_diff_eq!(result[[i, j]], expected[[i, j]], epsilon = 1e-8);
                }
            }
        }
    }
}
