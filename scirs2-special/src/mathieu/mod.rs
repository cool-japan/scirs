//! Mathieu Functions
//!
//! This module provides implementations of Mathieu functions, which are
//! solutions to the Mathieu differential equation:
//!
//! d²y/dz² + [a - 2q cos(2z)]y = 0
//!
//! These functions are important in the analysis of wave equations in elliptical
//! coordinates, vibration problems, and other physical systems with elliptical
//! symmetry.
//!
//! Characteristic values are computed via the Hill matrix eigenvalue method
//! (Sturm sequence bisection + inverse iteration), matching the approach used
//! by SciPy/DLMF §28.

pub mod advanced;

use crate::error::SpecialResult;
use scirs2_core::numeric::{Float, FromPrimitive};
use std::f64::consts::PI;
use std::fmt::Debug;

/// Helper to convert f64 constants to generic Float type with better error messages
#[inline(always)]
fn const_f64<F: Float + FromPrimitive>(value: f64) -> F {
    F::from(value).expect("Failed to convert constant to target float type - this indicates an incompatible numeric type")
}

// ────────────────────────────────────────────────────────────────────────────
// Hill matrix builder helpers
// ────────────────────────────────────────────────────────────────────────────

/// Number of Fourier terms used in the Hill matrix (must be > max order you want + margin).
const HILL_SIZE: usize = 60;

/// Build the Hill matrix diagonal and off-diagonal entries for a_{2m}(q)
/// (even-order, period-π, even Mathieu functions).
///
/// Basis: cos(2k·x), k = 0, 1, …
/// The matrix is symmetric tridiagonal with:
///   diag[k]     = (2k)²
///   off_diag[0] = √2·q   (k=0↔1 coupling, from normalizing the k=0 term)
///   off_diag[k] = q       for k ≥ 1
///
/// Reference: DLMF 28.4.5–28.4.6.
fn hill_even_a_matrix(q: f64, size: usize) -> (Vec<f64>, Vec<f64>) {
    let mut diag = Vec::with_capacity(size);
    let mut off = Vec::with_capacity(size - 1);
    for k in 0..size {
        diag.push((2 * k) as f64 * (2 * k) as f64);
    }
    if size > 1 {
        off.push(q * 2.0_f64.sqrt());
        for _ in 1..size - 1 {
            off.push(q);
        }
    }
    (diag, off)
}

/// Build the Hill matrix for a_{2m+1}(q)
/// (odd-order, period-2π, even Mathieu functions).
///
/// Basis: cos((2k+1)·x), k = 0, 1, …
/// Diagonal: (2k+1)²  with a correction of +q on diag[0]
/// Off-diagonal: all q.
///
/// Reference: DLMF 28.4.7–28.4.8.
fn hill_odd_a_matrix(q: f64, size: usize) -> (Vec<f64>, Vec<f64>) {
    let mut diag = Vec::with_capacity(size);
    let mut off = Vec::with_capacity(size - 1);
    for k in 0..size {
        let d = (2 * k + 1) as f64 * (2 * k + 1) as f64;
        diag.push(d);
    }
    // The period-2π boundary condition shifts diagonal[0] by +q
    if !diag.is_empty() {
        diag[0] += q;
    }
    for _ in 0..size.saturating_sub(1) {
        off.push(q);
    }
    (diag, off)
}

/// Build the Hill matrix for b_{2m+2}(q)
/// (even-order ≥ 2, odd Mathieu functions).
///
/// Basis: sin(2k·x), k = 1, 2, …
/// Diagonal: (2k)² for k = 1, 2, …
/// Off-diagonal: all q.
///
/// Reference: DLMF 28.4.11–28.4.12.
fn hill_even_b_matrix(q: f64, size: usize) -> (Vec<f64>, Vec<f64>) {
    let mut diag = Vec::with_capacity(size);
    let mut off = Vec::with_capacity(size - 1);
    for k in 1..=size {
        diag.push((2 * k) as f64 * (2 * k) as f64);
    }
    for _ in 0..size.saturating_sub(1) {
        off.push(q);
    }
    (diag, off)
}

/// Build the Hill matrix for b_{2m+1}(q)
/// (odd-order ≥ 1, odd Mathieu functions).
///
/// Basis: sin((2k+1)·x), k = 0, 1, …
/// Diagonal: (2k+1)²  with a correction of −q on diag[0]
/// Off-diagonal: all q.
///
/// Reference: DLMF 28.4.9–28.4.10.
fn hill_odd_b_matrix(q: f64, size: usize) -> (Vec<f64>, Vec<f64>) {
    let mut diag = Vec::with_capacity(size);
    let mut off = Vec::with_capacity(size - 1);
    for k in 0..size {
        let d = (2 * k + 1) as f64 * (2 * k + 1) as f64;
        diag.push(d);
    }
    // The period-2π boundary condition shifts diagonal[0] by -q
    if !diag.is_empty() {
        diag[0] -= q;
    }
    for _ in 0..size.saturating_sub(1) {
        off.push(q);
    }
    (diag, off)
}

// ────────────────────────────────────────────────────────────────────────────
// Sturm sequence eigenvalue isolation
// ────────────────────────────────────────────────────────────────────────────

/// Count eigenvalues of the symmetric tridiagonal matrix (diag, off_diag)
/// that are strictly less than `lambda`.
///
/// Uses the numerically stable LDL^T factorization of (M - λI).
/// The count of negative diagonal entries in D equals the count of eigenvalues < λ.
///
/// The off-diagonal array has length n-1 (element at index i couples diag[i] to diag[i+1]).
fn sturm_count(diag: &[f64], off_diag: &[f64], lambda: f64) -> usize {
    let n = diag.len();
    if n == 0 {
        return 0;
    }
    let mut count = 0usize;
    let mut d_prev = diag[0] - lambda;
    if d_prev < 0.0 {
        count += 1;
    }
    for i in 1..n {
        let e = if i <= off_diag.len() {
            off_diag[i - 1]
        } else {
            0.0
        };
        let d_curr = if d_prev.abs() > 1e-200 {
            (diag[i] - lambda) - e * e / d_prev
        } else if d_prev >= 0.0 {
            // d_prev ≈ 0+: e²/d_prev → +∞, so d_curr → -∞
            f64::NEG_INFINITY
        } else {
            // d_prev ≈ 0-: e²/d_prev → -∞, so d_curr → +∞
            f64::INFINITY
        };
        if d_curr < 0.0 {
            count += 1;
        }
        d_prev = d_curr;
    }
    count
}

/// Find the k-th eigenvalue (0-indexed, sorted ascending) of a symmetric
/// tridiagonal matrix via Sturm sequence bisection.
///
/// Searches within [global_lo, global_hi] for the k-th eigenvalue.
/// Returns the midpoint of the final bracket.
fn sturm_bisect_kth(
    diag: &[f64],
    off_diag: &[f64],
    k: usize,
    global_lo: f64,
    global_hi: f64,
) -> f64 {
    let mut lo = global_lo;
    let mut hi = global_hi;
    // Ensure bracket: count(lo) <= k < count(hi)
    // Expand if needed, with a cap on iterations
    for _ in 0..60 {
        if sturm_count(diag, off_diag, lo) <= k {
            break;
        }
        let width = (hi - lo).abs().max(1.0);
        lo -= width;
    }
    for _ in 0..60 {
        if sturm_count(diag, off_diag, hi) > k {
            break;
        }
        let width = (hi - lo).abs().max(1.0);
        hi += width;
    }
    for _ in 0..100 {
        let mid = (lo + hi) / 2.0;
        if sturm_count(diag, off_diag, mid) <= k {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo < 1e-13 * (1.0 + lo.abs() + hi.abs()) {
            break;
        }
    }
    (lo + hi) / 2.0
}

/// Find the eigenvalue of a symmetric tridiagonal matrix nearest to `target`.
///
/// Uses Sturm sequence to identify the correct eigenvalue index, then bisects
/// within a broad Gershgorin bracket.
fn find_eigenvalue_near(diag: &[f64], off_diag: &[f64], target: f64) -> f64 {
    let n = diag.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return diag[0];
    }

    // Gershgorin bound: all eigenvalues lie in [lo_bound, hi_bound]
    let (lo_bound, hi_bound) = gershgorin_bounds(diag, off_diag);
    let spread = (hi_bound - lo_bound).abs();
    let lo_global = lo_bound - spread - 2.0;
    let hi_global = hi_bound + spread + 2.0;

    // k_above = number of eigenvalues strictly < target
    // So eigenvalue k_above is the first one >= target (may not exist if k_above == n)
    // Eigenvalue k_above - 1 is the last one < target (may not exist if k_above == 0)
    // We want the eigenvalue NEAREST to target.
    // Candidates: index k_above (if k_above < n) and k_above-1 (if k_above > 0).
    let k_above = sturm_count(diag, off_diag, target);

    // Get the two candidate eigenvalues and pick the nearer one
    let candidate_above = if k_above < n {
        sturm_bisect_kth(diag, off_diag, k_above, lo_global, hi_global)
    } else {
        f64::INFINITY
    };
    let candidate_below = if k_above > 0 {
        sturm_bisect_kth(diag, off_diag, k_above - 1, lo_global, hi_global)
    } else {
        f64::NEG_INFINITY
    };

    // Pick nearest
    let dist_above = (candidate_above - target).abs();
    let dist_below = (candidate_below - target).abs();
    if dist_below <= dist_above {
        candidate_below
    } else {
        candidate_above
    }
}

/// Compute Gershgorin row-sum bounds for the eigenvalues.
fn gershgorin_bounds(diag: &[f64], off_diag: &[f64]) -> (f64, f64) {
    let n = diag.len();
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for i in 0..n {
        let radius = if i == 0 {
            if off_diag.is_empty() {
                0.0
            } else {
                off_diag[0].abs()
            }
        } else if i == n - 1 {
            if off_diag.len() >= i {
                off_diag[i - 1].abs()
            } else {
                0.0
            }
        } else {
            let l = if i < off_diag.len() {
                off_diag[i - 1].abs()
            } else {
                0.0
            };
            let r = if i < off_diag.len() {
                off_diag[i].abs()
            } else {
                0.0
            };
            l + r
        };
        if diag[i] - radius < lo {
            lo = diag[i] - radius;
        }
        if diag[i] + radius > hi {
            hi = diag[i] + radius;
        }
    }
    (lo, hi)
}

// ────────────────────────────────────────────────────────────────────────────
// Tridiagonal system solver (Thomas algorithm)
// ────────────────────────────────────────────────────────────────────────────

/// Solve (M - shift·I) x = rhs for a symmetric tridiagonal M using Thomas algorithm.
/// Returns None if the system is near-singular.
fn tridiag_solve(diag: &[f64], off_diag: &[f64], shift: f64, rhs: &[f64]) -> Option<Vec<f64>> {
    let n = diag.len();
    let mut a: Vec<f64> = diag.iter().map(|&d| d - shift).collect();
    // Build off-diagonal padded to length n-1
    let b: Vec<f64> = {
        let mut bv = off_diag.to_vec();
        bv.resize(n.saturating_sub(1), 0.0);
        bv
    };
    let mut d = rhs.to_vec();

    // Forward sweep
    for i in 1..n {
        if a[i - 1].abs() < 1e-30 {
            return None;
        }
        let m = b[i - 1] / a[i - 1];
        a[i] -= m * b[i - 1];
        d[i] -= m * d[i - 1];
    }
    if a[n - 1].abs() < 1e-30 {
        return None;
    }
    // Back substitution
    let mut x = vec![0.0_f64; n];
    x[n - 1] = d[n - 1] / a[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = (d[i] - b[i] * x[i + 1]) / a[i];
    }
    Some(x)
}

/// Normalize a vector in-place (L2 norm = 1).
fn normalize_vec(v: &mut [f64]) {
    let norm: f64 = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
    if norm > 1e-15 {
        for vi in v.iter_mut() {
            *vi /= norm;
        }
    }
}

/// Compute the eigenvector for the Hill matrix corresponding to `eigenvalue`
/// using shifted inverse iteration (10 steps is always sufficient).
fn hill_eigenvector(
    diag: &[f64],
    off_diag: &[f64],
    eigenvalue: f64,
    sign_index: usize,
) -> Vec<f64> {
    let n = diag.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![1.0];
    }
    // Use a slight shift to avoid exact singularity
    let shift = eigenvalue - 1e-8;
    let mut v = vec![0.0_f64; n];
    // Start with a guess that has energy near the target index
    let idx = sign_index.min(n - 1);
    v[idx] = 1.0;
    normalize_vec(&mut v);

    for _ in 0..30 {
        let v_old = v.clone();
        match tridiag_solve(diag, off_diag, shift, &v_old) {
            Some(mut w) => {
                normalize_vec(&mut w);
                v = w;
            }
            None => {
                // Singular: add small perturbation
                for (i, vi) in v.iter_mut().enumerate() {
                    *vi += 1e-10 * (i as f64 + 1.0).sin();
                }
                normalize_vec(&mut v);
            }
        }
    }
    normalize_vec(&mut v);
    v
}

// ────────────────────────────────────────────────────────────────────────────
// Characteristic value computation (f64 core, then generic wrappers)
// ────────────────────────────────────────────────────────────────────────────

/// Compute a_m(q) using the Hill matrix Sturm sequence method.
fn mathieu_a_f64(m: usize, q: f64) -> f64 {
    if q == 0.0 {
        return (m * m) as f64;
    }
    if m.is_multiple_of(2) {
        // Even-order a: Hill matrix for period-π even functions
        let (diag, off) = hill_even_a_matrix(q, HILL_SIZE);
        let target = (m * m) as f64;
        find_eigenvalue_near(&diag, &off, target)
    } else {
        // Odd-order a: Hill matrix for period-2π even functions
        let (diag, off) = hill_odd_a_matrix(q, HILL_SIZE);
        let target = (m * m) as f64;
        find_eigenvalue_near(&diag, &off, target)
    }
}

/// Compute b_m(q) using the Hill matrix Sturm sequence method.
fn mathieu_b_f64(m: usize, q: f64) -> f64 {
    if m == 0 {
        return f64::INFINITY;
    }
    if q == 0.0 {
        return (m * m) as f64;
    }
    if m.is_multiple_of(2) {
        // Even-order b: Hill matrix for period-π odd functions
        let (diag, off) = hill_even_b_matrix(q, HILL_SIZE);
        let target = (m * m) as f64;
        find_eigenvalue_near(&diag, &off, target)
    } else {
        // Odd-order b: Hill matrix for period-2π odd functions
        let (diag, off) = hill_odd_b_matrix(q, HILL_SIZE);
        let target = (m * m) as f64;
        find_eigenvalue_near(&diag, &off, target)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Public API
// ────────────────────────────────────────────────────────────────────────────

/// Characteristic value of even Mathieu functions
///
/// Computes the characteristic value for the even solution, ce_m(z, q),
/// of Mathieu's differential equation.
///
/// # Arguments
///
/// * `m` - Order of the function (non-negative integer)
/// * `q` - Parameter of the function
///
/// # Returns
///
/// * Characteristic value for the even Mathieu function
///
/// # Examples
///
/// ```
/// use scirs2_special::mathieu_a;
/// use approx::assert_relative_eq;
///
/// // a_0(0) = 0, a_1(0) = 1, etc.
/// assert_relative_eq!(mathieu_a(0, 0.0f64).unwrap(), 0.0, epsilon = 1e-10);
/// assert_relative_eq!(mathieu_a(1, 0.0f64).unwrap(), 1.0, epsilon = 1e-10);
///
/// // a_0(1.0) ≈ -0.4551386 (SciPy reference)
/// let a0 = mathieu_a(0, 1.0f64).unwrap();
/// assert_relative_eq!(a0, -0.4551386041, epsilon = 1e-6);
///
/// // a_1(1.0) ≈ 1.8591080725
/// let a1 = mathieu_a(1, 1.0f64).unwrap();
/// assert_relative_eq!(a1, 1.8591080725, epsilon = 1e-6);
/// ```
#[allow(dead_code)]
pub fn mathieu_a<F>(m: usize, q: F) -> SpecialResult<F>
where
    F: Float + FromPrimitive + Debug,
{
    let q_f64 = q.to_f64().ok_or_else(|| {
        crate::error::SpecialError::ComputationError("Failed to convert q to f64".to_string())
    })?;
    let result = mathieu_a_f64(m, q_f64);
    Ok(const_f64::<F>(result))
}

/// Characteristic value of odd Mathieu functions
///
/// Computes the characteristic value for the odd solution, se_m(z, q),
/// of Mathieu's differential equation.
///
/// # Arguments
///
/// * `m` - Order of the function (non-negative integer, ≥ 1)
/// * `q` - Parameter of the function
///
/// # Returns
///
/// * Characteristic value for the odd Mathieu function
///
/// # Examples
///
/// ```
/// use scirs2_special::mathieu_b;
/// use approx::assert_relative_eq;
///
/// // b_1(0) = 1, b_2(0) = 4, etc.
/// assert_relative_eq!(mathieu_b(1, 0.0f64).unwrap(), 1.0, epsilon = 1e-10);
/// assert_relative_eq!(mathieu_b(2, 0.0f64).unwrap(), 4.0, epsilon = 1e-10);
///
/// // b_1(1.0) ≈ -0.1102488170 (SciPy reference)
/// let b1 = mathieu_b(1, 1.0f64).unwrap();
/// assert_relative_eq!(b1, -0.1102488170, epsilon = 1e-6);
///
/// // b_2(1.0) ≈ 3.9170247730
/// let b2 = mathieu_b(2, 1.0f64).unwrap();
/// assert_relative_eq!(b2, 3.9170247730, epsilon = 1e-6);
/// ```
#[allow(dead_code)]
pub fn mathieu_b<F>(m: usize, q: F) -> SpecialResult<F>
where
    F: Float + FromPrimitive + Debug,
{
    if m == 0 {
        return Ok(F::infinity());
    }
    let q_f64 = q.to_f64().ok_or_else(|| {
        crate::error::SpecialError::ComputationError("Failed to convert q to f64".to_string())
    })?;
    let result = mathieu_b_f64(m, q_f64);
    Ok(const_f64::<F>(result))
}

/// Fourier coefficients for even Mathieu functions
///
/// Computes the Fourier coefficients for the even Mathieu functions.
/// For even m=2n, returns coefficients A_(2n)^(2k) for k=0,1,2,...
/// For odd m=2n+1, returns coefficients A_(2n+1)^(2k+1) for k=0,1,2,...
///
/// # Arguments
///
/// * `m` - Order of the Mathieu function (non-negative integer)
/// * `q` - Parameter of the function (non-negative)
///
/// # Returns
///
/// * Vector of Fourier coefficients for the even Mathieu function
///
/// # Examples
///
/// ```
/// use scirs2_special::mathieu_even_coef;
/// use approx::assert_relative_eq;
///
/// // For q=0, only one non-zero coefficient
/// let coeffs = mathieu_even_coef(0, 0.0f64).unwrap();
/// assert!(coeffs.len() > 0);
/// assert_relative_eq!(coeffs[0], 1.0, epsilon = 1e-10);
///
/// // For q=0, m=2: A_2^2 = 1 at index 1
/// let coeffs2 = mathieu_even_coef(2, 0.0f64).unwrap();
/// assert_relative_eq!(coeffs2[1], 1.0, epsilon = 1e-10);
///
/// // For q=1, m=0: dominant coefficient is near 1 (normalized)
/// let coeffs_q1 = mathieu_even_coef(0, 1.0f64).unwrap();
/// assert!(coeffs_q1[0].abs() > 0.9);
/// assert!(coeffs_q1[1].abs() < 0.5);
/// ```
#[allow(dead_code)]
pub fn mathieu_even_coef<F>(m: usize, q: F) -> SpecialResult<Vec<F>>
where
    F: Float + FromPrimitive + Debug + std::iter::Sum,
{
    let q_f64 = q.to_f64().ok_or_else(|| {
        crate::error::SpecialError::ComputationError("Failed to convert q to f64".to_string())
    })?;
    let coeffs_f64 = compute_even_coefficients_f64(m, q_f64);
    Ok(coeffs_f64.iter().map(|&c| const_f64::<F>(c)).collect())
}

/// Fourier coefficients for odd Mathieu functions
///
/// Computes the Fourier coefficients for the odd Mathieu functions.
/// For odd m=2n+1, returns coefficients B_(2n+1)^(2k+1) for k=0,1,2,...
/// For even m=2n+2, returns coefficients B_(2n+2)^(2k+2) for k=0,1,2,...
///
/// # Arguments
///
/// * `m` - Order of the Mathieu function (non-negative integer, ≥ 1)
/// * `q` - Parameter of the function (non-negative)
///
/// # Returns
///
/// * Vector of Fourier coefficients for the odd Mathieu function
///
/// # Examples
///
/// ```
/// use scirs2_special::mathieu_odd_coef;
/// use approx::assert_relative_eq;
///
/// // For q=0, m=1: B_1^1 = 1 at index 0
/// let coeffs = mathieu_odd_coef(1, 0.0f64).unwrap();
/// assert!(coeffs.len() > 0);
/// assert_relative_eq!(coeffs[0], 1.0, epsilon = 1e-10);
///
/// // For q=0, m=3: B_3^3 = 1 at index 1
/// let coeffs3 = mathieu_odd_coef(3, 0.0f64).unwrap();
/// assert_relative_eq!(coeffs3[1], 1.0, epsilon = 1e-10);
///
/// // For q=1, m=1: dominant coefficient is near 1 (normalized)
/// let coeffs_q1 = mathieu_odd_coef(1, 1.0f64).unwrap();
/// assert!(coeffs_q1[0].abs() > 0.5);
/// ```
#[allow(dead_code)]
pub fn mathieu_odd_coef<F>(m: usize, q: F) -> SpecialResult<Vec<F>>
where
    F: Float + FromPrimitive + Debug + std::iter::Sum,
{
    if m == 0 {
        return Ok(Vec::new());
    }
    let q_f64 = q.to_f64().ok_or_else(|| {
        crate::error::SpecialError::ComputationError("Failed to convert q to f64".to_string())
    })?;
    let coeffs_f64 = compute_odd_coefficients_f64(m, q_f64);
    Ok(coeffs_f64.iter().map(|&c| const_f64::<F>(c)).collect())
}

/// Even Mathieu function and its derivative
///
/// Computes the even Mathieu function, ce_m(x, q), and its derivative.
///
/// # Arguments
///
/// * `m` - Order of the function (non-negative integer)
/// * `q` - Parameter of the function
/// * `x` - Argument of the function (in radians)
///
/// # Returns
///
/// * Tuple containing (function value, derivative value)
///
/// # Examples
///
/// ```
/// use scirs2_special::mathieu_cem;
/// use approx::assert_relative_eq;
/// use std::f64::consts::PI;
///
/// // ce_0(0, x) = 1 for all x (q=0)
/// let (ce0, ce0_p) = mathieu_cem(0, 0.0f64, PI/4.0).unwrap();
/// assert_relative_eq!(ce0, 1.0, epsilon = 1e-10);
/// assert_relative_eq!(ce0_p, 0.0, epsilon = 1e-10);
///
/// // ce_1(0, π/4) = cos(π/4) (q=0)
/// let (ce1, _) = mathieu_cem(1, 0.0f64, PI/4.0).unwrap();
/// assert_relative_eq!(ce1, (PI/4.0).cos(), epsilon = 1e-10);
///
/// // ce_0(q=1, x=π/4): finite and bounded
/// let (ce, ce_p) = mathieu_cem(0, 1.0f64, PI/4.0).unwrap();
/// assert!(ce.is_finite() && ce_p.is_finite());
/// ```
#[allow(dead_code)]
pub fn mathieu_cem<F>(m: usize, q: F, x: F) -> SpecialResult<(F, F)>
where
    F: Float + FromPrimitive + Debug + std::iter::Sum,
{
    let coeffs = mathieu_even_coef(m, q)?;
    evaluate_even_mathieu(m, x, &coeffs)
}

/// Odd Mathieu function and its derivative
///
/// Computes the odd Mathieu function, se_m(x, q), and its derivative.
///
/// # Arguments
///
/// * `m` - Order of the function (non-negative integer)
/// * `q` - Parameter of the function
/// * `x` - Argument of the function (in radians)
///
/// # Returns
///
/// * Tuple containing (function value, derivative value)
///
/// # Examples
///
/// ```
/// use scirs2_special::mathieu_sem;
/// use approx::assert_relative_eq;
/// use std::f64::consts::PI;
///
/// // se_1(0, π/4) = sin(π/4)
/// let (se1, se1_p) = mathieu_sem(1, 0.0f64, PI/4.0).unwrap();
/// assert_relative_eq!(se1, (PI/4.0).sin(), epsilon = 1e-10);
/// assert_relative_eq!(se1_p, (PI/4.0).cos(), epsilon = 1e-10);
///
/// // se_2(0, π/4) = sin(2*π/4) = sin(π/2) = 1
/// let (se2, _) = mathieu_sem(2, 0.0f64, PI/4.0).unwrap();
/// assert_relative_eq!(se2, (PI/2.0).sin(), epsilon = 1e-10);
///
/// // se_1(q=1, x=π/4): finite
/// let (se, se_p) = mathieu_sem(1, 1.0f64, PI/4.0).unwrap();
/// assert!(se.is_finite() && se_p.is_finite());
/// ```
#[allow(dead_code)]
pub fn mathieu_sem<F>(m: usize, q: F, x: F) -> SpecialResult<(F, F)>
where
    F: Float + FromPrimitive + Debug + std::iter::Sum,
{
    if m == 0 {
        return Ok((F::zero(), F::zero()));
    }
    let coeffs = mathieu_odd_coef(m, q)?;
    evaluate_odd_mathieu(m, x, &coeffs)
}

// ────────────────────────────────────────────────────────────────────────────
// Fourier coefficient computation (f64 core)
// ────────────────────────────────────────────────────────────────────────────

/// Compute normalized Fourier coefficients for ce_m(q, x) from the Hill eigenvector.
fn compute_even_coefficients_f64(m: usize, q: f64) -> Vec<f64> {
    let num_coeffs = 30;
    if q == 0.0 {
        let mut coeffs = vec![0.0_f64; num_coeffs];
        let idx = if m.is_multiple_of(2) {
            m / 2
        } else {
            (m - 1) / 2
        };
        if idx < num_coeffs {
            coeffs[idx] = 1.0;
        }
        return coeffs;
    }

    let eigenvalue = mathieu_a_f64(m, q);

    let (diag, off) = if m.is_multiple_of(2) {
        hill_even_a_matrix(q, HILL_SIZE)
    } else {
        hill_odd_a_matrix(q, HILL_SIZE)
    };

    let sign_index = if m.is_multiple_of(2) {
        m / 2
    } else {
        (m - 1) / 2
    };
    let mut v = hill_eigenvector(&diag, &off, eigenvalue, sign_index);

    // Enforce sign convention: dominant coefficient is positive
    let idx = sign_index.min(v.len().saturating_sub(1));
    if idx < v.len() && v[idx] < 0.0 {
        for vi in v.iter_mut() {
            *vi = -*vi;
        }
    }

    // Truncate to num_coeffs
    v.truncate(num_coeffs);
    while v.len() < num_coeffs {
        v.push(0.0);
    }
    v
}

/// Compute normalized Fourier coefficients for se_m(q, x) from the Hill eigenvector.
fn compute_odd_coefficients_f64(m: usize, q: f64) -> Vec<f64> {
    if m == 0 {
        return vec![];
    }
    let num_coeffs = 30;
    if q == 0.0 {
        let mut coeffs = vec![0.0_f64; num_coeffs];
        let idx = if m % 2 == 1 { (m - 1) / 2 } else { (m - 2) / 2 };
        if idx < num_coeffs {
            coeffs[idx] = 1.0;
        }
        return coeffs;
    }

    let eigenvalue = mathieu_b_f64(m, q);

    let (diag, off) = if m.is_multiple_of(2) {
        hill_even_b_matrix(q, HILL_SIZE)
    } else {
        hill_odd_b_matrix(q, HILL_SIZE)
    };

    // For even-order b (sin(2k·x), k=1,…), sign_index is (m/2 - 1)
    // For odd-order b (sin((2k+1)·x), k=0,…), sign_index is (m-1)/2
    let sign_index = if m.is_multiple_of(2) {
        m / 2 - 1
    } else {
        (m - 1) / 2
    };
    let mut v = hill_eigenvector(&diag, &off, eigenvalue, sign_index);

    // Enforce sign convention: dominant coefficient is positive
    let idx = sign_index.min(v.len().saturating_sub(1));
    if idx < v.len() && v[idx] < 0.0 {
        for vi in v.iter_mut() {
            *vi = -*vi;
        }
    }

    // Truncate to num_coeffs
    v.truncate(num_coeffs);
    while v.len() < num_coeffs {
        v.push(0.0);
    }
    v
}

// ────────────────────────────────────────────────────────────────────────────
// Fourier series evaluation
// ────────────────────────────────────────────────────────────────────────────

fn evaluate_even_mathieu<F>(m: usize, x: F, coeffs: &[F]) -> SpecialResult<(F, F)>
where
    F: Float + FromPrimitive + Debug,
{
    let mut result = F::zero();
    let mut derivative = F::zero();

    if m.is_multiple_of(2) {
        // ce_{2n}(x) = Σ_k A_{2k} cos(2k·x)
        for (k, &coef) in coeffs.iter().enumerate() {
            let freq = const_f64::<F>((2 * k) as f64);
            let arg = freq * x;
            result = result + coef * arg.cos();
            derivative = derivative - coef * freq * arg.sin();
        }
    } else {
        // ce_{2n+1}(x) = Σ_k A_{2k+1} cos((2k+1)·x)
        for (k, &coef) in coeffs.iter().enumerate() {
            let freq = const_f64::<F>((2 * k + 1) as f64);
            let arg = freq * x;
            result = result + coef * arg.cos();
            derivative = derivative - coef * freq * arg.sin();
        }
    }

    Ok((result, derivative))
}

fn evaluate_odd_mathieu<F>(m: usize, x: F, coeffs: &[F]) -> SpecialResult<(F, F)>
where
    F: Float + FromPrimitive + Debug,
{
    if m == 0 || coeffs.is_empty() {
        return Ok((F::zero(), F::zero()));
    }

    let mut result = F::zero();
    let mut derivative = F::zero();

    if m % 2 == 1 {
        // se_{2n+1}(x) = Σ_k B_{2k+1} sin((2k+1)·x)
        for (k, &coef) in coeffs.iter().enumerate() {
            let freq = const_f64::<F>((2 * k + 1) as f64);
            let arg = freq * x;
            result = result + coef * arg.sin();
            derivative = derivative + coef * freq * arg.cos();
        }
    } else {
        // se_{2n+2}(x) = Σ_k B_{2k+2} sin((2k+2)·x)
        for (k, &coef) in coeffs.iter().enumerate() {
            let freq = const_f64::<F>((2 * k + 2) as f64);
            let arg = freq * x;
            result = result + coef * arg.sin();
            derivative = derivative + coef * freq * arg.cos();
        }
    }

    Ok((result, derivative))
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    // Reference values from scipy.special.mathieu_a / mathieu_b
    const A0_Q1: f64 = -0.4551386041;
    const A1_Q1: f64 = 1.8591080725;
    const A2_Q1: f64 = 4.3713009827;
    const A3_Q1: f64 = 9.0783688472;
    const B1_Q1: f64 = -0.1102488170;
    const B2_Q1: f64 = 3.9170247730;
    const B3_Q1: f64 = 9.0477392598;

    const A0_Q01: f64 = -0.0049945438;
    const A1_Q01: f64 = 1.0987343130;
    const A2_Q01: f64 = 4.0041611598;
    const B1_Q01: f64 = 0.8987655570;
    const B2_Q01: f64 = 3.9991667028;

    #[test]
    fn test_mathieu_a_special_cases() {
        // q = 0: a_m = m²
        assert_relative_eq!(mathieu_a::<f64>(0, 0.0).unwrap(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(mathieu_a::<f64>(1, 0.0).unwrap(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(mathieu_a::<f64>(2, 0.0).unwrap(), 4.0, epsilon = 1e-10);
        assert_relative_eq!(mathieu_a::<f64>(3, 0.0).unwrap(), 9.0, epsilon = 1e-10);
        assert_relative_eq!(mathieu_a::<f64>(4, 0.0).unwrap(), 16.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mathieu_b_special_cases() {
        // q = 0: b_m = m²
        assert_relative_eq!(mathieu_b::<f64>(1, 0.0).unwrap(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(mathieu_b::<f64>(2, 0.0).unwrap(), 4.0, epsilon = 1e-10);
        assert_relative_eq!(mathieu_b::<f64>(3, 0.0).unwrap(), 9.0, epsilon = 1e-10);
        assert_relative_eq!(mathieu_b::<f64>(4, 0.0).unwrap(), 16.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mathieu_a_q1_reference() {
        // Match SciPy reference values at q=1 to 1e-5 tolerance
        assert_relative_eq!(mathieu_a::<f64>(0, 1.0).unwrap(), A0_Q1, epsilon = 1e-5);
        assert_relative_eq!(mathieu_a::<f64>(1, 1.0).unwrap(), A1_Q1, epsilon = 1e-5);
        assert_relative_eq!(mathieu_a::<f64>(2, 1.0).unwrap(), A2_Q1, epsilon = 1e-5);
        assert_relative_eq!(mathieu_a::<f64>(3, 1.0).unwrap(), A3_Q1, epsilon = 1e-5);
    }

    #[test]
    fn test_mathieu_b_q1_reference() {
        // Match SciPy reference values at q=1 to 1e-5 tolerance
        assert_relative_eq!(mathieu_b::<f64>(1, 1.0).unwrap(), B1_Q1, epsilon = 1e-5);
        assert_relative_eq!(mathieu_b::<f64>(2, 1.0).unwrap(), B2_Q1, epsilon = 1e-5);
        assert_relative_eq!(mathieu_b::<f64>(3, 1.0).unwrap(), B3_Q1, epsilon = 1e-5);
    }

    #[test]
    fn test_mathieu_a_q01_reference() {
        assert_relative_eq!(mathieu_a::<f64>(0, 0.1).unwrap(), A0_Q01, epsilon = 1e-6);
        assert_relative_eq!(mathieu_a::<f64>(1, 0.1).unwrap(), A1_Q01, epsilon = 1e-6);
        assert_relative_eq!(mathieu_a::<f64>(2, 0.1).unwrap(), A2_Q01, epsilon = 1e-6);
    }

    #[test]
    fn test_mathieu_b_q01_reference() {
        assert_relative_eq!(mathieu_b::<f64>(1, 0.1).unwrap(), B1_Q01, epsilon = 1e-6);
        assert_relative_eq!(mathieu_b::<f64>(2, 0.1).unwrap(), B2_Q01, epsilon = 1e-6);
    }

    #[test]
    fn test_mathieu_small_q() {
        // Verify sign properties for small q
        let a0 = mathieu_a::<f64>(0, 0.1).unwrap();
        let a1 = mathieu_a::<f64>(1, 0.1).unwrap();
        let b1 = mathieu_b::<f64>(1, 0.1).unwrap();
        let b2 = mathieu_b::<f64>(2, 0.1).unwrap();

        // a₀ should be negative for small positive q
        assert!(a0 < 0.0);
        // a₁ should be > 1 for small positive q
        assert!(a1 > 1.0);
        // b₁ should be < 1 for small positive q
        assert!(b1 < 1.0);
        // b₂ should be < 4 for small positive q
        assert!(b2 < 4.0);
    }

    #[test]
    fn test_even_fourier_coefficients() {
        // For q=0, only one coefficient is non-zero
        let coeffs0 = mathieu_even_coef::<f64>(0, 0.0).unwrap();
        assert!(!coeffs0.is_empty());
        assert_relative_eq!(coeffs0[0], 1.0, epsilon = 1e-10);

        let coeffs2 = mathieu_even_coef::<f64>(2, 0.0).unwrap();
        assert!(coeffs2.len() > 1);
        assert_relative_eq!(coeffs2[1], 1.0, epsilon = 1e-10);

        // For small q, the dominant coefficient is large (near 1) and others are small
        let coeffs_small_q = mathieu_even_coef::<f64>(0, 0.1).unwrap();
        assert!(coeffs_small_q.len() > 1);
        // Tightened tolerance: dominant coefficient A_0 should be near 1
        assert!(coeffs_small_q[0].abs() > 0.99);
        // Second coefficient A_2 should be small
        assert!(coeffs_small_q[1].abs() < 0.15);
    }

    #[test]
    fn test_odd_fourier_coefficients() {
        // For q=0, only one coefficient is non-zero
        let coeffs1 = mathieu_odd_coef::<f64>(1, 0.0).unwrap();
        assert!(!coeffs1.is_empty());
        assert_relative_eq!(coeffs1[0], 1.0, epsilon = 1e-10);

        let coeffs3 = mathieu_odd_coef::<f64>(3, 0.0).unwrap();
        assert!(coeffs3.len() > 1);
        assert_relative_eq!(coeffs3[1], 1.0, epsilon = 1e-10);

        // For small q, the dominant coefficient is near 1
        let coeffs_small_q = mathieu_odd_coef::<f64>(1, 0.1).unwrap();
        assert!(coeffs_small_q.len() > 1);
        // Tightened tolerance: dominant coefficient B_1 should be near 1
        assert!(coeffs_small_q[0].abs() > 0.99);
        // Second coefficient B_3 should be small
        assert!(coeffs_small_q[1].abs() < 0.15);
    }

    #[test]
    fn test_mathieu_cem_sem_zero_q() {
        // For q=0, ce₀(x) = 1
        let (ce0, ce0_prime) = mathieu_cem(0, 0.0f64, PI / 4.0).unwrap();
        assert_relative_eq!(ce0, 1.0, epsilon = 1e-10);
        assert_relative_eq!(ce0_prime, 0.0, epsilon = 1e-10);

        // For q=0, ce₁(x) = cos(x)
        let (ce1, ce1_prime) = mathieu_cem(1, 0.0f64, PI / 4.0).unwrap();
        assert_relative_eq!(ce1, (PI / 4.0).cos(), epsilon = 1e-10);
        assert_relative_eq!(ce1_prime, -(PI / 4.0).sin(), epsilon = 1e-10);

        // For q=0, se₁(x) = sin(x)
        let (se1, se1_prime) = mathieu_sem(1, 0.0f64, PI / 4.0).unwrap();
        assert_relative_eq!(se1, (PI / 4.0).sin(), epsilon = 1e-10);
        assert_relative_eq!(se1_prime, (PI / 4.0).cos(), epsilon = 1e-10);

        // For q=0, se₂(x) = sin(2x)
        let (se2, se2_prime) = mathieu_sem(2, 0.0f64, PI / 4.0).unwrap();
        assert_relative_eq!(se2, (PI / 2.0).sin(), epsilon = 1e-10);
        assert_relative_eq!(se2_prime, 2.0 * (PI / 2.0).cos(), epsilon = 1e-10);
    }

    #[test]
    fn test_mathieu_cem_finite_nonzero_q() {
        let (ce, ce_p) = mathieu_cem(0, 1.0f64, PI / 4.0).unwrap();
        assert!(ce.is_finite() && ce_p.is_finite());
        let (ce1, ce1_p) = mathieu_cem(1, 1.0f64, PI / 4.0).unwrap();
        assert!(ce1.is_finite() && ce1_p.is_finite());
    }

    #[test]
    fn test_mathieu_sem_finite_nonzero_q() {
        let (se, se_p) = mathieu_sem(1, 1.0f64, PI / 4.0).unwrap();
        assert!(se.is_finite() && se_p.is_finite());
        let (se2, se2_p) = mathieu_sem(2, 1.0f64, PI / 4.0).unwrap();
        assert!(se2.is_finite() && se2_p.is_finite());
    }

    #[test]
    fn test_a_b_interleaving() {
        // For q > 0: a_0 < b_1 < a_1 < b_2 < a_2 < b_3 < a_3 < ...
        let q = 1.0_f64;
        let a0 = mathieu_a::<f64>(0, q).unwrap();
        let b1 = mathieu_b::<f64>(1, q).unwrap();
        let a1 = mathieu_a::<f64>(1, q).unwrap();
        let b2 = mathieu_b::<f64>(2, q).unwrap();
        let a2 = mathieu_a::<f64>(2, q).unwrap();
        let b3 = mathieu_b::<f64>(3, q).unwrap();
        let a3 = mathieu_a::<f64>(3, q).unwrap();
        assert!(a0 < b1, "a_0 < b_1 failed: {} vs {}", a0, b1);
        assert!(b1 < a1, "b_1 < a_1 failed: {} vs {}", b1, a1);
        assert!(a1 < b2, "a_1 < b_2 failed: {} vs {}", a1, b2);
        assert!(b2 < a2, "b_2 < a_2 failed: {} vs {}", b2, a2);
        assert!(a2 < b3, "a_2 < b_3 failed: {} vs {}", a2, b3);
        assert!(b3 < a3, "b_3 < a_3 failed: {} vs {}", b3, a3);
    }
}
