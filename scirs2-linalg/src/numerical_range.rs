//! Numerical range and spectral analysis for matrices
//!
//! This module provides:
//!
//! - **Field of Values** (`field_of_values`): W(A) = {x^H A x : ‖x‖ = 1}
//!   via Johnson's random-vector + boundary-tracing algorithm.
//! - **Numerical Radius** (`numerical_radius`): w(A) = max{|z| : z ∈ W(A)}
//! - **Kreiss Number** (`kreiss_number`): K(A) = sup_{Re(z)>0} Re(z) ‖(zI−A)^{-1}‖
//! - **ε-pseudospectrum** (`pseudospectrum`): Σ_ε(A) grid of σ_min(zI−A) values
//! - **Spectral Abscissa** (`spectral_abscissa`): max Re(λ)
//! - **Spectral Radius** (`spectral_radius`): max |λ|

use crate::decomposition::svd;
use crate::eigen::standard::eig;
use crate::error::{LinalgError, LinalgResult};
use crate::solve::solve;
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, ScalarOperand};
use scirs2_core::numeric::{Complex, Float, NumAssign};
use scirs2_core::random::prelude::Rng;
use scirs2_core::random::SeedableRng;
use scirs2_core::RngExt;
use std::iter::Sum;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Draw a unit vector in R^n with entries from a standard normal distribution,
/// then normalise it.  Returns `None` when the zero vector is sampled (negligible
/// probability; callers should retry).
fn random_unit_vector<F>(n: usize, rng: &mut impl Rng) -> Option<Array1<F>>
where
    F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static,
{
    let mut v = Array1::<F>::zeros(n);
    for i in 0..n {
        // Box-Muller or simple normal approximation via 12-uniform sum
        let u: f64 = rng.random_range(-1.0_f64..1.0_f64);
        v[i] = F::from(u).unwrap_or(F::zero());
    }
    let norm_sq: F = v.iter().fold(F::zero(), |acc, &x| acc + x * x);
    if norm_sq <= F::epsilon() {
        return None;
    }
    let norm = norm_sq.sqrt();
    for vi in v.iter_mut() {
        *vi /= norm;
    }
    Some(v)
}

/// Compute x^T A x (Rayleigh quotient, real version) for a real matrix A.
fn rayleigh_quotient_real<F>(a: &ArrayView2<F>, x: &Array1<F>) -> F
where
    F: Float + NumAssign + Sum + ScalarOperand,
{
    let ax = a.dot(x);
    x.iter()
        .zip(ax.iter())
        .fold(F::zero(), |acc, (&xi, &axi)| acc + xi * axi)
}

/// Compute the smallest singular value of the matrix M = zI - A.
///
/// We form the matrix explicitly and then compute the minimum singular value
/// via SVD.
fn smallest_singular_value_shifted<F>(a: &ArrayView2<F>, z_re: F, z_im: F) -> LinalgResult<F>
where
    F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static,
{
    let n = a.nrows();
    // Build real 2n×2n matrix representing (zI - A) acting on ℝ^{2n} ≅ ℂ^n
    // For z = a+bi, M_real = [a*I - A_re,  -b*I - A_im; b*I + A_im,  a*I - A_re]
    // Since A is real (A_im = 0): M_real = [a*I - A,  -b*I; b*I,  a*I - A]
    let mut m = Array2::<F>::zeros((2 * n, 2 * n));
    for i in 0..n {
        for j in 0..n {
            let aij = a[[i, j]];
            m[[i, j]] = if i == j { z_re - aij } else { -aij };
            m[[i + n, j + n]] = if i == j { z_re - aij } else { -aij };
            m[[i, j + n]] = if i == j { -z_im } else { F::zero() };
            m[[i + n, j]] = if i == j { z_im } else { F::zero() };
        }
    }
    let (_u, s, _vt) = svd(&m.view(), false, None)?;
    // The 2n×2n real representation has each complex singular value appearing
    // twice; we take the minimum and divide by sqrt(2) to get the true
    // complex singular value.
    let two = F::from(2.0_f64).unwrap_or(F::one() + F::one());
    let min_sv = s
        .iter()
        .cloned()
        .fold(F::infinity(), |a, b| if b < a { b } else { a });
    Ok(min_sv / two.sqrt())
}

// ---------------------------------------------------------------------------
// Field of Values
// ---------------------------------------------------------------------------

/// Compute the numerical range (field of values) of a real square matrix.
///
/// The numerical range W(A) = { x^H A x : x ∈ ℂ^n, ‖x‖ = 1 } is a convex
/// compact subset of ℂ.  For real matrices each x^H A x = x^T A x + iθ where
/// θ comes from the skew-symmetric part.
///
/// The boundary is traced using the **Johnson rotating-eigenvector** method:
/// for each angle θ the extreme point of W(A) in direction e^{iθ} is the
/// largest eigenvalue of (e^{-iθ} A + e^{iθ} A^H)/2 = cos θ · H + sin θ · S
/// where H = (A+Aᵀ)/2 (symmetric part) and S = (A−Aᵀ)/2 (skew-symmetric).
///
/// # Arguments
/// * `a`        - Input square real matrix (n×n)
/// * `n_samples` - Number of angular samples for boundary tracing (≥ 4)
///
/// # Returns
/// `(real_parts, imag_parts)` - boundary point coordinates
///
/// # Errors
/// Returns `LinalgError` on invalid input or computation failure.
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::numerical_range::field_of_values;
///
/// let a = array![[2.0_f64, 1.0], [0.0, 3.0]];
/// let (re, im) = field_of_values(&a.view(), 64).expect("fov failed");
/// assert_eq!(re.len(), im.len());
/// assert!(re.len() >= 4);
/// ```
pub fn field_of_values<F>(a: &ArrayView2<F>, n_samples: usize) -> LinalgResult<(Vec<F>, Vec<F>)>
where
    F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static,
{
    let n = a.nrows();
    if n == 0 {
        return Err(LinalgError::InvalidInputError(
            "field_of_values: matrix must be non-empty".to_string(),
        ));
    }
    if a.ncols() != n {
        return Err(LinalgError::ShapeError(
            "field_of_values: matrix must be square".to_string(),
        ));
    }
    if n_samples < 4 {
        return Err(LinalgError::ValueError(
            "field_of_values: n_samples must be >= 4".to_string(),
        ));
    }

    // Symmetric part H = (A + Aᵀ)/2 and skew-symmetric part S = (A - Aᵀ)/2
    let two = F::from(2.0_f64).unwrap_or(F::one() + F::one());
    let mut h = Array2::<F>::zeros((n, n));
    let mut s = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let aij = a[[i, j]];
            let aji = a[[j, i]];
            h[[i, j]] = (aij + aji) / two;
            s[[i, j]] = (aij - aji) / two;
        }
    }

    let pi = F::from(std::f64::consts::PI).unwrap_or(F::one());
    let two_pi = two * pi;

    let mut re_pts = Vec::with_capacity(n_samples);
    let mut im_pts = Vec::with_capacity(n_samples);

    for k in 0..n_samples {
        let theta =
            two_pi * F::from(k).unwrap_or(F::zero()) / F::from(n_samples).unwrap_or(F::one());
        let cos_t = theta.cos();
        let sin_t = theta.sin();

        // H(θ) = cos(θ) * H + sin(θ) * S  (real symmetric)
        let mut h_theta = Array2::<F>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                h_theta[[i, j]] = cos_t * h[[i, j]] + sin_t * s[[i, j]];
            }
        }

        // The boundary point in direction e^{iθ} is λ_max(H(θ)) * e^{iθ}
        // where λ_max is the largest eigenvalue of H(θ).
        //
        // For the symmetric matrix H(θ) use power iteration to find λ_max.
        let lambda_max = power_iteration_max_eigenvalue(&h_theta.view(), 200)?;

        re_pts.push(lambda_max * cos_t);
        im_pts.push(lambda_max * sin_t);
    }

    Ok((re_pts, im_pts))
}

/// Power iteration to approximate the largest-magnitude eigenvalue of a
/// symmetric matrix.  Returns the Rayleigh quotient at convergence.
fn power_iteration_max_eigenvalue<F>(a: &ArrayView2<F>, max_iter: usize) -> LinalgResult<F>
where
    F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static,
{
    let n = a.nrows();
    if n == 0 {
        return Ok(F::zero());
    }

    let mut rng = scirs2_core::random::rngs::SmallRng::seed_from_u64(12345);
    let mut v = loop {
        if let Some(u) = random_unit_vector(n, &mut rng) {
            break u;
        }
    };

    let tol = F::from(1e-10_f64).unwrap_or(F::epsilon());
    let mut lambda_prev = F::zero();

    for _iter in 0..max_iter {
        let av = a.dot(&v);
        let lambda = rayleigh_quotient_real(a, &v);

        // Check convergence
        if (lambda - lambda_prev).abs() < tol {
            return Ok(lambda);
        }
        lambda_prev = lambda;

        // Renormalise
        let norm_sq: F = av.iter().fold(F::zero(), |acc, &x| acc + x * x);
        if norm_sq <= F::epsilon() {
            break;
        }
        let norm = norm_sq.sqrt();
        v = av.mapv(|x| x / norm);
    }

    Ok(lambda_prev)
}

// ---------------------------------------------------------------------------
// Numerical Radius
// ---------------------------------------------------------------------------

/// Compute the numerical radius w(A) = max{ |z| : z ∈ W(A) }.
///
/// Uses a randomised power-iteration-like method combined with boundary
/// tracing of the field of values.
///
/// # Arguments
/// * `a` - Input square real matrix
///
/// # Returns
/// The numerical radius w(A) ≥ 0.
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::numerical_range::numerical_radius;
///
/// // Diagonal matrix: numerical radius = largest |eigenvalue|
/// let a = array![[3.0_f64, 0.0], [0.0, -2.0]];
/// let w = numerical_radius(&a.view()).expect("numerical_radius failed");
/// assert!((w - 3.0).abs() < 0.1);
/// ```
pub fn numerical_radius<F>(a: &ArrayView2<F>) -> LinalgResult<F>
where
    F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static,
{
    let n_samples = 256;
    let (re_pts, im_pts) = field_of_values(a, n_samples)?;

    let radius = re_pts
        .iter()
        .zip(im_pts.iter())
        .map(|(&r, &i)| (r * r + i * i).sqrt())
        .fold(F::zero(), |acc, v| if v > acc { v } else { acc });

    Ok(radius)
}

// ---------------------------------------------------------------------------
// Spectral Abscissa and Spectral Radius
// ---------------------------------------------------------------------------

/// Compute the spectral abscissa α(A) = max{ Re(λ) : λ eigenvalue of A }.
///
/// # Arguments
/// * `a` - Input square real matrix (n×n)
///
/// # Returns
/// Spectral abscissa (may be negative for stable matrices).
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::numerical_range::spectral_abscissa;
///
/// let a = array![[-1.0_f64, 0.0], [0.0, -2.0]];
/// let alpha = spectral_abscissa(&a.view()).expect("spectral_abscissa failed");
/// assert!((alpha - (-1.0)).abs() < 1e-10);
/// ```
pub fn spectral_abscissa<F>(a: &ArrayView2<F>) -> LinalgResult<F>
where
    F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static,
{
    let n = a.nrows();
    if n == 0 {
        return Err(LinalgError::InvalidInputError(
            "spectral_abscissa: matrix must be non-empty".to_string(),
        ));
    }
    if a.ncols() != n {
        return Err(LinalgError::ShapeError(
            "spectral_abscissa: matrix must be square".to_string(),
        ));
    }

    let (eigenvalues, _) = eig(a, None)?;

    let alpha = eigenvalues
        .iter()
        .map(|z| z.re)
        .fold(F::neg_infinity(), |acc, re| if re > acc { re } else { acc });

    Ok(alpha)
}

/// Compute the spectral radius ρ(A) = max{ |λ| : λ eigenvalue of A }.
///
/// # Arguments
/// * `a` - Input square real matrix (n×n)
///
/// # Returns
/// Spectral radius ≥ 0.
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::numerical_range::spectral_radius;
///
/// let a = array![[0.0_f64, -2.0], [2.0, 0.0]];
/// let rho = spectral_radius(&a.view()).expect("spectral_radius failed");
/// assert!((rho - 2.0).abs() < 1e-10);
/// ```
pub fn spectral_radius<F>(a: &ArrayView2<F>) -> LinalgResult<F>
where
    F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static,
{
    let n = a.nrows();
    if n == 0 {
        return Err(LinalgError::InvalidInputError(
            "spectral_radius: matrix must be non-empty".to_string(),
        ));
    }
    if a.ncols() != n {
        return Err(LinalgError::ShapeError(
            "spectral_radius: matrix must be square".to_string(),
        ));
    }

    let (eigenvalues, _) = eig(a, None)?;

    let rho = eigenvalues
        .iter()
        .map(|z| (z.re * z.re + z.im * z.im).sqrt())
        .fold(F::zero(), |acc, r| if r > acc { r } else { acc });

    Ok(rho)
}

// ---------------------------------------------------------------------------
// Kreiss Number
// ---------------------------------------------------------------------------

/// Compute the Kreiss number K(A) = sup_{ Re(z) > 0 } Re(z) · ‖(zI − A)^{-1}‖₂.
///
/// The Kreiss matrix theorem states ρ(A) ≤ K(A) and for any time-dependent
/// system ‖e^{tA}‖ ≤ e · K(A) · t (roughly).
///
/// The supremum is computed over a grid of points on the right half-plane.
/// We use a logarithmically spaced grid in the radial direction and a uniform
/// angular grid in (-π/2, π/2).
///
/// # Arguments
/// * `a`         - Input square real matrix (n×n)
/// * `grid_size` - Number of points in each direction (default 32)
///
/// # Returns
/// Kreiss number K(A) ≥ 0.
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::numerical_range::kreiss_number;
///
/// let a = array![[0.0_f64, 1.0], [-1.0, 0.0]];  // purely imaginary eigenvalues
/// let k = kreiss_number(&a.view(), 16).expect("kreiss_number failed");
/// assert!(k >= 0.0);
/// ```
pub fn kreiss_number<F>(a: &ArrayView2<F>, grid_size: usize) -> LinalgResult<F>
where
    F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static,
{
    let n = a.nrows();
    if n == 0 {
        return Err(LinalgError::InvalidInputError(
            "kreiss_number: matrix must be non-empty".to_string(),
        ));
    }
    if a.ncols() != n {
        return Err(LinalgError::ShapeError(
            "kreiss_number: matrix must be square".to_string(),
        ));
    }
    let gs = grid_size.max(4);

    // Determine scale from the spectral radius
    let rho = spectral_radius(a)?;
    let scale = (rho + F::one()).max(F::one());

    let pi = F::from(std::f64::consts::PI).unwrap_or(F::one());
    let half_pi = pi / (F::one() + F::one());
    let half = F::from(0.5_f64).unwrap_or(F::one() / (F::one() + F::one()));

    let mut kreiss = F::zero();

    // Iterate over a grid on the right half-plane
    for ir in 0..gs {
        // Logarithmically spaced radii: r from 1e-3*scale to 3*scale
        let r_log_min = -3.0_f64 + scale.to_f64().unwrap_or(1.0).log10();
        let r_log_max = scale.to_f64().unwrap_or(1.0).log10() + 0.6_f64;
        let r_log = r_log_min + (r_log_max - r_log_min) * ir as f64 / (gs - 1) as f64;
        let r = F::from(10.0_f64.powf(r_log)).unwrap_or(F::one());

        for ia in 0..gs {
            // Angles in (-π/2, π/2) excluding the boundary
            let frac = (ia as f64 + 0.5) / gs as f64;
            let theta = half_pi * (F::from(frac * 2.0 - 1.0).unwrap_or(F::zero()));

            let z_re = r * theta.cos();
            let z_im = r * theta.sin();

            // Only consider Re(z) > 0
            if z_re <= F::zero() {
                continue;
            }

            // σ_min(zI - A) via smallest singular value
            let sigma_min = match smallest_singular_value_shifted(a, z_re, z_im) {
                Ok(s) => s,
                Err(_) => continue,
            };

            if sigma_min <= F::zero() {
                continue;
            }

            // Re(z) / σ_min(zI - A)  (since ‖(zI-A)^{-1}‖₂ = 1/σ_min)
            let candidate = z_re / sigma_min;
            if candidate > kreiss {
                kreiss = candidate;
            }

            // Also check the purely real axis point (z_im = 0)
            if ia == gs / 2 {
                let sigma_real = match smallest_singular_value_shifted(a, r, F::zero()) {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                if sigma_real > F::zero() {
                    let c = r / sigma_real;
                    if c > kreiss {
                        kreiss = c;
                    }
                }
            }
        }

        // Also check a few points close to the imaginary axis
        for ia in 0..4 {
            let eps_re = scale * F::from(1e-4_f64 * (ia + 1) as f64).unwrap_or(F::epsilon());
            let y = scale * F::from(ia as f64 / 3.0).unwrap_or(F::zero());
            let sigma = match smallest_singular_value_shifted(a, eps_re, y) {
                Ok(s) => s,
                Err(_) => continue,
            };
            if sigma > F::zero() {
                let c = eps_re / sigma;
                if c > kreiss {
                    kreiss = c;
                }
            }
        }

        // Suppress unused warning
        let _ = half;
    }

    Ok(kreiss)
}

// ---------------------------------------------------------------------------
// ε-Pseudospectrum
// ---------------------------------------------------------------------------

/// Compute the ε-pseudospectrum on a grid.
///
/// The ε-pseudospectrum Σ_ε(A) = { z ∈ ℂ : σ_min(zI − A) ≤ ε }.
/// This function returns the **grid of σ_min values** so the caller can
/// identify regions where σ_min ≤ ε for any desired ε.
///
/// # Arguments
/// * `a`            - Input square real matrix (n×n)
/// * `epsilon_values` - Epsilon values used to define contour levels (unused
///   in the computation but returned for convenience)
/// * `grid_size`    - Grid resolution (produces a `grid_size × grid_size` output)
/// * `x_range`      - Optional `(x_min, x_max)` for the real axis; if `None`
///   an automatic range is derived from eigenvalues.
/// * `y_range`      - Optional `(y_min, y_max)` for the imaginary axis.
///
/// # Returns
/// `PseudospectrumResult` containing the grid coordinates and σ_min values.
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::numerical_range::pseudospectrum;
///
/// let a = array![[1.0_f64, 0.0], [0.0, 2.0]];
/// let result = pseudospectrum(&a.view(), &[0.1, 0.5], 8, None, None)
///     .expect("pseudospectrum failed");
/// assert_eq!(result.sigma_min.shape(), &[8, 8]);
/// ```
pub struct PseudospectrumResult<F> {
    /// Real-axis grid coordinates (length = grid_size)
    pub x_grid: Vec<F>,
    /// Imaginary-axis grid coordinates (length = grid_size)
    pub y_grid: Vec<F>,
    /// σ_min(zI − A) for each grid point; shape (grid_size, grid_size).
    /// Index: `sigma_min[[iy, ix]]`.
    pub sigma_min: Array2<F>,
}

pub fn pseudospectrum<F>(
    a: &ArrayView2<F>,
    _epsilon_values: &[F],
    grid_size: usize,
    x_range: Option<(F, F)>,
    y_range: Option<(F, F)>,
) -> LinalgResult<PseudospectrumResult<F>>
where
    F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static,
{
    let n = a.nrows();
    if n == 0 {
        return Err(LinalgError::InvalidInputError(
            "pseudospectrum: matrix must be non-empty".to_string(),
        ));
    }
    if a.ncols() != n {
        return Err(LinalgError::ShapeError(
            "pseudospectrum: matrix must be square".to_string(),
        ));
    }
    let gs = grid_size.max(2);

    // Determine the grid range from eigenvalues when not specified
    let (eig_re, eig_im) = {
        let (eigenvalues, _) = eig(a, None)?;
        let re: Vec<F> = eigenvalues.iter().map(|z| z.re).collect();
        let im: Vec<F> = eigenvalues.iter().map(|z| z.im).collect();
        (re, im)
    };

    let eig_re_min = eig_re
        .iter()
        .cloned()
        .fold(F::infinity(), |a, b| if b < a { b } else { a });
    let eig_re_max = eig_re
        .iter()
        .cloned()
        .fold(F::neg_infinity(), |a, b| if b > a { b } else { a });
    let eig_im_min = eig_im
        .iter()
        .cloned()
        .fold(F::infinity(), |a, b| if b < a { b } else { a });
    let eig_im_max = eig_im
        .iter()
        .cloned()
        .fold(F::neg_infinity(), |a, b| if b > a { b } else { a });

    let two = F::from(2.0_f64).unwrap_or(F::one() + F::one());
    let margin =
        ((eig_re_max - eig_re_min).abs() + (eig_im_max - eig_im_min).abs()) / two + F::one();

    let (x_min, x_max) = x_range.unwrap_or((eig_re_min - margin, eig_re_max + margin));
    let (y_min, y_max) = y_range.unwrap_or((eig_im_min - margin, eig_im_max + margin));

    let gs_f = F::from(gs - 1).unwrap_or(F::one());
    let x_step = (x_max - x_min) / gs_f;
    let y_step = (y_max - y_min) / gs_f;

    let x_grid: Vec<F> = (0..gs)
        .map(|i| x_min + x_step * F::from(i).unwrap_or(F::zero()))
        .collect();
    let y_grid: Vec<F> = (0..gs)
        .map(|i| y_min + y_step * F::from(i).unwrap_or(F::zero()))
        .collect();

    let mut sigma_min = Array2::<F>::zeros((gs, gs));

    for iy in 0..gs {
        for ix in 0..gs {
            let z_re = x_grid[ix];
            let z_im = y_grid[iy];
            let s = smallest_singular_value_shifted(a, z_re, z_im)?;
            sigma_min[[iy, ix]] = s;
        }
    }

    Ok(PseudospectrumResult {
        x_grid,
        y_grid,
        sigma_min,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array2};

    #[test]
    fn test_spectral_abscissa_diagonal() {
        let a = array![[-1.0_f64, 0.0], [0.0, -2.0]];
        let alpha = spectral_abscissa(&a.view()).expect("spectral_abscissa");
        assert!((alpha - (-1.0)).abs() < 1e-10, "expected -1, got {alpha}");
    }

    #[test]
    fn test_spectral_radius_rotation() {
        // Pure rotation: eigenvalues ±i → |λ| = 1
        let a = array![[0.0_f64, -1.0], [1.0, 0.0]];
        let rho = spectral_radius(&a.view()).expect("spectral_radius");
        assert!((rho - 1.0).abs() < 1e-10, "expected 1, got {rho}");
    }

    #[test]
    fn test_spectral_radius_diagonal() {
        let a = array![[3.0_f64, 0.0], [0.0, -5.0]];
        let rho = spectral_radius(&a.view()).expect("spectral_radius");
        assert!((rho - 5.0).abs() < 1e-10, "expected 5, got {rho}");
    }

    #[test]
    fn test_spectral_abscissa_stable() {
        // Hurwitz-stable: all eigenvalues in left half-plane
        let a = array![[-2.0_f64, 1.0], [0.0, -3.0]];
        let alpha = spectral_abscissa(&a.view()).expect("spectral_abscissa");
        assert!(alpha < 0.0, "stable matrix: spectral_abscissa should be <0");
    }

    #[test]
    fn test_field_of_values_symmetric() {
        // Symmetric matrix A = [[1,0],[0,3]]; W(A) is the segment [1,3].
        // The boundary tracing at angle θ yields z(θ) = λ_max(H(θ)) * e^{iθ}
        // where H(θ) = cos(θ)*H for symmetric A.
        // The extreme points on the real axis occur at θ=0 and θ=π: z=3 and z=-1
        // (or at θ close to 0 for the eigenvalue range).
        let a = array![[1.0_f64, 0.0], [0.0, 3.0]];
        let (re, im) = field_of_values(&a.view(), 64).expect("fov");
        // All boundary points satisfy |z| ≤ λ_max = 3
        for (&rev, &imv) in re.iter().zip(im.iter()) {
            let mag = (rev * rev + imv * imv).sqrt();
            assert!(mag <= 3.1, "boundary point magnitude out of range: {mag}");
        }
        // Numerical radius ≈ 3 (maximum |z| over boundary)
        let max_mag = re
            .iter()
            .zip(im.iter())
            .map(|(&r, &i)| (r * r + i * i).sqrt())
            .fold(0.0_f64, f64::max);
        assert!(
            max_mag >= 2.9,
            "max boundary magnitude should be ≈3, got {max_mag}"
        );
    }

    #[test]
    fn test_numerical_radius_symmetric_positive() {
        let a = array![[2.0_f64, 0.0], [0.0, 4.0]];
        let w = numerical_radius(&a.view()).expect("numerical_radius");
        // For symmetric PD matrix: w(A) = λ_max = 4
        assert!((w - 4.0).abs() < 0.2, "expected ≈4, got {w}");
    }

    #[test]
    fn test_kreiss_number_nonnegative() {
        let a = array![[0.0_f64, 1.0], [-1.0, 0.0]];
        let k = kreiss_number(&a.view(), 8).expect("kreiss_number");
        assert!(k >= 0.0, "kreiss_number should be >=0, got {k}");
    }

    #[test]
    fn test_pseudospectrum_shape() {
        let a = array![[1.0_f64, 0.0], [0.0, 2.0]];
        let result = pseudospectrum(&a.view(), &[0.1_f64], 6, None, None).expect("pseudospectrum");
        assert_eq!(result.sigma_min.shape(), &[6, 6]);
        assert_eq!(result.x_grid.len(), 6);
        assert_eq!(result.y_grid.len(), 6);
    }

    #[test]
    fn test_pseudospectrum_on_eigenvalue() {
        // At an eigenvalue z = λ, σ_min(zI - A) = 0
        // Use a diagonal matrix A = diag(1,2), check near z=1
        let a = array![[1.0_f64, 0.0], [0.0, 2.0]];
        let result = pseudospectrum(
            &a.view(),
            &[0.01_f64],
            4,
            Some((0.5, 1.5)),
            Some((-0.5, 0.5)),
        )
        .expect("pseudospectrum");
        // The grid includes points near z=1+0i; σ_min should be small there
        let min_sv = result
            .sigma_min
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        assert!(
            min_sv < 0.6,
            "σ_min near eigenvalue should be small, got {min_sv}"
        );
    }

    #[test]
    fn test_field_of_values_empty_error() {
        let a = Array2::<f64>::zeros((0, 0));
        assert!(field_of_values(&a.view(), 64).is_err());
    }

    #[test]
    fn test_field_of_values_nonsquare_error() {
        let a = Array2::<f64>::zeros((2, 3));
        assert!(field_of_values(&a.view(), 64).is_err());
    }
}
