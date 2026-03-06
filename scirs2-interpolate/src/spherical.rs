//! Interpolation on spherical surfaces and quaternion curves
//!
//! This module provides:
//!
//! 1. **[`SphericalHarmonicsInterpolator`]** — least-squares fit of real spherical
//!    harmonics to scattered data on the unit sphere, then evaluation at arbitrary
//!    (θ, φ) coordinates.
//!
//! 2. **[`slerp`]** — Spherical Linear intERPolation of unit quaternions.
//!
//! 3. **[`squad`]** — Spherical QUADrangle interpolation for smooth quaternion
//!    splines through four consecutive control points.
//!
//! # Coordinate conventions
//!
//! Unless otherwise noted, spherical coordinates follow the **physics convention**:
//!
//! - θ ∈ [0, π]  is the **colatitude** (polar angle from the north pole).
//! - φ ∈ [0, 2π) is the **longitude** (azimuthal angle from the x-axis).
//!
//! The Cartesian embedding is:
//!
//! ```text
//! x = sin(θ)·cos(φ),   y = sin(θ)·sin(φ),   z = cos(θ)
//! ```
//!
//! # Real Spherical Harmonics
//!
//! The interpolator uses the **real** (tesseral) spherical harmonic basis.
//! For degree `l` and order `m` ∈ {-l, …, l} the real harmonic is
//!
//! ```text
//! Y_l^m(θ,φ) = { √2 · K_l^|m| · cos(|m|φ) · P_l^|m|(cos θ)   if m > 0
//!              { K_l^0 · P_l^0(cos θ)                           if m = 0
//!              { √2 · K_l^|m| · sin(|m|φ) · P_l^|m|(cos θ)     if m < 0
//! ```
//!
//! where `K_l^m = sqrt((2l+1)/(4π) · (l-m)!/(l+m)!)` is the normalisation
//! constant and `P_l^m` is the associated Legendre polynomial.
//!
//! # References
//!
//! - Shoemake, K. (1985). "Animating rotation with quaternion curves."
//!   *SIGGRAPH '85*, 245–254.
//! - Blanco, M. A. et al. (1997). "Evaluation of the rotation matrices in the
//!   basis of real and complex spherical harmonics." *J. Mol. Struct. (Theochem)*
//!   **419**, 19–27.

use crate::error::{InterpolateError, InterpolateResult};
use std::f64::consts::{PI, SQRT_2};

// ---------------------------------------------------------------------------
// Associated Legendre polynomials (fully normalised, real convention)
// ---------------------------------------------------------------------------

/// Compute the unnormalised associated Legendre polynomial `P_l^m(x)` for
/// non-negative integer `l` and `m ∈ [0, l]`.
///
/// Uses the three-term recurrence relation:
/// ```text
/// (l-m) P_l^m = (2l-1) x P_{l-1}^m − (l+m-1) P_{l-2}^m
/// ```
/// with seeds
/// ```text
/// P_m^m   = (-1)^m (2m-1)!! (1-x²)^{m/2}
/// P_{m+1}^m = (2m+1) x P_m^m
/// ```
fn assoc_legendre(l: usize, m: usize, x: f64) -> f64 {
    debug_assert!(m <= l);
    debug_assert!(x.abs() <= 1.0 + 1e-10);

    // Seed: P_m^m
    let sin_theta = (1.0 - x.min(1.0).max(-1.0).powi(2)).sqrt();
    let mut pmm = 1.0f64;
    let mut factor = 1.0f64;
    for _ in 0..m {
        pmm *= -factor * sin_theta;
        factor += 2.0;
    }

    if l == m {
        return pmm;
    }

    // P_{m+1}^m
    let mut pm1 = (2 * m + 1) as f64 * x * pmm;
    if l == m + 1 {
        return pm1;
    }

    // Three-term recurrence
    let mut p_prev2 = pmm;
    let mut p_prev1 = pm1;
    let mut p_cur = 0.0;
    for ll in (m + 2)..=l {
        let lf = ll as f64;
        let mf = m as f64;
        p_cur = ((2.0 * lf - 1.0) * x * p_prev1 - (lf + mf - 1.0) * p_prev2) / (lf - mf);
        p_prev2 = p_prev1;
        p_prev1 = p_cur;
    }
    p_cur
}

/// Normalisation constant K_l^m for real spherical harmonics.
///
/// `K = sqrt((2l+1)/(4π) * (l-m)!/(l+m)!)`
#[allow(dead_code)]
fn normk(l: usize, m: usize) -> f64 {
    // Compute (l-m)!/(l+m)! incrementally to avoid huge factorials
    let mut ratio = 1.0f64;
    // ratio = (l-m)! / (l+m)!
    // = product of 1/k for k = (l-m+1) to (l+m)
    for k in (l - m + 1)..=(l + m) {
        ratio /= k as f64;
    }
    ((2 * l + 1) as f64 / (4.0 * PI) * ratio).sqrt()
}

/// Compute the fully-normalised Legendre value `K_l^m * P_l^m(x)` with the
/// Condon-Shortley phase `(-1)^m`, avoiding overflow for large `l` and `m`.
///
/// For large `l = m` the unnormalised `P_m^m ≈ (2m-1)!!` overflows f64 while
/// the normalisation constant `K_m^m ≈ 1/(2m)!` underflows.  This routine
/// folds both factors into a single recurrence so that neither intermediate
/// quantity grows beyond O(sqrt(l)).
fn normalized_legendre_cs(l: usize, m: usize, x: f64) -> f64 {
    if m > l {
        return 0.0;
    }

    // Clamp x to [-1, 1] to guard against floating-point rounding errors.
    let x = x.clamp(-1.0, 1.0);
    let sin_theta = ((1.0 - x) * (1.0 + x)).sqrt(); // = |sin θ|

    // Compute the normalised seed K_m^m * P_m^m with Condon-Shortley phase.
    //
    //   K_m^m * P_m^m = sqrt((2m+1)/(4π)) * prod_{k=1}^{m} (-sqrt((2k-1)/(2k))) * sin θ
    //
    // Each factor |−sqrt((2k-1)/(2k)) * sin θ| ≤ 1, so the running product
    // never overflows regardless of m.
    let mut bar_pmm = 1.0_f64 / (2.0 * PI.sqrt()); // = sqrt(1/(4π))
    for k in 1..=m {
        let kf = k as f64;
        // The leading (-1) per step contributes the overall (-1)^m CS phase.
        bar_pmm *= -((2.0 * kf - 1.0) / (2.0 * kf)).sqrt() * sin_theta;
    }
    bar_pmm *= ((2 * m + 1) as f64).sqrt();

    if l == m {
        return bar_pmm;
    }

    // bar_P_{m+1}^m = sqrt(2m+3) * x * bar_P_m^m
    let mut bar_pmmp1 = ((2 * m + 3) as f64).sqrt() * x * bar_pmm;

    if l == m + 1 {
        return bar_pmmp1;
    }

    // Three-term normalised recurrence for l >= m+2:
    //   bar_P_l^m = alpha * x * bar_P_{l-1}^m - beta * bar_P_{l-2}^m
    // with:
    //   alpha = sqrt((4l²-1) / (l²-m²))
    //   beta  = sqrt((2l+1)(l+m-1)(l-m-1) / ((2l-3)(l²-m²)))
    let mut result = 0.0;
    let mut prev2 = bar_pmm;
    let mut prev1 = bar_pmmp1;
    for ll in (m + 2)..=l {
        let lf = ll as f64;
        let mf = m as f64;
        let l2m2 = lf * lf - mf * mf;
        let alpha = ((4.0 * lf * lf - 1.0) / l2m2).sqrt();
        let beta = (((2.0 * lf + 1.0) * (lf + mf - 1.0) * (lf - mf - 1.0))
            / ((2.0 * lf - 3.0) * l2m2))
            .sqrt();
        result = alpha * x * prev1 - beta * prev2;
        prev2 = prev1;
        prev1 = result;
    }
    result
}

/// Evaluate a single real spherical harmonic Y_l^m(θ, φ).
///
/// `m` is signed: negative values use the sine component.
pub fn real_sph_harm(l: usize, m: i64, theta: f64, phi: f64) -> f64 {
    let abs_m = m.unsigned_abs() as usize;
    debug_assert!(abs_m <= l);
    let cos_theta = theta.cos().clamp(-1.0, 1.0);
    // Use the normalized computation to avoid overflow/underflow for large l and m.
    let bar_p = normalized_legendre_cs(l, abs_m, cos_theta);
    if m == 0 {
        bar_p
    } else if m > 0 {
        SQRT_2 * bar_p * (abs_m as f64 * phi).cos()
    } else {
        SQRT_2 * bar_p * (abs_m as f64 * phi).sin()
    }
}

/// Total number of real spherical harmonics up to and including degree `l_max`.
///
/// `(l_max + 1)^2`
pub fn n_sph_harm_basis(l_max: usize) -> usize {
    (l_max + 1) * (l_max + 1)
}

/// Evaluate all real spherical harmonics up to `l_max` at (θ, φ).
///
/// Returns a vector of length `(l_max+1)²`.
/// Ordering: `(l=0,m=0), (l=1,m=-1), (l=1,m=0), (l=1,m=1), (l=2,m=-2), …`
pub fn eval_sph_basis(l_max: usize, theta: f64, phi: f64) -> Vec<f64> {
    let nb = n_sph_harm_basis(l_max);
    let mut basis = Vec::with_capacity(nb);
    for l in 0..=l_max {
        let li = l as i64;
        for m in -li..=li {
            basis.push(real_sph_harm(l, m, theta, phi));
        }
    }
    basis
}

// ---------------------------------------------------------------------------
// Least-squares solve (SVD via Golub-Reinsch, or QR via Householder)
// ---------------------------------------------------------------------------

/// Solve the normal equations Aᵀ A x = Aᵀ b using a stabilised QR decomposition
/// (Householder reflections + back-substitution).  Rank deficiency is handled by
/// adding a small Tikhonov regularisation.
///
/// Returns the least-squares solution `x`.
fn lstsq(a: &[Vec<f64>], b: &[f64], reg: f64) -> Vec<f64> {
    let m = b.len();
    let n = if m > 0 { a[0].len() } else { 0 };

    // Build Aᵀ A and Aᵀ b
    let mut ata = vec![vec![0.0f64; n]; n];
    let mut atb = vec![0.0f64; n];
    for (i, row) in a.iter().enumerate() {
        for j in 0..n {
            atb[j] += row[j] * b[i];
            for k in 0..n {
                ata[j][k] += row[j] * row[k];
            }
        }
    }
    // Tikhonov regularisation
    for j in 0..n {
        ata[j][j] += reg;
    }

    // Cholesky of Aᵀ A + λI
    let mut l = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = ata[i][j];
            for k in 0..j {
                s -= l[i][k] * l[j][k];
            }
            if i == j {
                l[i][j] = if s > 0.0 { s.sqrt() } else { 0.0 };
            } else if l[j][j].abs() > 0.0 {
                l[i][j] = s / l[j][j];
            }
        }
    }

    // Forward sub: L y = Aᵀ b
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let mut s = atb[i];
        for k in 0..i {
            s -= l[i][k] * y[k];
        }
        if l[i][i].abs() > 1e-30 {
            y[i] = s / l[i][i];
        }
    }

    // Backward sub: Lᵀ x = y
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in i + 1..n {
            s -= l[k][i] * x[k];
        }
        if l[i][i].abs() > 1e-30 {
            x[i] = s / l[i][i];
        }
    }
    x
}

// ---------------------------------------------------------------------------
// SphericalHarmonicsInterpolator
// ---------------------------------------------------------------------------

/// Spherical harmonics interpolator for scattered data on the unit sphere.
///
/// Fits a truncated real spherical harmonics expansion to scattered data on the
/// sphere by solving a least-squares system.  The expansion has `(l_max+1)²`
/// terms and interpolates / approximates `n` data values at scattered
/// (θ, φ) coordinates.
///
/// # Examples
///
/// ```rust
/// use scirs2_interpolate::spherical::SphericalHarmonicsInterpolator;
/// use std::f64::consts::PI;
///
/// // Sample data: f(θ,φ) = cos(θ) (= Y_1^0 up to a scale)
/// let n = 50;
/// let coords: Vec<[f64; 2]> = (0..n)
///     .map(|i| {
///         let theta = PI * (i as f64 + 0.5) / n as f64;
///         let phi = 2.0 * PI * (i as f64) / n as f64;
///         [theta, phi]
///     })
///     .collect();
/// let values: Vec<f64> = coords.iter().map(|[t, _]| t.cos()).collect();
///
/// let interp = SphericalHarmonicsInterpolator::fit(&coords, &values, 3)
///     .expect("fit failed");
///
/// // Evaluate at the north pole
/// let v = interp.predict(0.0, 0.0);
/// assert!((v - 1.0).abs() < 0.1);
/// ```
#[derive(Clone, Debug)]
pub struct SphericalHarmonicsInterpolator {
    l_max: usize,
    coefficients: Vec<f64>,
}

impl SphericalHarmonicsInterpolator {
    /// Fit a spherical harmonics interpolator.
    ///
    /// # Arguments
    ///
    /// * `coords` — `n` sample locations as `[θ, φ]` in radians.
    /// * `values` — scalar data at each sample location.
    /// * `l_max`  — maximum spherical harmonic degree.  The basis has
    ///              `(l_max+1)²` terms; more data points than basis functions
    ///              is strongly recommended.
    ///
    /// # Errors
    ///
    /// Returns an error if `coords` and `values` have different lengths or if
    /// there are not enough sample points to determine the coefficients.
    pub fn fit(
        coords: &[[f64; 2]],
        values: &[f64],
        l_max: usize,
    ) -> InterpolateResult<Self> {
        let n = coords.len();
        if n != values.len() {
            return Err(InterpolateError::invalid_input(
                "SphericalHarmonics: coords and values must have the same length".to_string(),
            ));
        }
        let nb = n_sph_harm_basis(l_max);
        if n < nb {
            return Err(InterpolateError::insufficient_points(
                nb,
                n,
                "SphericalHarmonicsInterpolator",
            ));
        }

        // Build design matrix A: n × nb
        let mut a = Vec::with_capacity(n);
        for [theta, phi] in coords.iter() {
            a.push(eval_sph_basis(l_max, *theta, *phi));
        }

        // Regularisation: small relative to the data scale
        let data_scale = values.iter().map(|v| v.abs()).fold(0.0f64, f64::max).max(1.0);
        let reg = 1e-12 * data_scale * data_scale;

        let coefficients = lstsq(&a, values, reg);
        Ok(Self { l_max, coefficients })
    }

    /// Evaluate the spherical harmonics expansion at (θ, φ).
    pub fn predict(&self, theta: f64, phi: f64) -> f64 {
        let basis = eval_sph_basis(self.l_max, theta, phi);
        basis.iter().zip(self.coefficients.iter()).map(|(b, c)| b * c).sum()
    }

    /// Evaluate at a batch of (θ, φ) coordinates.
    pub fn predict_batch(&self, coords: &[[f64; 2]]) -> Vec<f64> {
        coords.iter().map(|[t, p]| self.predict(*t, *p)).collect()
    }

    /// Power spectrum: total variance at each degree `l`.
    ///
    /// `P_l = Σ_{m=-l}^{l} c_{lm}²`
    pub fn power_spectrum(&self) -> Vec<f64> {
        let mut ps = vec![0.0f64; self.l_max + 1];
        let mut idx = 0usize;
        for l in 0..=self.l_max {
            for _m in -(l as i64)..=(l as i64) {
                ps[l] += self.coefficients[idx] * self.coefficients[idx];
                idx += 1;
            }
        }
        ps
    }

    /// Maximum spherical harmonic degree.
    pub fn l_max(&self) -> usize {
        self.l_max
    }

    /// Number of basis functions: `(l_max + 1)²`.
    pub fn n_basis(&self) -> usize {
        n_sph_harm_basis(self.l_max)
    }

    /// Fitted coefficients (length `n_basis()`).
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }
}

// ---------------------------------------------------------------------------
// Quaternion helpers
// ---------------------------------------------------------------------------

/// Dot product of two unit quaternions (treated as 4-vectors).
#[inline]
fn quat_dot(q: [f64; 4], r: [f64; 4]) -> f64 {
    q[0] * r[0] + q[1] * r[1] + q[2] * r[2] + q[3] * r[3]
}

/// Multiply two unit quaternions (Hamilton product).
///
/// Quaternion layout: `[w, x, y, z]`.
#[inline]
fn quat_mul(q: [f64; 4], r: [f64; 4]) -> [f64; 4] {
    let [qw, qx, qy, qz] = q;
    let [rw, rx, ry, rz] = r;
    [
        qw * rw - qx * rx - qy * ry - qz * rz,
        qw * rx + qx * rw + qy * rz - qz * ry,
        qw * ry - qx * rz + qy * rw + qz * rx,
        qw * rz + qx * ry - qy * rx + qz * rw,
    ]
}

/// Conjugate of a unit quaternion `[w, x, y, z]` → `[w, -x, -y, -z]`.
#[inline]
fn quat_conj(q: [f64; 4]) -> [f64; 4] {
    [q[0], -q[1], -q[2], -q[3]]
}

/// Normalise a quaternion to unit length.
///
/// Returns `[1, 0, 0, 0]` if the norm is essentially zero.
#[inline]
fn quat_norm(q: [f64; 4]) -> [f64; 4] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if n < 1e-30 {
        [1.0, 0.0, 0.0, 0.0]
    } else {
        [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
    }
}

/// Quaternion natural logarithm `log(q)` for a unit quaternion.
///
/// For a unit quaternion `q = cos(θ)·1 + sin(θ)·n̂`, `log(q) = θ·n̂`.
/// The result is a pure quaternion `[0, x, y, z]`.
fn quat_log(q: [f64; 4]) -> [f64; 4] {
    // Clamp w to [-1, 1] for numerical safety
    let w = q[0].clamp(-1.0, 1.0);
    let sin_theta = (q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if sin_theta < 1e-14 {
        return [0.0, 0.0, 0.0, 0.0];
    }
    let theta = w.acos();
    let scale = theta / sin_theta;
    [0.0, scale * q[1], scale * q[2], scale * q[3]]
}

/// Quaternion exponential `exp(v)` for a pure quaternion `v = [0, x, y, z]`.
///
/// `exp(v) = cos(|v|)·1 + sin(|v|)·v/|v|`
fn quat_exp(v: [f64; 4]) -> [f64; 4] {
    let theta = (v[1] * v[1] + v[2] * v[2] + v[3] * v[3]).sqrt();
    if theta < 1e-14 {
        return [1.0, 0.0, 0.0, 0.0];
    }
    let sinc = theta.sin() / theta;
    [theta.cos(), sinc * v[1], sinc * v[2], sinc * v[3]]
}

/// Scale a pure quaternion by a scalar.
#[inline]
fn quat_scale(q: [f64; 4], s: f64) -> [f64; 4] {
    [q[0] * s, q[1] * s, q[2] * s, q[3] * s]
}

/// Add two quaternions component-wise.
#[inline]
fn quat_add(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
}

// ---------------------------------------------------------------------------
// slerp
// ---------------------------------------------------------------------------

/// Spherical Linear Interpolation (SLERP) between two unit quaternions.
///
/// Computes the shortest-arc constant-angular-velocity interpolation from `q1`
/// to `q2` at parameter `t ∈ [0, 1]`.
///
/// If `t = 0` the result is `q1`; if `t = 1` the result is `q2`.
///
/// # Quaternion convention
///
/// Quaternions are represented as `[w, x, y, z]`.  Both inputs should be unit
/// quaternions; non-unit inputs are normalised internally.
///
/// # Examples
///
/// ```rust
/// use scirs2_interpolate::spherical::slerp;
///
/// // Interpolate halfway between identity and 180° rotation about z
/// let q1 = [1.0_f64, 0.0, 0.0, 0.0]; // identity
/// let q2 = [0.0_f64, 0.0, 0.0, 1.0]; // 180° about z
/// let mid = slerp(q1, q2, 0.5);
/// // Should be 90° rotation about z: [cos45°, 0, 0, sin45°]
/// let expected = std::f64::consts::FRAC_PI_4.cos();
/// assert!((mid[0] - expected).abs() < 1e-10);
/// assert!((mid[3] - expected).abs() < 1e-10);
/// ```
pub fn slerp(q1: [f64; 4], q2: [f64; 4], t: f64) -> [f64; 4] {
    let q1 = quat_norm(q1);
    let mut q2 = quat_norm(q2);

    // Ensure shortest arc: flip q2 if the dot product is negative
    let mut dot = quat_dot(q1, q2);
    if dot < 0.0 {
        q2 = [-q2[0], -q2[1], -q2[2], -q2[3]];
        dot = -dot;
    }

    // If very close, fall back to linear interpolation to avoid division by zero
    if dot > 1.0 - 1e-10 {
        let q = [
            q1[0] + t * (q2[0] - q1[0]),
            q1[1] + t * (q2[1] - q1[1]),
            q1[2] + t * (q2[2] - q1[2]),
            q1[3] + t * (q2[3] - q1[3]),
        ];
        return quat_norm(q);
    }

    let theta = dot.clamp(-1.0, 1.0).acos();
    let sin_theta = theta.sin();
    let s1 = ((1.0 - t) * theta).sin() / sin_theta;
    let s2 = (t * theta).sin() / sin_theta;
    quat_norm([
        s1 * q1[0] + s2 * q2[0],
        s1 * q1[1] + s2 * q2[1],
        s1 * q1[2] + s2 * q2[2],
        s1 * q1[3] + s2 * q2[3],
    ])
}

// ---------------------------------------------------------------------------
// squad
// ---------------------------------------------------------------------------

/// Compute the SQUAD intermediate control point `s_i` for knot `q_i`.
///
/// The formula is:
/// ```text
/// s_i = q_i · exp( -(log(q_i⁻¹ q_{i+1}) + log(q_i⁻¹ q_{i-1})) / 4 )
/// ```
///
/// All inputs must be unit quaternions.
fn squad_inner(q_prev: [f64; 4], q_i: [f64; 4], q_next: [f64; 4]) -> [f64; 4] {
    let qi_inv = quat_conj(q_i);
    let log1 = quat_log(quat_mul(qi_inv, q_next));
    let log2 = quat_log(quat_mul(qi_inv, q_prev));
    let sum = quat_add(log1, log2);
    let term = quat_scale(sum, -0.25);
    quat_mul(q_i, quat_exp(term))
}

/// Spherical Quadrangle (SQUAD) interpolation for smooth quaternion curves.
///
/// SQUAD computes a smooth C¹ quaternion spline through a sequence of knots.
/// Given four consecutive control quaternions `q0, q1, q2, q3`, this function
/// interpolates **between `q1` and `q2`** at parameter `t ∈ [0, 1]`.
///
/// The cubic SQUAD formula is:
/// ```text
/// squad(q1, q2, s1, s2, t) = slerp(slerp(q1, q2, t), slerp(s1, s2, t), 2t(1-t))
/// ```
/// where `s1 = inner(q0, q1, q2)` and `s2 = inner(q1, q2, q3)`.
///
/// # Arguments
///
/// * `q0`, `q1`, `q2`, `q3` — four consecutive unit quaternions.
/// * `t` — interpolation parameter in `[0, 1]` (0 → `q1`, 1 → `q2`).
///
/// # Examples
///
/// ```rust
/// use scirs2_interpolate::spherical::squad;
///
/// let q0 = [1.0_f64, 0.0, 0.0, 0.0];
/// let q1 = [1.0_f64, 0.0, 0.0, 0.0];
/// let q2 = [0.0_f64, 0.0, 0.0, 1.0];
/// let q3 = [0.0_f64, 0.0, 0.0, 1.0];
///
/// // At t=0 → q1; at t=1 → q2
/// let start = squad(q0, q1, q2, q3, 0.0);
/// assert!((start[0] - 1.0).abs() < 1e-10);
/// let end = squad(q0, q1, q2, q3, 1.0);
/// assert!((end[3] - 1.0).abs() < 1e-10);
/// ```
pub fn squad(
    q0: [f64; 4],
    q1: [f64; 4],
    q2: [f64; 4],
    q3: [f64; 4],
    t: f64,
) -> [f64; 4] {
    let q0n = quat_norm(q0);
    let q1n = quat_norm(q1);
    let q2n = quat_norm(q2);
    let q3n = quat_norm(q3);

    // Compute inner control points
    let s1 = squad_inner(q0n, q1n, q2n);
    let s2 = squad_inner(q1n, q2n, q3n);

    // SQUAD: slerp(slerp(q1,q2,t), slerp(s1,s2,t), 2t(1-t))
    let ab = slerp(q1n, q2n, t);
    let cd = slerp(s1, s2, t);
    slerp(ab, cd, 2.0 * t * (1.0 - t))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // --- Associated Legendre polynomial basic tests ---

    #[test]
    fn test_legendre_p00() {
        // P_0^0(x) = 1 for all x
        for x in [-1.0, -0.5, 0.0, 0.5, 1.0] {
            assert!((assoc_legendre(0, 0, x) - 1.0).abs() < 1e-12, "P00({x})");
        }
    }

    #[test]
    fn test_legendre_p10() {
        // P_1^0(x) = x
        for x in [-0.8, 0.0, 0.5, 1.0] {
            assert!((assoc_legendre(1, 0, x) - x).abs() < 1e-10, "P10({x})");
        }
    }

    #[test]
    fn test_legendre_p20() {
        // P_2^0(x) = (3x²-1)/2
        for x in [-1.0, -0.5, 0.0, 0.5, 1.0] {
            let expected = (3.0 * x * x - 1.0) / 2.0;
            let got = assoc_legendre(2, 0, x);
            assert!((got - expected).abs() < 1e-10, "P20({x}) got={got} expected={expected}");
        }
    }

    // --- Spherical harmonics orthonormality (spot checks) ---

    #[test]
    fn test_sph_harm_y00_normalization() {
        // Y_0^0 = sqrt(1/(4π)) everywhere
        let expected = (1.0 / (4.0 * PI)).sqrt();
        let v = real_sph_harm(0, 0, 1.0, 2.0);
        assert!((v - expected).abs() < 1e-10, "Y00={v} expected={expected}");
    }

    #[test]
    fn test_sph_harm_y10_north_pole() {
        // At north pole (θ=0): Y_1^0 = sqrt(3/(4π))
        let expected = (3.0 / (4.0 * PI)).sqrt();
        let v = real_sph_harm(1, 0, 0.0, 0.0);
        assert!((v - expected).abs() < 1e-8, "Y10(pole)={v} expected={expected}");
    }

    #[test]
    fn test_sph_harm_nonzero() {
        // Just verify evaluation doesn't panic/NaN
        for l in 0..=4usize {
            for m in -(l as i64)..=(l as i64) {
                let v = real_sph_harm(l, m, 1.2, 0.7);
                assert!(v.is_finite(), "Y_{l}^{m} is not finite");
            }
        }
    }

    // --- SphericalHarmonicsInterpolator ---

    #[test]
    fn test_sph_interp_constant() {
        // f = 1 everywhere → only l=0 coefficient matters
        let n = 25;
        let coords: Vec<[f64; 2]> = (0..n)
            .map(|i| {
                let t = PI * (i as f64 + 0.5) / n as f64;
                let p = 2.0 * PI * (i as f64) / n as f64;
                [t, p]
            })
            .collect();
        let values = vec![1.0f64; n];
        let interp = SphericalHarmonicsInterpolator::fit(&coords, &values, 2).expect("test: should succeed");
        // Prediction at arbitrary point should be close to 1
        let pred = interp.predict(PI * 0.3, 1.0);
        assert!((pred - 1.0).abs() < 0.1, "pred={pred}");
    }

    #[test]
    fn test_sph_interp_cos_theta() {
        // f(θ,φ) = cos(θ) — should be captured by l=1 harmonics
        let n = 36;
        let coords: Vec<[f64; 2]> = (0..n)
            .map(|i| {
                let t = PI * (i as f64 + 0.5) / n as f64;
                let p = 2.0 * PI * (i as f64) / n as f64;
                [t, p]
            })
            .collect();
        let values: Vec<f64> = coords.iter().map(|[t, _]| t.cos()).collect();
        let interp = SphericalHarmonicsInterpolator::fit(&coords, &values, 3).expect("test: should succeed");

        // Test at north pole: cos(0) = 1
        let pred_np = interp.predict(0.01, 0.0);
        assert!((pred_np - 1.0).abs() < 0.15, "north pole pred={pred_np}");

        // Test at equator: cos(π/2) = 0
        let pred_eq = interp.predict(PI * 0.5, 1.0);
        assert!(pred_eq.abs() < 0.2, "equator pred={pred_eq}");
    }

    #[test]
    fn test_sph_interp_power_spectrum_length() {
        let n = 25;
        let coords: Vec<[f64; 2]> = (0..n)
            .map(|i| [PI * i as f64 / n as f64, 2.0 * PI * i as f64 / n as f64])
            .collect();
        let values = vec![1.0f64; n];
        let interp = SphericalHarmonicsInterpolator::fit(&coords, &values, 2).expect("test: should succeed");
        let ps = interp.power_spectrum();
        assert_eq!(ps.len(), 3); // l=0,1,2
        for p in &ps {
            assert!(p.is_finite() && *p >= 0.0);
        }
    }

    #[test]
    fn test_sph_interp_n_basis() {
        let n = 25;
        let coords: Vec<[f64; 2]> = (0..n)
            .map(|i| [PI * i as f64 / n as f64, 2.0 * PI * i as f64 / n as f64])
            .collect();
        let values = vec![1.0f64; n];
        let interp = SphericalHarmonicsInterpolator::fit(&coords, &values, 2).expect("test: should succeed");
        assert_eq!(interp.n_basis(), 9); // (2+1)^2 = 9
        assert_eq!(interp.l_max(), 2);
    }

    #[test]
    fn test_sph_interp_batch() {
        let n = 25;
        let coords: Vec<[f64; 2]> = (0..n)
            .map(|i| [PI * i as f64 / n as f64, 2.0 * PI * i as f64 / n as f64])
            .collect();
        let values = vec![1.0f64; n];
        let interp = SphericalHarmonicsInterpolator::fit(&coords, &values, 2).expect("test: should succeed");
        let queries = vec![[0.5, 0.5], [1.0, 1.0], [1.5, 2.0]];
        let preds = interp.predict_batch(&queries);
        assert_eq!(preds.len(), 3);
        for p in &preds {
            assert!(p.is_finite());
        }
    }

    #[test]
    fn test_sph_interp_insufficient_points() {
        // 4 points < (l_max=2+1)^2 = 9
        let coords: Vec<[f64; 2]> = (0..4).map(|i| [i as f64, 0.0]).collect();
        let values = vec![1.0f64; 4];
        assert!(SphericalHarmonicsInterpolator::fit(&coords, &values, 2).is_err());
    }

    // --- slerp tests ---

    #[test]
    fn test_slerp_endpoints() {
        let q1 = [1.0f64, 0.0, 0.0, 0.0];
        let q2 = [0.0f64, 1.0, 0.0, 0.0];
        let s0 = slerp(q1, q2, 0.0);
        let s1 = slerp(q1, q2, 1.0);
        for i in 0..4 {
            assert!((s0[i] - q1[i]).abs() < 1e-10, "slerp(0)[{i}]");
            assert!((s1[i] - q2[i]).abs() < 1e-10, "slerp(1)[{i}]");
        }
    }

    #[test]
    fn test_slerp_unit_norm() {
        let q1 = [1.0f64, 0.0, 0.0, 0.0];
        let q2 = [0.0f64, 0.0, 0.0, 1.0];
        for i in 0..=10 {
            let t = i as f64 / 10.0;
            let q = slerp(q1, q2, t);
            let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
            assert!((norm - 1.0).abs() < 1e-10, "norm={norm} at t={t}");
        }
    }

    #[test]
    fn test_slerp_halfway() {
        // 180° rotation about z: q2 = [0,0,0,1]
        // halfway should be 90°: [cos45°, 0, 0, sin45°]
        let q1 = [1.0f64, 0.0, 0.0, 0.0];
        let q2 = [0.0f64, 0.0, 0.0, 1.0];
        let mid = slerp(q1, q2, 0.5);
        let expected = (PI / 4.0).cos();
        assert!((mid[0] - expected).abs() < 1e-10, "mid[0]={}", mid[0]);
        assert!((mid[3] - expected).abs() < 1e-10, "mid[3]={}", mid[3]);
    }

    #[test]
    fn test_slerp_shortest_path() {
        // q2 and -q2 represent the same rotation; slerp should take the short path
        let q1 = [1.0f64, 0.0, 0.0, 0.0];
        let q2 = [0.0f64, 0.0, 0.0, 1.0];
        let q2_neg = [-q2[0], -q2[1], -q2[2], -q2[3]];
        let s_pos = slerp(q1, q2, 0.5);
        let s_neg = slerp(q1, q2_neg, 0.5);
        // Both should represent the same 90° rotation
        for i in 0..4 {
            assert!(
                (s_pos[i] - s_neg[i]).abs() < 1e-10
                    || (s_pos[i] + s_neg[i]).abs() < 1e-10,
                "slerp shortest path component {i}: pos={} neg={}",
                s_pos[i],
                s_neg[i]
            );
        }
    }

    // --- squad tests ---

    #[test]
    fn test_squad_endpoints() {
        let q0 = [1.0f64, 0.0, 0.0, 0.0];
        let q1 = [1.0f64, 0.0, 0.0, 0.0];
        let q2 = [0.0f64, 0.0, 0.0, 1.0];
        let q3 = [0.0f64, 0.0, 0.0, 1.0];

        let start = squad(q0, q1, q2, q3, 0.0);
        let end = squad(q0, q1, q2, q3, 1.0);

        // t=0 → q1 (identity)
        assert!((start[0] - 1.0).abs() < 1e-8, "squad start[0]={}", start[0]);
        // t=1 → q2 (180° about z)
        assert!((end[3] - 1.0).abs() < 1e-8, "squad end[3]={}", end[3]);
    }

    #[test]
    fn test_squad_unit_norm() {
        let q0 = [1.0f64, 0.0, 0.0, 0.0];
        let q1 = [0.7071f64, 0.7071, 0.0, 0.0];
        let q2 = [0.0f64, 1.0, 0.0, 0.0];
        let q3 = [-0.7071f64, 0.7071, 0.0, 0.0];
        for i in 0..=20 {
            let t = i as f64 / 20.0;
            let q = squad(q0, q1, q2, q3, t);
            let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
            assert!((norm - 1.0).abs() < 1e-8, "squad norm={norm} at t={t}");
        }
    }

    #[test]
    fn test_squad_smooth_midpoint() {
        // squad should produce a value distinct from both endpoints at t=0.5
        let q0 = [1.0f64, 0.0, 0.0, 0.0];
        let q1 = [1.0f64, 0.0, 0.0, 0.0];
        let q2 = [0.0f64, 0.0, 0.0, 1.0];
        let q3 = [0.0f64, 0.0, 0.0, 1.0];
        let mid = squad(q0, q1, q2, q3, 0.5);
        // Should not be identical to q1 or q2
        let same_q1 = mid.iter().zip(q1.iter()).all(|(a, b)| (a - b).abs() < 1e-8);
        let same_q2 = mid.iter().zip(q2.iter()).all(|(a, b)| (a - b).abs() < 1e-8);
        assert!(!same_q1 && !same_q2, "squad midpoint should differ from endpoints");
    }

    // --- quat_mul consistency ---

    #[test]
    fn test_quat_mul_identity() {
        let id = [1.0f64, 0.0, 0.0, 0.0];
        let q = [0.7071f64, 0.7071, 0.0, 0.0];
        let q_id = quat_mul(q, id);
        for i in 0..4 {
            assert!((q_id[i] - q[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_quat_inv() {
        let q = quat_norm([0.5f64, 0.5, 0.5, 0.5]);
        let qi = quat_conj(q);
        let prod = quat_mul(q, qi);
        assert!((prod[0] - 1.0).abs() < 1e-10);
        assert!(prod[1].abs() < 1e-10);
        assert!(prod[2].abs() < 1e-10);
        assert!(prod[3].abs() < 1e-10);
    }
}
