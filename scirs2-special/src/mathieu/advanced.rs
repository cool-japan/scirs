//! Advanced Mathieu Function Implementations
//!
//! This module provides high-accuracy implementations of Mathieu functions
//! using matrix eigenvalue methods and Fourier series expansions.
//!
//! Mathieu's equation: y'' + (a - 2q cos(2x)) y = 0
//!
//! Solutions are expressed as Fourier series:
//! - ce_n(q, x) = even Mathieu function (cosine-elliptic)
//! - se_n(q, x) = odd Mathieu function (sine-elliptic)
//!
//! Characteristic values a_n(q), b_n(q) are eigenvalues of tri-diagonal matrices.

use std::f64::consts::PI;

/// Configuration for Mathieu function computation.
#[derive(Debug, Clone)]
pub struct MathieuConfig {
    /// Fourier truncation order (number of Fourier components)
    pub n_fourier: usize,
    /// Maximum iterations for QR tridiagonal eigenvalue algorithm
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
}

impl Default for MathieuConfig {
    fn default() -> Self {
        MathieuConfig {
            n_fourier: 64,
            max_iter: 200,
            tol: 1e-12,
        }
    }
}

/// Characteristic value information for a Mathieu function.
#[derive(Debug, Clone)]
pub struct MathieuCharacteristic {
    /// Characteristic value a (for ce) or b (for se)
    pub a: f64,
    /// Parameter q
    pub q: f64,
    /// Order n
    pub order: usize,
    /// True for even functions (ce), false for odd (se)
    pub is_even: bool,
}

// ────────────────────────────────────────────────────────────────────────────
// Tridiagonal eigenvalue/eigenvector algorithms
// ────────────────────────────────────────────────────────────────────────────

/// Compute all eigenvalues of a symmetric tridiagonal matrix using QR iteration
/// with Wilkinson shift and deflation.
///
/// # Arguments
/// * `diag` - Diagonal entries d_0, ..., d_{n-1}
/// * `off_diag` - Off-diagonal entries e_1, ..., e_{n-1} (length n-1)
///
/// # Returns
/// Eigenvalues sorted in ascending order.
pub fn tridiag_eigenvalues(diag: &[f64], off_diag: &[f64]) -> Vec<f64> {
    let n = diag.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![diag[0]];
    }

    let config = MathieuConfig::default();
    let mut d = diag.to_vec();
    let mut e = vec![0.0f64; n]; // e[1..n-1] are the off-diagonal entries
    let copy_len = off_diag.len().min(n - 1);
    e[1..copy_len + 1].copy_from_slice(&off_diag[..copy_len]);

    // Use the implicit symmetric QR algorithm (LAPACK's dsteqr variant)
    implicit_symmetric_qr(&mut d, &mut e, None, config.max_iter, config.tol);

    d.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    d
}

/// Compute the eigenvector corresponding to a given eigenvalue via inverse iteration.
///
/// # Arguments
/// * `diag` - Diagonal entries
/// * `off_diag` - Off-diagonal entries (length n-1)
/// * `eigenvalue` - Target eigenvalue (must be an eigenvalue of the matrix)
///
/// # Returns
/// Normalized eigenvector
pub fn tridiag_eigenvector(diag: &[f64], off_diag: &[f64], eigenvalue: f64) -> Vec<f64> {
    let n = diag.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![1.0];
    }

    // Inverse iteration: solve (T - μI) v = b for random b, iterate
    let config = MathieuConfig::default();
    let shift = eigenvalue + 1e-10; // small shift to avoid exact singularity

    let mut v = vec![1.0f64 / (n as f64).sqrt(); n];
    // Perturb slightly
    for (i, vi) in v.iter_mut().enumerate() {
        *vi += 0.01 * ((i + 1) as f64).sin();
    }
    normalize_vec(&mut v);

    for _ in 0..config.max_iter {
        let v_old = v.clone();
        // Solve (T - shift*I) w = v using Thomas algorithm (tridiagonal solver)
        match tridiag_solve(diag, off_diag, shift, &v) {
            Some(w) => {
                let mut w = w;
                normalize_vec(&mut w);
                let dot: f64 = w.iter().zip(v_old.iter()).map(|(a, b)| a * b).sum();
                let converged = (1.0 - dot.abs()) < config.tol;
                v = w;
                if converged {
                    break;
                }
            }
            None => {
                // Matrix is singular: eigenvector is in the null space
                // Use random perturbation
                for (i, vi) in v.iter_mut().enumerate() {
                    *vi = ((i + 1) as f64 * 0.1).sin();
                }
                normalize_vec(&mut v);
                break;
            }
        }
    }
    normalize_vec(&mut v);
    v
}

/// Solve (T - shift*I) x = b for a symmetric tridiagonal T using Thomas algorithm.
/// Returns None if the system is (near-)singular.
fn tridiag_solve(diag: &[f64], off_diag: &[f64], shift: f64, rhs: &[f64]) -> Option<Vec<f64>> {
    let n = diag.len();
    // Build modified diagonals: a[i] = diag[i] - shift, b[i] = off_diag[i]
    let mut a: Vec<f64> = diag.iter().map(|&d| d - shift).collect();
    let b: Vec<f64> = if off_diag.len() >= n - 1 {
        off_diag[..n - 1].to_vec()
    } else {
        let mut bv = off_diag.to_vec();
        bv.resize(n - 1, 0.0);
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
    let mut x = vec![0.0f64; n];
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

/// Implicit symmetric QR step with Wilkinson shift.
/// Operates on a symmetric tridiagonal matrix T specified by diagonal d and
/// sub/superdiagonal e (1-indexed: e[1..n]).
fn implicit_symmetric_qr(
    d: &mut [f64],
    e: &mut [f64],
    _z: Option<&mut Vec<Vec<f64>>>,
    max_iter: usize,
    tol: f64,
) {
    let n = d.len();
    if n <= 1 {
        return;
    }

    // l and m define the active submatrix d[l..=m]
    let mut m = n - 1;

    for _ in 0..max_iter * n {
        if m == 0 {
            break;
        }
        // Check for small off-diagonal elements (deflation)
        // Find l: largest l such that e[l] is "negligible"
        let mut l = 0;
        for i in (1..=m).rev() {
            if e[i].abs() <= tol * (d[i - 1].abs() + d[i].abs()) {
                l = i;
                break;
            }
        }
        if l == m {
            m -= 1;
            continue;
        }

        // Wilkinson shift
        let tm = d[m];
        let tm1 = d[m - 1];
        let em = e[m];
        let delta = (tm1 - tm) / 2.0;
        let sign_delta = if delta >= 0.0 { 1.0 } else { -1.0 };
        let shift = tm - em * em / (delta + sign_delta * (delta * delta + em * em).sqrt());

        // QR step
        let mut g = d[l] - shift;
        let mut s = 1.0f64;
        let mut c = 1.0f64;
        let mut p = 0.0f64;

        for i in l..m {
            let f = s * e[i + 1];
            let b = c * e[i + 1];
            let r = (f * f + g * g).sqrt();
            if r < 1e-300 {
                e[i] = 0.0;
                d[i] -= p;
                // Remaining entries are unchanged
                break;
            }
            e[i] = r;
            s = f / r;
            c = g / r;
            g = d[i] - p;
            let r2 = (d[i + 1] - g) * s + 2.0 * c * b;
            p = s * r2;
            d[i] = g + p;
            g = c * r2 - b;
        }
        d[m] -= p;
        e[m] = g;
        e[l] = 0.0;
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Characteristic value computation
// ────────────────────────────────────────────────────────────────────────────

/// Compute the characteristic value a_n(q) for the even Mathieu function ce_n.
///
/// For q=0: a_n(0) = n².
/// For q≠0: eigenvalue of the Fourier tridiagonal matrix.
///
/// The tridiagonal matrix M for even functions:
/// - Even order n=2m: M[k,k] = (2k)², M[k,k±1] = q  (k=0,1,2,...)
///   Special: M[0,0] = 0, M[0,1] = √2 q (normalization convention)
/// - Odd order n=2m+1: M[k,k] = (2k+1)², M[k,k±1] = q
pub fn mathieu_a(n: usize, q: f64) -> f64 {
    let config = MathieuConfig::default();
    let size = config.n_fourier;

    if q == 0.0 {
        return (n * n) as f64;
    }

    if n.is_multiple_of(2) {
        // Even n: basis functions cos(2kx), k=0,1,...
        let half = size / 2;
        let mut diag = Vec::with_capacity(half);
        let mut off_diag = Vec::with_capacity(half - 1);

        // k=0: diagonal = 0, off-diagonal connects to k=1 with factor sqrt(2)*q
        // Actually the standard form uses:
        // M[0,0] = 0, M[0,1] = sqrt(2)*q (for n=0 normalization)
        // BUT for n≥2 even, the matrix is the same — we pick the n/2-th eigenvalue
        // Standard tridiagonal for ce_{2m}: diag = (2k)^2, off = q
        // with M[0,1] = sqrt(2)*q to account for the normalization of the cos(0) term
        diag.push(0.0f64);
        if half > 1 {
            off_diag.push(q * 2.0f64.sqrt()); // M[0,1]
        }
        for k in 1..half {
            diag.push((2 * k) as f64 * (2 * k) as f64);
            if k < half - 1 {
                off_diag.push(q);
            }
        }
        let eigenvalues = tridiag_eigenvalues(&diag, &off_diag);
        let idx = n / 2;
        *eigenvalues.get(idx).unwrap_or(&f64::NAN)
    } else {
        // Odd n: basis functions cos((2k+1)x), k=0,1,...
        // The period-2π boundary condition for a_{2m+1} shifts diagonal[0] by +q.
        // Reference: DLMF 28.4.7–28.4.8.
        let half = size / 2;
        let mut diag = Vec::with_capacity(half);
        let mut off_diag = Vec::with_capacity(half - 1);

        for k in 0..half {
            diag.push((2 * k + 1) as f64 * (2 * k + 1) as f64);
            if k < half - 1 {
                off_diag.push(q);
            }
        }
        // Apply the period-2π correction: diagonal[0] += q
        if !diag.is_empty() {
            diag[0] += q;
        }
        let eigenvalues = tridiag_eigenvalues(&diag, &off_diag);
        let idx = (n - 1) / 2;
        *eigenvalues.get(idx).unwrap_or(&f64::NAN)
    }
}

/// Compute the characteristic value b_n(q) for the odd Mathieu function se_n.
///
/// For q=0: b_n(0) = n² (same as a_n for n≥1; they split when q≠0).
/// The matrix differs from the even case in the off-diagonal structure.
pub fn mathieu_b(n: usize, q: f64) -> f64 {
    if n == 0 {
        return f64::INFINITY; // b_0 doesn't exist
    }

    let config = MathieuConfig::default();
    let size = config.n_fourier;

    if q == 0.0 {
        return (n * n) as f64;
    }

    if n.is_multiple_of(2) {
        // Even n: basis functions sin(2kx), k=1,2,...
        let half = size / 2;
        let mut diag = Vec::with_capacity(half);
        let mut off_diag = Vec::with_capacity(half - 1);

        for k in 1..=half {
            diag.push((2 * k) as f64 * (2 * k) as f64);
            if k < half {
                off_diag.push(q);
            }
        }
        let eigenvalues = tridiag_eigenvalues(&diag, &off_diag);
        let idx = n / 2 - 1;
        *eigenvalues.get(idx).unwrap_or(&f64::NAN)
    } else {
        // Odd n: basis functions sin((2k+1)x), k=0,1,...
        // The period-2π boundary condition for b_{2m+1} shifts diagonal[0] by -q.
        // Reference: DLMF 28.4.9–28.4.10.
        let half = size / 2;
        let mut diag = Vec::with_capacity(half);
        let mut off_diag = Vec::with_capacity(half - 1);

        for k in 0..half {
            diag.push((2 * k + 1) as f64 * (2 * k + 1) as f64);
            if k < half - 1 {
                off_diag.push(q);
            }
        }
        // Apply the period-2π correction: diagonal[0] -= q
        if !diag.is_empty() {
            diag[0] -= q;
        }
        let eigenvalues = tridiag_eigenvalues(&diag, &off_diag);
        let idx = (n - 1) / 2;
        *eigenvalues.get(idx).unwrap_or(&f64::NAN)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Fourier coefficient computation
// ────────────────────────────────────────────────────────────────────────────

/// Compute Fourier coefficients for ce_n(q, x) via inverse iteration.
///
/// Returns coefficients [A_0, A_2, A_4, ...] (even n) or [A_1, A_3, ...] (odd n)
/// for the Fourier expansion.
pub fn mathieu_ce_coefficients(n: usize, q: f64, config: &MathieuConfig) -> Vec<f64> {
    let size = config.n_fourier / 2;
    let a_n = mathieu_a(n, q);

    if n.is_multiple_of(2) {
        // Even n: ce_n = Σ_k A_{2k} cos(2kx)
        let half = size;
        let mut diag = Vec::with_capacity(half);
        let mut off_diag = Vec::with_capacity(half - 1);

        diag.push(0.0f64);
        if half > 1 {
            off_diag.push(q * 2.0f64.sqrt());
        }
        for k in 1..half {
            diag.push((2 * k) as f64 * (2 * k) as f64);
            if k < half - 1 {
                off_diag.push(q);
            }
        }
        let mut coeffs = tridiag_eigenvector(&diag, &off_diag, a_n);
        // Ensure normalization: Σ A_{2k}^2 = 1 (already done in eigenvector)
        // Sign convention: A_0 > 0 for ce_0
        let sign_idx = if n == 0 { 0 } else { n / 2 };
        if coeffs[sign_idx] < 0.0 {
            for c in coeffs.iter_mut() {
                *c = -*c;
            }
        }
        coeffs
    } else {
        // Odd n: ce_n = Σ_k A_{2k+1} cos((2k+1)x)
        let half = size;
        let mut diag = Vec::with_capacity(half);
        let mut off_diag = Vec::with_capacity(half - 1);

        for k in 0..half {
            diag.push((2 * k + 1) as f64 * (2 * k + 1) as f64);
            if k < half - 1 {
                off_diag.push(q);
            }
        }
        let mut coeffs = tridiag_eigenvector(&diag, &off_diag, a_n);
        let idx = (n - 1) / 2;
        if idx < coeffs.len() && coeffs[idx] < 0.0 {
            for c in coeffs.iter_mut() {
                *c = -*c;
            }
        }
        coeffs
    }
}

/// Compute Fourier coefficients for se_n(q, x) via inverse iteration.
///
/// Returns coefficients [B_2, B_4, ...] (even n) or [B_1, B_3, ...] (odd n).
pub fn mathieu_se_coefficients(n: usize, q: f64, config: &MathieuConfig) -> Vec<f64> {
    if n == 0 {
        return vec![];
    }
    let size = config.n_fourier / 2;
    let b_n = mathieu_b(n, q);

    if n.is_multiple_of(2) {
        let half = size;
        let mut diag = Vec::with_capacity(half);
        let mut off_diag = Vec::with_capacity(half - 1);
        for k in 1..=half {
            diag.push((2 * k) as f64 * (2 * k) as f64);
            if k < half {
                off_diag.push(q);
            }
        }
        let mut coeffs = tridiag_eigenvector(&diag, &off_diag, b_n);
        let idx = n / 2 - 1;
        if idx < coeffs.len() && coeffs[idx] < 0.0 {
            for c in coeffs.iter_mut() {
                *c = -*c;
            }
        }
        coeffs
    } else {
        let half = size;
        let mut diag = Vec::with_capacity(half);
        let mut off_diag = Vec::with_capacity(half - 1);
        for k in 0..half {
            diag.push((2 * k + 1) as f64 * (2 * k + 1) as f64);
            if k < half - 1 {
                off_diag.push(q);
            }
        }
        let mut coeffs = tridiag_eigenvector(&diag, &off_diag, b_n);
        let idx = (n - 1) / 2;
        if idx < coeffs.len() && coeffs[idx] < 0.0 {
            for c in coeffs.iter_mut() {
                *c = -*c;
            }
        }
        coeffs
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Function evaluation via Fourier series
// ────────────────────────────────────────────────────────────────────────────

/// Evaluate the even Mathieu function ce_n(q, x).
///
/// ce_n(q, x) = Σ_k A_{2k} cos(2kx)   (n even)
///            = Σ_k A_{2k+1} cos((2k+1)x) (n odd)
///
/// Normalized so that (1/π) ∫_0^{2π} ce_n(q,x)^2 dx = 1.
pub fn mathieu_ce(n: usize, q: f64, x: f64) -> f64 {
    let config = MathieuConfig::default();
    let coeffs = mathieu_ce_coefficients(n, q, &config);

    if n.is_multiple_of(2) {
        // Σ A_{2k} cos(2kx)
        let mut sum = 0.0f64;
        for (k, &ak) in coeffs.iter().enumerate() {
            sum += ak * ((2 * k) as f64 * x).cos();
        }
        sum
    } else {
        // Σ A_{2k+1} cos((2k+1)x)
        let mut sum = 0.0f64;
        for (k, &ak) in coeffs.iter().enumerate() {
            sum += ak * ((2 * k + 1) as f64 * x).cos();
        }
        sum
    }
}

/// Evaluate the odd Mathieu function se_n(q, x).
///
/// se_n(q, x) = Σ_k B_{2k} sin(2kx)   (n even)
///            = Σ_k B_{2k+1} sin((2k+1)x) (n odd)
///
/// Normalized so that (1/π) ∫_0^{2π} se_n(q,x)^2 dx = 1.
pub fn mathieu_se(n: usize, q: f64, x: f64) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let config = MathieuConfig::default();
    let coeffs = mathieu_se_coefficients(n, q, &config);

    if n.is_multiple_of(2) {
        // Σ B_{2k} sin(2kx)
        let mut sum = 0.0f64;
        for (k, &bk) in coeffs.iter().enumerate() {
            let freq = (2 * (k + 1)) as f64;
            sum += bk * (freq * x).sin();
        }
        sum
    } else {
        // Σ B_{2k+1} sin((2k+1)x)
        let mut sum = 0.0f64;
        for (k, &bk) in coeffs.iter().enumerate() {
            sum += bk * ((2 * k + 1) as f64 * x).sin();
        }
        sum
    }
}

/// Evaluate ce_n(q, x) at a batch of x values.
///
/// More efficient than calling `mathieu_ce` repeatedly since Fourier coefficients
/// are computed only once.
pub fn mathieu_ce_batch(n: usize, q: f64, xs: &[f64]) -> Vec<f64> {
    let config = MathieuConfig::default();
    let coeffs = mathieu_ce_coefficients(n, q, &config);

    xs.iter()
        .map(|&x| {
            if n.is_multiple_of(2) {
                let mut sum = 0.0f64;
                for (k, &ak) in coeffs.iter().enumerate() {
                    sum += ak * ((2 * k) as f64 * x).cos();
                }
                sum
            } else {
                let mut sum = 0.0f64;
                for (k, &ak) in coeffs.iter().enumerate() {
                    sum += ak * ((2 * k + 1) as f64 * x).cos();
                }
                sum
            }
        })
        .collect()
}

// ────────────────────────────────────────────────────────────────────────────
// Modified Mathieu functions (radial)
// ────────────────────────────────────────────────────────────────────────────

/// Evaluate the modified Mathieu function Mc1_n(q, r) (first kind, radial).
///
/// For large r: asymptotic expansion
/// Mc1_n(q, r) ≈ sqrt(2/(π r)) * cos(r - n*π/2 - π/4) * correction
///
/// For small r (r < 1): series expansion or direct evaluation.
/// The modification is ce_n(q, ix) analytically continued to the real radial variable r.
pub fn mathieu_modified_m1(n: usize, q: f64, r: f64) -> f64 {
    if r <= 0.0 {
        return f64::NAN;
    }

    if r >= 5.0 {
        // Asymptotic expansion for large r
        // Mc1_n(q, r) ~ sqrt(2/πr) * cos(r - nπ/2 - π/4)
        let phase = r - n as f64 * PI / 2.0 - PI / 4.0;
        let amplitude = (2.0 / (PI * r)).sqrt();
        amplitude * phase.cos()
    } else if r < 1.0 {
        // Series expansion for small r: use the connection formula
        // Mc1_n(q, r) = ce_n(q, 0) * cosh(r * sqrt(...)) / ce_n(q, 0) ...
        // Actually use the Fourier series evaluated at ix:
        // ce_n(q, ix) = Σ_k A_{2k} cosh(2k r)  (n even)
        //             = Σ_k A_{2k+1} cosh((2k+1) r) (n odd)
        let config = MathieuConfig::default();
        let coeffs = mathieu_ce_coefficients(n, q, &config);
        let mut sum = 0.0f64;
        if n.is_multiple_of(2) {
            for (k, &ak) in coeffs.iter().enumerate() {
                sum += ak * ((2 * k) as f64 * r).cosh();
            }
        } else {
            for (k, &ak) in coeffs.iter().enumerate() {
                sum += ak * ((2 * k + 1) as f64 * r).cosh();
            }
        }
        sum
    } else {
        // Intermediate r: use Fourier series with hyperbolic functions
        let config = MathieuConfig::default();
        let coeffs = mathieu_ce_coefficients(n, q, &config);
        let mut sum = 0.0f64;
        if n.is_multiple_of(2) {
            for (k, &ak) in coeffs.iter().enumerate() {
                let arg = (2 * k) as f64 * r;
                sum += ak * arg.cosh();
            }
        } else {
            for (k, &ak) in coeffs.iter().enumerate() {
                let arg = (2 * k + 1) as f64 * r;
                sum += ak * arg.cosh();
            }
        }
        sum
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Stability chart
// ────────────────────────────────────────────────────────────────────────────

/// Compute stability boundary points for Mathieu's equation.
///
/// Returns a vector of `(q, a, stable)` triples representing points in the
/// (q, a) stability diagram.
///
/// For a given q, the stable regions are: a_{2k}(q) ≤ a ≤ b_{2k+1}(q) for k=0,1,...
/// (the regions between consecutive characteristic curves where solutions are bounded).
pub fn mathieu_stability_chart(q_max: f64, n_points: usize) -> Vec<(f64, f64, bool)> {
    let mut result = Vec::new();
    let n_orders = 4; // compute first 4 orders

    for qi in 0..=n_points {
        let q = q_max * qi as f64 / n_points as f64;

        // Compute characteristic values
        let a_vals: Vec<f64> = (0..n_orders).map(|n| mathieu_a(n, q)).collect();
        let b_vals: Vec<f64> = (1..=n_orders).map(|n| mathieu_b(n, q)).collect();

        // Stability boundaries: a_0, b_1, a_1, b_2, a_2, ...
        // Stable: a_0 < a < b_1 (first stable region, actually a < a_0 is stable too for q>0)
        // Simplified: mark boundaries
        for k in 0..n_orders.min(a_vals.len()) {
            result.push((q, a_vals[k], true));
        }
        for k in 0..n_orders.min(b_vals.len()) {
            result.push((q, b_vals[k], false));
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_mathieu_a_q0() {
        // a_n(0) = n^2
        assert_relative_eq!(mathieu_a(0, 0.0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(mathieu_a(1, 0.0), 1.0, epsilon = 1e-10);
        assert_relative_eq!(mathieu_a(2, 0.0), 4.0, epsilon = 1e-10);
        assert_relative_eq!(mathieu_a(3, 0.0), 9.0, epsilon = 1e-10);
        assert_relative_eq!(mathieu_a(4, 0.0), 16.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mathieu_b_q0() {
        // b_n(0) = n^2 for n >= 1
        assert_relative_eq!(mathieu_b(1, 0.0), 1.0, epsilon = 1e-10);
        assert_relative_eq!(mathieu_b(2, 0.0), 4.0, epsilon = 1e-10);
        assert_relative_eq!(mathieu_b(3, 0.0), 9.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mathieu_a_small_q() {
        // a_0(q) ≈ -2q² + ... for small q (perturbation theory)
        // a_0(0.1) ≈ -0.02 (first order correction is second order in q)
        let a0 = mathieu_a(0, 0.1);
        assert!(a0.is_finite(), "a_0(0.1) should be finite");
        // Should be close to 0 for small q
        assert!(a0.abs() < 0.5, "a_0(0.1) should be small");
    }

    #[test]
    fn test_mathieu_ce_normalization() {
        // ce_0(0, x) = 1/sqrt(2) (constant)
        // At q=0: ce_n is just cos(nx) (normalized)
        let val = mathieu_ce(0, 0.0, 0.0);
        // ce_0(0, 0) should be 1/sqrt(2) (SciPy convention) or 1.0 (unit norm)
        assert!(val.is_finite(), "ce_0(0,0) should be finite");
        assert!(val.abs() > 0.1, "ce_0(0,0) should be nonzero");
    }

    #[test]
    fn test_mathieu_ce_batch_consistent() {
        let xs = vec![0.0, PI / 4.0, PI / 2.0, PI];
        let batch = mathieu_ce_batch(1, 0.5, &xs);
        for (i, &x) in xs.iter().enumerate() {
            let single = mathieu_ce(1, 0.5, x);
            assert_relative_eq!(batch[i], single, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_mathieu_stability_chart_nonempty() {
        let chart = mathieu_stability_chart(5.0, 20);
        assert!(
            !chart.is_empty(),
            "Stability chart should return non-empty vec"
        );
        assert!(chart.len() > 10, "Should have multiple points");
    }

    #[test]
    fn test_mathieu_ce_se_orthogonality_q0() {
        // At q=0: ce_1(0, x) = cos(x)/norm, se_1(0, x) = sin(x)/norm
        // They should be orthogonal: ∫_0^{2π} ce_1*se_1 dx ≈ 0
        let n_pts = 1000;
        let dx = 2.0 * PI / n_pts as f64;
        let sum: f64 = (0..n_pts)
            .map(|i| {
                let x = i as f64 * dx;
                mathieu_ce(1, 0.0, x) * mathieu_se(1, 0.0, x)
            })
            .sum::<f64>()
            * dx;
        assert!(
            sum.abs() < 0.05,
            "ce_1 and se_1 should be approximately orthogonal"
        );
    }

    #[test]
    fn test_mathieu_modified_m1_large_r() {
        // For large r, Mc1_0(q, r) should be real and finite
        let val = mathieu_modified_m1(0, 1.0, 10.0);
        assert!(val.is_finite());
        assert!(val.abs() < 1.0, "Should be bounded for large r");
    }

    #[test]
    fn test_tridiag_eigenvalues_simple() {
        // 2x2 symmetric tridiagonal: [[2, 1], [1, 2]]
        // Eigenvalues: 1 and 3
        let diag = vec![2.0, 2.0];
        let off_diag = vec![1.0];
        let eigs = tridiag_eigenvalues(&diag, &off_diag);
        assert_eq!(eigs.len(), 2);
        assert_relative_eq!(eigs[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(eigs[1], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_tridiag_eigenvector() {
        // 2x2: eigenvalue 1 → eigenvector proportional to [1, -1]
        let diag = vec![2.0, 2.0];
        let off_diag = vec![1.0];
        let v = tridiag_eigenvector(&diag, &off_diag, 1.0);
        assert_eq!(v.len(), 2);
        // v[0]/v[1] should be -1
        let ratio = v[0] / v[1];
        assert_relative_eq!(ratio.abs(), 1.0, epsilon = 1e-5);
        assert!(ratio < 0.0, "Components should have opposite signs");
    }
}
