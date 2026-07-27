//! ESPRIT (Estimation of Signal Parameters via Rotational Invariance Techniques).
//!
//! Reference: Roy & Kailath (1989). "ESPRIT — Estimation of signal parameters via
//! rotational invariance techniques." IEEE Trans. ASSP 37(7):984-995.

use std::f64::consts::PI;

use crate::error::{SignalError, SignalResult};

use super::types::{FrequencyComponent, PhaseEstResult, PhaseMethod};

// ─── Internal linear algebra helpers ─────────────────────────────────────────

/// Matrix-vector product y = A * x  (A is nrows×ncols row-major).
fn mat_vec(a: &[f64], nrows: usize, ncols: usize, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0f64; nrows];
    for i in 0..nrows {
        for j in 0..ncols {
            y[i] += a[i * ncols + j] * x[j];
        }
    }
    y
}

/// Dot product of two equal-length slices.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum()
}

/// Normalise a vector in-place; returns its norm.
fn normalise(v: &mut [f64]) -> f64 {
    let n = dot(v, v).sqrt();
    if n > 0.0 {
        v.iter_mut().for_each(|x| *x /= n);
    }
    n
}

/// In-place Gram-Schmidt orthogonalisation of `vec` against each column of `basis`.
fn gram_schmidt_step(vec: &mut Vec<f64>, basis: &[Vec<f64>]) {
    for b in basis {
        let proj = dot(vec, b);
        for (v, &bi) in vec.iter_mut().zip(b.iter()) {
            *v -= proj * bi;
        }
    }
}

/// Compute the top-`k` left singular vectors of `mat` (nrows × ncols, row-major)
/// using block simultaneous subspace iteration (Bai et al. "Templates for the
/// Solution of Algebraic Eigenvalue Problems").
///
/// Each iteration applies (A·A^T) simultaneously to all k columns and
/// re-orthogonalises via modified Gram-Schmidt. This converges orders of magnitude
/// faster than sequential deflation when singular values are close.
///
/// Returns `U` as a flat Vec<f64> of shape (nrows × k), column-major:
/// column j is `U[j*nrows .. (j+1)*nrows]`.
fn top_k_left_singular_vectors(
    mat: &[f64],
    nrows: usize,
    ncols: usize,
    k: usize,
    max_iter: usize,
    tol: f64,
) -> SignalResult<Vec<f64>> {
    if k == 0 || k > nrows.min(ncols) {
        return Err(SignalError::InvalidArgument(format!(
            "k={k} out of range for {}×{} matrix",
            nrows, ncols
        )));
    }

    // Initialise U as first k standard basis vectors (deterministic, good for convergence).
    // U is stored column-major: col j → U[j*nrows .. (j+1)*nrows].
    let mut u = vec![0.0f64; nrows * k];
    for col in 0..k {
        let idx = if col < nrows { col } else { 0 };
        u[col * nrows + idx] = 1.0;
    }

    let mut prev_u = u.clone();

    for _iter in 0..max_iter {
        // Apply A·A^T to each column simultaneously.
        // Step 1: W = A^T · U  (ncols × k)
        let mut w = vec![0.0f64; ncols * k];
        for col in 0..k {
            for j in 0..ncols {
                let mut s = 0.0f64;
                for i in 0..nrows {
                    s += mat[i * ncols + j] * u[col * nrows + i];
                }
                w[col * ncols + j] = s;
            }
        }

        // Step 2: Z = A · W  (nrows × k)
        let mut z = vec![0.0f64; nrows * k];
        for col in 0..k {
            for i in 0..nrows {
                let mut s = 0.0f64;
                for j in 0..ncols {
                    s += mat[i * ncols + j] * w[col * ncols + j];
                }
                z[col * nrows + i] = s;
            }
        }

        // Step 3: Modified Gram-Schmidt orthogonalisation of Z.
        for col in 0..k {
            // Orthogonalise column `col` against all previous columns.
            for prev in 0..col {
                let proj = (0..nrows)
                    .map(|i| z[col * nrows + i] * z[prev * nrows + i])
                    .sum::<f64>();
                for i in 0..nrows {
                    let sub = proj * z[prev * nrows + i];
                    z[col * nrows + i] -= sub;
                }
            }
            // Normalise.
            let norm = (0..nrows)
                .map(|i| z[col * nrows + i].powi(2))
                .sum::<f64>()
                .sqrt();
            if norm < 1e-14 {
                // Degenerate column: use a fresh standard basis vector.
                z[col * nrows..col * nrows + nrows].fill(0.0);
                let idx = if col < nrows { col } else { 0 };
                z[col * nrows + idx] = 1.0;
            } else {
                for i in 0..nrows {
                    z[col * nrows + i] /= norm;
                }
            }
        }

        // Check convergence: max change across all columns.
        let max_change = (0..k)
            .map(|col| {
                (0..nrows)
                    .map(|i| (z[col * nrows + i] - prev_u[col * nrows + i]).abs())
                    .fold(0.0f64, f64::max)
                    .min(
                        (0..nrows)
                            .map(|i| (z[col * nrows + i] + prev_u[col * nrows + i]).abs())
                            .fold(0.0f64, f64::max),
                    )
            })
            .fold(0.0f64, f64::max);

        prev_u = z.clone();
        u = z;

        if max_change < tol {
            break;
        }
    }

    Ok(u)
}

/// Pseudo-inverse of a small matrix via normal equations: pinv(A) = (A^T A)^{-1} A^T.
/// A is (m×n) row-major; result is (n×m) row-major.
fn pseudo_inverse(a: &[f64], m: usize, n: usize) -> SignalResult<Vec<f64>> {
    // Gram matrix G = A^T A  (n×n)
    let mut g = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            g[i * n + j] = (0..m).map(|k| a[k * n + i] * a[k * n + j]).sum();
        }
    }

    let g_inv = invert_symmetric(g, n)?;

    // pinv(A) = G^{-1} A^T  →  shape (n×m)
    let mut pinv = vec![0.0f64; n * m];
    for i in 0..n {
        for j in 0..m {
            pinv[i * m + j] = (0..n).map(|k| g_inv[i * n + k] * a[j * n + k]).sum();
        }
    }
    Ok(pinv)
}

/// Invert a real n×n matrix via Gaussian elimination with partial pivoting.
/// Adds Tikhonov regularisation to handle near-singular cases.
fn invert_symmetric(mut g: Vec<f64>, n: usize) -> SignalResult<Vec<f64>> {
    // Tikhonov regularisation.
    let trace = (0..n).map(|i| g[i * n + i]).sum::<f64>();
    let lambda = 1e-10 * trace / n as f64 + 1e-14;
    for i in 0..n {
        g[i * n + i] += lambda;
    }

    let mut aug = vec![0.0f64; n * (2 * n)];
    for i in 0..n {
        for j in 0..n {
            aug[i * (2 * n) + j] = g[i * n + j];
        }
        aug[i * (2 * n) + n + i] = 1.0;
    }

    for col in 0..n {
        // Partial pivot.
        let pivot_row = (col..n)
            .max_by(|&a, &b| {
                aug[a * (2 * n) + col]
                    .abs()
                    .partial_cmp(&aug[b * (2 * n) + col].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or_else(|| SignalError::ComputationError("Empty range".into()))?;

        if aug[pivot_row * (2 * n) + col].abs() < 1e-14 {
            return Err(SignalError::ComputationError(
                "Matrix is (near-)singular in pseudo-inverse".into(),
            ));
        }

        if pivot_row != col {
            for j in 0..(2 * n) {
                aug.swap(col * (2 * n) + j, pivot_row * (2 * n) + j);
            }
        }

        let scale = 1.0 / aug[col * (2 * n) + col];
        for j in 0..(2 * n) {
            aug[col * (2 * n) + j] *= scale;
        }

        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row * (2 * n) + col];
            for j in 0..(2 * n) {
                let v = aug[col * (2 * n) + j];
                aug[row * (2 * n) + j] -= factor * v;
            }
        }
    }

    let mut inv = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = aug[i * (2 * n) + n + j];
        }
    }
    Ok(inv)
}

// ─── Jacobi EVD (real symmetric) ─────────────────────────────────────────────

/// Jacobi EVD for a real symmetric matrix (n×n row-major).
/// After this call, `a` has eigenvalues on its diagonal.
/// Returns eigenvectors sorted by *ascending* eigenvalue, each as a Vec<f64> of length n.
fn jacobi_evd_sorted(a: &mut [f64], n: usize, max_sweep: usize) -> Vec<Vec<f64>> {
    // Initialise V = I.
    let mut v: Vec<f64> = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    for _sweep in 0..max_sweep {
        let mut max_off = 0.0f64;
        for p in 0..n {
            for q in (p + 1)..n {
                let val = a[p * n + q].abs();
                if val > max_off {
                    max_off = val;
                }
            }
        }
        if max_off < 1e-13 {
            break;
        }

        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.abs() < 1e-14 * max_off {
                    continue;
                }
                let app = a[p * n + p];
                let aqq = a[q * n + q];
                let theta = 0.5 * (aqq - app) / apq;
                let t = if theta >= 0.0 {
                    1.0 / (theta + (1.0 + theta * theta).sqrt())
                } else {
                    -1.0 / (-theta + (1.0 + theta * theta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                a[p * n + p] = app - t * apq;
                a[q * n + q] = aqq + t * apq;
                a[p * n + q] = 0.0;
                a[q * n + p] = 0.0;

                for r in 0..n {
                    if r == p || r == q {
                        continue;
                    }
                    let arp = a[r * n + p];
                    let arq = a[r * n + q];
                    a[r * n + p] = c * arp - s * arq;
                    a[p * n + r] = a[r * n + p];
                    a[r * n + q] = s * arp + c * arq;
                    a[q * n + r] = a[r * n + q];
                }

                for r in 0..n {
                    let vrp = v[r * n + p];
                    let vrq = v[r * n + q];
                    v[r * n + p] = c * vrp - s * vrq;
                    v[r * n + q] = s * vrp + c * vrq;
                }
            }
        }
    }

    // Extract eigenvalue / eigenvector pairs and sort ascending by eigenvalue.
    let mut pairs: Vec<(f64, Vec<f64>)> = (0..n)
        .map(|col| {
            let eval = a[col * n + col];
            let evec: Vec<f64> = (0..n).map(|row| v[row * n + col]).collect();
            (eval, evec)
        })
        .collect();
    pairs.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));

    // Put eigenvalues back on diagonal in sorted order and return sorted eigenvectors.
    for (i, (eval, _)) in pairs.iter().enumerate() {
        a[i * n + i] = *eval;
    }
    pairs.into_iter().map(|(_, evec)| evec).collect()
}

// ─── Real Schur / Francis QR algorithm for small matrices ─────────────────────
//
// Produces complex eigenvalues of a real n×n matrix via the implicit double-shift
// QR iteration (Francis 1961). Works correctly for conjugate-complex eigenvalue
// pairs which are what ESPRIT's Φ matrix has.

/// Hessenberg reduction in place (A becomes upper Hessenberg; H = Q^T A Q).
fn reduce_to_hessenberg(a: &mut [f64], n: usize) {
    for col in 0..n.saturating_sub(2) {
        // Householder reflector for column `col` below the sub-diagonal.
        let start = col + 1;
        let len = n - start;
        if len == 0 {
            break;
        }

        let mut x = vec![0.0f64; len];
        for (i, xi) in x.iter_mut().enumerate() {
            *xi = a[(start + i) * n + col];
        }

        let sigma = if x[0] >= 0.0 { 1.0 } else { -1.0 };
        let norm_x = x.iter().map(|&v| v * v).sum::<f64>().sqrt();
        x[0] += sigma * norm_x;
        let norm_v = x.iter().map(|&v| v * v).sum::<f64>().sqrt();
        if norm_v < 1e-14 {
            continue;
        }
        for v in x.iter_mut() {
            *v /= norm_v;
        }

        // Apply H from the left: A[start:, :] -= 2 v (v^T A[start:, :])
        for j in 0..n {
            let dot_val: f64 = (0..len).map(|i| x[i] * a[(start + i) * n + j]).sum();
            for i in 0..len {
                a[(start + i) * n + j] -= 2.0 * x[i] * dot_val;
            }
        }

        // Apply H from the right: A[:, start:] -= 2 (A[:, start:] v) v^T
        for i in 0..n {
            let dot_val: f64 = (0..len).map(|j| a[i * n + start + j] * x[j]).sum();
            for j in 0..len {
                a[i * n + start + j] -= 2.0 * dot_val * x[j];
            }
        }
    }
}

/// Francis double-shift QR step on the unreduced sub-Hessenberg block H[lo..hi+1, lo..hi+1].
fn francis_double_step(h: &mut [f64], n: usize, lo: usize, hi: usize) {
    let m = hi - 1;
    // Wilkinson double shifts: eigenvalues of the trailing 2×2.
    let s = h[m * n + m] + h[hi * n + hi];
    let t = h[m * n + m] * h[hi * n + hi] - h[m * n + hi] * h[hi * n + m];

    // First column of (H - s*I)(H - conj(s)*I) = H² - s·H + t·I.
    let mut x = h[lo * n + lo] * h[lo * n + lo] + h[lo * n + lo + 1] * h[(lo + 1) * n + lo]
        - s * h[lo * n + lo]
        + t;
    let mut y = h[(lo + 1) * n + lo] * (h[lo * n + lo] + h[(lo + 1) * n + lo + 1] - s);
    // z = H[lo+2][lo+1] · H[lo+1][lo]. It must read the *sub-diagonal* entry:
    // H[lo+2][lo] sits two rows below the diagonal and is structurally zero in a
    // Hessenberg matrix, so using it pins z to 0 and the double-shift bulge is
    // never formed.
    let mut z = if hi > lo + 1 {
        h[(lo + 2) * n + lo + 1] * h[(lo + 1) * n + lo]
    } else {
        0.0
    };

    for k in lo..hi {
        // Householder reflector for [x, y, z].
        let len = if k + 2 <= hi { 3 } else { 2 };
        let mut v = vec![0.0f64; len];
        v[0] = x;
        if len >= 2 {
            v[1] = y;
        }
        if len >= 3 {
            v[2] = z;
        }

        let sigma = if v[0] >= 0.0 { 1.0 } else { -1.0 };
        let norm_v = v.iter().map(|&vi| vi * vi).sum::<f64>().sqrt();
        if norm_v < 1e-14 {
            // Skip degenerate step.
            if k + 1 < n {
                x = h[(k + 1) * n + k];
                y = if k + 2 < n { h[(k + 2) * n + k] } else { 0.0 };
                z = if k + 3 < n { h[(k + 3) * n + k] } else { 0.0 };
            }
            continue;
        }
        v[0] += sigma * norm_v;
        let norm_v2 = v.iter().map(|&vi| vi * vi).sum::<f64>().sqrt();
        for vi in v.iter_mut() {
            *vi /= norm_v2;
        }

        let r = k.saturating_sub(1).max(lo); // clamp to block
                                             // Left multiply: H[k:k+len, r:] -= 2 v (v^T H[k:k+len, r:])
        for j in r..n {
            let dot_val: f64 = (0..len).map(|i| v[i] * h[(k + i) * n + j]).sum();
            for i in 0..len {
                h[(k + i) * n + j] -= 2.0 * v[i] * dot_val;
            }
        }

        // Right multiply: H[:, k:k+len] -= 2 (H[:, k:k+len] v) v^T
        //
        // The row limit must be `k + len + 1`, not `k + len`. Row `k+len` still has a
        // non-zero in this column range -- its sub-diagonal entry H[k+len][k+len-1] --
        // so stopping one row short leaves that entry un-rotated. The step then applies
        // P from the left but only part of P^T from the right, which is no longer a
        // similarity transform: the eigenvalues themselves drift away from the input
        // matrix's, silently and without any loss of Hessenberg structure to signal it.
        let r2 = (k + len + 1).min(n);
        for i in 0..r2 {
            let dot_val: f64 = (0..len).map(|j| h[i * n + k + j] * v[j]).sum();
            for j in 0..len {
                h[i * n + k + j] -= 2.0 * dot_val * v[j];
            }
        }

        // Update bulge.
        x = h[(k + 1) * n + k];
        y = if k + 2 <= hi { h[(k + 2) * n + k] } else { 0.0 };
        z = if k + 3 <= hi { h[(k + 3) * n + k] } else { 0.0 };
    }
}

/// Extract complex eigenvalues from a quasi-upper-triangular (real Schur form) matrix.
fn extract_eigenvalues_from_schur(h: &[f64], n: usize) -> Vec<(f64, f64)> {
    let mut eigs = Vec::with_capacity(n);
    let mut i = 0;
    while i < n {
        if i + 1 < n && h[(i + 1) * n + i].abs() > 1e-10 {
            // 2×2 block → complex conjugate pair.
            let a = h[i * n + i];
            let b = h[i * n + i + 1];
            let c = h[(i + 1) * n + i];
            let d = h[(i + 1) * n + i + 1];
            let tr = a + d;
            let det = a * d - b * c;
            let disc = tr * tr - 4.0 * det;
            if disc >= 0.0 {
                let s = disc.sqrt();
                eigs.push(((tr + s) / 2.0, 0.0));
                eigs.push(((tr - s) / 2.0, 0.0));
            } else {
                let im = (-disc).sqrt() / 2.0;
                eigs.push((tr / 2.0, im));
                eigs.push((tr / 2.0, -im));
            }
            i += 2;
        } else {
            eigs.push((h[i * n + i], 0.0));
            i += 1;
        }
    }
    eigs
}

/// Compute all eigenvalues of a real n×n matrix using the Francis implicit double-shift
/// QR algorithm. Returns (re, im) pairs.
fn eigenvalues_francis_qr(mat: &[f64], n: usize, max_iter: usize) -> Vec<(f64, f64)> {
    if n == 1 {
        return vec![(mat[0], 0.0)];
    }
    if n == 2 {
        let tr = mat[0] + mat[3];
        let det = mat[0] * mat[3] - mat[1] * mat[2];
        let disc = tr * tr - 4.0 * det;
        return if disc >= 0.0 {
            let s = disc.sqrt();
            vec![((tr + s) / 2.0, 0.0), ((tr - s) / 2.0, 0.0)]
        } else {
            let im = (-disc).sqrt() / 2.0;
            vec![(tr / 2.0, im), (tr / 2.0, -im)]
        };
    }

    let mut h = mat.to_vec();
    reduce_to_hessenberg(&mut h, n);

    // Standard deflation driver: repeatedly locate the *active* unreduced block
    // H[lo..=hi] at the bottom of the matrix, then either retire it (1×1 or 2×2,
    // which `extract_eigenvalues_from_schur` reads directly) or run one Francis
    // step on it. Searching upward from `hi` for the first negligible sub-diagonal
    // is what keeps the block unreduced -- a Francis step driven across an interior
    // zero sub-diagonal chases the bulge through a block boundary and corrupts both
    // sides.
    let mut hi = n - 1;
    let mut iters = 0;

    while hi > 0 && iters < max_iter {
        iters += 1;

        let mut lo = hi;
        while lo > 0 {
            let tol = 1e-14 * (h[(lo - 1) * n + lo - 1].abs() + h[lo * n + lo].abs());
            if h[lo * n + lo - 1].abs() <= tol.max(f64::MIN_POSITIVE) {
                h[lo * n + lo - 1] = 0.0;
                break;
            }
            lo -= 1;
        }

        if lo == hi {
            // Converged 1×1 block.
            hi -= 1;
        } else if lo + 1 == hi {
            // Converged 2×2 block (a real pair or a complex conjugate pair).
            if hi < 2 {
                break;
            }
            hi -= 2;
        } else {
            francis_double_step(&mut h, n, lo, hi);
        }
    }

    extract_eigenvalues_from_schur(&h, n)
}

// ─── Hankel matrix builder ────────────────────────────────────────────────────

/// Build the Hankel data matrix X of shape (n-L) × (L+1) where
/// `X[i, j] = signal[i + j]`.
pub(crate) fn build_hankel(signal: &[f64], l: usize) -> SignalResult<(Vec<f64>, usize, usize)> {
    let n = signal.len();
    if l >= n {
        return Err(SignalError::InvalidArgument(format!(
            "Hankel window L={l} must be < signal length {n}"
        )));
    }
    let nrows = n - l;
    let ncols = l + 1;
    let mut x = vec![0.0f64; nrows * ncols];
    for i in 0..nrows {
        for j in 0..ncols {
            x[i * ncols + j] = signal[i + j];
        }
    }
    Ok((x, nrows, ncols))
}

// ─── EspritEstimator ─────────────────────────────────────────────────────────

/// ESPRIT frequency and phase estimator.
#[derive(Debug, Clone)]
pub struct EspritEstimator {
    /// Number of signal components to extract.
    pub num_components: usize,
    /// Signal subspace dimension (must be ≥ num_components).
    pub subspace_dim: usize,
    /// Maximum iterations for power iteration / QR.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: f64,
    /// Sample rate in Hz (used only for converting output frequencies to Hz).
    pub fs: f64,
}

impl Default for EspritEstimator {
    fn default() -> Self {
        Self {
            num_components: 1,
            subspace_dim: 4,
            max_iter: 200,
            tol: 1e-10,
            fs: 1.0,
        }
    }
}

impl EspritEstimator {
    /// Create a new estimator.
    pub fn new(num_components: usize, fs: f64) -> Self {
        let subspace_dim = (3 * num_components).max(6);
        Self {
            num_components,
            subspace_dim,
            fs,
            ..Default::default()
        }
    }

    /// Compute the Hankel window length (L) from signal length.
    ///
    /// We keep `ncols = L+1` bounded to avoid huge correlation matrices in the EVD.
    /// A good tradeoff: L ≈ max(2·subspace_dim, sqrt(n)), capped at n/2.
    fn hankel_l(&self, n: usize) -> usize {
        let l_min = self.subspace_dim.saturating_sub(1);
        // Upper bound: keep ncols ≤ 4*subspace_dim to limit EVD cost,
        // but at least l_min and at most n/2.
        let l_max_affordable = (4 * self.subspace_dim).max(32);
        let l_default = l_max_affordable.min(n / 2);
        l_default.max(l_min).min(n.saturating_sub(1))
    }

    /// Run ESPRIT on `signal` and return the estimated components.
    ///
    /// Implementation note: we use the correlation-matrix variant of ESPRIT
    /// (Paulraj, Roy & Kailath 1985) for superior numerical stability:
    /// 1. Build Hankel data matrix X.
    /// 2. Compute correlation matrix R = X^T X.
    /// 3. EVD of R via Jacobi to get eigenvectors sorted by descending eigenvalue.
    /// 4. Signal subspace U_s = top-k eigenvectors.
    /// 5. Form X1 = U_s[0..m-1, :], X2 = U_s[1..m, :].
    /// 6. Solve TLS: Φ = pinv(X1) · X2, take eigenvalues.
    pub fn estimate(&self, signal: &[f64]) -> SignalResult<PhaseEstResult> {
        let n = signal.len();
        if n < 4 {
            return Err(SignalError::InvalidArgument(format!(
                "Signal too short for ESPRIT: need ≥ 4 samples, got {n}"
            )));
        }
        if self.num_components == 0 {
            return Err(SignalError::InvalidArgument(
                "num_components must be ≥ 1".into(),
            ));
        }

        let l = self.hankel_l(n);
        let (x, nrows, ncols) = build_hankel(signal, l)?;

        let k = self.subspace_dim.min(nrows.min(ncols));
        if k < self.num_components {
            return Err(SignalError::InvalidArgument(format!(
                "subspace_dim {k} < num_components {}",
                self.num_components
            )));
        }

        // Build correlation matrix R = X^T X  (ncols × ncols).
        let mut r = vec![0.0f64; ncols * ncols];
        for i in 0..ncols {
            for j in 0..ncols {
                r[i * ncols + j] = (0..nrows)
                    .map(|row| x[row * ncols + i] * x[row * ncols + j])
                    .sum();
            }
        }

        // EVD via Jacobi — eigenvectors are returned sorted by eigenvalue ascending.
        let eigvecs = jacobi_evd_sorted(&mut r, ncols, self.max_iter);

        // Signal subspace: top-k eigenvectors (largest eigenvalues → last k in ascending sort).
        // eigvecs[i] is the eigenvector for the i-th eigenvalue (ascending order).
        // So signal subspace = eigvecs[ncols-k..ncols].
        let start = ncols.saturating_sub(k);
        let u_s: Vec<Vec<f64>> = eigvecs[start..].to_vec(); // length k, each has ncols elements

        let m = ncols; // row dimension of subspace matrix = ncols

        // Split: X1 = U_s[:m-1, :] rows 0..m-1, X2 = U_s[1..m, :]
        // u_s[col][row] = element at (row, col) of the subspace matrix.
        // We want X1 and X2 as ((m-1) × k) matrices (row-major).
        let m_sub = m - 1;
        let mut x1 = vec![0.0f64; m_sub * k];
        let mut x2 = vec![0.0f64; m_sub * k];
        for col in 0..k {
            for row in 0..m_sub {
                x1[row * k + col] = u_s[col][row];
                x2[row * k + col] = u_s[col][row + 1];
            }
        }

        // Φ = pinv(X1) · X2  (k × k)
        let pinv_x1 = pseudo_inverse(&x1, m_sub, k)?;
        let mut phi = vec![0.0f64; k * k];
        for i in 0..k {
            for j in 0..k {
                phi[i * k + j] = (0..m_sub)
                    .map(|m| pinv_x1[i * m_sub + m] * x2[m * k + j])
                    .sum();
            }
        }

        // Eigenvalues of Φ via Francis QR (handles complex conjugate pairs).
        let eigs = eigenvalues_francis_qr(&phi, k, self.max_iter * k);

        // ESPRIT eigenvalues sit on (near) the unit circle in ℂ.
        // Convert to normalised frequencies: f_norm = angle(λ) / (2π).
        // Select num_components eigenvalues closest to the unit circle with positive imaginary part.
        let mut freq_amp: Vec<(f64, f64)> = eigs
            .iter()
            .filter_map(|&(re, im)| {
                if im.abs() < 1e-6 {
                    // Real eigenvalue → DC or Nyquist; skip.
                    return None;
                }
                // Only upper half-plane (positive normalised frequencies).
                if im < 0.0 {
                    return None;
                }
                let angle = im.atan2(re);
                let freq_norm = angle / (2.0 * PI);
                if !freq_norm.is_finite() || freq_norm <= 1e-6 || freq_norm >= 0.5 {
                    return None;
                }
                let mag = (re * re + im * im).sqrt();
                let unit_dist = (mag - 1.0).abs();
                Some((freq_norm, unit_dist))
            })
            .collect();

        // Sort by distance from unit circle (signal poles first).
        freq_amp.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        freq_amp.truncate(self.num_components);

        if freq_amp.is_empty() {
            // Fallback: include all complex eigenvalues.
            let mut fallback: Vec<(f64, f64)> = eigs
                .iter()
                .filter_map(|&(re, im)| {
                    let angle = im.atan2(re);
                    let freq_norm = angle.abs() / (2.0 * PI);
                    if freq_norm <= 1e-8 || !freq_norm.is_finite() {
                        return None;
                    }
                    let mag = (re * re + im * im).sqrt();
                    let unit_dist = (mag - 1.0).abs();
                    Some((freq_norm, unit_dist))
                })
                .collect();
            fallback.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            fallback.truncate(self.num_components);
            freq_amp = fallback;
        }

        // Estimate amplitude and phase via DFT projection.
        let components: Vec<FrequencyComponent> = freq_amp
            .into_iter()
            .map(|(freq_norm, _)| {
                let freq_hz = freq_norm * self.fs;
                let (amp, phase) = estimate_amplitude_phase(signal, freq_norm);
                FrequencyComponent {
                    frequency: freq_hz,
                    amplitude: amp,
                    phase,
                }
            })
            .collect();

        Ok(PhaseEstResult::new(components, PhaseMethod::Esprit))
    }

    /// Given a signal and known frequencies in Hz, estimate the phase of each component.
    pub fn estimate_phases(&self, signal: &[f64], freqs_hz: &[f64]) -> Vec<f64> {
        freqs_hz
            .iter()
            .map(|&fhz| {
                let fn_ = fhz / self.fs;
                let (_, phase) = estimate_amplitude_phase(signal, fn_);
                phase
            })
            .collect()
    }
}

/// Estimate amplitude and phase of a single sinusoid at normalised frequency `f_norm`
/// (cycles per sample) using a DFT projection.
///
/// For x[n] = A·sin(2πf_norm·n + φ):
///   cos_sum ≈ N·A/2 · cos(φ − π/2) = N·A/2 · sin(φ)
///   sin_sum ≈ N·A/2 · sin(φ − π/2) = -N·A/2 · cos(φ)
/// → re = 2·cos_sum/N, im = 2·sin_sum/N
/// → amp = sqrt(re² + im²), phase = atan2(-im, re) + π/2 ... complex.
///
/// Simpler: use atan2(im, re) which gives the DFT phase of the cosine component.
pub(crate) fn estimate_amplitude_phase(signal: &[f64], f_norm: f64) -> (f64, f64) {
    let n = signal.len() as f64;
    let mut cos_sum = 0.0f64;
    let mut sin_sum = 0.0f64;
    for (i, &s) in signal.iter().enumerate() {
        let theta = 2.0 * PI * f_norm * i as f64;
        cos_sum += s * theta.cos();
        sin_sum += s * theta.sin();
    }
    let re = cos_sum * 2.0 / n;
    let im = sin_sum * 2.0 / n;
    let amp = (re * re + im * im).sqrt();
    // Phase recovery for x[n] = A·sin(2πf·n + φ):
    // re = A·sin(φ), im = A·cos(φ)  (see derivation in module docstring)
    // → φ = atan2(re, im)   [not atan2(im, re)]
    let phase = re.atan2(im);
    (amp, phase)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn make_sine(freq_hz: f64, phase: f64, amp: f64, fs: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| amp * (2.0 * PI * freq_hz / fs * i as f64 + phase).sin())
            .collect()
    }

    #[test]
    fn test_hankel_matrix_construction() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (x, nrows, ncols) = build_hankel(&signal, 2).expect("hankel ok");
        // L=2 → nrows=3, ncols=3
        assert_eq!(nrows, 3);
        assert_eq!(ncols, 3);
        // Row 0: [1,2,3], Row 1: [2,3,4], Row 2: [3,4,5]
        assert!((x[0] - 1.0).abs() < 1e-12);
        assert!((x[1] - 2.0).abs() < 1e-12);
        assert!((x[5] - 4.0).abs() < 1e-12);
        assert!((x[8] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_esprit_single_tone() {
        let fs = 1000.0;
        let n = 512;
        let sig = make_sine(100.0, 0.0, 1.0, fs, n);
        let est = EspritEstimator::new(1, fs);
        let result = est.estimate(&sig).expect("esprit ok");
        assert_eq!(result.components.len(), 1);
        let f = result.components[0].frequency;
        assert!((f - 100.0).abs() < 2.0, "Expected ~100 Hz, got {f:.2} Hz");
    }

    #[test]
    fn test_esprit_two_tones() {
        let fs = 1000.0;
        let n = 512;
        let mut sig = make_sine(100.0, 0.0, 1.0, fs, n);
        let s2 = make_sine(300.0, 0.5, 1.0, fs, n);
        for (a, b) in sig.iter_mut().zip(s2.iter()) {
            *a += b;
        }
        let mut est = EspritEstimator::new(2, fs);
        est.subspace_dim = 6;
        let result = est.estimate(&sig).expect("esprit 2-tone ok");
        assert_eq!(result.components.len(), 2);
        let freqs: Vec<f64> = result.components.iter().map(|c| c.frequency).collect();
        let has_100 = freqs.iter().any(|&f| (f - 100.0).abs() < 3.0);
        let has_300 = freqs.iter().any(|&f| (f - 300.0).abs() < 3.0);
        assert!(has_100, "Missing 100 Hz component, freqs={freqs:?}");
        assert!(has_300, "Missing 300 Hz component, freqs={freqs:?}");
    }

    #[test]
    fn test_esprit_phase_estimation() {
        let fs = 1000.0;
        let n = 512;
        let true_phase = 0.7f64; // radians
        let sig = make_sine(100.0, true_phase, 2.0, fs, n);

        // Use estimate_amplitude_phase with known frequency.
        let f_norm = 100.0 / fs;
        let (_amp, phase) = estimate_amplitude_phase(&sig, f_norm);
        let diff = (phase - true_phase).abs();
        let diff = diff.min((2.0 * PI - diff).abs());
        assert!(
            diff < 0.15,
            "Phase error {diff:.4} > 0.15 rad (got {phase:.4}, expected {true_phase:.4})"
        );
    }

    #[test]
    fn test_esprit_empty_signal() {
        let est = EspritEstimator::new(1, 1000.0);
        let result = est.estimate(&[]);
        assert!(result.is_err(), "Empty signal should return error");
    }

    #[test]
    fn test_esprit_frequency_resolution() {
        // Two tones 10 Hz apart with long signal — should be separable.
        let fs = 1000.0;
        let n = 1024;
        let mut sig = make_sine(200.0, 0.0, 1.0, fs, n);
        let s2 = make_sine(210.0, 0.0, 1.0, fs, n);
        for (a, b) in sig.iter_mut().zip(s2.iter()) {
            *a += b;
        }
        let mut est = EspritEstimator::new(2, fs);
        est.subspace_dim = 6;
        let result = est.estimate(&sig).expect("esprit resolution ok");
        assert_eq!(result.components.len(), 2);
        let f0 = result.components[0].frequency;
        let f1 = result.components[1].frequency;
        assert!(f0 > 190.0 && f0 < 225.0, "f0={f0:.2}");
        assert!(f1 > 190.0 && f1 < 225.0, "f1={f1:.2}");
        assert!((f1 - f0).abs() > 1.0, "|f1-f0|={:.2}", (f1 - f0).abs());
    }

    #[test]
    fn test_freq_component_ordering() {
        let comps = vec![
            FrequencyComponent {
                frequency: 300.0,
                amplitude: 1.0,
                phase: 0.0,
            },
            FrequencyComponent {
                frequency: 100.0,
                amplitude: 1.0,
                phase: 0.0,
            },
            FrequencyComponent {
                frequency: 200.0,
                amplitude: 1.0,
                phase: 0.0,
            },
        ];
        let result = PhaseEstResult::new(comps, PhaseMethod::Esprit);
        assert!((result.components[0].frequency - 100.0).abs() < 1e-10);
        assert!((result.components[1].frequency - 200.0).abs() < 1e-10);
        assert!((result.components[2].frequency - 300.0).abs() < 1e-10);
    }
}
