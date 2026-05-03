//! Orthogonal Random Features (ORF) for improved approximation variance.
//!
//! Instead of drawing Ω rows i.i.d. from p(ω), ORF constructs an orthonormal
//! set of frequency vectors by QR-decomposing a Gaussian matrix and then
//! re-scaling each row by an independently drawn chi-distributed length so
//! that the marginal of each row still matches the target spectral density.
//!
//! This reduces the variance of the Monte-Carlo kernel estimator compared to
//! standard RFF at no extra prediction cost.
//!
//! # References
//! - Yu, F. X. et al. (2016). Orthogonal Random Features. NeurIPS.
//! - Choromanski, K. et al. (2017). The Geometry of Random Features. AISTATS.

use crate::error::InterpolateError;
use crate::random_features::feature_map::{FourierFeatureMap, Lcg64, RffKernel};
use scirs2_core::ndarray::{Array1, Array2, ArrayView2};

// ─── QR decomposition (Gram-Schmidt, stable) ─────────────────────────────────

/// Thin QR decomposition of a matrix `a` (rows × cols, rows ≥ cols).
/// Returns the Q factor (rows × cols), orthonormal columns.
fn gram_schmidt(a: &Array2<f64>) -> Array2<f64> {
    let m = a.nrows();
    let n = a.ncols();
    let mut q = Array2::<f64>::zeros((m, n));

    for j in 0..n {
        // Start with column j of a.
        let mut v: Vec<f64> = a.column(j).to_vec();

        // Subtract projections onto previous q columns.
        for k in 0..j {
            let qk: Vec<f64> = q.column(k).to_vec();
            let dot: f64 = v.iter().zip(qk.iter()).map(|(vi, qi)| vi * qi).sum();
            for i in 0..m {
                v[i] -= dot * qk[i];
            }
        }

        // Normalize.
        let norm: f64 = v.iter().map(|vi| vi * vi).sum::<f64>().sqrt();
        if norm > 1e-12 {
            for i in 0..m {
                q[(i, j)] = v[i] / norm;
            }
        }
        // If norm ≈ 0 (linearly dependent), leave column zero (won't happen
        // in practice for a random Gaussian matrix).
    }
    q
}

// ─── OrthogonalFourierFeatureMap ─────────────────────────────────────────────

/// Orthogonal Random Fourier Feature map.
///
/// Constructs frequency vectors by orthogonalising a Gaussian matrix (QR) and
/// re-scaling each row by a chi-distributed random length drawn to match the
/// target spectral density marginal.  This reduces estimator variance compared
/// to standard RFF while keeping the same API.
///
/// When `d_out > d_in`, multiple independent orthogonal blocks are stacked to
/// reach the requested number of features.
///
/// # Example
/// ```rust,ignore
/// use scirs2_interpolate::random_features::orthogonal::OrthogonalFourierFeatureMap;
/// use scirs2_interpolate::random_features::feature_map::RffKernel;
/// use scirs2_core::ndarray::Array2;
///
/// let map = OrthogonalFourierFeatureMap::new(
///     RffKernel::Gaussian { length_scale: 1.0 },
///     3,    // d_in
///     200,  // d_out
///     42,   // seed
/// );
/// let x = Array2::<f64>::zeros((10, 3));
/// let z = map.transform(&x.view()).expect("transform");
/// assert_eq!(z.shape(), &[10, 200]);
/// ```
#[derive(Debug, Clone)]
pub struct OrthogonalFourierFeatureMap {
    /// The underlying standard RFF map is stored for the `transform` call;
    /// the omega matrix is overwritten with the orthogonalised version.
    base: FourierFeatureMap,
}

impl OrthogonalFourierFeatureMap {
    /// Construct a new `OrthogonalFourierFeatureMap`.
    ///
    /// # Arguments
    /// * `kernel`  — kernel type and length-scale
    /// * `d_in`    — number of input dimensions
    /// * `d_out`   — total number of output features D (will be rounded to a
    ///               multiple of `d_in` upward)
    /// * `seed`    — RNG seed
    pub fn new(kernel: RffKernel, d_in: usize, d_out: usize, seed: u64) -> Self {
        assert!(d_in > 0, "d_in must be > 0");
        assert!(d_out > 0, "d_out must be > 0");

        let mut rng = Lcg64::new(seed);
        let ls = kernel.length_scale().max(1e-300);

        // Number of blocks needed.
        let n_blocks = (d_out + d_in - 1) / d_in; // ceiling division
        let d_out_padded = n_blocks * d_in;

        // Build padded omega (D_padded × d_in).
        let mut omega_rows: Vec<Vec<f64>> = Vec::with_capacity(d_out_padded);

        for _ in 0..n_blocks {
            // Sample a d_in × d_in Gaussian matrix.
            let g = Array2::from_shape_fn((d_in, d_in), |_| rng.next_normal());

            // Orthogonalise columns → Q (d_in × d_in).
            let q = gram_schmidt(&g);

            // For each row of Q (= each frequency direction), sample a scale
            // from the chi distribution to match the spectral density marginal.
            // For Gaussian kernel: each component of ω is N(0, 1/l²), so
            // ||ω||² ~ chi²(d_in)/l². We sample chi(d_in) and scale.
            // For other kernels we fall back to row-wise scaling derived from the
            // norm of an i.i.d. draw from the correct density.
            for qi_idx in 0..d_in {
                let q_row: Vec<f64> = (0..d_in).map(|j| q[(j, qi_idx)]).collect();
                let scale = sample_row_scale(&kernel, &mut rng, d_in, ls);
                let omega_row: Vec<f64> = q_row.iter().map(|v| v * scale).collect();
                omega_rows.push(omega_row);
            }
        }

        // Truncate to d_out rows.
        omega_rows.truncate(d_out);

        // Build the bias vector.
        let bias_data: Vec<f64> = (0..d_out)
            .map(|_| rng.next_f64() * 2.0 * std::f64::consts::PI)
            .collect();

        // Reconstruct the feature map with the orthogonal omega.
        // We build it via the public constructor and then use a "side-door"
        // reconstruction: build a standard map and swap the omega.
        // Since FourierFeatureMap's fields are private we build the orthogonal
        // map by storing omega/bias in flat form and delegating transform.
        let omega_flat: Vec<f64> = omega_rows.iter().flat_map(|r| r.iter().copied()).collect();
        let omega = Array2::from_shape_vec((d_out, d_in), omega_flat).expect("shape consistent");
        let bias = Array1::from_vec(bias_data);
        let scale = (2.0 / d_out as f64).sqrt();

        // Build via the public constructor (uses own omega), then replace
        // internal state via the accessor-based path.
        // We directly construct our ORF store.
        let base = FourierFeatureMap::from_parts(kernel, d_in, d_out, omega, bias, scale);

        Self { base }
    }

    /// Transform input `x` (shape `[n, d_in]`) to features (shape `[n, d_out]`).
    ///
    /// # Errors
    /// Returns an error if `x.ncols() != self.d_in`.
    pub fn transform(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>, InterpolateError> {
        self.base.transform(x)
    }

    /// Approximate the kernel value K(x1, x2) ≈ z(x1)ᵀz(y2).
    pub fn kernel_approx(&self, x1: &[f64], x2: &[f64]) -> Result<f64, InterpolateError> {
        self.base.kernel_approx(x1, x2)
    }

    /// Number of input dimensions.
    pub fn d_in(&self) -> usize {
        self.base.d_in
    }

    /// Number of output features.
    pub fn d_out(&self) -> usize {
        self.base.d_out
    }
}

/// Sample the row-wise scale factor for orthogonal frequency vectors.
///
/// For the Gaussian kernel, ||ω||² ~ chi²(d)/l², so sample chi(d)/l.
/// For other kernels, approximate by drawing one full frequency vector and
/// returning its norm; the direction is already fixed by the ORF Q matrix.
fn sample_row_scale(kernel: &RffKernel, rng: &mut Lcg64, d_in: usize, ls: f64) -> f64 {
    match kernel {
        RffKernel::Gaussian { .. } => {
            // chi(d_in) / ls
            let chi2: f64 = (0..d_in).map(|_| rng.next_normal().powi(2)).sum();
            chi2.sqrt() / ls
        }
        RffKernel::Laplacian { .. } => {
            // Use product-of-Cauchy norms — approximate chi for Laplacian:
            // sample a d_in-dim Cauchy vector and take its norm.
            let norm2: f64 = (0..d_in).map(|_| rng.next_cauchy().powi(2)).sum();
            norm2.sqrt() / ls
        }
        RffKernel::Matern32 { .. } => {
            let norm2: f64 = (0..d_in).map(|_| rng.next_student_t(3).powi(2)).sum();
            norm2.sqrt() / ls
        }
        RffKernel::Matern52 { .. } => {
            let norm2: f64 = (0..d_in).map(|_| rng.next_student_t(5).powi(2)).sum();
            norm2.sqrt() / ls
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_orf_output_shape() {
        let map =
            OrthogonalFourierFeatureMap::new(RffKernel::Gaussian { length_scale: 1.0 }, 3, 64, 99);
        let x = Array2::<f64>::zeros((5, 3));
        let z = map.transform(&x.view()).expect("transform");
        assert_eq!(z.shape(), &[5, 64]);
    }

    #[test]
    fn test_orf_kernel_approx_reasonable() {
        let map =
            OrthogonalFourierFeatureMap::new(RffKernel::Gaussian { length_scale: 1.0 }, 2, 512, 7);
        let x1 = [1.0f64, 0.0];
        let x2 = [0.0f64, 1.0];
        let true_k = (-1.0f64).exp();
        let approx_k = map.kernel_approx(&x1, &x2).expect("approx");
        let err = (approx_k - true_k).abs();
        assert!(err < 0.15, "ORF error={err:.4}, expected < 0.15 for D=512");
    }

    #[test]
    fn test_orf_all_features_finite() {
        for kernel in [
            RffKernel::Gaussian { length_scale: 1.0 },
            RffKernel::Laplacian { length_scale: 1.0 },
            RffKernel::Matern32 { length_scale: 1.0 },
            RffKernel::Matern52 { length_scale: 1.0 },
        ] {
            let x = Array2::from_shape_fn((4, 2), |(i, j)| (i + j) as f64 * 0.2);
            let map = OrthogonalFourierFeatureMap::new(kernel, 2, 32, 0);
            let z = map.transform(&x.view()).expect("transform");
            assert!(
                z.iter().all(|v| v.is_finite()),
                "ORF: all features must be finite"
            );
        }
    }

    #[test]
    fn test_orf_d_out_not_multiple_of_d_in() {
        // d_out=7, d_in=3 → padded to 9, then truncated to 7.
        let map =
            OrthogonalFourierFeatureMap::new(RffKernel::Gaussian { length_scale: 1.0 }, 3, 7, 5);
        let x = Array2::<f64>::zeros((2, 3));
        let z = map.transform(&x.view()).expect("transform");
        assert_eq!(z.shape(), &[2, 7]);
    }
}
