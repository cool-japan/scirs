//! Fourier feature maps for shift-invariant kernel approximation.
//!
//! Implements Rahimi & Recht (2007) random Fourier features (RFF):
//! `z(x) = sqrt(2/D) * [cos(ω_1^T x + b_1), ..., cos(ω_D^T x + b_D)]`
//! where ω_i are sampled from the spectral density of the target kernel.
//!
//! # References
//! - Rahimi, A. & Recht, B. (2007). Random features for large-scale kernel machines. NIPS.
//! - Yu, F. X. et al. (2016). Orthogonal Random Features. NeurIPS.

use crate::error::InterpolateError;
use scirs2_core::ndarray::{Array1, Array2, ArrayView2};

// ─── Internal RNG ────────────────────────────────────────────────────────────

/// Linear Congruential Generator (Knuth 64-bit multiplicative).
/// Exported for use in sibling modules.
pub(super) struct Lcg64 {
    state: u64,
}

impl Lcg64 {
    pub(super) fn new(seed: u64) -> Self {
        // Ensure non-zero state to avoid degenerate cycle.
        Self {
            state: seed.wrapping_add(1).max(1),
        }
    }

    pub(super) fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    /// Uniform float in [0, 1).
    pub(super) fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal via Box-Muller transform.
    pub(super) fn next_normal(&mut self) -> f64 {
        loop {
            let u1 = self.next_f64();
            if u1 > 1e-300 {
                let u2 = self.next_f64();
                return (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            }
        }
    }

    /// Cauchy sample: ratio of two normals.
    pub(super) fn next_cauchy(&mut self) -> f64 {
        loop {
            let b = self.next_normal();
            if b.abs() > 1e-15 {
                return self.next_normal() / b;
            }
        }
    }

    /// Student-t sample with `nu` degrees of freedom.
    /// Uses: t(ν) = Z / sqrt(χ²(ν)/ν) where Z ~ N(0,1) and χ²(ν) = sum of ν normals squared.
    pub(super) fn next_student_t(&mut self, nu: usize) -> f64 {
        let z = self.next_normal();
        let chi2: f64 = (0..nu).map(|_| self.next_normal().powi(2)).sum();
        let chi = (chi2 / nu as f64).sqrt().max(1e-300);
        z / chi
    }
}

// ─── RffKernel ───────────────────────────────────────────────────────────────

/// Shift-invariant kernel type for Random Fourier Feature approximation.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum RffKernel {
    /// Gaussian (RBF) kernel: K(x,y) = exp(-||x-y||²/(2l²)).
    /// Spectral density: p(ω) = N(0, I/l²).
    Gaussian {
        /// Length-scale parameter l > 0.
        length_scale: f64,
    },
    /// Laplacian kernel: K(x,y) = exp(-||x-y||₁/l).
    /// Spectral density: product of Cauchy distributions scaled by 1/l.
    Laplacian {
        /// Length-scale parameter l > 0.
        length_scale: f64,
    },
    /// Matérn 3/2 kernel.
    /// Spectral density: scaled Student-t with ν=3 degrees of freedom.
    Matern32 {
        /// Length-scale parameter l > 0.
        length_scale: f64,
    },
    /// Matérn 5/2 kernel.
    /// Spectral density: scaled Student-t with ν=5 degrees of freedom.
    Matern52 {
        /// Length-scale parameter l > 0.
        length_scale: f64,
    },
}

impl RffKernel {
    /// Length-scale for this kernel variant.
    pub fn length_scale(&self) -> f64 {
        match self {
            RffKernel::Gaussian { length_scale }
            | RffKernel::Laplacian { length_scale }
            | RffKernel::Matern32 { length_scale }
            | RffKernel::Matern52 { length_scale } => *length_scale,
        }
    }

    /// Sample one frequency component ω_k (scalar for one input dimension) from
    /// the spectral density of this kernel.
    pub(super) fn sample_omega_component(&self, rng: &mut Lcg64) -> f64 {
        let ls = self.length_scale().max(1e-300);
        match self {
            RffKernel::Gaussian { .. } => rng.next_normal() / ls,
            RffKernel::Laplacian { .. } => rng.next_cauchy() / ls,
            RffKernel::Matern32 { .. } => rng.next_student_t(3) / ls,
            RffKernel::Matern52 { .. } => rng.next_student_t(5) / ls,
        }
    }
}

// ─── FourierFeatureMap ───────────────────────────────────────────────────────

/// Random Fourier Feature (RFF) map using the Rahimi-Recht construction.
///
/// Maps `x ∈ R^d_in` to `z(x) ∈ R^d_out` via:
/// `z(x)_j = sqrt(2/D) * cos(Ω_j · x + b_j)`,
/// where `Ω ∈ R^(D×d)` and `b ∈ R^D` are drawn at construction time.
///
/// Then `E[z(x)ᵀz(y)] = K(x, y)` for the target kernel K.
///
/// # Example
/// ```rust,ignore
/// use scirs2_interpolate::random_features::feature_map::{FourierFeatureMap, RffKernel};
/// use scirs2_core::ndarray::Array2;
///
/// let map = FourierFeatureMap::new(
///     RffKernel::Gaussian { length_scale: 1.0 },
///     2,   // input dimensions
///     500, // output features D
///     42,  // seed
/// );
/// let x = Array2::zeros((10, 2));
/// let z = map.transform(&x.view()); // shape (10, 500)
/// ```
#[derive(Debug, Clone)]
pub struct FourierFeatureMap {
    /// Frequency matrix, shape `[D, d_in]`.
    omega: Array2<f64>,
    /// Phase bias vector, shape `[D]`.
    bias: Array1<f64>,
    /// The kernel this map approximates.
    pub kernel: RffKernel,
    /// `sqrt(2/D)` scaling factor.
    scale: f64,
    /// Number of input dimensions.
    pub d_in: usize,
    /// Number of output features (D).
    pub d_out: usize,
}

impl FourierFeatureMap {
    /// Construct a `FourierFeatureMap` from pre-computed matrices.
    ///
    /// Used internally by [`crate::random_features::orthogonal::OrthogonalFourierFeatureMap`]
    /// to inject an orthogonalised frequency matrix without re-sampling.
    pub(super) fn from_parts(
        kernel: RffKernel,
        d_in: usize,
        d_out: usize,
        omega: Array2<f64>,
        bias: Array1<f64>,
        scale: f64,
    ) -> Self {
        Self {
            omega,
            bias,
            kernel,
            scale,
            d_in,
            d_out,
        }
    }

    /// Construct a new `FourierFeatureMap`.
    ///
    /// # Arguments
    /// * `kernel` — kernel type and length-scale
    /// * `d_in`   — number of input dimensions
    /// * `d_out`  — number of random features D (larger = better approximation)
    /// * `seed`   — RNG seed for reproducibility
    ///
    /// # Panics
    /// Panics if `d_in == 0` or `d_out == 0`.
    pub fn new(kernel: RffKernel, d_in: usize, d_out: usize, seed: u64) -> Self {
        assert!(d_in > 0, "d_in must be > 0");
        assert!(d_out > 0, "d_out must be > 0");

        let mut rng = Lcg64::new(seed);

        // Sample Ω (D × d_in)
        let omega_data: Vec<f64> = (0..d_out * d_in)
            .map(|_| kernel.sample_omega_component(&mut rng))
            .collect();
        let omega = Array2::from_shape_vec((d_out, d_in), omega_data)
            .expect("shape is consistent by construction");

        // Sample b ~ Uniform[0, 2π]
        let bias_data: Vec<f64> = (0..d_out)
            .map(|_| rng.next_f64() * 2.0 * std::f64::consts::PI)
            .collect();
        let bias = Array1::from_vec(bias_data);

        let scale = (2.0 / d_out as f64).sqrt();

        Self {
            omega,
            bias,
            kernel,
            scale,
            d_in,
            d_out,
        }
    }

    /// Transform input `x` (shape `[n, d_in]`) into features (shape `[n, D]`).
    ///
    /// `z[i, j] = scale * cos(Ω[j, :] · x[i, :] + b[j])`
    ///
    /// # Errors
    /// Returns an error if `x.ncols() != self.d_in`.
    pub fn transform(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>, InterpolateError> {
        let n = x.nrows();
        let d = x.ncols();
        if d != self.d_in {
            return Err(InterpolateError::DimensionMismatch(format!(
                "FourierFeatureMap expects {d_in} input dimensions, got {d}",
                d_in = self.d_in,
            )));
        }

        let mut z = Array2::<f64>::zeros((n, self.d_out));
        for i in 0..n {
            let xi = x.row(i);
            for j in 0..self.d_out {
                let omega_j = self.omega.row(j);
                let dot: f64 = omega_j.iter().zip(xi.iter()).map(|(w, xv)| w * xv).sum();
                z[(i, j)] = self.scale * (dot + self.bias[j]).cos();
            }
        }
        Ok(z)
    }

    /// Approximate the kernel value K(x1, x2) ≈ z(x1)ᵀz(y2).
    ///
    /// # Errors
    /// Returns an error if either slice has length != `d_in`.
    pub fn kernel_approx(&self, x1: &[f64], x2: &[f64]) -> Result<f64, InterpolateError> {
        if x1.len() != self.d_in || x2.len() != self.d_in {
            return Err(InterpolateError::DimensionMismatch(format!(
                "kernel_approx expects {d_in} dimensions",
                d_in = self.d_in,
            )));
        }
        let mut result = 0.0f64;
        for j in 0..self.d_out {
            let omega_j = self.omega.row(j);
            let dot1: f64 = omega_j.iter().zip(x1.iter()).map(|(w, v)| w * v).sum();
            let dot2: f64 = omega_j.iter().zip(x2.iter()).map(|(w, v)| w * v).sum();
            result += (dot1 + self.bias[j]).cos() * (dot2 + self.bias[j]).cos();
        }
        Ok(2.0 / self.d_out as f64 * result)
    }

    /// Access the frequency matrix Ω (shape `[D, d_in]`).
    pub fn omega(&self) -> &Array2<f64> {
        &self.omega
    }

    /// Access the bias vector b (shape `[D]`).
    pub fn bias(&self) -> &Array1<f64> {
        &self.bias
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_rff_output_shape() {
        let map = FourierFeatureMap::new(RffKernel::Gaussian { length_scale: 1.0 }, 3, 64, 1);
        let x = Array2::<f64>::zeros((5, 3));
        let z = map.transform(&x.view()).expect("transform");
        assert_eq!(z.shape(), &[5, 64]);
    }

    #[test]
    fn test_gaussian_kernel_approx_close() {
        // For D=1000, E[z(x)ᵀz(y)] ≈ exp(-||x-y||²/2)
        let map = FourierFeatureMap::new(RffKernel::Gaussian { length_scale: 1.0 }, 2, 1000, 77);
        let x1 = [1.0f64, 0.0];
        let x2 = [0.0f64, 1.0];
        let true_k = (-1.0f64).exp(); // exp(-||x1-x2||²/(2·1²)) = exp(-1)
        let approx_k = map.kernel_approx(&x1, &x2).expect("approx");
        let err = (approx_k - true_k).abs();
        assert!(err < 0.1, "error={err:.4}, expected < 0.1 for D=1000");
    }

    #[test]
    fn test_all_kernel_types_output_finite() {
        let x = Array2::from_shape_fn((4, 2), |(i, j)| (i + j) as f64 * 0.2);
        for kernel in [
            RffKernel::Gaussian { length_scale: 1.0 },
            RffKernel::Laplacian { length_scale: 1.0 },
            RffKernel::Matern32 { length_scale: 1.0 },
            RffKernel::Matern52 { length_scale: 1.0 },
        ] {
            let map = FourierFeatureMap::new(kernel, 2, 32, 0);
            let z = map.transform(&x.view()).expect("transform");
            assert!(
                z.iter().all(|v| v.is_finite()),
                "all features must be finite"
            );
        }
    }
}
