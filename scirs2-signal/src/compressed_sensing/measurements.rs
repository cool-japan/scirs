//! Measurement matrices for compressed sensing.
//!
//! This module provides various measurement (sensing) matrix constructions used
//! in compressed sensing to obtain linear measurements of a sparse signal.
//! Each matrix type satisfies (or approximately satisfies) the Restricted
//! Isometry Property (RIP) under appropriate parameter settings.
//!
//! # Measurement Matrices
//!
//! - [`GaussianMeasurement`]   – i.i.d. Gaussian entries, optimal RIP
//! - [`BernoulliMeasurement`]  – ±1 Rademacher entries, communication-friendly
//! - [`PartialDFT`]            – random rows of the DFT matrix, MRI-friendly
//! - [`ToeplitzMeasurement`]   – random Toeplitz structure, convolution-friendly
//!
//! # Analysis
//!
//! - [`rip_check_estimate`] – Monte Carlo estimate of the RIP constant δ_k
//! - [`coherence`]          – mutual coherence μ(Φ) of a measurement matrix
//!
//! # References
//!
//! - Candès & Tao (2006) – Near-optimal signal recovery from random projections
//! - Baraniuk et al. (2008) – A simple proof of the restricted isometry property
//! - Donoho & Huo (2001) – Uncertainty principles and ideal atomic decomposition
//!
//! Pure Rust, no unwrap(), snake_case naming throughout.

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView2};
use scirs2_core::numeric::Complex64;
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{RngExt, SeedableRng};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// GaussianMeasurement
// ---------------------------------------------------------------------------

/// Random Gaussian measurement matrix.
///
/// Each entry is drawn i.i.d. from N(0, 1/m) so that the columns have unit
/// expected norm.  This gives the best-known RIP constants.
///
/// # Example
///
/// ```
/// use scirs2_signal::compressed_sensing::measurements::GaussianMeasurement;
/// let phi = GaussianMeasurement::new(32, 128, 42).expect("operation should succeed");
/// let mat = phi.matrix();
/// assert_eq!(mat.dim(), (32, 128));
/// ```
pub struct GaussianMeasurement {
    matrix: Array2<f64>,
}

impl GaussianMeasurement {
    /// Construct a new `m × n` Gaussian measurement matrix.
    ///
    /// # Arguments
    ///
    /// * `m`    – Number of measurements (rows).
    /// * `n`    – Signal dimension (columns).
    /// * `seed` – RNG seed for reproducibility.
    ///
    /// # Errors
    ///
    /// Returns [`SignalError::ValueError`] if `m == 0` or `n == 0`.
    pub fn new(m: usize, n: usize, seed: u64) -> SignalResult<Self> {
        if m == 0 || n == 0 {
            return Err(SignalError::ValueError(
                "GaussianMeasurement: dimensions must be positive".to_string(),
            ));
        }

        let mut rng = StdRng::seed_from_u64(seed);
        let scale = 1.0 / (m as f64).sqrt();

        // Box-Muller transform for Gaussian samples
        let total = m * n;
        let mut data = Vec::with_capacity(total);
        let mut i = 0usize;
        while i < total {
            let u1: f64 = rng.random::<f64>();
            let u2: f64 = rng.random::<f64>();
            // Avoid log(0)
            let u1_safe = if u1 < 1e-300 { 1e-300 } else { u1 };
            let r = (-2.0 * u1_safe.ln()).sqrt();
            let theta = 2.0 * PI * u2;
            data.push(r * theta.cos() * scale);
            if i + 1 < total {
                data.push(r * theta.sin() * scale);
            }
            i += 2;
        }
        data.truncate(total);

        let matrix = Array2::from_shape_vec((m, n), data).map_err(|e| {
            SignalError::ComputationError(format!("GaussianMeasurement shape error: {e}"))
        })?;

        Ok(Self { matrix })
    }

    /// Return a reference to the underlying matrix.
    #[inline]
    pub fn matrix(&self) -> &Array2<f64> {
        &self.matrix
    }

    /// Consume the struct and return the matrix.
    #[inline]
    pub fn into_matrix(self) -> Array2<f64> {
        self.matrix
    }

    /// Compute the measurement vector `y = Φ x`.
    ///
    /// # Errors
    ///
    /// Returns [`SignalError::DimensionMismatch`] when dimensions are incompatible.
    pub fn measure(&self, x: &Array1<f64>) -> SignalResult<Array1<f64>> {
        let (m, n) = self.matrix.dim();
        if x.len() != n {
            return Err(SignalError::DimensionMismatch(format!(
                "GaussianMeasurement::measure: signal length {} != matrix columns {n}",
                x.len()
            )));
        }
        let y = self.matrix.dot(x);
        let _ = m;
        Ok(y)
    }
}

// ---------------------------------------------------------------------------
// BernoulliMeasurement
// ---------------------------------------------------------------------------

/// Bernoulli (Rademacher) ±1 measurement matrix.
///
/// Each entry is independently +1/√m or -1/√m with equal probability.
/// This matrix has the same RIP guarantees as the Gaussian design with the
/// advantage of integer (hardware-friendly) entries before scaling.
///
/// # Example
///
/// ```
/// use scirs2_signal::compressed_sensing::measurements::BernoulliMeasurement;
/// let phi = BernoulliMeasurement::new(32, 128, 7).expect("operation should succeed");
/// assert_eq!(phi.matrix().dim(), (32, 128));
/// ```
pub struct BernoulliMeasurement {
    matrix: Array2<f64>,
}

impl BernoulliMeasurement {
    /// Construct a new `m × n` Bernoulli (±1) measurement matrix.
    ///
    /// # Arguments
    ///
    /// * `m`    – Number of measurements.
    /// * `n`    – Signal dimension.
    /// * `seed` – RNG seed.
    ///
    /// # Errors
    ///
    /// Returns [`SignalError::ValueError`] if `m == 0` or `n == 0`.
    pub fn new(m: usize, n: usize, seed: u64) -> SignalResult<Self> {
        if m == 0 || n == 0 {
            return Err(SignalError::ValueError(
                "BernoulliMeasurement: dimensions must be positive".to_string(),
            ));
        }

        let mut rng = StdRng::seed_from_u64(seed);
        let scale = 1.0 / (m as f64).sqrt();

        let total = m * n;
        let data: Vec<f64> = (0..total)
            .map(|_| if rng.random::<bool>() { scale } else { -scale })
            .collect();

        let matrix = Array2::from_shape_vec((m, n), data).map_err(|e| {
            SignalError::ComputationError(format!("BernoulliMeasurement shape error: {e}"))
        })?;

        Ok(Self { matrix })
    }

    /// Return a reference to the underlying matrix.
    #[inline]
    pub fn matrix(&self) -> &Array2<f64> {
        &self.matrix
    }

    /// Consume the struct and return the matrix.
    #[inline]
    pub fn into_matrix(self) -> Array2<f64> {
        self.matrix
    }

    /// Compute the measurement vector `y = Φ x`.
    ///
    /// # Errors
    ///
    /// Returns [`SignalError::DimensionMismatch`] when dimensions are incompatible.
    pub fn measure(&self, x: &Array1<f64>) -> SignalResult<Array1<f64>> {
        let (_m, n) = self.matrix.dim();
        if x.len() != n {
            return Err(SignalError::DimensionMismatch(format!(
                "BernoulliMeasurement::measure: signal length {} != matrix columns {n}",
                x.len()
            )));
        }
        Ok(self.matrix.dot(x))
    }
}

// ---------------------------------------------------------------------------
// PartialDFT
// ---------------------------------------------------------------------------

/// Partial DFT measurement matrix.
///
/// Selects `m` rows uniformly at random from the `n × n` DFT matrix.
/// After normalization the resulting matrix satisfies the RIP with high
/// probability when `m = O(k log(n/k))` — this is the foundation of MRI
/// compressed sensing.
///
/// The matrix is returned in its real-valued representation as a stacked
/// `[2m × n]` matrix with rows ordered `[Re(row_0), Im(row_0), …]`.
/// A purely real-valued interface is provided via [`PartialDFT::measure_real`].
///
/// # Example
///
/// ```
/// use scirs2_signal::compressed_sensing::measurements::PartialDFT;
/// let phi = PartialDFT::new(16, 64, 0).expect("operation should succeed");
/// assert_eq!(phi.row_indices().len(), 16);
/// ```
pub struct PartialDFT {
    n: usize,
    row_indices: Vec<usize>,
}

impl PartialDFT {
    /// Construct a new partial DFT with `m` randomly chosen rows of size `n`.
    ///
    /// # Arguments
    ///
    /// * `m`    – Number of frequency measurements.
    /// * `n`    – Signal dimension (DFT size).
    /// * `seed` – RNG seed.
    ///
    /// # Errors
    ///
    /// Returns [`SignalError::ValueError`] if `m == 0`, `n == 0`, or `m > n`.
    pub fn new(m: usize, n: usize, seed: u64) -> SignalResult<Self> {
        if m == 0 || n == 0 {
            return Err(SignalError::ValueError(
                "PartialDFT: dimensions must be positive".to_string(),
            ));
        }
        if m > n {
            return Err(SignalError::ValueError(format!(
                "PartialDFT: m ({m}) cannot exceed n ({n})"
            )));
        }

        // Fisher-Yates partial shuffle to pick m distinct rows
        let mut indices: Vec<usize> = (0..n).collect();
        let mut rng = StdRng::seed_from_u64(seed);
        for i in 0..m {
            let j = i + (rng.random::<u64>() as usize % (n - i));
            indices.swap(i, j);
        }
        let row_indices = indices[..m].to_vec();

        Ok(Self { n, row_indices })
    }

    /// The DFT row indices selected.
    #[inline]
    pub fn row_indices(&self) -> &[usize] {
        &self.row_indices
    }

    /// Signal dimension.
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Number of measurements.
    #[inline]
    pub fn m(&self) -> usize {
        self.row_indices.len()
    }

    /// Compute the complex DFT measurements of a real signal `x` of length `n`.
    ///
    /// Returns a vector of `m` complex values Φx.
    ///
    /// # Errors
    ///
    /// Returns [`SignalError::DimensionMismatch`] if `x.len() != n`.
    pub fn measure_complex(&self, x: &Array1<f64>) -> SignalResult<Vec<Complex64>> {
        let n = self.n;
        if x.len() != n {
            return Err(SignalError::DimensionMismatch(format!(
                "PartialDFT::measure_complex: signal length {} != {n}",
                x.len()
            )));
        }

        let scale = 1.0 / (n as f64).sqrt();
        let result: Vec<Complex64> = self
            .row_indices
            .iter()
            .map(|&k| {
                let mut acc = Complex64::new(0.0, 0.0);
                for j in 0..n {
                    let angle = -2.0 * PI * (k as f64) * (j as f64) / (n as f64);
                    acc += x[j] * Complex64::new(angle.cos(), angle.sin());
                }
                acc * scale
            })
            .collect();

        Ok(result)
    }

    /// Compute measurements as a real-valued vector of length `2m` by
    /// stacking real and imaginary parts: `[Re y_0, Im y_0, …, Re y_{m-1}, Im y_{m-1}]`.
    ///
    /// # Errors
    ///
    /// Returns [`SignalError::DimensionMismatch`] if `x.len() != n`.
    pub fn measure_real(&self, x: &Array1<f64>) -> SignalResult<Array1<f64>> {
        let complex = self.measure_complex(x)?;
        let m = complex.len();
        let mut out = Array1::zeros(2 * m);
        for (i, c) in complex.iter().enumerate() {
            out[2 * i] = c.re;
            out[2 * i + 1] = c.im;
        }
        Ok(out)
    }

    /// Build the explicit real-valued sensing matrix of shape `(2m, n)`.
    ///
    /// Row `2k` = `Re(DFT_row_{row_indices[k]})`, row `2k+1` = Im(…).
    pub fn to_real_matrix(&self) -> Array2<f64> {
        let m = self.m();
        let n = self.n;
        let scale = 1.0 / (n as f64).sqrt();
        let mut mat = Array2::zeros((2 * m, n));

        for (i, &k) in self.row_indices.iter().enumerate() {
            for j in 0..n {
                let angle = -2.0 * PI * (k as f64) * (j as f64) / (n as f64);
                mat[[2 * i, j]] = angle.cos() * scale;
                mat[[2 * i + 1, j]] = angle.sin() * scale;
            }
        }
        mat
    }
}

// ---------------------------------------------------------------------------
// ToeplitzMeasurement
// ---------------------------------------------------------------------------

/// Random Toeplitz measurement matrix.
///
/// A Toeplitz matrix is defined by a single row and column vector.  A random
/// Toeplitz sensing matrix can implement measurements via convolution, making
/// it hardware-friendly.  The first row and first column are drawn i.i.d. from
/// N(0, 1/m).
///
/// # References
///
/// Haupt, Bajwa, Raz & Nowak (2010) – "Toeplitz compressed sensing matrices
/// with applications to sparse channel estimation"
///
/// # Example
///
/// ```
/// use scirs2_signal::compressed_sensing::measurements::ToeplitzMeasurement;
/// let phi = ToeplitzMeasurement::new(32, 64, 99).expect("operation should succeed");
/// assert_eq!(phi.matrix().dim(), (32, 64));
/// ```
pub struct ToeplitzMeasurement {
    matrix: Array2<f64>,
}

impl ToeplitzMeasurement {
    /// Construct a Toeplitz measurement matrix of size `m × n`.
    ///
    /// # Arguments
    ///
    /// * `m`    – Number of measurements (rows).
    /// * `n`    – Signal dimension (columns).
    /// * `seed` – RNG seed.
    ///
    /// # Errors
    ///
    /// Returns [`SignalError::ValueError`] if dimensions are zero.
    pub fn new(m: usize, n: usize, seed: u64) -> SignalResult<Self> {
        if m == 0 || n == 0 {
            return Err(SignalError::ValueError(
                "ToeplitzMeasurement: dimensions must be positive".to_string(),
            ));
        }

        let mut rng = StdRng::seed_from_u64(seed);
        let scale = 1.0 / (m as f64).sqrt();

        // We need m + n - 1 independent Gaussian values
        let len = m + n - 1;
        let mut gen_gaussian = move || -> f64 {
            let u1: f64 = rng.random::<f64>();
            let u2: f64 = rng.random::<f64>();
            let u1_safe = if u1 < 1e-300 { 1e-300 } else { u1 };
            (-2.0 * u1_safe.ln()).sqrt() * (2.0 * PI * u2).cos() * scale
        };

        let values: Vec<f64> = (0..len).map(|_| gen_gaussian()).collect();

        // Toeplitz matrix: mat[i][j] = values[abs(i - j + n - 1)] using a
        // standard parameterization: first column = values[0..m],
        // first row = values[0], values[m], values[m+1], …, values[m+n-2]
        //
        // We use the convention mat[i][j] = v[i - j + (n-1)] where the
        // extended vector v has length m + n - 1.
        let mut data = Vec::with_capacity(m * n);
        for i in 0..m {
            for j in 0..n {
                // index into extended vector: n - 1 + i - j
                let idx = (n - 1 + i).saturating_sub(j);
                let idx = idx.min(len - 1);
                data.push(values[idx]);
            }
        }

        let matrix = Array2::from_shape_vec((m, n), data).map_err(|e| {
            SignalError::ComputationError(format!("ToeplitzMeasurement shape error: {e}"))
        })?;

        Ok(Self { matrix })
    }

    /// Return a reference to the underlying matrix.
    #[inline]
    pub fn matrix(&self) -> &Array2<f64> {
        &self.matrix
    }

    /// Consume the struct and return the matrix.
    #[inline]
    pub fn into_matrix(self) -> Array2<f64> {
        self.matrix
    }

    /// Compute the measurement vector `y = Φ x`.
    ///
    /// # Errors
    ///
    /// Returns [`SignalError::DimensionMismatch`] when dimensions are incompatible.
    pub fn measure(&self, x: &Array1<f64>) -> SignalResult<Array1<f64>> {
        let (_m, n) = self.matrix.dim();
        if x.len() != n {
            return Err(SignalError::DimensionMismatch(format!(
                "ToeplitzMeasurement::measure: signal length {} != {n}",
                x.len()
            )));
        }
        Ok(self.matrix.dot(x))
    }
}

// ---------------------------------------------------------------------------
// RIP constant estimation
// ---------------------------------------------------------------------------

/// Estimate the Restricted Isometry Property (RIP) constant δ_k via Monte Carlo.
///
/// The RIP constant of order `k` for matrix `Φ` is the smallest δ such that
/// `(1 - δ) ‖x‖² ≤ ‖Φ x‖² ≤ (1 + δ) ‖x‖²` for all k-sparse vectors `x`.
///
/// This function estimates δ_k by evaluating `‖Φ x‖² / ‖x‖²` for `n_trials`
/// randomly generated k-sparse vectors and returning `max(|ratio - 1|)`.
///
/// # Arguments
///
/// * `phi`      – The measurement matrix (m × n).
/// * `sparsity` – Sparsity level k.
/// * `n_trials` – Number of Monte Carlo trials (higher = more accurate).
/// * `seed`     – RNG seed.
///
/// # Returns
///
/// Estimated RIP constant in [0, ∞).  Values < 1 indicate the RIP holds.
///
/// # Errors
///
/// Returns [`SignalError::ValueError`] for invalid parameters.
///
/// # Example
///
/// ```
/// use scirs2_signal::compressed_sensing::measurements::{GaussianMeasurement, rip_check_estimate};
/// let phi = GaussianMeasurement::new(48, 128, 1).expect("operation should succeed");
/// let delta = rip_check_estimate(phi.matrix(), 5, 200, 2).expect("operation should succeed");
/// assert!(delta < 1.0, "RIP constant should be < 1 for Gaussian measurements");
/// ```
pub fn rip_check_estimate(
    phi: &Array2<f64>,
    sparsity: usize,
    n_trials: usize,
    seed: u64,
) -> SignalResult<f64> {
    let (m, n) = phi.dim();
    if m == 0 || n == 0 {
        return Err(SignalError::ValueError(
            "rip_check_estimate: matrix must be non-empty".to_string(),
        ));
    }
    if sparsity == 0 {
        return Err(SignalError::ValueError(
            "rip_check_estimate: sparsity must be positive".to_string(),
        ));
    }
    if sparsity > n {
        return Err(SignalError::ValueError(format!(
            "rip_check_estimate: sparsity {sparsity} > signal dimension {n}"
        )));
    }
    if n_trials == 0 {
        return Err(SignalError::ValueError(
            "rip_check_estimate: n_trials must be positive".to_string(),
        ));
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut max_deviation: f64 = 0.0;

    for _ in 0..n_trials {
        // Choose k distinct support indices
        let mut indices: Vec<usize> = (0..n).collect();
        for i in 0..sparsity {
            let j = i + (rng.random::<u64>() as usize % (n - i));
            indices.swap(i, j);
        }
        let support = &indices[..sparsity];

        // Build a random k-sparse vector on that support
        let mut x = Array1::<f64>::zeros(n);
        let mut x_norm_sq = 0.0f64;
        for &idx in support.iter() {
            let val = {
                let u1: f64 = rng.random::<f64>();
                let u2: f64 = rng.random::<f64>();
                let u1_safe = if u1 < 1e-300 { 1e-300 } else { u1 };
                (-2.0 * u1_safe.ln()).sqrt() * (2.0 * PI * u2).cos()
            };
            x[idx] = val;
            x_norm_sq += val * val;
        }

        if x_norm_sq < 1e-14 {
            continue;
        }

        // Compute Φx and its squared norm
        let y = phi.dot(&x);
        let y_norm_sq: f64 = y.iter().map(|&v| v * v).sum();

        let ratio = y_norm_sq / x_norm_sq;
        let deviation = (ratio - 1.0).abs();
        if deviation > max_deviation {
            max_deviation = deviation;
        }
    }

    Ok(max_deviation)
}

// ---------------------------------------------------------------------------
// Mutual Coherence
// ---------------------------------------------------------------------------

/// Compute the mutual coherence μ(Φ) of a measurement matrix.
///
/// The mutual coherence is defined as
///
/// μ(Φ) = max_{i ≠ j} |φ_i^T φ_j| / (‖φ_i‖ ‖φ_j‖)
///
/// where φ_i are the columns of Φ.  A small coherence implies the columns are
/// nearly orthogonal and guarantees exact recovery by OMP for signals of
/// sparsity `k < (1 + 1/μ)/2`.
///
/// # Arguments
///
/// * `phi` – Measurement matrix (m × n).  Each column is an atom.
///
/// # Returns
///
/// Mutual coherence in [0, 1].
///
/// # Errors
///
/// Returns [`SignalError::ValueError`] if the matrix has fewer than 2 columns.
///
/// # Example
///
/// ```
/// use scirs2_signal::compressed_sensing::measurements::{GaussianMeasurement, coherence};
/// let phi = GaussianMeasurement::new(64, 128, 5).expect("operation should succeed");
/// let mu = coherence(phi.matrix()).expect("operation should succeed");
/// // Gaussian matrices typically have low coherence
/// assert!(mu < 1.0);
/// ```
pub fn coherence(phi: &Array2<f64>) -> SignalResult<f64> {
    let (m, n) = phi.dim();
    if m == 0 {
        return Err(SignalError::ValueError(
            "coherence: matrix has no rows".to_string(),
        ));
    }
    if n < 2 {
        return Err(SignalError::ValueError(
            "coherence: matrix must have at least 2 columns".to_string(),
        ));
    }

    // Compute column norms
    let col_norms: Vec<f64> = (0..n)
        .map(|j| {
            let col = phi.column(j);
            col.iter().map(|&v| v * v).sum::<f64>().sqrt()
        })
        .collect();

    let mut max_coherence: f64 = 0.0;

    for i in 0..n {
        let ni = col_norms[i];
        if ni < 1e-14 {
            continue;
        }
        for j in (i + 1)..n {
            let nj = col_norms[j];
            if nj < 1e-14 {
                continue;
            }
            // dot product of columns i and j
            let dot: f64 = phi
                .column(i)
                .iter()
                .zip(phi.column(j).iter())
                .map(|(&a, &b)| a * b)
                .sum();
            let c = (dot / (ni * nj)).abs();
            if c > max_coherence {
                max_coherence = c;
            }
        }
    }

    Ok(max_coherence)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_measurement_shape() {
        let phi = GaussianMeasurement::new(32, 128, 1).expect("should construct");
        assert_eq!(phi.matrix().dim(), (32, 128));
    }

    #[test]
    fn test_gaussian_measure() {
        let phi = GaussianMeasurement::new(16, 64, 2).expect("should construct");
        let x = Array1::ones(64);
        let y = phi.measure(&x).expect("should measure");
        assert_eq!(y.len(), 16);
    }

    #[test]
    fn test_bernoulli_measurement_values() {
        let phi = BernoulliMeasurement::new(8, 16, 3).expect("should construct");
        let scale = 1.0 / 8f64.sqrt();
        for &v in phi.matrix().iter() {
            assert!((v - scale).abs() < 1e-12 || (v + scale).abs() < 1e-12);
        }
    }

    #[test]
    fn test_partial_dft_row_count() {
        let phi = PartialDFT::new(8, 32, 0).expect("should construct");
        assert_eq!(phi.m(), 8);
        assert_eq!(phi.n(), 32);
    }

    #[test]
    fn test_partial_dft_measure_real_length() {
        let phi = PartialDFT::new(4, 16, 1).expect("should construct");
        let x = Array1::zeros(16);
        let y = phi.measure_real(&x).expect("should measure");
        assert_eq!(y.len(), 8);
    }

    #[test]
    fn test_toeplitz_measurement_shape() {
        let phi = ToeplitzMeasurement::new(12, 24, 7).expect("should construct");
        assert_eq!(phi.matrix().dim(), (12, 24));
    }

    #[test]
    fn test_rip_gaussian() {
        let phi = GaussianMeasurement::new(48, 64, 99).expect("should construct");
        let delta = rip_check_estimate(phi.matrix(), 3, 100, 1).expect("should estimate");
        assert!(delta < 1.0, "RIP constant {delta} should be < 1");
    }

    #[test]
    fn test_coherence_unit_matrix() {
        // Identity columns have zero pairwise inner products
        let eye = Array2::eye(4);
        let mu = coherence(&eye).expect("should compute");
        assert!(mu < 1e-12);
    }
}
