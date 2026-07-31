// SOBI - Second Order Blind Identification
//
// Temporal decorrelation-based BSS using multiple lagged covariance matrices.
// Reference: Belouchrani et al. (1997). "A blind source separation technique
// using second-order statistics." IEEE Trans. Signal Processing, 45(2), 434-444.

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{s, Array1, Array2, Axis};
use scirs2_linalg::{eigh, svd};

/// Result from the SOBI algorithm
#[derive(Debug, Clone)]
pub struct SOBIResult {
    /// Unmixing matrix W (n_components x n_channels)
    pub unmixing_matrix: Array2<f64>,
    /// Separated sources (n_components x n_samples)
    pub sources: Array2<f64>,
    /// Mixing matrix A ≈ inv(W) (n_channels x n_components)
    pub mixing_matrix: Array2<f64>,
    /// Time lags used for covariance estimation
    pub time_lags: Vec<usize>,
    /// Whether the algorithm converged
    pub converged: bool,
    /// Number of Jacobi sweeps performed
    pub n_iterations: usize,
}

/// Lag-covariance estimator for SOBI.
///
/// Computes normalised covariance matrices at multiple time lags for a
/// whitened multichannel signal.
pub struct LagCovariance;

impl LagCovariance {
    /// Compute covariance matrices at a set of lags.
    ///
    /// For whitened signal z of shape (n, T), the lag-τ covariance is:
    ///   `R_τ[i,j] = (1/T) Σ_{t=0}^{T-τ-1} z[i,t] z[j, t+τ]`
    ///
    /// The matrices are symmetrised: R_τ = (R_τ + R_τ^T) / 2.
    ///
    /// # Arguments
    ///
    /// * `z`    - Whitened data (n_components x n_samples)
    /// * `lags` - Time lags τ > 0
    ///
    /// # Returns
    ///
    /// Vector of (lag, R_lag) pairs.
    pub fn compute(z: &Array2<f64>, lags: &[usize]) -> SignalResult<Vec<(usize, Array2<f64>)>> {
        let (n, t) = z.dim();
        if t == 0 {
            return Err(SignalError::ValueError("Data has zero samples".to_string()));
        }
        let mut result = Vec::with_capacity(lags.len());

        for &lag in lags {
            if lag == 0 {
                return Err(SignalError::ValueError(
                    "Time lag must be strictly positive for SOBI".to_string(),
                ));
            }
            if lag >= t {
                // Skip lags longer than the signal
                continue;
            }

            let effective = (t - lag) as f64;
            let mut r = Array2::<f64>::zeros((n, n));

            for i in 0..n {
                for j in 0..n {
                    let mut sum = 0.0f64;
                    for s_idx in 0..(t - lag) {
                        sum += z[[i, s_idx]] * z[[j, s_idx + lag]];
                    }
                    r[[i, j]] = sum / effective;
                }
            }

            // Symmetrise
            let r_sym = (&r + &r.t()) * 0.5;
            result.push((lag, r_sym));
        }

        if result.is_empty() {
            return Err(SignalError::ValueError(
                "No valid lag-covariance matrices could be computed (all lags ≥ n_samples)"
                    .to_string(),
            ));
        }

        Ok(result)
    }

    /// Choose lags automatically: geometric spacing up to `n_lags` lags,
    /// bounded by `max_lag`.
    ///
    /// # Arguments
    ///
    /// * `n_samples` - Length of the signal
    /// * `n_lags`    - Number of lags to use (default 100)
    /// * `max_lag`   - Maximum lag (default n_samples / 4)
    ///
    /// # Returns
    ///
    /// Sorted list of unique positive lags.
    pub fn auto_lags(
        n_samples: usize,
        n_lags: Option<usize>,
        max_lag: Option<usize>,
    ) -> Vec<usize> {
        let nl = n_lags.unwrap_or(100).max(1);
        let ml = max_lag
            .unwrap_or(n_samples / 4)
            .max(1)
            .min(n_samples.saturating_sub(1));

        if ml == 0 || n_samples < 2 {
            return vec![];
        }

        let mut lags: Vec<usize> = if nl == 1 || ml == 1 {
            (1..=ml.min(nl)).collect()
        } else {
            // Logarithmic spacing from 1..=ml
            let mut v = Vec::with_capacity(nl);
            for k in 0..nl {
                let t = k as f64 / (nl - 1) as f64;
                let lag = (ml as f64).powf(t).round() as usize;
                let lag = lag.max(1).min(ml);
                v.push(lag);
            }
            v
        };

        lags.sort();
        lags.dedup();
        lags
    }
}

/// Joint diagonaliser used internally by SOBI.
///
/// Finds an orthogonal V such that V^T M_k V is approximately diagonal
/// for all lag-covariance matrices M_k. Uses Jacobi sweeps.
pub struct JointDiag {
    /// Maximum Jacobi sweeps
    pub max_iterations: usize,
    /// Convergence threshold
    pub tolerance: f64,
}

impl Default for JointDiag {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-6,
        }
    }
}

impl JointDiag {
    /// Create with custom parameters.
    pub fn new(max_iterations: usize, tolerance: f64) -> Self {
        Self {
            max_iterations,
            tolerance,
        }
    }

    /// Jointly diagonalise the matrices in-place, accumulating rotations in V.
    ///
    /// # Returns
    ///
    /// (V, n_iters, converged)
    pub fn run(&self, matrices: &mut Vec<Array2<f64>>) -> SignalResult<(Array2<f64>, usize, bool)> {
        if matrices.is_empty() {
            return Err(SignalError::ValueError(
                "No matrices provided for joint diagonalisation".to_string(),
            ));
        }

        let n = matrices[0].nrows();
        for m in matrices.iter() {
            if m.dim() != (n, n) {
                return Err(SignalError::DimensionMismatch(
                    "All matrices must be square with the same size".to_string(),
                ));
            }
        }

        let mut v = Array2::<f64>::eye(n);
        let mut n_iters = 0usize;

        for sweep in 0..self.max_iterations {
            let mut max_off = 0.0f64;

            for i in 0..n {
                for j in (i + 1)..n {
                    // Cardoso & Souloumiac SOBI rotation: minimise off-diagonal energy
                    // theta = 0.5 * atan2(2 * G12, G11 - G22)
                    let mut g11 = 0.0f64;
                    let mut g22 = 0.0f64;
                    let mut g12 = 0.0f64;

                    for m in matrices.iter() {
                        let mii = m[[i, i]];
                        let mjj = m[[j, j]];
                        let mij = m[[i, j]];
                        let mji = m[[j, i]];

                        g11 += (mii - mjj) * (mii - mjj);
                        g22 += (mij + mji) * (mij + mji);
                        g12 += (mii - mjj) * (mij + mji);

                        max_off = max_off.max(mij.abs()).max(mji.abs());
                    }

                    let theta = if g11 + g22 < 1e-15 {
                        0.0
                    } else {
                        0.5 * g12.atan2(0.5 * (g11 - g22))
                    };

                    if theta.abs() < self.tolerance * 1e-3 {
                        continue;
                    }

                    let c = theta.cos();
                    let s = theta.sin();

                    // Apply G^T M G to each matrix
                    for m in matrices.iter_mut() {
                        givens_rotate_inplace(m, n, i, j, c, s);
                    }

                    // Accumulate: V <- V G
                    for row in 0..n {
                        let vi = v[[row, i]];
                        let vj = v[[row, j]];
                        v[[row, i]] = c * vi + s * vj;
                        v[[row, j]] = -s * vi + c * vj;
                    }
                }
            }

            n_iters = sweep + 1;
            if max_off < self.tolerance {
                return Ok((v, n_iters, true));
            }
        }

        Ok((v, n_iters, false))
    }
}

/// Apply in-place similarity transform G^T M G for Givens rotation G[i,j].
fn givens_rotate_inplace(m: &mut Array2<f64>, n: usize, i: usize, j: usize, c: f64, s: f64) {
    // Left: G^T * M (rows i and j)
    for col in 0..n {
        let mi = m[[i, col]];
        let mj = m[[j, col]];
        m[[i, col]] = c * mi + s * mj;
        m[[j, col]] = -s * mi + c * mj;
    }
    // Right: M * G (cols i and j)
    for row in 0..n {
        let mi = m[[row, i]];
        let mj = m[[row, j]];
        m[[row, i]] = c * mi - s * mj;
        m[[row, j]] = s * mi + c * mj;
    }
}

/// SOBI: Second Order Blind Identification.
///
/// Uses second-order temporal statistics (multiple lagged covariance matrices)
/// to identify and separate statistically independent sources that have
/// different spectral profiles (coloured spectra).
///
/// ## Algorithm
///
/// 1. Centre and whiten the observed data.
/// 2. Compute lagged covariance matrices R_τ of the whitened data at
///    multiple time lags τ.
/// 3. Jointly diagonalise all R_τ using Jacobi sweeps.
/// 4. The joint diagonaliser V gives the separation: S = V^T Z = V^T W X.
///
/// ## When to use SOBI vs JADE
///
/// SOBI is preferred when sources have **different temporal spectra** but
/// may be Gaussian. JADE exploits **fourth-order statistics** and works
/// when sources are non-Gaussian but independent.
///
/// # Arguments
///
/// * `x`             - Observed mixed signals, shape `(n_channels, n_samples)`.
/// * `n_components`  - Number of sources to separate.
/// * `time_lags`     - Explicit list of time lags. If `None`, chosen automatically.
/// * `n_lags`        - Number of automatic lags (default 100). Ignored if `time_lags` set.
/// * `max_iterations`- Maximum Jacobi sweeps (default 1000).
/// * `tolerance`     - Convergence threshold (default 1e-6).
///
/// # Returns
///
/// A [`SOBIResult`] containing unmixing matrix, sources, mixing matrix, and diagnostics.
///
/// # Errors
///
/// Returns [`SignalError`] if inputs are invalid or if numerical issues arise.
///
/// # Example
///
/// ```rust
/// use scirs2_signal::bss::sobi::{sobi, SOBIResult};
/// use scirs2_core::ndarray::Array2;
///
/// let x = Array2::<f64>::eye(2);
/// let result = sobi(&x, 2, None, None, None, None).expect("operation should succeed");
/// assert_eq!(result.sources.nrows(), 2);
/// ```
pub fn sobi(
    x: &Array2<f64>,
    n_components: usize,
    time_lags: Option<&[usize]>,
    n_lags: Option<usize>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> SignalResult<SOBIResult> {
    let (n_channels, n_samples) = x.dim();

    if n_channels == 0 || n_samples == 0 {
        return Err(SignalError::ValueError(
            "Input matrix must be non-empty".to_string(),
        ));
    }
    if n_components == 0 || n_components > n_channels {
        return Err(SignalError::ValueError(format!(
            "n_components ({}) must be in [1, n_channels ({})]",
            n_components, n_channels
        )));
    }

    let max_iters = max_iterations.unwrap_or(1000);
    let tol = tolerance.unwrap_or(1e-6);

    // ----------------------------------------------------------------
    // Step 1: centre and whiten
    // ----------------------------------------------------------------
    let (whitened, w_white) = whiten_sobi(x, n_components)?;

    // ----------------------------------------------------------------
    // Step 2: determine time lags
    // ----------------------------------------------------------------
    let lags: Vec<usize> = match time_lags {
        Some(l) => {
            if l.is_empty() {
                return Err(SignalError::ValueError(
                    "time_lags slice must not be empty".to_string(),
                ));
            }
            l.to_vec()
        }
        None => {
            let auto = LagCovariance::auto_lags(n_samples, n_lags, None);
            if auto.is_empty() {
                return Err(SignalError::ValueError(
                    "Could not determine valid time lags (signal too short?)".to_string(),
                ));
            }
            auto
        }
    };

    // ----------------------------------------------------------------
    // Step 3: compute lagged covariance matrices
    // ----------------------------------------------------------------
    let lag_covs = LagCovariance::compute(&whitened, &lags)?;
    let used_lags: Vec<usize> = lag_covs.iter().map(|(l, _)| *l).collect();

    let mut matrices: Vec<Array2<f64>> = lag_covs.into_iter().map(|(_, m)| m).collect();

    // ----------------------------------------------------------------
    // Step 4: joint diagonalisation
    // ----------------------------------------------------------------
    let jd = JointDiag::new(max_iters, tol);
    let (v, n_iters, converged) = jd.run(&mut matrices)?;

    // ----------------------------------------------------------------
    // Step 5: unmixing matrix in original space
    // ----------------------------------------------------------------
    // Z = W_white * X, S = V^T * Z  =>  S = (V^T W_white) * X
    let w_unmix = v.t().dot(&w_white);
    let sources = w_unmix.dot(x);

    // ----------------------------------------------------------------
    // Step 6: mixing matrix (pseudo-inverse)
    // ----------------------------------------------------------------
    let mixing_matrix = pseudoinverse_sobi(&w_unmix)?;

    Ok(SOBIResult {
        unmixing_matrix: w_unmix,
        sources,
        mixing_matrix,
        time_lags: used_lags,
        converged,
        n_iterations: n_iters,
    })
}

/// Centre and whiten data for SOBI: returns (whitened, whitening_matrix).
/// whitened: (n_components, n_samples)
/// whitening_matrix: (n_components, n_channels)
fn whiten_sobi(x: &Array2<f64>, n_components: usize) -> SignalResult<(Array2<f64>, Array2<f64>)> {
    let (n_channels, n_samples) = x.dim();
    let t_f = n_samples as f64;

    // Centre
    let mean = x.mean_axis(Axis(1)).ok_or_else(|| {
        SignalError::ComputationError("Failed to compute column mean".to_string())
    })?;

    let mut centered = x.clone();
    for ch in 0..n_channels {
        let m = mean[ch];
        for s in 0..n_samples {
            centered[[ch, s]] -= m;
        }
    }

    // Covariance and eigen-decomposition
    let cov = centered.dot(&centered.t()) / t_f;
    let (eigvals, eigvecs) = eigh(&cov.view(), None)
        .map_err(|e| SignalError::ComputationError(format!("Eigendecomposition failed: {e}")))?;

    // Sort descending
    let mut idx: Vec<usize> = (0..n_channels).collect();
    idx.sort_by(|&a, &b| {
        eigvals[b]
            .partial_cmp(&eigvals[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let n_comp = n_components.min(n_channels);
    let mut w_white = Array2::<f64>::zeros((n_comp, n_channels));
    for (new_i, &old_i) in idx[..n_comp].iter().enumerate() {
        let scale = if eigvals[old_i] > 1e-12 {
            1.0 / eigvals[old_i].sqrt()
        } else {
            0.0
        };
        for ch in 0..n_channels {
            w_white[[new_i, ch]] = scale * eigvecs[[ch, old_i]];
        }
    }

    let whitened = w_white.dot(&centered);
    Ok((whitened, w_white))
}

/// Compute pseudo-inverse via SVD.
fn pseudoinverse_sobi(m: &Array2<f64>) -> SignalResult<Array2<f64>> {
    let (u, s, vt) = svd(&m.view(), false, None).map_err(|e| {
        SignalError::ComputationError(format!("SVD for pseudo-inverse failed: {e}"))
    })?;

    let rank = s.len();
    let mut s_inv = Array2::<f64>::zeros((vt.nrows(), u.nrows()));
    for i in 0..rank.min(s_inv.nrows()).min(s_inv.ncols()) {
        if s[i] > 1e-10 {
            s_inv[[i, i]] = 1.0 / s[i];
        }
    }

    Ok(vt.t().dot(&s_inv).dot(&u.t()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;
    use std::f64::consts::PI;

    fn make_test_signal(n_channels: usize, n_samples: usize) -> Array2<f64> {
        let mut x = Array2::<f64>::zeros((n_channels, n_samples));
        for ch in 0..n_channels {
            let freq = (ch + 1) as f64;
            for t in 0..n_samples {
                x[[ch, t]] = (2.0 * PI * freq * t as f64 / n_samples as f64).sin();
            }
        }
        x
    }

    #[test]
    fn test_sobi_basic() {
        let x = make_test_signal(2, 256);
        let result = sobi(&x, 2, None, Some(10), Some(200), Some(1e-6));
        match result {
            Ok(res) => {
                assert_eq!(res.sources.nrows(), 2);
                assert_eq!(res.sources.ncols(), 256);
                assert!(!res.time_lags.is_empty());
            }
            Err(e) => panic!("SOBI failed: {e}"),
        }
    }

    #[test]
    fn test_lag_covariance_auto() {
        let lags = LagCovariance::auto_lags(1000, Some(20), None);
        assert!(!lags.is_empty());
        assert!(lags.iter().all(|&l| l > 0));
        // Check sorted and unique
        for w in lags.windows(2) {
            assert!(w[0] < w[1]);
        }
    }

    #[test]
    fn test_lag_covariance_compute() {
        let z = Array2::<f64>::eye(3);
        // lag=1 is valid since n_samples=3 and lag < n_samples
        let covs = LagCovariance::compute(&z, &[1]);
        // With only 3 samples and lag=1, T-lag=2 pairs; should succeed
        assert!(covs.is_ok());
    }

    #[test]
    fn test_sobi_explicit_lags() {
        let x = make_test_signal(3, 512);
        let lags: Vec<usize> = (1..=20).collect();
        let result = sobi(&x, 3, Some(&lags), None, Some(500), Some(1e-5));
        assert!(result.is_ok());
        let res = result.expect("failed to create res");
        assert_eq!(res.time_lags.len(), 20);
    }

    #[test]
    fn test_sobi_invalid_input() {
        let x = Array2::<f64>::zeros((2, 100));
        // n_components > n_channels should error
        let result = sobi(&x, 5, None, None, None, None);
        assert!(result.is_err());
    }
}
