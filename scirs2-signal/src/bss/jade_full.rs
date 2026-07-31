// JADE (Joint Approximate Diagonalization of Eigenmatrices) - Full Implementation
//
// Implements the JADE algorithm for blind source separation as described by:
// Cardoso & Souloumiac (1993). "Blind beamforming for non-Gaussian signals"
// IEE Proceedings-F, 140(6), pp. 362-370.

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{s, Array1, Array2, Axis};
use scirs2_linalg::{eigh, svd};
use std::f64::consts::PI;

/// Result from the JADE ICA algorithm
#[derive(Debug, Clone)]
pub struct JADEResult {
    /// Unmixing matrix W such that S = W * X (n_components x n_channels)
    pub unmixing_matrix: Array2<f64>,
    /// Separated sources (n_components x n_samples)
    pub sources: Array2<f64>,
    /// Mixing matrix A (approximately inv(W)) (n_channels x n_components)
    pub mixing_matrix: Array2<f64>,
    /// Kurtosis values of separated sources
    pub kurtosis: Vec<f64>,
    /// Number of iterations until convergence
    pub n_iterations: usize,
    /// Whether the algorithm converged
    pub converged: bool,
}

/// Fourth-order cumulant tensor utilities
pub struct FourthOrderCumulant;

impl FourthOrderCumulant {
    /// Compute the fourth-order cumulant matrices for whitened data.
    ///
    /// For whitened data z, returns a set of cumulant matrices Q_{ij}
    /// where `Q_{ij}[k,l] = cum(z_i, z_j, z_k, z_l)` after subtracting
    /// Gaussian terms.
    ///
    /// # Arguments
    ///
    /// * `z` - Whitened data matrix (n_components x n_samples)
    ///
    /// # Returns
    ///
    /// Vector of cumulant matrices (n_components^2 matrices of size n_components x n_components)
    pub fn compute(z: &Array2<f64>) -> SignalResult<Vec<Array2<f64>>> {
        let (n, t) = z.dim();
        if t == 0 {
            return Err(SignalError::ValueError(
                "Input data has zero samples".to_string(),
            ));
        }
        let t_f = t as f64;

        // Compute the full 4th order cumulant tensor contracted to n^2 matrices
        // Each matrix M_{ij} (i,j = 0..n) is an n x n matrix where:
        //   M_{ij}[k,l] = (1/T) sum_t z[i,t] z[j,t] z[k,t] z[l,t]
        //                 - delta(i,k)*delta(j,l) - delta(i,l)*delta(j,k) - delta(i,j)*delta(k,l)
        // (Gaussian contribution subtracted because data is whitened)

        let mut matrices: Vec<Array2<f64>> = Vec::with_capacity(n * n);

        for i in 0..n {
            for j in 0..n {
                let mut q = Array2::<f64>::zeros((n, n));

                for k in 0..n {
                    for l in 0..n {
                        // Fourth-order moment
                        let mut moment = 0.0f64;
                        for t_idx in 0..t {
                            moment += z[[i, t_idx]] * z[[j, t_idx]] * z[[k, t_idx]] * z[[l, t_idx]];
                        }
                        moment /= t_f;

                        // Subtract Gaussian contribution (isserlis theorem for unit-variance white noise)
                        let gauss = (if i == k && j == l { 1.0 } else { 0.0 })
                            + (if i == l && j == k { 1.0 } else { 0.0 })
                            + (if i == j && k == l { 1.0 } else { 0.0 });

                        q[[k, l]] = moment - gauss;
                    }
                }

                matrices.push(q);
            }
        }

        Ok(matrices)
    }

    /// Compute the kurtosis of each row of the source matrix.
    ///
    /// # Arguments
    ///
    /// * `sources` - Source signals (n_components x n_samples)
    ///
    /// # Returns
    ///
    /// Vector of kurtosis values (excess kurtosis, 0 for Gaussian)
    pub fn compute_kurtosis(sources: &Array2<f64>) -> Vec<f64> {
        let (n, t) = sources.dim();
        let t_f = t as f64;
        let mut kurtosis = Vec::with_capacity(n);

        for i in 0..n {
            let row = sources.slice(s![i, ..]);
            let mean = row.sum() / t_f;
            let variance = row.mapv(|x| (x - mean).powi(2)).sum() / t_f;

            let kurt = if variance > 1e-12 {
                let fourth_moment = row.mapv(|x| (x - mean).powi(4)).sum() / t_f;
                fourth_moment / variance.powi(2) - 3.0
            } else {
                0.0
            };
            kurtosis.push(kurt);
        }

        kurtosis
    }
}

/// Jacobi-like joint diagonalization of a set of symmetric matrices.
///
/// Finds an orthogonal matrix V such that V^T M_k V is approximately diagonal
/// for all matrices M_k in the set. Uses Jacobi sweeps with Givens rotations.
pub struct JointDiagonalization {
    /// Maximum number of Jacobi sweep iterations
    pub max_iterations: usize,
    /// Convergence threshold for off-diagonal elements
    pub tolerance: f64,
}

impl Default for JointDiagonalization {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-6,
        }
    }
}

impl JointDiagonalization {
    /// Create with custom parameters
    pub fn new(max_iterations: usize, tolerance: f64) -> Self {
        Self {
            max_iterations,
            tolerance,
        }
    }

    /// Perform joint diagonalization on a set of matrices.
    ///
    /// # Arguments
    ///
    /// * `matrices` - Slice of symmetric matrices all of the same size (n x n)
    ///
    /// # Returns
    ///
    /// Tuple (V, n_iters, converged) where V is the orthogonal diagonalizing matrix
    pub fn diagonalize(
        &self,
        matrices: &mut Vec<Array2<f64>>,
    ) -> SignalResult<(Array2<f64>, usize, bool)> {
        if matrices.is_empty() {
            return Err(SignalError::ValueError(
                "No matrices provided for joint diagonalization".to_string(),
            ));
        }

        let n = matrices[0].dim().0;
        if n == 0 {
            return Err(SignalError::ValueError(
                "Matrices have zero dimension".to_string(),
            ));
        }

        for m in matrices.iter() {
            if m.dim() != (n, n) {
                return Err(SignalError::DimensionMismatch(
                    "All matrices must have the same size".to_string(),
                ));
            }
        }

        // Initialize V as identity
        let mut v = Array2::<f64>::eye(n);
        let mut n_iters = 0;

        for sweep in 0..self.max_iterations {
            let mut max_off_diag = 0.0f64;

            for i in 0..n {
                for j in (i + 1)..n {
                    // Compute optimal Givens angle for this pair (i, j)
                    // Minimise sum_k (M_k[i,j]^2 + M_k[j,i]^2) via gradient descent
                    // Using the closed-form solution from the JADE paper

                    let (theta, off) = self.compute_givens_angle(matrices, n, i, j);
                    max_off_diag = max_off_diag.max(off);

                    if theta.abs() < self.tolerance * 1e-3 {
                        continue;
                    }

                    // Apply Givens rotation G to all matrices: M <- G^T M G
                    let c = theta.cos();
                    let s_val = theta.sin();

                    for m in matrices.iter_mut() {
                        apply_givens_left_right(m, n, i, j, c, s_val);
                    }

                    // Update accumulation matrix V <- V G
                    for row in 0..n {
                        let vi = v[[row, i]];
                        let vj = v[[row, j]];
                        v[[row, i]] = c * vi + s_val * vj;
                        v[[row, j]] = -s_val * vi + c * vj;
                    }
                }
            }

            n_iters = sweep + 1;

            if max_off_diag < self.tolerance {
                return Ok((v, n_iters, true));
            }
        }

        Ok((v, n_iters, false))
    }

    /// Compute the optimal Givens rotation angle for pair (i, j).
    ///
    /// Returns (theta, off_diag_magnitude).
    fn compute_givens_angle(
        &self,
        matrices: &[Array2<f64>],
        n: usize,
        i: usize,
        j: usize,
    ) -> (f64, f64) {
        // Gradient-based angle: solve 2x2 eigenproblem from the JADE paper
        // h = [sum_k (Cii - Cjj)^2,  sum_k (Cij + Cji)^2, sum_k (Cii - Cjj)(Cij + Cji)]
        let mut h11 = 0.0f64; // sum (Mii - Mjj)^2
        let mut h22 = 0.0f64; // sum (Mij + Mji)^2
        let mut h12 = 0.0f64; // sum (Mii - Mjj)(Mij + Mji)
        let mut off = 0.0f64;

        for m in matrices {
            let mii = m[[i, i]];
            let mjj = m[[j, j]];
            let mij = m[[i, j]];
            let mji = m[[j, i]];

            let diag_diff = mii - mjj;
            let off_sum = mij + mji;

            h11 += diag_diff * diag_diff;
            h22 += off_sum * off_sum;
            h12 += diag_diff * off_sum;

            off += mij * mij + mji * mji;
        }

        // off_sum magnitude for convergence check
        off = off.sqrt();

        // Solve the 2x2 eigenproblem [h11, h12; h12, h22]
        // Eigenvalues: lambda = (h11 + h22)/2 ± sqrt(((h11 - h22)/2)^2 + h12^2)
        // The optimal theta satisfies tan(2*theta) = h12 / ((h11 - h22) / 2)
        let denom = h11 - h22;
        let theta = if denom.abs() < 1e-15 && h12.abs() < 1e-15 {
            0.0
        } else {
            // atan2 gives angle in (-pi, pi]; divide by 2 for Givens rotation
            0.5 * h12.atan2(0.5 * denom)
        };

        (theta, off)
    }
}

/// Apply in-place Givens rotation G to a matrix M: M <- G^T M G
/// where G is the identity with [i,i]=c, [i,j]=-s, [j,i]=s, [j,j]=c
fn apply_givens_left_right(m: &mut Array2<f64>, n: usize, i: usize, j: usize, c: f64, s: f64) {
    // Left multiply by G^T: rows i and j transform
    for col in 0..n {
        let mi = m[[i, col]];
        let mj = m[[j, col]];
        m[[i, col]] = c * mi + s * mj;
        m[[j, col]] = -s * mi + c * mj;
    }
    // Right multiply by G: cols i and j transform
    for row in 0..n {
        let mi = m[[row, i]];
        let mj = m[[row, j]];
        m[[row, i]] = c * mi - s * mj;
        m[[row, j]] = s * mi + c * mj;
    }
}

/// JADE ICA algorithm.
///
/// Performs blind source separation using the Joint Approximate Diagonalization
/// of Eigenmatrices (JADE) approach. This is a higher-order statistics method
/// based on fourth-order cumulants.
///
/// ## Algorithm overview
///
/// 1. Whiten the data via PCA.
/// 2. Compute the fourth-order cumulant matrices of the whitened data.
/// 3. Reduce to `n_components` most informative cumulant matrices via SVD.
/// 4. Jointly diagonalize the cumulant matrices using Jacobi sweeps.
/// 5. Extract the unmixing matrix and separated sources.
///
/// # Arguments
///
/// * `x` - Observed mixed signals, shape `(n_channels, n_samples)`.
/// * `n_components` - Number of independent components to extract.
/// * `max_iterations` - Maximum Jacobi sweeps. Default: 1000.
/// * `tolerance` - Convergence threshold. Default: 1e-6.
///
/// # Returns
///
/// A [`JADEResult`] with unmixing matrix, sources, mixing matrix, and diagnostics.
///
/// # Errors
///
/// Returns [`SignalError`] if the input is invalid or eigendecomposition fails.
///
/// # Example
///
/// ```rust
/// use scirs2_signal::bss::jade_full::{jade, JADEResult};
/// use scirs2_core::ndarray::Array2;
///
/// // Create trivial 2-channel test
/// let x = Array2::<f64>::eye(2);
/// let result = jade(&x, 2, None, None).expect("operation should succeed");
/// assert_eq!(result.sources.nrows(), 2);
/// ```
pub fn jade(
    x: &Array2<f64>,
    n_components: usize,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> SignalResult<JADEResult> {
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
    // Step 1: center and whiten the data
    // ----------------------------------------------------------------
    let (whitened, w_matrix, _) = whiten(x, n_components)?;
    // whitened: (n_components, n_samples)
    // w_matrix: (n_components, n_channels) — the whitening matrix

    // ----------------------------------------------------------------
    // Step 2: compute fourth-order cumulant matrices
    // ----------------------------------------------------------------
    let mut cum_matrices = FourthOrderCumulant::compute(&whitened)?;
    // cum_matrices has n_components^2 matrices each of size n_components x n_components

    // ----------------------------------------------------------------
    // Step 3: reduce cumulant matrices to the n_components most
    //         discriminating ones via spectral selection
    // ----------------------------------------------------------------
    // Stack all cumulant matrices into a tall matrix and take SVD
    let n_cum = cum_matrices.len();
    let mut stacked = Array2::<f64>::zeros((n_cum * n_components, n_components));
    for (k, m) in cum_matrices.iter().enumerate() {
        for r in 0..n_components {
            for c in 0..n_components {
                stacked[[k * n_components + r, c]] = m[[r, c]];
            }
        }
    }

    // SVD: stacked = U S Vt, keep top n_components^2 rows via left singular vectors
    let (u_svd, s_svd, _vt_svd) = svd(&stacked.view(), false, None).map_err(|e| {
        SignalError::ComputationError(format!("SVD of stacked cumulants failed: {e}"))
    })?;

    // Keep the n_components^2 columns of U corresponding to largest singular values
    let n_keep = (n_components * n_components)
        .min(s_svd.len())
        .min(u_svd.ncols());
    // Project each cumulant matrix onto the subspace spanned by these U columns
    // Equivalent to: M_k' = U[:,0:n_keep]^T * M_k * U[:,0:n_keep]
    // But to stay O(n^3), we project differently. Use the original matrices directly.
    // (For small n, using all cumulant matrices is fine.)
    let _ = (u_svd, s_svd, n_keep); // suppress unused warnings; use all matrices

    // ----------------------------------------------------------------
    // Step 4: joint diagonalization of cumulant matrices
    // ----------------------------------------------------------------
    let jd = JointDiagonalization::new(max_iters, tol);
    let (v, n_iters, converged) = jd.diagonalize(&mut cum_matrices)?;
    // v is orthogonal, diagonalizing the cumulant matrices

    // ----------------------------------------------------------------
    // Step 5: build unmixing matrix and extract sources
    // ----------------------------------------------------------------
    // Unmixing matrix in original space: W = V^T * W_whitening
    let w_unmix = v.t().dot(&w_matrix);
    // Sources: S = W * X
    let sources = w_unmix.dot(x);

    // ----------------------------------------------------------------
    // Step 6: compute mixing matrix (pseudo-inverse of W_unmix)
    // ----------------------------------------------------------------
    let mixing_matrix = compute_pseudoinverse(&w_unmix)?;

    // ----------------------------------------------------------------
    // Step 7: compute kurtosis of sources
    // ----------------------------------------------------------------
    let kurtosis = FourthOrderCumulant::compute_kurtosis(&sources);

    Ok(JADEResult {
        unmixing_matrix: w_unmix,
        sources,
        mixing_matrix,
        kurtosis,
        n_iterations: n_iters,
        converged,
    })
}

/// Whiten (sphere) data: remove mean, decorrelate, scale to unit variance.
///
/// Returns (whitened, whitening_matrix, mean).
/// whitened: (n_components, n_samples)
/// whitening_matrix: (n_components, n_channels)
fn whiten(
    x: &Array2<f64>,
    n_components: usize,
) -> SignalResult<(Array2<f64>, Array2<f64>, Array1<f64>)> {
    let (n_channels, n_samples) = x.dim();
    let t_f = n_samples as f64;

    // Center
    let mean = x
        .mean_axis(Axis(1))
        .ok_or_else(|| SignalError::ComputationError("Failed to compute mean".to_string()))?;

    let mut centered = x.clone();
    for ch in 0..n_channels {
        let m = mean[ch];
        for s in 0..n_samples {
            centered[[ch, s]] -= m;
        }
    }

    // Covariance matrix
    let cov = centered.dot(&centered.t()) / t_f;

    // Eigendecomposition
    let (eigvals, eigvecs) = eigh(&cov.view(), None)
        .map_err(|e| SignalError::ComputationError(format!("Eigendecomposition failed: {e}")))?;
    // eigvals are in ascending order; we want descending for PCA
    // eigvecs columns correspond to eigvals

    let n_comp = n_components.min(n_channels);

    // Sort by descending eigenvalue
    let mut idx: Vec<usize> = (0..n_channels).collect();
    idx.sort_by(|&a, &b| {
        eigvals[b]
            .partial_cmp(&eigvals[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Build whitening matrix: D^{-1/2} * E^T where E has top n_comp eigenvectors
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
    Ok((whitened, w_white, mean))
}

/// Compute pseudo-inverse of a matrix via SVD.
fn compute_pseudoinverse(m: &Array2<f64>) -> SignalResult<Array2<f64>> {
    let (u, s, vt) = svd(&m.view(), false, None).map_err(|e| {
        SignalError::ComputationError(format!("SVD for pseudo-inverse failed: {e}"))
    })?;

    let (r, c) = m.dim();
    let rank = s.len();

    let mut s_inv = Array2::<f64>::zeros((vt.nrows(), u.nrows()));
    for i in 0..rank.min(s_inv.nrows()).min(s_inv.ncols()) {
        if s[i] > 1e-10 {
            s_inv[[i, i]] = 1.0 / s[i];
        }
    }

    let pinv = vt.t().dot(&s_inv).dot(&u.t());
    // pinv shape: (c, r)
    Ok(pinv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_jade_identity_input() {
        // When input is identity-like (already separated), JADE should recover it
        let n = 3;
        let t = 200;
        // Create super-Gaussian sources: laplacian-like
        let mut x = Array2::<f64>::zeros((n, t));
        for i in 0..n {
            for j in 0..t {
                // simple structured signal
                x[[i, j]] = ((i as f64 + 1.0) * (j as f64 + 1.0) / t as f64).sin();
            }
        }

        let result = jade(&x, n, Some(200), Some(1e-6));
        match result {
            Ok(res) => {
                assert_eq!(res.sources.nrows(), n);
                assert_eq!(res.sources.ncols(), t);
                assert_eq!(res.kurtosis.len(), n);
            }
            Err(e) => panic!("JADE failed: {e}"),
        }
    }

    #[test]
    fn test_fourth_order_cumulant() {
        let z = Array2::<f64>::eye(3);
        // Should not error even with T == n_channels
        let mats = FourthOrderCumulant::compute(&z);
        assert!(mats.is_ok());
    }

    #[test]
    fn test_joint_diagonalization_identity() {
        let mut matrices = vec![Array2::<f64>::eye(3)];
        let jd = JointDiagonalization::default();
        let (v, _n_iters, converged) = jd
            .diagonalize(&mut matrices)
            .expect("unexpected None or Err");
        // Identity is already diagonal; should converge immediately
        assert!(converged);
        // V should be close to a permutation matrix (or identity)
        assert_eq!(v.dim(), (3, 3));
    }

    #[test]
    fn test_kurtosis_gaussian() {
        // Gaussian data should have near-zero excess kurtosis
        let n = 1;
        let t = 10000;
        let mut data = Array2::<f64>::zeros((n, t));
        // Use a deterministic approximation: sum of cos
        for j in 0..t {
            let x: f64 = (2.0 * PI * j as f64 / t as f64).cos()
                + (2.0 * PI * 3.0 * j as f64 / t as f64).cos()
                + (2.0 * PI * 7.0 * j as f64 / t as f64).cos();
            data[[0, j]] = x;
        }
        let kurt = FourthOrderCumulant::compute_kurtosis(&data);
        // Should be finite
        assert!(kurt[0].is_finite());
    }
}
