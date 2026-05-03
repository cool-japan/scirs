//! Graph Signal Processing (GSP) — spectral and wavelet methods on graphs.
//!
//! This module implements the core GSP toolkit:
//! - **Graph Fourier Transform (GFT)** and its inverse using Laplacian eigenvectors.
//! - **Spectral graph filters** — ideal low-pass, high-pass, band-pass.
//! - **Graph Wavelets** via diffusion (heat-kernel wavelets).
//! - **Graph Signal Smoother** via Tikhonov (graph-Laplacian) regularization.
//!
//! All algorithms operate on `Array2<f64>` weighted adjacency matrices and
//! `Array1<f64>` graph signals (one value per node).
//!
//! ## Mathematical Background
//!
//! Let `L = D − A` be the combinatorial graph Laplacian with eigendecomposition
//! `L = U Λ Uᵀ`.  The **Graph Fourier Transform** of a signal `x` is
//! `x̂ = Uᵀ x` and the inverse is `x = U x̂`.  Spectral filters are applied
//! by multiplying `x̂` component-wise: `ŷ = h(Λ) x̂`.
//!
//! ## Example
//! ```rust,no_run
//! use scirs2_core::ndarray::{Array1, Array2};
//! use scirs2_graph::signal_processing::gsp::{GraphFourierTransform, IdealLowPass, GraphFilter};
//!
//! // Path graph: 0-1-2-3
//! let mut adj = Array2::<f64>::zeros((4, 4));
//! adj[[0,1]] = 1.0; adj[[1,0]] = 1.0;
//! adj[[1,2]] = 1.0; adj[[2,1]] = 1.0;
//! adj[[2,3]] = 1.0; adj[[3,2]] = 1.0;
//!
//! let gft = GraphFourierTransform::from_adjacency(&adj).unwrap();
//! let signal = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
//! let freq = gft.transform(&signal).unwrap();
//! let rec  = gft.inverse(&freq).unwrap();
//!
//! // Low-pass filter retaining lowest 2 frequency components
//! let lp = IdealLowPass::new(2);
//! let smoothed = lp.apply(&gft, &signal).unwrap();
//! ```

use scirs2_core::ndarray::{Array1, Array2};

use crate::error::{GraphError, Result};
use crate::spectral_graph::graph_laplacian;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers: symmetric tridiagonal eigendecomposition via Jacobi iterations
// ─────────────────────────────────────────────────────────────────────────────

/// Compute eigenvalues and eigenvectors of a real symmetric matrix via the
/// classical Jacobi iteration method.
///
/// Returns `(eigenvalues, eigenvectors)` where eigenvectors are stored as
/// columns of the returned matrix.
fn symmetric_eigen(a: &Array2<f64>) -> Result<(Array1<f64>, Array2<f64>)> {
    let n = a.nrows();
    if n == 0 {
        return Err(GraphError::InvalidGraph("empty matrix".into()));
    }
    if a.ncols() != n {
        return Err(GraphError::InvalidGraph("matrix must be square".into()));
    }

    // Work copy
    let mut m = a.clone();
    // Accumulate rotations in V (starts as identity)
    let mut v = Array2::<f64>::eye(n);

    const MAX_SWEEPS: usize = 500;
    const TOL: f64 = 1e-12;

    for _ in 0..MAX_SWEEPS {
        // Find the largest off-diagonal element
        let mut max_val = 0.0_f64;
        let mut p = 0_usize;
        let mut q = 1_usize;
        for i in 0..n {
            for j in (i + 1)..n {
                let v_ij = m[[i, j]].abs();
                if v_ij > max_val {
                    max_val = v_ij;
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < TOL {
            break;
        }

        // Compute Jacobi rotation angle
        let theta = if (m[[q, q]] - m[[p, p]]).abs() < TOL {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * ((2.0 * m[[p, q]]) / (m[[q, q]] - m[[p, p]])).atan()
        };
        let cos_t = theta.cos();
        let sin_t = theta.sin();

        // Apply rotation: M' = R^T M R, V' = V R
        // Update rows / cols p and q of m
        let mut new_m = m.clone();
        for r in 0..n {
            if r != p && r != q {
                new_m[[r, p]] = cos_t * m[[r, p]] - sin_t * m[[r, q]];
                new_m[[p, r]] = new_m[[r, p]];
                new_m[[r, q]] = sin_t * m[[r, p]] + cos_t * m[[r, q]];
                new_m[[q, r]] = new_m[[r, q]];
            }
        }
        new_m[[p, p]] =
            cos_t * cos_t * m[[p, p]] - 2.0 * sin_t * cos_t * m[[p, q]] + sin_t * sin_t * m[[q, q]];
        new_m[[q, q]] =
            sin_t * sin_t * m[[p, p]] + 2.0 * sin_t * cos_t * m[[p, q]] + cos_t * cos_t * m[[q, q]];
        new_m[[p, q]] = 0.0;
        new_m[[q, p]] = 0.0;
        m = new_m;

        // Update eigenvector matrix
        let v_old = v.clone();
        for r in 0..n {
            v[[r, p]] = cos_t * v_old[[r, p]] - sin_t * v_old[[r, q]];
            v[[r, q]] = sin_t * v_old[[r, p]] + cos_t * v_old[[r, q]];
        }
    }

    // Eigenvalues are diagonal entries of m
    let eigenvalues = Array1::from_iter((0..n).map(|i| m[[i, i]]));

    // Sort eigenvalues (and eigenvectors) in ascending order
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        eigenvalues[a]
            .partial_cmp(&eigenvalues[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let sorted_evals = Array1::from_iter(idx.iter().map(|&i| eigenvalues[i]));
    let mut sorted_evecs = Array2::<f64>::zeros((n, n));
    for (new_col, &old_col) in idx.iter().enumerate() {
        for row in 0..n {
            sorted_evecs[[row, new_col]] = v[[row, old_col]];
        }
    }

    Ok((sorted_evals, sorted_evecs))
}

// ─────────────────────────────────────────────────────────────────────────────
// GraphFourierTransform
// ─────────────────────────────────────────────────────────────────────────────

/// Graph Fourier Transform (GFT) based on the graph Laplacian eigenvectors.
///
/// The GFT projects a graph signal onto the frequency basis defined by the
/// eigenvectors of the graph Laplacian `L = D − A`.  Low-frequency components
/// correspond to smooth signals (slowly varying across edges); high-frequency
/// components to rapidly oscillating signals.
///
/// # Fields
/// - `eigenvalues` — sorted Laplacian spectrum (graph frequencies) λ₀ ≤ λ₁ ≤ …
/// - `eigenvectors` — columns are eigenvectors (basis functions); shape `(n, n)`
#[derive(Debug, Clone)]
pub struct GraphFourierTransform {
    /// Graph frequencies (Laplacian eigenvalues), sorted ascending.
    pub eigenvalues: Array1<f64>,
    /// Frequency basis: eigenvectors as columns, shape `(n, n)`.
    pub eigenvectors: Array2<f64>,
}

impl GraphFourierTransform {
    /// Build a GFT from a weighted adjacency matrix.
    ///
    /// Computes `L = D − A` and its full eigendecomposition.
    pub fn from_adjacency(adj: &Array2<f64>) -> Result<Self> {
        let n = adj.nrows();
        if n == 0 {
            return Err(GraphError::InvalidGraph("empty adjacency matrix".into()));
        }
        let lap = graph_laplacian(adj);
        let (eigenvalues, eigenvectors) = symmetric_eigen(&lap)?;
        Ok(Self {
            eigenvalues,
            eigenvectors,
        })
    }

    /// Build a GFT directly from a precomputed Laplacian matrix.
    pub fn from_laplacian(laplacian: &Array2<f64>) -> Result<Self> {
        let (eigenvalues, eigenvectors) = symmetric_eigen(laplacian)?;
        Ok(Self {
            eigenvalues,
            eigenvectors,
        })
    }

    /// Number of nodes (= size of frequency basis).
    pub fn num_nodes(&self) -> usize {
        self.eigenvalues.len()
    }

    /// Forward GFT: `x̂ = Uᵀ x`.
    ///
    /// # Arguments
    /// * `signal` — graph signal of length `n` (one value per node).
    ///
    /// # Returns
    /// Spectral coefficients `x̂` of length `n`.
    pub fn transform(&self, signal: &Array1<f64>) -> Result<Array1<f64>> {
        let n = self.num_nodes();
        if signal.len() != n {
            return Err(GraphError::InvalidParameter {
                param: "signal.len()".into(),
                value: signal.len().to_string(),
                expected: n.to_string(),
                context: "GFT forward transform".into(),
            });
        }
        // x̂_k = sum_i U[i,k] * x[i]  (Uᵀ applied to x)
        let mut x_hat = Array1::<f64>::zeros(n);
        for k in 0..n {
            let mut acc = 0.0_f64;
            for i in 0..n {
                acc += self.eigenvectors[[i, k]] * signal[i];
            }
            x_hat[k] = acc;
        }
        Ok(x_hat)
    }

    /// Inverse GFT: `x = U x̂`.
    ///
    /// # Arguments
    /// * `freq_signal` — spectral coefficients `x̂` of length `n`.
    ///
    /// # Returns
    /// Reconstructed graph signal `x` of length `n`.
    pub fn inverse(&self, freq_signal: &Array1<f64>) -> Result<Array1<f64>> {
        let n = self.num_nodes();
        if freq_signal.len() != n {
            return Err(GraphError::InvalidParameter {
                param: "freq_signal.len()".into(),
                value: freq_signal.len().to_string(),
                expected: n.to_string(),
                context: "GFT inverse transform".into(),
            });
        }
        // x_i = sum_k U[i,k] * x̂_k
        let mut x = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut acc = 0.0_f64;
            for k in 0..n {
                acc += self.eigenvectors[[i, k]] * freq_signal[k];
            }
            x[i] = acc;
        }
        Ok(x)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GraphFilter trait
// ─────────────────────────────────────────────────────────────────────────────

/// Trait for spectral graph filters.
///
/// A filter takes a graph signal and returns a filtered version by modifying
/// the spectral coefficients according to a frequency-response function `h(λ)`.
pub trait GraphFilter {
    /// Apply the filter to `signal` using the precomputed GFT basis.
    fn apply(&self, gft: &GraphFourierTransform, signal: &Array1<f64>) -> Result<Array1<f64>>;

    /// Return the frequency response `h(λ)` for each eigenvalue in `gft`.
    fn frequency_response(&self, gft: &GraphFourierTransform) -> Array1<f64>;
}

// ─────────────────────────────────────────────────────────────────────────────
// IdealLowPass
// ─────────────────────────────────────────────────────────────────────────────

/// Ideal low-pass spectral graph filter.
///
/// Retains the `k` lowest-frequency graph Fourier components and zeroes out
/// all higher-frequency components.  This is the graph analogue of the ideal
/// rectangular low-pass filter in classical DSP.
#[derive(Debug, Clone)]
pub struct IdealLowPass {
    /// Number of low-frequency components to retain.
    pub k: usize,
}

impl IdealLowPass {
    /// Create a new ideal low-pass filter retaining `k` components.
    pub fn new(k: usize) -> Self {
        Self { k }
    }
}

impl GraphFilter for IdealLowPass {
    fn frequency_response(&self, gft: &GraphFourierTransform) -> Array1<f64> {
        let n = gft.num_nodes();
        Array1::from_iter((0..n).map(|i| if i < self.k { 1.0 } else { 0.0 }))
    }

    fn apply(&self, gft: &GraphFourierTransform, signal: &Array1<f64>) -> Result<Array1<f64>> {
        let x_hat = gft.transform(signal)?;
        let h = self.frequency_response(gft);
        let filtered_hat = Array1::from_iter(x_hat.iter().zip(h.iter()).map(|(a, b)| a * b));
        gft.inverse(&filtered_hat)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// IdealHighPass
// ─────────────────────────────────────────────────────────────────────────────

/// Ideal high-pass spectral graph filter.
///
/// Zeroes out the `k` lowest-frequency graph Fourier components and retains
/// all higher-frequency components.  Useful for highlighting rapidly varying
/// parts of a graph signal (e.g. edge features, anomalies).
#[derive(Debug, Clone)]
pub struct IdealHighPass {
    /// Number of low-frequency components to suppress.
    pub k: usize,
}

impl IdealHighPass {
    /// Create a new ideal high-pass filter suppressing the `k` lowest components.
    pub fn new(k: usize) -> Self {
        Self { k }
    }
}

impl GraphFilter for IdealHighPass {
    fn frequency_response(&self, gft: &GraphFourierTransform) -> Array1<f64> {
        let n = gft.num_nodes();
        Array1::from_iter((0..n).map(|i| if i < self.k { 0.0 } else { 1.0 }))
    }

    fn apply(&self, gft: &GraphFourierTransform, signal: &Array1<f64>) -> Result<Array1<f64>> {
        let x_hat = gft.transform(signal)?;
        let h = self.frequency_response(gft);
        let filtered_hat = Array1::from_iter(x_hat.iter().zip(h.iter()).map(|(a, b)| a * b));
        gft.inverse(&filtered_hat)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GraphBandpass
// ─────────────────────────────────────────────────────────────────────────────

/// Ideal band-pass spectral graph filter.
///
/// Retains only frequency components whose indices fall in `[low_k, high_k)`,
/// i.e. the band between the `low_k`-th and `high_k`-th eigenvalue.
#[derive(Debug, Clone)]
pub struct GraphBandpass {
    /// Inclusive lower index of the retained frequency band.
    pub low_k: usize,
    /// Exclusive upper index of the retained frequency band.
    pub high_k: usize,
}

impl GraphBandpass {
    /// Create a new band-pass filter for the frequency band `[low_k, high_k)`.
    pub fn new(low_k: usize, high_k: usize) -> Self {
        Self { low_k, high_k }
    }
}

impl GraphFilter for GraphBandpass {
    fn frequency_response(&self, gft: &GraphFourierTransform) -> Array1<f64> {
        let n = gft.num_nodes();
        Array1::from_iter((0..n).map(|i| {
            if i >= self.low_k && i < self.high_k {
                1.0
            } else {
                0.0
            }
        }))
    }

    fn apply(&self, gft: &GraphFourierTransform, signal: &Array1<f64>) -> Result<Array1<f64>> {
        let x_hat = gft.transform(signal)?;
        let h = self.frequency_response(gft);
        let filtered_hat = Array1::from_iter(x_hat.iter().zip(h.iter()).map(|(a, b)| a * b));
        gft.inverse(&filtered_hat)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GraphWavelet (diffusion / heat-kernel wavelets)
// ─────────────────────────────────────────────────────────────────────────────

/// Graph wavelet based on the heat / diffusion kernel.
///
/// The diffusion kernel at scale `t` is `K_t = U exp(−t Λ) Uᵀ`.
/// The wavelet at node `s` is the `s`-th column (or row) of `K_t`.
/// This provides a spatially-localized, multi-scale representation of
/// graph signals.
///
/// # Reference
/// Hammond et al. (2011). "Wavelets on graphs via spectral graph theory."
/// *Applied and Computational Harmonic Analysis*, 30(2), 129–150.
#[derive(Debug, Clone)]
pub struct GraphWavelet {
    /// Diffusion scale parameter `t > 0`.
    pub scale: f64,
    /// Pre-computed kernel matrix `K_t = U exp(−t Λ) Uᵀ`, shape `(n, n)`.
    kernel: Array2<f64>,
}

impl GraphWavelet {
    /// Build the diffusion wavelet kernel at scale `t` from a GFT.
    ///
    /// # Arguments
    /// * `gft` — precomputed graph Fourier transform.
    /// * `scale` — diffusion scale `t > 0`.
    pub fn new(gft: &GraphFourierTransform, scale: f64) -> Result<Self> {
        if scale <= 0.0 {
            return Err(GraphError::InvalidParameter {
                param: "scale".into(),
                value: scale.to_string(),
                expected: "> 0".into(),
                context: "GraphWavelet construction".into(),
            });
        }
        let n = gft.num_nodes();
        // h(λ) = exp(−t λ)
        let h: Vec<f64> = gft
            .eigenvalues
            .iter()
            .map(|&lam| (-scale * lam).exp())
            .collect();

        // K_t[i,j] = sum_k U[i,k] h[k] U[j,k]
        let mut kernel = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0_f64;
                for k in 0..n {
                    acc += gft.eigenvectors[[i, k]] * h[k] * gft.eigenvectors[[j, k]];
                }
                kernel[[i, j]] = acc;
            }
        }
        Ok(Self { scale, kernel })
    }

    /// Apply the wavelet kernel to a signal: `y = K_t x`.
    pub fn apply(&self, signal: &Array1<f64>) -> Result<Array1<f64>> {
        let n = self.kernel.nrows();
        if signal.len() != n {
            return Err(GraphError::InvalidParameter {
                param: "signal.len()".into(),
                value: signal.len().to_string(),
                expected: n.to_string(),
                context: "GraphWavelet apply".into(),
            });
        }
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut acc = 0.0_f64;
            for j in 0..n {
                acc += self.kernel[[i, j]] * signal[j];
            }
            out[i] = acc;
        }
        Ok(out)
    }

    /// Return the wavelet atom centered at node `s` (column `s` of `K_t`).
    pub fn wavelet_atom(&self, s: usize) -> Result<Array1<f64>> {
        let n = self.kernel.nrows();
        if s >= n {
            return Err(GraphError::InvalidParameter {
                param: "s".into(),
                value: s.to_string(),
                expected: format!("< {n}"),
                context: "GraphWavelet atom".into(),
            });
        }
        Ok(self.kernel.column(s).to_owned())
    }

    /// Return the full kernel matrix `K_t` (shape `n × n`).
    pub fn kernel(&self) -> &Array2<f64> {
        &self.kernel
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GraphSignalSmoother (Tikhonov / graph Laplacian regularization)
// ─────────────────────────────────────────────────────────────────────────────

/// Graph signal smoother via Tikhonov (graph Laplacian) regularization.
///
/// Solves the regularized least-squares problem:
///
///   minimize  ‖x − y‖² + α xᵀ L x
///
/// where `y` is the observed (potentially noisy) signal, `L` is the graph
/// Laplacian, and `α > 0` is the regularization strength.
///
/// The closed-form solution is:
///
///   x* = (I + α L)⁻¹ y  =  U (I + α Λ)⁻¹ Uᵀ y
///
/// which can be computed efficiently using the GFT as a spectral filter with
/// frequency response `h(λ) = 1 / (1 + α λ)`.
#[derive(Debug, Clone)]
pub struct GraphSignalSmoother {
    /// Regularization strength α > 0.
    pub alpha: f64,
}

impl GraphSignalSmoother {
    /// Create a smoother with regularization strength `alpha`.
    pub fn new(alpha: f64) -> Result<Self> {
        if alpha <= 0.0 {
            return Err(GraphError::InvalidParameter {
                param: "alpha".into(),
                value: alpha.to_string(),
                expected: "> 0".into(),
                context: "GraphSignalSmoother construction".into(),
            });
        }
        Ok(Self { alpha })
    }

    /// Smooth the observed signal `y` using the GFT basis.
    ///
    /// Returns `x* = U (I + α Λ)⁻¹ Uᵀ y`.
    pub fn smooth(&self, gft: &GraphFourierTransform, signal: &Array1<f64>) -> Result<Array1<f64>> {
        let n = gft.num_nodes();
        if signal.len() != n {
            return Err(GraphError::InvalidParameter {
                param: "signal.len()".into(),
                value: signal.len().to_string(),
                expected: n.to_string(),
                context: "GraphSignalSmoother smooth".into(),
            });
        }
        let y_hat = gft.transform(signal)?;
        // Apply frequency response h(λ) = 1 / (1 + α λ)
        let x_hat = Array1::from_iter(
            y_hat
                .iter()
                .zip(gft.eigenvalues.iter())
                .map(|(&c, &lam)| c / (1.0 + self.alpha * lam)),
        );
        gft.inverse(&x_hat)
    }

    /// Return the frequency response function h(λ) = 1/(1 + α λ) evaluated at
    /// each graph frequency.
    pub fn frequency_response(&self, gft: &GraphFourierTransform) -> Array1<f64> {
        Array1::from_iter(
            gft.eigenvalues
                .iter()
                .map(|&lam| 1.0 / (1.0 + self.alpha * lam)),
        )
    }

    /// Compute the total variation (graph smoothness) of a signal:
    /// `TV(x) = xᵀ L x = Σ_{(i,j)∈E} w_ij (x_i − x_j)²`.
    pub fn total_variation(adj: &Array2<f64>, signal: &Array1<f64>) -> Result<f64> {
        let n = adj.nrows();
        if signal.len() != n {
            return Err(GraphError::InvalidParameter {
                param: "signal.len()".into(),
                value: signal.len().to_string(),
                expected: n.to_string(),
                context: "total_variation".into(),
            });
        }
        let mut tv = 0.0_f64;
        for i in 0..n {
            for j in (i + 1)..n {
                let w = adj[[i, j]];
                if w != 0.0 {
                    let diff = signal[i] - signal[j];
                    tv += w * diff * diff;
                }
            }
        }
        Ok(tv)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    fn path_graph_adj(n: usize) -> Array2<f64> {
        let mut adj = Array2::<f64>::zeros((n, n));
        for i in 0..(n - 1) {
            adj[[i, i + 1]] = 1.0;
            adj[[i + 1, i]] = 1.0;
        }
        adj
    }

    #[test]
    fn test_gft_reconstruction() {
        let adj = path_graph_adj(5);
        let gft = GraphFourierTransform::from_adjacency(&adj).unwrap();
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 2.0, 1.0]);
        let freq = gft.transform(&signal).unwrap();
        let rec = gft.inverse(&freq).unwrap();
        for (a, b) in signal.iter().zip(rec.iter()) {
            assert!((a - b).abs() < 1e-9, "Reconstruction error: {a} vs {b}");
        }
    }

    #[test]
    fn test_low_pass_smoothing() {
        let adj = path_graph_adj(6);
        let gft = GraphFourierTransform::from_adjacency(&adj).unwrap();
        let signal = Array1::from_vec(vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0]);
        let lp = IdealLowPass::new(2);
        let smoothed = lp.apply(&gft, &signal).unwrap();
        // The smoothed signal should have lower total variation
        let tv_orig = GraphSignalSmoother::total_variation(&adj, &signal).unwrap();
        let tv_smooth = GraphSignalSmoother::total_variation(&adj, &smoothed).unwrap();
        assert!(
            tv_smooth < tv_orig,
            "LP filter should reduce TV: {tv_smooth} vs {tv_orig}"
        );
    }

    #[test]
    fn test_high_pass_removes_dc() {
        let adj = path_graph_adj(5);
        let gft = GraphFourierTransform::from_adjacency(&adj).unwrap();
        // Constant signal = DC component only
        let dc_signal = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0]);
        let hp = IdealHighPass::new(1);
        let out = hp.apply(&gft, &dc_signal).unwrap();
        for v in out.iter() {
            assert!(v.abs() < 1e-9, "HP filter should remove DC: got {v}");
        }
    }

    #[test]
    fn test_bandpass() {
        let adj = path_graph_adj(8);
        let gft = GraphFourierTransform::from_adjacency(&adj).unwrap();
        let signal = Array1::from_vec(vec![1.0, 0.5, 0.0, -0.5, -1.0, -0.5, 0.0, 0.5]);
        let bp = GraphBandpass::new(2, 5);
        let out = bp.apply(&gft, &signal).unwrap();
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn test_wavelet_kernel_symmetry() {
        let adj = path_graph_adj(5);
        let gft = GraphFourierTransform::from_adjacency(&adj).unwrap();
        let wv = GraphWavelet::new(&gft, 1.0).unwrap();
        let k = wv.kernel();
        for i in 0..5 {
            for j in 0..5 {
                assert!((k[[i, j]] - k[[j, i]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_smoother_reduces_variation() {
        let adj = path_graph_adj(6);
        let gft = GraphFourierTransform::from_adjacency(&adj).unwrap();
        let noisy = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        let smoother = GraphSignalSmoother::new(5.0).unwrap();
        let smoothed = smoother.smooth(&gft, &noisy).unwrap();
        let tv_noisy = GraphSignalSmoother::total_variation(&adj, &noisy).unwrap();
        let tv_smooth = GraphSignalSmoother::total_variation(&adj, &smoothed).unwrap();
        assert!(tv_smooth < tv_noisy);
    }
}
