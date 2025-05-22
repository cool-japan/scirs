//! Blind source separation module
//!
//! This module implements various blind source separation (BSS) techniques for signal processing,
//! including Independent Component Analysis (ICA), Principal Component Analysis (PCA),
//! Non-negative Matrix Factorization (NMF), and related methods.

use crate::error::{SignalError, SignalResult};
use ndarray::{s, Array1, Array2, Axis};
use scirs2_linalg::eigh;

/// Type alias for multi-dataset JADE result
pub type JadeMultiResult = (Vec<Array2<f64>>, Vec<Array2<f64>>);

/// Configuration for blind source separation
#[derive(Debug, Clone)]
pub struct BssConfig {
    /// Maximum number of iterations for iterative methods
    pub max_iterations: usize,
    /// Convergence threshold for iterative methods
    pub convergence_threshold: f64,
    /// Whether to use whitening preprocessing
    pub apply_whitening: bool,
    /// Whether to perform dimensionality reduction
    pub dimension_reduction: bool,
    /// Target dimensionality (if None, determined automatically)
    pub target_dimension: Option<usize>,
    /// Variance to preserve in dimensionality reduction (0.0-1.0)
    pub variance_threshold: f64,
    /// Learning rate for gradient-based methods
    pub learning_rate: f64,
    /// Whether to apply non-negativity constraint
    pub non_negative: bool,
    /// Regularization parameter
    pub regularization: f64,
    /// Random seed for initialization
    pub random_seed: Option<u64>,
    /// Whether to use fixed-point algorithm for ICA
    pub use_fixed_point: bool,
    /// Whether to use parallel implementation for efficiency
    pub parallel: bool,
}

impl Default for BssConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            convergence_threshold: 1e-6,
            apply_whitening: true,
            dimension_reduction: false,
            target_dimension: None,
            variance_threshold: 0.95,
            learning_rate: 0.1,
            non_negative: false,
            regularization: 1e-4,
            random_seed: None,
            use_fixed_point: true,
            parallel: false,
        }
    }
}

/// ICA algorithms
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IcaMethod {
    /// FastICA algorithm
    FastICA,
    /// Infomax algorithm
    Infomax,
    /// JADE algorithm (Joint Approximate Diagonalization of Eigen-matrices)
    JADE,
    /// Extended Infomax algorithm (works with sub and super-Gaussian sources)
    ExtendedInfomax,
}

/// ICA nonlinearity functions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NonlinearityFunction {
    /// Logistic function (G(x) = tanh(x))
    Logistic,
    /// Exponential function (G(x) = x*exp(-x²/2))
    Exponential,
    /// Cubic function (G(x) = x³)
    Cubic,
    /// Hyperbolic tangent
    Tanh,
    /// Hyperbolic cosine
    Cosh,
}

/// Perform whitening (sphering) of signals
///
/// Whitening decorrelates the signals and scales them to unit variance.
///
/// # Arguments
///
/// * `signals` - Matrix of signals (rows are signals, columns are samples)
///
/// # Returns
///
/// * Tuple containing (whitened signals, whitening matrix)
pub fn whiten_signals(signals: &Array2<f64>) -> SignalResult<(Array2<f64>, Array2<f64>)> {
    let (n_signals, n_samples) = signals.dim();

    // Compute covariance matrix
    let cov = signals.dot(&signals.t()) / (n_samples as f64 - 1.0);

    // Perform eigendecomposition
    let (eigvals, eigvecs) = match eigh(&cov.view()) {
        Ok((vals, vecs)) => (vals, vecs),
        Err(_) => {
            return Err(SignalError::Compute(
                "Failed to compute eigendecomposition".to_string(),
            ));
        }
    };

    // Create diagonal matrix of scaled eigenvalues
    let mut d_inv_sqrt = Array2::<f64>::zeros((n_signals, n_signals));
    for i in 0..n_signals {
        if eigvals[i] > 1e-10 {
            d_inv_sqrt[[i, i]] = 1.0 / eigvals[i].sqrt();
        }
    }

    // Compute whitening matrix
    let whitening_matrix = d_inv_sqrt.dot(&eigvecs.t());

    // Apply whitening
    let whitened = whitening_matrix.dot(signals);

    Ok((whitened, whitening_matrix))
}

/// Sort decomposition components by explained variance
///
/// # Arguments
///
/// * `sources` - Source signals from decomposition
/// * `mixing` - Mixing matrix from decomposition
///
/// # Returns
///
/// * Tuple containing (sorted sources, sorted mixing matrix)
pub fn sort_components(
    sources: &Array2<f64>,
    mixing: &Array2<f64>,
) -> SignalResult<(Array2<f64>, Array2<f64>)> {
    let (n_components, n_samples) = sources.dim();

    // Calculate variance of each component
    let mut variances = Vec::with_capacity(n_components);

    for i in 0..n_components {
        let component = sources.slice(s![i, ..]);
        let mean = component.mean().unwrap();
        let var = component.mapv(|x: f64| (x - mean).powi(2)).sum() / (n_samples as f64 - 1.0);
        variances.push((i, var));
    }

    // Sort components by variance
    variances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Reorder components
    let mut sorted_sources = Array2::<f64>::zeros(sources.dim());
    let mut sorted_mixing = Array2::<f64>::zeros(mixing.dim());

    for (new_idx, (old_idx, _)) in variances.into_iter().enumerate() {
        sorted_sources
            .slice_mut(s![new_idx, ..])
            .assign(&sources.slice(s![old_idx, ..]));
        sorted_mixing
            .slice_mut(s![.., new_idx])
            .assign(&mixing.slice(s![.., old_idx]));
    }

    Ok((sorted_sources, sorted_mixing))
}

/// Calculate correlation matrix between signals
///
/// # Arguments
///
/// * `signals` - Matrix of signals (rows are signals, columns are samples)
///
/// # Returns
///
/// * Correlation matrix
pub fn calculate_correlation_matrix(signals: &Array2<f64>) -> SignalResult<Array2<f64>> {
    let (n_signals, n_samples) = signals.dim();

    // Center and normalize signals
    let mut normalized = Array2::<f64>::zeros(signals.dim());

    for i in 0..n_signals {
        let signal = signals.slice(s![i, ..]);
        let mean = signal.mean().unwrap();
        let std_dev = (signal.mapv(|x: f64| (x - mean).powi(2)).sum() / n_samples as f64).sqrt();

        if std_dev > 1e-10 {
            for j in 0..n_samples {
                normalized[[i, j]] = (signals[[i, j]] - mean) / std_dev;
            }
        }
    }

    // Calculate correlation matrix
    let corr = normalized.dot(&normalized.t()) / (n_samples as f64 - 1.0);

    Ok(corr)
}

/// Calculate mutual information between signals
///
/// # Arguments
///
/// * `signals` - Matrix of signals (rows are signals, columns are samples)
/// * `n_bins` - Number of bins for histogram estimation
///
/// # Returns
///
/// * Matrix of pairwise mutual information values
pub fn calculate_mutual_information(
    signals: &Array2<f64>,
    n_bins: usize,
) -> SignalResult<Array2<f64>> {
    let (n_signals, n_samples) = signals.dim();
    let mut mi_matrix = Array2::<f64>::zeros((n_signals, n_signals));

    // For each pair of signals
    for i in 0..n_signals {
        for j in 0..n_signals {
            if i == j {
                continue;
            }

            let x = signals.slice(s![i, ..]);
            let y = signals.slice(s![j, ..]);

            // Find min and max for each signal to define histogram bins
            let x_min = x.iter().fold(f64::INFINITY, |a: f64, &b| a.min(b));
            let x_max = x.iter().fold(f64::NEG_INFINITY, |a: f64, &b| a.max(b));
            let y_min = y.iter().fold(f64::INFINITY, |a: f64, &b| a.min(b));
            let y_max = y.iter().fold(f64::NEG_INFINITY, |a: f64, &b| a.max(b));

            let x_bin_width = (x_max - x_min) / n_bins as f64;
            let y_bin_width = (y_max - y_min) / n_bins as f64;

            // Create joint histogram
            let mut joint_hist = Array2::<f64>::zeros((n_bins, n_bins));
            let mut x_hist = Array1::<f64>::zeros(n_bins);
            let mut y_hist = Array1::<f64>::zeros(n_bins);

            // Fill histograms
            for s in 0..n_samples {
                let x_bin = ((x[s] - x_min) / x_bin_width).floor() as usize;
                let y_bin = ((y[s] - y_min) / y_bin_width).floor() as usize;

                // Handle edge cases
                let x_idx = x_bin.min(n_bins - 1);
                let y_idx = y_bin.min(n_bins - 1);

                joint_hist[[x_idx, y_idx]] += 1.0;
                x_hist[x_idx] += 1.0;
                y_hist[y_idx] += 1.0;
            }

            // Normalize histograms
            joint_hist /= n_samples as f64;
            x_hist /= n_samples as f64;
            y_hist /= n_samples as f64;

            // Calculate mutual information
            let mut mi = 0.0;

            for xi in 0..n_bins {
                for yi in 0..n_bins {
                    if joint_hist[[xi, yi]] > 1e-10 && x_hist[xi] > 1e-10 && y_hist[yi] > 1e-10 {
                        mi += joint_hist[[xi, yi]]
                            * (joint_hist[[xi, yi]] / (x_hist[xi] * y_hist[yi])).ln();
                    }
                }
            }

            mi_matrix[[i, j]] = mi;
        }
    }

    Ok(mi_matrix)
}

/// Estimate number of sources using eigenvalue analysis
///
/// # Arguments
///
/// * `signals` - Matrix of mixed signals (rows are signals, columns as samples)
/// * `threshold` - Eigenvalue ratio threshold
///
/// # Returns
///
/// * Estimated number of sources
pub fn estimate_source_count(signals: &Array2<f64>, threshold: f64) -> SignalResult<usize> {
    let (n_signals, n_samples) = signals.dim();

    // Center the signals
    let means = signals.mean_axis(Axis(1)).unwrap();
    let mut centered = signals.clone();

    for i in 0..n_signals {
        for j in 0..n_samples {
            centered[[i, j]] -= means[i];
        }
    }

    // Compute covariance matrix
    let cov = centered.dot(&centered.t()) / (n_samples as f64 - 1.0);

    // Perform eigendecomposition
    let eigvals = match eigh(&cov.view()) {
        Ok((vals, _)) => vals,
        Err(_) => {
            return Err(SignalError::Compute(
                "Failed to compute eigendecomposition".to_string(),
            ));
        }
    };

    // Calculate eigenvalue ratios
    let mut ratios = Vec::with_capacity(n_signals - 1);

    for i in 0..n_signals - 1 {
        if eigvals[i + 1] > 1e-10 {
            ratios.push(eigvals[i] / eigvals[i + 1]);
        }
    }

    // Find the largest gap
    let mut max_ratio_idx = 0;
    let mut max_ratio = 0.0;

    for (i, &ratio) in ratios.iter().enumerate() {
        if ratio > max_ratio {
            max_ratio = ratio;
            max_ratio_idx = i;
        }
    }

    // Check if max ratio exceeds threshold
    if max_ratio > threshold {
        Ok(max_ratio_idx + 1)
    } else {
        // If no clear gap, return a conservative estimate
        Ok(n_signals / 2)
    }
}

// Public module exports
mod fastica;
mod ica;
mod infomax;
mod jade;
mod joint;
mod kernel;
mod memd;
mod nmf;
mod pca;
mod sparse;

// Re-export public functions
pub use ica::ica;
pub use joint::{joint_bss, joint_diagonalization};
pub use kernel::kernel_ica;
pub use memd::multivariate_emd;
pub use nmf::nmf;
pub use pca::pca;
pub use sparse::sparse_component_analysis;
