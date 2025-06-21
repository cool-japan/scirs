//! Principal Component Analysis (PCA) for blind source separation
//!
//! This module implements PCA techniques for signal processing.

use super::BssConfig;
use crate::error::{SignalError, SignalResult};
use ndarray::{s, Array2, Axis};
use scirs2_linalg::eigh;

/// Apply Principal Component Analysis (PCA) to separate mixed signals
///
/// PCA finds uncorrelated components that maximize variance.
///
/// # Arguments
///
/// * `signals` - Matrix of mixed signals (rows are signals, columns are samples)
/// * `config` - BSS configuration
///
/// # Returns
///
/// * Tuple containing (extracted sources, mixing matrix)
pub fn pca(signals: &Array2<f64>, config: &BssConfig) -> SignalResult<(Array2<f64>, Array2<f64>)> {
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
    let (eigvals, eigvecs) = match eigh(&cov.view(), None) {
        Ok((vals, vecs)) => (vals, vecs),
        Err(_) => {
            return Err(SignalError::Compute(
                "Failed to compute eigendecomposition".to_string(),
            ));
        }
    };

    // Sort eigenvectors by eigenvalues in descending order
    let mut indices: Vec<usize> = (0..n_signals).collect();
    indices.sort_by(|&i, &j| eigvals[j].partial_cmp(&eigvals[i]).unwrap());

    let mut sorted_eigvecs = Array2::<f64>::zeros((n_signals, n_signals));
    for (i, &idx) in indices.iter().enumerate() {
        for j in 0..n_signals {
            sorted_eigvecs[[j, i]] = eigvecs[[j, idx]];
        }
    }

    // Determine how many components to keep
    let n_components = if let Some(dim) = config.target_dimension {
        dim.min(n_signals)
    } else if config.dimension_reduction {
        // Calculate the number of components to keep based on variance threshold
        let total_var = eigvals.sum();
        let mut cum_var = 0.0;
        let mut k = 0;

        for i in 0..indices.len() {
            cum_var += eigvals[indices[i]];
            k += 1;
            if cum_var / total_var >= config.variance_threshold {
                break;
            }
        }

        k
    } else {
        n_signals
    };

    // Extract principal components
    let transform = sorted_eigvecs.slice(s![.., 0..n_components]);
    let sources = transform.t().dot(&centered);

    // Calculate the mixing matrix (transform)
    let mixing = transform.to_owned();

    Ok((sources, mixing))
}
