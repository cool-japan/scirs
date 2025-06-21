//! Joint Blind Source Separation and Joint Diagonalization
//!
//! This module implements Joint BSS techniques for multi-dataset blind source separation.

use super::{BssConfig, JadeMultiResult};
use crate::error::{SignalError, SignalResult};
use ndarray::{s, Array2, Axis};
use scirs2_linalg::{eigh, svd};

/// Apply Joint Blind Source Separation (JBSS) to separate mixed signals from multiple datasets
///
/// JBSS extends BSS to multiple datasets with shared sources but different mixing matrices.
///
/// # Arguments
///
/// * `datasets` - Vector of mixed signal matrices (each with rows as signals, columns as samples)
/// * `n_components` - Number of components to extract
/// * `config` - BSS configuration
///
/// # Returns
///
/// * Tuple containing (vector of extracted sources, vector of mixing matrices)
pub fn joint_bss(
    datasets: &[Array2<f64>],
    n_components: usize,
    _config: &BssConfig,
) -> SignalResult<JadeMultiResult> {
    if datasets.is_empty() {
        return Err(SignalError::ValueError("No datasets provided".to_string()));
    }

    // Number of datasets
    let n_datasets = datasets.len();

    // Build joint covariance matrices
    let mut joint_cov = Array2::<f64>::zeros((0, 0));
    let mut dataset_dims = Vec::with_capacity(n_datasets);

    for dataset in datasets {
        let (n_signals, n_samples) = dataset.dim();
        dataset_dims.push(n_signals);

        // Center the dataset
        let means = dataset.mean_axis(Axis(1)).unwrap();
        let mut centered = dataset.clone();

        for i in 0..n_signals {
            for j in 0..n_samples {
                centered[[i, j]] -= means[i];
            }
        }

        // Calculate covariance matrix
        let cov = centered.dot(&centered.t()) / (n_samples as f64 - 1.0);

        // Extend joint covariance matrix
        if joint_cov.dim().0 == 0 {
            joint_cov = cov;
        } else {
            // Create block diagonal matrix
            let current_dim = joint_cov.dim().0;
            let new_dim = current_dim + n_signals;
            let mut new_joint_cov = Array2::<f64>::zeros((new_dim, new_dim));

            // Copy existing joint covariance
            for i in 0..current_dim {
                for j in 0..current_dim {
                    new_joint_cov[[i, j]] = joint_cov[[i, j]];
                }
            }

            // Add new covariance block
            for i in 0..n_signals {
                for j in 0..n_signals {
                    new_joint_cov[[current_dim + i, current_dim + j]] = cov[[i, j]];
                }
            }

            joint_cov = new_joint_cov;
        }
    }

    // Perform joint diagonalization on the covariance matrices
    let (_eigvals, eigvecs) = match eigh(&joint_cov.view(), None) {
        Ok((vals, vecs)) => (vals, vecs),
        Err(_) => {
            return Err(SignalError::Compute(
                "Failed to compute eigendecomposition of joint covariance".to_string(),
            ));
        }
    };

    // Extract the unmixing matrices for each dataset
    let mut unmixing_matrices = Vec::with_capacity(n_datasets);
    let mut offset = 0;

    for &dim in &dataset_dims {
        let unmixing = eigvecs.slice(s![offset..offset + dim, ..]).to_owned();
        unmixing_matrices.push(unmixing);
        offset += dim;
    }

    // Apply the unmixing matrices to each dataset
    let mut extracted_sources = Vec::with_capacity(n_datasets);
    let mut mixing_matrices = Vec::with_capacity(n_datasets);

    for (i, dataset) in datasets.iter().enumerate() {
        let unmixing = &unmixing_matrices[i];
        let sources = unmixing.t().slice(s![0..n_components, ..]).dot(dataset);
        extracted_sources.push(sources);

        // Calculate mixing matrix (pseudoinverse of unmixing)
        let (u, s, vt) = match svd(&unmixing.view(), false, None) {
            Ok((u, s, vt)) => (u, s, vt),
            Err(_) => {
                return Err(SignalError::Compute(
                    "Failed to compute SVD of unmixing matrix".to_string(),
                ));
            }
        };

        let mut s_inv = Array2::<f64>::zeros((vt.dim().0, u.dim().0));
        for i in 0..s.len() {
            if s[i] > 1e-10 {
                s_inv[[i, i]] = 1.0 / s[i];
            }
        }

        let mixing = vt.t().dot(&s_inv).dot(&u.t());
        mixing_matrices.push(mixing.slice(s![.., 0..n_components]).to_owned());
    }

    Ok((extracted_sources, mixing_matrices))
}

/// Apply Joint Approximate Diagonalization (JAD) for blind source separation
///
/// JAD simultaneously diagonalizes multiple matrices for BSS.
///
/// # Arguments
///
/// * `signals` - Matrix of mixed signals (rows are signals, columns are samples)
/// * `n_components` - Number of components to extract
/// * `config` - BSS configuration
///
/// # Returns
///
/// * Tuple containing (extracted sources, mixing matrix)
pub fn joint_diagonalization(
    signals: &Array2<f64>,
    n_components: usize,
    config: &BssConfig,
) -> SignalResult<(Array2<f64>, Array2<f64>)> {
    let (n_signals, n_samples) = signals.dim();

    // Center the signals
    let means = signals.mean_axis(Axis(1)).unwrap();
    let mut centered = signals.clone();

    for i in 0..n_signals {
        for j in 0..n_samples {
            centered[[i, j]] -= means[i];
        }
    }

    // Create a set of time-delayed covariance matrices
    let mut cov_matrices = Vec::new();

    // Add the standard covariance matrix
    let cov = centered.dot(&centered.t()) / (n_samples as f64 - 1.0);
    cov_matrices.push(cov);

    // Add time-delayed covariance matrices
    let max_lag = 10.min(n_samples / 4);

    for lag in 1..=max_lag {
        let mut cov_lagged = Array2::<f64>::zeros((n_signals, n_signals));

        for i in 0..n_signals {
            for j in 0..n_signals {
                let mut sum = 0.0;

                for t in 0..n_samples - lag {
                    sum += centered[[i, t]] * centered[[j, t + lag]];
                }

                cov_lagged[[i, j]] = sum / (n_samples as f64 - lag as f64);
            }
        }

        cov_matrices.push(cov_lagged);
    }

    // Perform approximate joint diagonalization
    let mut v = Array2::<f64>::eye(n_signals);

    for _ in 0..config.max_iterations {
        let mut diagonalized = true;

        for i in 0..n_signals - 1 {
            for j in i + 1..n_signals {
                // Calculate rotation angle to jointly diagonalize matrices
                let mut num = 0.0;
                let mut denom = 0.0;

                for cov_matrix in &cov_matrices {
                    let cij = cov_matrix[[i, j]];
                    let cji = cov_matrix[[j, i]];
                    let cii = cov_matrix[[i, i]];
                    let cjj = cov_matrix[[j, j]];

                    num += 2.0 * (cij + cji);
                    denom += cii - cjj;
                }

                // Avoid division by zero
                if denom.abs() < 1e-10 {
                    continue;
                }

                let theta = 0.5 * (num / denom).atan();

                // Check if rotation is needed
                if theta.abs() > config.convergence_threshold {
                    diagonalized = false;

                    // Apply Givens rotation
                    let cos_t = theta.cos();
                    let sin_t = theta.sin();

                    // Create Givens rotation matrix
                    let mut g = Array2::<f64>::eye(n_signals);
                    g[[i, i]] = cos_t;
                    g[[i, j]] = -sin_t;
                    g[[j, i]] = sin_t;
                    g[[j, j]] = cos_t;

                    // Update V
                    v = v.dot(&g);

                    // Update covariance matrices
                    for matrix in &mut cov_matrices {
                        *matrix = g.t().dot(matrix).dot(&g);
                    }
                }
            }
        }

        if diagonalized {
            break;
        }
    }

    // Extract unmixing matrix (take the first n_components rows)
    let w = v.slice(s![0..n_components, ..]).to_owned();

    // Extract sources
    let sources = w.dot(&centered);

    // Calculate mixing matrix (pseudoinverse of w)
    let (u, s, vt) = match svd(&w.view(), false, None) {
        Ok((u, s, vt)) => (u, s, vt),
        Err(_) => {
            return Err(SignalError::Compute(
                "Failed to compute SVD of unmixing matrix".to_string(),
            ));
        }
    };

    let mut s_inv = Array2::<f64>::zeros((vt.dim().0, u.dim().0));
    for i in 0..s.len().min(s_inv.dim().0).min(s_inv.dim().1) {
        if s[i] > 1e-10 {
            s_inv[[i, i]] = 1.0 / s[i];
        }
    }

    let mixing = vt.t().dot(&s_inv).dot(&u.t());

    Ok((sources, mixing))
}
