use scirs2_core::ndarray::s;
// JADE (Joint Approximate Diagonalization of Eigenmatrices) for ICA
//
// This module implements the JADE algorithm for blind source separation.

use super::{pca, BssConfig};
use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::Array2;
use scirs2_linalg::solve_multiple;

#[allow(unused_imports)]
/// Implement JADE algorithm for ICA
///
/// JADE uses Joint Approximate Diagonalization of Eigenmatrices.
///
/// # Arguments
///
/// * `signals` - Matrix of signals (rows are signals, columns are samples)
/// * `n_components` - Number of independent components to extract
/// * `config` - BSS configuration
///
/// # Returns
///
/// * Tuple containing (extracted sources, unmixing matrix)
#[allow(dead_code)]
pub fn jade_ica(
    signals: &Array2<f64>,
    n_components: usize,
    config: &BssConfig,
) -> SignalResult<(Array2<f64>, Array2<f64>)> {
    let (n_signals, n_samples) = signals.dim();

    // Calculate covariance matrices
    let mut cumulants = Vec::new();

    // Use PCA as initial guess
    let (pca_sources, pca_mixing) = pca(signals, config)?;
    let pca_unmixing = match solve_multiple(
        &pca_mixing.view(),
        &Array2::<f64>::eye(n_signals).view(),
        None,
    ) {
        Ok(inv) => inv.slice(s![0..n_components, ..]).to_owned(),
        Err(_) => {
            return Err(crate::error::SignalError::ComputationError(
                "Failed to compute PCA unmixing matrix".to_string(),
            ));
        }
    };

    // Calculate cumulant matrices
    for k in 0..n_components {
        for l in k..n_components {
            let mut q = Array2::<f64>::zeros((n_components, n_components));

            // Compute fourth-order cross-cumulants
            for i in 0..n_components {
                for j in 0..n_components {
                    let mut cum = 0.0;

                    for t in 0..n_samples {
                        cum += pca_sources[[i, t]]
                            * pca_sources[[j, t]]
                            * pca_sources[[k, t]]
                            * pca_sources[[l, t]];
                    }

                    cum /= n_samples as f64;

                    // Remove Gaussian contribution
                    if i == j && k == l {
                        cum -= 1.0;
                    }
                    if i == k && j == l {
                        cum -= 1.0;
                    }
                    if i == l && j == k {
                        cum -= 1.0;
                    }

                    q[[i, j]] = cum;
                }
            }

            cumulants.push(q);
        }
    }

    // Joint diagonalization
    let mut v = Array2::<f64>::eye(n_components);

    for _ in 0..config.max_iterations {
        let mut diagonalized = true;

        for i in 0..n_components - 1 {
            for j in i + 1..n_components {
                // Givens rotation parameters
                let mut g11 = 0.0;
                let mut g12 = 0.0;
                let mut g21 = 0.0;
                let mut g22 = 0.0;

                for q in &cumulants {
                    g11 += q[[i, i]].powi(2) + q[[j, j]].powi(2);
                    g12 += q[[i, j]].powi(2) + q[[j, i]].powi(2);
                    g21 += q[[i, i]] * q[[i, j]] + q[[j, j]] * q[[j, i]];
                    g22 += q[[i, j]] * q[[j, i]] - q[[i, i]] * q[[j, j]];
                }

                // Calculate rotation angle
                let gamma = g12 - g11;
                let theta = if g22.abs() < 1e-10 {
                    PI / 4.0 * (g21 >= 0.0) as i32 as f64
                } else {
                    0.5 * (g21 / g22).atan()
                };

                // Check if rotation is needed
                if gamma.abs() > config.convergence_threshold
                    || theta.abs() > config.convergence_threshold
                {
                    diagonalized = false;

                    // Givens rotation matrix
                    let c = theta.cos();
                    let s = theta.sin();
                    let mut g = Array2::<f64>::eye(n_components);
                    g[[i, i]] = c;
                    g[[i, j]] = -s;
                    g[[j, i]] = s;
                    g[[j, j]] = c;

                    // Update V
                    v = v.dot(&g);

                    // Update cumulant matrices
                    for q in &mut cumulants {
                        *q = g.t().dot(q).dot(&g);
                    }
                }
            }
        }

        // Check if all matrices are jointly diagonalized
        if diagonalized {
            break;
        }
    }

    // Unmixing matrix is V * Wpca
    let w = v.dot(&pca_unmixing);

    // Extract the independent _components
    let sources = w.dot(signals);

    Ok((sources, w))
}
