//! Kernel ICA for nonlinear blind source separation
//!
//! This module implements kernel-based ICA methods.

use super::{pca, BssConfig};
use crate::error::{SignalError, SignalResult};
use ndarray::{s, Array1, Array2, Axis};
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use scirs2_linalg::{eigh, svd};

/// Apply Kernel ICA for nonlinear blind source separation
///
/// Kernel ICA uses kernel methods to handle nonlinearities in the data.
///
/// # Arguments
///
/// * `signals` - Matrix of mixed signals (rows are signals, columns as samples)
/// * `n_components` - Number of independent components to extract
/// * `kernel_width` - Width parameter for Gaussian kernel
/// * `config` - BSS configuration
///
/// # Returns
///
/// * Tuple containing (extracted sources, mixing matrix)
pub fn kernel_ica(
    signals: &Array2<f64>,
    n_components: usize,
    kernel_width: f64,
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

    // Use PCA as prewhitening
    let (pca_sources, pca_mixing) = pca(&centered, config)?;

    // Initialize random unmixing matrix
    let mut rng = if let Some(seed) = config.random_seed {
        rand::rngs::StdRng::from_seed([seed as u8; 32])
    } else {
        {
            // In rand 0.9, from_rng doesn't return Result but directly returns the PRNG
            rand::rngs::StdRng::from_rng(&mut rand::rng())
        }
    };

    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut w = Array2::<f64>::zeros((n_components, n_components));

    for i in 0..n_components {
        for j in 0..n_components {
            w[[i, j]] = normal.sample(&mut rng) * 0.1;
        }
    }

    // Start with identity matrix
    for i in 0..n_components {
        w[[i, i]] = 1.0;
    }

    // Compute whitened data
    let whitened = pca_sources.slice(s![0..n_components, ..]).to_owned();

    // Compute Gram matrices for each component
    let compute_gram_matrix = |component: &Array1<f64>| -> Array2<f64> {
        let mut gram = Array2::<f64>::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            for j in 0..n_samples {
                let diff = component[i] - component[j];
                gram[[i, j]] = (-diff * diff / (2.0 * kernel_width * kernel_width)).exp();
            }
        }

        // Center Gram matrix
        let row_means = gram.mean_axis(Axis(0)).unwrap();
        let col_means = gram.mean_axis(Axis(1)).unwrap();
        let total_mean = gram.mean().unwrap();

        for i in 0..n_samples {
            for j in 0..n_samples {
                gram[[i, j]] -= row_means[j] + col_means[i] - total_mean;
            }
        }

        gram
    };

    // Optimize using gradient descent
    let mut learning_rate = config.learning_rate;
    let min_learning_rate = 0.001;
    let decay_rate = 0.95;

    for iteration in 0..config.max_iterations {
        // Apply current unmixing matrix
        let mut y = Array2::<f64>::zeros((n_components, n_samples));
        for i in 0..n_components {
            for j in 0..n_samples {
                for k in 0..n_components {
                    y[[i, j]] += w[[i, k]] * whitened[[k, j]];
                }
            }
        }

        // Compute Gram matrices
        let mut gram_matrices = Vec::with_capacity(n_components);
        for i in 0..n_components {
            gram_matrices.push(compute_gram_matrix(&y.slice(s![i, ..]).to_owned()));
        }

        // Compute HSIC (Hilbert-Schmidt Independence Criterion)
        let mut hsic = 0.0;
        for i in 0..n_components {
            for j in 0..n_components {
                if i != j {
                    let gram_i = &gram_matrices[i];
                    let gram_j = &gram_matrices[j];

                    // Calculate Frobenius inner product tr(gram_i * gram_j)
                    for k in 0..n_samples {
                        for l in 0..n_samples {
                            hsic += gram_i[[k, l]] * gram_j[[k, l]];
                        }
                    }
                }
            }
        }
        hsic /= (n_samples * n_samples) as f64;

        // Compute gradient
        let mut gradient = Array2::<f64>::zeros((n_components, n_components));

        for c in 0..n_components {
            for d in 0..n_components {
                let mut grad_cd = 0.0;

                for i in 0..n_components {
                    if i != c {
                        let gram_i = &gram_matrices[i];
                        let gram_c = &gram_matrices[c];

                        for s in 0..n_samples {
                            for t in 0..n_samples {
                                // Compute partial derivative
                                let mut kernel_deriv_sum = 0.0;

                                for u in 0..n_samples {
                                    let diff_su = y[[c, s]] - y[[c, u]];
                                    let kernel_su = gram_c[[s, u]];
                                    kernel_deriv_sum +=
                                        kernel_su * (whitened[[d, s]] - whitened[[d, u]]) * diff_su;
                                }

                                kernel_deriv_sum /= -kernel_width * kernel_width;
                                grad_cd += kernel_deriv_sum * gram_i[[s, t]];
                            }
                        }
                    }
                }

                gradient[[c, d]] = grad_cd / (n_samples * n_samples) as f64;
            }
        }

        // Update unmixing matrix
        w = &w - &(&gradient * learning_rate);

        // Decorrelate using symmetric decorrelation
        let ww_t = w.dot(&w.t());
        let (eigvals, eigvecs) = match eigh(&ww_t.view(), None) {
            Ok((vals, vecs)) => (vals, vecs),
            Err(_) => {
                return Err(SignalError::Compute(
                    "Failed to compute eigendecomposition in KernelICA".to_string(),
                ));
            }
        };

        let mut d_inv_sqrt = Array2::<f64>::zeros((n_components, n_components));
        for i in 0..n_components {
            if eigvals[i] > 1e-10 {
                d_inv_sqrt[[i, i]] = 1.0 / eigvals[i].sqrt();
            }
        }

        w = eigvecs.dot(&d_inv_sqrt).dot(&eigvecs.t()).dot(&w);

        // Reduce learning rate
        learning_rate = (learning_rate * decay_rate).max(min_learning_rate);

        // Check convergence criterion
        if iteration > 10 && hsic < config.convergence_threshold {
            break;
        }
    }

    // Apply final unmixing matrix to whitened data
    let sources = w.dot(&whitened);

    // Compute the total unmixing matrix
    let unmixing = w.dot(&pca_mixing.slice(s![.., 0..n_components]).t());

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
    for i in 0..s.len().min(s_inv.dim().0).min(s_inv.dim().1) {
        if s[i] > 1e-10 {
            s_inv[[i, i]] = 1.0 / s[i];
        }
    }

    let mixing = vt.t().dot(&s_inv).dot(&u.t());

    Ok((sources, mixing))
}
