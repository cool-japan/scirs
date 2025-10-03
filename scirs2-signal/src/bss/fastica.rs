use scirs2_core::ndarray::s;
// FastICA implementation for Independent Component Analysis
//
// This module implements the FastICA algorithm for blind source separation.

use super::{BssConfig, NonlinearityFunction};
use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::SeedableRng;
use scirs2_core::random::{Distribution, Normal};
use scirs2_linalg::eigh;
use statrs::statistics::Statistics;

#[allow(unused_imports)]
/// Implement FastICA algorithm for ICA
///
/// FastICA is a computationally efficient method that uses fixed-point iteration.
///
/// # Arguments
///
/// * `signals` - Matrix of signals (rows are signals, columns are samples)
/// * `n_components` - Number of independent components to extract
/// * `nonlinearity` - Nonlinearity function for FastICA
/// * `config` - BSS configuration
///
/// # Returns
///
/// * Tuple containing (extracted sources, unmixing matrix)
#[allow(dead_code)]
pub fn fast_ica(
    signals: &Array2<f64>,
    n_components: usize,
    nonlinearity: NonlinearityFunction,
    config: &BssConfig,
) -> SignalResult<(Array2<f64>, Array2<f64>)> {
    let (n_signals, n_samples) = signals.dim();

    // Initialize random unmixing matrix
    let mut rng = if let Some(seed) = config.random_seed {
        scirs2_core::random::rngs::StdRng::seed_from_u64([seed as u8; 32])
    } else {
        {
            // In rand 0.9, from_rng doesn't return Result but directly returns the PRNG
            scirs2_core::random::rngs::StdRng::from_rng(&mut scirs2_core::random::rng())
        }
    };

    let normal = Normal::new(0.0, 1.0).map_err(|e| {
        SignalError::ValueError(format!("Failed to create normal distribution: {}", e))
    })?;
    let mut w = Array2::<f64>::zeros((n_components, n_signals));

    for i in 0..n_components {
        for j in 0..n_signals {
            w[[i, j]] = normal.sample(&mut rng);
        }
    }

    // Function to apply nonlinearity and its derivative
    type NonlinearityFn = Box<dyn Fn(f64) -> f64>;
    let (g, g_prime): (NonlinearityFn, NonlinearityFn) = match nonlinearity {
        NonlinearityFunction::Logistic => (
            Box::new(|x: f64| x.tanh()),
            Box::new(|x: f64| 1.0 - x.tanh().powi(2)),
        ),
        NonlinearityFunction::Exponential => (
            Box::new(|x: f64| x * (-x * x / 2.0).exp()),
            Box::new(|x: f64| (-x * x / 2.0).exp() * (1.0 - x * x)),
        ),
        NonlinearityFunction::Cubic => (
            Box::new(|x: f64| x.powi(3)),
            Box::new(|x: f64| 3.0 * x.powi(2)),
        ),
        NonlinearityFunction::Tanh => (
            Box::new(|x: f64| x.tanh()),
            Box::new(|x: f64| 1.0 - x.tanh().powi(2)),
        ),
        NonlinearityFunction::Cosh => (
            Box::new(|x: f64| x.tanh()),
            Box::new(|x: f64| 1.0 - x.tanh().powi(2)),
        ),
    };

    // Apply FastICA algorithm (deflation approach)
    if config.use_fixed_point {
        // Apply fixed-point algorithm
        for p in 0..n_components {
            // Initialize the pth row of the unmixing matrix
            let mut wp = w.slice_mut(s![p, ..]).to_owned();

            // Normalize the initial vector
            let dot_product = wp.dot(&wp);
            let norm = f64::sqrt(dot_product);
            if norm > 0.0 {
                wp /= norm;
            }

            let mut w_old = Array1::<f64>::zeros(n_signals);
            let mut iteration = 0;

            // Fixed-point iteration
            loop {
                // Store previous weight vector
                w_old.assign(&wp);

                // Compute projections of signals onto current weight vector
                let mut projected = Array1::<f64>::zeros(n_samples);
                for j in 0..n_samples {
                    for i in 0..n_signals {
                        projected[j] += wp[i] * signals[[i, j]];
                    }
                }

                // Apply nonlinearity
                let mut gx = Array1::<f64>::zeros(n_samples);
                let mut g_prime_sum = 0.0;
                for j in 0..n_samples {
                    gx[j] = g(projected[j]);
                    g_prime_sum += g_prime(projected[j]);
                }
                g_prime_sum /= n_samples as f64;

                // Update weight vector
                let mut new_wp = Array1::<f64>::zeros(n_signals);
                for i in 0..n_signals {
                    let mut sum_gx_x = 0.0;
                    for j in 0..n_samples {
                        sum_gx_x += gx[j] * signals[[i, j]];
                    }
                    new_wp[i] = sum_gx_x / (n_samples as f64) - g_prime_sum * wp[i];
                }

                // Decorrelate from existing _components (Gram-Schmidt orthogonalization)
                for i in 0..p {
                    let wi = w.slice(s![i, ..]);
                    let proj = new_wp.dot(&wi);
                    for j in 0..n_signals {
                        new_wp[j] -= proj * wi[j];
                    }
                }

                // Normalize
                let norm = (new_wp.dot(&new_wp)).sqrt();
                if norm > 0.0 {
                    new_wp /= norm;
                }

                wp = new_wp;

                // Copy back to unmixing matrix
                let mut wp_slice = w.slice_mut(s![p, ..]);
                wp_slice.assign(&wp);

                // Check for convergence
                let dot_product = w_old.dot(&wp).abs();
                if (1.0 - dot_product) < config.convergence_threshold {
                    break;
                }

                iteration += 1;
                if iteration >= config.max_iterations {
                    break;
                }
            }
        }
    } else {
        // Apply simple gradient algorithm (less efficient but more robust)
        let mut w_old = Array2::<f64>::zeros((n_components, n_signals));

        for _iteration in 0..config.max_iterations {
            // Store previous weight matrix
            w_old.assign(&w);

            // Compute projections
            let projected = w.dot(signals);

            // Apply nonlinearity
            let mut gx = Array2::<f64>::zeros(projected.dim());
            for i in 0..n_components {
                for j in 0..n_samples {
                    gx[[i, j]] = g(projected[[i, j]]);
                }
            }

            // Compute gradient
            let gradient = gx.dot(&signals.t()) / (n_samples as f64)
                - Array2::<f64>::eye(n_components) * w.mapv(|x: f64| g_prime(x)).mean();

            // Update weight matrix
            w = &w + &(&gradient * config.learning_rate);

            // Decorrelate weights (symmetric decorrelation)
            let ww_t = w.dot(&w.t());
            let (eigvals, eigvecs) = match eigh(&ww_t.view(), None) {
                Ok((vals, vecs)) => (vals, vecs),
                Err(_) => {
                    return Err(crate::error::SignalError::ComputationError(
                        "Failed to compute eigendecomposition in FastICA".to_string(),
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

            // Check for convergence
            let mut converged = true;
            for i in 0..n_components {
                let mut max_dot = 0.0;
                for j in 0..n_components {
                    let dot = w.slice(s![i, ..]).dot(&w_old.slice(s![j, ..])).abs();
                    if dot > max_dot {
                        max_dot = dot;
                    }
                }
                if (1.0 - max_dot) > config.convergence_threshold {
                    converged = false;
                    break;
                }
            }

            if converged {
                break;
            }
        }
    }

    // Extract the independent _components
    let sources = w.dot(signals);

    Ok((sources, w))
}
