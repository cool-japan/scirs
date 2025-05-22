//! Infomax and Extended Infomax implementations for ICA
//!
//! This module implements Infomax-based algorithms for blind source separation.

use super::BssConfig;
use crate::error::SignalResult;
use ndarray::{s, Array2};
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

/// Implement Infomax algorithm for ICA
///
/// Infomax is a neural network-based approach to ICA that maximizes information flow.
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
pub fn infomax_ica(
    signals: &Array2<f64>,
    n_components: usize,
    config: &BssConfig,
) -> SignalResult<(Array2<f64>, Array2<f64>)> {
    let (n_signals, n_samples) = signals.dim();

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
    let mut w = Array2::<f64>::zeros((n_components, n_signals));

    for i in 0..n_components {
        for j in 0..n_signals {
            w[[i, j]] = normal.sample(&mut rng) * 0.1;
        }
    }

    // Use identity matrix as initial unmixing matrix
    let eye = Array2::<f64>::eye(n_components);
    for i in 0..n_components.min(n_signals) {
        w[[i, i]] = 1.0;
    }

    // Learning rate schedule
    let mut learning_rate = 0.01;
    let min_learning_rate = 0.0001;
    let decay_rate = 0.9;

    // Batch size for stochastic gradient descent
    let batch_size = 128.min(n_samples);
    let n_batches = n_samples / batch_size;

    // Apply Infomax algorithm
    for _iteration in 0..config.max_iterations {
        let mut delta_w_sum = Array2::<f64>::zeros((n_components, n_signals));

        // Process in batches
        for batch in 0..n_batches {
            let start = batch * batch_size;
            let end = (batch + 1) * batch_size;

            let x_batch = signals.slice(s![.., start..end]);
            let y = w.dot(&x_batch);

            // Apply logistic nonlinearity
            let mut y_sigmoid = Array2::<f64>::zeros(y.dim());
            for i in 0..n_components {
                for j in 0..batch_size {
                    y_sigmoid[[i, j]] = 1.0 / (1.0 + (-y[[i, j]]).exp());
                }
            }

            // Compute gradient
            let block = &eye - &(&y_sigmoid * 2.0).dot(&Array2::ones((batch_size, n_components)));
            let delta_w =
                &block.dot(&y_sigmoid.dot(&x_batch.t())) * (learning_rate / batch_size as f64);

            delta_w_sum += &delta_w;
        }

        // Update unmixing matrix
        let delta_w_avg = delta_w_sum / n_batches as f64;
        w = &w + &delta_w_avg;

        // Reduce learning rate
        learning_rate = (learning_rate * decay_rate).max(min_learning_rate);

        // Check for convergence (simplified)
        if delta_w_avg.mapv(|x: f64| x.abs()).mean().unwrap() < config.convergence_threshold {
            break;
        }
    }

    // Extract the independent components
    let sources = w.dot(signals);

    Ok((sources, w))
}

/// Implement Extended Infomax algorithm for ICA
///
/// Extended Infomax works with both sub-Gaussian and super-Gaussian sources.
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
pub fn extended_infomax_ica(
    signals: &Array2<f64>,
    n_components: usize,
    config: &BssConfig,
) -> SignalResult<(Array2<f64>, Array2<f64>)> {
    let (n_signals, n_samples) = signals.dim();

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
    let mut w = Array2::<f64>::zeros((n_components, n_signals));

    for i in 0..n_components {
        for j in 0..n_signals {
            w[[i, j]] = normal.sample(&mut rng) * 0.1;
        }
    }

    // Use identity matrix as initial unmixing matrix
    let eye = Array2::<f64>::eye(n_components);
    for i in 0..n_components.min(n_signals) {
        w[[i, i]] = 1.0;
    }

    // Learning rate schedule
    let mut learning_rate = 0.01;
    let min_learning_rate = 0.0001;
    let decay_rate = 0.9;

    // Batch size for stochastic gradient descent
    let batch_size = 128.min(n_samples);
    let n_batches = n_samples / batch_size;

    // Sub-Gaussian or super-Gaussian detection
    let mut is_super_gaussian = vec![true; n_components];

    // Apply Extended Infomax algorithm
    for iteration in 0..config.max_iterations {
        let mut delta_w_sum = Array2::<f64>::zeros((n_components, n_signals));

        // Process in batches
        for batch in 0..n_batches {
            let start = batch * batch_size;
            let end = (batch + 1) * batch_size;

            let x_batch = signals.slice(s![.., start..end]);
            let y = w.dot(&x_batch);

            // Determine if signals are sub or super-Gaussian
            if iteration % 10 == 0 {
                for i in 0..n_components {
                    let mut kurtosis = 0.0;
                    for j in 0..batch_size {
                        kurtosis += y[[i, j]].powi(4);
                    }
                    kurtosis = kurtosis / batch_size as f64 - 3.0;
                    is_super_gaussian[i] = kurtosis > 0.0;
                }
            }

            // Compute nonlinearity based on sub/super-Gaussian nature
            let mut k = Array2::<f64>::zeros(y.dim());
            let mut k_prime = Array2::<f64>::zeros((n_components, n_components));

            for i in 0..n_components {
                if is_super_gaussian[i] {
                    // Super-Gaussian: tanh nonlinearity
                    for j in 0..batch_size {
                        k[[i, j]] = y[[i, j]].tanh();
                    }
                    k_prime[[i, i]] =
                        1.0 - k.slice(s![i, ..]).mapv(|x: f64| x.powi(2)).mean().unwrap();
                } else {
                    // Sub-Gaussian: cubic nonlinearity
                    for j in 0..batch_size {
                        k[[i, j]] = y[[i, j]].powi(3);
                    }
                    k_prime[[i, i]] = 3.0;
                }
            }

            // Compute gradient
            let block = &eye - &k.dot(&y.t()) / batch_size as f64 + &k_prime;
            let delta_w = &block.dot(&w) * learning_rate;

            delta_w_sum += &delta_w;
        }

        // Update unmixing matrix
        let delta_w_avg = delta_w_sum / n_batches as f64;
        w = &w + &delta_w_avg;

        // Reduce learning rate
        learning_rate = (learning_rate * decay_rate).max(min_learning_rate);

        // Check for convergence (simplified)
        if delta_w_avg.mapv(|x: f64| x.abs()).mean().unwrap() < config.convergence_threshold {
            break;
        }
    }

    // Extract the independent components
    let sources = w.dot(signals);

    Ok((sources, w))
}
