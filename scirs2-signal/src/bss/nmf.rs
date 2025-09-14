use ndarray::s;
// Non-negative Matrix Factorization (NMF) for blind source separation
//
// This module implements NMF techniques for signal processing.

use super::BssConfig;
use crate::error::{SignalError, SignalResult};
use ndarray::Array2;
use rand::{Rng, SeedableRng};

#[allow(unused_imports)]
/// Apply Non-negative Matrix Factorization (NMF) to separate mixed signals
///
/// NMF decomposes non-negative data into non-negative factors.
///
/// # Arguments
///
/// * `signals` - Matrix of mixed signals (rows are signals, columns are samples)
/// * `n_components` - Number of components to extract
/// * `config` - BSS configuration
///
/// # Returns
///
/// * Tuple containing (sources matrix H, mixing matrix W)
#[allow(dead_code)]
pub fn nmf(
    signals: &Array2<f64>,
    n_components: usize,
    config: &BssConfig,
) -> SignalResult<(Array2<f64>, Array2<f64>)> {
    let (n_signals, n_samples) = signals.dim();

    // Check that the signals are non-negative
    if signals.iter().any(|&x| x < 0.0) {
        return Err(SignalError::ValueError(
            "NMF requires non-negative signals".to_string(),
        ));
    }

    // Initialize random W and H matrices
    let mut rng = if let Some(seed) = config.random_seed {
        rand::rngs::StdRng::seed_from_u64([seed as u8; 32])
    } else {
        {
            // In rand 0.9, from_rng doesn't return Result but directly returns the PRNG
            rand::rngs::StdRng::from_rng(&mut rand::rng())
        }
    };

    let mut w = Array2::<f64>::zeros((n_signals, n_components));
    let mut h = Array2::<f64>::zeros((n_components, n_samples));

    for i in 0..n_signals {
        for j in 0..n_components {
            w[[i, j]] = rng.gen_range(0.0..1.0);
        }
    }

    for i in 0..n_components {
        for j in 0..n_samples {
            h[[i, j]] = rng.gen_range(0.0..1.0);
        }
    }

    // Normalize columns of W
    for j in 0..n_components {
        let norm = w.slice(s![.., j]).mapv(|x: f64| x.powi(2)).sum().sqrt();
        if norm > 0.0 {
            for i in 0..n_signals {
                w[[i, j]] /= norm;
            }
        }
    }
    // Perform NMF using multiplicative update rules
    for _ in 0..config.max_iterations {
        // Update H (sources)
        let w_t = w.t();
        let w_t_v = w_t.dot(signals);
        let w_t_w_h = w_t.dot(&w).dot(&h);

        for i in 0..n_components {
            for j in 0..n_samples {
                if w_t_w_h[[i, j]] > 1e-10 {
                    h[[i, j]] *= w_t_v[[i, j]] / w_t_w_h[[i, j]];
                }
            }
        }

        // Update W (mixing matrix)
        let v_h_t = signals.dot(&h.t());
        let w_h_h_t = w.dot(&h).dot(&h.t());

        for i in 0..n_signals {
            for j in 0..n_components {
                if w_h_h_t[[i, j]] > 1e-10 {
                    w[[i, j]] *= v_h_t[[i, j]] / w_h_h_t[[i, j]];
                }
            }
        }

        // Normalize columns of W
        for j in 0..n_components {
            let norm = w.slice(s![.., j]).mapv(|x: f64| x.powi(2)).sum().sqrt();
            if norm > 0.0 {
                for i in 0..n_signals {
                    w[[i, j]] /= norm;
                }

                // Scale H accordingly
                for i in 0..n_samples {
                    h[[j, i]] *= norm;
                }
            }
        }
    }

    // H contains the separated sources, W contains the mixing matrix
    Ok((h, w))
}
