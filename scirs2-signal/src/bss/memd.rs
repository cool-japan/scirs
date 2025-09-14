// Multivariate Empirical Mode Decomposition (MEMD) for signal separation
//
// This module implements MEMD for multivariate signal processing.

use super::BssConfig;
use crate::error::SignalResult;
use ndarray::Array2;
use rand::{Rng, SeedableRng};

#[allow(unused_imports)]
/// Apply Multivariate Empirical Mode Decomposition (MEMD) to separate components
///
/// MEMD decomposes signals into a set of Intrinsic Mode Functions (IMFs).
///
/// # Arguments
///
/// * `signals` - Matrix of signals (rows are signals, columns are samples)
/// * `n_directions` - Number of projection directions for MEMD
/// * `max_imfs` - Maximum number of IMFs to extract (or None for automatic)
/// * `config` - BSS configuration
///
/// # Returns
///
/// * Array of decomposed IMFs for each signal
#[allow(dead_code)]
pub fn multivariate_emd(
    signals: &Array2<f64>,
    n_directions: usize,
    max_imfs: Option<usize>,
    config: &BssConfig,
) -> SignalResult<Vec<Array2<f64>>> {
    let (n_signals, n_samples) = signals.dim();

    // Generate direction vectors on a hypersphere
    let mut _directions = Vec::with_capacity(n_directions);
    let mut rng = if let Some(seed) = config.random_seed {
        rand::rngs::StdRng::seed_from_u64([seed as u8; 32])
    } else {
        {
            // In rand 0.9, from_rng doesn't return Result but directly returns the PRNG
            rand::rngs::StdRng::from_rng(&mut rand::rng())
        }
    };

    for _ in 0..n_directions {
        let mut v = Vec::with_capacity(n_signals);

        // Generate random normal vector
        for _ in 0..n_signals {
            v.push(rng.gen_range(-1.0..1.0));
        }

        // Normalize
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }

        directions.push(v);
    }

    // Initialize IMF arrays
    let mut all_imfs = Vec::new();

    // Determine maximum number of IMFs
    let max_possible_imfs = (n_samples as f64).log2().floor() as usize;
    let max_imf_count = max_imfs.unwrap_or(max_possible_imfs);

    // Initialize with input signals
    let mut residuals = signals.clone();

    // Extract IMFs
    for _imf_idx in 0..max_imf_count {
        // Current IMF
        let mut imf = residuals.clone();
        let mut prev_imf = Array2::<f64>::zeros(imf.dim());

        // Apply sifting process
        for _iteration in 0..config.max_iterations {
            // Save previous IMF
            prev_imf.assign(&imf);

            // For each direction
            let mut envelopes = Vec::with_capacity(n_directions);

            for dir in &_directions {
                // Project signals onto direction
                let mut projection = Vec::with_capacity(n_samples);

                for j in 0..n_samples {
                    let mut proj = 0.0;
                    for i in 0..n_signals {
                        proj += imf[[i, j]] * dir[i];
                    }
                    projection.push(proj);
                }

                // Find extrema (maxima and minima)
                let mut extrema_indices = Vec::new();

                for j in 1..n_samples - 1 {
                    if (projection[j] > projection[j - 1] && projection[j] > projection[j + 1])
                        || (projection[j] < projection[j - 1] && projection[j] < projection[j + 1])
                    {
                        extrema_indices.push(j);
                    }
                }

                // Add first and last points
                extrema_indices.insert(0, 0);
                extrema_indices.push(n_samples - 1);

                // Compute envelope for this direction
                let mut envelope = Array2::<f64>::zeros((n_signals, n_samples));

                // Simple linear interpolation for envelope
                for i in 0..extrema_indices.len() - 1 {
                    let idx1 = extrema_indices[i];
                    let idx2 = extrema_indices[i + 1];

                    if idx2 > idx1 {
                        let step = 1.0 / (idx2 - idx1) as f64;

                        for j in idx1..=idx2 {
                            let t = (j - idx1) as f64 * step;

                            for k in 0..n_signals {
                                envelope[[k, j]] = (1.0 - t) * imf[[k, idx1]] + t * imf[[k, idx2]];
                            }
                        }
                    }
                }

                envelopes.push(envelope);
            }

            // Compute mean of envelopes
            let mut mean_envelope = Array2::<f64>::zeros((n_signals, n_samples));

            for envelope in &envelopes {
                mean_envelope = &mean_envelope + envelope;
            }

            mean_envelope /= n_directions as f64;

            // Subtract envelope mean from current IMF
            imf = &imf - &mean_envelope;

            // Check if IMF criteria are met
            let diff = (&imf - &prev_imf).mapv(|x: f64| x.powi(2)).sum().sqrt();
            let norm = imf.mapv(|x: f64| x.powi(2)).sum().sqrt();

            if diff / norm < config.convergence_threshold {
                break;
            }
        }

        // Store extracted IMF
        all_imfs.push(imf.clone());

        // Update residuals
        residuals = &residuals - &imf;

        // Check if residual is monotonic
        let mut is_monotonic = true;
        for i in 0..n_signals {
            let mut increasing_count = 0;
            let mut decreasing_count = 0;

            for j in 1..n_samples {
                if residuals[[i, j]] > residuals[[i, j - 1]] {
                    increasing_count += 1;
                } else if residuals[[i, j]] < residuals[[i, j - 1]] {
                    decreasing_count += 1;
                }
            }

            if increasing_count > 0 && decreasing_count > 0 {
                is_monotonic = false;
                break;
            }
        }

        if is_monotonic {
            break;
        }
    }

    // Add final residual as last IMF
    all_imfs.push(residuals);

    Ok(all_imfs)
}
