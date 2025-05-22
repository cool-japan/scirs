//! Independent Component Analysis (ICA) for blind source separation
//!
//! This module implements the main ICA interface for BSS techniques.

use super::fastica::fast_ica;
use super::infomax::{extended_infomax_ica, infomax_ica};
use super::jade::jade_ica;
use super::{whiten_signals, BssConfig, IcaMethod, NonlinearityFunction};
use crate::error::{SignalError, SignalResult};
use ndarray::{Array2, Axis};
use scirs2_linalg::solve_multiple;

/// Apply Independent Component Analysis (ICA) to separate mixed signals
///
/// ICA finds statistically independent components that generated the mixed signals.
///
/// # Arguments
///
/// * `signals` - Matrix of mixed signals (rows are signals, columns are samples)
/// * `n_components` - Number of independent components to extract (default: same as signals)
/// * `method` - ICA algorithm to use
/// * `nonlinearity` - Nonlinearity function for FastICA
/// * `config` - BSS configuration
///
/// # Returns
///
/// * Tuple containing (extracted sources, mixing matrix)
pub fn ica(
    signals: &Array2<f64>,
    n_components: Option<usize>,
    method: IcaMethod,
    nonlinearity: NonlinearityFunction,
    config: &BssConfig,
) -> SignalResult<(Array2<f64>, Array2<f64>)> {
    let (n_signals, n_samples) = signals.dim();

    // Determine number of components
    let n_comp = n_components.unwrap_or(n_signals);
    if n_comp > n_signals {
        return Err(SignalError::ValueError(format!(
            "Number of components ({}) cannot exceed number of signals ({})",
            n_comp, n_signals
        )));
    }

    // Center the signals
    let means = signals.mean_axis(Axis(1)).unwrap();
    let mut centered = signals.clone();

    for i in 0..n_signals {
        for j in 0..n_samples {
            centered[[i, j]] -= means[i];
        }
    }

    // Whitening (decorrelation + scaling)
    let (whitened, whitening_matrix) = if config.apply_whitening {
        whiten_signals(&centered)?
    } else {
        (centered.clone(), Array2::<f64>::eye(n_signals))
    };

    // Apply the requested ICA method
    let (sources, unmixing) = match method {
        IcaMethod::FastICA => fast_ica(&whitened, n_comp, nonlinearity, config)?,
        IcaMethod::Infomax => infomax_ica(&whitened, n_comp, config)?,
        IcaMethod::JADE => jade_ica(&whitened, n_comp, config)?,
        IcaMethod::ExtendedInfomax => extended_infomax_ica(&whitened, n_comp, config)?,
    };

    // Calculate mixing matrix
    // A = W^-1 * whitening_matrix^-1
    let ica_mixing = match solve_multiple(
        &unmixing.view(),
        &Array2::<f64>::eye(unmixing.dim().0).view(),
    ) {
        Ok(inv) => inv,
        Err(_) => {
            return Err(SignalError::Compute(
                "Failed to compute ICA mixing matrix".to_string(),
            ));
        }
    };

    let whitening_inv = match solve_multiple(
        &whitening_matrix.view(),
        &Array2::<f64>::eye(whitening_matrix.dim().0).view(),
    ) {
        Ok(inv) => inv,
        Err(_) => {
            return Err(SignalError::Compute(
                "Failed to invert whitening matrix".to_string(),
            ));
        }
    };

    let mixing = ica_mixing.dot(&whitening_inv);

    Ok((sources, mixing))
}
