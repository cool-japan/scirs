//! Utility functions for STFT operations
//!
//! This module provides helper functions for dual window calculation,
//! COLA condition checking, and other STFT-related utilities.

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::Array1;

/// Calculate the closest STFT dual window
///
/// # Arguments
///
/// * `win` - Analysis window
/// * `hop` - Hop size
/// * `desired_dual` - Desired dual window (optional)
/// * `scaled` - Whether to apply scaling
///
/// # Returns
///
/// * Tuple of (dual window, scaling factor)
#[allow(dead_code)]
pub fn closest_stft_dual_window(
    win: &[f64],
    hop: usize,
    desired_dual: Option<&[f64]>,
    scaled: bool,
) -> SignalResult<(Vec<f64>, f64)> {
    // Validate inputs
    if win.is_empty() {
        return Err(SignalError::ValueError(
            "Window cannot be empty".to_string(),
        ));
    }

    let desired = if let Some(d) = desired_dual {
        if d.len() != win.len() {
            return Err(SignalError::ValueError(
                "Desired dual window must have the same length as window".to_string(),
            ));
        }
        Array1::from_vec(d.to_vec())
    } else {
        // Default to rectangular window
        Array1::ones(win.len())
    };

    if hop < 1 || hop > win.len() {
        return Err(SignalError::ValueError(format!(
            "Hop size must be between 1 and {}, got {}",
            win.len(),
            hop
        )));
    }

    // Calculate the canonical dual window
    let w_d = calc_dual_window_internal(win, hop)?;

    // Calculate correlations
    let win_array = Array1::from_vec(win.to_vec());
    let wdd = &win_array * &desired;

    let mut q_d = wdd.clone();
    for k in (hop..win.len()).step_by(hop) {
        for i in k..win.len() {
            q_d[i] += wdd[i - k];
        }
        for i in 0..(win.len() - k) {
            q_d[i] += wdd[i + k];
        }
    }

    q_d = &w_d * &q_d;

    if !scaled {
        let result = &w_d + &desired - &q_d;
        return Ok((result.to_vec(), 1.0));
    }

    // Calculate scaling factor
    let numerator = (q_d.iter().map(|&x| x * x).sum::<f64>()).sqrt();
    let denominator = q_d.iter().map(|&x| x * x).sum::<f64>();

    if !(numerator > 0.0 && denominator > f64::EPSILON) {
        return Err(SignalError::ValueError(
            "Unable to calculate scaled closest dual window due to numerically unstable scaling factor!".to_string(),
        ));
    }

    let alpha = numerator / denominator;
    let result = &w_d + (alpha * (&desired - &q_d));

    Ok((result.to_vec(), alpha))
}

/// Create a STFT window that satisfies the COLA condition
///
/// # Arguments
///
/// * `m` - Window length
/// * `hop` - Hop size
///
/// # Returns
///
/// * COLA window
#[allow(dead_code)]
pub fn create_cola_window(m: usize, hop: usize) -> SignalResult<Vec<f64>> {
    // Create initial rectangular window
    let rect_win = vec![1.0; m];

    // Find closest STFT dual window
    let (cola_win, _) = closest_stft_dual_window(&rect_win, hop, None, true)?;

    Ok(cola_win)
}

/// Internal function to calculate canonical dual window
fn calc_dual_window_internal(win: &[f64], hop: usize) -> SignalResult<Array1<f64>> {
    let m = win.len();
    let win_array = Array1::from_vec(win.to_vec());

    // Calculate sum of shifted windows
    let mut window_sum = Array1::zeros(m);

    // Add the window at different hop positions
    for shift in (0..m).step_by(hop) {
        for i in 0..m {
            let shifted_idx = (i + shift) % m;
            window_sum[i] += win[shifted_idx] * win[shifted_idx];
        }
    }

    // Calculate dual window
    let mut dual_window = Array1::zeros(m);
    for i in 0..m {
        if window_sum[i] > 1e-12 {
            dual_window[i] = win[i] / window_sum[i];
        } else {
            dual_window[i] = 0.0;
        }
    }

    Ok(dual_window)
}

/// Check if a window satisfies the COLA (Constant OverLap Add) condition
///
/// # Arguments
///
/// * `win` - Window function
/// * `hop` - Hop size
/// * `tolerance` - Numerical tolerance for checking
///
/// # Returns
///
/// * True if COLA condition is satisfied
#[allow(dead_code)]
pub fn check_cola_condition(win: &[f64], hop: usize, tolerance: f64) -> bool {
    let m = win.len();

    // Calculate overlap-add sum
    let mut ola_sum = vec![0.0; m];

    for shift in (0..m).step_by(hop) {
        for i in 0..m {
            let ola_idx = (i + shift) % m;
            ola_sum[ola_idx] += win[i];
        }
    }

    // Check if sum is approximately constant
    let mean_sum = ola_sum.iter().sum::<f64>() / m as f64;

    for &sum_val in &ola_sum {
        if (sum_val - mean_sum).abs() > tolerance {
            return false;
        }
    }

    true
}

/// Calculate window normalization factor for different scaling modes
///
/// # Arguments
///
/// * `win` - Window function
/// * `scaling` - Scaling mode
///
/// # Returns
///
/// * Normalization factor
#[allow(dead_code)]
pub fn calculate_window_normalization(win: &[f64], scaling: &str) -> f64 {
    match scaling.to_lowercase().as_str() {
        "magnitude" => {
            let sum_sq = win.iter().map(|&w| w * w).sum::<f64>();
            if sum_sq > 0.0 {
                1.0 / sum_sq.sqrt()
            } else {
                1.0
            }
        }
        "psd" => {
            let sum_sq = win.iter().map(|&w| w * w).sum::<f64>();
            if sum_sq > 0.0 {
                1.0 / sum_sq
            } else {
                1.0
            }
        }
        "energy" => {
            let sum = win.iter().sum::<f64>();
            if sum > 0.0 {
                1.0 / sum
            } else {
                1.0
            }
        }
        _ => 1.0, // No scaling
    }
}

/// Estimate optimal hop size for a given window
///
/// # Arguments
///
/// * `win` - Window function
/// * `overlap_ratio` - Desired overlap ratio (0.0 to 1.0)
///
/// # Returns
///
/// * Optimal hop size
#[allow(dead_code)]
pub fn estimate_optimal_hop_size(win: &[f64], overlap_ratio: f64) -> usize {
    let overlap_ratio = overlap_ratio.clamp(0.0, 0.99);
    let hop = (win.len() as f64 * (1.0 - overlap_ratio)).round() as usize;
    hop.max(1)
}

/// Calculate effective window length (accounting for zero padding)
///
/// # Arguments
///
/// * `win` - Window function
/// * `threshold` - Threshold for considering values significant
///
/// # Returns
///
/// * Effective window length
#[allow(dead_code)]
pub fn effective_window_length(win: &[f64], threshold: f64) -> usize {
    let max_val = win.iter().cloned().fold(0.0f64, f64::max);
    let effective_threshold = max_val * threshold;

    let mut start = 0;
    let mut end = win.len();

    // Find first significant value
    for (i, &val) in win.iter().enumerate() {
        if val.abs() >= effective_threshold {
            start = i;
            break;
        }
    }

    // Find last significant value
    for (i, &val) in win.iter().enumerate().rev() {
        if val.abs() >= effective_threshold {
            end = i + 1;
            break;
        }
    }

    end.saturating_sub(start)
}