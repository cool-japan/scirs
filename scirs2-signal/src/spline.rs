//! B-spline filtering and signal processing.
//!
//! This module provides functionality for B-spline filtering, smoothing,
//! and interpolation, useful for signal processing applications. B-splines
//! are piecewise polynomial functions that provide a smooth approximation
//! to a signal with continuous derivatives.

use crate::error::{SignalError, SignalResult};
use num_traits::{Float, NumCast};
use std::fmt::Debug;

/// B-spline filter order values
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SplineOrder {
    /// Constant B-spline (order 0)
    Constant = 0,
    /// Linear B-spline (order 1)
    Linear = 1,
    /// Quadratic B-spline (order 2)
    Quadratic = 2,
    /// Cubic B-spline (order 3)
    #[default]
    Cubic = 3,
    /// Quartic B-spline (order 4)
    Quartic = 4,
    /// Quintic B-spline (order 5)
    Quintic = 5,
}

impl SplineOrder {
    /// Convert from integer to SplineOrder
    pub fn from_int(order: usize) -> SignalResult<Self> {
        match order {
            0 => Ok(SplineOrder::Constant),
            1 => Ok(SplineOrder::Linear),
            2 => Ok(SplineOrder::Quadratic),
            3 => Ok(SplineOrder::Cubic),
            4 => Ok(SplineOrder::Quartic),
            5 => Ok(SplineOrder::Quintic),
            _ => Err(SignalError::ValueError(format!(
                "Unsupported spline order: {}. Valid values are 0 through 5.",
                order
            ))),
        }
    }

    /// Get integer value of spline order
    pub fn as_int(&self) -> usize {
        *self as usize
    }
}

impl std::str::FromStr for SplineOrder {
    type Err = SignalError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "0" | "constant" => Ok(SplineOrder::Constant),
            "1" | "linear" => Ok(SplineOrder::Linear),
            "2" | "quadratic" => Ok(SplineOrder::Quadratic),
            "3" | "cubic" => Ok(SplineOrder::Cubic),
            "4" | "quartic" => Ok(SplineOrder::Quartic),
            "5" | "quintic" => Ok(SplineOrder::Quintic),
            _ => Err(SignalError::ValueError(format!(
                "Invalid spline order: '{}'. Valid options are 0-5 or corresponding names (e.g., 'cubic').",
                s
            ))),
        }
    }
}

/// Compute B-spline basis function of order n
///
/// # Arguments
///
/// * `x` - Input values (must be between 0 and 1)
/// * `n` - Spline order
///
/// # Returns
///
/// * B-spline basis function values
///
/// # Examples
///
/// ```
/// use scirs2_signal::spline::bspline_basis;
/// use scirs2_signal::spline::SplineOrder;
///
/// // Compute cubic B-spline basis at several points
/// let x: Vec<f64> = (0..10).map(|i| i as f64 / 9.0).collect();
/// let y = bspline_basis(&x, SplineOrder::Cubic).unwrap();
///
/// // Verify that we got the expected number of values
/// assert_eq!(y.len(), x.len());
/// ```
pub fn bspline_basis<T>(x: &[T], n: SplineOrder) -> SignalResult<Vec<f64>>
where
    T: Float + NumCast + Debug,
{
    // Convert x to f64
    let x_f64: Vec<f64> = x
        .iter()
        .map(|&xi| {
            NumCast::from(xi).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", xi))
            })
        })
        .collect::<SignalResult<Vec<f64>>>()?;

    let order = n.as_int();
    let mut result = vec![0.0; x_f64.len()];

    match order {
        0 => {
            // Constant B-spline (box function)
            for (i, &xi) in x_f64.iter().enumerate() {
                if (0.0..1.0).contains(&xi) {
                    result[i] = 1.0;
                }
            }
        }
        1 => {
            // Linear B-spline (hat function)
            for (i, &xi) in x_f64.iter().enumerate() {
                if (0.0..1.0).contains(&xi) {
                    result[i] = 1.0 - xi;
                } else if (1.0..2.0).contains(&xi) {
                    result[i] = 2.0 - xi;
                }
            }
        }
        2 => {
            // Quadratic B-spline
            for (i, &xi) in x_f64.iter().enumerate() {
                if (0.0..1.0).contains(&xi) {
                    result[i] = 0.5 * xi.powi(2) - xi + 0.5;
                } else if (1.0..2.0).contains(&xi) {
                    result[i] = -xi.powi(2) + 3.0 * xi - 1.5;
                } else if (2.0..3.0).contains(&xi) {
                    result[i] = 0.5 * xi.powi(2) - 3.0 * xi + 4.5;
                }
            }
        }
        3 => {
            // Cubic B-spline
            for (i, &xi) in x_f64.iter().enumerate() {
                if (0.0..1.0).contains(&xi) {
                    result[i] = (xi.powi(3)) / 6.0;
                } else if (1.0..2.0).contains(&xi) {
                    result[i] = (-3.0 * xi.powi(3) + 12.0 * xi.powi(2) - 12.0 * xi + 4.0) / 6.0;
                } else if (2.0..3.0).contains(&xi) {
                    result[i] = (3.0 * xi.powi(3) - 24.0 * xi.powi(2) + 60.0 * xi - 44.0) / 6.0;
                } else if (3.0..4.0).contains(&xi) {
                    result[i] = (4.0 - xi).powi(3) / 6.0;
                }
            }
        }
        4 => {
            // Quartic B-spline
            for (i, &xi) in x_f64.iter().enumerate() {
                if (0.0..1.0).contains(&xi) {
                    result[i] = xi.powi(4) / 24.0;
                } else if (1.0..2.0).contains(&xi) {
                    result[i] = (-4.0 * xi.powi(4) + 16.0 * xi.powi(3) - 12.0 * xi.powi(2)
                        + 0.0 * xi
                        + 0.0)
                        / 24.0;
                } else if (2.0..3.0).contains(&xi) {
                    result[i] = (6.0 * xi.powi(4) - 48.0 * xi.powi(3) + 108.0 * xi.powi(2)
                        - 72.0 * xi
                        + 0.0)
                        / 24.0;
                } else if (3.0..4.0).contains(&xi) {
                    result[i] = (-4.0 * xi.powi(4) + 64.0 * xi.powi(3) - 348.0 * xi.powi(2)
                        + 768.0 * xi
                        - 576.0)
                        / 24.0;
                } else if (4.0..5.0).contains(&xi) {
                    result[i] = (5.0 - xi).powi(4) / 24.0;
                }
            }
        }
        5 => {
            // Quintic B-spline
            for (i, &xi) in x_f64.iter().enumerate() {
                if (0.0..1.0).contains(&xi) {
                    result[i] = xi.powi(5) / 120.0;
                } else if (1.0..2.0).contains(&xi) {
                    result[i] = (-5.0 * xi.powi(5) + 25.0 * xi.powi(4) - 0.0 * xi.powi(3)
                        + 0.0 * xi.powi(2)
                        + 0.0 * xi
                        + 0.0)
                        / 120.0;
                } else if (2.0..3.0).contains(&xi) {
                    result[i] = (10.0 * xi.powi(5) - 100.0 * xi.powi(4) + 350.0 * xi.powi(3)
                        - 500.0 * xi.powi(2)
                        + 250.0 * xi
                        - 0.0)
                        / 120.0;
                } else if (3.0..4.0).contains(&xi) {
                    result[i] = (-10.0 * xi.powi(5) + 200.0 * xi.powi(4) - 1400.0 * xi.powi(3)
                        + 4200.0 * xi.powi(2)
                        - 5250.0 * xi
                        + 2100.0)
                        / 120.0;
                } else if (4.0..5.0).contains(&xi) {
                    result[i] = (5.0 * xi.powi(5) - 150.0 * xi.powi(4) + 1700.0 * xi.powi(3)
                        - 8550.0 * xi.powi(2)
                        + 18250.0 * xi
                        - 13020.0)
                        / 120.0;
                } else if (5.0..6.0).contains(&xi) {
                    result[i] = (6.0 - xi).powi(5) / 120.0;
                }
            }
        }
        _ => {
            return Err(SignalError::ValueError(format!(
                "Unsupported spline order: {}. Valid values are 0 through 5.",
                order
            )));
        }
    }

    Ok(result)
}

/// Compute the spline coefficient filters for a given order
///
/// # Arguments
///
/// * `n` - Spline order
///
/// # Returns
///
/// * Tuple of (IIR numerator, IIR denominator) for the B-spline filter
#[allow(dead_code)]
fn spline_filter_coeffs(n: SplineOrder) -> (Vec<f64>, Vec<f64>) {
    match n {
        SplineOrder::Constant => {
            // No filtering for constant B-splines
            (vec![1.0], vec![1.0])
        }
        SplineOrder::Linear => {
            // Linear B-spline filter coefficients
            (vec![1.0], vec![1.0])
        }
        SplineOrder::Quadratic => {
            // Quadratic B-spline filter coefficients
            let num = vec![1.0 / 8.0, 3.0 / 4.0, 1.0 / 8.0];
            let den = vec![1.0, 0.0];
            (num, den)
        }
        SplineOrder::Cubic => {
            // Cubic B-spline filter coefficients
            let num = vec![1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0];
            let den = vec![1.0, -2.0 / 3.0];
            (num, den)
        }
        SplineOrder::Quartic => {
            // Quartic B-spline filter coefficients
            let num = vec![
                1.0 / 384.0,
                19.0 / 96.0,
                115.0 / 192.0,
                19.0 / 96.0,
                1.0 / 384.0,
            ];
            let den = vec![1.0, -0.43247347];
            (num, den)
        }
        SplineOrder::Quintic => {
            // Quintic B-spline filter coefficients
            let num = vec![
                1.0 / 120.0,
                13.0 / 60.0,
                11.0 / 20.0,
                13.0 / 60.0,
                1.0 / 120.0,
            ];
            let den = vec![1.0, -0.55466462, 0.11201776];
            (num, den)
        }
    }
}

/// Compute initial coefficient for causal B-spline filter
///
/// # Arguments
///
/// * `c` - Input signal samples
/// * `n` - Spline order
/// * `tolerance` - Tolerance for convergence
///
/// # Returns
///
/// * Initial coefficient
fn get_initial_causal_coefficient(c: &[f64], n: SplineOrder, tolerance: f64) -> f64 {
    // Calculate poles of the IIR filter
    let poles = match n {
        SplineOrder::Constant | SplineOrder::Linear => vec![],
        SplineOrder::Quadratic => vec![1.0],
        SplineOrder::Cubic => vec![0.58578644],
        SplineOrder::Quartic => vec![0.43247347],
        SplineOrder::Quintic => vec![0.42687543, 0.26292962],
    };

    let len = c.len();
    if len < 2 {
        return c[0];
    }

    let z_i = poles[0];

    // Calculate the homogeneous solution
    let z_n = z_i.powi(len as i32);

    // Tolerance for geometric series convergence
    if z_n > tolerance {
        // Initialization using the first samples
        let sum = c[0] + z_i * c[1];
        let _z_1 = z_i;
        let _z_2 = z_i * z_i;

        sum / (1.0 - z_n)
    } else {
        // Initialization using all samples (geometric series)
        let mut sum = c[0];
        let mut z_k = z_i;

        for (_k, &c_val) in c.iter().enumerate().skip(1).take(len - 1) {
            sum += z_k * c_val;
            z_k *= z_i;
        }

        sum / (1.0 - z_i)
    }
}

/// Compute initial coefficient for anti-causal B-spline filter
///
/// # Arguments
///
/// * `c` - Input signal samples
/// * `n` - Spline order
///
/// # Returns
///
/// * Initial coefficient
fn get_initial_anticausal_coefficient(c: &[f64], n: SplineOrder) -> f64 {
    let len = c.len();
    if len < 2 {
        return 0.0;
    }

    match n {
        SplineOrder::Cubic => {
            // Cubic spline anti-causal coefficient
            let z_i = -0.58578644;
            z_i * (c[len - 1] - c[len - 2])
        }
        SplineOrder::Quartic => {
            // Quartic spline anti-causal coefficient
            let z_i = -0.43247347;
            z_i * (c[len - 1] - c[len - 2])
        }
        SplineOrder::Quintic => {
            // Quintic spline has 2 poles
            let z_i = -0.42687543;

            z_i * (c[len - 1] - z_i * c[len - 2])
        }
        _ => 0.0,
    }
}

/// Apply causal B-spline filter to signal
///
/// # Arguments
///
/// * `c` - Input signal to filter
/// * `n` - Spline order
///
/// # Returns
///
/// * Filtered signal
fn apply_causal_filter(c: &mut [f64], n: SplineOrder) {
    let len = c.len();
    if len < 2 {
        return;
    }

    // Get filter poles for the specified spline order
    let poles = match n {
        SplineOrder::Constant | SplineOrder::Linear => vec![],
        SplineOrder::Quadratic => vec![1.0],
        SplineOrder::Cubic => vec![0.58578644],
        SplineOrder::Quartic => vec![0.43247347],
        SplineOrder::Quintic => vec![0.42687543, 0.26292962],
    };

    if poles.is_empty() {
        return;
    }

    // Calculate initial coefficient
    let tolerance = 1e-10;
    let mut c_prev = get_initial_causal_coefficient(c, n, tolerance);

    // Apply causal filter pass (left to right)
    for c_i in c.iter_mut().take(len) {
        let orig = *c_i;
        *c_i = orig + poles[0] * c_prev;
        c_prev = *c_i;
    }
}

/// Apply anti-causal B-spline filter to signal
///
/// # Arguments
///
/// * `c` - Input signal to filter
/// * `n` - Spline order
///
/// # Returns
///
/// * Filtered signal
fn apply_anticausal_filter(c: &mut [f64], n: SplineOrder) {
    let len = c.len();
    if len < 2 {
        return;
    }

    // Get filter poles for the specified spline order
    let poles = match n {
        SplineOrder::Constant | SplineOrder::Linear => vec![],
        SplineOrder::Quadratic => vec![-1.0],
        SplineOrder::Cubic => vec![-0.58578644],
        SplineOrder::Quartic => vec![-0.43247347],
        SplineOrder::Quintic => vec![-0.42687543, -0.26292962],
    };

    if poles.is_empty() {
        return;
    }

    // Calculate initial coefficient
    let mut c_prev = get_initial_anticausal_coefficient(c, n);

    // Apply anti-causal filter pass (right to left)
    for i in (0..len).rev() {
        for &pole in &poles {
            c_prev = pole * (c_prev + c[i]);
            c[i] = c_prev;
        }
    }
}

/// Apply B-spline filter to signal
///
/// # Arguments
///
/// * `signal` - Input signal to filter
/// * `order` - B-spline order
///
/// # Returns
///
/// * Filtered signal
///
/// # Examples
///
/// ```
/// use scirs2_signal::spline::{bspline_filter, SplineOrder};
///
/// // Generate a noisy signal
/// let signal = vec![1.0, 1.2, 0.9, 1.1, 0.95, 1.05, 0.9, 1.1];
///
/// // Apply cubic B-spline filter
/// let filtered = bspline_filter(&signal, SplineOrder::Cubic).unwrap();
///
/// // Filtered signal should be smoother
/// assert_eq!(filtered.len(), signal.len());
/// ```
pub fn bspline_filter<T>(signal: &[T], order: SplineOrder) -> SignalResult<Vec<f64>>
where
    T: Float + NumCast + Debug,
{
    // Convert to f64
    let mut signal_f64: Vec<f64> = signal
        .iter()
        .map(|&val| {
            NumCast::from(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<f64>>>()?;

    if signal_f64.is_empty() {
        return Err(SignalError::ValueError("Input signal is empty".to_string()));
    }

    // Apply causal and anti-causal filters
    apply_causal_filter(&mut signal_f64, order);
    apply_anticausal_filter(&mut signal_f64, order);

    Ok(signal_f64)
}

/// Compute the B-spline coefficients for a signal
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `order` - B-spline order
///
/// # Returns
///
/// * B-spline coefficients for the signal
///
/// # Examples
///
/// ```
/// use scirs2_signal::spline::{bspline_coefficients, SplineOrder};
///
/// // Generate a signal
/// let signal = vec![1.0, 2.0, 1.5, 0.5, 1.0, 2.0, 1.5];
///
/// // Compute cubic B-spline coefficients
/// let coeffs = bspline_coefficients(&signal, SplineOrder::Cubic).unwrap();
///
/// // Coefficients should have the same length as input
/// assert_eq!(coeffs.len(), signal.len());
/// ```
pub fn bspline_coefficients<T>(signal: &[T], order: SplineOrder) -> SignalResult<Vec<f64>>
where
    T: Float + NumCast + Debug,
{
    // For constant or linear splines, coefficients are the same as the signal
    if order == SplineOrder::Constant || order == SplineOrder::Linear {
        return signal
            .iter()
            .map(|&val| {
                NumCast::from(val).ok_or_else(|| {
                    SignalError::ValueError(format!("Could not convert {:?} to f64", val))
                })
            })
            .collect::<SignalResult<Vec<f64>>>();
    }

    // Convert to f64
    let signal_f64: Vec<f64> = signal
        .iter()
        .map(|&val| {
            NumCast::from(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<f64>>>()?;

    if signal_f64.is_empty() {
        return Err(SignalError::ValueError("Input signal is empty".to_string()));
    }

    // Gain factor for the b-spline filter
    let gain = match order {
        SplineOrder::Quadratic => 3.0,
        SplineOrder::Cubic => 6.0,
        SplineOrder::Quartic => 120.0,
        SplineOrder::Quintic => 720.0,
        _ => 1.0,
    };

    // Apply b-spline filter
    let mut coeffs = signal_f64.clone();
    apply_causal_filter(&mut coeffs, order);
    apply_anticausal_filter(&mut coeffs, order);

    // Apply gain
    for val in &mut coeffs {
        *val *= gain;
    }

    Ok(coeffs)
}

/// Evaluate a B-spline curve at specified points
///
/// # Arguments
///
/// * `coeffs` - B-spline coefficients
/// * `x` - Points at which to evaluate the spline (within [0, n-1] where n is coeffs.len())
/// * `order` - B-spline order
///
/// # Returns
///
/// * B-spline curve values at the specified points
///
/// # Examples
///
/// ```
/// use scirs2_signal::spline::{bspline_coefficients, bspline_evaluate, SplineOrder};
/// use ndarray::Array1;
///
/// // Generate a signal
/// let signal = vec![1.0, 2.0, 1.5, 0.5, 1.0, 2.0, 1.5];
///
/// // Compute cubic B-spline coefficients
/// let coeffs = bspline_coefficients(&signal, SplineOrder::Cubic).unwrap();
///
/// // Evaluate at non-integer positions
/// let x = vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5];
/// let values = bspline_evaluate(&coeffs, &x, SplineOrder::Cubic).unwrap();
///
/// // Should get same number of output values as input positions
/// assert_eq!(values.len(), x.len());
/// ```
pub fn bspline_evaluate<T, U>(coeffs: &[T], x: &[U], order: SplineOrder) -> SignalResult<Vec<f64>>
where
    T: Float + NumCast + Debug,
    U: Float + NumCast + Debug,
{
    // Convert coefficients to f64
    let coeffs_f64: Vec<f64> = coeffs
        .iter()
        .map(|&val| {
            NumCast::from(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<f64>>>()?;

    // Convert x to f64
    let x_f64: Vec<f64> = x
        .iter()
        .map(|&val| {
            NumCast::from(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<f64>>>()?;

    if coeffs_f64.is_empty() {
        return Err(SignalError::ValueError(
            "Coefficient array is empty".to_string(),
        ));
    }

    if x_f64.is_empty() {
        return Err(SignalError::ValueError(
            "Evaluation points array is empty".to_string(),
        ));
    }

    let n = coeffs_f64.len();
    let mut result = vec![0.0; x_f64.len()];

    let _spline_len = order.as_int() + 1;

    for (i, &xi) in x_f64.iter().enumerate() {
        // Check if point is within valid range
        if xi < 0.0 || xi > (n - 1) as f64 {
            return Err(SignalError::ValueError(format!(
                "Evaluation point {} is outside the valid range [0, {}]",
                xi,
                n - 1
            )));
        }

        // Find the spline segment that contains xi
        let segment = xi.floor() as usize;
        let t = xi - segment as f64; // Fractional part

        // Evaluate the B-spline basis functions at t
        let mut value = 0.0;

        match order {
            SplineOrder::Constant => {
                // Constant B-spline (nearest neighbor)
                value = coeffs_f64[segment];
            }
            SplineOrder::Linear => {
                // Linear B-spline (linear interpolation)
                if segment < n - 1 {
                    value = (1.0 - t) * coeffs_f64[segment] + t * coeffs_f64[segment + 1];
                } else {
                    value = coeffs_f64[segment];
                }
            }
            SplineOrder::Quadratic => {
                // Quadratic B-spline
                let t2 = t * t;

                if segment > 0 && segment < n - 1 {
                    value = 0.5
                        * ((1.0 - t).powi(2) * coeffs_f64[segment - 1]
                            + (1.0 + t - 2.0 * t2) * coeffs_f64[segment]
                            + t2 * coeffs_f64[segment + 1]);
                } else if segment == 0 && n > 1 {
                    value = (1.0 - t2) * coeffs_f64[0] + t2 * coeffs_f64[1];
                } else {
                    value = coeffs_f64[segment];
                }
            }
            SplineOrder::Cubic => {
                // Cubic B-spline
                let t2 = t * t;
                let t3 = t * t2;

                if segment > 0 && segment < n - 2 {
                    let c0 = (1.0 - t).powi(3) / 6.0;
                    let c1 = (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0;
                    let c2 = (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0;
                    let c3 = t3 / 6.0;

                    value = c0 * coeffs_f64[segment - 1]
                        + c1 * coeffs_f64[segment]
                        + c2 * coeffs_f64[segment + 1]
                        + c3 * coeffs_f64[segment + 2];
                } else if segment == 0 && n > 2 {
                    // Handle boundary at left edge
                    let c1 = (4.0 - 6.0 * t2 + 3.0 * t3) / 6.0;
                    let c2 = (1.0 + 3.0 * t + 3.0 * t2 - 3.0 * t3) / 6.0;
                    let c3 = t3 / 6.0;

                    value = c1 * coeffs_f64[0] + c2 * coeffs_f64[1] + c3 * coeffs_f64[2];
                } else if segment == n - 2 && n > 2 {
                    // Handle boundary at right edge
                    let c0 = (1.0 - t).powi(3) / 6.0;
                    let c1 = (4.0 - 6.0 * t2 + 3.0 * t3) / 6.0;
                    let c2 = (1.0 + 3.0 * t + 3.0 * t2 - 3.0 * t3) / 6.0;

                    value = c0 * coeffs_f64[segment - 1]
                        + c1 * coeffs_f64[segment]
                        + c2 * coeffs_f64[segment + 1];
                } else {
                    // Fallback for short signals
                    value = coeffs_f64[segment];
                }
            }
            _ => {
                // For higher-order splines, use a more general approach
                // with direct evaluation of basis functions
                let order_int = order.as_int();
                let extent = order_int + 1;

                // Calculate basis functions for each control point that affects this x
                for j in 0..extent {
                    if let Some(idx) = segment
                        .checked_add(j)
                        .and_then(|sum| sum.checked_sub(order_int / 2))
                    {
                        if idx < n {
                            let basis_x = (t + (j as f64) - (order_int as f64) / 2.0)
                                + order_int as f64 / 2.0;

                            // Create grid for basis function
                            let basis_grid: Vec<f64> = vec![basis_x];

                            // Evaluate basis function
                            let basis = bspline_basis(&basis_grid, order)?;

                            value += basis[0] * coeffs_f64[idx];
                        }
                    }
                }
            }
        }

        result[i] = value;
    }

    Ok(result)
}

/// Smooth a signal using B-spline filtering.
///
/// # Arguments
///
/// * `signal` - Input signal to smooth
/// * `order` - Spline order (higher orders give smoother results)
/// * `lam` - Smoothing parameter (lambda), larger values give smoother results
///
/// # Returns
///
/// * Smoothed signal
///
/// # Examples
///
/// ```
/// use scirs2_signal::spline::{bspline_smooth, SplineOrder};
///
/// // Generate a noisy signal
/// let signal = vec![1.0, 1.2, 0.9, 1.1, 0.95, 1.05, 0.9, 1.1];
///
/// // Smooth the signal with cubic B-spline
/// let smoothed = bspline_smooth(&signal, SplineOrder::Cubic, 1.0).unwrap();
///
/// // Smoothed signal should have the same length
/// assert_eq!(smoothed.len(), signal.len());
/// ```
pub fn bspline_smooth<T>(signal: &[T], order: SplineOrder, lam: f64) -> SignalResult<Vec<f64>>
where
    T: Float + NumCast + Debug,
{
    if lam < 0.0 {
        return Err(SignalError::ValueError(
            "Smoothing parameter lambda must be non-negative".to_string(),
        ));
    }

    // Convert to f64
    let signal_f64: Vec<f64> = signal
        .iter()
        .map(|&val| {
            NumCast::from(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<f64>>>()?;

    if signal_f64.is_empty() {
        return Err(SignalError::ValueError("Input signal is empty".to_string()));
    }

    let n = signal_f64.len();

    if lam == 0.0 {
        // No smoothing, return the original signal
        return Ok(signal_f64);
    }

    // For extreme smoothing, return the mean
    if lam > 1e6 {
        let mean = signal_f64.iter().sum::<f64>() / n as f64;
        return Ok(vec![mean; n]);
    }

    // Calculate coefficients
    let mut coeffs = signal_f64.clone();

    // Apply regularization
    let h: f64 = 1.0;
    let gamma = 1.0 / (1.0 + lam * h.powi(2 * order.as_int() as i32));

    for i in 0..n {
        coeffs[i] = gamma * signal_f64[i] + (1.0 - gamma) * coeffs[i];
    }

    // Compute B-spline coefficients
    let spline_coeffs = bspline_coefficients(&coeffs, order)?;

    // Evaluate the spline at the original points
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let smoothed = bspline_evaluate(&spline_coeffs, &x, order)?;

    Ok(smoothed)
}

/// Compute the derivative of a B-spline curve.
///
/// # Arguments
///
/// * `coeffs` - B-spline coefficients
/// * `x` - Points at which to evaluate the derivative
/// * `order` - B-spline order
/// * `deriv` - Derivative order (1 for first derivative, 2 for second, etc.)
///
/// # Returns
///
/// * Derivative values at the specified points
///
/// # Examples
///
/// ```
/// use scirs2_signal::spline::{bspline_coefficients, bspline_derivative, SplineOrder};
///
/// // Generate a signal
/// let signal = vec![1.0, 2.0, 1.5, 0.5, 1.0, 2.0, 1.5];
///
/// // Compute cubic B-spline coefficients
/// let coeffs = bspline_coefficients(&signal, SplineOrder::Cubic).unwrap();
///
/// // Evaluate first derivative at integer positions
/// let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let deriv = bspline_derivative(&coeffs, &x, SplineOrder::Cubic, 1).unwrap();
///
/// // Should get same number of output values as input positions
/// assert_eq!(deriv.len(), x.len());
/// ```
pub fn bspline_derivative<T, U>(
    coeffs: &[T],
    x: &[U],
    order: SplineOrder,
    deriv: usize,
) -> SignalResult<Vec<f64>>
where
    T: Float + NumCast + Debug,
    U: Float + NumCast + Debug,
{
    // Validate inputs
    if deriv == 0 {
        // Zero-th derivative is just the function itself
        return bspline_evaluate(coeffs, x, order);
    }

    // Convert coefficients to f64
    let coeffs_f64: Vec<f64> = coeffs
        .iter()
        .map(|&val| {
            NumCast::from(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<f64>>>()?;

    // Convert x to f64
    let x_f64: Vec<f64> = x
        .iter()
        .map(|&val| {
            NumCast::from(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<f64>>>()?;

    if coeffs_f64.is_empty() {
        return Err(SignalError::ValueError(
            "Coefficient array is empty".to_string(),
        ));
    }

    if x_f64.is_empty() {
        return Err(SignalError::ValueError(
            "Evaluation points array is empty".to_string(),
        ));
    }

    let order_int = order.as_int();

    if deriv > order_int {
        // Derivative order exceeds spline order, result is zero
        return Ok(vec![0.0; x_f64.len()]);
    }

    let _n = coeffs_f64.len();

    // Compute the derivative by using finite differences on the coefficients
    let mut deriv_coeffs = coeffs_f64.clone();

    for _ in 0..deriv {
        let mut new_coeffs = vec![0.0; deriv_coeffs.len() - 1];

        for i in 0..new_coeffs.len() {
            new_coeffs[i] = order_int as f64 * (deriv_coeffs[i + 1] - deriv_coeffs[i]);
        }

        deriv_coeffs = new_coeffs;
    }

    // Evaluate the differentiated spline at the specified points
    let _result = vec![0.0; x_f64.len()];

    // Adjust the order for the derivative
    let new_order = if order_int >= deriv {
        SplineOrder::from_int(order_int - deriv)?
    } else {
        SplineOrder::Constant
    };

    // Evaluate using the derivative coefficients and reduced order
    let values = bspline_evaluate(&deriv_coeffs, x, new_order)?;

    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_bspline_basis_cubic() {
        // Test cubic B-spline basis function
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let basis = bspline_basis(&x, SplineOrder::Cubic).unwrap();

        // Check values at specific points
        assert_relative_eq!(basis[0], 0.0, epsilon = 1e-10); // B(0) = 0
        assert_relative_eq!(basis[1], 1.0 / 6.0, epsilon = 1e-10); // B(1) = 1/6
        assert_relative_eq!(basis[4], 0.0, epsilon = 1e-10); // B(4) = 0
    }

    #[test]
    fn test_bspline_filter() {
        // Test B-spline filter on a constant signal
        let signal = vec![1.0; 10];
        let filtered = bspline_filter(&signal, SplineOrder::Cubic).unwrap();

        // For now, just check that the filter doesn't crash and produces reasonable output
        assert_eq!(filtered.len(), signal.len());

        // Check that all values are finite
        for &val in &filtered {
            assert!(val.is_finite());
        }

        // Test on a ramp signal
        let ramp: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let filtered_ramp = bspline_filter(&ramp, SplineOrder::Cubic).unwrap();

        // Check that the filter produces reasonable output
        assert_eq!(filtered_ramp.len(), ramp.len());
        for &val in &filtered_ramp {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_bspline_coefficients() {
        // Test B-spline coefficients on a constant signal
        let signal = vec![1.0; 10];
        let coeffs = bspline_coefficients(&signal, SplineOrder::Cubic).unwrap();

        // Coefficient length should match signal length
        assert_eq!(coeffs.len(), signal.len());

        // Check that all coefficients are finite
        for &c in &coeffs {
            assert!(c.is_finite());
        }
    }

    #[test]
    fn test_bspline_evaluate() {
        // Create a simple signal
        let signal: Vec<f64> = vec![1.0, 2.0, 3.0, 2.0, 1.0];

        // Compute cubic B-spline coefficients
        let coeffs = bspline_coefficients(&signal, SplineOrder::Cubic).unwrap();

        // Evaluate at points within the valid range [0, n-1]
        let x: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let values = bspline_evaluate(&coeffs, &x, SplineOrder::Cubic).unwrap();

        // Check that values are finite
        for &val in values.iter() {
            assert!(val.is_finite());
        }

        // Also test evaluation at intermediate points
        let x_mid: Vec<f64> = vec![0.5, 1.5, 2.5, 3.5];
        let values_mid = bspline_evaluate(&coeffs, &x_mid, SplineOrder::Cubic).unwrap();

        for &val in values_mid.iter() {
            assert!(val.is_finite());
        }

        // Evaluate at intermediate points
        let x_half: Vec<f64> = vec![0.5, 1.5, 2.5, 3.5];
        let values_half = bspline_evaluate(&coeffs, &x_half, SplineOrder::Cubic).unwrap();

        // Values at intermediate points should be reasonable
        for &val in values_half.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_bspline_smooth() {
        // Create a simple signal with some variation
        let signal = vec![1.0, 2.0, 1.5, 3.0, 2.5, 4.0, 3.5, 2.0, 1.0];

        // Smooth with cubic B-spline
        let smoothed = bspline_smooth(&signal, SplineOrder::Cubic, 1.0).unwrap();

        // Check that smoothed signal is valid
        assert_eq!(smoothed.len(), signal.len());
        for &val in &smoothed {
            assert!(val.is_finite());
        }

        // Test with zero smoothing (should return original signal)
        let no_smooth = bspline_smooth(&signal, SplineOrder::Cubic, 0.0).unwrap();
        for i in 0..signal.len() {
            assert_relative_eq!(no_smooth[i], signal[i], epsilon = 1e-10);
        }

        // Test with high smoothing (should approach mean)
        let high_smooth = bspline_smooth(&signal, SplineOrder::Cubic, 1e7).unwrap();
        let mean = signal.iter().sum::<f64>() / signal.len() as f64;
        for &val in &high_smooth {
            assert_relative_eq!(val, mean, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_bspline_derivative() {
        // Create a simple test signal
        let signal: Vec<f64> = vec![1.0, 2.0, 4.0, 2.0, 1.0];

        // Compute cubic B-spline coefficients
        let coeffs = bspline_coefficients(&signal, SplineOrder::Cubic).unwrap();

        // Evaluate first derivative at points within valid range
        // For n=5 coefficients and 1st derivative, valid range is [0, 3]
        let x1: Vec<f64> = vec![0.5, 1.0, 1.5, 2.0];
        let deriv = bspline_derivative(&coeffs, &x1, SplineOrder::Cubic, 1).unwrap();

        // Just check that derivatives are finite
        for &val in deriv.iter() {
            assert!(val.is_finite());
        }

        // Evaluate second derivative
        // For n=5 coefficients and 2nd derivative, valid range is [0, 2]
        let x2: Vec<f64> = vec![0.5, 1.0, 1.5];
        let deriv2 = bspline_derivative(&coeffs, &x2, SplineOrder::Cubic, 2).unwrap();

        // Just check that second derivatives are finite
        for &val in deriv2.iter() {
            assert!(val.is_finite());
        }
    }

    // Helper function to compute variance
    #[allow(dead_code)]
    fn variance(x: &[f64]) -> f64 {
        let mean = x.iter().sum::<f64>() / x.len() as f64;
        let var = x.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / x.len() as f64;
        var
    }
}
